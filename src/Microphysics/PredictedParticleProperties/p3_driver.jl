using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Center, ZeroField
using Oceananigans.Grids: inactive_cell
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: Utils, launch!

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState, patankar_fraction, patankar_sink_fraction,
                                sequential_sink_fractions

using Breeze: Microphysics

using KernelAbstractions: @kernel, @index

const P3 = PredictedParticlePropertiesMicrophysics

# P3 evolves through RK-stage tendencies; it has no full-Δt operator-split update.
AM.microphysics_model_update!(::P3, model) = nothing

#####
##### Surface air temperature
#####
##### Hallett-Mossop rime splintering is switched off over a warm surface
##### (`maximum_splintering_surface_temperature`); that shutoff is the only part of the
##### scheme that needs a column-bottom value rather than a local one, so the bottom-most
##### active temperature of each column is gathered once per stage into `μ.surface_temperature`.
#####

@kernel function _p3_surface_temperature_kernel!(surface_temperature, temperature_field)
    i, j = @index(Global, NTuple)
    @inbounds surface_temperature[i, j, 1] = temperature_field[i, j, 1]
end

# With an immersed bottom, k = 1 can sit inside the topography, so the lowest active cell
# has to be found by scanning up the column. Branch-free, so it still compiles for the GPU.
@kernel function _p3_immersed_surface_temperature_kernel!(surface_temperature, temperature_field, grid)
    i, j = @index(Global, NTuple)

    FT = eltype(grid)
    bottom_temperature = zero(FT)
    found_active_cell = false

    for k in 1:grid.Nz
        active_cell = !inactive_cell(i, j, k, grid)
        use_this_cell = active_cell & !found_active_cell
        @inbounds local_temperature = temperature_field[i, j, k]
        bottom_temperature = ifelse(use_this_cell, local_temperature, bottom_temperature)
        found_active_cell = found_active_cell | active_cell
    end

    @inbounds surface_temperature[i, j, 1] = bottom_temperature
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Fill `surface_temperature` with the air temperature of the bottom-most active cell in
each column. Without an immersed boundary every column is active down to `k = 1`, so
this is a plain copy; `ImmersedBoundaryGrid` dispatches to a column scan.
"""
function compute_p3_surface_temperature!(surface_temperature, temperature_field, grid)
    launch!(grid.architecture, grid, :xy, _p3_surface_temperature_kernel!,
            surface_temperature, temperature_field)
    return nothing
end

# TODO: Add a vertically distributed column reduction in Oceananigans. Its distributed
# top/bottom halo fills are currently no-ops, so this column scan is correct for serial
# and horizontally partitioned grids, but cannot broadcast across a z partition.
function compute_p3_surface_temperature!(surface_temperature, temperature_field,
                                         grid::ImmersedBoundaryGrid)
    launch!(grid.architecture, grid, :xy, _p3_immersed_surface_temperature_kernel!,
            surface_temperature, temperature_field, grid)
    return nothing
end

#####
##### Process-rate tendencies
#####
##### "The P3 tendencies" are the tendencies of P3's own microphysical prognostics
##### (ρqᵛ, ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ, and the optional ρnᶜˡ/ρnᵃ/ρδqᵛ⁺ˡ).
##### They are computed *jointly*: the coupled donor-budget limiters see every species at
##### once, so one kernel evaluates all of them per cell and adds each straight into `Gⁿ`.
##### The evaluation is fully inlined. Table 1 is bracketed twice per cell: once at the
##### diagnostic ice population (`p3_ice_moments`) and once at the bounded population
##### (`P3IceLookups`, inside `compute_p3_process_rates`). Each ice integral is read at
##### whichever bracket matches the population it belongs to.
#####

# Everything `compute_p3_process_rates` needs at cell (i, j, k), assembled exactly as the
# tendency kernel assembles it, so that the process-rate diagnostics
# (`MicrophysicalProcessRate`) evaluate the very rates that drive the model.
@inline function p3_grid_process_inputs(i, j, k, grid, p3, μ, formulation, dynamics,
                                        constants, ρ_field, velocities)
    @inbounds begin
        ρ = ρ_field[i, j, k]
        qᵛᵉ = μ.qᵛ[i, j, k]
        surface_temperature = μ.surface_temperature[i, j, 1]

        ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, nothing, velocities)
        q = AM.moisture_fractions(p3, ℳ, qᵛᵉ)
        𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
        𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, qᵛᵉ, constants)

        # Approximate the resolved diffusional-growth driver with hydrostatic adiabatic
        # expansion from the local vertical velocity. External vapor forcing is
        # intentionally neglected.
        temperature_tendency = p3_adiabatic_temperature_tendency(ℳ, 𝒰, constants)
        pressure_tendency = p3_adiabatic_pressure_tendency(ρ, ℳ, constants)
    end

    properties = p3_process_properties(p3, ρ, ℳ)

    return (; ρ, ℳ, 𝒰, properties, surface_temperature, temperature_tendency, pressure_tendency)
end

@kernel function _p3_add_tendencies_kernel!(G, stage, μ, formulation, dynamics, grid, constants, p3, ρ_field, velocities,
                                            condensation_fields)
    i, j, k = @index(Global, NTuple)

    inputs = p3_grid_process_inputs(i, j, k, grid, p3, μ, formulation, dynamics,
                                    constants, ρ_field, velocities)

    # Statistics of the `condensation` option; `nothing` for `GridMeanCondensation`
    statistics = subgrid_supersaturation_statistics(i, j, k, grid, p3.condensation, condensation_fields)

    rates = compute_p3_process_rates(p3, inputs.ρ, inputs.ℳ, inputs.𝒰, constants, inputs.properties,
                                     inputs.surface_temperature, inputs.temperature_tendency,
                                     inputs.pressure_tendency, statistics,
                                     GridLiquidRelaxationStage(i, j, k, G, stage))

    # Scale every transfer to what the coming stage leaves of the species it draws on.
    φ = p3_sink_fractions(i, j, k, G, stage, p3, inputs.ρ, inputs.properties, rates)
    rates = patankar_limited_rates(rates, φ)

    result = p3_tendencies(p3, inputs.ρ, inputs.ℳ, inputs.properties, rates)

    # Read `G` before the process tendencies land on it: the relaxations are advanced
    # implicitly from what every other tendency leaves of each number.
    relaxations = p3_number_relaxations(i, j, k, G, stage, p3, inputs.ρ, inputs.ℳ,
                                        inputs.properties, rates)

    rime = p3_rime_limits(i, j, k, G, stage, result)

    add_p3_tendencies!(G, i, j, k, grid, p3, result, relaxations, rime)
end

#####
##### The stage of the implicit liquid relaxation
#####

# The stage that the implicit liquid relaxation advances over, at cell `(i, j, k)`. It holds only
# what the kernel already has, and the tendencies `G` are read where the relaxation needs them
# (`liquid_relaxation_stage`), so nothing more is carried through the process rates. `G` is read
# before the P3 tendencies are added.
struct GridLiquidRelaxationStage{I, G, S}
    i :: I
    j :: I
    k :: I
    tendencies :: G
    stage :: S
end

# What the stage start and every other tendency leave of the cloud, per unit mass.
@inline function liquid_relaxation_stage(grid_stage::GridLiquidRelaxationStage, aerosol, ρ)
    i, j, k = grid_stage.i, grid_stage.j, grid_stage.k
    G = grid_stage.tendencies
    stage = grid_stage.stage
    active = AM.stage_interval_is_known(stage)
    Δt = ifelse(active, stage.Δt, zero(stage.Δt))
    qᶜˡ = AM.stage_remainder(i, j, k, G.ρqᶜˡ, stage, Val(:ρqᶜˡ)) / ρ
    nᶜˡ = p3_cloud_number_remainder(i, j, k, G, stage, aerosol) / ρ
    return LiquidRelaxationStage(Δt, ifelse(active, qᶜˡ, zero(qᶜˡ)), ifelse(active, nᶜˡ, zero(nᶜˡ)), active)
end

@inline liquid_relaxation_is_active(grid_stage::GridLiquidRelaxationStage) =
    AM.stage_interval_is_known(grid_stage.stage)

@inline p3_cloud_number_remainder(i, j, k, G, stage, ::Nothing) = zero(stage.Δt)
@inline p3_cloud_number_remainder(i, j, k, G, stage, ::AerosolActivation) =
    AM.stage_remainder(i, j, k, G.ρnᶜˡ, stage, Val(:ρnᶜˡ))

#####
##### Patankar-limited sinks
#####
#
# The process rates budget each sink against the state they are evaluated at, over
# `sink_limiting_timescale`. The time stepper applies them to the state the coming stage
# starts from, over the stage interval `Δt`, alongside transport. On the grid every explicit
# sink is therefore advanced in Patankar form: each transfer is scaled by the
# `AtmosphereModels.patankar_sink_fraction` of the species it draws on.
#
# The transfers that are relaxed implicitly over the stage (`implicit_liquid_relaxation.jl`:
# the vapor exchange of cloud liquid and rain, and droplet activation) are already advanced
# over `Δt`, and the Patankar fraction would limit them a second time. They are drawn first
# from their donor, in full, and the explicit sinks of that donor are advanced in Patankar
# form against what they leave (`AtmosphereModels.sequential_sink_fractions`):
#
#   * cloud evaporation from cloud liquid, a relaxation of the cloud itself, which asks for no
#     more than the stage leaves of it;
#   * condensation on cloud and rain and the seed mass of activation from vapor, which they
#     relax toward saturation and so leave positive;
#   * rain evaporation from rain. It relaxes the vapor deficit, not the rain, so it can ask for
#     more rain than there is, and is then limited to the rain the stage leaves;
#   * activation from the aerosol reservoir, whose only sink it is.
#
# Each first draw is limited to what the stage leaves of its donor, which is what keeps the
# donor non-negative where the stage starts from a state other than the one the rates saw.
#
# A number sink that removes no mass (rain self-collection, ice aggregation) is budgeted
# against less than the number the stage leaves. The mass that the stage leaves of rain or ice
# needs a minimum number `n_min` to be carried by the largest particles the scheme admits
# (`minimum_rain_slope` for rain, the tabulated lower λ bound for ice), and such a sink is
# advanced in Patankar form against what the number holds above it, `max(0, ñ - n_min)`, with
# `ñ` what the sinks that remove number with mass leave. It therefore approaches `n_min` and
# never passes it, for any `Δt`. Where `ñ` is already below `n_min` the sink is switched off.
# The sinks that remove number with mass keep the fraction of the mass they remove and are not
# floored. Limited against the whole number, a number-only sink can leave mass with almost no
# number, which the size bounds then give the largest, fastest-falling particles.
#
# Before the first time step the rates are the explicit ones and every fraction is one.

struct P3SinkFractions{FT}
    vapor :: FT              # explicit vapor sinks: deposition, nucleation, coating condensation
    cloud :: FT              # explicit cloud sinks
    rain :: FT               # explicit rain sinks
    rain_number :: FT
    ice :: FT
    ice_number :: FT
    coating :: FT
    aerosol :: FT            # activation, drawn from the aerosol reservoir
    condensation :: FT       # implicit vapor sinks, drawn first: condensation on cloud and rain, activation
    cloud_evaporation :: FT  # drawn first from cloud liquid
    rain_evaporation :: FT   # drawn first from rain
    rain_number_evaporation :: FT  # drawn first from rain number, with the rain mass
    rain_self_collection :: FT  # number-only rain sink, floored at the size bound of the rain left
    aggregation :: FT           # number-only ice sink, floored at the size bound of the ice left
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The fraction on a number-only loss rate `number_only_loss` of a number density of which the
stage and the sinks that remove number with mass leave `ñ`, where the mass left needs at
least `n_min`. The loss is weighted by the Patankar fraction against `max(0, ñ - n_min)`, so
the number approaches `n_min` and does not pass it for any `Δt`, and is zero where `ñ` is
below `n_min`. It is no larger than `φ`, the Patankar fraction of all the explicit number
sinks together, to which it reduces where `n_min` is zero. One where the stage interval is
not `known`.
"""
@inline function floored_sink_fraction(φ, ñ, n_min, Δt, number_only_loss, known)
    φᶠ = min(φ, patankar_fraction(ñ - n_min, Δt, number_only_loss))
    return ifelse(known, φᶠ, one(φ))
end

@inline function p3_sink_fractions(i, j, k, G, stage, p3::P3, ρ, properties, rates::P3ProcessRates)
    liquid_on_ice = liquid_fraction_routing_active(p3.process_rates)
    condensation = max(0, rates.condensation)
    cloud_evaporation = max(0, -rates.condensation)
    deposition = max(0, rates.deposition)
    sublimation = max(0, -rates.deposition)

    # Vapor: the implicit condensation and activation first, then the explicit sinks
    liquid_condensation = condensation + rates.ccn_activation_mass + rates.rain_condensation
    vapor_loss = deposition + rates.nucleation_mass + rates.coating_condensation

    cloud_loss = rates.autoconversion + rates.accretion + rates.cloud_riming +
                 rates.cloud_freezing_mass + rates.cloud_homogeneous_mass +
                 rates.cloud_warm_collection + rates.wet_growth_cloud + rates.wet_growth_shedding

    rain_loss = rates.rain_riming + rates.rain_freezing_mass +
                rates.rain_homogeneous_mass + rates.rain_warm_collection + rates.wet_growth_rain

    # Evaporation removes drops in proportion to mass (`rain_evaporation_number = nʳ/qʳ ×
    # rain_evaporation`), so it is drawn first from rain number as from rain mass, and both lose
    # the same fraction of what they hold.
    # The other rain number sinks accompany a mass transfer, except self-collection.
    rain_number_loss_with_mass = rates.rain_riming_number + rates.rain_freezing_number +
                                 rates.rain_homogeneous_number + rates.rain_warm_collection_number
    rain_number_loss = rates.rain_self_collection + rain_number_loss_with_mass

    ice_loss = sublimation + rates.partial_melting + rates.complete_melting

    # `ni_limit` is a relaxation, advanced implicitly with the number corrections.
    ice_number_loss_with_mass = rates.melting_number + rates.sublimation_number
    ice_number_loss = ice_number_loss_with_mass + rates.aggregation + rates.homogeneous_number_excess

    coating_shedding = ifelse(liquid_on_ice, rates.wet_growth_shedding, zero(ρ))
    coating_loss = rates.shedding + rates.refreezing + rates.coating_evaporation + coating_shedding

    κᵛ, φᵛ = sequential_sink_fractions(i, j, k, G.ρqᵛ, stage, Val(:ρqᵛ), ρ * liquid_condensation, ρ * vapor_loss)
    κᶜˡ, φᶜˡ = sequential_sink_fractions(i, j, k, G.ρqᶜˡ, stage, Val(:ρqᶜˡ), ρ * cloud_evaporation, ρ * cloud_loss)
    κʳ, φʳ = sequential_sink_fractions(i, j, k, G.ρqʳ, stage, Val(:ρqʳ), ρ * rates.rain_evaporation, ρ * rain_loss)
    κⁿʳ, φⁿʳ = sequential_sink_fractions(i, j, k, G.ρnʳ, stage, Val(:ρnʳ),
                                         ρ * rates.rain_evaporation_number, ρ * rain_number_loss)

    φⁱ = patankar_sink_fraction(i, j, k, G.ρqⁱ, stage, Val(:ρqⁱ), ρ * ice_loss)
    φⁿⁱ = patankar_sink_fraction(i, j, k, G.ρnⁱ, stage, Val(:ρnⁱ), ρ * ice_number_loss)
    φʷⁱ = patankar_sink_fraction(i, j, k, G.ρqʷⁱ, stage, Val(:ρqʷⁱ), ρ * coating_loss)

    # The number-only sinks, floored at the fewest particles that can carry the mass the stage
    # and the mass sinks leave. A Patankar-limited loss leaves the fraction `φ` of what it is
    # drawn from. The sinks that are scaled by the smaller of two fractions are counted as
    # `patankar_limited_rates` scales them.
    Δt = stage.Δt
    known = AM.stage_interval_is_known(stage)

    ρqʳ_left = φʳ * max(0, max(0, AM.stage_remainder(i, j, k, G.ρqʳ, stage, Val(:ρqʳ))) -
                           Δt * κʳ * ρ * rates.rain_evaporation)
    ρnʳ_min = rain_number_from_slope(ρqʳ_left, p3.process_rates.minimum_rain_slope, p3.process_rates)
    ρnʳ_left = max(0, AM.stage_remainder(i, j, k, G.ρnʳ, stage, Val(:ρnʳ))) -
               Δt * ρ * (min(κʳ, κⁿʳ) * rates.rain_evaporation_number +
                         min(φʳ, φⁿʳ) * rain_number_loss_with_mass)
    φˢᶜ = floored_sink_fraction(φⁿʳ, ρnʳ_left, ρnʳ_min, Δt, ρ * rates.rain_self_collection, known)

    # The ice mass that sets the size bound includes the liquid coating where it is tracked.
    ρqʷⁱ_left = max(0, AM.stage_remainder(i, j, k, G.ρqʷⁱ, stage, Val(:ρqʷⁱ))) -
                Δt * ρ * (φʷⁱ * (coating_loss - coating_shedding) + min(φᶜˡ, φʷⁱ) * coating_shedding)
    ρqⁱ_left = φⁱ * max(0, AM.stage_remainder(i, j, k, G.ρqⁱ, stage, Val(:ρqⁱ))) +
               ifelse(liquid_on_ice, max(0, ρqʷⁱ_left), zero(ρ))
    ρnⁱ_min = properties.nⁱ_minimum_per_mass * ρqⁱ_left
    ρnⁱ_left = max(0, AM.stage_remainder(i, j, k, G.ρnⁱ, stage, Val(:ρnⁱ))) -
               Δt * ρ * min(φⁱ, φⁿⁱ) * ice_number_loss_with_mass
    φᵃᵍᵍ = floored_sink_fraction(φⁿⁱ, ρnⁱ_left, ρnⁱ_min, Δt, ρ * rates.aggregation, known)

    return P3SinkFractions(
        φᵛ, φᶜˡ, φʳ, φⁿʳ, φⁱ, φⁿⁱ, φʷⁱ,
        aerosol_sink_fraction(i, j, k, G, stage, p3.aerosol, ρ * rates.ccn_activation_number),
        κᵛ, κᶜˡ, κʳ, κⁿʳ, φˢᶜ, φᵃᵍᵍ)
end

# Activation draws on the aerosol reservoir only where that reservoir is prognostic. It is the
# reservoir's only sink, so it is limited to what the stage leaves of the reservoir and no further.
@inline aerosol_sink_fraction(i, j, k, G, stage, aerosol, L) = one(L)
@inline aerosol_sink_fraction(i, j, k, G, stage, ::AerosolActivation{<:Any, true}, L) =
    first(sequential_sink_fractions(i, j, k, G.ρnᵃ, stage, Val(:ρnᵃ), L, zero(L)))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The process `rates` with each transfer scaled by the `P3SinkFractions` `φ` of the
species it draws on: the Patankar fraction for an explicit transfer, and the fraction of an
implicitly relaxed transfer that the stage leaves of its donor, which is one wherever the
transfer fits. A number rate that accompanies a mass transfer takes the smaller of
the mass and number fractions, so it never removes more particles than its mass does mass.
A number rate that removes no mass (rain self-collection, ice aggregation) takes the
fraction floored at the size bound of the mass that is left.
Sources that draw on no prognostic, and the number relaxations, are unchanged.
"""
@inline function patankar_limited_rates(rates::P3ProcessRates, φ::P3SinkFractions)
    v, c, r, i, w = φ.vapor, φ.cloud, φ.rain, φ.ice, φ.coating
    rn = min(r, φ.rain_number)
    in = min(i, φ.ice_number)
    cw = min(c, w)
    condensation = φ.condensation
    activation = min(condensation, φ.aerosol)
    evaporation = φ.rain_evaporation

    return P3ProcessRates(
        rates.condensation * ifelse(rates.condensation > 0, condensation, φ.cloud_evaporation),
        rates.autoconversion * c,
        rates.accretion * c,
        rates.cloud_self_collection * c,
        rates.rain_evaporation * evaporation,
        rates.rain_evaporation_number * min(evaporation, φ.rain_number_evaporation),
        rates.rain_self_collection * φ.rain_self_collection,
        rates.rain_breakup,
        rates.deposition * ifelse(rates.deposition > 0, v, i),
        rates.partial_melting * i,
        rates.complete_melting * i,
        rates.melting_number * in,
        rates.clipping_dry_mass * i,
        rates.clipping_rime_mass * i,
        rates.clipping_rime_volume * i,
        rates.post_process_clipping,
        rates.sublimation_number * in,
        rates.aggregation * φ.aggregation,
        rates.ni_limit,
        rates.cloud_riming * c,
        rates.cloud_riming_number * c,
        rates.rain_riming * r,
        rates.rain_riming_number * rn,
        rates.rime_density_new,
        rates.shedding * w,
        rates.shedding_number * w,
        rates.refreezing * w,
        rates.nucleation_mass * v,
        rates.nucleation_number * v,
        rates.cloud_freezing_mass * c,
        rates.cloud_freezing_number * c,
        rates.rain_freezing_mass * r,
        rates.rain_freezing_number * rn,
        rates.splintering_mass,
        rates.splintering_number,
        rates.cloud_homogeneous_mass * c,
        rates.cloud_homogeneous_number * c,
        rates.rain_homogeneous_mass * r,
        rates.rain_homogeneous_number * rn,
        rates.homogeneous_number_excess * min(c, rn),
        rates.cloud_warm_collection * c,
        rates.cloud_warm_collection_number * c,
        rates.rain_warm_collection * r,
        rates.rain_warm_collection_number * rn,
        rates.wet_growth_cloud * c,
        rates.wet_growth_rain * r,
        rates.wet_growth_shedding * cw,
        rates.wet_growth_shedding_number * cw,
        rates.ccn_activation_mass * activation,
        rates.ccn_activation_number * activation,
        rates.rain_condensation * condensation,
        rates.coating_condensation * v,
        rates.coating_evaporation * w,
        rates.wet_growth_densification_mass,
        rates.wet_growth_densification_volume,
        rates.cloud_number_correction,
        rates.rain_number_correction,
        rates.ice_number_correction,
        rates.prognostic_vapor_excess_adjustment,
        rates.prognostic_vapor_excess_tendency)
end

#####
##### Rime limits
#####
#
# Rime mass and volume are properties of the ice, not separate reservoirs: they lose in
# proportion to the ice that melts or sublimates, at the rime fraction and density of the
# state the rates were evaluated at. The stage-start state can hold less rime than that,
# so their loss is bounded by what the coming stage leaves of each. No mass is exchanged
# with another species by this bound.

struct P3RimeLimits{FT}
    mass :: FT    # tendency added to the rime mass tendency so it removes no more than there is
    volume :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The tendency to add to the process tendency `Gᵖ` so that it removes no more than
`c̃ = c⁰ + Δt Gᵒ`, what the state the coming stage starts from and every other tendency
`Gᵒ` leave. Zero before the first time step, when the stage interval is not known.
"""
@inline function nonnegative_tendency_limit(i, j, k, G, stage, name, Gᵖ)
    Δt = stage.Δt
    c⁰ = AM.stage_start_value(i, j, k, stage, name)
    Gᵒ = AM.accumulated_tendency(i, j, k, G, stage, name)
    c̃ = max(0, c⁰ + Δt * Gᵒ)
    limited = max(Gᵖ, -c̃ / Δt)
    known_interval = isfinite(Δt) & (Δt > 0)
    return ifelse(known_interval, limited - Gᵖ, zero(Gᵖ))
end

@inline p3_rime_limits(i, j, k, G, stage, result) =
    P3RimeLimits(nonnegative_tendency_limit(i, j, k, G.ρqᶠ, stage, Val(:ρqᶠ), result.tendency_ρqᶠ),
                 nonnegative_tendency_limit(i, j, k, G.ρbᶠ, stage, Val(:ρbᶠ), result.tendency_ρbᶠ))

#####
##### Semi-implicit number relaxations
#####
#
# P3 keeps each number concentration consistent with its mass by relaxing it towards a
# bounded value `nᵗ` over `τ = sink_limiting_timescale`: the size-distribution bounds on
# cloud, rain and ice number, and the cap on ice number. The gridless rates carry these as
# the explicit tendencies `(nᵗ - n) / τ`, which overshoot through the target when a stage
# is longer than `τ`, are unstable when it is longer than about `2.5 τ`, and act on the
# wrong state when a stage starts from the step-start state instead of the state the rates
# were evaluated at.
#
# On the grid the relaxation is advanced by backward Euler over the coming stage instead.
# With `c⁰` the number density the stage starts from, `Gᵒ` every other tendency, and
# `c̃ = c⁰ + Δt Gᵒ` what they leave of it, the relaxed number density is
#
#     c⁺ = (c̃ + Δt ρ nᵗ / τ) / (1 + Δt / τ),
#
# which lies between `c̃` and the target for any `Δt`. It is applied as the equivalent
# tendency `(c⁺ - c̃) / Δt = (ρ nᵗ - c̃) / (τ + Δt)`, so the time stepper is unchanged.

struct P3NumberRelaxations{FT}
    cloud :: FT  # tendency replacing the explicit cloud number relaxation, minus that relaxation
    rain :: FT
    ice :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The tendency that advances a number relaxation towards the target density `ρnᵗ` implicitly
over the coming stage, minus the explicit relaxation tendency `explicit` that the process
rates already carry. `G` is the tendency of every other process, read before the P3 process
tendencies are added, and `Gᵖ` the P3 process tendency including `explicit`.

Falls back to the explicit relaxation (returns zero) where the relaxation is not `active`,
and before the first time step, when the stage interval is not known.
"""
@inline function semi_implicit_relaxation(i, j, k, G, stage, name, Gᵖ, explicit, ρnᵗ, τ, active)
    Δt = stage.Δt
    c⁰ = AM.stage_start_value(i, j, k, stage, name)
    Gᵒ = AM.accumulated_tendency(i, j, k, G, stage, name)
    c̃ = c⁰ + Δt * (Gᵒ + Gᵖ - explicit)
    implicit = (ρnᵗ - c̃) / (τ + Δt)
    semi_implicit = active & isfinite(Δt) & (Δt > 0)
    return ifelse(semi_implicit, implicit - explicit, zero(explicit))
end

@inline function p3_number_relaxations(i, j, k, G, stage, p3::P3, ρ, ℳ, properties, rates)
    τ = p3.process_rates.sink_limiting_timescale

    # Rain: the correction relaxes the raw number towards the slope-bounded one.
    rain_explicit = ρ * rates.rain_number_correction
    rain_target = max(0, ρ * ℳ.nʳ + τ * rain_explicit)
    rain = semi_implicit_relaxation(i, j, k, G.ρnʳ, stage, Val(:ρnʳ), tendency_ρnʳ(rates, ρ, p3),
                                    rain_explicit, rain_target, τ, rain_explicit != 0)

    # Ice: the λ-limiter correction and the cap together relax the raw number towards the
    # bounded number `properties.nⁱ`. A negative number relaxes towards it too; the explicit
    # correction is measured from `max(0, nⁱ)` and leaves a negative number where it is.
    ice_explicit = ρ * (rates.ice_number_correction - rates.ni_limit)
    ice_active = (ice_explicit != 0) | (ℳ.nⁱ < 0)
    ice = semi_implicit_relaxation(i, j, k, G.ρnⁱ, stage, Val(:ρnⁱ), tendency_ρnⁱ(rates, ρ),
                                   ice_explicit, ρ * properties.nⁱ, τ, ice_active)

    cloud = p3_cloud_number_relaxation(i, j, k, G, stage, p3, p3.aerosol, ρ, ℳ, properties, rates, τ)

    return P3NumberRelaxations(cloud, rain, ice)
end

# The prescribed-Nᶜˡ path has no cloud number prognostic.
@inline p3_cloud_number_relaxation(i, j, k, G, stage, p3, ::Nothing, ρ, ℳ, properties, rates, τ) = zero(ρ)

@inline function p3_cloud_number_relaxation(i, j, k, G, stage, p3, ::AerosolActivation, ρ, ℳ,
                                            properties, rates, τ)
    explicit = ρ * rates.cloud_number_correction
    target = max(0, ρ * ℳ.nᶜˡ + τ * explicit)
    Gᵖ = tendency_ρnᶜˡ(rates, ρ, properties.Nᶜˡ, ℳ.qᶜˡ, p3)
    return semi_implicit_relaxation(i, j, k, G.ρnᶜˡ, stage, Val(:ρnᶜˡ), Gᵖ, explicit, target, τ,
                                    explicit != 0)
end

#####
##### Fused tendency override (fast path for AtmosphereModel)
#####
#
# P3 evaluates its process rates from the current state with the adiabatic-only
# diffusional-growth driver described above, and adds them to `Gⁿ` in the same kernel.
# P3 sedimentation is assembled separately by the scalar transport operators using the
# fall speeds `update_microphysical_auxiliaries!` established during `update_state!`.
# The state-based `microphysical_tendency` methods above remain the gridless fallback
# used by ParcelModels.

# Just the `Gⁿ` slots P3 writes, not the model's whole tendency tuple. Dispatch is on a
# *type*, so the reduced tuple is a compile-time constant.
@inline p3_tendency_fields(G, p3::P3) =
    merge((; G.ρqᵛ, G.ρqᶜˡ, G.ρqʳ, G.ρnʳ, G.ρqⁱ, G.ρnⁱ, G.ρqᶠ, G.ρbᶠ, G.ρqʷⁱ),
          p3_aerosol_tendency_fields(G, p3.aerosol),
          p3_vapor_excess_tendency_fields(G, p3.process_rates))

@inline p3_aerosol_tendency_fields(G, ::Nothing) = (;)
@inline p3_aerosol_tendency_fields(G, ::AerosolActivation{<:Any, false}) = (; G.ρnᶜˡ)
@inline p3_aerosol_tendency_fields(G, ::AerosolActivation{<:Any, true}) = (; G.ρnᶜˡ, G.ρnᵃ)

@inline p3_vapor_excess_tendency_fields(G, ::ProcessRate{FT, DiagnosticVaporExcess}) where FT = (;)
@inline p3_vapor_excess_tendency_fields(G, ::ProcessRate{FT, PrognosticVaporExcess}) where FT = (; G.ρδqᵛ⁺ˡ)

function AM.compute_microphysical_tendencies!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    μ = model.microphysical_fields

    ρ_field = AM.total_density(model.dynamics)

    compute_p3_surface_temperature!(μ.surface_temperature, model.temperature, grid)

    stage = AM.StageStart(model, fields -> p3_tendency_fields(fields, p3))

    launch!(arch, grid, :xyz, _p3_add_tendencies_kernel!,
            p3_tendency_fields(model.timestepper.Gⁿ, p3), stage,
            μ, model.formulation, model.dynamics, grid,
            model.thermodynamic_constants, p3, ρ_field,
            model.velocities,
            subgrid_supersaturation_fields(p3.condensation, model))

    return nothing
end

#####
##### Phase relaxation rate (see `AtmosphereModels.phase_relaxation_rate`)
#####

@inline function AM.phase_relaxation_rate(i, j, k, grid, p3::P3, μ, 𝒰, constants)
    ρ = density(𝒰, constants)
    ℳ = p3_relaxation_microphysical_state(i, j, k, p3, μ, ρ)
    return p3_phase_relaxation_rates(p3, ρ, ℳ, 𝒰, constants).ωᶜ
end

# The microphysical state that the phase relaxation rate reads, at density ρ (no velocities).
@inline function p3_relaxation_microphysical_state(i, j, k, p3, μ, ρ)
    FT = typeof(ρ)
    nᶜˡ = grid_cloud_droplet_number(p3, p3.aerosol, μ, i, j, k, ρ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ  = μ.ρqʳ[i, j, k] / ρ
        nʳ  = μ.ρnʳ[i, j, k] / ρ
        qⁱ  = μ.ρqⁱ[i, j, k] / ρ
        nⁱ  = μ.ρnⁱ[i, j, k] / ρ
        qᶠ  = μ.ρqᶠ[i, j, k] / ρ
        bᶠ  = μ.ρbᶠ[i, j, k] / ρ
        qʷⁱ = μ.ρqʷⁱ[i, j, k] / ρ
    end
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ, zero(FT), zero(FT), zero(FT))
end

#####
##### Number concentration diagnostic
#####
#
# P3 carries prognostic number-density fields for rain and ice, and for cloud liquid when
# aerosol activation is enabled. The default prescribed-Nᶜˡ path has no `ρnᶜˡ` prognostic,
# but cloud droplets are still part of the model: their total number concentration is the
# constant `p3.cloud.number_concentration`. Represent that constant as a lazy operation so
# `number_concentration_field` works uniformly.

struct PrescribedCloudNumberKernelFunction{FT}
    number_concentration :: FT
end

Utils.prettysummary(::PrescribedCloudNumberKernelFunction) =
    "PrescribedCloudNumberKernelFunction"

@inline (kernel::PrescribedCloudNumberKernelFunction)(i, j, k, grid) =
    kernel.number_concentration

Microphysics.number_concentration(model, ::P3, ::Val{:rain}) =
    get(model.microphysical_fields, :ρnʳ, nothing)

Microphysics.number_concentration(model, p3::P3, ::Val{:cloud_liquid}) =
    cloud_number_concentration(model, p3, p3.aerosol)

function cloud_number_concentration(model, p3, ::Nothing)
    kernel = PrescribedCloudNumberKernelFunction(p3.cloud.number_concentration)
    return KernelFunctionOperation{Center, Center, Center}(kernel, model.grid)
end

cloud_number_concentration(model, p3, aerosol) =
    get(model.microphysical_fields, :ρnᶜˡ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val{:ice}) =
    get(model.microphysical_fields, :ρnⁱ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val) = nothing
