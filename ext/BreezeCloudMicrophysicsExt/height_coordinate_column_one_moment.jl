#####
##### Warm-phase one-moment microphysics with saturation adjustment (WP1M) as column physics on a height
##### coordinate: rain sources and sedimentation once per time step, after the dynamics
#####
#
# PROVISIONAL (AMIP pilot). Not included by `BreezeCloudMicrophysicsExt`: a driver evaluates this file in the
# extension module. It adds no rate and no fall speed: the conversion of cloud liquid to rain, the evaporation
# of rain and the fall speed are `HydrostaticEquations.rain_source_rates` and `rain_fall_speed` of WP1M, the
# functions the column physics of moist `HydrostaticDynamics` uses.
#
# Why. Through the tendencies, the rain of WP1M is not positive under a multi-stage time stepper at time
# steps of minutes: the sink of each stage is limited to the rain of that stage's state over `τⁿᵘᵐ`, and is
# applied from the state at the start of the step. Here the same operators as `warm_rain_column_physics!`
# act on the cells of a height grid:
#
#   sources        each sink is limited to what its donor holds over the time step (a limit on a rate
#                  between two reservoirs: nothing is made negative, what leaves one density enters the
#                  other). The densities of dry air and of all water of the cell do not change.
#   sedimentation  implicit in time and upwind, from the model top down: with λ = V Δt / Δz, a cell keeps
#                  (M + inflow) / (1 + λ) of its rain mass per unit area M = ρqʳ Δz and passes on λ times
#                  that. Positive for any λ. What reaches the ground is accumulated in a ledger.
#
# A model takes this path when its microphysics is built with
# `precipitation_boundary_condition = ColumnPhysicsPrecipitation()`: its rain then has no tendency from the
# microphysics and no sedimentation velocity in the dynamics, and the driver calls
# `height_coordinate_warm_rain_column_physics!` after each time step.
#
# Energy. The cells have fixed volume and exchange no heat in this operator, so what is conserved is their internal
# energy per unit volume plus the potential energy of their mass,
#
#   U = ρ (cᵛᵐ T - ℒˡ₀ qˡ),   cᵛᵐ = cᵖᵐ - Rᵐ,        (the enthalpy ρ (cᵖᵐ T - ℒˡ₀ qˡ) less the pressure p = ρ Rᵐ T)
#
# with Breeze's convention for the enthalpy of water (vapor cᵖᵛ T, liquid cˡ T - ℒˡ₀), in which liquid has no pressure
# term: its internal energy per unit mass is eˡ = cˡ T - ℒˡ₀.
#
#   sources        no mass and no energy crosses the cell: U and ρ are unchanged while rain becomes vapor or cloud
#                  becomes rain. T follows from U at the new composition, T⁺ = (U / ρ + ℒˡ₀ qˡ⁺) / cᵛᵐ⁺, the pressure from
#                  the equation of state, and θˡⁱ from ĉᵖ Π θˡⁱ = cᵖᵐ T - ℒˡ₀ qˡ. (At fixed θˡⁱ, which is what a pressure
#                  coordinate keeps, U would change by the pressure work that a fixed-volume cell cannot do.)
#   sedimentation  rain leaving a cell carries eˡ at the temperature of that cell, and the potential energy g z of the
#                  level it leaves. Falling to the level below it releases g (zₖ₊₁ - zₖ) per unit mass, which is
#                  dissipated as heat in the cell it enters: Uₖ⁺ = Uₖ + [inflow (eˡₖ₊₁ + g (zₖ₊₁ - zₖ)) - outflow eˡₖ] / Δz.
#                  Rain reaching the ground leaves with eˡ₁ + g z₁: the ledger `precipitation_energy` holds that sum.
#
# Then Σ (U + ρ g z) Δz + precipitation_energy is unchanged by the operator, to rounding. The saturation adjustment that
# the model applies when its state is diagnosed again is Breeze's own and is not part of this statement.

using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity, mixture_gas_constant,
                             vapor_equivalent_heat_capacity, condensate_latent_energy,
                             absolute_zero_latent_heat, dry_air_gas_constant
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: znode, Center
using Oceananigans.TimeSteppers: update_state!

struct ColumnPhysicsPrecipitation end

const ColumnWP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories, <:ColumnPhysicsPrecipitation}

# The rain does not fall in the stages, and has no source there
@inline function AM.write_sedimentation_velocity!(w_field, i, j, k, ::ColumnPhysicsPrecipitation, 𝕎)
    @inbounds w_field[i, j, k] = zero(𝕎)
    return nothing
end

# Where the scheme builds its tendencies in one fused kernel (`compute_fused_bulk_tendencies!`), that kernel is what
# must not run: the methods of `microphysical_tendency` below are not called on that path.
if isdefined(@__MODULE__, :compute_fused_bulk_tendencies!)
    AM.compute_microphysical_tendencies!(::ColumnWP1M, model) = nothing
end

@inline AM.microphysical_tendency(bμp::ColumnWP1M, ::Val{:ρqʳ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants) = zero(ρ)
@inline AM.microphysical_tendency(bμp::ColumnWP1M, ::Val{:ρqᵉ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants) = zero(ρ)
@inline AM.microphysical_tendency(bμp::ColumnWP1M, ::Val{:ρqᵉ}, ρ, ℳ, 𝒰, constants) = zero(ρ)

# The sources of `_apply_rain_sources!` of the hydrostatic column physics, on the cells of a height grid
@kernel function _apply_height_coordinate_rain_sources!(ρθ, ρqᵉ, ρqʳ, grid, microphysics, ρᵈ, ρ, T, qᵛ, qˡ, qᶜˡ, constants, pˢᵗ, κ, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρᵏ = ρ[i, j, k]
        ρqʳᵏ = ρqʳ[i, j, k]
        qʳ = ρqʳᵏ / ρᵏ
        qᶜˡᵏ = qᶜˡ[i, j, k]
        q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])

        conversion, evaporation = HydrostaticEquations.rain_source_rates(microphysics, ρᵏ, qᶜˡᵏ, qʳ, q, T[i, j, k], constants)
        conversion = ifelse(qᶜˡᵏ > 0, conversion, zero(conversion))
        evaporation = ifelse(qʳ > 0, evaporation, zero(evaporation))

        to_rain = min(ρᵏ * Δt * conversion, ρᵏ * max(0, qᶜˡᵏ))
        to_vapor = max(ρᵏ * Δt * evaporation, - max(0, ρqʳᵏ))
        δρqʳ = to_rain + to_vapor

        ρqʳ[i, j, k] = ρqʳᵏ + δρqʳ
        ρqᵉ[i, j, k] -= δρqʳ

        # Only the evaporation changes the liquid of the cell. Its internal energy does not change:
        # temperature at the new composition, then the potential temperature that diagnoses it
        ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
        cᵛᵐ = mixture_heat_capacity(q, constants) - mixture_gas_constant(q, constants)
        U = ρᵏ * (cᵛᵐ * T[i, j, k] - condensate_latent_energy(q, constants))
        q⁺ = MoistureMassFractions(q.vapor - to_vapor / ρᵏ, q.liquid + to_vapor / ρᵏ)
        θ⁺ = potential_temperature_from_internal_energy(U, ρᵏ, q⁺, constants, pˢᵗ, κ)
        ρθ[i, j, k] = ifelse(to_vapor != 0, ρᵈ[i, j, k] * θ⁺, ρθ[i, j, k])
    end
end

# θˡⁱ of a cell of density ρ and composition q that holds the internal energy U per unit volume
@inline function potential_temperature_from_internal_energy(U, ρ, q, constants, pˢᵗ, κ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    ℒq = condensate_latent_energy(q, constants)
    T = (U / ρ + ℒq) / (cᵖᵐ - Rᵐ)
    Π = (ρ * Rᵐ * T / pˢᵗ)^κ
    return (cᵖᵐ * T - ℒq) / (vapor_equivalent_heat_capacity(q, constants) * Π)
end

@kernel function _sediment_height_coordinate_rain!(ρθ, ρqʳ, ledger, grid, microphysics, ρᵈ, ρqᵉ, T, qᵛ, qᶜˡ,
                                                   constants, pˢᵗ, κ, g, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    @inbounds begin
        inflow = zero(grid)           # rain mass per unit area entering through the upper face
        energy_inflow = zero(grid)    # its internal energy, and the potential energy its fall released

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            zᵏ = znode(i, j, k, grid, Center(), Center(), Center())
            z⁻ = ifelse(k == 1, zero(zᵏ), znode(i, j, max(k - 1, 1), grid, Center(), Center(), Center()))
            ρqʳᵏ = ρqʳ[i, j, k]
            ρᵏ = ρᵈ[i, j, k] + ρqᵉ[i, j, k] + ρqʳᵏ
            ρqᵛ = qᵛ[i, j, k] * ρᵏ
            ρqᶜˡ = qᶜˡ[i, j, k] * ρᵏ
            Tᵏ = T[i, j, k]

            V = HydrostaticEquations.positive_rain_fall_speed(microphysics, ρᵏ, ρqʳᵏ / ρᵏ)
            λ = V * Δt / Δz
            M = ρqʳᵏ * Δz
            M⁺ = (M + inflow) / (1 + λ)
            outflow = λ * M⁺
            eˡ = cˡ * Tᵏ - ℒˡ₀
            crossed = (inflow != 0) | (outflow != 0)

            # Internal energy of the cell per unit volume before and after
            q = MoistureMassFractions(ρqᵛ / ρᵏ, (ρqᶜˡ + ρqʳᵏ) / ρᵏ)
            cᵛᵐ = mixture_heat_capacity(q, constants) - mixture_gas_constant(q, constants)
            U = ρᵏ * (cᵛᵐ * Tᵏ - condensate_latent_energy(q, constants))
            U⁺ = U + (energy_inflow - outflow * eˡ) / Δz

            ρqʳ⁺ = M⁺ / Δz
            ρ⁺ = ρᵈ[i, j, k] + ρqᵉ[i, j, k] + ρqʳ⁺
            q⁺ = MoistureMassFractions(ρqᵛ / ρ⁺, (ρqᶜˡ + ρqʳ⁺) / ρ⁺)
            θ⁺ = potential_temperature_from_internal_energy(U⁺, ρ⁺, q⁺, constants, pˢᵗ, κ)

            ρθ[i, j, k] = ifelse(crossed, ρᵈ[i, j, k] * θ⁺, ρθ[i, j, k])
            ρqʳ[i, j, k] = ifelse(crossed, ρqʳ⁺, ρqʳᵏ)

            # What leaves carries eˡ of this cell; falling to the level below (or leaving at the ground with its
            # potential energy) it releases g (zᵏ - z⁻), heat of the cell it enters
            inflow = outflow
            energy_inflow = outflow * (eˡ + ifelse(k == 1, g * zᵏ, g * (zᵏ - z⁻)))
        end

        ledger.surface_rain[i, j, 1] += inflow
        ledger.precipitation_energy[i, j, 1] += energy_inflow
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the column physics of WP1M to `model`, an `AtmosphereModel` on a height coordinate whose microphysics
was built with `precipitation_boundary_condition = ColumnPhysicsPrecipitation()`, over `Δt` in `substeps`
equal applications: the conversion of cloud liquid to rain and the evaporation of rain, then the
sedimentation of rain, each on the state the previous one leaves (`sources = false` or `sedimentation = false` leaves one out). `ledger` holds the fields `surface_rain`
(kg m⁻²) and `precipitation_energy` (J m⁻², internal plus potential energy), to which what reaches the ground is added. `regions` are the
regions of a multi-region grid, or `nothing`.
"""
function height_coordinate_warm_rain_column_physics!(model, ledger; Δt, substeps=1, regions=nothing, sources=true, sedimentation=true)
    microphysics = model.microphysics
    microphysics isa ColumnWP1M ||
        throw(ArgumentError("height_coordinate_warm_rain_column_physics! needs warm-phase one-moment microphysics built " *
                            "with precipitation_boundary_condition = ColumnPhysicsPrecipitation()."))

    constants = model.thermodynamic_constants
    FT = eltype(model.grid)
    κ = convert(FT, dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
    pˢᵗ = convert(FT, model.dynamics.standard_pressure)
    δt = convert(FT, Δt / substeps)
    g = convert(FT, constants.gravitational_acceleration)
    μ = model.microphysical_fields
    part(field, r) = isnothing(r) ? field : Oceananigans.Utils.getregion(field, r)
    parts(fields::NamedTuple, r) = map(field -> part(field, r), fields)
    each_region = isnothing(regions) ? (nothing,) : regions

    for n in 1:substeps
        for r in each_region
            sources || continue
            grid = part(model.grid, r)
            launch!(grid.architecture, grid, :xyz, _apply_height_coordinate_rain_sources!,
                    part(model.formulation.potential_temperature_density, r),
                    part(model.moisture_density, r), part(μ.ρqʳ, r), grid, microphysics,
                    part(model.dynamics.dry_density, r), part(AM.total_density(model.dynamics), r), part(model.temperature, r),
                    part(μ.qᵛ, r), part(μ.qˡ, r), part(μ.qᶜˡ, r), constants, pˢᵗ, κ, δt)
        end

        update_state!(model; compute_tendencies=false)

        for r in each_region
            sedimentation || continue
            grid = part(model.grid, r)
            launch!(grid.architecture, grid, :xy, _sediment_height_coordinate_rain!,
                    part(model.formulation.potential_temperature_density, r), part(μ.ρqʳ, r), parts(ledger, r),
                    grid, microphysics, part(model.dynamics.dry_density, r), part(model.moisture_density, r),
                    part(model.temperature, r), part(μ.qᵛ, r), part(μ.qᶜˡ, r),
                    constants, pˢᵗ, κ, g, δt)
        end

        update_state!(model; compute_tendencies = n == substeps)
    end

    return nothing
end
