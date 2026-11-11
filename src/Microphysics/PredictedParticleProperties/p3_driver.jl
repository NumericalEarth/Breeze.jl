using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: Center, ZeroField
using Oceananigans.Grids: inactive_cell
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: Utils, launch!

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

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

@kernel function _p3_add_tendencies_kernel!(G, stage, μ, formulation, dynamics, grid, constants, p3, ρ_field, velocities)
    i, j, k = @index(Global, NTuple)

    inputs = p3_grid_process_inputs(i, j, k, grid, p3, μ, formulation, dynamics,
                                    constants, ρ_field, velocities)

    rates = compute_p3_process_rates(p3, inputs.ρ, inputs.ℳ, inputs.𝒰, constants, inputs.properties,
                                     inputs.surface_temperature, inputs.temperature_tendency,
                                     inputs.pressure_tendency)

    result = p3_tendencies(p3, inputs.ρ, inputs.ℳ, inputs.properties, rates)

    # Read `G` before the process tendencies land on it: the relaxations are advanced
    # implicitly from what every other tendency leaves of each number.
    relaxations = p3_number_relaxations(i, j, k, G, stage, p3, inputs.ρ, inputs.ℳ,
                                        inputs.properties, rates)

    add_p3_tendencies!(G, i, j, k, grid, p3, result)
    add_p3_number_relaxations!(G, i, j, k, p3, relaxations)
end

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
    @inbounds Gᵒ = G[i, j, k]
    c̃ = c⁰ + Δt * (Gᵒ + Gᵖ - explicit)
    implicit = (ρnᵗ - c̃) / (τ + Δt)
    semi_implicit = active & isfinite(Δt) & (Δt > 0)
    return ifelse(semi_implicit, implicit - explicit, zero(explicit))
end

@inline function p3_number_relaxations(i, j, k, G, stage, p3::P3, ρ, ℳ, properties, rates)
    τ = p3.process_rates.sink_limiting_timescale

    # Rain: the correction relaxes the raw number towards the slope-bounded one.
    rain_explicit = ρ * rates.rain_number_correction
    rain_target = ρ * ℳ.nʳ + τ * rain_explicit
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
    target = ρ * ℳ.nᶜˡ + τ * explicit
    Gᵖ = tendency_ρnᶜˡ(rates, ρ, properties.Nᶜˡ, ℳ.qᶜˡ, p3)
    return semi_implicit_relaxation(i, j, k, G.ρnᶜˡ, stage, Val(:ρnᶜˡ), Gᵖ, explicit, target, τ,
                                    explicit != 0)
end

@inline function add_p3_number_relaxations!(G, i, j, k, p3::P3, relaxations)
    @inbounds begin
        G.ρnʳ[i, j, k] += relaxations.rain
        G.ρnⁱ[i, j, k] += relaxations.ice
    end
    add_p3_cloud_number_relaxation!(G, i, j, k, p3.aerosol, relaxations)
    return nothing
end

@inline add_p3_cloud_number_relaxation!(G, i, j, k, ::Nothing, relaxations) = nothing

@inline function add_p3_cloud_number_relaxation!(G, i, j, k, ::AerosolActivation, relaxations)
    @inbounds G.ρnᶜˡ[i, j, k] += relaxations.cloud
    return nothing
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

# The number densities that are relaxed towards their bounds.
@inline p3_number_fields(fields, p3::P3) =
    merge((; fields.ρnʳ, fields.ρnⁱ), p3_cloud_number_fields(fields, p3.aerosol))

@inline p3_cloud_number_fields(fields, ::Nothing) = (;)
@inline p3_cloud_number_fields(fields, ::AerosolActivation) = (; fields.ρnᶜˡ)

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

    stage = AM.StageStart(model, fields -> p3_number_fields(fields, p3))

    launch!(arch, grid, :xyz, _p3_add_tendencies_kernel!,
            p3_tendency_fields(model.timestepper.Gⁿ, p3), stage,
            μ, model.formulation, model.dynamics, grid,
            model.thermodynamic_constants, p3, ρ_field,
            model.velocities)

    return nothing
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
