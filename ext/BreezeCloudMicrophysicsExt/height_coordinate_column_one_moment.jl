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
#                  other). θˡⁱ, the densities of dry air and of all water of the cell do not change.
#   sedimentation  implicit in time and upwind, from the model top down: with λ = V Δt / Δz, a cell keeps
#                  (M + inflow) / (1 + λ) of its rain mass per unit area M = ρqʳ Δz and passes on λ times
#                  that. Positive for any λ. The rain that leaves a cell carries the enthalpy of liquid water
#                  at the temperature of that cell, hˡ = cˡ T - ℒˡ₀, and the potential temperature of every
#                  cell the rain crossed is expressed again from its enthalpy ρ (cᵖᵐ T - ℒˡ₀ qˡ) = ρ ĉᵖ Π θˡⁱ
#                  at its new composition. What reaches the ground is accumulated in a ledger.
#
# A model takes this path when its microphysics is built with
# `precipitation_boundary_condition = ColumnPhysicsPrecipitation()`: its rain then has no tendency from the
# microphysics and no sedimentation velocity in the dynamics, and the driver calls
# `height_coordinate_warm_rain_column_physics!` after each time step.
#
# The enthalpy of the cells is conserved by the sedimentation, as in the hydrostatic column physics. The cells
# have fixed volume, so the conserved energy of the dynamics is the internal energy: the pressure work of the
# change of mass of a cell is not accounted for. It is of the order of Rᵐ T / (cᵖ T) of the enthalpy that the
# rain carries.

using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity, mixture_gas_constant,
                             vapor_equivalent_heat_capacity, condensate_latent_energy,
                             absolute_zero_latent_heat, dry_air_gas_constant
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!

struct ColumnPhysicsPrecipitation end

const ColumnWP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories, <:ColumnPhysicsPrecipitation}

# The rain does not fall in the stages, and has no source there
@inline function AM.write_sedimentation_velocity!(w_field, i, j, k, ::ColumnPhysicsPrecipitation, 𝕎)
    @inbounds w_field[i, j, k] = zero(𝕎)
    return nothing
end

@inline AM.microphysical_tendency(bμp::ColumnWP1M, ::Val{:ρqʳ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants) = zero(ρ)
@inline AM.microphysical_tendency(bμp::ColumnWP1M, ::Val{:ρqᵉ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants) = zero(ρ)
@inline AM.microphysical_tendency(bμp::ColumnWP1M, ::Val{:ρqᵉ}, ρ, ℳ, 𝒰, constants) = zero(ρ)

# The sources of `_apply_rain_sources!` of the hydrostatic column physics, on the cells of a height grid
@kernel function _apply_height_coordinate_rain_sources!(ρqᵉ, ρqʳ, grid, microphysics, ρ, T, qᵛ, qˡ, qᶜˡ, constants, Δt)
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
    end
end

@kernel function _sediment_height_coordinate_rain!(ρθ, ρqʳ, ledger, grid, microphysics, ρᵈ, ρqᵉ, θ, T, p, qᵛ, qᶜˡ,
                                                   constants, pˢᵗ, κ, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    @inbounds begin
        inflow = zero(grid)           # rain mass per unit area entering through the upper face
        enthalpy_inflow = zero(grid)

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
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
            hˡ = cˡ * Tᵏ - ℒˡ₀
            crossed = (inflow != 0) | (outflow != 0)

            # Enthalpy of the cell per unit volume before, ρ ĉᵖ Π θ, and after
            q = MoistureMassFractions(ρqᵛ / ρᵏ, (ρqᶜˡ + ρqʳᵏ) / ρᵏ)
            Π = (p[i, j, k] / pˢᵗ)^κ
            𝓗 = ρᵏ * vapor_equivalent_heat_capacity(q, constants) * Π * θ[i, j, k]
            𝓗⁺ = 𝓗 + (enthalpy_inflow - outflow * hˡ) / Δz

            # The cell with its new rain: temperature from its enthalpy, pressure from the equation of state
            ρqʳ⁺ = M⁺ / Δz
            ρ⁺ = ρᵈ[i, j, k] + ρqᵉ[i, j, k] + ρqʳ⁺
            q⁺ = MoistureMassFractions(ρqᵛ / ρ⁺, (ρqᶜˡ + ρqʳ⁺) / ρ⁺)
            T⁺ = (𝓗⁺ / ρ⁺ + condensate_latent_energy(q⁺, constants)) / mixture_heat_capacity(q⁺, constants)
            p⁺ = ρ⁺ * mixture_gas_constant(q⁺, constants) * T⁺
            Π⁺ = (p⁺ / pˢᵗ)^κ
            θ⁺ = 𝓗⁺ / (ρ⁺ * vapor_equivalent_heat_capacity(q⁺, constants) * Π⁺)

            ρθ[i, j, k] = ifelse(crossed, ρᵈ[i, j, k] * θ⁺, ρθ[i, j, k])
            ρqʳ[i, j, k] = ifelse(crossed, ρqʳ⁺, ρqʳᵏ)

            inflow = outflow
            enthalpy_inflow = outflow * hˡ
        end

        ledger.surface_rain[i, j, 1] += inflow
        ledger.precipitation_energy[i, j, 1] += enthalpy_inflow
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the column physics of WP1M to `model`, an `AtmosphereModel` on a height coordinate whose microphysics
was built with `precipitation_boundary_condition = ColumnPhysicsPrecipitation()`, over `Δt` in `substeps`
equal applications: the conversion of cloud liquid to rain and the evaporation of rain, then the
sedimentation of rain, each on the state the previous one leaves. `ledger` holds the fields `surface_rain`
(kg m⁻²) and `precipitation_energy` (J m⁻²), to which what reaches the ground is added. `regions` are the
regions of a multi-region grid, or `nothing`.
"""
function height_coordinate_warm_rain_column_physics!(model, ledger; Δt, substeps=1, regions=nothing)
    microphysics = model.microphysics
    microphysics isa ColumnWP1M ||
        throw(ArgumentError("height_coordinate_warm_rain_column_physics! needs warm-phase one-moment microphysics built " *
                            "with precipitation_boundary_condition = ColumnPhysicsPrecipitation()."))

    constants = model.thermodynamic_constants
    FT = eltype(model.grid)
    κ = convert(FT, dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
    pˢᵗ = convert(FT, model.dynamics.standard_pressure)
    δt = convert(FT, Δt / substeps)
    μ = model.microphysical_fields
    part(field, r) = isnothing(r) ? field : Oceananigans.Utils.getregion(field, r)
    parts(fields::NamedTuple, r) = map(field -> part(field, r), fields)
    each_region = isnothing(regions) ? (nothing,) : regions

    for n in 1:substeps
        for r in each_region
            grid = part(model.grid, r)
            launch!(grid.architecture, grid, :xyz, _apply_height_coordinate_rain_sources!,
                    part(model.moisture_density, r), part(μ.ρqʳ, r), grid, microphysics,
                    part(AM.total_density(model.dynamics), r), part(model.temperature, r),
                    part(μ.qᵛ, r), part(μ.qˡ, r), part(μ.qᶜˡ, r), constants, δt)
        end

        update_state!(model; compute_tendencies=false)

        for r in each_region
            grid = part(model.grid, r)
            launch!(grid.architecture, grid, :xy, _sediment_height_coordinate_rain!,
                    part(model.formulation.potential_temperature_density, r), part(μ.ρqʳ, r), parts(ledger, r),
                    grid, microphysics, part(model.dynamics.dry_density, r), part(model.moisture_density, r),
                    part(model.formulation.potential_temperature, r), part(model.temperature, r),
                    part(AM.dynamics_pressure(model.dynamics), r), part(μ.qᵛ, r), part(μ.qᶜˡ, r),
                    constants, pˢᵗ, κ, δt)
        end

        update_state!(model; compute_tendencies = n == substeps)
    end

    return nothing
end
