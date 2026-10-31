#####
##### Operator-split saturation adjustment of a prognostic cloud (CM1's `satadj`)
#####

using ..AtmosphereModels:
    AtmosphereModels,
    compute_microphysical_tendencies!,
    dynamics_density,
    dynamics_pressure,
    diagnose_thermodynamic_state,
    kernel_time_step,
    thermodynamic_density,
    thermodynamic_density_name,
    total_density

using ..PotentialTemperatureFormulations:
    PotentialTemperatureFormulation,
    DryPotentialTemperatureFormulation,
    IsochoricPressureWork,
    IsobaricPressureWork

using ..Thermodynamics: dry_air_gas_constant, vapor_gas_constant, absolute_zero_latent_heat, air_pressure

using DocStringExtensions: TYPEDEF

"""
$(TYPEDEF)

Saturation adjustment of a prognostic cloud liquid `ρqᶜˡ` and vapor `ρqᵛ`, applied once per time step after
the Runge-Kutta stages: the port of CM1's `satadj` (ptype = 6 with no fall speed), for which
[Bryan and Fritsch (2002)](@cite BryanFritsch2002) define their benchmark. At fixed density and
internal energy it iterates

```math
Δrᵛ = \\frac{rᵛ⁺ - rᵛ}{1 + d₁ rᵛ⁺ / (T - δT)²}, \\quad T → T - Δrᵛ \\frac{ℒˡ(T₀) - Rᵛ T₀}{cᵛᵐ},
\\quad d₁ = a (Tᵣ - δT) \\frac{ℒˡ(T₀) - Rᵛ T₀}{cᵛᵐ}
```

(mixing ratios and heat capacities per unit dry air), with the Tetens coefficients ``a``, ``δT`` and ``Tᵣ`` of the
model's `TetensFormula`, and writes the temperature it reaches into the thermodynamic variable.
It serves every potential temperature flavor and so isolates the choice of the variable from the
microphysics. A [`Dry`](@ref Breeze.PotentialTemperatureFormulations.Dry) flavor with
[`IsobaricPressureWork`](@ref Breeze.PotentialTemperatureFormulations.IsobaricPressureWork) adjusts
at fixed pressure instead, with ``ℒˡ / cᵖᵐ``.

With `stage_estimate = true`, the default, the Runge-Kutta stages feel the increments of the previous
adjustment as tendencies of the water and of `ρθ`, which are removed again before the next adjustment, so
that the dynamics sees the heating during the stages (`solve1.F:1747-1751`, `solve2.F:1561-1577`, `solve3.F:325-336`
in CM1). `tolerance` is the temperature change at which the iteration stops.

Available on `CompressibleDynamics` only.

```jldoctest
using Breeze

OperatorSplitSaturationAdjustment()

# output
OperatorSplitSaturationAdjustment(tolerance = 0.0002, stage_estimate = true)
```
"""
struct OperatorSplitSaturationAdjustment{FT, E}
    tolerance :: FT
    stage_estimate :: E # Val(true) or Val(false)
end

OperatorSplitSaturationAdjustment(FT = Oceananigans.defaults.FloatType; tolerance = 2e-4, stage_estimate = true) =
    OperatorSplitSaturationAdjustment(convert(FT, tolerance), Val(stage_estimate))

const OSSA = OperatorSplitSaturationAdjustment

Base.summary(m::OSSA{<:Any, Val{E}}) where E =
    string("OperatorSplitSaturationAdjustment(tolerance = ", m.tolerance, ", stage_estimate = ", E, ")")
Base.show(io::IO, m::OSSA) = print(io, summary(m))

function AtmosphereModels.validate_microphysics(::OSSA, thermodynamic_constants)
    thermodynamic_constants isa TetensFormulaThermodynamicConstants ||
        throw(ArgumentError("OperatorSplitSaturationAdjustment requires `thermodynamic_constants` with a `TetensFormula` " *
                            "saturation vapor pressure formulation."))
    return nothing
end

AtmosphereModels.prognostic_field_names(::OSSA) = (:ρqᶜˡ,)
AtmosphereModels.moisture_prognostic_name(::OSSA) = :ρqᵛ
AtmosphereModels.phase_change_source_applies(::OSSA) = false
AtmosphereModels.condensate_liquid_fraction(::OSSA, ::Val{:ρqᶜˡ}) = 1

function AtmosphereModels.materialize_microphysical_fields(::OSSA, grid, boundary_conditions)
    ρqᶜˡ = CenterField(grid, boundary_conditions=boundary_conditions.ρqᶜˡ)
    qᵛ  = CenterField(grid)
    qᶜˡ = CenterField(grid)
    # The increments of the last adjustment per unit time
    Ṡᵛ  = CenterField(grid)
    Ṡᶜˡ = CenterField(grid)
    Θ̇   = CenterField(grid)
    return (; ρqᶜˡ, qᵛ, qᶜˡ, Ṡᵛ, Ṡᶜˡ, Θ̇)
end

@inline function AtmosphereModels.microphysical_state(::OSSA, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    return AtmosphereModels.WarmRainState(qᶜˡ, zero(qᶜˡ))
end

@inline AtmosphereModels.microphysical_state(::OSSA, ρ, ::Nothing, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))
@inline AtmosphereModels.microphysical_state(::OSSA, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))

@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰, ::OSSA, qᵛ, constants) = 𝒰
@inline AtmosphereModels.microphysical_tendency(::OSSA, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

AtmosphereModels.liquid_mass_fraction(::OSSA, model) = model.microphysical_fields.qᶜˡ
AtmosphereModels.ice_mass_fraction(::OSSA, model) = nothing

@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::OSSA, ρ, qᵛ, μ)
    qᶜˡ = @inbounds μ.qᶜˡ[i, j, k]
    return MoistureMassFractions(qᵛ, qᶜˡ)
end

@inline function AtmosphereModels.update_microphysical_auxiliaries!(μ, i, j, k, grid, ::OSSA, ℳ::AtmosphereModels.WarmRainState, ρ, 𝒰, constants)
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end

#####
##### The Runge-Kutta stages feel the increments of the last adjustment
#####

AtmosphereModels.compute_microphysical_tendencies!(microphysics::OSSA, model) =
    add_adjustment_estimate!(microphysics.stage_estimate, model)

add_adjustment_estimate!(::Val{false}, model) = nothing

function add_adjustment_estimate!(::Val{true}, model)
    grid = model.grid
    G = model.timestepper.Gⁿ
    Gρθ = getproperty(G, thermodynamic_density_name(model.formulation))
    launch!(architecture(grid), grid, :xyz, _add_adjustment_estimate!, Gρθ, G.ρqᵛ, G.ρqᶜˡ, model.microphysical_fields)
    return nothing
end

@kernel function _add_adjustment_estimate!(Gρθ, Gρqᵛ, Gρqᶜˡ, μ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Gρθ[i, j, k] += μ.Θ̇[i, j, k]
        Gρqᵛ[i, j, k] += μ.Ṡᵛ[i, j, k]
        Gρqᶜˡ[i, j, k] += μ.Ṡᶜˡ[i, j, k]
    end
end

#####
##### The adjustment
#####

adjustment_pressure_work(::PotentialTemperatureFormulation) = IsochoricPressureWork()
adjustment_pressure_work(formulation::DryPotentialTemperatureFormulation) = formulation.flavor.pressure_work

function AtmosphereModels.microphysics_model_update!(microphysics::OSSA, model)
    grid = model.grid
    Δt = model.clock.last_Δt

    # No update before the first time step
    (isnan(Δt) || isinf(Δt) || Δt ≤ 0) && return nothing

    ρ = total_density(model.dynamics)
    ρᵈ = dynamics_density(model.dynamics)
    ρ === ρᵈ && throw(ArgumentError("OperatorSplitSaturationAdjustment needs CompressibleDynamics"))

    formulation = model.formulation
    launch!(architecture(grid), grid, :xyz, _operator_split_saturation_adjustment!,
            microphysics, adjustment_pressure_work(formulation), grid, kernel_time_step(architecture(grid), grid, Δt),
            formulation, model.dynamics, thermodynamic_density(formulation), formulation.potential_temperature,
            model.moisture_density, model.microphysical_fields, ρ, ρᵈ, model.thermodynamic_constants)

    # The kernel wrote the prognostic fields in the interior: refill the halos and recompute
    # the diagnostics and the tendencies that the first stage of the next step uses.
    update_state!(model)

    return nothing
end

@kernel function _operator_split_saturation_adjustment!(microphysics, pressure_work, grid, Δt, formulation, dynamics,
                                                        ρθ, θ, ρqᵛ, μ, ρ_field, ρᵈ_field, constants)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρ = ρ_field[i, j, k]
        ρᵈ = ρᵈ_field[i, j, k]

        # Remove the estimate the stages applied
        ρθ⁻, ρqᵛ⁻, ρqᶜˡ⁻ = without_stage_estimate(microphysics.stage_estimate, Δt, ρθ[i, j, k], ρqᵛ[i, j, k], μ.ρqᶜˡ[i, j, k],
                                                  μ.Θ̇[i, j, k], μ.Ṡᵛ[i, j, k], μ.Ṡᶜˡ[i, j, k])
        θ⁻ = ρθ⁻ / ρᵈ
        θ[i, j, k] = θ⁻
    end

    q⁻ = MoistureMassFractions(ρqᵛ⁻ / ρ, ρqᶜˡ⁻ / ρ)
    𝒰⁻ = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q⁻)
    T⁻ = temperature(𝒰⁻, constants)

    T⁺, rᵛ⁺, rᶜˡ⁺ = saturation_adjustment_step(pressure_work, T⁻, ρqᵛ⁻ / ρᵈ, ρqᶜˡ⁻ / ρᵈ, ρᵈ, microphysics.tolerance, constants)

    q⁺ = MoistureMassFractions(rᵛ⁺ * ρᵈ / ρ, rᶜˡ⁺ * ρᵈ / ρ)
    θ⁺ = adjusted_potential_temperature(pressure_work, 𝒰⁻, q⁺, T⁺, ρ, standard_pressure(dynamics), constants)

    ρθ⁺ = ρᵈ * θ⁺
    ρqᵛ⁺ = rᵛ⁺ * ρᵈ
    ρqᶜˡ⁺ = rᶜˡ⁺ * ρᵈ

    @inbounds begin
        θ[i, j, k] = θ⁺
        ρθ[i, j, k] = ρθ⁺
        ρqᵛ[i, j, k] = ρqᵛ⁺
        μ.ρqᶜˡ[i, j, k] = ρqᶜˡ⁺
        μ.Θ̇[i, j, k] = (ρθ⁺ - ρθ⁻) / Δt
        μ.Ṡᵛ[i, j, k] = (ρqᵛ⁺ - ρqᵛ⁻) / Δt
        μ.Ṡᶜˡ[i, j, k] = (ρqᶜˡ⁺ - ρqᶜˡ⁻) / Δt
    end
end

@inline without_stage_estimate(::Val{false}, Δt, ρθ, ρqᵛ, ρqᶜˡ, Θ̇, Ṡᵛ, Ṡᶜˡ) = (ρθ, ρqᵛ, ρqᶜˡ)
@inline without_stage_estimate(::Val{true}, Δt, ρθ, ρqᵛ, ρqᶜˡ, Θ̇, Ṡᵛ, Ṡᶜˡ) = (ρθ - Δt * Θ̇, ρqᵛ - Δt * Ṡᵛ, ρqᶜˡ - Δt * Ṡᶜˡ)

@inline adjusted_potential_temperature(::IsochoricPressureWork, 𝒰⁻, q⁺, T⁺, ρ, pˢᵗ, constants) =
    with_temperature(with_moisture(𝒰⁻, q⁺), T⁺, constants).potential_temperature

# The pressure is held fixed through the adjustment
@inline function adjusted_potential_temperature(::IsobaricPressureWork, 𝒰⁻, q⁺, T⁺, ρ, pˢᵗ, constants)
    κᵈ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    return T⁺ * (pˢᵗ / air_pressure(𝒰⁻, constants))^κᵈ
end

# The heat capacity that the condensation heats per unit dry air, the heat released per unit
# condensed vapor at T, and the pressure after the iteration at (T, rᵛ).
@inline adjustment_latent_heat(::IsochoricPressureWork, ℒ, Rᵛ, T) = ℒ - Rᵛ * T
@inline adjustment_latent_heat(::IsobaricPressureWork, ℒ, Rᵛ, T) = ℒ

@inline adjustment_heat_capacity(::IsochoricPressureWork, c, Rᵐ) = c - Rᵐ
@inline adjustment_heat_capacity(::IsobaricPressureWork, c, Rᵐ) = c

@inline adjustment_pressure(::IsochoricPressureWork, ρᵈ, Rᵐ, T, p₀) = ρᵈ * Rᵐ * T
@inline adjustment_pressure(::IsobaricPressureWork, ρᵈ, Rᵐ, T, p₀) = p₀

"""
$(TYPEDSIGNATURES)

Return `(T, rᵛ, rᶜˡ)` after the saturation adjustment of air at temperature `T₀` with mixing ratios `rᵛ₀` and
`rᶜˡ₀` and dry-air density `ρᵈ`: the iteration of CM1's `satadj` (`kessler.F:178-275`).
A cloud of negative mass (from advection) condenses vapor until it is empty.
"""
@inline function saturation_adjustment_step(pressure_work, T₀, rᵛ₀, rᶜˡ₀, ρᵈ, tolerance, constants::TetensFormulaThermodynamicConstants)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    tetens = constants.saturation_vapor_pressure
    pᵣ = tetens.reference_saturation_vapor_pressure
    Tᵣ = tetens.reference_temperature
    a = tetens.liquid_coefficient
    δT = tetens.liquid_temperature_offset
    ε = Rᵈ / Rᵛ

    Rᵐ₀ = Rᵈ + Rᵛ * rᵛ₀
    p₀ = ρᵈ * Rᵐ₀ * T₀

    # Quantities frozen through the iteration, per unit dry air
    c = cᵖᵈ + cᵖᵛ * rᵛ₀ + cˡ * max(0, rᶜˡ₀)
    ℒ = ℒ₀ + (cᵖᵛ - cˡ) * T₀
    t₁ = adjustment_latent_heat(pressure_work, ℒ, Rᵛ, T₀) / adjustment_heat_capacity(pressure_work, c, Rᵐ₀)
    d₁ = t₁ * a * (Tᵣ - δT)

    pᵛ⁺ = min(pᵣ * exp(a * (T₀ - Tᵣ) / (T₀ - δT)), p₀ / 2)
    rᵛ⁺ = ε * pᵛ⁺ / (p₀ - pᵛ⁺)

    T = T₀
    rᵛ = rᵛ₀
    rᶜˡ = rᶜˡ₀
    active = (rᶜˡ₀ > 1e-12) | (rᶜˡ₀ < 0) | (rᵛ₀ > rᵛ⁺)
    iteration = 0

    while active & (iteration < 100)
        iteration += 1
        Δrᵛ = (rᵛ⁺ - rᵛ) / (1 + d₁ * rᵛ⁺ / (T - δT)^2)
        Δrᵛ = min(Δrᵛ, rᶜˡ)
        Δrᵛ = max(Δrᵛ, 1e-20 - rᵛ)

        rᵛ += Δrᵛ
        rᶜˡ -= Δrᵛ
        T⁻ = T
        T -= Δrᵛ * t₁

        p = adjustment_pressure(pressure_work, ρᵈ, Rᵈ + Rᵛ * rᵛ, T, p₀)
        pᵛ⁺ = min(pᵣ * exp(a * (T - Tᵣ) / (T - δT)), p₀ / 2)
        rᵛ⁺ = ε * pᵛ⁺ / (p - pᵛ⁺)

        active = abs(T - T⁻) > tolerance
    end

    return T, rᵛ, rᶜˡ
end
