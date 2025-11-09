module Microphysics

export WarmPhaseSaturationAdjustment, compute_temperature

using ..Thermodynamics:
    MoistureMassFractions,
    ThermodynamicConstants,
    ReferenceState,
    mixture_heat_capacity,
    mixture_gas_constant,
    dry_air_gas_constant,
    vapor_gas_constant,
    saturation_vapor_pressure,
    density,
    total_specific_humidity,
    AnelasticThermodynamicState,
    MoistStaticEnergyState,
    adiabatic_hydrostatic_pressure

import ..Thermodynamics: saturation_specific_humidity

using Adapt: Adapt, adapt

"""
    WarmPhaseSaturationAdjustment(reference_state, thermodynamics)

Simple warm-phase saturation adjustment microphysics that computes temperature
via a saturation adjustment similar to MoistAirBuoyancy, adapted for the
anelastic thermodynamic state used in AtmosphereModel.
"""
struct WarmPhaseSaturationAdjustment{RS, AT}
    reference_state :: RS
    thermodynamics :: AT
end

Adapt.adapt_structure(to, m::WarmPhaseSaturationAdjustment) =
    WarmPhaseSaturationAdjustment(adapt(to, m.reference_state),
                                  adapt(to, m.thermodynamics))

#####
##### Saturation adjustment utilities (copy-adapted from MoistAirBuoyancy)
#####

@inline function exner(𝒰::AnelasticThermodynamicState, m::WarmPhaseSaturationAdjustment)
    p₀ = m.reference_state.base_pressure
    pᵣ = 𝒰.reference_pressure
    q = 𝒰.moisture_fractions
    Rᵐ = mixture_gas_constant(q, m.thermodynamics)
    cᵖᵐ = mixture_heat_capacity(q, m.thermodynamics)
    return (pᵣ / p₀)^(Rᵐ / cᵖᵐ)
end

@inline function adjustment_saturation_specific_humidity(T, 𝒰::AnelasticThermodynamicState, m::WarmPhaseSaturationAdjustment)
    pᵛ⁺ = saturation_vapor_pressure(T, m.thermodynamics, m.thermodynamics.liquid)
    pᵣ = 𝒰.reference_pressure
    qᵗ = total_specific_humidity(𝒰)
    Rᵈ = dry_air_gas_constant(m.thermodynamics)
    Rᵛ = vapor_gas_constant(m.thermodynamics)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀::AnelasticThermodynamicState, T, m::WarmPhaseSaturationAdjustment)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, 𝒰₀, m)
    qᵗ = total_specific_humidity(𝒰₀)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q₁ = MoistureMassFractions(qᵛ⁺, qˡ, zero(qˡ))
    return AnelasticThermodynamicState(𝒰₀.potential_temperature,
                                       q₁,
                                       𝒰₀.reference_density,
                                       𝒰₀.reference_pressure,
                                       𝒰₀.exner_function)
end

@inline function saturation_adjustment_residual(T, 𝒰::AnelasticThermodynamicState, m::WarmPhaseSaturationAdjustment)
    Π = exner(𝒰, m)
    q = 𝒰.moisture_fractions
    θ = 𝒰.potential_temperature
    ℒˡᵣ = m.thermodynamics.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, m.thermodynamics)
    qˡ = q.liquid
    return T - Π * θ - ℒˡᵣ * qˡ / cᵖᵐ
end

"""
    compute_temperature(state::AnelasticThermodynamicState, microphysics::WarmPhaseSaturationAdjustment)

Return the saturation-adjusted temperature using a secant iteration identical to
that used in MoistAirBuoyancy, adapted to AnelasticThermodynamicState.
"""
@inline function compute_temperature(𝒰₀::AnelasticThermodynamicState{FT}, m::WarmPhaseSaturationAdjustment) where FT
    θ = 𝒰₀.potential_temperature
    θ == 0 && return zero(FT)

    # Unsaturated initial guess
    qᵗ = total_specific_humidity(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰₁ = AnelasticThermodynamicState(θ, q₁, 𝒰₀.reference_density, 𝒰₀.reference_pressure, 𝒰₀.exner_function)
    Π₁ = exner(𝒰₁, m)
    T₁ = Π₁ * θ

    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, m.thermodynamics)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, m.thermodynamics, m.thermodynamics.liquid)
    qᵗ <= qᵛ⁺₁ && return T₁

    # If saturated, modify state to include qˡ
    qᵛ⁺₁ = adjustment_saturation_specific_humidity(T₁, 𝒰₁, m)
    qˡ₁ = qᵗ - qᵛ⁺₁
    q₁ = MoistureMassFractions(qᵛ⁺₁, qˡ₁, zero(qˡ₁))
    𝒰₁ = AnelasticThermodynamicState(θ, q₁, 𝒰₀.reference_density, 𝒰₀.reference_pressure, 𝒰₀.exner_function)

    # Second guess
    T₂ = T₁ + one(FT)
    𝒰₂ = adjust_state(𝒰₁, T₂, m)

    # Initialize secant iteration
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, m)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, m)
    δ = convert(FT, 1e-3)

    while abs(T₂ - T₁) > δ
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂
        𝒰₁ = 𝒰₂

        # Update
        T₂ -= r₂ * ΔTΔr
        𝒰₂ = adjust_state(𝒰₂, T₂, m)
        r₂ = saturation_adjustment_residual(T₂, 𝒰₂, m)
    end

    return T₂
end

#####
##### Moist static energy formulation
#####

@inline function adjustment_saturation_specific_humidity(T, 𝒰::MoistStaticEnergyState, m::WarmPhaseSaturationAdjustment)
    p₀ = m.reference_state.base_pressure
    θ₀ = m.reference_state.potential_temperature
    pᵣ = adiabatic_hydrostatic_pressure(𝒰.height, p₀, θ₀, m.thermodynamics)
    pᵛ⁺ = saturation_vapor_pressure(T, m.thermodynamics, m.thermodynamics.liquid)
    qᵗ = total_specific_humidity(𝒰)
    Rᵈ = dry_air_gas_constant(m.thermodynamics)
    Rᵛ = vapor_gas_constant(m.thermodynamics)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀::MoistStaticEnergyState, T, m::WarmPhaseSaturationAdjustment)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, 𝒰₀, m)
    qᵗ = total_specific_humidity(𝒰₀)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q₁ = MoistureMassFractions(qᵛ⁺, qˡ, zero(qˡ))
    return MoistStaticEnergyState(𝒰₀.moist_static_energy, q₁, 𝒰₀.height)
end

@inline function saturation_adjustment_residual(T, 𝒰::MoistStaticEnergyState, m::WarmPhaseSaturationAdjustment)
    q = 𝒰.moisture_fractions
    cᵖᵐ = mixture_heat_capacity(q, m.thermodynamics)
    g = m.thermodynamics.gravitational_acceleration
    ℒˡᵣ = m.thermodynamics.liquid.reference_latent_heat
    qᵗ = total_specific_humidity(𝒰)
    h = 𝒰.moist_static_energy
    z = 𝒰.height
    return h - (cᵖᵐ * T + g * z + ℒˡᵣ * qᵗ)
end

@inline function compute_temperature(𝒰₀::MoistStaticEnergyState{FT}, m::WarmPhaseSaturationAdjustment) where FT
    h = 𝒰₀.moist_static_energy
    z = 𝒰₀.height
    qᵗ = total_specific_humidity(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    cᵖᵐ = mixture_heat_capacity(q₁, m.thermodynamics)
    g = m.thermodynamics.gravitational_acceleration
    ℒˡᵣ = m.thermodynamics.liquid.reference_latent_heat

    # Dry/unsaturated initial guess from moist static energy definition
    T₁ = (h - g * z - ℒˡᵣ * qᵗ) / cᵖᵐ
    𝒰₁ = MoistStaticEnergyState(h, q₁, z)
    𝒰₁ = adjust_state(𝒰₁, T₁, m)

    # Second guess
    T₂ = T₁ + one(FT)
    𝒰₂ = adjust_state(𝒰₁, T₂, m)

    # Secant iteration on h - (cpm T + gz + L0 qᵗ)
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, m)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, m)
    δ = convert(FT, 1e-3)

    while abs(T₂ - T₁) > δ
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)
        r₁ = r₂
        T₁ = T₂
        𝒰₁ = 𝒰₂
        T₂ -= r₂ * ΔTΔr
        𝒰₂ = adjust_state(𝒰₂, T₂, m)
        r₂ = saturation_adjustment_residual(T₂, 𝒰₂, m)
    end

    return T₂
end

end # module Microphysics
