module Microphysics

export WarmPhaseSaturationAdjustment

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
    total_moisture_fraction,
    MoistStaticEnergyState,
    adiabatic_hydrostatic_pressure

import ..Thermodynamics: saturation_specific_humidity
import ..AtmosphereModels: compute_temperature

using Adapt: Adapt, adapt

"""
    WarmPhaseSaturationAdjustment(reference_state, thermodynamics)

Simple warm-phase saturation adjustment microphysics that computes temperature
via a saturation adjustment similar to MoistAirBuoyancy, adapted for the
anelastic thermodynamic state used in AtmosphereModel.
"""
struct WarmPhaseSaturationAdjustment{FT}
    tolerance :: FT
end

#####
##### Saturation adjustment utilities (copy-adapted from MoistAirBuoyancy)
#####

@inline function adjustment_saturation_specific_humidity(T, 𝒰::MoistStaticEnergyState, thermo)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, thermo.liquid)
    pᵣ = 𝒰.reference_pressure
    qᵗ = total_moisture_fraction(𝒰)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀::MoistStaticEnergyState, T, m::WarmPhaseSaturationAdjustment)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, 𝒰₀, m)
    qᵗ = total_moisture_fraction(𝒰₀)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q₁ = MoistureMassFractions(qᵛ⁺, qˡ, zero(qˡ))
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰::MoistStaticEnergyState, m::WarmPhaseSaturationAdjustment)
    Π = exner(𝒰, m)
    q = 𝒰.moisture_fractions
    θ = 𝒰.potential_temperature
    ℒˡᵣ = m.thermodynamics.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, m.thermodynamics)
    qˡ = q.liquid
    return T - Π * θ - ℒˡᵣ * qˡ / cᵖᵐ
end

"""
    compute_temperature(state::MoistStaticEnergyState, microphysics::WarmPhaseSaturationAdjustment)

Return the saturation-adjusted temperature using a secant iteration identical to
that used in MoistAirBuoyancy, adapted to MoistStaticEnergyState.
"""
@inline function compute_temperature(𝒰₀::MoistStaticEnergyState, microphysics::WarmPhaseSaturationAdjustment, thermo)
    FT = eltype(𝒰₀)
    e = 𝒰₀.moist_static_energy
    e == 0 && return zero(FT)

    # Unsaturated initial guess
    q = 𝒰₀.moisture_fractions
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    T₁ = e / cᵖᵐ

    # If saturated, modify state to include qˡ
    qᵛ⁺₁ = adjustment_saturation_specific_humidity(T₁, 𝒰₀, thermo)
    qˡ₁ = qᵗ - qᵛ⁺₁
    q₁ = MoistureMassFractions(qᵛ⁺₁, qˡ₁, zero(qˡ₁))
    𝒰₁ = MoistStaticEnergyState(e, q₁, 𝒰₀.height)

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

end # module Microphysics
