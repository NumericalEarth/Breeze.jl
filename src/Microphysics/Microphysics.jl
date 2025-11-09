module Microphysics

export WarmPhaseSaturationAdjustment

using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    dry_air_gas_constant,
    vapor_gas_constant,
    saturation_vapor_pressure,
    saturation_specific_humidity,
    density,
    with_moisture,
    total_moisture_mass_fraction,
    MoistStaticEnergyState

using Oceananigans: CenterField

import ..AtmosphereModels:
    compute_temperature,
    prognostic_field_names,
    materialize_microphysical_fields

using Adapt: Adapt, adapt
using Oceananigans: Oceananigans

"""
    WarmPhaseSaturationAdjustment(reference_state, thermodynamics)

Simple warm-phase saturation adjustment microphysics that computes temperature
via a saturation adjustment similar to MoistAirBuoyancy, adapted for the
anelastic thermodynamic state used in AtmosphereModel.
"""
struct WarmPhaseSaturationAdjustment{FT}
    tolerance :: FT
end

function WarmPhaseSaturationAdjustment(FT::DataType=Oceananigans.defaults.FloatType; tolerance = 1e-3)
    tolerance = convert(FT, tolerance)
    return WarmPhaseSaturationAdjustment(tolerance)
end

function materialize_microphysical_fields(microphysics::WarmPhaseSaturationAdjustment, grid, boundary_conditions)
    liquid_density = CenterField(grid)
    vapor_density = CenterField(grid)
    return (; liquid_density, vapor_density)
end

prognostic_field_names(::WarmPhaseSaturationAdjustment) = tuple()

#####
##### Saturation adjustment utilities (copy-adapted from MoistAirBuoyancy)
#####

@inline function adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, thermo.liquid)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀::MoistStaticEnergyState, T, thermo)
    pᵣ = 𝒰₀.reference_pressure
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q₁ = MoistureMassFractions(qᵛ⁺, qˡ, zero(qˡ))
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰::MoistStaticEnergyState, thermo)
    q = 𝒰.moisture_mass_fractions
    e = 𝒰.moist_static_energy
    g = thermo.gravitational_acceleration
    z = 𝒰.height
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    qˡ = q.liquid

    # e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ
    return T - (e - g * z + ℒˡᵣ * qˡ) / cᵖᵐ
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
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    T₁ = e / cᵖᵐ

    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, thermo)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, thermo, thermo.liquid)
    qᵗ <= qᵛ⁺₁ && return T₁

    # Re-initialize first guess assuming saturation
    𝒰₁ = with_moisture(𝒰₀, q₁)
    qᵛ⁺₁ = adjustment_saturation_specific_humidity(T₁, pᵣ, qᵗ, thermo)
    qˡ₁ = qᵗ - qᵛ⁺₁
    q₁ = MoistureMassFractions(qᵛ⁺₁, qˡ₁, zero(qˡ₁))
    𝒰₁ = with_moisture(𝒰₀, q₁)

    # Generate a second guess
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    T₂ = T₁ + 1e-2 #ℒˡᵣ * qˡ₁ / cᵖᵐ
    𝒰₂ = adjust_state(𝒰₁, T₂, thermo)

    # Initialize secant iteration
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, thermo)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo)
    δ = microphysics.tolerance

    while abs(T₂ - T₁) > δ
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂
        𝒰₁ = 𝒰₂

        # Update
        T₂ -= r₂ * ΔTΔr
        𝒰₂ = adjust_state(𝒰₂, T₂, thermo)
        r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo)
    end

    return T₂
end

end # module Microphysics
