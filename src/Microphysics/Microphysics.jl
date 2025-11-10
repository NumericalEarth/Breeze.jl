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
    temperature,
    is_absolute_zero,
    with_moisture,
    total_moisture_mass_fraction,
    AbstractThermodynamicState

using Oceananigans: Oceananigans, CenterField
using DocStringExtensions: TYPEDSIGNATURES

import ..AtmosphereModels:
    compute_thermodynamic_state,
    update_microphysical_fields,
    prognostic_field_names,
    materialize_microphysical_fields

"""
    WarmPhaseSaturationAdjustment(reference_state, thermodynamics)

Simple warm-phase saturation adjustment microphysics that computes temperature
via a saturation adjustment.
"""
struct WarmPhaseSaturationAdjustment{FT}
    tolerance :: FT
end

function WarmPhaseSaturationAdjustment(FT::DataType=Oceananigans.defaults.FloatType; tolerance=1e-3)
    tolerance = convert(FT, tolerance)
    return WarmPhaseSaturationAdjustment(tolerance)
end

prognostic_field_names(::WarmPhaseSaturationAdjustment) = tuple()

function materialize_microphysical_fields(microphysics::WarmPhaseSaturationAdjustment, grid, boundary_conditions)
    liquid_mass_fraction = CenterField(grid)
    specific_humidity = CenterField(grid)
    return (; liquid_mass_fraction, specific_humidity)
end

@inline function update_microphysical_fields(microphysical_fields, ::WarmPhaseSaturationAdjustment, i, j, k, grid, 𝒰, thermo)
    qˡ = microphysical_fields.liquid_mass_fraction
    qᵛ = microphysical_fields.specific_humidity
    @inbounds begin
        qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
        qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    end
    return nothing
end

#####
##### Saturation adjustment utilities
#####

@inline function adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, thermo.liquid)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀, T, thermo)
    pᵣ = 𝒰₀.reference_pressure
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo)
    qˡ = max(0, qᵗ - qᵛ⁺)
    qᵛ = qᵗ - qˡ
    q₁ = MoistureMassFractions(qᵛ, qˡ, zero(qˡ))
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰₀, thermo)
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    pᵣ = 𝒰₀.reference_pressure

    # Adjust the moisture and compute a new temperature
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo)
    qˡ = max(0, qᵗ - qᵛ⁺)
    q = MoistureMassFractions(qᵛ⁺, qˡ, zero(qˡ))
    𝒰₁ = with_moisture(𝒰₀, q)
    T₁ = temperature(𝒰₁, thermo)

    return T - T₁
end

"""
$(TYPEDSIGNATURES)

Return the saturation-adjusted thermodynamic state using a secant iteration.
"""
@inline function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, microphysics::WarmPhaseSaturationAdjustment, thermo)
    FT = eltype(𝒰₀)
    is_absolute_zero(𝒰₀) && return 𝒰₀

    # Unsaturated initial guess
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰₁ = with_moisture(𝒰₀, q₁)
    T₁ = temperature(𝒰₁, thermo)

    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, thermo)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, thermo, thermo.liquid)
    qᵗ <= qᵛ⁺₁ && return 𝒰₁

    # Re-initialize first guess assuming saturation
    𝒰₁ = with_moisture(𝒰₀, q₁)
    qᵛ⁺₁ = adjustment_saturation_specific_humidity(T₁, pᵣ, qᵗ, thermo)
    qˡ₁ = qᵗ - qᵛ⁺₁
    q₁ = MoistureMassFractions(qᵛ⁺₁, qˡ₁, zero(qˡ₁))
    𝒰₁ = with_moisture(𝒰₀, q₁)

    # Generate a second guess
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    ΔT = ℒˡᵣ * qˡ₁ / cᵖᵐ
    T₂ = T₁ + ΔT / 2
    𝒰₂ = adjust_state(𝒰₁, T₂, thermo)

    # Initialize secant iteration
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, thermo)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo)
    δ = microphysics.tolerance
    iter = 0

    while abs(r₂) > δ
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
        iter += 1
    end

    return 𝒰₂
end

end # module Microphysics
