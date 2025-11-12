using ..Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    dry_air_gas_constant,
    vapor_gas_constant,
    PlanarLiquidSurface,
    PlanarIceSurface,
    PlanarMixedPhaseSurface,
    saturation_vapor_pressure,
    density,
    temperature,
    is_absolute_zero,
    with_moisture,
    total_moisture_mass_fraction,
    AbstractThermodynamicState

import ..Thermodynamics: saturation_specific_humidity

using Oceananigans: Oceananigans, CenterField
using DocStringExtensions: TYPEDSIGNATURES

import ..AtmosphereModels:
    compute_thermodynamic_state,
    update_microphysical_fields!,
    prognostic_field_names,
    materialize_microphysical_fields,
    moisture_mass_fractions

abstract type AbstractEquilibrium end

"""
    WarmPhaseSaturationAdjustment(reference_state, thermodynamics)

Simple warm-phase saturation adjustment microphysics that computes temperature
via a saturation adjustment.
"""
struct SaturationAdjustment{E, FT}
    tolerance :: FT
    maxiter :: FT
    equilibrium :: E
end

"""
    $(TYPEDSIGNATURES)

Return `SaturationAdjustment` microphysics representing an instantaneous adjustment to
`equilibrium` between condensates and water vapor, computed by a solver with `tolerance` and `maxiter`.

The options for `equilibrium` are
    * `WarmPhaseEquilibrium` represneting an equilibrium between water vapor and liquid water.

    * `MixedPhaseEquilibrium` representing a temperature-dependent equilibrium between
      water vapor, possibly supercooled liquid water, and ice. The equilibrium state is modeled as a linear
      variation of the equilibrium liquid fraction with temperature, between
      the freezing temperature (e.g. 273.15 K) below which liquid water is supercooled,
      and the temperature of homogeneous ice nucleation temperature (e.g. 233.15 K) at which
      the supercooled liquid fraction vanishes.
"""
function SaturationAdjustment(FT::DataType=Oceananigans.defaults.FloatType;
                              tolerance = 1e-3,
                              maxiter = Inf,
                              equilibrium = MixedPhaseEquilibrium(FT))
    tolerance = convert(FT, tolerance)
    maxiter = convert(FT, maxiter)
    return SaturationAdjustment(tolerance, maxiter, equilibrium)
end

#####
##### Warm-phase equilibrium
#####

"""
    $(TYPEDSIGNATURES)

Return `WarmPhaseEquilibrium` representing an equilibrium between water vapor and liquid water.
"""
struct WarmPhaseEquilibrium <: AbstractEquilibrium end
@inline equilibrated_surface(::WarmPhaseEquilibrium, T) = PlanarLiquidSurface()

@inline function equilibrated_moisture_mass_fractions(::WarmPhaseEquilibrium, T, qᵗ)
    qˡ = max(0, qᵗ - qᵛ⁺)
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ, zero(qˡ))
end

#####
##### Mixed-phase equilibrium
#####

struct MixedPhaseEquilibrium{FT} <: AbstractEquilibrium
    freezing_temperature :: FT
    homogeneous_ice_nucleation_temperature :: FT
end

"""
    $(TYPEDSIGNATURES)

Return `MixedPhaseEquilibrium` representing a temperature-dependent equilibrium between
water vapor, possibly supercooled liquid water, and ice.

The equilibrium state is modeled as a linear variation of the equilibrium liquid fraction with temperature,
between the freezing temperature (e.g. 273.15 K) below which liquid water is supercooled,
and the temperature of homogeneous ice nucleation temperature (e.g. 233.15 K) at which
the supercooled liquid fraction vanishes.
"""
function MixedPhaseEquilibrium(FT = Oceananigans.defaults.FloatType;
                               freezing_temperature = 273.15,
                               homogeneous_ice_nucleation_temperature = 233.15)
    freezing_temperature = convert(FT, freezing_temperature)
    homogeneous_ice_nucleation_temperature = convert(FT, homogeneous_ice_nucleation_temperature)
    return MixedPhaseEquilibrium(freezing_temperature, homogeneous_ice_nucleation_temperature)
end

@inline function equilibrated_surface(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    T′ = clamp(T, Tʰ, Tᶠ)
    λ = (T′ - Tʰ) / (Tᶠ - Tʰ)
    return PlanarMixedPhaseSurface(λ)
end

@inline function equilibrated_moisture_mass_fractions(equilibrium::MixedPhaseEquilibrium, T, qᵗ)
    surface = equilibrated_surface(equilibrium, T)
    λ = surface.liquid_fraction
    qᶜ = max(0, qᵗ - qᵛ⁺)
    qˡ = λ * qᶜ
    qⁱ = (1 - λ) * qᶜ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

const WarmPhaseSaturationAdjustment{FT} = SaturationAdjustment{WarmPhaseEquilibrium, FT} where FT
const MixedPhaseSaturationAdjustment{FT} = SaturationAdjustment{MixedPhaseEquilibrium{FT}, FT} where FT

const WPSA = WarmPhaseSaturationAdjustment
const MPSA = MixedPhaseSaturationAdjustment

prognostic_field_names(::WPSA) = tuple()
prognostic_field_names(::MPSA) = tuple()

function materialize_microphysical_fields(::WPSA, grid, boundary_conditions)
    liquid_mass_fraction = CenterField(grid)
    specific_humidity = CenterField(grid)
    return (; liquid_mass_fraction, specific_humidity)
end

function materialize_microphysical_fields(::MPSA, grid, boundary_conditions)
    ice_mass_fraction = CenterField(grid)
    liquid_mass_fraction = CenterField(grid)
    specific_humidity = CenterField(grid)
    return (; ice_mass_fraction, liquid_mass_fraction, specific_humidity)
end

@inline function update_microphysical_fields!(microphysical_fields, ::WPSA, i, j, k, grid, 𝒰, thermo)
    qˡ = microphysical_fields.liquid_mass_fraction
    qᵛ = microphysical_fields.specific_humidity
    @inbounds begin
        qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
        qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
    end
    return nothing
end

@inline function update_microphysical_fields!(microphysical_fields, ::MPSA, i, j, k, grid, 𝒰, thermo)
    qᵛ = microphysical_fields.specific_humidity
    qˡ = microphysical_fields.liquid_mass_fraction
    qⁱ = microphysical_fields.ice_mass_fraction
    @inbounds begin
        qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
        qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
        qⁱ[i, j, k] = 𝒰.moisture_mass_fractions.ice
    end
    return nothing
end 

@inline function moisture_mass_fractions(i, j, k, grid, ::WPSA, μ, qᵗ)
    @inbounds begin
        qᵛ = μ.microphysical_fields.specific_humidity[i, j, k]
        qˡ = μ.microphysical_fields.liquid_mass_fraction[i, j, k]
    end
    return MoistureMassFractions(qᵛ, qˡ, zero(qᵛ))
end

@inline function moisture_mass_fractions(i, j, k, grid, ::MPSA, μ, qᵗ)
    @inbounds begin
        qᵛ = μ.microphysical_fields.specific_humidity[i, j, k]
        qˡ = μ.microphysical_fields.liquid_mass_fraction[i, j, k]
        qⁱ = μ.microphysical_fields.ice_mass_fraction[i, j, k]
    end
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Saturation adjustment utilities
#####

@inline function saturation_specific_humidity(T, ρ, thermo, equilibrium::AbstractEquilibrium)
    surface = equilibrated_surface(equilibrium, T)
    return saturation_specific_humidity(T, ρ, thermo, surface)
end

@inline function adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equilibrium)
    surface = equilibrated_surface(equilibrium, T)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, surface)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀, T, thermo, equilibrium)
    pᵣ = 𝒰₀.reference_pressure
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equilibrium)
    q = equilibrated_moisture_mass_fractions(equilibrium, T, qᵗ)
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰₀, thermo, equilibrium)
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    pᵣ = 𝒰₀.reference_pressure

    # Adjust the moisture and compute a new temperature
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equilibrium)
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
@inline function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, microphysics::SaturationAdjustment, thermo)
    FT = eltype(𝒰₀)
    is_absolute_zero(𝒰₀) && return 𝒰₀
    equilibrium = microphysics.equilibrium

    # Unsaturated initial guess
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ, zero(qᵗ), zero(qᵗ))
    𝒰₁ = with_moisture(𝒰₀, q₁)
    T₁ = temperature(𝒰₁, thermo)

    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, thermo)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, thermo, equilibrium)
    qᵗ <= qᵛ⁺₁ && return 𝒰₁

    # Re-initialize first guess assuming saturation
    𝒰₁ = with_moisture(𝒰₀, q₁)
    qᵛ⁺₁ = adjustment_saturation_specific_humidity(T₁, pᵣ, qᵗ, thermo, equilibrium)
    qˡ₁ = qᵗ - qᵛ⁺₁
    q₁ = MoistureMassFractions(qᵛ⁺₁, qˡ₁, zero(qˡ₁))
    𝒰₁ = with_moisture(𝒰₀, q₁)

    # Generate a second guess
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    ΔT = ℒˡᵣ * qˡ₁ / cᵖᵐ
    T₂ = T₁ + ΔT / 2
    𝒰₂ = adjust_state(𝒰₁, T₂, thermo, equilibrium)

    # Initialize secant iteration
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, thermo, equilibrium)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo, equilibrium)
    δ = microphysics.tolerance
    iter = 0

    while abs(r₂) > δ && iter < microphysics.maxiter
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂
        𝒰₁ = 𝒰₂

        # Update
        T₂ -= r₂ * ΔTΔr
        𝒰₂ = adjust_state(𝒰₂, T₂, thermo, equilibrium)
        r₂ = saturation_adjustment_residual(T₂, 𝒰₂, thermo, equilibrium)
        iter += 1
    end

    return 𝒰₂
end
