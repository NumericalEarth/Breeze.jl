using ..Thermodynamics:
    ThermodynamicConstants,
    dry_air_gas_constant,
    vapor_gas_constant,
    density,
    temperature,
    MoistureMassFractions,
    mixture_heat_capacity,
    PlanarLiquidSurface,
    PlanarMixedPhaseSurface,
    saturation_vapor_pressure,
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

struct SaturationAdjustment{E, FT}
    tolerance :: FT
    maxiter :: FT
    equilibrium :: E
end

"""
$(TYPEDSIGNATURES)

Return `SaturationAdjustment` microphysics representing an instantaneous adjustment
to `equilibrium` between condensates and water vapor, computed by a solver with
`tolerance` and `maxiter`.

The options for `equilibrium` are:

* [`WarmPhaseEquilibrium()`](@ref WarmPhaseEquilibrium) representing an equilibrium between
  water vapor and liquid water.

* [`MixedPhaseEquilibrium()`](@ref MixedPhaseEquilibrium) representing a temperature-dependent
  equilibrium between water vapor, possibly supercooled liquid water, and ice. The equilibrium
  state is modeled as a linear variation of the equilibrium liquid fraction with temperature,
  between the freezing temperature (e.g. 273.15 K) below which liquid water is supercooled,
  and the temperature of homogeneous ice nucleation temperature (e.g. 233.15 K) at which
  the supercooled liquid fraction vanishes.
"""
function SaturationAdjustment(FT::DataType=Oceananigans.defaults.FloatType;
                              tolerance = 1e-3,
                              maxiter = Inf,
                              equilibrium = MixedPhaseEquilibrium(FT))
    tolerance = convert(FT, tolerance)
    maxiter = convert(FT, maxiter)
    equilibrium = convert_equilibrium(FT, equilibrium)
    return SaturationAdjustment(tolerance, maxiter, equilibrium)
end

convert_equilibrium(FT, equil) = equil # fallback 
abstract type AbstractEquilibrium end

#####
##### Warm-phase equilibrium
#####

"""
$(TYPEDSIGNATURES)

Return `WarmPhaseEquilibrium` representing an equilibrium between water vapor and liquid water.
"""
struct WarmPhaseEquilibrium <: AbstractEquilibrium end
@inline equilibrated_surface(::WarmPhaseEquilibrium, T) = PlanarLiquidSurface()
convert_equilibrium(FT, ::WarmPhaseEquilibrium) = WarmPhaseEquilibrium()

@inline function equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, ::WarmPhaseEquilibrium)
    qˡ = max(0, qᵗ - qᵛ⁺)
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

#####
##### Mixed-phase equilibrium
#####

struct MixedPhaseEquilibrium{FT} <: AbstractEquilibrium
    freezing_temperature :: FT
    homogeneous_ice_nucleation_temperature :: FT
end

function convert_equilibrium(FT, equilibrium::MixedPhaseEquilibrium)
    Tᶠ = convert(FT, equilibrium.freezing_temperature)
    Tʰ = convert(FT, equilibrium.homogeneous_ice_nucleation_temperature)
    return MixedPhaseEquilibrium{FT}(Tᶠ, Tʰ)
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

    if freezing_temperature < homogeneous_ice_nucleation_temperature
        throw(ArgumentError("`freezing_temperature` must be greater than `homogeneous_ice_nucleation_temperature`"))
    end

    Tᶠ = convert(FT, freezing_temperature)
    Tʰ = convert(FT, homogeneous_ice_nucleation_temperature)
    return MixedPhaseEquilibrium{FT}(Tᶠ, Tʰ)
end

@inline function equilibrated_surface(equilibrium::MixedPhaseEquilibrium{FT}, T::FT) where FT
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    T′ = clamp(T, Tʰ, Tᶠ)
    λ = (T′ - Tʰ) / (Tᶠ - Tʰ)
    return PlanarMixedPhaseSurface(λ)
end

@inline function equilibrated_moisture_mass_fractions(T::FT, qᵗ::FT, qᵛ⁺::FT, equilibrium::MixedPhaseEquilibrium{FT}) where FT
    surface = equilibrated_surface(equilibrium, T)
    λ = surface.liquid_fraction
    qᶜ = max(0, qᵗ - qᵛ⁺)
    qᵛ = qᵗ - qᶜ
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

center_field_tuple(grid, names...) = NamedTuple{names}(CenterField(grid) for name in names)
materialize_microphysical_fields(::WPSA, grid, bcs) = center_field_tuple(grid, :qᵛ, :qˡ)
materialize_microphysical_fields(::MPSA, grid, bcs) = center_field_tuple(grid, :qᵛ, :qˡ, :qⁱ)

@inline @inbounds function update_microphysical_fields!(μ, ::WPSA, i, j, k, grid, 𝒰, thermo)
    μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    μ.qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
    return nothing
end

@inline @inbounds function update_microphysical_fields!(μ, ::MPSA, i, j, k, grid, 𝒰, thermo)
    μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    μ.qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
    μ.qⁱ[i, j, k] = 𝒰.moisture_mass_fractions.ice
    return nothing
end

@inline @inbounds function moisture_mass_fractions(i, j, k, grid, ::WPSA, μ, qᵗ)
    qᵛ = μ.qᵛ[i, j, k]
    qˡ = μ.qˡ[i, j, k]
    return MoistureMassFractions(qᵛ, qˡ)
end

@inline @inbounds function moisture_mass_fractions(i, j, k, grid, ::MPSA, μ, qᵗ)
    qᵛ = μ.qᵛ[i, j, k]
    qˡ = μ.qˡ[i, j, k]
    qⁱ = μ.qⁱ[i, j, k]
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Saturation adjustment utilities
#####

@inline function saturation_specific_humidity(T, ρ, thermo, equilibrium::AbstractEquilibrium)
    surface = equilibrated_surface(equilibrium, T)
    return saturation_specific_humidity(T, ρ, thermo, surface)
end

@inline function adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equil)
    surface = equilibrated_surface(equil, T)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, surface)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    ϵᵈᵛ = Rᵈ / Rᵛ
    return ϵᵈᵛ * (1 - qᵗ) * pᵛ⁺ / (pᵣ - pᵛ⁺)
end

@inline function adjust_state(𝒰₀::AbstractThermodynamicState{FT}, T::FT,
                              thermo::ThermodynamicConstants{FT},
                              equilibrium::MixedPhaseEquilibrium{FT}) where FT

    pᵣ = 𝒰₀.reference_pressure
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, thermo, equilibrium)
    q₁ = equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium)
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T::FT, 𝒰₀::AbstractThermodynamicState{FT},
                                                thermo::ThermodynamicConstants{FT},
                                                equilibrium::MixedPhaseEquilibrium{FT}) where FT

    𝒰₁ = adjust_state(𝒰₀, T, thermo, equilibrium)
    T₁ = temperature(𝒰₁, thermo)
    return T - T₁
end

"""
$(TYPEDSIGNATURES)

Return the saturation-adjusted thermodynamic state using a secant iteration.
"""
@inline function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState{FT},
                                             microphysics::SaturationAdjustment,
                                             thermo::ThermodynamicConstants{FT}) where FT
    # FT = eltype(𝒰₀)
    is_absolute_zero(𝒰₀) && return 𝒰₀

    # Compute an initial guess assuming unsaturated conditions
    qᵗ = total_moisture_mass_fraction(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ)
    𝒰₁ = with_moisture(𝒰₀, q₁)
    T₁ = temperature(𝒰₁, thermo)

    equilibrium = microphysics.equilibrium
    pᵣ = 𝒰₀.reference_pressure
    ρ₁ = density(pᵣ, T₁, q₁, thermo)
    qᵛ⁺₁ = saturation_specific_humidity(T₁, ρ₁, thermo, equilibrium)
    qᵗ <= qᵛ⁺₁ && return 𝒰₁

    # If we made it here, the state is saturated.
    # So, we re-initialize our first guess assuming saturation
    𝒰₁ = adjust_state(𝒰₀, T₁, thermo, equilibrium)

    # Next, we generate a second guess that scaled by the supersaturation implied by T₁
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat
    qˡ₁ = q₁.liquid
    qⁱ₁ = q₁.ice
    cᵖᵐ = mixture_heat_capacity(q₁, thermo)
    ΔT = (ℒˡᵣ * qˡ₁ + ℒⁱᵣ * qⁱ₁) / cᵖᵐ
    ϵT = convert(FT, 0.01) # minimum increment for second guess
    T₂ = T₁ + max(ϵT, ΔT / 2) # reduce the increment, recognizing it is an overshoot
    # T₂ = T₁ + ΔT / 2 # reduce the increment, recognizing it is an overshoot
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
