#####
##### Translations of CloudMicrophysics functions that depend on Thermodynamics.jl
#####
#
# These functions mirror physics from CloudMicrophysics.jl but use Breeze's
# internal thermodynamics infrastructure instead of Thermodynamics.jl.
# This avoids a dependency on the Thermodynamics.jl package.
#
# CRITERIA: A function appears here ONLY if it depends on Thermodynamics.jl
# in CloudMicrophysics. Helper functions that don't depend on thermodynamics
# are imported directly from CloudMicrophysics when possible.
#
# Reference: CloudMicrophysics.jl Common.jl and Microphysics1M.jl

# Import CloudMicrophysics internals that we need
# (these don't depend on Thermodynamics.jl)
using CloudMicrophysics.Utilities: ϵ_numerics
using CloudMicrophysics.Microphysics1M: lambda_inverse, get_n0, get_v0, SF

# gamma function from SpecialFunctions (via CloudMicrophysics)
const Γ = SF.gamma

#####
##### Diffusional growth factor (TRANSLATION: uses Thermodynamics.jl in CloudMicrophysics)
#####

"""
    diffusional_growth_factor(aps::AirProperties, T, constants)

Compute the thermodynamic factor ``G`` that controls the rate of diffusional
growth of cloud droplets and rain drops.

The ``G`` factor combines the effects of thermal conductivity and vapor diffusivity
on phase change. It appears in the Mason equation for droplet growth:

```math
\\frac{dm}{dt} = 4π r G 𝒮
```

where ``𝒮`` is supersaturation and ``r`` is droplet radius.

This is a translation of `CloudMicrophysics.Common.G_func_liquid`
using Breeze's thermodynamics instead of Thermodynamics.jl.

See Eq. (13.28) by [Pruppacher & Klett (2010)](@cite pruppacher2010microphysics).

# References
* Pruppacher, H. R., Klett, J. D. (2010). Microphysics of clouds and precipitation. Springer Netherlands. 2nd Edition
"""
@inline function diffusional_growth_factor(aps::AirProperties{FT}, T, constants) where {FT}
    (; K_therm, D_vapor) = aps
    Rᵛ = vapor_gas_constant(constants)
    ℒˡ = liquid_latent_heat(T, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())

    return 1 / (ℒˡ / K_therm / T * (ℒˡ / Rᵛ / T - 1) + Rᵛ * T / D_vapor / pᵛ⁺)
end

@inline function diffusional_growth_factor_ice(aps::AirProperties{FT}, T, constants) where {FT}
    (; K_therm, D_vapor) = aps
    Rᵛ = vapor_gas_constant(constants)
    ℒⁱ = ice_latent_heat(T, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarIceSurface())

    Dᵛ = D_vapor

    # TODO: notation for the thermal diffusivity K_therm?
    return 1 / (ℒⁱ / (K_therm * T) * (ℒⁱ / (Rᵛ * T) - 1) + Rᵛ * T / (Dᵛ * pᵛ⁺))
end

#####
##### Rain evaporation (TRANSLATION: uses the above thermodynamics-dependent functions)
#####

"""
    rain_evaporation(rain_params, vel, aps, q, qʳ, ρ, T, constants)

Compute the rain evaporation rate (dqʳ/dt, negative for evaporation).

This is a translation of `CloudMicrophysics.Microphysics1M.evaporation_sublimation`
that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `rain_params`: Rain microphysics parameters (pdf, mass, vent)
- `vel`: Terminal velocity parameters
- `aps`: Air properties (kinematic viscosity, vapor diffusivity, thermal conductivity)
- `q`: `MoistureMassFractions` containing vapor, liquid, and ice mass fractions
- `qʳ`: Rain specific humidity
- `ρ`: Air density
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Rate of change of rain specific humidity (negative = evaporation)
"""
@inline function rain_evaporation(
    (; pdf, mass, vent)::Rain{FT},
    vel::Blk1MVelTypeRain{FT},
    aps::AirProperties{FT},
    q::MoistureMassFractions{FT},
    qʳ::FT,
    ρ::FT,
    T::FT,
    constants,
) where {FT}
    (; ν_air, D_vapor) = aps
    (; χv, ve, Δv) = vel
    (; r0) = mass
    aᵥ = vent.a
    bᵥ = vent.b

    # Compute supersaturation (𝒮 < 0 means subsaturated → evaporation)
    𝒮 = supersaturation(T, ρ, q, constants, PlanarLiquidSurface())

    G = diffusional_growth_factor(aps, T, constants)
    n₀ = get_n0(pdf, qʳ, ρ)
    v₀ = get_v0(vel, ρ)
    λ⁻¹ = lambda_inverse(pdf, mass, qʳ, ρ)

    # Ventilated evaporation rate from Mason equation
    # Base evaporation rate (unventilated)
    base_rate = 4π * n₀ / ρ * 𝒮 * G * λ⁻¹^2

    # Ventilation correction terms
    Sc = ν_air / D_vapor
    Re = 2v₀ * χv / ν_air * λ⁻¹
    size_factor = (r0 / λ⁻¹)^((ve + Δv) / 2)
    gamma_factor = Γ((ve + Δv + 5) / 2)

    ventilation = aᵥ + bᵥ * cbrt(Sc) * sqrt(Re) / size_factor * gamma_factor

    evap_rate = base_rate * ventilation

    # Only evaporate if subsaturated (𝒮 < 0) and rain exists
    evaporating = (qʳ > ϵ_numerics(FT)) & (𝒮 < 0)

    # Only evaporation (negative tendency) is considered for rain
    return ifelse(evaporating, min(zero(FT), evap_rate), zero(FT))
end

#####
##### Snow sublimation/deposition (TRANSLATION: uses Breeze thermodynamics over ice surface)
#####

"""
    snow_sublimation_deposition(snow_params, vel, aps, q, qˢ, ρ, T, constants)

Compute the snow sublimation/deposition rate (dqˢ/dt).

Positive values mean deposition (vapor → snow), negative means sublimation (snow → vapor).
Unlike rain evaporation, both signs are physical for snow.

This is a translation of `CloudMicrophysics.Microphysics1M.evaporation_sublimation`
for snow that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `snow_params`: Snow microphysics parameters (pdf, mass, vent)
- `vel`: Snow terminal velocity parameters
- `aps`: Air properties (kinematic viscosity, vapor diffusivity, thermal conductivity)
- `q`: `MoistureMassFractions` containing vapor, liquid, and ice mass fractions
- `qˢ`: Snow specific humidity
- `ρ`: Air density
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Rate of change of snow specific humidity (positive = deposition, negative = sublimation)
"""
@inline function snow_sublimation_deposition(
    (; pdf, mass, vent)::Snow{FT},
    vel::Blk1MVelTypeSnow{FT},
    aps::AirProperties{FT},
    q::MoistureMassFractions{FT},
    qˢ::FT,
    ρ::FT,
    T::FT,
    constants,
) where {FT}
    (; ν_air, D_vapor) = aps
    (; χv, ve, Δv) = vel
    (; r0) = mass
    aᵥ = vent.a
    bᵥ = vent.b

    # Supersaturation over ice (𝒮 > 0 → deposition, 𝒮 < 0 → sublimation)
    𝒮 = supersaturation(T, ρ, q, constants, PlanarIceSurface())

    G = diffusional_growth_factor_ice(aps, T, constants)
    n₀ = get_n0(pdf, qˢ, ρ)
    v₀ = get_v0(vel, ρ)
    λ⁻¹ = lambda_inverse(pdf, mass, qˢ, ρ)

    # Ventilated sublimation/deposition rate from Mason equation
    base_rate = 4π * n₀ / ρ * 𝒮 * G * λ⁻¹^2

    # Ventilation correction terms
    Sc = ν_air / D_vapor
    Re = 2v₀ * χv / ν_air * λ⁻¹
    size_factor = (r0 / λ⁻¹)^((ve + Δv) / 2)
    gamma_factor = Γ((ve + Δv + 5) / 2)

    ventilation = aᵥ + bᵥ * cbrt(Sc) * sqrt(Re) / size_factor * gamma_factor

    rate = base_rate * ventilation

    # Both sublimation (𝒮 < 0) and deposition (𝒮 > 0) are physical for snow
    has_snow = qˢ > ϵ_numerics(FT)
    return ifelse(has_snow, rate, zero(FT))
end

#####
##### Snow melting (TRANSLATION: uses Breeze thermodynamics for latent heat of fusion)
#####

"""
    snow_melting(snow_params, vel, aps, qˢ, ρ, T, constants)

Compute the snow melting rate (dqˢ/dt due to melting, always non-negative).

Sensible-heat-driven melting: heat from warm air (T > T_freeze) melts snow to rain.
The rate is proportional to (T - T_freeze) and includes ventilation corrections.

This is a translation of `CloudMicrophysics.Microphysics1M.snow_melt`
that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `snow_params`: Snow microphysics parameters (T_freeze, pdf, mass, vent)
- `vel`: Snow terminal velocity parameters
- `aps`: Air properties (kinematic viscosity, vapor diffusivity, thermal conductivity)
- `qˢ`: Snow specific humidity
- `ρ`: Air density
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Rate of snow mass lost to melting [kg/kg/s] (always non-negative)
"""
@inline function snow_melting(
    (; T_freeze, pdf, mass, vent)::Snow{FT},
    vel::Blk1MVelTypeSnow{FT},
    aps::AirProperties{FT},
    qˢ::FT,
    ρ::FT,
    T::FT,
    constants,
) where {FT}
    (; ν_air, D_vapor, K_therm) = aps
    (; χv, ve, Δv) = vel
    (; r0) = mass
    aᵥ = vent.a
    bᵥ = vent.b

    # Latent heat of fusion: ℒⁱ(vapor→ice) - ℒˡ(vapor→liquid) = ℒf(liquid→ice)
    ℒf = ice_latent_heat(T, constants) - liquid_latent_heat(T, constants)

    n₀ = get_n0(pdf, qˢ, ρ)
    v₀ = get_v0(vel, ρ)
    λ⁻¹ = lambda_inverse(pdf, mass, qˢ, ρ)

    # Sensible-heat-driven melting rate
    base_rate = 4π * n₀ / ρ * K_therm / ℒf * (T - T_freeze) * λ⁻¹^2

    # Ventilation correction terms
    Sc = ν_air / D_vapor
    Re = 2v₀ * χv / ν_air * λ⁻¹
    size_factor = (r0 / λ⁻¹)^((ve + Δv) / 2)
    gamma_factor = Γ((ve + Δv + 5) / 2)

    ventilation = aᵥ + bᵥ * cbrt(Sc) * sqrt(Re) / size_factor * gamma_factor

    melt_rate = base_rate * ventilation

    # Only melt when snow exists and temperature is above freezing
    melting = (qˢ > ϵ_numerics(FT)) & (T > T_freeze)
    return ifelse(melting, melt_rate, zero(FT))
end

#####
##### Warm accretion melt factor (TRANSLATION: uses Breeze thermodynamics for Lf and cv_l)
#####

"""
    warm_accretion_melt_factor(snow_params, T, constants)

Compute the thermal melt factor for warm accretion processes.

When cloud liquid or rain collides with snow above freezing, the sensible heat
carried by the warm hydrometeor melts additional snow. The factor ``α`` gives
the mass ratio of melted snow to accreted warm hydrometeor mass:

``α = cˡ (T - T_{freeze}) / ℒf``

This is a translation of `CloudMicrophysics.BulkMicrophysicsTendencies.warm_accretion_melt_factor`
that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `snow_params`: Snow parameters (contains T_freeze)
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Thermal melt factor α (zero when T <= T_freeze)
"""
@inline function warm_accretion_melt_factor(
    (; T_freeze)::Snow{FT},
    T::FT,
    constants,
) where {FT}
    cˡ = constants.liquid.heat_capacity
    ℒf = ice_latent_heat(T, constants) - liquid_latent_heat(T, constants)
    ΔT = T - T_freeze
    return ifelse(T <= T_freeze, zero(FT), cˡ / ℒf * ΔT)
end

#####
##### Two-moment rain evaporation (TRANSLATION: SB2006 evaporation using Breeze thermodynamics)
#####

# Import SB2006 PDF helper functions from CloudMicrophysics.Microphysics2M
using CloudMicrophysics.Microphysics2M: pdf_rain_parameters, Γ_incl

"""
    rain_evaporation_2m(sb, aps, q, qʳ, ρ, Nʳ, T, constants)

Compute the two-moment rain evaporation rate returning both number and mass tendencies.

This is a translation of `CloudMicrophysics.Microphysics2M.rain_evaporation`
that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `sb`: SB2006 parameters containing pdf_r and evap
- `aps`: Air properties (kinematic viscosity, vapor diffusivity, thermal conductivity)
- `q`: `MoistureMassFractions` containing vapor, liquid, and ice mass fractions
- `qʳ`: Rain specific humidity [kg/kg]
- `ρ`: Air density [kg/m³]
- `Nʳ`: Rain number concentration [1/m³]
- `T`: Temperature [K]
- `constants`: Breeze ThermodynamicConstants

# Returns
Named tuple `(; evap_rate_0, evap_rate_1)` where:
- `evap_rate_0`: Rate of change of number concentration [1/(m³·s)], negative for evaporation
- `evap_rate_1`: Rate of change of mass mixing ratio [kg/kg/s], negative for evaporation
"""
@inline function rain_evaporation_2m(
    (; pdf_r, evap)::SB2006{FT},
    aps::AirProperties{FT},
    q::MoistureMassFractions{FT},
    qʳ::FT,
    ρ::FT,
    Nʳ::FT,
    T::FT,
    constants,
) where {FT}

    (; ν_air, D_vapor) = aps
    (; av, bv, α, β, ρ0) = evap
    x_star = pdf_r.xr_min
    ρᴸ = pdf_r.ρw

    # Compute supersaturation over liquid (negative means subsaturated)
    𝒮 = supersaturation(T, ρ, q, constants, PlanarLiquidSurface())

    # Condition: evaporate only when rain exists and air is subsaturated
    evaporating = (Nʳ > ϵ_numerics(FT)) & (𝒮 < zero(FT))

    # Use safe positive values to avoid NaN/Inf in intermediate computations
    Nʳ_safe = max(Nʳ, ϵ_numerics(FT))
    qʳ_safe = max(qʳ, eps(FT))

    # Diffusional growth factor (G function)
    G = diffusional_growth_factor(aps, T, constants)

    # Mean rain drop mass and diameter
    (; xr_mean) = pdf_rain_parameters(pdf_r, qʳ_safe, ρ, Nʳ_safe)
    xr_mean_safe = max(xr_mean, eps(FT))
    Dʳ = cbrt(6 * xr_mean_safe / (π * ρᴸ))

    # Ventilation factors for number and mass tendencies
    t_star = cbrt(6 * x_star / xr_mean_safe)
    a_vent_0 = av * Γ_incl(FT(-1), t_star) / FT(6)^(FT(-2) / FT(3))
    b_vent_0 = bv * Γ_incl(FT(-0.5) + FT(1.5) * β, t_star) / FT(6)^((β - one(FT)) / FT(2))

    a_vent_1 = av * Γ(FT(2)) / cbrt(FT(6))
    b_vent_1 = bv * Γ(FT(2.5) + FT(1.5) * β) / FT(6)^((β + one(FT)) / FT(2))

    # Reynolds number
    Re = α * xr_mean_safe^β * sqrt(ρ0 / ρ) * Dʳ / ν_air
    Fv0 = a_vent_0 + b_vent_0 * cbrt(ν_air / D_vapor) * sqrt(Re)
    Fv1 = a_vent_1 + b_vent_1 * cbrt(ν_air / D_vapor) * sqrt(Re)

    # Evaporation rates (negative for evaporation)
    evap_rate_0 = min(zero(FT), 2π * G * 𝒮 * Nʳ_safe * Dʳ * Fv0 / xr_mean_safe)
    evap_rate_1 = min(zero(FT), 2π * G * 𝒮 * Nʳ_safe * Dʳ * Fv1 / ρ)

    # Handle edge cases where xr_mean approaches zero
    evap_rate_0 = ifelse(xr_mean / x_star < eps(FT), zero(FT), evap_rate_0)
    evap_rate_1 = ifelse(qʳ < eps(FT), zero(FT), evap_rate_1)

    # Zero out when no evaporation should occur
    evap_rate_0 = ifelse(evaporating, evap_rate_0, zero(FT))
    evap_rate_1 = ifelse(evaporating, evap_rate_1, zero(FT))

    return (; evap_rate_0, evap_rate_1)
end

#####
##### Two-moment microphysical state (defined here for use in translations below)
#####

using Breeze.AtmosphereModels: AbstractMicrophysicalState
using CloudMicrophysics.AerosolModel: Mode_B, Mode_κ

"""
    WarmPhaseTwoMomentState{FT, V} <: AbstractMicrophysicalState{FT}

Microphysical state for warm-phase two-moment bulk microphysics.

Contains the local mixing ratios and number concentrations needed to compute
tendencies for cloud liquid and rain following the Seifert-Beheng 2006 scheme.

# Fields
- `qᶜˡ`: Cloud liquid mixing ratio (kg/kg)
- `nᶜˡ`: Cloud liquid number per unit mass (1/kg)
- `qʳ`: Rain mixing ratio (kg/kg)
- `nʳ`: Rain number per unit mass (1/kg)
- `nᵃ`: Aerosol number per unit mass (1/kg)
- `velocities`: NamedTuple of velocity components `(; u, v, w)` [m/s].
  The vertical velocity `w` is used for aerosol activation.
"""
struct WarmPhaseTwoMomentState{FT, V} <: AbstractMicrophysicalState{FT}
    qᶜˡ :: FT         # cloud liquid mixing ratio
    nᶜˡ :: FT         # cloud liquid number per unit mass
    qʳ  :: FT         # rain mixing ratio
    nʳ  :: FT         # rain number per unit mass
    nᵃ  :: FT         # aerosol number per unit mass
    velocities :: V   # velocity components (; u, v, w)
end

"""
    AerosolActivation{AP, AD, FT}

Aerosol activation parameters for two-moment microphysics.

Aerosol activation is the physical process that creates cloud droplets from aerosol
particles when air becomes supersaturated. This struct bundles the parameters needed
to compute the activation source term for cloud droplet number concentration.

# Fields
- `activation_parameters`: [`AerosolActivationParameters`] from CloudMicrophysics.jl
- `aerosol_distribution`: Aerosol size distribution (modes with number, size, hygroscopicity)
- `nucleation_timescale`: Nucleation timescale [s] for converting activation deficit to rate (default: 1s)

# References
* Abdul-Razzak, H. and Ghan, S.J. (2000). A parameterization of aerosol activation:
  2. Multiple aerosol types. J. Geophys. Res., 105(D5), 6837-6844.
"""
struct AerosolActivation{AP, AD, FT}
    activation_parameters :: AP
    aerosol_distribution :: AD
    nucleation_timescale :: FT
end

Base.summary(::AerosolActivation) = "AerosolActivation"

#####
##### Aerosol activation (TRANSLATION: uses AerosolActivation.jl in CloudMicrophysics with Breeze thermodynamics)
#####
#
# Aerosol activation computes the number of cloud droplets formed when aerosol
# particles are exposed to supersaturated conditions. This is the source term
# for cloud droplet number in two-moment microphysics.
#
# Reference: Abdul-Razzak, H. and Ghan, S.J. (2000). A parameterization of aerosol
#            activation: 2. Multiple aerosol types. J. Geophys. Res., 105(D5), 6837-6844.
#####

"""
    max_supersaturation_breeze(aerosol_activation, aps, ρ, ℳ, 𝒰, constants)

Compute the maximum supersaturation using the Abdul-Razzak and Ghan (2000) parameterization.

This is a translation of `CloudMicrophysics.AerosolActivation.max_supersaturation` that uses
Breeze's thermodynamics instead of Thermodynamics.jl.

# Arguments
- `aerosol_activation`: AerosolActivation containing activation parameters and aerosol distribution
- `aps`: AirProperties (thermal conductivity, vapor diffusivity)
- `ρ`: Air density [kg/m³]
- `ℳ`: Microphysical state containing updraft velocity and number concentrations
- `𝒰`: Thermodynamic state
- `constants`: Breeze ThermodynamicConstants

# Returns
Maximum supersaturation (dimensionless, e.g., 0.01 = 1% supersaturation)
"""
@inline function max_supersaturation_breeze(
    aerosol_activation::AerosolActivation,
    aps::AirProperties{FT},
    ρ::FT,
    ℳ::WarmPhaseTwoMomentState{FT},
    𝒰,
    constants,
) where {FT}

    # Extract from thermodynamic state
    T = temperature(𝒰, constants)
    p = 𝒰.reference_pressure
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor
    qˡ = q.liquid
    qⁱ = q.ice

    # Extract from microphysical state
    w = ℳ.velocities.w  # vertical velocity for aerosol activation
    Nˡ = ℳ.nᶜˡ * ρ  # convert from per-mass to per-volume
    Nⁱ = zero(FT)   # warm phase: no ice

    ap = aerosol_activation.activation_parameters
    ad = aerosol_activation.aerosol_distribution

    # Thermodynamic properties from Breeze
    Rᵛ = vapor_gas_constant(constants)
    ℒˡ = liquid_latent_heat(T, constants)
    ℒⁱ = ice_latent_heat(T, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    pᵛ⁺ⁱ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
    g = constants.gravitational_acceleration
    ρᴸ = ap.ρ_w  # intrinsic density of liquid water
    ρᴵ = ap.ρ_i  # intrinsic density of ice

    # Mixture properties
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)

    # Vapor pressure
    pᵛ = qᵛ * ρ * Rᵛ * T

    # Diffusional growth factor G (Eq. 13.28 in Pruppacher & Klett)
    G = diffusional_growth_factor(aps, T, constants) / ρᴸ

    # ARG parameters (Eq. 11, 12 in Abdul-Razzak et al. 1998)
    # α = rate of change of saturation ratio due to adiabatic cooling
    α = pᵛ / pᵛ⁺ * (ℒˡ * g / (Rᵛ * cᵖᵐ * T^2) - g / (Rᵐ * T))
    # γ = thermodynamic factor for condensation
    γ = Rᵛ * T / pᵛ⁺ + pᵛ / pᵛ⁺ * Rᵐ * ℒˡ^2 / (Rᵛ * cᵖᵐ * T * p)

    # Curvature coefficient (Kelvin effect)
    # Formula: A = 2σ / (ρᴸ * R_v * T)
    A = 2 * ap.σ / (ρᴸ * Rᵛ * T)

    # Maximum supersaturation from ARG 2000 (only valid for w > 0)
    Sᵐᵃˣ₀ = compute_smax(aerosol_activation, A, α, γ, G, w, ρᴸ)

    # Correction for existing liquid and ice (phase relaxation)
    # See Eq. A13 in Korolev and Mazin (2003) or CloudMicrophysics implementation

    # Liquid relaxation
    rˡ = ifelse(Nˡ > eps(FT), cbrt(ρ * qˡ / (Nˡ * ρᴸ * (4π / 3))), zero(FT))
    Kˡ = 4π * ρᴸ * Nˡ * rˡ * G * γ

    # Ice relaxation
    γⁱ = Rᵛ * T / pᵛ⁺ + pᵛ / pᵛ⁺ * Rᵐ * ℒˡ * ℒⁱ / (Rᵛ * cᵖᵐ * T * p)
    rⁱ = ifelse(Nⁱ > eps(FT), cbrt(ρ * qⁱ / (Nⁱ * ρᴵ * (4π / 3))), zero(FT))
    Gⁱ = diffusional_growth_factor_ice(aps, T, constants)
    Kⁱ = 4π * Nⁱ * rⁱ * Gⁱ * γⁱ

    ξ = pᵛ⁺ / pᵛ⁺ⁱ

    # Phase-relaxation corrected Sᵐᵃˣ (Eq. A13 in Korolev and Mazin 2003)
    # Use safe denominator conditioned on w > 0 to avoid NaN
    denominator = α * w + (Kˡ + Kⁱ * ξ) * Sᵐᵃˣ₀
    safe_denominator = ifelse(w > zero(FT), denominator, one(FT))
    Sᵐᵃˣ_computed = Sᵐᵃˣ₀ * (α * w - Kⁱ * (ξ - 1)) / safe_denominator

    # Activation only occurs with positive updraft velocity
    Sᵐᵃˣ = ifelse(w > zero(FT), Sᵐᵃˣ_computed, zero(FT))

    return max(zero(FT), Sᵐᵃˣ)
end

# Helper function to compute mean hygroscopicity
@inline function mean_hygroscopicity(ap, mode::Mode_κ{T, FT}) where {T <: Tuple, FT}
    κ̄ = zero(FT)
    @inbounds for α in 1:fieldcount(T)
        κ̄ += mode.vol_mix_ratio[α] * mode.kappa[α]
    end
    return κ̄
end

@inline mean_hygroscopicity(ap, mode::Mode_κ{T, FT}) where {T <: Real, FT} = mode.vol_mix_ratio * mode.kappa

@inline function mean_hygroscopicity(ap, mode::Mode_B{T, FT}) where {T <: Tuple, FT}
    numerator = zero(FT)
    @inbounds for α in 1:fieldcount(T)
        numerator += mode.mass_mix_ratio[α] * mode.dissoc[α] * mode.osmotic_coeff[α] *
                     mode.soluble_mass_frac[α] / mode.molar_mass[α]
    end

    denominator = zero(FT)
    @inbounds for α in 1:fieldcount(T)
        denominator += mode.mass_mix_ratio[α] / mode.aerosol_density[α]
    end

    return numerator / denominator * ap.M_w / ap.ρ_w
end

@inline function mean_hygroscopicity(ap, mode::Mode_B{T, FT}) where {T <: Real, FT}
    numerator = mode.mass_mix_ratio * mode.dissoc * mode.osmotic_coeff * mode.soluble_mass_frac / mode.molar_mass
    denominator = mode.mass_mix_ratio / mode.aerosol_density
    return numerator / denominator * ap.M_w / ap.ρ_w
end

# Helper function to compute Sᵐᵃˣ
# Dispatches on aerosol_activation type to enable different activation schemes
@inline function compute_smax(aerosol_activation, A::FT, α::FT, γ::FT, G::FT, w::FT, ρᴸ::FT) where FT
    ap = aerosol_activation.activation_parameters
    ad = aerosol_activation.aerosol_distribution

    # Use safe positive w to avoid NaN in computation; result is 0 when w <= 0
    # ARG 2000 parameterization is only valid for positive updraft velocities
    w⁺ = max(eps(FT), w)

    # All intermediate quantities should be non-negative for physical states.
    # Guard with max(0, ...) to handle extreme/unphysical transient states.
    αwG = max(0, α * w⁺ / G)
    ζ = max(0, 2A / 3) * sqrt(αwG)

    # Compute critical supersaturation and contribution from each mode
    Σ_inv_Sᵐᵃˣ² = zero(FT)
    for mode in ad.modes

        # Mean hygroscopicity for mode (volume-weighted κ)
        κ̄ = max(eps(FT), mean_hygroscopicity(ap, mode))

        # Critical supersaturation (Eq. 9 in ARG 2000)
        Sᶜʳⁱᵗ = max(eps(FT), 2 / sqrt(κ̄) * sqrt(max(0, A / (3 * mode.r_dry)))^3)

        # Fitting parameters (fᵥ and gᵥ are ventilation-related)
        fᵥ = ap.f1 * exp(ap.f2 * log(mode.stdev)^2)
        gᵥ = ap.g1 + ap.g2 * log(mode.stdev)

        # η parameter
        η = max(eps(FT), sqrt(αwG)^3 / (2π * ρᴸ * γ * mode.N))

        # Contribution to 1/Sᵐᵃˣ² (Eq. 6 in ARG 2000)
        # All bases of fractional exponents are guaranteed positive by guards above
        Σ_inv_Sᵐᵃˣ² += 1 / Sᶜʳⁱᵗ^2 * (fᵥ * (ζ / η)^ap.p1 + gᵥ * (Sᶜʳⁱᵗ^2 / (η + 3 * ζ))^ap.p2)
    end

    Sᵐᵃˣ_computed = 1 / sqrt(max(eps(FT), Σ_inv_Sᵐᵃˣ²))

    # Return 0 for no updraft (w <= 0), otherwise return computed value
    return ifelse(w > zero(FT), Sᵐᵃˣ_computed, zero(FT))
end
