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
using CloudMicrophysics.Common: ϵ_numerics
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

    evap_rate_0 = zero(FT)
    evap_rate_1 = zero(FT)

    # Compute supersaturation over liquid (negative means subsaturated)
    𝒮 = supersaturation(T, ρ, q, constants, PlanarLiquidSurface())

    # Only evaporate if there's rain and air is subsaturated
    if (Nʳ > ϵ_numerics(FT)) && (𝒮 < zero(FT))
        (; ν_air, D_vapor) = aps
        (; av, bv, α, β, ρ0) = evap
        x_star = pdf_r.xr_min
        ρw = pdf_r.ρw

        # Diffusional growth factor (G function)
        G = diffusional_growth_factor(aps, T, constants)

        # Mean rain drop mass and diameter
        (; xr_mean) = pdf_rain_parameters(pdf_r, qʳ, ρ, Nʳ)
        Dr = cbrt(6 * xr_mean / (π * ρw))

        # Ventilation factors for number and mass tendencies
        t_star = cbrt(6 * x_star / xr_mean)
        a_vent_0 = av * Γ_incl(FT(-1), t_star) / FT(6)^(-2 // 3)
        b_vent_0 = bv * Γ_incl(-1 // 2 + 3 // 2 * β, t_star) / FT(6)^(β / 2 - 1 // 2)

        a_vent_1 = av * Γ(FT(2)) / cbrt(FT(6))
        b_vent_1 = bv * Γ(5 // 2 + 3 // 2 * β) / 6^(β / 2 + 1 // 2)

        # Reynolds number
        N_Re = α * xr_mean^β * sqrt(ρ0 / ρ) * Dr / ν_air
        Fv0 = a_vent_0 + b_vent_0 * cbrt(ν_air / D_vapor) * sqrt(N_Re)
        Fv1 = a_vent_1 + b_vent_1 * cbrt(ν_air / D_vapor) * sqrt(N_Re)

        # Evaporation rates (negative for evaporation)
        evap_rate_0 = min(zero(FT), FT(2) * FT(π) * G * 𝒮 * Nʳ * Dr * Fv0 / xr_mean)
        evap_rate_1 = min(zero(FT), FT(2) * FT(π) * G * 𝒮 * Nʳ * Dr * Fv1 / ρ)

        # Handle edge cases where xr_mean approaches zero
        evap_rate_0 = ifelse(xr_mean / x_star < eps(FT), zero(FT), evap_rate_0)
        evap_rate_1 = ifelse(qʳ < eps(FT), zero(FT), evap_rate_1)
    end

    return (; evap_rate_0, evap_rate_1)
end
