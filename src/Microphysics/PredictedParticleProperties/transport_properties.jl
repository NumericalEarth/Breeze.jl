#####
##### Air transport properties
#####
##### T,P-dependent thermal conductivity Kᵃ, vapor diffusivity Dᵛ, and
##### kinematic viscosity ν. Writing the dynamic viscosity as η (μ is the PSD
##### shape parameter everywhere else in this module), the fits are
#####   Dᵛ = 8.794e-5 * T^1.81 / P        [m²/s]
#####   η  = 1.496e-6 * T^1.5 / (T + 120) [Pa s]   (Sutherland's law)
#####   Kᵃ = 1414 * η                     [W/m/K]
#####   ν  = η / ρ = η * Rᵈ * T / P       [m²/s]
#####

export air_transport_properties

# Coefficients of the transport fits, named so that the provenance of each number
# is visible where it is used.
const VAPOR_DIFFUSIVITY_COEFFICIENT = 8.794e-5           # [m² s⁻¹ Pa K^-1.81]
const VAPOR_DIFFUSIVITY_TEMPERATURE_EXPONENT = 1.81      # [-]
const SUTHERLAND_COEFFICIENT = 1.496e-6                  # [Pa s K^-1/2]
const SUTHERLAND_TEMPERATURE = 120                       # [K]
const AIR_HEAT_CAPACITY_PRANDTL_RATIO = 1414             # Kᵃ / η = cᵖ / Pr [J kg⁻¹ K⁻¹]
const MINIMUM_TRANSPORT_TEMPERATURE = 1                  # [K], keeps the power laws finite

@inline function sutherland_dynamic_viscosity(T_safe)
    FT = typeof(T_safe)
    return FT(SUTHERLAND_COEFFICIENT) * sqrt(T_safe)^3 /
           (T_safe + FT(SUTHERLAND_TEMPERATURE))
end

@inline function air_kinematic_viscosity(T, P, constants)
    FT = typeof(T)
    T_safe = max(T, FT(MINIMUM_TRANSPORT_TEMPERATURE))
    η = sutherland_dynamic_viscosity(T_safe)
    Rᵈ = FT(dry_air_gas_constant(constants))
    return η * Rᵈ * T_safe / P
end

"""
$(TYPEDSIGNATURES)

Compute T,P-dependent air transport properties following
[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021).

Returns a named tuple `(; Dᵛ, Kᵃ, ν)`:
- `Dᵛ`: vapor diffusivity [m²/s], from `8.794e-5 × T^1.81 / P`
- `Kᵃ`: thermal conductivity of air [W/m/K], from `1414 × η`
- `ν`: kinematic viscosity [m²/s], from `η × Rᵈ × T / P`

where `η = 1.496e-6 × T^1.5 / (T + 120)` is the dynamic viscosity (Pa s) from
Sutherland's law. It is written `η` rather than `μ`, which denotes the particle
size distribution shape parameter throughout this module.

# Arguments
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `constants`: Thermodynamic constants supplying the dry-air gas constant used
  in the kinematic-viscosity calculation.

# Reference values

At T = 273.15 K, P = 101325 Pa:
- Dᵛ ≈ 2.23e-5 m²/s
- Kᵃ ≈ 0.024 W/m/K
- ν ≈ 1.33e-5 m²/s

# Example

```jldoctest
using Breeze
using Breeze.Microphysics.PredictedParticleProperties: air_transport_properties
constants = ThermodynamicConstants()
properties = air_transport_properties(273.15, 101325.0, constants)
map(x -> round(x, sigdigits=3), properties)

# output
(Dᵛ = 2.23e-5, Kᵃ = 0.0243, ν = 1.33e-5)
```
"""
@inline function air_transport_properties(T, P, constants)
    FT = typeof(T)
    T_safe = max(T, FT(MINIMUM_TRANSPORT_TEMPERATURE))

    Dᵛ = FT(VAPOR_DIFFUSIVITY_COEFFICIENT) *
         T_safe^FT(VAPOR_DIFFUSIVITY_TEMPERATURE_EXPONENT) / P

    # Sutherland's law. The dynamic viscosity is η, not μ, which denotes the
    # PSD shape parameter everywhere else in this module.
    η = sutherland_dynamic_viscosity(T_safe)

    Kᵃ = FT(AIR_HEAT_CAPACITY_PRANDTL_RATIO) * η
    Rᵈ = FT(dry_air_gas_constant(constants))
    ν = η * Rᵈ * T_safe / P

    return (; Dᵛ, Kᵃ, ν)
end
