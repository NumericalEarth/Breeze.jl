#####
##### Air transport properties
#####
##### T,P-dependent thermal conductivity K_a, vapor diffusivity D_v, and
##### kinematic viscosity nu. Following Fortran P3 v5.5.0 conventions:
#####   dv  = 8.794e-5 * T^1.81 / P        [m²/s]
#####   mu  = 1.496e-6 * T^1.5 / (T + 120) [Pa·s]   (Sutherland's law)
#####   kap = 1414 * mu                     [W/m/K]
#####   nu  = mu / rho = mu * R_d * T / P   [m²/s]
#####

export air_transport_properties

"""
$(TYPEDSIGNATURES)

Compute T,P-dependent air transport properties following [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021)
and the Fortran P3 v5.5.0 reference implementation.

Returns a named tuple `(; D_v, K_a, nu)`:
- `D_v`: vapor diffusivity [m²/s], from `8.794e-5 × T^1.81 / P`
- `K_a`: thermal conductivity of air [W/m/K], from Sutherland's law `1414 × μ`
- `nu`: kinematic viscosity [m²/s], from `μ × R_d × T / P`

where `μ = 1.496e-6 × T^1.5 / (T + 120)` is the dynamic viscosity (Pa·s).

# Arguments
- `T`: Temperature [K]
- `P`: Pressure [Pa]

# Reference values

At T = 273.15 K, P = 101325 Pa:
- D_v ≈ 2.23e-5 m²/s
- K_a ≈ 0.024 W/m/K
- nu ≈ 1.33e-5 m²/s

# Example

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: air_transport_properties
props = air_transport_properties(273.15, 101325.0)
typeof(props.D_v)

# output
Float64
```
"""
@inline function air_transport_properties(T, P)
    FT = typeof(T)
    D_v = FT(8.794e-5) * T^FT(1.81) / P
    mu_air = FT(1.496e-6) * T^FT(1.5) / (T + FT(120))
    K_a = FT(1414) * mu_air
    R_d = FT(287.0)
    nu = mu_air * R_d * T / P
    return (; D_v, K_a, nu)
end
