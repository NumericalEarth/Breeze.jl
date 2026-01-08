#####
##### Ice Fall Speed
#####
##### Terminal velocity integrals over the ice particle size distribution.
##### P3 computes number-, mass-, and reflectivity-weighted fall speeds.
#####

"""
    IceFallSpeed{FT, N, M, Z}

Ice particle fall speed properties and integrals.

The terminal velocity of ice particles follows a power law:

```math
V(D) = a_V \\left(\\frac{\\rho_0}{\\rho}\\right)^{0.5} D^{b_V}
```

where `a_V` is the `fall_speed_coefficient`, `b_V` is the `fall_speed_exponent`,
`ρ_0` is the `reference_air_density`, and `ρ` is the local air density.

# Fields

## Parameters
- `reference_air_density`: Reference air density ρ₀ for fall speed correction [kg/m³]
- `fall_speed_coefficient`: Coefficient a_V in V(D) = a_V D^{b_V} [m^{1-b_V}/s]
- `fall_speed_exponent`: Exponent b_V in V(D) = a_V D^{b_V} [-]

## Integrals (or `TabulatedIntegral` after tabulation)
- `number_weighted`: Number-weighted fall speed V_n
- `mass_weighted`: Mass-weighted fall speed V_m
- `reflectivity_weighted`: Reflectivity-weighted fall speed V_z (3-moment)

# References

Morrison and Milbrandt (2015), Milbrandt and Morrison (2016)
"""
struct IceFallSpeed{FT, N, M, Z}
    # Parameters
    reference_air_density :: FT
    fall_speed_coefficient :: FT
    fall_speed_exponent :: FT
    # Integrals
    number_weighted :: N
    mass_weighted :: M
    reflectivity_weighted :: Z
end

"""
    IceFallSpeed(FT=Float64)

Construct `IceFallSpeed` with default parameters and quadrature-based integrals.

Default parameters from Morrison and Milbrandt (2015).
"""
function IceFallSpeed(FT::Type{<:AbstractFloat} = Float64)
    return IceFallSpeed(
        FT(1.225),   # reference_air_density [kg/m³] at sea level
        FT(11.72),   # fall_speed_coefficient [m^{1-b}/s]
        FT(0.41),    # fall_speed_exponent [-]
        NumberWeightedFallSpeed(),
        MassWeightedFallSpeed(),
        ReflectivityWeightedFallSpeed()
    )
end

Base.summary(::IceFallSpeed) = "IceFallSpeed"

function Base.show(io::IO, fs::IceFallSpeed)
    print(io, summary(fs), "(")
    print(io, "ρ₀=", fs.reference_air_density, ", ")
    print(io, "aᵥ=", fs.fall_speed_coefficient, ", ")
    print(io, "bᵥ=", fs.fall_speed_exponent, ")")
end

