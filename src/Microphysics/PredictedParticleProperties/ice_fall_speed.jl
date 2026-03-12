#####
##### Ice Fall Speed
#####
##### Terminal velocity integrals over the ice particle size distribution.
##### P3 computes number-, mass-, and reflectivity-weighted fall speeds.
#####

"""
    IceFallSpeed

Ice terminal velocity power law parameters and weighted fall speed integrals.
See [`IceFallSpeed`](@ref) constructor for details.
"""
struct IceFallSpeed{FT, N, M, Z}
    reference_air_density :: FT
    fall_speed_coefficient :: FT
    fall_speed_exponent :: FT
    number_weighted :: N
    mass_weighted :: M
    reflectivity_weighted :: Z
end

"""
$(TYPEDSIGNATURES)

Construct `IceFallSpeed` with parameters and quadrature-based integrals.

Ice particle terminal velocity follows a power law with air density correction:

```math
V(D) = a_V \\left(\\frac{ρ_0}{ρ}\\right)^{0.54} D^{b_V}
```

where ``a_V`` is `fall_speed_coefficient`, ``b_V`` is `fall_speed_exponent`,
and ``ρ_0`` is `reference_air_density`. The density correction exponent 0.54
follows [Heymsfield et al. (2006)](@cite HeymsfieldEtAl2006), and
``ρ_0`` matches the reference conditions (T=253.15 K, P=600 hPa) at which
the P3 lookup tables are computed.

Three weighted fall speeds are computed by integrating over the size distribution:

- **Number-weighted** ``V_n``: For number flux (sedimentation of particle count)
- **Mass-weighted** ``V_m``: For mass flux (precipitation rate)
- **Reflectivity-weighted** ``V_z``: For 3-moment scheme (6th moment flux)

# Keyword Arguments

- `reference_air_density`: Reference ρ₀ [kg/m³], default ≈0.825 (P3 mid-troposphere reference)
- `fall_speed_coefficient`: Coefficient aᵥ [m^{1-b}/s], default 11.72
- `fall_speed_exponent`: Exponent bᵥ [-], default 0.41

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 20,
[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) for reflectivity weighting.
"""
function IceFallSpeed(FT::Type{<:AbstractFloat} = Float64;
                      reference_air_density = 60000 / (287.15 * 253.15),
                      fall_speed_coefficient = 11.72,
                      fall_speed_exponent = 0.41)
    return IceFallSpeed(
        FT(reference_air_density),
        FT(fall_speed_coefficient),
        FT(fall_speed_exponent),
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
