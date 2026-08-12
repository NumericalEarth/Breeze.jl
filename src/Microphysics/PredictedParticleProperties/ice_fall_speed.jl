#####
##### Ice Fall Speed
#####
##### Terminal velocity integrals over the ice particle size distribution.
##### P3 computes number- and mass-weighted fall speeds.
#####

"""
    IceFallSpeed

Ice terminal velocity power law parameters and weighted fall speed integrals.
See [`IceFallSpeed`](@ref) constructor for details.
"""
struct IceFallSpeed{FT, N, M}
    reference_air_density :: FT
    number_weighted :: N
    mass_weighted :: M
end

"""
$(TYPEDSIGNATURES)

Construct `IceFallSpeed` with parameters and quadrature-based integrals.

Ice particle terminal velocity uses the [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005)
Best-number formulation with air density correction exponent 0.54 from
[Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007). The reference density
``ρ_0`` matches the reference conditions (T=253.15 K, P=600 hPa) at which
the P3 lookup tables are computed.

Two weighted fall speeds are computed by integrating over the size distribution:

- **Number-weighted** ``V_n``: For number flux (sedimentation of particle count)
- **Mass-weighted** ``V_m``: For mass flux (precipitation rate)

# Keyword Arguments

- `thermodynamic_constants`: Source of the dry-air gas constant used to diagnose
  the default reference-air density.
- `reference_air_density`: Reference ρ₀ [kg/m³], default ≈0.825 (P3 mid-troposphere reference)

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 20.
"""
function IceFallSpeed(FT::Type{<:AbstractFloat} = Float64;
                      thermodynamic_constants = ThermodynamicConstants(FT),
                      reference_air_density = 60000 / (dry_air_gas_constant(thermodynamic_constants) * 253.15))
    return IceFallSpeed(FT(reference_air_density), nothing, nothing)
end

Base.summary(::IceFallSpeed) = "IceFallSpeed"

function Base.show(io::IO, fs::IceFallSpeed)
    print(io, summary(fs), "(")
    print(io, "ρ₀=", fs.reference_air_density, ")")
end
