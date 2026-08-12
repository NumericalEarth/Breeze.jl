#####
##### Ice Properties
#####
##### Container combining all ice particle property concepts.
#####

"""
    IceProperties

Ice particle properties for P3. See [`IceProperties()`](@ref) constructor.
"""
struct IceProperties{FT, FS, DP, BP, CL, LL, IR, TABLES}
    # Top-level parameters
    minimum_rime_density :: FT
    maximum_rime_density :: FT
    maximum_shape_parameter :: FT
    # Concept containers
    fall_speed :: FS
    deposition :: DP
    bulk_properties :: BP
    collection :: CL
    lambda_limiter :: LL
    ice_rain :: IR
    lookup_tables :: TABLES
end

function IceProperties(minimum_rime_density,
                       maximum_rime_density,
                       maximum_shape_parameter,
                       fall_speed,
                       deposition,
                       bulk_properties,
                       collection,
                       lambda_limiter,
                       ice_rain;
                       lookup_tables = nothing)
    return IceProperties(
        minimum_rime_density,
        maximum_rime_density,
        maximum_shape_parameter,
        fall_speed,
        deposition,
        bulk_properties,
        collection,
        lambda_limiter,
        ice_rain,
        lookup_tables)
end

"""
$(TYPEDSIGNATURES)

Construct ice particle properties with parameters and integrals for the P3 scheme.

Ice particles in P3 span a continuum from small pristine crystals to large
heavily-rimed graupel. The particle mass ``m(D)`` follows a piecewise power
law depending on size ``D``, rime fraction ``Fᶠ``, and rime density ``ρᶠ``.

# Physical Concepts

This container organizes all ice-related computations:

- **Fall speed**: Terminal velocity integrals for sedimentation
  (number-weighted, mass-weighted)
- **Deposition**: Ventilation integrals for vapor diffusion growth
- **Bulk properties**: Population-averaged diameter, density, reflectivity
- **Collection**: Integrals for aggregation and riming rates
- **Lambda limiter**: Constraints on size distribution slope

# Keyword Arguments

- `minimum_rime_density`: Lower bound for ρᶠ [kg/m³], default 50
- `maximum_rime_density`: Upper bound for ρᶠ [kg/m³], default 900 (pure ice)
- `maximum_shape_parameter`: Upper limit on μ [-], default 20

# References

The mass-diameter relationship is from
[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
function IceProperties(FT::Type{<:AbstractFloat} = Float64;
                       minimum_rime_density = 50,
                       maximum_rime_density = 900,
                       maximum_shape_parameter = 20,
                       lookup_tables = nothing)
    return IceProperties(
        FT(minimum_rime_density),
        FT(maximum_rime_density),
        FT(maximum_shape_parameter),
        IceFallSpeed(FT),
        IceDeposition(FT),
        IceBulkProperties(FT),
        IceCollection(),
        IceLambdaLimiter(),
        IceRainCollection();
        lookup_tables
    )
end

Base.summary(::IceProperties) = "IceProperties"

function Base.show(io::IO, ice::IceProperties)
    print(io, summary(ice), '\n')
    print(io, "├── ρᶠ: [", ice.minimum_rime_density, ", ", ice.maximum_rime_density, "] kg/m³\n")
    print(io, "├── μmax: ", ice.maximum_shape_parameter, "\n")
    print(io, "├── ", ice.fall_speed, "\n")
    print(io, "├── ", ice.deposition, "\n")
    print(io, "├── ", ice.bulk_properties, "\n")
    print(io, "├── ", ice.collection, "\n")
    print(io, "├── ", ice.lambda_limiter, "\n")
    print(io, "└── ", ice.ice_rain)
end
