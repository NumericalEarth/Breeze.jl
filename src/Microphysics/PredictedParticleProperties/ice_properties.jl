#####
##### Ice Properties
#####
##### Container combining all ice particle property concepts.
#####

"""
    IceProperties{FT, FS, DP, BP, CL, M6, LL, IR}

Complete ice particle properties for the P3 scheme.

This container combines all ice-related concepts: fall speed, deposition,
bulk properties, collection, sixth moment evolution, lambda limiting,
and ice-rain collection.

# Fields

## Top-level parameters
- `minimum_rime_density`: Minimum rime density ρ_rim,min [kg/m³]
- `maximum_rime_density`: Maximum rime density ρ_rim,max [kg/m³]
- `maximum_shape_parameter`: Maximum shape parameter μ_max [-]
- `minimum_reflectivity`: Minimum reflectivity for 3-moment [m⁶/m³]

## Concept containers (each with parameters + integrals)
- `fall_speed`: [`IceFallSpeed`](@ref) - terminal velocity integrals
- `deposition`: [`IceDeposition`](@ref) - vapor diffusion integrals
- `bulk_properties`: [`IceBulkProperties`](@ref) - population averages
- `collection`: [`IceCollection`](@ref) - collision-coalescence
- `sixth_moment`: [`IceSixthMoment`](@ref) - M₆ tendencies (3-moment)
- `lambda_limiter`: [`IceLambdaLimiter`](@ref) - PSD constraints
- `ice_rain`: [`IceRainCollection`](@ref) - ice collecting rain

# References

Morrison and Milbrandt (2015), Milbrandt and Morrison (2016), Milbrandt et al. (2024)
"""
struct IceProperties{FT, FS, DP, BP, CL, M6, LL, IR}
    # Top-level parameters
    minimum_rime_density :: FT
    maximum_rime_density :: FT
    maximum_shape_parameter :: FT
    minimum_reflectivity :: FT
    # Concept containers
    fall_speed :: FS
    deposition :: DP
    bulk_properties :: BP
    collection :: CL
    sixth_moment :: M6
    lambda_limiter :: LL
    ice_rain :: IR
end

"""
    IceProperties(FT=Float64)

Construct `IceProperties` with default parameters and all concept containers.

Default parameters from Morrison and Milbrandt (2015).
"""
function IceProperties(FT::Type{<:AbstractFloat} = Float64)
    return IceProperties(
        # Top-level parameters
        FT(50.0),    # minimum_rime_density [kg/m³]
        FT(900.0),   # maximum_rime_density [kg/m³] (pure ice)
        FT(10.0),    # maximum_shape_parameter [-]
        FT(1e-22),   # minimum_reflectivity [m⁶/m³]
        # Concept containers
        IceFallSpeed(FT),
        IceDeposition(FT),
        IceBulkProperties(FT),
        IceCollection(FT),
        IceSixthMoment(),
        IceLambdaLimiter(),
        IceRainCollection()
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
    print(io, "├── ", ice.sixth_moment, "\n")
    print(io, "├── ", ice.lambda_limiter, "\n")
    print(io, "└── ", ice.ice_rain)
end

