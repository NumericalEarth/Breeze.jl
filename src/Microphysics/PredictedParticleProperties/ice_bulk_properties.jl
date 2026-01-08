#####
##### Ice Bulk Properties
#####
##### Population-averaged properties computed by integrating over the
##### ice particle size distribution.
#####

"""
    IceBulkProperties{FT, EF, DM, RH, RF, LA, MU, SH}

Ice bulk property integrals over the size distribution.

These integrals compute population-averaged quantities used for
radiation, radar reflectivity, and diagnostics.

# Fields

## Parameters
- `maximum_mean_diameter`: Upper limit on mean diameter D_m [m]
- `minimum_mean_diameter`: Lower limit on mean diameter D_m [m]

## Integrals
- `effective_radius`: Effective radius for radiation [m]
- `mean_diameter`: Mass-weighted mean diameter [m]
- `mean_density`: Mass-weighted mean particle density [kg/m³]
- `reflectivity`: Radar reflectivity factor Z [m⁶/m³]
- `slope`: Slope parameter λ of gamma distribution [1/m]
- `shape`: Shape parameter μ of gamma distribution [-]
- `shedding`: Meltwater shedding rate [kg/kg/s]

# References

Morrison and Milbrandt (2015), Field et al. (2007)
"""
struct IceBulkProperties{FT, EF, DM, RH, RF, LA, MU, SH}
    # Parameters
    maximum_mean_diameter :: FT
    minimum_mean_diameter :: FT
    # Integrals
    effective_radius :: EF
    mean_diameter :: DM
    mean_density :: RH
    reflectivity :: RF
    slope :: LA
    shape :: MU
    shedding :: SH
end

"""
    IceBulkProperties(FT=Float64)

Construct `IceBulkProperties` with default parameters and quadrature-based integrals.
"""
function IceBulkProperties(FT::Type{<:AbstractFloat} = Float64)
    return IceBulkProperties(
        FT(2e-2),    # maximum_mean_diameter [m] = 2 cm
        FT(1e-5),    # minimum_mean_diameter [m] = 10 μm
        EffectiveRadius(),
        MeanDiameter(),
        MeanDensity(),
        Reflectivity(),
        SlopeParameter(),
        ShapeParameter(),
        SheddingRate()
    )
end

Base.summary(::IceBulkProperties) = "IceBulkProperties"

function Base.show(io::IO, bp::IceBulkProperties)
    print(io, summary(bp), "(")
    print(io, "D_max=", bp.maximum_mean_diameter, ", ")
    print(io, "D_min=", bp.minimum_mean_diameter, ")")
end

