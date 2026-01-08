"""
    PredictedParticleProperties

Predicted Particle Properties (P3) microphysics scheme implementation.

P3 is a bulk microphysics scheme that uses a single ice category with
continuously predicted properties (rime fraction, rime density, liquid fraction)
rather than multiple discrete ice categories.

# Key Features

- Single ice category with predicted properties
- 3-moment ice (mass, number, reflectivity/6th moment)
- Predicted liquid fraction on ice particles
- Rime fraction and rime density evolution
- Compatible with both quadrature and lookup table evaluation

# References

- Morrison and Milbrandt (2015), J. Atmos. Sci. - Original P3 scheme
- Milbrandt and Morrison (2016), J. Atmos. Sci. - 3-moment ice
- Milbrandt et al. (2024), J. Adv. Model. Earth Syst. - Predicted liquid fraction

# Source Code

Based on [P3-microphysics v5.5.0](https://github.com/P3-microphysics/P3-microphysics)
"""
module PredictedParticleProperties

export
    # Main scheme type
    PredictedParticlePropertiesMicrophysics,
    P3Microphysics,
    
    # Ice properties
    IceProperties,
    IceFallSpeed,
    IceDeposition,
    IceBulkProperties,
    IceCollection,
    IceSixthMoment,
    IceLambdaLimiter,
    IceRainCollection,
    
    # Rain and cloud properties
    RainProperties,
    CloudProperties,
    
    # Integral types (abstract)
    AbstractP3Integral,
    AbstractIceIntegral,
    AbstractRainIntegral,
    AbstractFallSpeedIntegral,
    AbstractDepositionIntegral,
    AbstractBulkPropertyIntegral,
    AbstractCollectionIntegral,
    AbstractSixthMomentIntegral,
    AbstractLambdaLimiterIntegral,
    
    # Integral types (concrete) - Fall speed
    NumberWeightedFallSpeed,
    MassWeightedFallSpeed,
    ReflectivityWeightedFallSpeed,
    
    # Integral types (concrete) - Deposition
    Ventilation,
    VentilationEnhanced,
    SmallIceVentilationConstant,
    SmallIceVentilationReynolds,
    LargeIceVentilationConstant,
    LargeIceVentilationReynolds,
    
    # Integral types (concrete) - Bulk properties
    EffectiveRadius,
    MeanDiameter,
    MeanDensity,
    Reflectivity,
    SlopeParameter,
    ShapeParameter,
    SheddingRate,
    
    # Integral types (concrete) - Collection
    AggregationNumber,
    RainCollectionNumber,
    
    # Integral types (concrete) - Sixth moment
    SixthMomentRime,
    SixthMomentDeposition,
    SixthMomentDeposition1,
    SixthMomentMelt1,
    SixthMomentMelt2,
    SixthMomentAggregation,
    SixthMomentShedding,
    SixthMomentSublimation,
    SixthMomentSublimation1,
    
    # Integral types (concrete) - Lambda limiter
    SmallQLambdaLimit,
    LargeQLambdaLimit,
    
    # Integral types (concrete) - Rain
    RainShapeParameter,
    RainVelocityNumber,
    RainVelocityMass,
    RainEvaporation,
    
    # Integral types (concrete) - Ice-rain collection
    IceRainMassCollection,
    IceRainNumberCollection,
    IceRainSixthMomentCollection,
    
    # Tabulated wrapper
    TabulatedIntegral,
    
    # Interface functions
    prognostic_field_names,
    
    # Quadrature
    evaluate,
    IceSizeDistributionState,
    chebyshev_gauss_nodes_weights,
    
    # Tabulation
    tabulate,
    TabulationParameters

#####
##### Integral types (must be first - no dependencies)
#####

include("integral_types.jl")

#####
##### Ice concept containers
#####

include("ice_fall_speed.jl")
include("ice_deposition.jl")
include("ice_bulk_properties.jl")
include("ice_collection.jl")
include("ice_sixth_moment.jl")
include("ice_lambda_limiter.jl")
include("ice_rain_collection.jl")
include("ice_properties.jl")

#####
##### Rain and cloud properties
#####

include("rain_properties.jl")
include("cloud_properties.jl")

#####
##### Main scheme type
#####

include("p3_scheme.jl")

#####
##### Size distribution and quadrature (depends on types above)
#####

include("size_distribution.jl")
include("quadrature.jl")
include("tabulation.jl")

end # module PredictedParticleProperties

