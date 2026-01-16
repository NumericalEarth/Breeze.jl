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

# Complete Reference List

This implementation is based on the following P3 papers:

1. **Morrison & Milbrandt (2015a)** - Original P3: m(D), A(D), V(D), process rates
   [Morrison2015parameterization](@citet)

2. **Morrison et al. (2015b)** - Part II: Case study validation
   [Morrison2015part2](@citet)

3. **Milbrandt & Morrison (2016)** - Part III: Multiple ice categories (NOT implemented)
   [MilbrandtMorrison2016](@citet)

4. **Milbrandt et al. (2021)** - Three-moment ice: Z as prognostic, size sorting
   [MilbrandtEtAl2021](@citet)

5. **Milbrandt et al. (2024)** - Updated triple-moment formulation
   [MilbrandtEtAl2024](@citet)

6. **Milbrandt et al. (2025)** - Predicted liquid fraction: shedding, refreezing
   [MilbrandtEtAl2025liquidfraction](@citet)

7. **Morrison et al. (2025)** - Complete three-moment implementation
   [Morrison2025complete3moment](@citet)

# Source Code

Based on [P3-microphysics v5.5.0](https://github.com/P3-microphysics/P3-microphysics)

# Not Implemented

- Multiple free ice categories from Milbrandt & Morrison (2016)
- Full process rate tendency functions (infrastructure is ready, rates are TODO)
"""
module PredictedParticleProperties

export
    # Main scheme type
    PredictedParticlePropertiesMicrophysics,
    P3Microphysics,
    ProcessRateParameters,

    # Ice properties
    IceProperties,
    IceFallSpeed,
    IceDeposition,
    IceBulkProperties,
    IceCollection,
    IceSixthMoment,
    IceLambdaLimiter,
    IceRainCollection,

    # Rain and cloud droplet properties
    RainProperties,
    CloudDropletProperties,

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
    NumberMomentLambdaLimit,
    MassMomentLambdaLimit,

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
    TabulationParameters,

    # Lambda solver
    IceMassPowerLaw,
    TwoMomentClosure,
    ThreeMomentClosure,
    ShapeParameterRelation,  # alias for TwoMomentClosure
    IceRegimeThresholds,
    IceDistributionParameters,
    solve_lambda,
    solve_shape_parameter,
    distribution_parameters,
    shape_parameter,
    ice_regime_thresholds,
    ice_mass,
    ice_mass_coefficients,
    intercept_parameter

using DocStringExtensions: TYPEDSIGNATURES
using SpecialFunctions: loggamma, gamma_inc

using Oceananigans: Oceananigans
using Breeze.AtmosphereModels: prognostic_field_names

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
include("cloud_droplet_properties.jl")

#####
##### Process rate parameters
#####

include("process_rate_parameters.jl")

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

#####
##### Lambda solver (depends on mass-diameter relationship)
#####

include("lambda_solver.jl")

#####
##### Process rates (Phase 1: rain, deposition, melting)
#####

include("process_rates.jl")

#####
##### AtmosphereModel interface (must be last - depends on all types)
#####

include("p3_interface.jl")

end # module PredictedParticleProperties
