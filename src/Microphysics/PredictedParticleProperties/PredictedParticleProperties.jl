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
   [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)

2. **Morrison et al. (2015b)** - Part II: Case study validation
   [Morrison et al. (2015b)](@cite Morrison2015part2)

3. **Milbrandt & Morrison (2016)** - Part III: Multiple ice categories (NOT implemented)
   [Milbrandt and Morrison (2016)](@cite MilbrandtMorrison2016)

4. **Milbrandt et al. (2021)** - Three-moment ice: Z as prognostic, size sorting
   [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021)

5. **Milbrandt et al. (2024)** - Updated triple-moment formulation
   [Milbrandt et al. (2024)](@cite MilbrandtEtAl2024)

6. **Milbrandt et al. (2025)** - Predicted liquid fraction: shedding, refreezing
   [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction)

7. **Morrison et al. (2025)** - Complete three-moment implementation
   [Morrison et al. (2025)](@cite Morrison2025complete3moment)

# Source Code

Based on [P3-microphysics v5.5.0](https://github.com/P3-microphysics/P3-microphysics)

# Not Implemented

- Full multiple free ice categories from Milbrandt & Morrison (2016)
  (initial framework exists via `MultiIceCategory`)
- Full Fortran-format lookup table I/O for all table types
"""
module PredictedParticleProperties

export
    # Main scheme type
    PredictedParticlePropertiesMicrophysics,
    P3Microphysics,
    P3MicrophysicalState,
    ProcessRateParameters,

    # Multi-category ice
    MultiIceCategory,
    multi_category_ice_field_names,
    inter_category_collection,

    # Ice properties
    IceProperties,
    IceFallSpeed,
    IceDeposition,
    IceBulkProperties,
    IceCollection,
    IceSixthMoment,
    IceLambdaLimiter,
    IceRainCollection,
    P3LookupTable1,
    P3LookupTable2,
    P3LookupTable3,
    P3LookupTables,

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
    RayleighReflectivity,
    SlopeParameter,
    ShapeParameter,
    SheddingRate,

    # Integral types (concrete) - Collection
    AggregationNumber,
    RainCollectionNumber,
    CloudAerosolCollection,
    IceAerosolCollection,

    # Integral types (concrete) - Sixth moment
    SixthMomentRime,
    SixthMomentDeposition,
    SixthMomentDeposition1,
    SixthMomentMelt1,
    SixthMomentMelt2,
    SixthMomentMeltAll1,
    SixthMomentMeltAll2,
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

    # Rain PSD quadrature evaluators
    RainMassWeightedVelocityEvaluator,
    RainNumberWeightedVelocityEvaluator,
    RainEvaporationVentilationEvaluator,

    # Tabulated wrapper
    TabulatedIntegral,
    TabulatedFunction1D,
    TabulatedFunction2D,
    TabulatedFunction3D,
    TabulatedFunction4D,
    TabulatedFunction5D,
    TabulatedFunction6D,
    tabulated_function_1d,
    FortranTabulatedFunction5D,
    FortranTabulatedFunction6D,
    FortranTabulatedFunction3,
    P3Table5D,
    P3Table6D,

    # Transport properties
    air_transport_properties,

    # PSD correction functions
    psd_correction_spherical_volume,
    liu_daum_shape_parameter,

    # Interface functions
    prognostic_field_names,

    # Quadrature
    evaluate,
    IceSizeDistributionState,
    chebyshev_gauss_nodes_weights,

    # Fortran table reader
    read_fortran_lookup_tables,
    tabulate_rain_from_quadrature,
    rime_density_index,

    # Aerosol activation (prognostic CCN)
    AerosolMode,
    AerosolActivation,
    activated_number,
    total_activated_number,
    sum_aerosol_number,
    prognostic_ccn_activation_rate,

    # Lambda solver
    IceMassPowerLaw,
    P3Closure,
    TwoMomentClosure,
    ThreeMomentClosure,
    ThreeMomentClosureExact,
    ThreeMomentLookupClosure,
    FixedShapeParameter,
    ShapeParameterRelation,  # alias for TwoMomentClosure
    IceRegimeThresholds,
    IceDistributionParameters,
    DiameterBounds,
    solve_lambda,
    distribution_parameters,
    shape_parameter,
    ice_regime_thresholds,
    ice_mass,
    ice_mass_coefficients,
    intercept_parameter,
    lambda_bounds_from_diameter,
    enforce_diameter_bounds

using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS
using SpecialFunctions: loggamma, gamma_inc, erf

using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant

# Precomputed gas constants for GPU compatibility.
# ThermodynamicConstants() allocates and cannot be called inside GPU kernels,
# but the gas constants (R/M_d and R/M_v) are universal physical constants.
const DRY_AIR_GAS_CONSTANT = dry_air_gas_constant(ThermodynamicConstants())
const VAPOR_GAS_CONSTANT = vapor_gas_constant(ThermodynamicConstants())

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
include("lookup_tables.jl")
include("ice_properties.jl")

#####
##### PSD correction functions (analytical gamma-distribution factors)
##### Must precede cloud_droplet_properties.jl which uses psd_correction_spherical_volume.
#####

include("psd_corrections.jl")

#####
##### Rain and cloud properties
#####

include("rain_properties.jl")
include("cloud_droplet_properties.jl")

#####
##### Transport properties (T,P-dependent K_a, D_v, nu)
#####

include("transport_properties.jl")

#####
##### Process rate parameters
#####

include("process_rate_parameters.jl")

#####
##### Aerosol activation (prognostic CCN)
#####

include("aerosol_activation.jl")

#####
##### Main scheme type
#####

include("p3_scheme.jl")

#####
##### Size distribution and quadrature (depends on types above)
#####

include("size_distribution.jl")
include("quadrature.jl")

#####
##### Lambda solver (depends on mass-diameter relationship)
#####

include("lambda_solver.jl")

#####
##### Tabulation (depends on quadrature and lambda solver)
#####

include("table_generation_common.jl")
include("lookup_table_3.jl")
include("tabulated_function_adapters.jl")
include("fortran_table_reader.jl")

#####
##### Rain PSD quadrature evaluators (must follow quadrature.jl)
#####

include("rain_quadrature.jl")

#####
##### GPU/architecture adaptation methods
#####

include("gpu_adaptation.jl")

#####
##### Process rates
#####

include("process_rates.jl")
include("rain_process_rates.jl")
include("melting_rates.jl")
include("ice_nucleation_rates.jl")
include("collection_rates.jl")
include("terminal_velocities.jl")

#####
##### Multi-ice category support
#####

include("multi_ice_category.jl")

#####
##### AtmosphereModel interface (must be last - depends on all types)
#####

include("p3_interface.jl")

end # module PredictedParticleProperties
