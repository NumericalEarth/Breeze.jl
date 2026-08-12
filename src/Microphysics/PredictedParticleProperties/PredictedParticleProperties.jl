"""
    PredictedParticleProperties

Predicted Particle Properties (P3) microphysics scheme implementation.

P3 is a bulk microphysics scheme that uses a single ice category with
continuously predicted properties (rime fraction, rime density, liquid fraction)
rather than multiple discrete ice categories.

# Key Features

- Single ice category with predicted properties
- Two-moment ice (mass and number)
- Predicted liquid fraction on ice particles
- Rime fraction and rime density evolution
- Ice-side integrals are read from the ASCII lookup tables; rain 1D
  integrals are tabulated at startup using Chebyshev–Gauss quadrature

# Complete Reference List

This implementation is based on the following P3 papers:

1. **Morrison & Milbrandt (2015a)** - Original P3: m(D), A(D), V(D), process rates
   [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)

2. **Morrison et al. (2015b)** - Part II: Case study validation
   [Morrison et al. (2015b)](@cite Morrison2015part2)

3. **Milbrandt & Morrison (2016)** - Part III: Multiple ice categories (NOT implemented)
   [Milbrandt and Morrison (2016)](@cite MilbrandtMorrison2016)

4. **Milbrandt et al. (2025)** - Predicted liquid fraction: shedding, refreezing
   [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction)

# Source Code

Based on [P3-microphysics v5.5.0](https://github.com/P3-microphysics/P3-microphysics)

# Not Implemented

- Three-moment ice (prognostic reflectivity) from Milbrandt et al. (2021)
- Multiple free ice categories from Milbrandt & Morrison (2016)
- Lookup table I/O for all table types
"""
module PredictedParticleProperties

export
    # Main scheme type
    PredictedParticlePropertiesMicrophysics,
    P3Microphysics,
    P3MicrophysicalState,
    ProcessRateParameters,
    NumericalFloors,

    # Ice properties
    IceProperties,
    IceFallSpeed,
    IceDeposition,
    IceBulkProperties,
    IceCollection,
    IceLambdaLimiter,
    IceRainCollection,
    P3IceIntegralsTable,
    P3RainIceCollectionTable,
    P3LookupTables,

    # Rain and cloud droplet properties
    RainProperties,
    CloudDropletProperties,

    # Rain PSD quadrature evaluators
    RainMassWeightedVelocityEvaluator,
    RainNumberWeightedVelocityEvaluator,
    RainEvaporationVentilationEvaluator,

    # Tabulated wrapper
    TabulatedFunction1D,
    TabulatedFunction5D,
    TabulatedFunction6D,
    RimeDensityIndexedTable5D,
    RimeDensityIndexedTable6D,
    P3Table5D,

    # Transport properties
    air_transport_properties,

    # PSD correction functions
    psd_correction_spherical_volume,
    liu_daum_shape_parameter,

    # Interface functions
    prognostic_field_names,

    # Quadrature helpers
    chebyshev_gauss_nodes_weights,

    # Lookup table reader
    read_lookup_tables,
    tabulate_rain_from_quadrature,
    rime_density_index,

    # Aerosol activation (prognostic CCN)
    AerosolMode,
    AerosolActivation,
    activated_number,
    total_activated_number,
    sum_aerosol_number,
    prognostic_ccn_activation_rate

using DocStringExtensions: TYPEDSIGNATURES, TYPEDFIELDS, TYPEDEF
using SpecialFunctions: erf

using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant

#####
##### Ice concept containers
#####

include("ice_fall_speed.jl")
include("ice_deposition.jl")
include("ice_bulk_properties.jl")
include("ice_collection.jl")
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
##### Transport properties (T,P-dependent Kᵃ, Dᵛ, ν)
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
##### Warm-rain scheme selector types
#####

include("warm_rain_schemes.jl")

#####
##### Main scheme type
#####

include("p3_scheme.jl")

#####
##### Quadrature (depends on types above)
#####

include("quadrature.jl")

#####
##### Tabulation (depends on quadrature)
#####

include("tabulated_function_adapters.jl")
include("lookup_table_format.jl")
include("lookup_table_reader.jl")

#####
##### Rain PSD quadrature evaluators (must follow quadrature.jl)
#####

include("rain_quadrature.jl")

#####
##### GPU/architecture adaptation methods
#####

include("gpu_adaptation.jl")

#####
##### Process-rate helpers, tabulated kernels, CCN activation,
##### and the coupled saturation-adjustment solver
#####

include("process_rate_helpers.jl")
include("tabulated_kernels.jl")
include("ccn_activation_rates.jl")
include("coupled_saturation_adjustment.jl")

#####
##### P3ProcessRates struct + constructor and the per-process rate calculations
#####

include("process_rates.jl")
include("rain_process_rates.jl")
include("melting_rates.jl")
include("ice_nucleation_rates.jl")
include("ice_aggregation_rates.jl")
include("riming_rates.jl")
include("wet_ice_processes.jl")
include("terminal_velocities.jl")

#####
##### Prognostic tendency functions (operate on the P3ProcessRates struct)
#####

include("prognostic_tendencies.jl")

#####
##### AtmosphereModel interface (must be last - depends on all types)
#####

include("p3_microphysical_state.jl")
include("p3_microphysical_tendencies.jl")
include("p3_driver.jl")

end # module PredictedParticleProperties
