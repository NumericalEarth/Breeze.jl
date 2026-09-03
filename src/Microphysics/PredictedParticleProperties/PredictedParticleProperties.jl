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
    ProcessRate,
    NumericalFloors,

    # Ice properties
    Ice,
    IceFallSpeed,
    IceDeposition,
    IceBulk,
    IceCollection,
    IceLambdaLimiter,
    IceRainCollection,

    # Rain and cloud droplet properties
    Rain,
    Cloud,

    # Empirical parameter containers for the warm-phase fits
    CloudShape,
    RainFallSpeed,
    RainVentilation,

    # Rain PSD quadrature evaluators
    RainMassWeightedVelocity,
    RainNumberWeightedVelocity,
    RainVelocityDiameterIntegral,

    # Tabulated wrapper
    TabulatedFunction1D,
    TabulatedFunction4D,
    TabulatedFunction5D,
    RimeDensityIndexedTable4D,
    RimeDensityIndexedTable5D,
    P3Table4D,

    # Transport properties
    air_transport_properties,

    # PSD correction functions
    psd_correction_spherical_volume,
    liu_daum_shape_parameter,

    # Interface functions
    prognostic_field_names,

    # Quadrature helpers (defined in `Breeze.Utils`, re-exported for convenience)
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

using Adapt: Adapt
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU, on_architecture
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: MoistureMassFractions,
                             PlanarIceSurface,
                             PlanarLiquidSurface,
                             ThermodynamicConstants,
                             adjustment_saturation_specific_humidity,
                             air_pressure,
                             density,
                             dry_air_gas_constant,
                             ice_latent_heat,
                             liquid_latent_heat,
                             mixture_heat_capacity,
                             psychrometric_correction,
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             temperature,
                             vapor_gas_constant,
                             with_moisture
using Breeze.Utils: @adapt_architecture,
                    chebyshev_gauss_nodes_weights,
                    jacobian_diameter_transform,
                    safe_divide,
                    transform_to_diameter

#####
##### Ice concept containers and the lookup tables they are read from
#####

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
##### Tabulation
#####

include("tabulated_function_adapters.jl")
include("lookup_table_format.jl")
include("lookup_table_reader.jl")

#####
##### Rain PSD quadrature evaluators
#####

include("rain_quadrature.jl")

#####
##### GPU/architecture adaptation
#####
##### The ice integrals (4D/5D lookup tables) and rain integrals (1D tables built
##### from quadrature) hold lookup arrays that must be transferred to the GPU.
##### `@adapt_architecture` (from `Breeze.Utils`) generates the `Adapt.adapt_structure`
##### and `on_architecture` methods that walk each container's fields, so the field
##### list is not repeated twice per type. Scalar fields pass through unchanged.
#####

@adapt_architecture RimeDensityIndexedTable4D
@adapt_architecture RimeDensityIndexedTable5D
@adapt_architecture IceFallSpeed
@adapt_architecture IceDeposition
@adapt_architecture IceBulk
@adapt_architecture IceCollection
@adapt_architecture IceLambdaLimiter
@adapt_architecture IceRainCollection
@adapt_architecture Ice
@adapt_architecture Rain
@adapt_architecture PredictedParticlePropertiesMicrophysics

#####
##### Process-rate helpers, the shared Table-1 lookups (`P3IceLookups`), CCN activation,
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
