#####
##### GPU/architecture support for P3 container structs
#####
##### When ice/rain integrals are tabulated (TabulatedFunction3D / TabulatedFunction1D),
##### the lookup table arrays must be transferred to the GPU. Scalar fields and
##### singleton integral types pass through unchanged.
#####

using Adapt: Adapt
using Oceananigans.Architectures: on_architecture

# --- IceFallSpeed ---

Adapt.adapt_structure(to, x::IceFallSpeed) =
    IceFallSpeed(x.reference_air_density,
                 Adapt.adapt(to, x.number_weighted),
                 Adapt.adapt(to, x.mass_weighted),
                 Adapt.adapt(to, x.reflectivity_weighted))

Oceananigans.Architectures.on_architecture(arch, x::IceFallSpeed) =
    IceFallSpeed(x.reference_air_density,
                 on_architecture(arch, x.number_weighted),
                 on_architecture(arch, x.mass_weighted),
                 on_architecture(arch, x.reflectivity_weighted))

# --- IceDeposition ---

Adapt.adapt_structure(to, x::IceDeposition) =
    IceDeposition(x.thermal_conductivity,
                  x.vapor_diffusivity,
                  Adapt.adapt(to, x.ventilation),
                  Adapt.adapt(to, x.ventilation_enhanced),
                  Adapt.adapt(to, x.small_ice_ventilation_constant),
                  Adapt.adapt(to, x.small_ice_ventilation_reynolds),
                  Adapt.adapt(to, x.large_ice_ventilation_constant),
                  Adapt.adapt(to, x.large_ice_ventilation_reynolds))

Oceananigans.Architectures.on_architecture(arch, x::IceDeposition) =
    IceDeposition(x.thermal_conductivity,
                  x.vapor_diffusivity,
                  on_architecture(arch, x.ventilation),
                  on_architecture(arch, x.ventilation_enhanced),
                  on_architecture(arch, x.small_ice_ventilation_constant),
                  on_architecture(arch, x.small_ice_ventilation_reynolds),
                  on_architecture(arch, x.large_ice_ventilation_constant),
                  on_architecture(arch, x.large_ice_ventilation_reynolds))

# --- IceBulkProperties ---

Adapt.adapt_structure(to, x::IceBulkProperties) =
    IceBulkProperties(x.maximum_mean_diameter,
                      x.minimum_mean_diameter,
                      Adapt.adapt(to, x.effective_radius),
                      Adapt.adapt(to, x.mean_diameter),
                      Adapt.adapt(to, x.mean_density),
                      Adapt.adapt(to, x.reflectivity),
                      Adapt.adapt(to, x.slope),
                      Adapt.adapt(to, x.shape),
                      Adapt.adapt(to, x.shedding))

Oceananigans.Architectures.on_architecture(arch, x::IceBulkProperties) =
    IceBulkProperties(x.maximum_mean_diameter,
                      x.minimum_mean_diameter,
                      on_architecture(arch, x.effective_radius),
                      on_architecture(arch, x.mean_diameter),
                      on_architecture(arch, x.mean_density),
                      on_architecture(arch, x.reflectivity),
                      on_architecture(arch, x.slope),
                      on_architecture(arch, x.shape),
                      on_architecture(arch, x.shedding))

# --- IceCollection ---

Adapt.adapt_structure(to, x::IceCollection) =
    IceCollection(x.ice_cloud_collection_efficiency,
                  x.ice_rain_collection_efficiency,
                  Adapt.adapt(to, x.aggregation),
                  Adapt.adapt(to, x.rain_collection))

Oceananigans.Architectures.on_architecture(arch, x::IceCollection) =
    IceCollection(x.ice_cloud_collection_efficiency,
                  x.ice_rain_collection_efficiency,
                  on_architecture(arch, x.aggregation),
                  on_architecture(arch, x.rain_collection))

# --- IceSixthMoment ---

Adapt.adapt_structure(to, x::IceSixthMoment) =
    IceSixthMoment(Adapt.adapt(to, x.rime),
                   Adapt.adapt(to, x.deposition),
                   Adapt.adapt(to, x.deposition1),
                   Adapt.adapt(to, x.melt1),
                   Adapt.adapt(to, x.melt2),
                   Adapt.adapt(to, x.melt_all1),
                   Adapt.adapt(to, x.melt_all2),
                   Adapt.adapt(to, x.shedding),
                   Adapt.adapt(to, x.aggregation),
                   Adapt.adapt(to, x.sublimation),
                   Adapt.adapt(to, x.sublimation1))

Oceananigans.Architectures.on_architecture(arch, x::IceSixthMoment) =
    IceSixthMoment(on_architecture(arch, x.rime),
                   on_architecture(arch, x.deposition),
                   on_architecture(arch, x.deposition1),
                   on_architecture(arch, x.melt1),
                   on_architecture(arch, x.melt2),
                   on_architecture(arch, x.melt_all1),
                   on_architecture(arch, x.melt_all2),
                   on_architecture(arch, x.shedding),
                   on_architecture(arch, x.aggregation),
                   on_architecture(arch, x.sublimation),
                   on_architecture(arch, x.sublimation1))

# --- IceLambdaLimiter ---

Adapt.adapt_structure(to, x::IceLambdaLimiter) =
    IceLambdaLimiter(Adapt.adapt(to, x.small_q),
                     Adapt.adapt(to, x.large_q))

Oceananigans.Architectures.on_architecture(arch, x::IceLambdaLimiter) =
    IceLambdaLimiter(on_architecture(arch, x.small_q),
                     on_architecture(arch, x.large_q))

# --- IceRainCollection ---

Adapt.adapt_structure(to, x::IceRainCollection) =
    IceRainCollection(Adapt.adapt(to, x.mass),
                      Adapt.adapt(to, x.number),
                      Adapt.adapt(to, x.sixth_moment))

Oceananigans.Architectures.on_architecture(arch, x::IceRainCollection) =
    IceRainCollection(on_architecture(arch, x.mass),
                      on_architecture(arch, x.number),
                      on_architecture(arch, x.sixth_moment))

# --- P3 lookup table wrappers ---

Adapt.adapt_structure(to, x::P3LookupTable1) =
    P3LookupTable1(Adapt.adapt(to, x.fall_speed),
                   Adapt.adapt(to, x.deposition),
                   Adapt.adapt(to, x.bulk_properties),
                   Adapt.adapt(to, x.collection),
                   Adapt.adapt(to, x.sixth_moment),
                   Adapt.adapt(to, x.lambda_limiter),
                   Adapt.adapt(to, x.ice_rain))

Oceananigans.Architectures.on_architecture(arch, x::P3LookupTable1) =
    P3LookupTable1(on_architecture(arch, x.fall_speed),
                   on_architecture(arch, x.deposition),
                   on_architecture(arch, x.bulk_properties),
                   on_architecture(arch, x.collection),
                   on_architecture(arch, x.sixth_moment),
                   on_architecture(arch, x.lambda_limiter),
                   on_architecture(arch, x.ice_rain))

Adapt.adapt_structure(to, x::P3LookupTable2) =
    P3LookupTable2(Adapt.adapt(to, x.mass),
                   Adapt.adapt(to, x.number),
                   Adapt.adapt(to, x.sixth_moment))

Oceananigans.Architectures.on_architecture(arch, x::P3LookupTable2) =
    P3LookupTable2(on_architecture(arch, x.mass),
                   on_architecture(arch, x.number),
                   on_architecture(arch, x.sixth_moment))

Adapt.adapt_structure(to, x::P3LookupTable3) =
    P3LookupTable3(Adapt.adapt(to, x.shape),
                   Adapt.adapt(to, x.slope),
                   Adapt.adapt(to, x.mean_density))

Oceananigans.Architectures.on_architecture(arch, x::P3LookupTable3) =
    P3LookupTable3(on_architecture(arch, x.shape),
                   on_architecture(arch, x.slope),
                   on_architecture(arch, x.mean_density))

Adapt.adapt_structure(to, x::P3LookupTables) =
    P3LookupTables(Adapt.adapt(to, x.table_1),
                   Adapt.adapt(to, x.table_2),
                   Adapt.adapt(to, x.table_3))

Oceananigans.Architectures.on_architecture(arch, x::P3LookupTables) =
    P3LookupTables(on_architecture(arch, x.table_1),
                   on_architecture(arch, x.table_2),
                   on_architecture(arch, x.table_3))

# --- IceProperties ---

Adapt.adapt_structure(to, x::IceProperties) =
    IceProperties(x.minimum_rime_density,
                  x.maximum_rime_density,
                  x.maximum_shape_parameter,
                  x.minimum_reflectivity,
                  Adapt.adapt(to, x.fall_speed),
                  Adapt.adapt(to, x.deposition),
                  Adapt.adapt(to, x.bulk_properties),
                  Adapt.adapt(to, x.collection),
                  Adapt.adapt(to, x.sixth_moment),
                  Adapt.adapt(to, x.lambda_limiter),
                  Adapt.adapt(to, x.ice_rain);
                  lookup_tables = Adapt.adapt(to, x.lookup_tables))

Oceananigans.Architectures.on_architecture(arch, x::IceProperties) =
    IceProperties(x.minimum_rime_density,
                  x.maximum_rime_density,
                  x.maximum_shape_parameter,
                  x.minimum_reflectivity,
                  on_architecture(arch, x.fall_speed),
                  on_architecture(arch, x.deposition),
                  on_architecture(arch, x.bulk_properties),
                  on_architecture(arch, x.collection),
                  on_architecture(arch, x.sixth_moment),
                  on_architecture(arch, x.lambda_limiter),
                  on_architecture(arch, x.ice_rain);
                  lookup_tables = on_architecture(arch, x.lookup_tables))

# --- RainProperties ---

Adapt.adapt_structure(to, x::RainProperties) =
    RainProperties(x.maximum_mean_diameter,
                   x.fall_speed_coefficient,
                   x.fall_speed_exponent,
                   Adapt.adapt(to, x.shape_parameter),
                   Adapt.adapt(to, x.velocity_number),
                   Adapt.adapt(to, x.velocity_mass),
                   Adapt.adapt(to, x.evaporation))

Oceananigans.Architectures.on_architecture(arch, x::RainProperties) =
    RainProperties(x.maximum_mean_diameter,
                   x.fall_speed_coefficient,
                   x.fall_speed_exponent,
                   on_architecture(arch, x.shape_parameter),
                   on_architecture(arch, x.velocity_number),
                   on_architecture(arch, x.velocity_mass),
                   on_architecture(arch, x.evaporation))

# --- PredictedParticlePropertiesMicrophysics ---

Adapt.adapt_structure(to, x::PredictedParticlePropertiesMicrophysics) =
    PredictedParticlePropertiesMicrophysics(
        x.water_density,
        x.minimum_mass_mixing_ratio,
        x.minimum_number_mixing_ratio,
        Adapt.adapt(to, x.ice),
        Adapt.adapt(to, x.rain),
        x.cloud,
        x.process_rates,
        Adapt.adapt(to, x.precipitation_boundary_condition))

Oceananigans.Architectures.on_architecture(arch, x::PredictedParticlePropertiesMicrophysics) =
    PredictedParticlePropertiesMicrophysics(
        x.water_density,
        x.minimum_mass_mixing_ratio,
        x.minimum_number_mixing_ratio,
        on_architecture(arch, x.ice),
        on_architecture(arch, x.rain),
        x.cloud,
        x.process_rates,
        on_architecture(arch, x.precipitation_boundary_condition))
