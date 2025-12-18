module BreezeCloudMicrophysicsExt

using CloudMicrophysics: CloudMicrophysics
using CloudMicrophysics.Parameters: Parameters0M, Rain, Snow, CloudIce, CloudLiquid, CollisionEff
using CloudMicrophysics.Parameters: Blk1MVelType, Blk1MVelTypeRain, Blk1MVelTypeSnow
using CloudMicrophysics.Parameters: AirProperties
using CloudMicrophysics.Microphysics0M: remove_precipitation

using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    accretion,
    terminal_velocity

using Breeze
using Breeze.AtmosphereModels

using Breeze.Thermodynamics:
    MoistureMassFractions,
    density,
    with_moisture,
    temperature,
    PlanarLiquidSurface,
    saturation_vapor_pressure,
    saturation_specific_humidity,
    supersaturation,
    liquid_latent_heat,
    vapor_gas_constant,
    mixture_heat_capacity

using Breeze.Microphysics:
    center_field_tuple,
    BulkMicrophysics,
    FourCategories,
    SaturationAdjustment,
    WarmPhaseSaturationAdjustment,
    MixedPhaseSaturationAdjustment,
    NonEquilibriumCloudFormation,
    ImpenetrableBoundaryCondition,
    adjust_thermodynamic_state

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Center, Face, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ZeroField, ZFaceField
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Adapt: Adapt, adapt

import Breeze.AtmosphereModels:
    maybe_adjust_thermodynamic_state,
    prognostic_field_names,
    materialize_microphysical_fields,
    update_microphysical_fields!,
    compute_moisture_fractions,
    microphysical_tendency,
    microphysical_velocities,
    precipitation_rate,
    surface_precipitation_flux

include("cloud_microphysics_translations.jl")
include("zero_moment_microphysics.jl")
include("one_moment_microphysics.jl")
include("saturation_adjustment_one_moment.jl")
include("nonequilibrium_one_moment.jl")

end # module BreezeCloudMicrophysicsExt
