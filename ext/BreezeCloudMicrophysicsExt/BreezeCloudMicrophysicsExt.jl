module BreezeCloudMicrophysicsExt

using CloudMicrophysics: CloudMicrophysics
using CloudMicrophysics.Parameters: Parameters0M, Rain, Snow, CloudIce, CloudLiquid, CollisionEff
using CloudMicrophysics.Parameters: Blk1MVelType, Blk1MVelTypeRain, Blk1MVelTypeSnow
using CloudMicrophysics.Parameters: AirProperties
# Two-moment parameters
using CloudMicrophysics.Parameters: SB2006, StokesRegimeVelType, SB2006VelType, Chen2022VelTypeRain

using CloudMicrophysics.Microphysics0M: remove_precipitation

using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    accretion,
    terminal_velocity

# Two-moment microphysics
using CloudMicrophysics: Microphysics2M as CM2

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels,
    materialize_microphysical_fields,
    update_microphysical_fields!,
    compute_moisture_fractions,
    microphysical_velocities

using Breeze.Thermodynamics:
    MoistureMassFractions,
    density,
    with_moisture,
    temperature,
    PlanarLiquidSurface,
    PlanarIceSurface,
    saturation_vapor_pressure,
    saturation_specific_humidity,
    supersaturation,
    liquid_latent_heat,
    vapor_gas_constant

using Breeze.Microphysics:
    center_field_tuple,
    BulkMicrophysics,
    FourCategories,
    SaturationAdjustment,
    WarmPhaseSaturationAdjustment,
    MixedPhaseSaturationAdjustment,
    AbstractCondensateFormation,
    ConstantRateCondensateFormation,
    NonEquilibriumCloudFormation,
    condensation_rate,
    deposition_rate,
    adjust_thermodynamic_state

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Center, Face, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ZeroField, ZFaceField
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, BoundaryCondition, Open
using Adapt: Adapt, adapt

include("cloud_microphysics_translations.jl")
include("zero_moment_microphysics.jl")
include("one_moment_microphysics.jl")
include("one_moment_helpers.jl")
include("two_moment_microphysics.jl")
include("two_moment_helpers.jl")

end # module BreezeCloudMicrophysicsExt
