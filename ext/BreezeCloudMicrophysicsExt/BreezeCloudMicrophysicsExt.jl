module BreezeCloudMicrophysicsExt

using CloudMicrophysics: CloudMicrophysics
using CloudMicrophysics.Parameters:
    Rain,
    Snow,
    CloudIce,
    CloudLiquid,
    Microphysics1MParams,
    TemperatureDependent,
    WithSupersaturation,
    SublimationOnly,
    CloudIceMelt
using CloudMicrophysics.Parameters: Blk1MVelType, Blk1MVelTypeRain, Blk1MVelTypeSnow
using CloudMicrophysics.Parameters: AirProperties
# Two-moment parameters
using CloudMicrophysics.Parameters: SB2006, StokesRegimeVelType, SB2006VelType, Chen2022VelTypeRain
using CloudMicrophysics.Parameters: TerminalVelocityParams
# Aerosol activation parameters
using CloudMicrophysics.Parameters: AerosolActivationParameters
using CloudMicrophysics: AerosolModel as CMAM
# SpecialFunctions for error function
using SpecialFunctions: erf

using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    conv_q_icl_to_q_sno,
    accretion,
    accretion_rain_sink,
    accretion_snow_rain,
    terminal_velocity,
    get_n0,
    lambda_inverse

# Two-moment microphysics
using CloudMicrophysics: Microphysics2M as CM2
# Non-equilibrium cloud condensate terminal velocities
using CloudMicrophysics: MicrophysicsNonEq as CMNonEq

using Breeze.AtmosphereModels: AtmosphereModels,
    AbstractNumberConcentrationCategories,
    is_moisture_mass_tracer,
    total_density,
    moisture_phase,
    prognostic_field_names,
    sedimentation_velocity,
    specific_field_name,
    transport_velocities

using Breeze.Thermodynamics:
    MoistureMassFractions,
    with_moisture,
    temperature,
    PlanarLiquidSurface,
    PlanarIceSurface,
    saturation_vapor_pressure,
    saturation_specific_humidity,
    supersaturation,
    liquid_latent_heat,
    ice_latent_heat,
    vapor_gas_constant,
    mixture_gas_constant,
    mixture_heat_capacity

using Breeze: Microphysics

using Breeze.Microphysics:
    center_field_tuple,
    BulkMicrophysics,
    WarmPhaseSaturationAdjustment,
    MixedPhaseSaturationAdjustment,
    AbstractCondensateFormation,
    ConstantRateCondensateFormation,
    NonEquilibriumCloudFormation,
    NumberConcentrationKernelFunction,
    condensation_rate,
    deposition_rate,
    adjust_thermodynamic_state

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Center, Face, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: ZFaceField
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, BoundaryCondition, NormalFlow
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using Adapt: Adapt, adapt

using Breeze.Advection: SurfacePrecipitationFluxKernel

include("cloud_microphysics_translations.jl")
include("one_moment_microphysics.jl")
include("one_moment_helpers.jl")
include("two_moment_microphysics.jl")
include("two_moment_helpers.jl")

end # module BreezeCloudMicrophysicsExt
