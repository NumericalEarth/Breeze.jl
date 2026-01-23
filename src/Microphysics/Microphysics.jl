module Microphysics

export
    # Microphysics schemes and utilities
    compute_temperature,
    adjust_thermodynamic_state,
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium,
    AbstractCondensateFormation,
    ConstantRateCondensateFormation,
    NonEquilibriumCloudFormation,
    BulkMicrophysics,
    FourCategories,
    SaturationSpecificHumidity,
    SaturationSpecificHumidityField,
    DCMIP2016KesslerMicrophysics,
    kessler_terminal_velocity,
    saturation_adjustment_coefficient,
    RelativeHumidity,
    RelativeHumidityField

using ..AtmosphereModels: AtmosphereModels, moisture_fractions, grid_moisture_fractions,
    materialize_microphysical_fields, update_microphysical_fields!,
    NothingMicrophysicalState, WarmRainState

include("saturation_adjustment.jl")
include("bulk_microphysics.jl")
include("microphysics_diagnostics.jl")
include("dcmip2016_kessler.jl")

end # module Microphysics
