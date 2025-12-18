module Microphysics

export
    compute_temperature,
    adjust_thermodynamic_state,
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium,
    BulkMicrophysics,
    FourCategories,
    SaturationSpecificHumidity,
    SaturationSpecificHumidityField,
    RelativeHumidity,
    RelativeHumidityField,
    KesslerMicrophysics

using ..AtmosphereModels: AtmosphereModels, compute_moisture_fractions,
    materialize_microphysical_fields, update_microphysical_fields!

include("saturation_adjustment.jl")
include("bulk_microphysics.jl")
include("kessler_microphysics.jl")
include("microphysics_diagnostics.jl")

end # module Microphysics
