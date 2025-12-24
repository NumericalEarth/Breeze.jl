module Microphysics

export
    compute_temperature,
    adjust_thermodynamic_state,
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium,
    NonEquilibriumCloudFormation,
    BulkMicrophysics,
    FourCategories,
    SaturationSpecificHumidity,
    SaturationSpecificHumidityField,
    DCMIP2016KesslerMicrophysics,
    RelativeHumidity,
    RelativeHumidityField

using ..AtmosphereModels: AtmosphereModels

import ..AtmosphereModels:
    compute_moisture_fractions,
    materialize_microphysical_fields,
    update_microphysical_fields!

include("saturation_adjustment.jl")
include("bulk_microphysics.jl")
include("microphysics_diagnostics.jl")
include("dcmip2016_kessler.jl")

end # module Microphysics
