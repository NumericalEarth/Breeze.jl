module Microphysics

export
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
    RelativeHumidity,
    RelativeHumidityField,
    KesslerMicrophysics

import ..AtmosphereModels: compute_moisture_fractions,
    materialize_microphysical_fields, update_microphysical_fields!,
    maybe_adjust_thermodynamic_state

using ..AtmosphereModels: AtmosphereModels

include("saturation_adjustment.jl")
include("bulk_microphysics.jl")
include("kessler_microphysics.jl")
include("microphysics_diagnostics.jl")

end # module Microphysics
