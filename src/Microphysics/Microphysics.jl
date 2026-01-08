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
    RelativeHumidityField

using ..AtmosphereModels: AtmosphereModels, compute_moisture_fractions,
    materialize_microphysical_fields, update_microphysical_fields!

include("saturation_adjustment.jl")
include("bulk_microphysics.jl")
include("microphysics_diagnostics.jl")

#####
##### Predicted Particle Properties (P3) submodule
#####

include("PredictedParticleProperties/PredictedParticleProperties.jl")
using .PredictedParticleProperties

# Re-export key P3 types
export PredictedParticlePropertiesMicrophysics, P3Microphysics

end # module Microphysics
