module Microphysics

export
    compute_temperature,
    adjust_thermodynamic_state,
    SaturationAdjustment,
    MixedPhaseEquilibrium,
    WarmPhaseEquilibrium,
    BulkMicrophysics,
    FourCategories,
    KesslerMicrophysics

import ..AtmosphereModels:
    maybe_adjust_thermodynamic_state,
    update_microphysical_fields!,
    prognostic_field_names,
    materialize_microphysical_fields,
    microphysical_velocities,
    compute_moisture_fractions,
    microphysical_tendency

include("saturation_adjustment.jl")
include("bulk_microphysics.jl")
include("dcmip2016_kessler.jl")

end # module Microphysics
