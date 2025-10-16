module Microphysics

import CloudMicrophysics

const CM1 = CloudMicrophysics.Microphysics1M
const CM2 = CloudMicrophysics.Microphysics2M
const CMNe = CloudMicrophysics.MicrophysicsNonEq
const CMP = CloudMicrophysics.Parameters

export AbstractMicrophysics,
       DefaultMicrophysics,
       Microphysics1MParameters,
       Microphysics1MCache,
       Microphysics1M,
       update_microphysics_state!,
       microphysics_transition,
       microphysics_drift_velocity,
       microphysics_auxiliary_fields

include("default_microphysics.jl")
include("Microphysics1M.jl")

end
