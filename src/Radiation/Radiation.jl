module Radiation

export
    AbstractRadiationModel,
    RRTMGPModel,
    initialize_rrtmgp_model,
    compute_vertical_fluxes!,
    flux_results

include("rrtmgp_backend.jl")
include("radiation_model.jl")

end
