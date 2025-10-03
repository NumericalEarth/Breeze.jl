module Radiation

export
    AbstractRadiationModel,
    GrayRadiationModel,
    update_radative_fluxes!

include("rrtmgp_interface.jl")
include("radiation_model.jl")
include("radiation_model_utils.jl")
include("radiation_model_gray.jl")

end
