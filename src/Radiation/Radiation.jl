module Radiation

export
    AbstractRadiationModel,
    GrayRadiationModel

include("rrtmgp_interface.jl")
include("radiation_model.jl")
include("radiation_model_gray.jl")

end
