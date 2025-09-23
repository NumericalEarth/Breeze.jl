module AtmosphereModels

export AtmosphereModel, prognostic_fields   

include("atmosphere_model.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")

end
