module AtmosphereModels

export AtmosphereModel

include("atmosphere_model.jl")
include("anelastic_formulation.jl")
include("saturation_adjustment.jl")
# include("anelastic_pressure_correction.jl")
include("update_hydrostatic_pressure.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")

end
