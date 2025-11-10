module AtmosphereModels

export AtmosphereModel, AnelasticFormulation

using DocStringExtensions: TYPEDSIGNATURES

include("atmosphere_model.jl")
include("anelastic_formulation.jl")
include("saturation_adjustment.jl")
# include("update_hydrostatic_pressure.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")

end
