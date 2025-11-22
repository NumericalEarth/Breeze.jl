module AtmosphereModels

export
    AtmosphereModel,
    AnelasticFormulation,
    PotentialTemperatureOperation,
    PotentialTemperature,
    PotentialTemperatureField

using DocStringExtensions: TYPEDSIGNATURES

include("atmosphere_model.jl")
include("diagnostic_fields.jl")
include("anelastic_formulation.jl")
include("microphysics_interface.jl")
# include("update_hydrostatic_pressure.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")

end
