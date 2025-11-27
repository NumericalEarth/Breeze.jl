module AtmosphereModels

export
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    AnelasticFormulation,
    PotentialTemperature,
    PotentialTemperatureField

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt


include("atmosphere_model.jl")
include("diagnostic_fields.jl")
include("anelastic_formulation.jl")
include("atmosphere_model_buoyancy.jl")
include("microphysics_interface.jl")
# include("update_hydrostatic_pressure.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")

end
