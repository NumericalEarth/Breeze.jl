module AtmosphereModels

export
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    AnelasticFormulation,
    StaticEnergyThermodynamics,
    PotentialTemperature,
    PotentialTemperatureField,
    energy_density,
    specific_energy,
    compute_hydrostatic_pressure!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt


include("atmosphere_model.jl")
include("diagnostic_fields.jl")
include("anelastic_formulation.jl")
include("atmosphere_model_buoyancy.jl")
include("microphysics_interface.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")
include("compute_hydrostatic_pressure.jl")

end
