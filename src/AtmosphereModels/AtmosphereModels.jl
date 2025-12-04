module AtmosphereModels

export
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    AnelasticFormulation,
    StaticEnergyThermodynamics,
    LiquidIcePotentialTemperatureThermodynamics,
    energy_density,
    specific_energy,
    potential_temperature_density,
    potential_temperature

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt


include("atmosphere_model.jl")
include("diagnostic_fields.jl")
include("anelastic_formulation.jl")
include("static_energy_thermodynamics.jl")
include("potential_temperature_thermodynamics.jl")
include("atmosphere_model_buoyancy.jl")
include("microphysics_interface.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("set_atmosphere_model.jl")
include("compute_hydrostatic_pressure.jl")

end
