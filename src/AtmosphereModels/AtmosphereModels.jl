module AtmosphereModels

export
    # AtmosphereModel core
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    AnelasticFormulation,
    StaticEnergyThermodynamics,
    LiquidIcePotentialTemperatureThermodynamics,
    static_energy_density,
    static_energy,
    total_energy,
    liquid_ice_potential_temperature_density,
    liquid_ice_potential_temperature,

    # Diagnostics (re-exported from Diagnostics submodule)
    PotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt

include("atmosphere_model.jl")
include("set_atmosphere_model.jl")
include("anelastic_formulation.jl")
include("static_energy_thermodynamics.jl")
include("potential_temperature_thermodynamics.jl")
include("atmosphere_model_buoyancy.jl")
include("microphysics_interface.jl")
include("radiation_interface.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("compute_hydrostatic_pressure.jl")

# Include Diagnostics submodule after AtmosphereModel is defined
include("Diagnostics/Diagnostics.jl")
using .Diagnostics

end
