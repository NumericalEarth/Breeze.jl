module AtmosphereModels

export
    # AtmosphereModel core
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    AnelasticFormulation,
    StaticEnergyThermodynamics,
    LiquidIcePotentialTemperatureThermodynamics,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    formulation_density,
    static_energy_density,
    static_energy,
    total_energy,
    liquid_ice_potential_temperature_density,
    liquid_ice_potential_temperature,
    precipitation_rate,
    surface_precipitation_flux,

    # Interface functions (extended by BoundaryConditions and Forcings)
    regularize_atmosphere_model_boundary_conditions,
    materialize_atmosphere_model_forcing,
    compute_forcing!,

    # Radiation (implemented by extensions)
    RadiativeTransferModel,

    # Diagnostics (re-exported from Diagnostics submodule)
    PotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy,
    compute_hydrostatic_pressure!

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, fields
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Solvers: Solvers
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

#####
##### Formulation interface and AtmosphereModel
#####

include("formulation_interface.jl")
include("forcing_interface.jl")
include("microphysics_interface.jl")
include("atmosphere_model.jl")
include("set_atmosphere_model.jl")

#####
##### AnelasticFormulation submodule
#####

include("AnelasticFormulations/AnelasticFormulations.jl")
using .AnelasticFormulations:
    AnelasticFormulation,
    AnelasticModel

#####
##### Thermodynamics implementations
#####

include("static_energy_thermodynamics.jl")
include("potential_temperature_thermodynamics.jl")

#####
##### Remaining AtmosphereModel components
#####

include("atmosphere_model_buoyancy.jl")
include("radiation_interface.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("compute_hydrostatic_pressure.jl")

#####
##### Diagnostics submodule
#####

include("Diagnostics/Diagnostics.jl")
using .Diagnostics

end
