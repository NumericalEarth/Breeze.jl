module AtmosphereModels

export
    # AtmosphereModel core
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    # Dynamics
    AnelasticDynamics,
    AnelasticModel,
    CompressibleDynamics,
    CompressibleModel,
    dynamics_density,
    dynamics_pressure,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    buoyancy_forceᶜᶜᶜ,
    # Thermodynamic formulations
    StaticEnergyFormulation,
    LiquidIcePotentialTemperatureFormulation,
    thermodynamic_density_name,
    thermodynamic_density,
    # Helpers
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
##### Interfaces
#####

include("forcing_interface.jl")
include("microphysics_interface.jl")
include("dynamics_interface.jl")
include("formulation_interface.jl")

#####
##### Dynamics submodules
#####

include("AnelasticDynamicses/AnelasticDynamicses.jl")
using .AnelasticDynamicses:
    AnelasticDynamics,
    solve_for_anelastic_pressure!

include("CompressibleDynamicses/CompressibleDynamicses.jl")
using .CompressibleDynamicses:
    CompressibleDynamics

#####
##### AtmosphereModel core (needed before formulation submodules for type aliases)
#####

include("atmosphere_model.jl")
include("set_atmosphere_model.jl")

# Define model type aliases after AtmosphereModel is defined
const AnelasticModel = AtmosphereModel{<:AnelasticDynamics}
const CompressibleModel = AtmosphereModel{<:CompressibleDynamics}

#####
##### Remaining AtmosphereModel components
#####

include("atmosphere_model_buoyancy.jl")
include("radiation_interface.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("compute_hydrostatic_pressure.jl")

#####
##### Diagnostics submodule (needed before formulation submodules for helper accessors)
#####

include("Diagnostics/Diagnostics.jl")
using .Diagnostics

#####
##### Thermodynamic formulation submodules
#####

include("StaticEnergyFormulations/StaticEnergyFormulations.jl")
using .StaticEnergyFormulations:
    StaticEnergyFormulation

include("PotentialTemperatureFormulations/PotentialTemperatureFormulations.jl")
using .PotentialTemperatureFormulations:
    LiquidIcePotentialTemperatureFormulation

# Note: Type aliases StaticEnergyModel and PotentialTemperatureModel are defined
# in their respective formulation submodules and used internally for dispatch.
# They are not exported from AtmosphereModels.

#####
##### Include dynamics-specific time stepping after formulations
#####

include("anelastic_time_stepping.jl")
include("compressible_time_stepping.jl")

end
