module AtmosphereModels

export
    # AtmosphereModel core
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    # Dynamics
    AnelasticDynamics,
    AnelasticModel,
    dynamics_density,
    dynamics_pressure,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
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
    specific_humidity,

    # Interface functions (extended by BoundaryConditions and Forcings)
    regularize_atmosphere_model_boundary_conditions,
    materialize_atmosphere_model_forcing,
    compute_forcing!,

    # Radiation (implemented by extensions)
    RadiativeTransferModel,
    BackgroundAtmosphere,

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

#####
##### Dynamics submodule
#####

include("Dynamics/Dynamics.jl")
using .Dynamics:
    AnelasticDynamics,
    default_dynamics,
    materialize_dynamics,
    materialize_momentum_and_velocities,
    dynamics_pressure_solver,
    dynamics_density,
    dynamics_pressure,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    solve_for_anelastic_pressure!

#####
##### Thermodynamic formulations submodule
#####

include("ThermodynamicFormulations/ThermodynamicFormulations.jl")
using .ThermodynamicFormulations:
    StaticEnergyFormulation,
    LiquidIcePotentialTemperatureFormulation,
    materialize_formulation,
    prognostic_thermodynamic_field_names,
    additional_thermodynamic_field_names,
    thermodynamic_density_name,
    thermodynamic_density,
    collect_prognostic_fields,
    diagnose_thermodynamic_state

# Import with `import` (not `using`) to allow extension
import .ThermodynamicFormulations: compute_auxiliary_thermodynamic_variables!, compute_thermodynamic_tendency!

#####
##### AtmosphereModel core
#####

include("atmosphere_model.jl")
include("set_atmosphere_model.jl")

# Define AnelasticModel type alias after AtmosphereModel is defined
const AnelasticModel = AtmosphereModel{<:AnelasticDynamics}

# Include anelastic time stepping after AnelasticModel is defined
include("anelastic_time_stepping.jl")

#####
##### Remaining AtmosphereModel components
#####

include("atmosphere_model_buoyancy.jl")
include("radiation_interface.jl")
include("dynamics_kernel_functions.jl")
include("update_atmosphere_model_state.jl")
include("compute_hydrostatic_pressure.jl")

#####
##### Thermodynamics tendencies and set! implementations
#####

include("static_energy_tendency.jl")
include("potential_temperature_tendency.jl")

#####
##### Diagnostics submodule
#####

include("Diagnostics/Diagnostics.jl")
using .Diagnostics

end
