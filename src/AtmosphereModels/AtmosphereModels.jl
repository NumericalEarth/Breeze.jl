module AtmosphereModels

export
    # AtmosphereModel core
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    # Dynamics interface (dynamics types exported by their respective modules)
    dynamics_density,
    dynamics_pressure,
    mean_pressure,
    pressure_anomaly,
    total_pressure,
    buoyancy_forceᶜᶜᶜ,
    # Thermodynamic formulation interface (formulation types exported by their respective modules)
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

    # Microphysics interface
    AbstractMicrophysicalState,
    NothingMicrophysicalState,
    WarmRainState,
    microphysical_state,
    microphysical_tendency,
    grid_microphysical_tendency,
    update_microphysical_fields!,
    update_microphysical_auxiliaries!,

    # Interface functions (extended by BoundaryConditions and Forcings)
    regularize_atmosphere_model_boundary_conditions,
    materialize_atmosphere_model_forcing,
    compute_forcing!,

    # Radiation (implemented by extensions)
    RadiativeTransferModel,
    BackgroundAtmosphere,
    GrayOptics,
    ClearSkyOptics,
    AllSkyOptics,

    # Cloud effective radius
    ConstantRadiusParticles,

    # Diagnostics (re-exported from Diagnostics submodule)
    PotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy,
    compute_hydrostatic_pressure!,
    set_to_mean!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, fields
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Fields: Field
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Solvers: Solvers
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

#####
##### Interfaces (define the contract that dynamics implementations must fulfill)
#####

include("forcing_interface.jl")
include("microphysics_interface.jl")
include("dynamics_interface.jl")
include("formulation_interface.jl")

#####
##### AtmosphereModel core
#####

include("atmosphere_model.jl")

#####
##### Remaining AtmosphereModel components
#####

include("atmosphere_model_buoyancy.jl")
include("radiation_interface.jl")
include("dynamics_kernel_functions.jl")

# Column fields (Field{Nothing, Nothing, <:Any}) have singleton x/y dimensions
# and cannot participate in MPI halo communication on distributed grids.
_is_column_field(::Field{Nothing, Nothing}) = true
_is_column_field(::Any) = false

include("update_atmosphere_model_state.jl")
include("compute_hydrostatic_pressure.jl")

#####
##### Diagnostics submodule (needed before formulation submodules for helper accessors)
#####

include("Diagnostics/Diagnostics.jl")
using .Diagnostics

# set_atmosphere_model requires Diagnostics for SaturationSpecificHumidity
include("set_atmosphere_model.jl")
include("set_to_mean.jl")

end
