module AtmosphereModels

export
    # AtmosphereModel core
    AtmosphereModel,
    AtmosphereModelBuoyancy,
    # Adiabatic (FV3 na_init) initialization
    balance_adiabatically!,
    AdiabaticBalancer,
    HydrostaticallyBalancedDensity,
    # Dynamics interface (dynamics types exported by their respective modules)
    dynamics_density,
    dynamics_pressure,
    total_density,
    sedimentation_composition_increment,
    pressure_anomaly,
    total_pressure,
    buoyancy_forceᶜᶜᶜ,
    SlowDynamics,
    compute_pressure_correction!,
    make_pressure_correction!,
    # Thermodynamic formulation interface (formulation types exported by their respective modules)
    specific_thermodynamic_field,
    thermodynamic_density_name,
    thermodynamic_density,
    total_energy_density_name,
    DefaultTemperatureSolver,
    default_temperature_solver,
    # Helpers
    static_energy_density,
    static_energy,
    total_energy,
    liquid_ice_potential_temperature_density,
    liquid_ice_potential_temperature,
    precipitation_rate,
    bottom_precipitation_flux,
    specific_humidity,
    moisture_prognostic_name,
    total_moisture_density_name,
    moisture_specific_name,
    specific_prognostic_moisture,

    # Negative moisture correction types
    AbstractNegativeMoistureCorrection,
    VerticalBorrowing,
    SpeciesBorrowing,
    AbstractNumberConcentrationCategories,

    # Microphysics interface
    AbstractMicrophysicalState,
    NothingMicrophysicalState,
    WarmRainState,
    microphysical_state,
    microphysical_tendency,
    microphysical_tendencies,
    compute_microphysical_tendencies!,
    moisture_fractions,
    grid_moisture_fractions,
    update_microphysical_fields!,
    update_microphysical_auxiliaries!,
    aerosol_field_names,
    initial_aerosol_number,
    initial_aerosol_number_density,
    sedimentation_velocity,
    condensate_liquid_fraction,
    sedimentation_velocity_field,
    write_sedimentation_velocity!,
    microphysical_velocities,

    # Interface functions (extended by BoundaryConditions and Forcings)
    materialize_atmosphere_model_boundary_conditions,
    materialize_atmosphere_model_forcing,
    compute_forcing!,
    is_density_tendency_forcing,

    # Radiation (implemented by extensions)
    RadiativeTransferModel,
    BackgroundAtmosphere,
    standard_ozone_profile,
    materialize_background_atmosphere,
    materialize_surface_property,
    resolve_column_batch_rows,
    GrayOptics,
    ClearSkyOptics,
    AllSkyOptics,
    EcCKDOptics,
    CloudScatteringTables,
    OverlyingAtmosphereColumn,
    standard_atmosphere_temperature,

    # Solar position specifications
    AbstractSolarPosition,
    ApparentSolarPosition,
    DiurnalSolarPosition,
    FixedCosineZenith,

    # Cloud effective radius
    ConstantRadiusParticles,

    # Microphysics interface
    prognostic_field_names,

    # Diagnostics (re-exported from Diagnostics submodule)
    PotentialTemperature,
    VirtualPotentialTemperature,
    EquivalentPotentialTemperature,
    StabilityEquivalentPotentialTemperature,
    LiquidIcePotentialTemperature,
    StaticEnergy,
    azimuthal_mean,
    azimuthal_mean!,
    compute_hydrostatic_pressure!,
    set_to_mean!,

    # Transport interface (for terrain-following coordinates)
    transport_velocities,
    advecting_momentum,

    # Advective timescale for the time-step wizard
    CellAdvectionTimescale,

    # Momentum tendency kernels (used by TimeSteppers for acoustic substepping)
    compute_x_momentum_tendency!,
    compute_y_momentum_tendency!,
    compute_z_momentum_tendency!,

    # Architecture-dispatched helper for safely converting Δt to a kernel arg
    kernel_time_step

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index, @localmem, @uniform, get_backend

using Oceananigans: Oceananigans, Center, Face, CenterField, fields
using Oceananigans.BoundaryConditions: BoundaryConditions, FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Operators: Δzᶜᶜᶜ, ∂zᶜᶜᶜ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
using Oceananigans.Solvers: Solvers
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Utils: prettysummary, launch!

# Convert Δt to the kernel-compatible time type for `grid`'s architecture.
# Metal cannot load Float64 kernel arguments, so we must convert at the launch
# site for Float32 grids. Reactant tracing breaks if we convert outside the
# kernel — BreezeReactantExt overrides this for `ReactantState` archs to pass
# Δt through unchanged.
@inline kernel_time_step(arch, grid, Δt) = convert(eltype(grid), Δt)

#####
##### Halo-fill / launch-count switches (T16 fills unit). Read from ENV when the module is loaded
##### (`__init__`), not at precompile; set the `Ref`s directly to A/B in one process.
#####

"`BREEZE_FILLS_TOGETHER` (default true): consecutive independent fills go through one tupled `fill_halo_regions!`."
const FILLS_TOGETHER = Ref(true)
# Handoff C6 (e97db3f8): the scalar RK3 update of the multi-right-hand-side group formed inside its solve.
const FOLDED_RK3_SOLVES = Ref(true)
"`BREEZE_BATCHED_SCALAR_TENDENCIES` (default true): the moisture and every tracer tendency in one launch (T22 scalars)."
const BATCHED_SCALAR_TENDENCIES = Ref(true)
"`BREEZE_SKIP_DEAD_FILLS` (default true): leave out halo fills that a later fill in the same pass overwrites unread."
const SKIP_DEAD_FILLS = Ref(true)
"`BREEZE_BATCHED_Z_BCS` (default true): the z flux-BC tendencies of every prognostic in one launch."
const BATCHED_Z_BCS = Ref(true)
"`BREEZE_CELLWISE_MOISTURE_CORRECTION` (default true): species borrowing and number repairs over every cell (PR #1092); only vertical borrowing walks a column."
const CELLWISE_MOISTURE_CORRECTION = Ref(true)

env_switch(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

function __init__()
    FILLS_TOGETHER[] = env_switch("BREEZE_FILLS_TOGETHER", true)
    SKIP_DEAD_FILLS[] = env_switch("BREEZE_SKIP_DEAD_FILLS", true)
    BATCHED_Z_BCS[] = env_switch("BREEZE_BATCHED_Z_BCS", true)
    COLUMN_SEDIMENTATION[] = env_switch("BREEZE_COLUMN_SEDIMENTATION", false)  # density_weighted_implicit_diffusion.jl
    CELLWISE_MOISTURE_CORRECTION[] = env_switch("BREEZE_CELLWISE_MOISTURE_CORRECTION", true)
    read_kernel_argument_cache_switches!()  # kernel_arguments.jl
    FOLDED_RK3_SOLVES[] = env_switch("BREEZE_FOLDED_RK3_SOLVES", true)
    BATCHED_SCALAR_TENDENCIES[] = env_switch("BREEZE_BATCHED_SCALAR_TENDENCIES", true)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Fill the halos of independent `fields` (no field's boundary condition reads another field of the
call) with one tupled `fill_halo_regions!`, so Oceananigans fills fields that share location, kernels
and argument-free conditions with one launch per side. With `FILLS_TOGETHER[] = false`, field by field.
"""
function fill_halo_regions_together!(fields::Tuple, args...)
    if FILLS_TOGETHER[]
        fill_halo_regions!(fields, args...)
    else
        foreach(f -> fill_halo_regions!(f, args...), fields)
    end
    return nothing
end

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
include("kernel_arguments.jl")

#####
##### Remaining AtmosphereModel components
#####

include("atmosphere_model_buoyancy.jl")
include("solar_position.jl")
include("radiation_interface.jl")
include("dynamics_kernel_functions.jl")
include("implicit_vertical_advection.jl")
include("density_weighted_implicit_diffusion.jl")
include("carried_implicit_rows.jl")
include("cell_advection_timescale.jl")
include("negative_moisture_correction.jl")
include("update_atmosphere_model_state.jl")
include("checkpointing.jl")
include("compute_hydrostatic_pressure.jl")

#####
##### Diagnostics submodule (needed before formulation submodules for helper accessors)
#####

include("Diagnostics/Diagnostics.jl")
using .Diagnostics

# set_atmosphere_model requires Diagnostics for SaturationSpecificHumidity
include("set_moisture.jl")
include("set_atmosphere_model.jl")
include("set_to_mean.jl")

# Adiabatic (FV3 na_init) initialization. The dynamics-specific twin construction
# (`adiabatic_twin_dynamics`) is extended in CompressibleEquations; the generic fallback here keeps
# the balance solver-agnostic.
include("adiabatic_balance.jl")

end
