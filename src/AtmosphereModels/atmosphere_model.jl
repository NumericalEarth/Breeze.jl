using ..Thermodynamics: Thermodynamics, ThermodynamicConstants

using Oceananigans: Oceananigans, AbstractModel, Center, CenterField, Clock, Field,
                    Centered, fields, prognostic_fields
using Oceananigans.Advection: Advection, adapt_advection_order, cell_advection_timescale
using Oceananigans.AbstractOperations: @at
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Diagnostics: Diagnostics as OceananigansDiagnostics, NaNChecker
using Oceananigans.Models: Models, validate_model_halo, validate_tracer_advection
using Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_momentum_advection
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, time_discretization, build_closure_fields
using Oceananigans.Utils: launch!, prettytime, prettykeys, with_tracers

struct DefaultValue end

tupleit(t::Tuple) = t
tupleit(t) = tuple(t)

validate_tracers(tracers) = throw(ArgumentError("tracers for AtmosphereModel must be a tuple of symbols"))

function validate_tracers(tracers::Tuple)
    for name in tracers
        name isa Symbol || throw(ArgumentError("The names of tracers for AtmosphereModel must be symbols, got $name"))
    end
    return tracers
end

mutable struct AtmosphereModel{Dyn, Frm, Arc, Tst, Grd, Clk, Thm, Mom, Moi, Mfr, Buy,
                               Tmp, Sol, Vel, Trc, Adv, Cor, Frc, Mic, Cnd, Cls, Cfs, Rad} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    dynamics :: Dyn
    formulation :: Frm
    thermodynamic_constants :: Thm
    momentum :: Mom
    moisture_density :: Moi
    specific_moisture :: Mfr
    temperature :: Tmp
    pressure_solver :: Sol
    velocities :: Vel
    tracers :: Trc
    buoyancy :: Buy
    advection :: Adv
    coriolis :: Cor
    forcing :: Frc
    microphysics :: Mic
    microphysical_fields :: Cnd
    timestepper :: Tst
    closure :: Cls
    closure_fields :: Cfs
    radiation :: Rad
end

"""
$(TYPEDSIGNATURES)

Return an `AtmosphereModel` that uses the anelastic approximation following
[Pauluis (2008)](@cite Pauluis2008).

Arguments
=========

   * The default `dynamics` is [`AnelasticDynamics`](@ref Breeze.AnelasticEquations.AnelasticDynamics).

   * The default `formulation` is `:LiquidIcePotentialTemperature`.

   * The default `advection` scheme is `Centered(order=2)` for both momentum
     and scalars. If a single `advection` is provided, it is used for both momentum
     and scalars.

   * Alternatively, specific `momentum_advection` and `scalar_advection`
     schemes may be provided. `scalar_advection` may be a `NamedTuple` with
     a different scheme for each respective scalar, identified by name.

Example
=======

```jldoctest
julia> using Breeze

julia> grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 2, 3));

julia> model = AtmosphereModel(grid)
AtmosphereModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── dynamics: AnelasticDynamics(p₀=101325.0, θ₀=288.0)
├── formulation: LiquidIcePotentialTemperatureFormulation
├── timestepper: SSPRungeKutta3
├── advection scheme:
│   ├── momentum: Centered(order=2)
│   ├── ρθ: Centered(order=2)
│   └── ρqᵗ: Centered(order=2)
├── tracers: ()
├── coriolis: Nothing
└── microphysics: Nothing
```

References
==========
Pauluis, O. (2008). Thermodynamic consistency of the anelastic approximation for a moist atmosphere.
  Journal of the Atmospheric Sciences 65, 2719–2729.
"""
function AtmosphereModel(grid;
                         clock = Clock(grid),
                         thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                         formulation = :LiquidIcePotentialTemperature,
                         dynamics = nothing,
                         velocities = nothing,
                         moisture_density = DefaultValue(),
                         tracers = tuple(),
                         coriolis = nothing,
                         boundary_conditions = NamedTuple(),
                         forcing = NamedTuple(),
                         advection = DefaultValue(),
                         momentum_advection = DefaultValue(),
                         scalar_advection = DefaultValue(),
                         closure = nothing,
                         microphysics = nothing,
                         timestepper = nothing,
                         timestepper_kwargs = NamedTuple(),
                         radiation = nothing)

    # Use default dynamics if not specified
    isnothing(dynamics) && (dynamics = default_dynamics(grid, thermodynamic_constants))

    # Use default timestepper for the dynamics if not specified
    isnothing(timestepper) && (timestepper = default_timestepper(dynamics))

    # Validate that velocity boundary conditions are only provided for dynamics that support them
    validate_velocity_boundary_conditions(dynamics, boundary_conditions)

    if !(advection isa DefaultValue)
        # TODO: check that tracer+momentum advection were not independently set.
        scalar_advection = momentum_advection = advection
    else
        (momentum_advection isa DefaultValue) && (momentum_advection = Centered(order=2))
        (scalar_advection isa DefaultValue) && (scalar_advection = Centered(order=2))
    end

    # Check halos and throw an error if the grid's halo is too small
    validate_model_halo(grid, momentum_advection, scalar_advection, closure)

    # Reduce the advection order in directions that do not have enough grid points
    momentum_advection = validate_momentum_advection(momentum_advection, grid)
    default_scalar_advection, scalar_advection = validate_tracer_advection(scalar_advection, grid)

    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    tracer_names = validate_tracers(tracers)

    # Get field names from dynamics and formulation
    prognostic_names = prognostic_field_names(dynamics, formulation, microphysics, tracers)
    velocity_bc_names = velocity_boundary_condition_names(dynamics)
    default_bc_names = tuple(prognostic_names..., velocity_bc_names...)
    default_boundary_conditions = NamedTuple{default_bc_names}(FieldBoundaryConditions() for _ in default_bc_names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)

    # Pre-regularize AtmosphereModel boundary conditions (fill in reference_density, compute saturation humidity, etc.)
    # Also converts ρe boundary conditions to ρθ for potential temperature formulations
    p₀ = surface_pressure(dynamics)
    boundary_conditions = regularize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                                          dynamics, microphysics, p₀, thermodynamic_constants)

    all_names = field_names(dynamics, formulation, microphysics, tracers)
    regularized_boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, all_names)

    # Materialize dynamics and formulation
    dynamics = materialize_dynamics(dynamics, grid, regularized_boundary_conditions, thermodynamic_constants, microphysics)
    formulation = materialize_formulation(formulation, dynamics, grid, regularized_boundary_conditions)

    # Materialize momentum and velocities
    # If velocities is provided (e.g., PrescribedVelocityFields), use it
    if isnothing(velocities)
        momentum, velocities = materialize_momentum_and_velocities(dynamics, grid, regularized_boundary_conditions)
    else
        # Store velocity specification in dynamics for dispatch (e.g., PrescribedVelocityFields)
        dynamics = update_dynamics_with_velocities(dynamics, velocities)
        momentum, _ = materialize_momentum_and_velocities(dynamics, grid, regularized_boundary_conditions)
        velocities = materialize_velocities(velocities, grid)
    end
    microphysical_fields = materialize_microphysical_fields(microphysics, grid, regularized_boundary_conditions)

    tracers = NamedTuple(name => CenterField(grid, boundary_conditions=regularized_boundary_conditions[name]) for name in tracer_names)

    if moisture_density isa DefaultValue
        moisture_density = CenterField(grid, boundary_conditions=regularized_boundary_conditions.ρqᵗ)
    end

    # Diagnostic fields
    specific_moisture = CenterField(grid)
    temperature = CenterField(grid)

    prognostic_microphysical_fields = NamedTuple(name => microphysical_fields[name] for name in prognostic_field_names(microphysics))
    prognostic_model_fields = collect_prognostic_fields(formulation,
                                                        dynamics,
                                                        momentum,
                                                        moisture_density,
                                                        prognostic_microphysical_fields,
                                                        tracers)

    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper_kwargs = augment_timestepper_kwargs(dynamics, timestepper_kwargs)
    timestepper = TimeStepper(timestepper, grid, prognostic_model_fields; implicit_solver, timestepper_kwargs...)
    pressure_solver = dynamics_pressure_solver(dynamics, grid)

    model_fields = merge(prognostic_model_fields, velocities, (; T=temperature, qᵗ=specific_moisture))
    density = dynamics_density(dynamics)
    forcing = atmosphere_model_forcing(forcing, prognostic_model_fields, model_fields,
                                       grid, coriolis, density,
                                       velocities, dynamics, formulation, specific_moisture)

    # Include thermodynamic density (ρe or ρθ), ρqᵗ, microphysical prognostic fields, plus user tracers
    closure_thermo_name = thermodynamic_density_name(formulation)
    microphysical_names = prognostic_field_names(microphysics)
    scalar_names = tuple(closure_thermo_name, :ρqᵗ, microphysical_names..., tracer_names...)
    closure = Oceananigans.Utils.with_tracers(scalar_names, closure)
    closure_fields = build_closure_fields(nothing, grid, clock, scalar_names, regularized_boundary_conditions, closure)

    # Generate tracer advection scheme for each tracer
    # scalar_advection is always a NamedTuple after validate_tracer_advection (either user's partial NamedTuple or empty)
    # with_tracers fills in missing names using default_generator
    default_generator(names, initial_tuple) = default_scalar_advection
    scalar_advection_tuple = with_tracers(scalar_names, scalar_advection, default_generator, with_velocities=false)
    momentum_advection_tuple = (; momentum = momentum_advection)
    advection = merge(momentum_advection_tuple, scalar_advection_tuple)
    advection = NamedTuple(name => adapt_advection_order(scheme, grid) for (name, scheme) in pairs(advection))

    model = AtmosphereModel(arch,
                            grid,
                            clock,
                            dynamics,
                            formulation,
                            thermodynamic_constants,
                            momentum,
                            moisture_density,
                            specific_moisture,
                            temperature,
                            pressure_solver,
                            velocities,
                            tracers,
                            nothing, # buoyancy, temporary solution for compatibility with Oceananigans.TurbulenceClosures
                            advection,
                            coriolis,
                            forcing,
                            microphysics,
                            microphysical_fields,
                            timestepper,
                            closure,
                            closure_fields,
                            radiation)

    # Initialize thermodynamics (dynamics-specific)
    initialize_model_thermodynamics!(model)

    return model
end

function Base.summary(model::AtmosphereModel)
    A = nameof(typeof(model.grid.architecture))
    G = nameof(typeof(model.grid))
    return string("AtmosphereModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::AtmosphereModel)
    TS = nameof(typeof(model.timestepper))
    Mic = nameof(typeof(model.microphysics))
    tracernames = prettykeys(model.tracers)

    print(io, summary(model), "\n",
              "├── grid: ", summary(model.grid), "\n",
              "├── dynamics: ", summary(model.dynamics), "\n",
              "├── formulation: ", summary(model.formulation), "\n",
              "├── timestepper: ", TS, "\n")

    if model.advection !== nothing
        print(io, "├── advection scheme: ", "\n")
        names = keys(model.advection)
        for name in names[1:end-1]
            print(io, "│   ├── " * string(name) * ": " * summary(model.advection[name]), "\n")
        end
        name = names[end]
        print(io, "│   └── " * string(name) * ": " * summary(model.advection[name]), "\n")
    end

    print(io, "├── tracers: ", tracernames, "\n",
              "├── coriolis: ", summary(model.coriolis), "\n",
              "└── microphysics: ", Mic)
end

Advection.cell_advection_timescale(model::AtmosphereModel) = cell_advection_timescale(model.grid, model.velocities)

# Prognostic field names from dynamics + thermodynamic formulation + microphysics + tracers
function prognostic_field_names(dynamics, formulation, microphysics, tracer_names)
    momentum_names = prognostic_momentum_field_names(dynamics)
    formulation_names = prognostic_thermodynamic_field_names(formulation)
    microphysical_names = prognostic_field_names(microphysics)
    return tuple(momentum_names..., :ρqᵗ, formulation_names..., microphysical_names..., tracer_names...)
end

function field_names(dynamics, formulation, microphysics, tracer_names)
    prog_names = prognostic_field_names(dynamics, formulation, microphysics, tracer_names)
    default_additional_names = (:u, :v, :w, :T, :qᵗ)
    formulation_additional_names = additional_thermodynamic_field_names(formulation)
    return tuple(prog_names..., default_additional_names..., formulation_additional_names...)
end

function atmosphere_model_forcing(user_forcings, prognostic_fields, model_fields,
                                  grid, coriolis, density,
                                  velocities, dynamics, formulation, specific_moisture)
    forcings_type = typeof(user_forcings)
    msg = string("AtmosphereModel forcing must be a NamedTuple, got $forcings_type")
    throw(ArgumentError(msg))
    return nothing
end

function atmosphere_model_forcing(::Nothing, prognostic_fields, model_fields,
                                  grid, coriolis, density,
                                  velocities, dynamics, formulation, specific_moisture)
    names = keys(prognostic_fields)
    return NamedTuple{names}(Returns(zero(eltype(prognostic_fields[name]))) for name in names)
end

function atmosphere_model_forcing(user_forcings::NamedTuple, prognostic_fields, model_fields,
                                  grid, coriolis, density,
                                  velocities, dynamics, formulation, specific_moisture)

    user_forcing_names = keys(user_forcings)

    if :ρe ∈ keys(prognostic_fields)
        forcing_fields = prognostic_fields
    else
        forcing_fields = merge(prognostic_fields, (; ρe=prognostic_fields.ρθ))
    end

    forcing_names = keys(forcing_fields)

    for name in user_forcing_names
        if name ∉ forcing_names
            msg = string("Invalid forcing: forcing contains an entry for $name, but $name is not a prognostic field!", '\n',
                         "The forcing fields are ", forcing_names)
            throw(ArgumentError(msg))
        end
    end

    model_names = keys(model_fields)

    # Build specific fields for subsidence forcing (maps specific field names like :u, :θ to fields)
    formulation_fields = fields(formulation)
    specific_fields = merge(velocities, formulation_fields, (; qᵗ=specific_moisture))

    # Build context for special forcing types (used by extended materialize_forcing in Forcings module)
    forcing_context = (; coriolis, density, specific_fields)

    materialized = Tuple(
        n in keys(user_forcings) ?
            materialize_atmosphere_model_forcing(user_forcings[n], f, n, model_names, forcing_context) :
            Returns(zero(eltype(f)))
            for (n, f) in pairs(forcing_fields)
    )

    forcings = NamedTuple{forcing_names}(materialized)

    return forcings
end

function Oceananigans.fields(model::AtmosphereModel)
    formulation_fields = fields(model.formulation)
    auxiliary = (; T=model.temperature, qᵗ=model.specific_moisture)
    return merge(prognostic_fields(model), formulation_fields, model.velocities, auxiliary, model.microphysical_fields)
end

function Oceananigans.prognostic_fields(model::AtmosphereModel)
    dynamics_fields = dynamics_prognostic_fields(model.dynamics)
    prognostic_formulation_fields = prognostic_fields(model.formulation)
    thermodynamic_fields = merge(prognostic_formulation_fields, (; ρqᵗ=model.moisture_density))
    μ_names = prognostic_field_names(model.microphysics)
    μ_fields= NamedTuple{μ_names}(model.microphysical_fields[name] for name in μ_names)
    return merge(dynamics_fields, model.momentum, thermodynamic_fields, μ_fields, model.tracers)
end

Models.boundary_condition_args(model::AtmosphereModel) = (model.clock, fields(model))

function total_energy(model)
    u, v, w = model.velocities
    k = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2 |> Field
    e = static_energy(model) |> Field
    return k + e
end

# Check for NaNs in the first prognostic field
function OceananigansDiagnostics.default_nan_checker(model::AtmosphereModel)
    model_fields = prognostic_fields(model)

    if isempty(model_fields)
        return nothing
    end

    first_name = first(keys(model_fields))
    field_to_check_nans = NamedTuple{tuple(first_name)}(model_fields)
    nan_checker = NaNChecker(field_to_check_nans)
    return nan_checker
end

# For compatibility with Oceananigans JLD2Writer
Oceananigans.OutputWriters.default_included_properties(::AtmosphereModel) = [:grid, :thermodynamic_constants]
