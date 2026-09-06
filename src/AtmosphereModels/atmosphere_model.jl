using ..Thermodynamics: Thermodynamics, ThermodynamicConstants

using Oceananigans: Oceananigans, AbstractModel, Center, CenterField, Clock, Field,
                    Centered, fields, prognostic_fields
using Oceananigans.Advection: Advection, adapt_advection_order, cell_advection_timescale, materialize_advection
using Oceananigans.AbstractOperations: @at
using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions, needs_implicit_solver
using Oceananigans.Diagnostics: Diagnostics as OceananigansDiagnostics, NaNChecker
using Oceananigans.Models: Models, validate_model_halo, validate_tracer_advection
using Oceananigans.TimeSteppers: TimeSteppers, TimeStepper, AbstractLagrangianParticles, step_lagrangian_particles!
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, build_closure_fields,
                                       closure_required_tracers, VerticallyImplicitTimeDiscretization
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.Utils: launch!, prettytime, prettykeys, with_tracers

# AtmosphereModel-specific momentum-advection validation. The compressible core advects momentum in
# flux form (`div_𝐯u`) plus the curvilinear curvature term `U_dot_∇u_metric`, so any flux-form scheme
# is valid on every grid — including `OrthogonalSphericalShellGrid` (where the hydrostatic ocean model
# restricts to `VectorInvariant`). We define our own validator rather than importing Oceananigans' and
# accept the requested scheme as-is.
validate_momentum_advection(momentum_advection, grid) = momentum_advection

struct DefaultValue end

const ParticlesOrNothing = Union{Nothing, AbstractLagrangianParticles}

"""
$(TYPEDSIGNATURES)

Return `particles` unchanged. Extended for grids on which Lagrangian particle
tracking is not supported, so that the combination is rejected at construction
rather than producing wrong trajectories at run time (see
`TerrainFollowingDiscretization/lagrangian_particles.jl`).
"""
validate_particles(particles, grid) = particles

tupleit(t::Tuple) = t
tupleit(t) = tuple(t)

validate_tracers(tracers) = throw(ArgumentError("tracers for AtmosphereModel must be a tuple of symbols"))

function validate_tracers(tracers::Tuple)
    for name in tracers
        name isa Symbol || throw(ArgumentError("The names of tracers for AtmosphereModel must be symbols, got $name"))
    end
    return tracers
end

mutable struct AtmosphereModel{Dyn, Frm, Arc, Tst, Grd, Clk, Thm, Mom, Moi, Buy,
                               Tmp, Sol, Vel, Trc, Adv, Cor, Frc, Mic, Cnd, Cls, Cfs, Rad, Prt} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    dynamics :: Dyn
    formulation :: Frm
    thermodynamic_constants :: Thm
    momentum :: Mom
    moisture_density :: Moi
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
    particles :: Prt
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

   * `particles` are Lagrangian particles to be advected with the flow,
     constructed with `Oceananigans.LagrangianParticles`. Particles are advected
     with the Cartesian velocities `model.velocities` once per time step, over the
     full `Δt`. Default: `nothing`. See the "Lagrangian particles" section of the
     documentation for details, including the treatment on terrain-following grids.

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
├── thermodynamic_constants: ThermodynamicConstants{Float64}
├── timestepper: SSPRungeKutta3
├── advection scheme:
│   ├── momentum: Centered(order=2)
│   ├── ρθ: Centered(order=2)
│   └── ρqᵛ: Centered(order=2)
├── forcing: @NamedTuple{ρu::Returns{Float64}, ρv::Returns{Float64}, ρw::Returns{Float64}, ρθ::Returns{Float64}, ρqᵛ::Returns{Float64}, ρs::Returns{Float64}}
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
                         radiation = nothing,
                         particles::ParticlesOrNothing = nothing)

    # Use default dynamics if not specified
    isnothing(dynamics) && (dynamics = default_dynamics(grid, thermodynamic_constants))

    # Use default timestepper for the dynamics if not specified
    isnothing(timestepper) && (timestepper = default_timestepper(dynamics))

    # Validate that velocity boundary conditions are only provided for dynamics that support them
    validate_velocity_boundary_conditions(dynamics, boundary_conditions)

    # Validate that the microphysics scheme is compatible with the thermodynamic constants
    validate_microphysics(microphysics, thermodynamic_constants)

    if !(advection isa DefaultValue)
        # TODO: check that tracer+momentum advection were not independently set.
        scalar_advection = momentum_advection = advection
    else
        (momentum_advection isa DefaultValue) && (momentum_advection = Centered(order=2))
        (scalar_advection isa DefaultValue) && (scalar_advection = Centered(order=2))
    end

    # Check halos and throw an error if the grid's halo is too small
    validate_model_halo(grid, momentum_advection, scalar_advection, closure)

    momentum_advection = validate_momentum_advection(momentum_advection, grid)
    default_scalar_advection, scalar_advection = validate_tracer_advection(scalar_advection, grid)
    particles = validate_particles(particles, grid)

    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    user_tracer_names = validate_tracers(tracers)

    # Prognostic-TKE closures carry their own prognostic scalar (`:ρe`). Appending it here,
    # before `prognostic_field_names`, the boundary-condition defaults, the tracer-field allocation
    # and `scalar_names`, is what makes it a first-class tracer everywhere downstream.
    # The captured name must differ from the assigned one, or the closure boxes it.
    closure_tracer_names = filter(∉(user_tracer_names), closure_required_tracers(closure))
    tracer_names = tuple(user_tracer_names..., closure_tracer_names...)
    tracers = tracer_names

    # Get field names from dynamics and formulation
    prognostic_names = prognostic_field_names(dynamics, formulation, microphysics, tracers)
    allunique(prognostic_names) ||
        throw(ArgumentError("Prognostic field names must be unique, but got $prognostic_names. " *
                            "A closure-required tracer ($(closure_required_tracers(closure))) cannot " *
                            "share its name with another prognostic field."))
    velocity_bc_names = velocity_boundary_condition_names(dynamics)
    default_bc_names = tuple(prognostic_names..., velocity_bc_names...)
    default_boundary_conditions = NamedTuple{default_bc_names}(FieldBoundaryConditions() for _ in default_bc_names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)

    # Pre-create diagnostic fields needed for VirtualPotentialTemperature
    # (used in stability-dependent boundary conditions like PolynomialCoefficient)
    temperature = CenterField(grid)

    # Regularize boundary conditions for grid topology before creating microphysical fields
    all_names = field_names(dynamics, formulation, microphysics, tracers)
    field_boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, all_names)

    # Create temporary microphysical fields for BC materialization (using pre-regularized BCs)
    preliminary_microphysical_fields = materialize_microphysical_fields(microphysics, grid, field_boundary_conditions)

    # Materialize atmosphere-specific boundary conditions (fill in VPT diagnostic,
    # surface pressure, thermodynamic constants, convert ρs → ρθ for potential temperature formulations)
    p₀ = surface_pressure(dynamics)
    # Pass preliminary microphysical fields for BC materialization; the qᵛ field within
    # provides the specific_prognostic_moisture reference needed by VirtualPotentialTemperature.
    specific_moisture_field = haskey(preliminary_microphysical_fields, :qᵛ) ? preliminary_microphysical_fields.qᵛ : CenterField(grid)
    boundary_conditions = materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                                           dynamics, microphysics, p₀, thermodynamic_constants,
                                                                           preliminary_microphysical_fields, specific_moisture_field, temperature)

    # Re-regularize after materialization (materialization may modify boundary conditions)
    regularized_boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, all_names)

    # Materialize dynamics and formulation
    dynamics = materialize_dynamics(dynamics, grid, regularized_boundary_conditions, thermodynamic_constants, microphysics)
    formulation = materialize_formulation(formulation, dynamics, grid, regularized_boundary_conditions)

    # Adaptive implicit vertical advection is supported for all prognostics with SSPRungeKutta3
    # (per-substep solve) and with AcousticRungeKutta3 (moisture and tracers via the generic
    # implicit step; momentum and the thermodynamic variable via a per-stage solve after the
    # acoustic substep loop — see TimeSteppers/acoustic_substep_helpers.jl). On terrain-following
    # grids the split partitions the contravariant velocity (see `advecting_vertical_velocity`).
    advection_needs_solver = needs_implicit_solver(momentum_advection) ||
                             needs_implicit_solver(default_scalar_advection) ||
                             any(needs_implicit_solver, values(scalar_advection))

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

    # Microphysical fields, including a prognostic aerosol reservoir `ρnᵃ`, start at zero. `ρnᵃ`
    # holds a ρ-weighted count, so it is filled in by `set_default_aerosol_number!` at the end of
    # this constructor, once the dynamics has been materialized and has a density to weight by.
    microphysical_fields = materialize_microphysical_fields(microphysics, grid, regularized_boundary_conditions)

    tracers = NamedTuple(name => CenterField(grid, boundary_conditions=regularized_boundary_conditions[name]) for name in tracer_names)

    moisture_name = moisture_prognostic_name(microphysics)
    if moisture_density isa DefaultValue
        moisture_density = CenterField(grid, boundary_conditions=regularized_boundary_conditions[moisture_name])
    end

    prognostic_microphysical_fields = NamedTuple(name => microphysical_fields[name] for name in prognostic_field_names(microphysics))
    prognostic_model_fields = collect_prognostic_fields(formulation,
                                                        dynamics,
                                                        momentum,
                                                        moisture_density,
                                                        moisture_name,
                                                        prognostic_microphysical_fields,
                                                        tracers)

    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)

    # Build a vertical tridiagonal solver for adaptive implicit vertical advection even when the
    # closure is explicit. When both are present, the diffusion and advection diagonals are summed
    # into a single system (see mass_weighted_implicit_diffusion.jl for the z-Center prognostics
    # and implicit_vertical_advection.jl for `ρw`).
    if implicit_solver === nothing && advection_needs_solver
        implicit_solver = implicit_diffusion_solver(VerticallyImplicitTimeDiscretization(), grid)
    end

    # Only pass `dynamics` to time steppers that accept it (Breeze's acoustic and SSP steppers).
    # Oceananigans' built-in time steppers (RungeKutta3, QuasiAdamsBashforth2) do not.
    if timestepper_uses_dynamics(timestepper)
        timestepper = TimeStepper(timestepper, grid, prognostic_model_fields; dynamics, implicit_solver,
                                  cache_advecting_state = advection_needs_solver, timestepper_kwargs...)
    else
        timestepper = TimeStepper(timestepper, grid, prognostic_model_fields; implicit_solver, timestepper_kwargs...)
    end
    pressure_solver = dynamics_pressure_solver(dynamics, grid)

    moisture_specific = moisture_specific_name(microphysics)
    specific_prognostic_moisture = microphysical_fields[moisture_specific]
    # Build `model_fields` with the same key order as Oceananigans.fields(model::AtmosphereModel)
    # below. ContinuousForcing resolves `field_dependencies` to positional indices at
    # materialize time and looks them up positionally at runtime; the two tuples must
    # agree on ordering, or forcings will read the wrong field.
    model_fields = merge(prognostic_model_fields, fields(formulation), velocities,
                         (; T=temperature), microphysical_fields)
    coupling_density = dynamics_density(dynamics)
    mass_density = total_density(dynamics)
    forcing = atmosphere_model_forcing(forcing, prognostic_model_fields, model_fields,
                                       grid, coriolis, coupling_density, mass_density,
                                       velocities, dynamics, formulation, microphysics,
                                       specific_prognostic_moisture)

    # The closure's scalars — thermodynamic density, moisture, microphysical prognostic fields, user
    # tracers — in the order the vertically-implicit solve indexes them (see `closure_scalar_index`)
    scalar_names = closure_scalar_names(formulation, microphysics, tracer_names)
    closure = Oceananigans.Utils.with_tracers(scalar_names, closure)
    closure_fields = build_closure_fields(nothing, grid, clock, scalar_names, regularized_boundary_conditions, closure)

    # Generate tracer advection scheme for each tracer
    # scalar_advection is always a NamedTuple after validate_tracer_advection (either user's partial NamedTuple or empty)
    # with_tracers fills in missing names using default_generator
    default_generator(names, initial_tuple) = default_scalar_advection
    scalar_advection_tuple = with_tracers(scalar_names, scalar_advection, default_generator, with_velocities=false)
    momentum_advection_tuple = (; momentum = momentum_advection)
    advection = merge(momentum_advection_tuple, scalar_advection_tuple)
    materialized_advection = NamedTuple(name => adapt_advection_order(materialize_advection(scheme, grid), grid) for (name, scheme) in pairs(advection))

    # Move microphysics lookup tables to the grid architecture (CPU → GPU)
    microphysics = on_architecture(arch, microphysics)

    model = AtmosphereModel(arch,
                            grid,
                            clock,
                            dynamics,
                            formulation,
                            thermodynamic_constants,
                            momentum,
                            moisture_density,
                            temperature,
                            pressure_solver,
                            velocities,
                            tracers,
                            nothing, # buoyancy, temporary solution for compatibility with Oceananigans.TurbulenceClosures
                            materialized_advection,
                            coriolis,
                            forcing,
                            microphysics,
                            microphysical_fields,
                            timestepper,
                            closure,
                            closure_fields,
                            radiation,
                            particles)

    # Initialize thermodynamics (dynamics-specific)
    initialize_model_thermodynamics!(model)

    # Seed the prognostic aerosol reservoir from the microphysics scheme's distribution. Dynamics
    # whose density is physical at construction (the anelastic reference state, a prescribed
    # density) are fully initialized here, so a model that is never `set!` still activates.
    # Compressible density fields are still zero, so this writes zero and the first `set!` that
    # supplies a density fills it in. Idempotent: every `set!` rewrites it.
    #
    # This belongs in the constructor rather than in `initialize!(model)` because it is a
    # *default*: `initialize!` runs after `set!`, which is where a user supplies `nᵃ` or `ρnᵃ`,
    # so re-seeding there would overwrite a user-supplied aerosol reservoir.
    set_default_aerosol_number!(model)

    return model
end

# Breeze's acoustic and SSP time steppers accept a `dynamics` keyword;
# Oceananigans' built-in steppers (RungeKutta3, QuasiAdamsBashforth2) do not.
timestepper_uses_dynamics(::Val) = false
timestepper_uses_dynamics(::Val{:SSPRungeKutta3}) = true
timestepper_uses_dynamics(::Val{:AcousticRungeKutta3}) = true
timestepper_uses_dynamics(s::Symbol) = timestepper_uses_dynamics(Val(s))

function Base.summary(model::AtmosphereModel)
    A = nameof(typeof(model.grid.architecture))
    G = nameof(typeof(model.grid))
    return string("AtmosphereModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

is_default_atmosphere_model_forcing(::Returns) = true
is_default_atmosphere_model_forcing(_) = false

function atmosphere_model_forcing_summary(model::AtmosphereModel)
    forcing = model.forcing
    names = Tuple(name for name in keys(forcing) if !is_default_atmosphere_model_forcing(forcing[name]))

    isempty(names) && return summary(model.forcing)

    summary_tuple = Tuple(string(name, "=>", nameof(typeof(forcing[name]))) for name in names)
    return join(summary_tuple, ", ")
end

# AtmosphereModel has a grid, so we can use grid-based implementations
Base.eltype(model::AtmosphereModel) = eltype(model.grid)
Architectures.architecture(model::AtmosphereModel) = model.grid.architecture

function Base.show(io::IO, model::AtmosphereModel)
    TS = nameof(typeof(model.timestepper))
    Mic = nameof(typeof(model.microphysics))
    tracernames = prettykeys(model.tracers)
    forcing_summary = atmosphere_model_forcing_summary(model)

    print(io, summary(model), "\n",
              "├── grid: ", summary(model.grid), "\n",
              "├── dynamics: ", summary(model.dynamics), "\n",
              "├── formulation: ", summary(model.formulation), "\n",
              "├── thermodynamic_constants: ", summary(model.thermodynamic_constants), "\n",
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

    print(io, "├── forcing: ", forcing_summary, "\n",
              "├── tracers: ", tracernames, "\n",
              "├── coriolis: ", summary(model.coriolis), "\n")

    if isnothing(model.particles)
        print(io, "└── microphysics: ", Mic)
    else
        print(io, "├── microphysics: ", Mic, "\n",
                  "└── particles: ", summary(model.particles))
    end
end

# `cell_advection_timescale(model::AtmosphereModel)` and the direction-aware `CellAdvectionTimescale`
# callable are defined in cell_advection_timescale.jl (included after this file).

# Prognostic field names from dynamics + thermodynamic formulation + microphysics + tracers
function prognostic_field_names(dynamics, formulation, microphysics, tracer_names)
    dynamics_names = prognostic_dynamics_field_names(dynamics)
    momentum_names = prognostic_momentum_field_names(dynamics)
    formulation_names = prognostic_thermodynamic_field_names(formulation)
    microphysical_names = prognostic_field_names(microphysics)
    moist_name = moisture_prognostic_name(microphysics)
    return tuple(dynamics_names..., momentum_names..., formulation_names..., moist_name, microphysical_names..., tracer_names...)
end

# The scalars a turbulence closure sees, in the order `with_tracers` and `build_closure_fields`
# receive them: the thermodynamic density, moisture, microphysical prognostics, user tracers.
function closure_scalar_names(formulation, microphysics, tracer_names)
    thermodynamic_name = thermodynamic_density_name(formulation)
    moisture_name = moisture_prognostic_name(microphysics)
    microphysical_names = prognostic_field_names(microphysics)
    return tuple(thermodynamic_name, moisture_name, microphysical_names..., tracer_names...)
end

closure_scalar_names(model::AtmosphereModel) =
    closure_scalar_names(model.formulation, model.microphysics, keys(model.tracers))

"""
$(TYPEDSIGNATURES)

The index under which the prognostic field `name` enters the vertically-implicit solve: `nothing`
for momentum, which is diffused with the closure's viscosity, and `Val(i)` for the `i`th closure
scalar, which is diffused with `diffusivity(closure, closure_fields, Val(i))`.
"""
function closure_scalar_index(model::AtmosphereModel, name::Symbol)
    name ∈ keys(model.momentum) && return nothing
    return Val(findfirst(==(name), closure_scalar_names(model)))
end

"""
$(TYPEDSIGNATURES)

Whether the prognostic field `name` sits out the vertically-implicit solve. The dynamics-specific
prognostics — the compressible dry density, the kinematic driver's density — are advanced explicitly
and have no diffusivity to apply; momentum and every scalar take the solve.
"""
skip_vertical_diffusion(model::AtmosphereModel, name::Symbol) =
    name ∈ keys(dynamics_prognostic_fields(model.dynamics))

function field_names(dynamics, formulation, microphysics, tracer_names)
    prog_names = prognostic_field_names(dynamics, formulation, microphysics, tracer_names)
    moist_specific = moisture_specific_name(microphysics)
    formulation_additional_names = additional_thermodynamic_field_names(formulation)
    default_additional_names = (:u, :v, :w, :T, moist_specific)
    return tuple(prog_names..., formulation_additional_names..., default_additional_names...)
end

function atmosphere_model_forcing(user_forcings, prognostic_fields, model_fields,
                                  grid, coriolis, coupling_density, mass_density,
                                  velocities, dynamics, formulation, microphysics,
                                  specific_prognostic_moisture)
    forcings_type = typeof(user_forcings)
    msg = string("AtmosphereModel forcing must be a NamedTuple, got $forcings_type")
    throw(ArgumentError(msg))
    return nothing
end

function atmosphere_model_forcing(::Nothing, prognostic_fields, model_fields,
                                  grid, coriolis, coupling_density, mass_density,
                                  velocities, dynamics, formulation, microphysics,
                                  specific_prognostic_moisture)
    names = keys(prognostic_fields)
    return NamedTuple{names}(Returns(zero(eltype(prognostic_fields[name]))) for name in names)
end

function atmosphere_model_forcing(user_forcings::NamedTuple, prognostic_fields, model_fields,
                                  grid, coriolis, coupling_density, mass_density,
                                  velocities, dynamics, formulation, microphysics,
                                  specific_prognostic_moisture)

    user_forcing_names = keys(user_forcings)

    if :ρs ∈ keys(prognostic_fields)
        forcing_fields = prognostic_fields
    else
        forcing_fields = merge(prognostic_fields, (; ρs=prognostic_fields.ρθ))
    end

    forcing_names = keys(forcing_fields)

    # Build a specific→density name map for any prognostic name that starts with `ρ`.
    # For example, :ρθ contributes :θ => :ρθ. Users may supply forcings under either key,
    # and the dispatch routes specific-keyed values through wrap_specific_forcing.
    specific_to_density = NamedTuple(specific_to_density_pairs(forcing_names))
    valid_specific_names = keys(specific_to_density)

    for name in user_forcing_names
        if name ∉ forcing_names && name ∉ valid_specific_names
            msg = string("Invalid forcing: forcing contains an entry for $name, but $name is not a prognostic field!", '\n',
                         "The forcing fields are ", forcing_names,
                         "; specific-key aliases are ", valid_specific_names, '.')
            throw(ArgumentError(msg))
        end
    end

    model_names = keys(model_fields)

    # Build specific fields for subsidence forcing (maps specific field names like :u, :θ to fields)
    formulation_fields = fields(formulation)
    moist_specific = moisture_specific_name(microphysics)
    specific_fields = merge(velocities, formulation_fields, NamedTuple{(moist_specific,)}((specific_prognostic_moisture,)))

    # Momentum, the dynamics mass variable, and thermodynamic density are weighted by the
    # coupling density (ρᵈ for CompressibleDynamics). Moisture, microphysical moments, and
    # user tracers are total-air mass fractions and therefore use total density. The extra
    # :ρs entry is the energy-forcing alias retained by potential-temperature formulations.
    coupling_density_names = tuple(prognostic_dynamics_field_names(dynamics)...,
                                   prognostic_momentum_field_names(dynamics)...,
                                   thermodynamic_density_name(formulation),
                                   :ρs)

    # Keep `density` as the coupling-density compatibility entry for other forcing
    # materializers; SpecificForcing selects between the two explicit carriers by target.
    forcing_context = (; coriolis,
                         density=coupling_density,
                         coupling_density,
                         total_density=mass_density,
                         coupling_density_names,
                         specific_fields)

    materialized = Tuple(
        assemble_field_forcing(n, f, user_forcings, model_names, forcing_context)
        for (n, f) in pairs(forcing_fields)
    )

    forcings = NamedTuple{forcing_names}(materialized)

    return forcings
end

# Assemble the materialized forcing for one prognostic field. Dispatch on the result of
# `specific_name_of(prognostic_name)`: a `Symbol` is the specific alias of a ρ-prefixed
# prognostic; `Nothing` means the prognostic name is already in specific form (e.g. a
# user tracer like `:c`) and has no separate density alias.
function assemble_field_forcing(prognostic_name, target_field, user_forcings, model_names, context)
    return assemble_field_forcing(prognostic_name, specific_name_of(prognostic_name),
                                  target_field, user_forcings, model_names, context)
end

# ρ-prefixed prognostic: combine any user-supplied ρ-keyed entry with any specific-keyed
# entry on the same prognostic, wrapping the specific-keyed entry in SpecificForcing.
function assemble_field_forcing(density_name, specific_name::Symbol, target_field,
                                user_forcings, model_names, context)
    raw_specific_forcing = get(user_forcings, specific_name, nothing)
    wrapped_specific_forcing = wrap_specific_forcing(raw_specific_forcing, density_name)
    raw_density_forcing = get(user_forcings, density_name, nothing)
    combined = combine_forcing_values(raw_density_forcing, wrapped_specific_forcing)
    return materialize_or_default(combined, target_field, density_name, model_names, context)
end

# Unprefixed prognostic (a user tracer like `:c`): the prognostic name *is* in specific
# form, so any forcing supplied under that name is a specific tendency and gets wrapped
# in SpecificForcing — the ρ factor is then applied at kernel time, matching the
# convention used for ρ-prefixed prognostics' specific aliases.
function assemble_field_forcing(tracer_name, ::Nothing, target_field,
                                user_forcings, model_names, context)
    raw_forcing = get(user_forcings, tracer_name, nothing)
    wrapped_forcing = wrap_specific_forcing(raw_forcing, tracer_name)
    return materialize_or_default(wrapped_forcing, target_field, tracer_name, model_names, context)
end

# Strip the `ρ` prefix from a density-weighted prognostic name; returns `nothing` for
# any name that is not ρ-prefixed.
function specific_name_of(density_name)
    s = string(density_name)
    return startswith(s, "ρ") ? Symbol(s[nextind(s, 1):end]) : nothing
end

# Default forcing for fields the user did not supply: a Returns that yields zero at every
# grid point. Dispatch on Nothing keeps the assemble path branch-free.
materialize_or_default(::Nothing, target_field, density_name, model_names, context) =
    Returns(zero(target_field.grid))

materialize_or_default(forcing, target_field, density_name, model_names, context) =
    materialize_atmosphere_model_forcing(forcing, target_field, density_name, model_names, context)

# Build (specific_name => density_name) pairs from a tuple of prognostic ρ-names.
function specific_to_density_pairs(forcing_names)
    pairs = Pair{Symbol, Symbol}[]
    for name in forcing_names
        specific_name = specific_name_of(name)
        isnothing(specific_name) || push!(pairs, specific_name => name)
    end
    return Tuple(pairs)
end

# Combine a density-keyed forcing value with the specific-keyed forcing value supplied
# for the same prognostic. `nothing` denotes "user did not supply this side". The two
# sides are flattened into a single tuple so the existing tuple-materialization path
# wraps them in MultipleForcings.
combine_forcing_values(::Nothing, ::Nothing) = nothing
combine_forcing_values(a::Tuple, ::Nothing) = a
combine_forcing_values(::Nothing, b::Tuple) = b
combine_forcing_values(a, ::Nothing) = a
combine_forcing_values(::Nothing, b) = b
combine_forcing_values(a::Tuple, b::Tuple) = (a..., b...)
combine_forcing_values(a::Tuple, b) = (a..., b)
combine_forcing_values(a, b::Tuple) = (a, b...)
combine_forcing_values(a, b) = (a, b)

function Oceananigans.fields(model::AtmosphereModel)
    formulation_fields = fields(model.formulation)
    auxiliary = (; T=model.temperature)
    return merge(prognostic_fields(model), formulation_fields, model.velocities, auxiliary, model.microphysical_fields)
end

function Oceananigans.prognostic_fields(model::AtmosphereModel)
    dynamics_fields = dynamics_prognostic_fields(model.dynamics)
    prognostic_formulation_fields = prognostic_fields(model.formulation)
    moist_name = moisture_prognostic_name(model.microphysics)
    thermodynamic_fields = merge(prognostic_formulation_fields, NamedTuple{(moist_name,)}((model.moisture_density,)))
    μ_names = prognostic_field_names(model.microphysics)
    μ_fields = NamedTuple{μ_names}(model.microphysical_fields[name] for name in μ_names)
    return merge(dynamics_fields, model.momentum, thermodynamic_fields, μ_fields, model.tracers)
end

Models.boundary_condition_args(model::AtmosphereModel) = (model.clock, fields(model))

#####
##### Lagrangian particle tracking
#####

# Velocities are diagnostic (u = ρu/ρ) and refreshed by `update_state!`, so they are
# current when the time steppers advect particles at the end of each step.
@inline Models.total_velocities(model::AtmosphereModel) = model.velocities

TimeSteppers.step_lagrangian_particles!(model::AtmosphereModel, Δt) =
    step_lagrangian_particles!(model.particles, model, Δt)

function total_energy(model)
    u, v, w = model.velocities
    k = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2 |> Field
    s = static_energy(model) |> Field
    return k + s
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
Oceananigans.OutputWriters.default_included_properties(::AtmosphereModel) = [:thermodynamic_constants]
