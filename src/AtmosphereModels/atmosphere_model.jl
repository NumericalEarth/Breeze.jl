using ..Thermodynamics: Thermodynamics, ThermodynamicConstants, ReferenceState

using Oceananigans: AbstractModel, Center, CenterField, Clock, Field
using Oceananigans: Centered, XFaceField, YFaceField, ZFaceField
using Oceananigans.Advection: adapt_advection_order
using Oceananigans.AbstractOperations: @at
using Oceananigans.Forcings: materialize_forcing
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Grids: ZDirection
using Oceananigans.Models: validate_model_halo, validate_tracer_advection
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, time_discretization, build_closure_fields
using Oceananigans.Utils: launch!, prettytime, prettykeys, with_tracers

import Oceananigans: fields, prognostic_fields
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_momentum_advection

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

formulation_pressure_solver(formulation, grid) = nothing

mutable struct AtmosphereModel{Frm, Arc, Tst, Grd, Clk, Thm, Mom, Moi, Mfr, Buy,
                               Tmp, Prs, Sol, Vel, Trc, Adv, Cor, Frc, Mic, Cnd, Cls, Cfs} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    formulation :: Frm
    thermodynamic_constants :: Thm
    momentum :: Mom
    moisture_density :: Moi
    specific_moisture :: Mfr
    temperature :: Tmp
    pressure :: Prs
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
end

# Stub functions to be overloaded by formulation-specific files
function default_formulation end
function materialize_formulation end

"""
    $(TYPEDSIGNATURES)

Return an AtmosphereModel that uses the anelastic approximation following
[Pauluis2008](@citet).

Example
=======

```jldoctest
julia> using Breeze

julia> grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 2, 3));

julia> model = AtmosphereModel(grid)
AtmosphereModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── formulation: AnelasticFormulation(p₀=101325.0, θ₀=288.0)
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: 
│   ├── momentum: Centered(order=2)
│   ├── ρe: Centered(order=2)
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
                         formulation = default_formulation(grid, thermodynamic_constants),
                         moisture_density = DefaultValue(),
                         tracers = tuple(),
                         coriolis = nothing,
                         boundary_conditions = NamedTuple(),
                         forcing = NamedTuple(),
                         advection = nothing,
                         momentum_advection = nothing,
                         scalar_advection = nothing,
                         closure = nothing,
                         microphysics = nothing, # WarmPhaseSaturationAdjustment(),
                         timestepper = :RungeKutta3)

    if !isnothing(advection)
        # TODO: check that tracer+momentum advection were not independently set.
        scalar_advection = momentum_advection = advection
    else
        isnothing(momentum_advection) && (momentum_advection = Centered(order=2))
        isnothing(scalar_advection) && (scalar_advection = Centered(order=2))
    end

    # Check halos and throw an error if the grid's halo is too small
    validate_model_halo(grid, momentum_advection, scalar_advection, closure)

    # Reduce the advection order in directions that do not have enough grid points
    momentum_advection = validate_momentum_advection(momentum_advection, grid)
    default_scalar_advection, scalar_advection = validate_tracer_advection(scalar_advection, grid)
    
    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    tracer_names = validate_tracers(tracers)

    # Next, we form a list of default boundary conditions:
    prognostic_names = prognostic_field_names(formulation, microphysics, tracers)
    default_boundary_conditions = NamedTuple{prognostic_names}(FieldBoundaryConditions() for _ in prognostic_names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    all_names = field_names(formulation, microphysics, tracers)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, all_names)

    # Materialize the full formulation with thermodynamic fields and pressure
    formulation = materialize_formulation(formulation, grid, boundary_conditions)


    velocities, momentum = materialize_momentum_and_velocities(formulation, grid, boundary_conditions)
    microphysical_fields = materialize_microphysical_fields(microphysics, grid, boundary_conditions)

    tracers = NamedTuple(name => CenterField(grid, boundary_conditions=boundary_conditions[name]) for name in tracer_names)

    if moisture_density isa DefaultValue
        moisture_density = CenterField(grid, boundary_conditions=boundary_conditions.ρqᵗ)
    end

    # Diagnostic fields
    specific_moisture = CenterField(grid)
    temperature = CenterField(grid)
    pressure = formulation.pressure_anomaly

    prognostic_microphysical_fields = NamedTuple(microphysical_fields[name] for name in prognostic_field_names(microphysics))
    prognostic_fields = collect_prognostic_fields(formulation,
                                                  momentum,
                                                  moisture_density,
                                                  prognostic_microphysical_fields,
                                                  tracers)

    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(timestepper, grid, prognostic_fields; implicit_solver)
    pressure_solver = formulation_pressure_solver(formulation, grid)

    model_fields = merge(prognostic_fields, velocities, (; T=temperature, qᵗ=specific_moisture))
    forcing = atmosphere_model_forcing(forcing, prognostic_fields, model_fields)

    # Include thermodynamic density (ρe or ρθ), ρqᵗ plus user tracers for closure field construction
    closure_thermo_name = thermodynamic_density_name(formulation)
    scalar_names = tuple(closure_thermo_name, :ρqᵗ, tracer_names...)
    closure = Oceananigans.Utils.with_tracers(scalar_names, closure)
    closure_fields = build_closure_fields(nothing, grid, clock, scalar_names, boundary_conditions, closure)

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
                            formulation,
                            thermodynamic_constants,
                            momentum,
                            moisture_density,
                            specific_moisture,
                            temperature,
                            pressure,
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
                            closure_fields)

    update_state!(model)

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

cell_advection_timescale(model::AtmosphereModel) = cell_advection_timescale(model.grid, model.velocities)

# Default prognostic field names - overloaded by formulation-specific files
function prognostic_field_names(formulation, microphysics, tracer_names)
    default_names = (:ρu, :ρv, :ρw, :ρqᵗ)
    formulation_names = prognostic_field_names(formulation)
    microphysical_names = prognostic_field_names(microphysics)
    return tuple(default_names..., formulation_names..., microphysical_names..., tracer_names...)
end

function field_names(formulation, microphysics, tracer_names)
    prog_names = prognostic_field_names(formulation, microphysics, tracer_names)
    default_additional_names = (:u, :v, :w, :T, :qᵗ)
    formulation_additional_names = additional_field_names(formulation)
    return tuple(prog_names..., default_additional_names..., formulation_additional_names...)
end

function atmosphere_model_forcing(user_forcings, prognostic_fields, model_fields)
    forcings_type = typeof(user_forcings)
    msg = string("AtmosphereModel forcing must be a NamedTuple, got $forcings_type")
    throw(ArgumentError(msg))
    return nothing
end

function atmosphere_model_forcing(::Nothing, prognostic_fields, model_fields)
    names = keys(prognostic_fields)
    return NamedTuple{names}(Returns(zero(eltype(prognostic_fields[name]))) for name in names)
end

function atmosphere_model_forcing(user_forcings::NamedTuple, prognostic_fields, model_fields)
    user_forcing_names = keys(user_forcings)
    for name in user_forcing_names
        if name ∉ keys(prognostic_fields)
            msg = string("Invalid forcing: forcing contains an entry for $name, but $name is not a prognostic field!", '\n',
                         "The prognostic fields are ", keys(prognostic_fields))
            throw(ArgumentError(msg))
        end
    end

    model_field_names = keys(model_fields)

    materialized = Tuple(
        name in keys(user_forcings) ?
            materialize_forcing(user_forcings[name], field, name, model_field_names) :
            Returns(zero(eltype(field)))
            for (name, field) in pairs(prognostic_fields)
    )

    prognostic_names = keys(prognostic_fields)
    forcings = NamedTuple{prognostic_names}(materialized)

    return forcings
end

function fields(model::AtmosphereModel)
    formulation_fields = fields(model.formulation)
    auxiliary = (; T=model.temperature, qᵗ=model.specific_moisture)
    return merge(prognostic_fields(model), formulation_fields, model.velocities, auxiliary)
end

function prognostic_fields(model::AtmosphereModel)
    prognostic_formulation_fields = prognostic_fields(model.formulation)
    thermodynamic_fields = merge(prognostic_formulation_fields, (; ρqᵗ=model.moisture_density))
    μ_names = prognostic_field_names(model.microphysics)
    μ_fields= NamedTuple{μ_names}(model.microphysical_fields[name] for name in μ_names)
    return merge(model.momentum, thermodynamic_fields, μ_fields, model.tracers)
end

#####
##### Helper functions for accessing thermodynamic fields
#####

# Stub function - implementation in anelastic_formulation.jl
function thermodynamic_density_name end

"""
    static_energy_density(model::AtmosphereModel)

Return an `AbstractField` representing static energy density for `model`.
"""
function static_energy_density end

"""
    static_energy(model::AtmosphereModel)

Return an `AbstractField` representing the (specific) static energy
for `model`.
"""
function static_energy end

"""
    potential_temperature_density(model::AtmosphereModel)

Return an `AbstractField` representing potential temperature density
for `model`.
"""
function potential_temperature_density end

"""
    liquid_ice_potential_temperature(model::AtmosphereModel)

Return an `AbstractField` representing potential temperature `θ`
for `model`.
"""
function liquid_ice_potential_temperature end

function total_energy(model)
    u, v, w = model.velocities
    k = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2 |> Field
    e = static_energy(model) |> Field
    return k + e
end