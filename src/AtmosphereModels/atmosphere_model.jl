using ..Thermodynamics: Thermodynamics, ThermodynamicConstants, ReferenceState

using Oceananigans: AbstractModel, Center, CenterField, Clock, Field
using Oceananigans: Centered, XFaceField, YFaceField, ZFaceField
using Oceananigans.Advection: adapt_advection_order
using Oceananigans.Forcings: materialize_forcing
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, time_discretization, build_closure_fields
using Oceananigans.Utils: launch!, prettytime, prettykeys

import Oceananigans: fields, prognostic_fields
import Oceananigans.Advection: cell_advection_timescale

materialize_density(formulation, grid) = CenterField(grid)

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

mutable struct AtmosphereModel{Frm, Arc, Tst, Grd, Clk, Thm, Den, Mom, Eng, Mse, Moi, Mfr,
                               Tmp, Prs, Sol, Vel, Trc, Adv, Cor, Frc, Mic, Cnd, Cls, Cfs} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    formulation :: Frm
    thermodynamics :: Thm
    density :: Den
    momentum :: Mom
    energy_density :: Eng
    specific_energy :: Mse
    moisture_density :: Moi
    specific_moisture :: Mfr
    temperature :: Tmp
    pressure :: Prs
    pressure_solver :: Sol
    velocities :: Vel
    tracers :: Trc
    advection :: Adv
    coriolis :: Cor
    forcing :: Frc
    microphysics :: Mic
    microphysical_fields :: Cnd
    timestepper :: Tst
    closure :: Cls
    closure_fields :: Cfs
end

function default_formulation(grid, thermo)
    reference_state = ReferenceState(grid, thermo)
    return AnelasticFormulation(reference_state)
end

"""
    $(TYPEDSIGNATURES)

Return an AtmosphereModel that uses the anelastic approximation following
[Pauluis2008](@citet).

Example
=======

```jldoctest
julia> using Breeze, Breeze.AtmosphereModels, Oceananigans

julia> grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 2, 3));

julia> model = AtmosphereModel(grid)
AtmosphereModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── formulation: AnelasticFormulation(p₀=101325.0, θ₀=288.0)
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
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
                         thermodynamics = ThermodynamicConstants(eltype(grid)),
                         formulation = default_formulation(grid, thermodynamics),
                         moisture_density = DefaultValue(),
                         tracers = tuple(),
                         coriolis = nothing,
                         boundary_conditions = NamedTuple(),
                         forcing = NamedTuple(),
                         advection = Centered(order=2),
                         closure = nothing,
                         microphysics = nothing, # WarmPhaseSaturationAdjustment(),
                         timestepper = :RungeKutta3)

    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    tracer_names = validate_tracers(tracers)

    # Next, we form a list of default boundary conditions:
    prognostic_names = prognostic_field_names(formulation, microphysics, tracers)
    FT = eltype(grid)
    default_boundary_conditions = NamedTuple{prognostic_names}(FieldBoundaryConditions() for _ in prognostic_names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    all_names = field_names(formulation, microphysics, tracers)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, all_names)

    density = materialize_density(formulation, grid)
    velocities, momentum = materialize_momentum_and_velocities(formulation, grid, boundary_conditions)
    microphysical_fields = materialize_microphysical_fields(microphysics, grid, boundary_conditions)
    advection = adapt_advection_order(advection, grid)

    tracers = NamedTuple(name => CenterField(grid, boundary_conditions=boundary_conditions[name]) for name in tracer_names)

    if moisture_density isa DefaultValue
        moisture_density = CenterField(grid, boundary_conditions=boundary_conditions.ρqᵗ)
    end

    energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρe)

    # Diagnostic fields
    specific_energy = CenterField(grid) # e = ρe / ρᵣ (diagnostic per-mass energy)
    specific_moisture = CenterField(grid)
    temperature = CenterField(grid)
    pressure = CenterField(grid)

    prognostic_microphysical_fields = NamedTuple(microphysical_fields[name] for name in prognostic_field_names(microphysics))
    prognostic_fields = collect_prognostic_fields(formulation,
                                                  density,
                                                  momentum,
                                                  energy_density,
                                                  moisture_density,
                                                  prognostic_microphysical_fields,
                                                  tracers)

    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(timestepper, grid, prognostic_fields; implicit_solver)
    pressure_solver = formulation_pressure_solver(formulation, grid)

    model_fields = merge(prognostic_fields, velocities, (; T=temperature, qᵗ=specific_moisture))
    forcing = atmosphere_model_forcing(forcing, prognostic_fields, model_fields)

    # May need to use more names in `tracers` for this to work
    closure_names = tuple(:ρe, :ρqᵗ, tracer_names...)
    closure = Oceananigans.Utils.with_tracers(closure_names, closure)
    closure_fields = build_closure_fields(nothing, grid, clock, tracer_names, boundary_conditions, closure)

    model = AtmosphereModel(arch,
                            grid,
                            clock,
                            formulation,
                            thermodynamics,
                            density,
                            momentum,
                            energy_density,
                            specific_energy,
                            moisture_density,
                            specific_moisture,
                            temperature,
                            pressure,
                            pressure_solver,
                            velocities,
                            tracers,
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
              "├── timestepper: ", TS, "\n",
              "├── advection scheme: ", summary(model.advection), "\n",
              "├── tracers: ", tracernames, "\n",
              "├── coriolis: ", summary(model.coriolis), "\n",
              "└── microphysics: ", Mic)
end

cell_advection_timescale(model::AtmosphereModel) = cell_advection_timescale(model.grid, model.velocities)

function prognostic_field_names(formulation, microphysics, tracer_names)
    default_names = (:ρu, :ρv, :ρw, :ρe, :ρqᵗ)
    microphysical_names = prognostic_field_names(microphysics)
    return tuple(default_names..., microphysical_names..., tracer_names...)
end

function field_names(formulation, microphysics, tracer_names)
    prognostic_names = prognostic_field_names(formulation, microphysics, tracer_names)
    additional_names = (:u, :v, :w, :e, :T, :qᵗ)
    return tuple(prognostic_names..., additional_names...)
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
    auxiliary_thermodynamic_fields = (e=model.specific_energy, T=model.temperature, qᵗ=model.specific_moisture)
    return merge(prognostic_fields(model),
                 model.velocities,
                 auxiliary_thermodynamic_fields)
end

function prognostic_fields(model::AtmosphereModel)
    thermodynamic_fields = (ρe=model.energy_density, ρqᵗ=model.moisture_density)
    microphysical_names = prognostic_field_names(model.microphysics)
    prognostic_microphysical_fields = NamedTuple{microphysical_names}(
        model.microphysical_fields[name] for name in microphysical_names)

    return merge(model.momentum, thermodynamic_fields, prognostic_microphysical_fields, model.tracers)
end
