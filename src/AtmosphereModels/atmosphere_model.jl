using ..Thermodynamics: ThermodynamicConstants, ReferenceState

using Oceananigans: AbstractModel, Center, CenterField, Clock, Field
using Oceananigans: WENO, XFaceField, YFaceField, ZFaceField
using Oceananigans.Advection: adapt_advection_order
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Grids: ZDirection
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.Utils: launch!, prettytime, prettykeys

import Oceananigans.Advection: cell_advection_timescale

using KernelAbstractions: @kernel, @index

materialize_density(formulation, grid) = CenterField(grid)

struct DefaultValue end

tupleit(t::Tuple) = t
tupleit(t) = tuple(t)

formulation_pressure_solver(formulation, grid) = nothing

mutable struct AtmosphereModel{Frm, Arc, Tst, Grd, Clk, Thm, Den, Mom, Eng, Moi, Mfr,
                               Tmp, Prs, Ppa, Sol, Vel, Trc, Adv, Cor, Frc, Mic, Cnd, Cls, Dif} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    formulation :: Frm
    thermodynamics :: Thm
    density :: Den
    momentum :: Mom
    energy_density :: Eng
    moisture_density :: Moi
    moisture_mass_fraction :: Mfr
    temperature :: Tmp
    nonhydrostatic_pressure :: Prs
    hydrostatic_pressure_anomaly :: Ppa
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
    diffusivity_fields :: Dif
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
├── advection scheme: WENO{3, Float64, Float32}(order=5)
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
                         advection = WENO(order=5),
                         microphysics = nothing, # WarmPhaseSaturationAdjustment(),
                         timestepper = :RungeKutta3)

    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    hydrostatic_pressure_anomaly = CenterField(grid)
    nonhydrostatic_pressure = CenterField(grid)

    # Next, we form a list of default boundary conditions:
    names = prognostic_field_names(formulation, microphysics, tracers)
    FT = eltype(grid)
    forcing = NamedTuple{names}(Returns(zero(FT)) for _ in names)
    default_boundary_conditions = NamedTuple{names}(FieldBoundaryConditions() for _ in names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, names)

    density = materialize_density(formulation, grid)
    velocities, momentum = materialize_momentum_and_velocities(formulation, grid, boundary_conditions)
    tracers = NamedTuple(n => CenterField(grid, boundary_conditions=boundary_conditions[n]) for name in tracers)
    microphysical_fields = materialize_microphysical_fields(microphysics, grid, boundary_conditions)
    advection = adapt_advection_order(advection, grid)

    if moisture_density isa DefaultValue
        moisture_density = CenterField(grid, boundary_conditions=boundary_conditions.ρqᵗ)
    end

    energy_density = CenterField(grid, boundary_conditions=boundary_conditions.ρe)
    moisture_mass_fraction = CenterField(grid, boundary_conditions=boundary_conditions.ρqᵗ)
    temperature = CenterField(grid)

    prognostic_microphysical_fields = NamedTuple(microphysics_fields[name] for name in prognostic_field_names(microphysics))
    prognostic_fields = collect_prognostic_fields(formulation,
                                                  density,
                                                  momentum,
                                                  energy_density,
                                                  moisture_density,
                                                  prognostic_microphysical_fields,
                                                  tracers)

    timestepper = TimeStepper(timestepper, grid, prognostic_fields)
    pressure_solver = formulation_pressure_solver(formulation, grid)

    # TODO: support these
    closure = nothing
    diffusivity_fields = nothing

    model = AtmosphereModel(arch,
                            grid,
                            clock,
                            formulation,
                            thermodynamics,
                            density,
                            momentum,
                            energy_density,
                            moisture_density,
                            moisture_mass_fraction,
                            temperature,
                            nonhydrostatic_pressure,
                            hydrostatic_pressure_anomaly,
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
                            diffusivity_fields)

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
