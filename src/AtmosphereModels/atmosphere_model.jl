using ..Thermodynamics:
    AtmosphereThermodynamics,
    ReferenceConstants,
    reference_pressure,
    reference_density,
    mixture_gas_constant,
    mixture_heat_capacity,
    dry_air_gas_constant

using Oceananigans
using Oceananigans.Advection: Centered, adapt_advection_order
using Oceananigans.Grids: ZDirection
using Oceananigans.Models: AbstractModel
using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver, HomogeneousNeumannFormulation
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

materialize_condenstates(microphysics, grid) = NamedTuple() #(; qˡ=CenterField(grid), qᵛ=CenterField(grid))
materialize_density(formulation, grid) = CenterField(grid)

struct WarmPhaseSaturationAdjustment end
struct DefaultValue end

tupleit(t::Tuple) = t
tupleit(t) = tuple(t)

formulation_pressure_solver(formulation, grid) = nothing

mutable struct AtmosphereModel{Frm, Arc, Tst, Grd, Clk, Thm, Den, Mom, Eng, Wat, Hum,
                               Tmp, Prs, Ppa, Sol, Vel, Trc, Adv, Cor, Frc, Mic, Cnd, Cls, Dif} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    formulation :: Frm
    thermodynamics :: Thm
    density :: Den
    momentum :: Mom
    energy :: Eng
    absolute_humidity :: Wat
    specific_humidity :: Hum
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
    condensates :: Cnd
    timestepper :: Tst
    closure :: Cls
    diffusivity_fields :: Dif
end

function default_formulation(grid, thermo)
    FT = eltype(grid)
    base_pressure = convert(FT, 101325)
    potential_temperature = convert(FT, 288)
    constants = ReferenceConstants(base_pressure, potential_temperature)
    return AnelasticFormulation(grid, constants, thermo)
end

"""
    AtmosphereModel(grid;
                    clock = Clock(grid),
                    thermodynamics = AtmosphereThermodynamics(eltype(grid)),
                    formulation = default_formulation(grid, thermodynamics),
                    absolute_humidity = DefaultValue(),
                    tracers = tuple(),
                    coriolis = nothing,
                    boundary_conditions = NamedTuple(),
                    forcing = NamedTuple(),
                    advection = WENO(order=5),
                    microphysics = WarmPhaseSaturationAdjustment(),
                    timestepper = :RungeKutta3)

Return an AtmosphereModel that uses the anelastic approximation following
[Pauluis (2008)](https://journals.ametsoc.org/view/journals/atsc/65/8/200s7jas2475.1.xml).
"""
function AtmosphereModel(grid;
                         clock = Clock(grid),
                         thermodynamics = AtmosphereThermodynamics(eltype(grid)),
                         formulation = default_formulation(grid, thermodynamics),
                         absolute_humidity = DefaultValue(),
                         tracers = tuple(),
                         coriolis = nothing,
                         boundary_conditions = NamedTuple(),
                         forcing = NamedTuple(),
                         advection = WENO(order=5),
                         microphysics = WarmPhaseSaturationAdjustment(),
                         timestepper = :RungeKutta3)

    arch = grid.architecture
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    hydrostatic_pressure_anomaly = CenterField(grid)
    nonhydrostatic_pressure = CenterField(grid)

    # Next, we form a list of default boundary conditions:
    names = field_names(formulation, tracers)
    FT = eltype(grid)
    forcing = NamedTuple{names}(Returns(zero(FT)) for _ in names)
    default_boundary_conditions = NamedTuple{names}(FieldBoundaryConditions() for _ in names)
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, names)

    density = materialize_density(formulation, grid)
    velocities, momentum = materialize_momentum_and_velocities(formulation, grid, boundary_conditions)
    tracers = NamedTuple(n => CenterField(grid, boundary_conditions=boundary_conditions[n]) for name in tracers)
    condensates = materialize_condenstates(microphysics, grid)
    advection = adapt_advection_order(advection, grid)

    if absolute_humidity isa DefaultValue
        absolute_humidity = CenterField(grid, boundary_conditions=boundary_conditions.ρq)
    end

    energy = CenterField(grid, boundary_conditions=boundary_conditions.e)
    specific_humidity = CenterField(grid)
    temperature = CenterField(grid)

    prognostic_fields = collect_prognostic_fields(formulation,
                                                  density,
                                                  momentum,
                                                  energy,
                                                  absolute_humidity,
                                                  condensates,
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
                            energy,
                            absolute_humidity,
                            specific_humidity,
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
                            condensates,
                            timestepper,
                            closure,
                            diffusivity_fields)

    update_state!(model)

    return model
end

using Oceananigans.Utils: prettytime, ordered_dict_show, prettykeys
using Oceananigans.TurbulenceClosures: closure_summary

function Base.summary(model::AtmosphereModel)
    A = nameof(typeof(model.grid.architecture))
    G = nameof(typeof(model.grid))
    return string("AtmosphereModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::AtmosphereModel)
    TS = nameof(typeof(model.timestepper))
    tracernames = prettykeys(model.tracers)

    print(io, summary(model), "\n",
        "├── grid: ", summary(model.grid), "\n",
        "├── formulation: ", summary(model.formulation), "\n",
        "├── timestepper: ", TS, "\n",
        "├── advection scheme: ", summary(model.advection), "\n",
        "├── tracers: ", tracernames, "\n",
        "└── coriolis: ", summary(model.coriolis))
end
