using Oceananigans
using Oceananigans.Advection: adapt_advection_order
using Oceananigans.Models: AbstractModel
using Oceananigans.TimeSteppers: TimeStepper
using Oceananigans.Fields: CenterField


struct DefaultThermodynamics end
materialize_thermo_vars(thermodynamics, grid) = NamedTuple()
materialize_thermo_vars(::DefaultThermodynamics, grid) = (; T=CenterField(grid))

struct DefaultMicrophysics end
materialize_condenstates(microphysics, grid) = NamedTuple()
materialize_condenstates(::DefaultMicrophysics, grid) = (; ql=CenterField(grid))

mutable struct RainshaftModel{Arc, Grd, Clk, Den, Tmp, Wat, Cnd, Prs, Thm, Mic, Adv, Tst} <: AbstractModel{Tst, Arc}
    architecture :: Arc
    grid :: Grd
    clock :: Clk
    density :: Den
    temperature :: Tmp
    water_vapor :: Wat
    water_condensates :: Cnd
    pressure :: Prs
    thermodynamics :: Thm
    microphysics :: Mic
    advection :: Adv
    timestepper :: Tst
end

"""
    RainshaftModel(grid;
                   clock = Clock(grid),
                   thermodynamics = DefaultThermodynamics(),
                   microphysics = DefaultMicrophysics(),
                   advection = WENO(order=5),
                   timestepper = :RungeKutta3)

Return an RainshaftModel that for learning purposes.

Example
=======
```jldoctest
julia> using Breeze, Breeze.RainshaftsModels, Oceananigans

julia> grid = RectilinearGrid(size=(1, 1, 8), extent=(1, 1, 3));

julia> model = RainshaftModel(grid)
RainshaftModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: WENO(order=5)
├── thermodynamics scheme: DefaultThermodynamics
└── microphysics scheme: DefaultMicrophysics
```
"""
function RainshaftModel(grid;
                        clock = Clock(grid),
                        thermodynamics = DefaultThermodynamics(),
                        microphysics = DefaultMicrophysics(),
                        advection = WENO(order=5),
                        timestepper = :RungeKutta3)
    if size(grid)[1:2] != (1,1)
        msg = "`Grid needs to be 1x1 in the horizontal dimensions."
        throw(ArgumentError(msg))
    end
    if !(thermodynamics isa DefaultThermodynamics)
        throw(ArgumentError("Only DefaultThermodynamics is supported."))
    end

    # Setup
    FT = eltype(grid)
    arch = grid.architecture
    advection = adapt_advection_order(advection, grid)

    # Allocate prognostic fields
    density = CenterField(grid)
    water_vapor = CenterField(grid)
    temperature = materialize_thermo_vars(thermodynamics, grid)
    water_condensates = materialize_condenstates(microphysics, grid)
    thermo_vars = temperature
    
    # Allocate derived fields
    pressure = CenterField(grid)

    # Set up time stepper
    prognostic_fields = (; ρ=density, qv=water_vapor)
    prognostic_fields = merge(prognostic_fields, thermo_vars)
    prognostic_fields = merge(prognostic_fields, water_condensates)
    timestepper = TimeStepper(timestepper, grid, prognostic_fields)

    model = RainshaftModel(arch,
                           grid,
                           clock,
                           density,
                           temperature,
                           water_vapor,
                           water_condensates,
                           pressure, # derived field
                           thermodynamics,
                           microphysics,
                           advection,
                           timestepper)

    update_state!(model)

    return model
end

using Oceananigans.Utils: prettytime, prettykeys

function Base.summary(model::RainshaftModel)
    A = nameof(typeof(model.grid.architecture))
    G = nameof(typeof(model.grid))
    return string("RainshaftModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::RainshaftModel)
    TS = nameof(typeof(model.timestepper))
    tracernames = prettykeys(model.tracers)

    print(io, summary(model), "\n",
        "├── grid: ", summary(model.grid), "\n",
        "├── timestepper: ", TS, "\n",
        "├── advection scheme: ", summary(model.advection), "\n",
        "├── thermodynamics scheme: ", summary(model.thermodynamics), "\n",
        "└── microphysics scheme: ", summary(model.microphysics))
end
