"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Standard SSP RK3 scheme for explicit time stepping
- `AcousticRungeKutta3`: Wicker-Skamarock RK3 with acoustic substepping for compressible dynamics
"""
module TimeSteppers

export SSPRungeKutta3, AcousticRungeKutta3,
       store_initial_state!,
       ssp_rk3_substep!,
       maybe_prepare_first_time_step!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: Oceananigans
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers,
                                 update_state!, maybe_prepare_first_time_step!, reconcile_state!
using Breeze.AtmosphereModels: kernel_time_step

include("ssp_runge_kutta_3.jl")
include("acoustic_substep_helpers.jl")
include("acoustic_runge_kutta_3.jl")

# Tier B (handoff B3 61c105f9): the vertically-implicit solves receive the stage interval in the grid's
# float type (`kernel_time_step`) instead of the host's Float64 value, so on a Float32 grid the rows,
# pivots and folded update run in Float32 (rounding-level change). Default off;
# `BREEZE_TIER_B_FLOAT32_SOLVES=true` at load.
const SOLVE_TIME_STEP_IN_GRID_FLOAT_TYPE = Ref(false)

implicit_solve_time_step(grid, Δt) =
    SOLVE_TIME_STEP_IN_GRID_FLOAT_TYPE[] ? kernel_time_step(grid.architecture, grid, Δt) : Δt

function __init__()
    SOLVE_TIME_STEP_IN_GRID_FLOAT_TYPE[] = lowercase(get(ENV, "BREEZE_TIER_B_FLOAT32_SOLVES", "false")) in ("1", "true")
    return nothing
end

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

end # module
