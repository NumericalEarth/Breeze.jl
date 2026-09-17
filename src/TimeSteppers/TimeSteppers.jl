"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Three-stage, third-order SSP Runge-Kutta scheme for explicit time stepping
- `SSPRungeKutta43`: Four-stage, third-order SSP Runge-Kutta scheme with twice the SSP time step
- `AcousticRungeKutta3`: Wicker-Skamarock RK3 with acoustic substepping for compressible dynamics
"""
module TimeSteppers

export SSPRungeKutta, SSPRungeKutta3, SSPRungeKutta43, AcousticRungeKutta3,
       store_initial_state!,
       ssp_runge_kutta_substep!,
       maybe_prepare_first_time_step!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: Oceananigans
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers,
                                 update_state!, maybe_prepare_first_time_step!, reconcile_state!
using Breeze.AtmosphereModels: kernel_time_step

include("ssp_runge_kutta.jl")
include("acoustic_substep_helpers.jl")
include("acoustic_runge_kutta_3.jl")

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta43}, args...; kwargs...) =
    SSPRungeKutta43(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

end # module
