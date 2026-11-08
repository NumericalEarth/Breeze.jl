"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Standard SSP RK3 scheme for explicit time stepping
- `AcousticRungeKutta3`: Wicker-Skamarock RK3 with acoustic substepping for compressible dynamics
- `MultirateInfinitesimalStep`: multirate infinitesimal step methods with acoustic substepping
"""
module TimeSteppers

export SSPRungeKutta3, AcousticRungeKutta3, MultirateInfinitesimalStep,
       MultirateInfinitesimalStepTableau, multirate_infinitesimal_step_tableau,
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
include("multirate_infinitesimal_step_coefficients.jl")
include("multirate_infinitesimal_step.jl")

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:MultirateInfinitesimalStep}, args...; kwargs...) =
    MultirateInfinitesimalStep(args...; kwargs...)

end # module
