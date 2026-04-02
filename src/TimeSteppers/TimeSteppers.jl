"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Standard SSP RK3 scheme for explicit time stepping
- `AcousticSSPRungeKutta3`: SSP RK3 with acoustic substepping for compressible dynamics
- `AcousticRungeKutta3`: Wicker-Skamarock RK3 with acoustic substepping for compressible dynamics
"""
module TimeSteppers

export SSPRungeKutta3, AcousticSSPRungeKutta3, AcousticRungeKutta3,
       IMEXRungeKuttaARS232,
       IMEXRungeKuttaSSP3332,
       store_initial_state!,
       ssp_rk3_substep!,
       maybe_prepare_first_time_step!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers,
                                 update_state!, maybe_prepare_first_time_step!

include("ssp_runge_kutta_3.jl")
include("acoustic_ssp_runge_kutta_3.jl")
include("acoustic_runge_kutta_3.jl")
include("imex_ark_common.jl")
include("imex_ark_ars232.jl")
include("imex_ark_ssp3332.jl")

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticSSPRungeKutta3}, args...; kwargs...) =
    AcousticSSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:IMEXRungeKuttaARS232}, args...; kwargs...) =
    IMEXRungeKuttaARS232(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:IMEXRungeKuttaSSP3332}, args...; kwargs...) =
    IMEXRungeKuttaSSP3332(args...; kwargs...)

end # module
