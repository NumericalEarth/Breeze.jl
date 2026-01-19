"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Standard SSP RK3 scheme for explicit time stepping
- `AcousticSSPRungeKutta3`: SSP RK3 with acoustic substepping for compressible dynamics
"""
module TimeSteppers

export SSPRungeKutta3, AcousticSSPRungeKutta3

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers

include("ssp_runge_kutta_3.jl")
include("acoustic_ssp_runge_kutta_3.jl")

# Extend TimeStepper to support SSPRK3 via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticSSPRungeKutta3}, args...; kwargs...) =
    AcousticSSPRungeKutta3(args...; kwargs...)

end # module
