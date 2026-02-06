"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including the SSP RK3 scheme
which is TVD (total variation diminishing) with CFL coefficient = 1.
"""
module TimeSteppers

export SSPRungeKutta3,
       store_initial_state!,
       ssp_rk3_substep!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers

include("ssp_runge_kutta_3.jl")

# Extend TimeStepper to support SSPRK3 via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

end # module
