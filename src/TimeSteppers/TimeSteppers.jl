"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including the SSP RK3 scheme
which is TVD (total variation diminishing) with CFL coefficient = 1.
"""
module TimeSteppers

export SSPRungeKutta3TimeStepper

import Oceananigans.TimeSteppers: TimeStepper
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

include("ssp_rk3.jl")

# Extend TimeStepper to support SSPRK3 via Symbol
TimeStepper(::Val{:SSPRK3}, args...; kwargs...) =
    SSPRungeKutta3TimeStepper(args...; kwargs...)

end # module
