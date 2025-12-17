"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including a positivity-preserving
variant that uses directionally-split advection to maintain tracer bounds.
"""
module TimeSteppers

export
    PositivityPreservingRK3TimeStepper,
    compute_split_advection_tendency!

using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures: architecture
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: TimeStepper

include("positivity_preserving_rk3.jl")

# Extend TimeStepper to support PositivityPreservingRK3 via Symbol
TimeStepper(::Val{:PositivityPreservingRK3}, args...; split_advection=NamedTuple(), kwargs...) =
    PositivityPreservingRK3TimeStepper(args...; split_advection, kwargs...)

end # module

