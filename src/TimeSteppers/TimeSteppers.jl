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

include("positivity_preserving_rk3.jl")

end # module

