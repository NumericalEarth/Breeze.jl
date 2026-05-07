module BreezeBenchmarks

export
    # Benchmark cases
    convective_boundary_layer,

    # Benchmark utilities
    many_time_steps!,
    benchmark_time_stepping,
    run_benchmark_simulation,
    BenchmarkResult,
    SimulationResult,
    BenchmarkMetadata

using Dates
using JLD2
using Printf
using Statistics

using Oceananigans
using Oceananigans.Architectures: GPU, ReactantState
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.OutputWriters: JLD2Writer, IterationInterval, TimeInterval, write_output!
using Oceananigans.Simulations: SpecifiedTimes

using Breeze

# Reactant must be loaded before CUDA so that ReactantCUDAExt activates with
# CUDA's kernel compilation hooks correctly registered.
using Reactant: Reactant, @trace
using Enzyme: Enzyme
using CUDA: CUDA, CUDABackend
# Compatibility for CUDA v5 and v6
if isdefined(CUDA, :CUDACore)
    using CUDA: CUDACore
else
    const CUDACore = CUDA
end
using AMDGPU: AMDGPU, ROCBackend

# Base functionalities
include("metadata.jl")
include("result.jl")
include("timestepping.jl")
include("utils.jl")
# Specific models to benchmark
include("convective_boundary_layer.jl")

end # module
