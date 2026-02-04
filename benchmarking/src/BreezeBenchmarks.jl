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
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval, TimeInterval, write_output!
using Oceananigans.Simulations: SpecifiedTimes

using Breeze

using CUDA: CUDA, CUDABackend

# Base functionalities
include("metadata.jl")
include("result.jl")
include("timestepping.jl")
include("utils.jl")
# Specific models to benchmark
include("convective_boundary_layer.jl")

end # module
