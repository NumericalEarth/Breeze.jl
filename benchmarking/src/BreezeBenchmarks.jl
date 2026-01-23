module BreezeBenchmarks

export
    # Benchmark cases
    convective_boundary_layer,

    # Benchmark utilities
    benchmark_time_stepping,
    BenchmarkResult

using Dates
using Printf
using Statistics

using Oceananigans
using Oceananigans.Units

using Breeze

include("convective_boundary_layer.jl")

#####
##### Benchmark result container
#####

struct BenchmarkResult
    name::String
    architecture::String
    float_type::String
    grid_size::Tuple{Int, Int, Int}
    time_steps::Int
    total_time_seconds::Float64
    time_per_step_seconds::Float64
    grid_points_per_second::Float64
    timestamp::DateTime
end

function Base.show(io::IO, r::BenchmarkResult)
    print(io, "BenchmarkResult: ", r.name, " on ", r.architecture)
end

function Base.show(io::IO, ::MIME"text/plain", r::BenchmarkResult)
    println(io, "BenchmarkResult")
    println(io, "├── name: ", r.name)
    println(io, "├── architecture: ", r.architecture)
    println(io, "├── float_type: ", r.float_type)
    println(io, "├── grid_size: ", r.grid_size)
    println(io, "├── time_steps: ", r.time_steps)
    println(io, "├── total_time: ", @sprintf("%.3f s", r.total_time_seconds))
    println(io, "├── time_per_step: ", @sprintf("%.6f s", r.time_per_step_seconds))
    println(io, "├── grid_points_per_second: ", @sprintf("%.2e", r.grid_points_per_second))
    print(io,   "└── timestamp: ", r.timestamp)
end

#####
##### Benchmark utilities
#####

"""
    benchmark_time_stepping(model;
                            time_steps = 100,
                            Δt = 0.05,
                            warmup_steps = 10,
                            name = "benchmark",
                            verbose = true)

Run a benchmark by executing `time_steps` time steps of the given model.

Returns a `BenchmarkResult` containing timing information.

# Arguments
- `model`: An `AtmosphereModel` to benchmark

# Keyword Arguments
- `time_steps`: Number of time steps to benchmark (default: 100)
- `Δt`: Time step size in seconds (default: 0.05)
- `warmup_steps`: Number of warmup steps before timing (default: 10)
- `name`: Name for this benchmark (default: "benchmark")
- `verbose`: Print progress information (default: true)
"""
function benchmark_time_stepping(model;
                                 time_steps = 100,
                                 Δt = 0.05,
                                 warmup_steps = 10,
                                 name = "benchmark",
                                 verbose = true)

    grid = model.grid
    arch = Oceananigans.Architectures.architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    simulation = Simulation(model; Δt)

    if verbose
        @info "Benchmark: $name"
        @info "  Architecture: $arch"
        @info "  Float type: $FT"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Time step: $Δt s"
        @info "  Warmup steps: $warmup_steps"
        @info "  Benchmark steps: $time_steps"
    end

    # Warmup phase
    if verbose
        @info "  Running warmup..."
    end
    for _ in 1:warmup_steps
        time_step!(simulation)
    end

    # Synchronize device before timing
    synchronize_device(arch)

    # Benchmark phase
    if verbose
        @info "  Running benchmark..."
    end
    start_time = time_ns()
    for _ in 1:time_steps
        time_step!(simulation)
    end
    synchronize_device(arch)
    end_time = time_ns()

    total_time_seconds = (end_time - start_time) / 1e9
    time_per_step_seconds = total_time_seconds / time_steps
    grid_points_per_second = total_points / time_per_step_seconds

    result = BenchmarkResult(
        name,
        string(typeof(arch)),
        string(FT),
        (Nx, Ny, Nz),
        time_steps,
        total_time_seconds,
        time_per_step_seconds,
        grid_points_per_second,
        now()
    )

    if verbose
        @info "  Results:"
        @info "    Total time: $(@sprintf("%.3f", total_time_seconds)) s"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
    end

    return result
end

#####
##### Device synchronization
#####

synchronize_device(::Oceananigans.Architectures.CPU) = nothing

function synchronize_device(::Oceananigans.Architectures.GPU)
    CUDA.synchronize()
    return nothing
end

# Lazy-load CUDA for GPU synchronization
using CUDA: CUDA

end # module
