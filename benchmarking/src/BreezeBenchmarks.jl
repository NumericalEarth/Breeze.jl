module BreezeBenchmarks

export
    # Benchmark cases
    convective_boundary_layer,

    # Benchmark utilities
    many_time_steps!,
    benchmark_time_stepping,
    BenchmarkResult,
    BenchmarkMetadata,

    # I/O
    save_benchmark,
    load_benchmark

using Dates
using JLD2
using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!

using Breeze

using CUDA: CUDA, CUDABackend

include("convective_boundary_layer.jl")

#####
##### System metadata
#####

struct BenchmarkMetadata
    julia_version::String
    oceananigans_version::String
    breeze_version::String
    architecture::String
    gpu_name::Union{String, Nothing}
    cuda_version::Union{String, Nothing}
    cpu_model::String
    num_threads::Int
    hostname::String
    timestamp::DateTime
end

function BenchmarkMetadata(arch)
    gpu_name = nothing
    cuda_version = nothing

    if arch isa Oceananigans.Architectures.GPU{CUDABackend}
        try
            gpu_name = CUDA.name(CUDA.device())
            cuda_version = string(CUDA.runtime_version())
        catch
            gpu_name = "Unknown GPU"
            cuda_version = "Unknown"
        end
    end

    # Get CPU model
    cpu_model = "$(Sys.cpu_info()[1].model) ($(Sys.CPU_NAME))"

    return BenchmarkMetadata(
        string(VERSION),
        string(pkgversion(Oceananigans)),
        string(pkgversion(Breeze)),
        string(typeof(arch)),
        gpu_name,
        cuda_version,
        cpu_model,
        Threads.nthreads(),
        gethostname(),
        now()
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::BenchmarkMetadata)
    println(io, "BenchmarkMetadata")
    println(io, "├── julia_version: ", m.julia_version)
    println(io, "├── oceananigans_version: ", m.oceananigans_version)
    println(io, "├── breeze_version: ", m.breeze_version)
    println(io, "├── architecture: ", m.architecture)
    if !isnothing(m.gpu_name)
        println(io, "├── gpu_name: ", m.gpu_name)
        println(io, "├── cuda_version: ", m.cuda_version)
    end
    println(io, "├── cpu_model: ", m.cpu_model)
    println(io, "├── num_threads: ", m.num_threads)
    println(io, "├── hostname: ", m.hostname)
    print(io,   "└── timestamp: ", m.timestamp)
end

#####
##### Benchmark result container
#####

struct BenchmarkResult
    name::String
    float_type::String
    grid_size::Tuple{Int, Int, Int}
    time_steps::Int
    Δt::Float64
    total_time_seconds::Float64
    time_per_step_seconds::Float64
    grid_points_per_second::Float64
    metadata::BenchmarkMetadata
end

function Base.show(io::IO, r::BenchmarkResult)
    print(io, "BenchmarkResult: ", r.name, " on ", r.metadata.architecture)
end

function Base.show(io::IO, ::MIME"text/plain", r::BenchmarkResult)
    println(io, "BenchmarkResult")
    println(io, "├── name: ", r.name)
    println(io, "├── float_type: ", r.float_type)
    println(io, "├── grid_size: ", r.grid_size)
    println(io, "├── time_steps: ", r.time_steps)
    println(io, "├── Δt: ", r.Δt)
    println(io, "├── total_time: ", @sprintf("%.3f s", r.total_time_seconds))
    println(io, "├── time_per_step: ", @sprintf("%.6f s", r.time_per_step_seconds))
    println(io, "├── grid_points_per_second: ", @sprintf("%.2e", r.grid_points_per_second))
    print(io,   "└── metadata: ", r.metadata.architecture, " @ ", r.metadata.timestamp)
end

#####
##### Time stepping without Simulation overhead
#####

"""
    many_time_steps!(model, Δt, N=100)

Execute `N` time steps of `model` with time step `Δt`.
This directly calls `time_step!` without any Simulation overhead.
"""
function many_time_steps!(model, Δt, N=100)
    for _ in 1:N
        time_step!(model, Δt)
    end
    return nothing
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
Uses `many_time_steps!` to avoid Simulation overhead.

Returns a `BenchmarkResult` containing timing information and system metadata.
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
    many_time_steps!(model, Δt, warmup_steps)

    # Synchronize device before timing
    synchronize_device(arch)

    # Benchmark phase
    if verbose
        @info "  Running benchmark..."
    end
    start_time = time_ns()
    many_time_steps!(model, Δt, time_steps)
    synchronize_device(arch)
    end_time = time_ns()

    total_time_seconds = (end_time - start_time) / 1e9
    time_per_step_seconds = total_time_seconds / time_steps
    grid_points_per_second = total_points / time_per_step_seconds

    metadata = BenchmarkMetadata(arch)

    result = BenchmarkResult(
        name,
        string(FT),
        (Nx, Ny, Nz),
        time_steps,
        Δt,
        total_time_seconds,
        time_per_step_seconds,
        grid_points_per_second,
        metadata
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

#####
##### I/O
#####

"""
    save_benchmark(filename, results)

Save benchmark results to a JLD2 file.
`results` can be a single `BenchmarkResult` or a vector of results.
"""
function save_benchmark(filename, results)
    jldopen(filename, "w") do file
        file["results"] = results
    end
    return nothing
end

"""
    load_benchmark(filename)

Load benchmark results from a JLD2 file.
Returns either a single `BenchmarkResult` or a vector of results.
"""
function load_benchmark(filename)
    return jldopen(filename, "r") do file
        file["results"]
    end
end

end # module
