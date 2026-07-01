#####
##### Benchmark result container
#####

struct BenchmarkResult
    name::String
    float_type::String
    advection::String
    closure::String
    dynamics::String
    microphysics::String
    backend::String
    mode::String  # "forward" (forward stepping only) or "ad" (forward+backward via Enzyme)
    grid_size::Tuple{Int, Int, Int}
    time_steps::Int
    Δt::Float64
    total_time_seconds::Float64
    time_per_step_seconds::Float64
    steps_per_second::Float64
    grid_points_per_second::Float64
    compile_time_seconds::Float64
    gpu_memory_used::Int64
    metadata::BenchmarkMetadata
end

function Base.show(io::IO, r::BenchmarkResult)
    print(io, "BenchmarkResult: ", r.name, " on ", r.metadata.architecture)
end

function Base.show(io::IO, ::MIME"text/plain", r::BenchmarkResult)
    println(io, "BenchmarkResult")
    println(io, "├── name: ", r.name)
    println(io, "├── float_type: ", r.float_type)
    println(io, "├── advection: ", r.advection)
    println(io, "├── closure: ", r.closure)
    println(io, "├── dynamics: ", r.dynamics)
    println(io, "├── microphysics: ", r.microphysics)
    println(io, "├── backend: ", r.backend)
    println(io, "├── mode: ", r.mode)
    println(io, "├── grid_size: ", r.grid_size)
    println(io, "├── time_steps: ", r.time_steps)
    println(io, "├── Δt: ", r.Δt)
    println(io, "├── total_time: ", @sprintf("%.3f s", r.total_time_seconds))
    println(io, "├── time_per_step: ", @sprintf("%.6f s", r.time_per_step_seconds))
    println(io, "├── steps_per_second: ", @sprintf("%.6f/s", r.steps_per_second))
    println(io, "├── grid_points_per_second: ", @sprintf("%.2e", r.grid_points_per_second))
    println(io, "├── compile_time: ", @sprintf("%.3f s", r.compile_time_seconds))
    println(io, "├── gpu_memory_used: ", Base.format_bytes(r.gpu_memory_used))
    print(io,   "└── metadata: ", r.metadata.architecture, " @ ", r.metadata.timestamp)
end

"""
    log_benchmark_result(r::BenchmarkResult)

Emit the standard `@info` summary lines for a benchmark result. Shared by the
time-stepping and tendency benchmarks so they report identically (total time,
time per step, throughput, and — when applicable — compile time and GPU memory).
"""
function log_benchmark_result(r::BenchmarkResult)
    @info "  Results:"
    @info "    Total time: $(@sprintf("%.3f", r.total_time_seconds)) s"
    @info "    Time per step: $(@sprintf("%.6f", r.time_per_step_seconds)) s"
    @info "    Grid points/s: $(@sprintf("%.2e", r.grid_points_per_second))"
    r.compile_time_seconds > 0 && @info "    Compile time: $(@sprintf("%.3f", r.compile_time_seconds)) s"
    r.gpu_memory_used > 0 && @info "    GPU memory usage: $(Base.format_bytes(r.gpu_memory_used))"
    return nothing
end

#####
##### Simulation result container (for full simulations with output)
#####

struct SimulationResult
    name::String
    float_type::String
    advection::String
    closure::String
    dynamics::String
    microphysics::String
    grid_size::Tuple{Int, Int, Int}
    simulation_time_seconds::Float64
    time_steps::Int
    Δt::Float64
    wall_time_seconds::Float64
    time_per_step_seconds::Float64
    steps_per_second::Float64
    grid_points_per_second::Float64
    output_file::String
    gpu_memory_used::Int64
    metadata::BenchmarkMetadata
end

function Base.show(io::IO, r::SimulationResult)
    print(io, "SimulationResult: ", r.name, " (", r.simulation_time_seconds, " s sim time)")
end

function Base.show(io::IO, ::MIME"text/plain", r::SimulationResult)
    println(io, "SimulationResult")
    println(io, "├── name: ", r.name)
    println(io, "├── float_type: ", r.float_type)
    println(io, "├── advection: ", r.advection)
    println(io, "├── closure: ", r.closure)
    println(io, "├── dynamics: ", r.dynamics)
    println(io, "├── microphysics: ", r.microphysics)
    println(io, "├── grid_size: ", r.grid_size)
    println(io, "├── simulation_time: ", @sprintf("%.1f s (%.2f hours)", r.simulation_time_seconds, r.simulation_time_seconds / 3600))
    println(io, "├── time_steps: ", r.time_steps)
    println(io, "├── Δt: ", r.Δt)
    println(io, "├── wall_time: ", @sprintf("%.1f s (%.2f hours)", r.wall_time_seconds, r.wall_time_seconds / 3600))
    println(io, "├── time_per_step: ", @sprintf("%.6f s", r.time_per_step_seconds))
    println(io, "├── steps_per_second: ", @sprintf("%.6f/s", r.steps_per_second))
    println(io, "├── grid_points_per_second: ", @sprintf("%.2e", r.grid_points_per_second))
    println(io, "├── output_file: ", r.output_file)
    println(io, "├── gpu_memory_used: ", Base.format_bytes(r.gpu_memory_used))
    print(io,   "└── metadata: ", r.metadata.architecture, " @ ", r.metadata.timestamp)
end
