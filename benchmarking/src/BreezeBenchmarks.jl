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
using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval, TimeInterval, write_output!
using Oceananigans.Simulations: SpecifiedTimes

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
##### Simulation result container (for full simulations with output)
#####

struct SimulationResult
    name::String
    float_type::String
    grid_size::Tuple{Int, Int, Int}
    simulation_time_seconds::Float64
    time_steps::Int
    Δt::Float64
    wall_time_seconds::Float64
    time_per_step_seconds::Float64
    grid_points_per_second::Float64
    output_file::String
    metadata::BenchmarkMetadata
end

function Base.show(io::IO, r::SimulationResult)
    print(io, "SimulationResult: ", r.name, " (", r.simulation_time_seconds, " s sim time)")
end

function Base.show(io::IO, ::MIME"text/plain", r::SimulationResult)
    println(io, "SimulationResult")
    println(io, "├── name: ", r.name)
    println(io, "├── float_type: ", r.float_type)
    println(io, "├── grid_size: ", r.grid_size)
    println(io, "├── simulation_time: ", @sprintf("%.1f s (%.2f hours)", r.simulation_time_seconds, r.simulation_time_seconds / 3600))
    println(io, "├── time_steps: ", r.time_steps)
    println(io, "├── Δt: ", r.Δt)
    println(io, "├── wall_time: ", @sprintf("%.1f s (%.2f hours)", r.wall_time_seconds, r.wall_time_seconds / 3600))
    println(io, "├── time_per_step: ", @sprintf("%.6f s", r.time_per_step_seconds))
    println(io, "├── grid_points_per_second: ", @sprintf("%.2e", r.grid_points_per_second))
    println(io, "├── output_file: ", r.output_file)
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
##### Full simulation with output (for validation and longer runs)
#####

"""
    run_benchmark_simulation(model;
                             stop_time = 2hours,
                             Δt = 0.5,
                             output_interval = 10minutes,
                             output_dir = ".",
                             name = "benchmark_simulation",
                             output_fields = (:u, :v, :w, :θ),
                             verbose = true)

Run a full simulation with output writers for validation and longer benchmarks.

Based on the FastEddy CBL case (Sauer & Munoz-Esparza 2020), which runs for
approximately 2 hours to reach quasi-steady convective state.

# Arguments
- `model`: The AtmosphereModel to simulate

# Keyword Arguments
- `stop_time`: Total simulation time (default: 2 hours)
- `Δt`: Time step size (default: 0.5 s, suitable for production runs)
- `output_interval`: Interval between output snapshots (default: 10 minutes)
- `output_dir`: Directory for output files (default: current directory)
- `name`: Name for the simulation (used in output filename)
- `output_fields`: Tuple of field names to save (default: u, v, w, θ)
- `verbose`: Print progress information

# Returns
A `SimulationResult` containing timing information and the output file path.
"""
function run_benchmark_simulation(model;
                                  stop_time = 2hours,
                                  Δt = 0.5,
                                  output_interval = 10minutes,
                                  output_dir = ".",
                                  name = "benchmark_simulation",
                                  output_fields = (:u, :v, :w, :θ),
                                  verbose = true)

    grid = model.grid
    arch = Oceananigans.Architectures.architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    # Build output filename
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    output_filename = joinpath(output_dir, "$(name)_$(timestamp).jld2")

    # Final snapshot filename
    final_filename = replace(output_filename, ".jld2" => "_final.jld2")

    if verbose
        @info "Benchmark Simulation: $name"
        @info "  Architecture: $arch"
        @info "  Float type: $FT"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Time step: $Δt s"
        @info "  Stop time: $(stop_time) s ($(stop_time / 3600) hours)"
        @info "  Output interval: $(output_interval) s ($(output_interval / 60) minutes)"
        @info "  Surface output: $output_filename (2D slices at z=0)"
        @info "  Final 3D snapshot: $final_filename"
    end

    # Create simulation
    simulation = Simulation(model; Δt, stop_time)

    # Add progress callback
    if verbose
        wall_time_ref = Ref(time_ns())
        function progress(sim)
            elapsed = (time_ns() - wall_time_ref[]) / 1e9
            wall_time_ref[] = time_ns()
            u_max = maximum(abs, sim.model.velocities.u)
            @info @sprintf("Time: %.1f/%.1f hours, Δt: %.2f s, max|u|: %.2f m/s, wall: %.1f s",
                           time(sim) / 3600, stop_time / 3600, sim.Δt, u_max, elapsed)
        end
        simulation.callbacks[:progress] = Callback(progress, TimeInterval(output_interval))
    end

    # Build outputs dictionary from field names
    outputs = Dict{Symbol, Any}()
    for field_name in output_fields
        if haskey(model.velocities, field_name)
            outputs[field_name] = model.velocities[field_name]
        elseif hasproperty(model, :tracers) && haskey(model.tracers, field_name)
            outputs[field_name] = model.tracers[field_name]
        elseif field_name == :θ
            # Potential temperature is a common diagnostic
            outputs[:θ] = model.thermodynamic_state.θ
        end
    end

    # Periodic output: only save the lowest level (2D slices) to reduce I/O cost
    simulation.output_writers[:surface] = JLD2OutputWriter(model, outputs;
        filename = output_filename,
        indices = (:, :, 1),
        schedule = TimeInterval(output_interval),
        overwrite_existing = true
    )

    # Final snapshot: save full 3D fields at the end of the simulation
    simulation.output_writers[:final_3d] = JLD2OutputWriter(model, outputs;
        filename = final_filename,
        schedule = IterationInterval(typemax(Int)),  # Never triggers during run
        overwrite_existing = true
    )

    # Add callback to write final 3D snapshot at end of simulation
    function save_final_snapshot(sim)
        @info "  Saving final 3D snapshot to: $final_filename"
        # Manually trigger the output writer
        Oceananigans.OutputWriters.write_output!(sim.output_writers[:final_3d], sim)
    end
    simulation.callbacks[:final_snapshot] = Callback(save_final_snapshot, SpecifiedTimes(stop_time))

    # Synchronize before timing
    synchronize_device(arch)

    if verbose
        @info "  Starting simulation..."
    end

    # Run simulation and time it
    start_time = time_ns()
    run!(simulation)
    synchronize_device(arch)
    end_time = time_ns()

    wall_time_seconds = (end_time - start_time) / 1e9
    time_steps = iteration(simulation)
    time_per_step_seconds = wall_time_seconds / time_steps
    grid_points_per_second = total_points / time_per_step_seconds

    metadata = BenchmarkMetadata(arch)

    result = SimulationResult(
        name,
        string(FT),
        (Nx, Ny, Nz),
        Float64(stop_time),
        time_steps,
        Float64(Δt),
        wall_time_seconds,
        time_per_step_seconds,
        grid_points_per_second,
        output_filename,
        metadata
    )

    if verbose
        @info "  Simulation complete!"
        @info "    Wall time: $(@sprintf("%.1f", wall_time_seconds)) s ($(@sprintf("%.2f", wall_time_seconds / 3600)) hours)"
        @info "    Time steps: $time_steps"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        @info "    Surface timeseries: $output_filename"
        @info "    Final 3D snapshot: $final_filename"
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
