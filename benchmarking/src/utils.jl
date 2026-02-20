
#####
##### Benchmark utilities
#####

"""
    benchmark_time_stepping(model;
                            time_steps = 100,
                            Δt = 0.05,
                            warmup_steps = 10,
                            name = "benchmark",
                            verbose = true,
                            advection::AbstractString = "",
                            closure::AbstractString = "",
                            dynamics::AbstractString = "",
                            microphysics::AbstractString = "",
                            )

Run a benchmark by executing `time_steps` time steps of the given model.
Uses `many_time_steps!` to avoid Simulation overhead.

Returns a `BenchmarkResult` containing timing information and system metadata.
"""
function benchmark_time_stepping(model;
                                 time_steps = 100,
                                 Δt = 0.05,
                                 warmup_steps = 10,
                                 name = "benchmark",
                                 verbose = true,
                                 advection::AbstractString = "",
                                 closure::AbstractString = "",
                                 dynamics::AbstractString = "",
                                 microphysics::AbstractString = "",
                                 )

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
    steps_per_second = time_steps / total_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    gpu_memory_used = arch isa GPU ? CUDA.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = BenchmarkResult(
        name,
        string(FT),
        String(advection),
        String(closure),
        String(dynamics),
        String(microphysics),
        (Nx, Ny, Nz),
        time_steps,
        Δt,
        total_time_seconds,
        time_per_step_seconds,
        steps_per_second,
        grid_points_per_second,
        gpu_memory_used,
        metadata,
    )

    if verbose
        @info "  Results:"
        @info "    Total time: $(@sprintf("%.3f", total_time_seconds)) s"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        if arch isa GPU
            @info "    GPU memory usage: $(Base.format_bytes(gpu_memory_used))"
        end
    end

    if arch isa GPU
        # Reclaim memory, so that next benchmarks will start from a clean state.
        CUDA.reclaim()
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
                             verbose = true,
                             advection::AbstractString = "",
                             closure::AbstractString = "",
                             dynamics::AbstractString = "",
                             microphysics::AbstractString = "",
                             )

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
- `advection`: Name of the advection scheme
- `closure`: Name of the closure
- `dynamics`: Name of the dynamics
- `microphysics`: Name of the microphysics

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
                                  verbose = true,
                                  advection::AbstractString = "",
                                  closure::AbstractString = "",
                                  dynamics::AbstractString = "",
                                  microphysics::AbstractString = "",
                                  )

    grid = model.grid
    arch = Oceananigans.Architectures.architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    # Build output filename
    timestamp = Dates.format(now(UTC), "yyyy-mm-dd_HHMMSS")
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
    simulation.output_writers[:surface] = JLD2Writer(model, outputs;
        filename = output_filename,
        indices = (:, :, 1),
        schedule = TimeInterval(output_interval),
        overwrite_existing = true
    )

    # Final snapshot: save full 3D fields at the end of the simulation
    simulation.output_writers[:final_3d] = JLD2Writer(model, outputs;
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
    steps_per_second = time_steps / wall_time_seconds
    grid_points_per_second = total_points / time_per_step_seconds

    gpu_memory_used = arch isa GPU ? CUDA.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    result = SimulationResult(
        name,
        string(FT),
        String(advection),
        String(closure),
        String(dynamics),
        String(microphysics),
        (Nx, Ny, Nz),
        Float64(stop_time),
        time_steps,
        Float64(Δt),
        wall_time_seconds,
        time_per_step_seconds,
        steps_per_second,
        grid_points_per_second,
        output_filename,
        gpu_memory_used,
        metadata,
    )

    if verbose
        @info "  Simulation complete!"
        @info "    Wall time: $(@sprintf("%.1f", wall_time_seconds)) s ($(@sprintf("%.2f", wall_time_seconds / 3600)) hours)"
        @info "    Time steps: $time_steps"
        @info "    Time per step: $(@sprintf("%.6f", time_per_step_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        @info "    Surface timeseries: $output_filename"
        @info "    Final 3D snapshot: $final_filename"
        if arch isa GPU
            @info "    GPU memory usage: $(Base.format_bytes(gpu_memory_used))"
        end
    end

    if arch isa GPU
        # Reclaim memory, so that next simulations will start from a clean state.
        CUDA.reclaim()
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
