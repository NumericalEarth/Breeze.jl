#####
##### Scalar-tendency kernel profiling (no Simulation, no model)
#####
##### Times the bare scalar tendency kernel on either backend so they can be
##### compared. On a `ReactantState` grid the launch is compiled into a single
##### optimized XLA program (`raise=true`) and profiled with
##### `Reactant.Profiler.@timed`; on a vanilla (eager KA/CUDA) grid the kernel
##### is launched directly and timed with device synchronization. Results reuse
##### `BenchmarkResult` with `mode = "tendency"` so they flow through the
##### existing JSON / publish pipeline unchanged.
#####

"""
    benchmark_scalar_tendency(tendency!, args;
                              nrepeat = 100,
                              name = "scalar_tendency",
                              advection::AbstractString = "",
                              backend::AbstractString = "reactant",
                              verbose = true)

Time `tendency!(args...)` on the architecture of the grid in `args` (the tuple
`(Gc, grid, advection, U, c)` from `scalar_tendency_problem`). On a
`ReactantState` grid the launch is compiled with `Reactant.@compile raise=true
raise_first=true sync=true` (compile time recorded separately) and profiled via
`Reactant.Profiler.@timed`. On a vanilla grid the kernel is launched eagerly and
timed over `nrepeat` calls with device synchronization (no Reactant compile).

Returns a `BenchmarkResult` with `mode = "tendency"`, where `time_per_step_seconds`
is the mean wall time of one tendency evaluation and `grid_points_per_second` is
the corresponding throughput.
"""
function benchmark_scalar_tendency(tendency!, args;
                                   nrepeat = 100,
                                   name = "scalar_tendency",
                                   advection::AbstractString = "",
                                   backend::AbstractString = "reactant",
                                   verbose = true)

    grid = args[2]  # args = (Gc, grid, advection, U, c)
    arch = Oceananigans.Architectures.architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz
    is_reactant = arch isa ReactantState

    if verbose
        @info "Scalar tendency benchmark: $name"
        @info "  Architecture: $arch"
        @info "  Backend: $backend"
        @info "  Advection: $advection"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Repeats: $nrepeat"
    end

    if is_reactant
        # Compile the launch into a single optimized XLA program, then profile
        # it. Passing the already-compiled thunk makes @timed profile it
        # directly (it special-cases `::Reactant.Compiler.Thunk`). `runtime_ns`
        # is the mean over `nrepeat`.
        verbose && @info "  Compiling scalar_tendency! with Reactant (raise=true)..."
        compile_start = time_ns()
        compiled! = Reactant.@compile raise=true raise_first=true sync=true tendency!(args...)
        compile_time_seconds = (time_ns() - compile_start) / 1e9
        prof = Reactant.Profiler.@timed nrepeat=nrepeat compiled!(args...)
        time_per_call_seconds = prof.runtime_ns / 1e9
    else
        # Vanilla backend: launch the kernel eagerly (no Reactant compile). Warm
        # up once, then time `nrepeat` launches with device synchronization.
        compile_time_seconds = 0.0
        tendency!(args...)
        synchronize_device(arch)
        start_time = time_ns()
        for _ in 1:nrepeat
            tendency!(args...)
        end
        synchronize_device(arch)
        time_per_call_seconds = ((time_ns() - start_time) / 1e9) / nrepeat
    end

    total_time_seconds = time_per_call_seconds * nrepeat
    calls_per_second = 1 / time_per_call_seconds
    grid_points_per_second = total_points / time_per_call_seconds

    gpu_memory_used = arch isa GPU{CUDABackend} ? CUDACore.MemoryInfo().pool_used_bytes : 0
    metadata = BenchmarkMetadata(arch)

    # Reuse BenchmarkResult: `time_steps` -> nrepeat, per-step fields hold
    # per-call timings, and the non-applicable model fields are "none".
    result = BenchmarkResult(
        name,
        string(FT),
        String(advection),
        "none",      # closure
        "none",      # dynamics
        "none",      # microphysics
        String(backend),
        "tendency",  # mode
        (Nx, Ny, Nz),
        nrepeat,
        0.0,         # Δt — not applicable to a single tendency evaluation
        total_time_seconds,
        time_per_call_seconds,
        calls_per_second,
        grid_points_per_second,
        compile_time_seconds,
        gpu_memory_used,
        metadata,
    )

    if verbose
        @info "  Results:"
        @info "    Time per call: $(@sprintf("%.6f", time_per_call_seconds)) s"
        @info "    Grid points/s: $(@sprintf("%.2e", grid_points_per_second))"
        @info "    Compile time: $(@sprintf("%.3f", compile_time_seconds)) s"
    end

    memory_reclaim(arch)

    return result
end
