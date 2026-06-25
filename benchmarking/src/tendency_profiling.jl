#####
##### Scalar-tendency kernel profiling (no Simulation, no model)
#####
##### Compiles the bare scalar tendency kernel through Reactant (raise=true) and
##### profiles its execution with `Reactant.Profiler.@timed`, which auto-detects
##### the already-compiled program and reports the mean runtime per call. Results
##### reuse `BenchmarkResult` with `mode = "tendency"` so they flow through the
##### existing JSON / publish pipeline unchanged.
#####

"""
    benchmark_scalar_tendency(tendency!, args;
                              nrepeat = 100,
                              name = "scalar_tendency",
                              advection::AbstractString = "",
                              optimize = true,
                              verbose = true)

Compile `tendency!(args...)` with `Reactant.@compile optimize=optimize raise=true
raise_first=true sync=true` (recording compile time separately, since the KA
kernel must be raised to XLA) and profile its runtime with
`Reactant.Profiler.@timed`. `args` is the tuple `(Gc, grid, advection, U, c)`
returned by `scalar_tendency_problem`. Pass `optimize=false` to skip XLA's
optimization passes (`:none`) and benchmark / dump the unoptimized program.

Returns a `BenchmarkResult` with `mode = "tendency"`, where `time_per_step_seconds`
is the mean wall time of one tendency evaluation and `grid_points_per_second` is
the corresponding throughput.
"""
function benchmark_scalar_tendency(tendency!, args;
                                   nrepeat = 100,
                                   name = "scalar_tendency",
                                   advection::AbstractString = "",
                                   optimize = true,
                                   verbose = true)

    grid = args[2]  # args = (Gc, grid, advection, U, c)
    arch = Oceananigans.Architectures.architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    total_points = Nx * Ny * Nz

    if verbose
        @info "Scalar tendency benchmark: $name"
        @info "  Architecture: $arch"
        @info "  Advection: $advection"
        @info "  Grid size: $Nx × $Ny × $Nz ($total_points points)"
        @info "  Repeats: $nrepeat"
        @info "  Optimize: $optimize"
        @info "  Compiling scalar_tendency! with Reactant (raise=true, optimize=$optimize)..."
    end

    compile_start = time_ns()
    compiled! = Reactant.@compile optimize=optimize raise=true raise_first=true sync=true tendency!(args...)
    compile_time_seconds = (time_ns() - compile_start) / 1e9

    # Mean runtime per call. Passing the already-compiled thunk makes @timed
    # profile it directly (it special-cases `::Reactant.Compiler.Thunk`) rather
    # than recompiling without `raise`. `runtime_ns` is the mean over `nrepeat`.
    prof = Reactant.Profiler.@timed nrepeat=nrepeat compiled!(args...)
    time_per_call_seconds = prof.runtime_ns / 1e9

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
        "reactant",  # backend
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
