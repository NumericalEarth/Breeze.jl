#####
##### Per-kernel GPU times of the vanilla time step
#####
##### `run_benchmarks.jl --profile_kernels` profiles each vanilla time-stepping case after
##### every case has been timed and saved, in the same process: its kernels are already
##### compiled, and the profiler cannot touch any published timing. The profile uses
##### CUDA.jl's integrated profiler (`CUDA.@profile`, CUPTI underneath; no Nsight install
##### needed). Each kernel's launches are grouped by its name up to the first `(` or `<`,
##### and the `TOP_KERNELS` kernels with the most GPU time are written to
##### `kernel_profile.json`, with a Markdown table in `kernel_profile.md`.
#####

using BreezeBenchmarks: many_time_steps!, BenchmarkMetadata
using CUDA: CUDA
using Printf: @sprintf
using Statistics: median

const TOP_KERNELS = 10

"""
    kernel_times(device)

Group the kernel launches of a `CUDA.@profile` device trace by kernel name, up to the
first `(` or `<`, skipping memory copies and sets. Returns one `NamedTuple` per kernel,
sorted by total GPU time, longest first.
"""
function kernel_times(device)
    launches = Dict{String, Vector{Float64}}()
    for (name, start, stop) in zip(device.name, device.start, device.stop)
        (ismissing(name) || startswith(name, "[")) && continue  # "[copy …]", "[set …]"
        short_name = strip(first(split(name, r"[(<]")))
        push!(get!(launches, short_name, Float64[]), stop - start)
    end

    kernels = [(name = String(name),
                instances = length(times),
                total_time_seconds = sum(times),
                median_time_seconds = median(times),
                min_time_seconds = minimum(times),
                max_time_seconds = maximum(times)) for (name, times) in launches]

    return sort!(kernels; by = k -> k.total_time_seconds, rev = true)
end

"""
    profile_kernels(cases)

Rebuild the model of each vanilla time-stepping case recorded by `run_benchmarks`,
take its warmup steps, and profile its `time_steps` steps with `CUDA.@profile`.
Returns one entry per case, with the `TOP_KERNELS` kernels by GPU time.
"""
function profile_kernels(cases)
    entries = []
    for case in cases
        println("\n", "-" ^ 70)
        println("Profiling kernels: $(case.name)")
        println("-" ^ 70)

        model = case.build_model()
        Δt = case.float_type(case.Δt)
        many_time_steps!(model, Δt, case.warmup_steps)
        CUDA.synchronize()

        profile = CUDA.@profile many_time_steps!(model, Δt, case.time_steps)
        kernels = kernel_times(profile.device)
        total_kernel_time = sum(k -> k.total_time_seconds, kernels; init = 0.0)

        push!(entries, (; case.name,
                        float_type = string(case.float_type),
                        case.advection,
                        case.closure,
                        case.dynamics,
                        case.microphysics,
                        backend = "vanilla",
                        grid_size = size(model.grid),
                        case.time_steps,
                        total_kernel_time_seconds = total_kernel_time,
                        kernel_count = length(kernels),
                        top_kernels = first(kernels, TOP_KERNELS),
                        metadata = BenchmarkMetadata(Oceananigans.Architectures.architecture(model.grid))))

        @info "  $(length(kernels)) kernels, $(@sprintf("%.3f", 1e3 * total_kernel_time / case.time_steps)) ms of GPU time per step"
        for k in first(kernels, TOP_KERNELS)
            @info @sprintf("    %-60s %8.3f ms/step  %5.1f%%  (%d launches, median %.3f ms)",
                           first(k.name, 60), 1e3 * k.total_time_seconds / case.time_steps,
                           100 * k.total_time_seconds / total_kernel_time, k.instances, 1e3 * k.median_time_seconds)
        end
    end

    return entries
end

#####
##### Per-kernel GPU times of the scalar tendency gradient (--mode scalar_tendency --ad)
#####

"""
    reactant_kernel_times(xplane_file)

Group the GPU kernels in a Reactant xprof trace by name, as `kernel_times` does for a
`CUDA.@profile` trace. The trace records each kernel's total, minimum and maximum time
but not its median, so `median_time_seconds` holds the mean.
"""
function reactant_kernel_times(xplane_file)
    reports = Reactant.Profiler.get_kernel_stats(xplane_file).reports
    grouped = Dict{String, Vector{eltype(reports)}}()
    for report in reports
        short_name = strip(first(split(report.name, r"[(<]")))
        push!(get!(grouped, short_name, eltype(reports)[]), report)
    end

    kernels = map(collect(grouped)) do (name, group)
        instances = sum(r -> Int(r.occurrences), group)
        total_time_seconds = sum(r -> r.total_duration_ns, group) / 1e9
        (name = String(name),
         instances,
         total_time_seconds,
         median_time_seconds = total_time_seconds / instances,
         min_time_seconds = minimum(r -> r.min_duration_ns, group) / 1e9,
         max_time_seconds = maximum(r -> r.max_duration_ns, group) / 1e9)
    end

    return sort!(kernels; by = k -> k.total_time_seconds, rev = true)
end

"""
    profile_tendency_kernels(cases)

Per-kernel GPU times of the gradient cases recorded by `run_benchmarks`, one entry per
case in the format of `profile_kernels`, with `time_steps` holding the number of
gradient evaluations. A vanilla case is profiled here with `CUDA.@profile` over its
`nrepeat` evaluations; a Reactant case is read from the xprof trace its timing run
wrote to `profile_dir`, so it is not run again.
"""
function profile_tendency_kernels(cases)
    entries = []
    for case in cases
        println("\n", "-" ^ 70)
        println("Profiling kernels: $(case.name)")
        println("-" ^ 70)

        kernels = if isnothing(case.profile_dir)
            case.run!()
            CUDA.synchronize()
            profile = CUDA.@profile for _ in 1:case.nrepeat
                case.run!()
            end
            kernel_times(profile.device)
        else
            reactant_kernel_times(Reactant.Profiler.find_xplane_file(case.profile_dir))
        end

        total_kernel_time = sum(k -> k.total_time_seconds, kernels; init = 0.0)

        push!(entries, (; case.name,
                        float_type = string(case.float_type),
                        case.advection,
                        closure = "none",
                        dynamics = "none",
                        microphysics = "none",
                        case.backend,
                        case.mode,
                        case.grid_size,
                        time_steps = case.nrepeat,
                        total_kernel_time_seconds = total_kernel_time,
                        kernel_count = length(kernels),
                        top_kernels = first(kernels, TOP_KERNELS),
                        metadata = BenchmarkMetadata(Oceananigans.Architectures.architecture(case.grid))))

        @info "  $(length(kernels)) kernels, $(@sprintf("%.3f", 1e3 * total_kernel_time / case.nrepeat)) ms of GPU time per gradient evaluation"
        for k in first(kernels, TOP_KERNELS)
            @info @sprintf("    %-60s %8.3f ms/eval  %5.1f%%  (%d launches, median %.3f ms)",
                           first(k.name, 60), 1e3 * k.total_time_seconds / case.nrepeat,
                           100 * k.total_time_seconds / total_kernel_time, k.instances, 1e3 * k.median_time_seconds)
        end
    end

    return entries
end

"""
    write_kernel_profiles(entries; output_file = "kernel_profile.json")

Write the kernel profiles to `output_file`, and a Markdown table of each to the same
name with a `.md` extension.
"""
function write_kernel_profiles(entries; output_file = "kernel_profile.json")
    open(output_file, "w") do io
        JSON.json(io, entries; pretty = true)
    end

    open(replace(output_file, ".json" => ".md"), "w") do io
        println(io, "## Kernel profile\n")
        for entry in entries
            gpu = something(entry.metadata.gpu_name, "GPU")
            per_step_ms = 1e3 * entry.total_kernel_time_seconds / entry.time_steps
            # A gradient case's "step" is one gradient evaluation
            gradient = get(entry, :mode, "") == "scalar_tendency_ad"
            unit, steps = gradient ? ("eval", "gradient evaluations") : ("step", "steps")
            println(io, "### $(entry.name) on $gpu\n")
            println(io, "$(entry.kernel_count) kernels, $(@sprintf("%.3f", per_step_ms)) ms of GPU time per $unit over $(entry.time_steps) $steps. Top $(length(entry.top_kernels)):\n")
            println(io, "| Kernel | ms/$unit | Share | Launches | Median (ms) | Min (ms) | Max (ms) |")
            println(io, "|--------|---------|-------|----------|-------------|----------|----------|")
            for k in entry.top_kernels
                println(io, @sprintf("| `%s` | %.3f | %.1f%% | %d | %.3f | %.3f | %.3f |",
                                     k.name, 1e3 * k.total_time_seconds / entry.time_steps,
                                     100 * k.total_time_seconds / entry.total_kernel_time_seconds, k.instances,
                                     1e3 * k.median_time_seconds, 1e3 * k.min_time_seconds, 1e3 * k.max_time_seconds))
            end
            println(io)
        end
    end

    println("\nKernel profiles saved to: $output_file")
    return nothing
end
