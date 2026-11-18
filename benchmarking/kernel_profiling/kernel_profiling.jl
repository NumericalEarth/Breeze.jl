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
        println(io, "## Kernel profile (vanilla)\n")
        for entry in entries
            gpu = something(entry.metadata.gpu_name, "GPU")
            per_step_ms = 1e3 * entry.total_kernel_time_seconds / entry.time_steps
            println(io, "### $(entry.name) on $gpu\n")
            println(io, "$(entry.kernel_count) kernels, $(@sprintf("%.3f", per_step_ms)) ms of GPU time per step over $(entry.time_steps) steps. Top $(length(entry.top_kernels)):\n")
            println(io, "| Kernel | ms/step | Share | Launches | Median (ms) | Min (ms) | Max (ms) |")
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
