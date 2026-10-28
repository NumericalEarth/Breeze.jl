#####
##### Per-kernel GPU times of the vanilla time-stepping benchmark
#####
##### Builds the same convective boundary layer model as `run_benchmarks.jl`, from the
##### same command-line arguments, on the vanilla CUDA backend, and profiles
##### `--time_steps` time steps with CUDA.jl's integrated profiler (`CUDA.@profile`,
##### CUPTI underneath; no Nsight install needed). Each kernel's launches are grouped by
##### its name up to the first `(` or `<`, and the `TOP_KERNELS` kernels with the most
##### GPU time are written to `kernel_profile.json` in the working directory, with a
##### Markdown table in `kernel_profile.md`.
#####
##### Usage, from benchmarking/ (takes `run_benchmarks.jl`'s arguments; `--backend` is
##### ignored, the run is always vanilla):
#####   julia --project kernel_profiling/profile_kernels.jl --device GPU --size 128x128x32 \
#####       --dynamics compressible_splitexplicit --microphysics nothing --topology PBB \
#####       --grid_type latitude_longitude --simplified --float_type Float32
#####

# Argument parsing and the model's building blocks are `run_benchmarks.jl`'s, which
# only runs its `main` when invoked as a script
include(joinpath(@__DIR__, "..", "run_benchmarks.jl"))

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

function profile_kernels(args)
    args["device"] == "GPU" || error("Kernel profiling needs --device GPU (got $(args["device"]))")

    sizes = [parse_size(s) for s in parse_list(args["size"])]
    float_types = [make_float_type(s) for s in parse_list(args["float_type"])]
    dynamics_names = parse_list(args["dynamics"])
    advections = parse_list(args["advection"])
    closures = parse_list(args["closure"])
    microphysics_schemes = parse_list(args["microphysics"])
    topologies = parse_list(args["topology"])
    grid_type = isempty(args["grid_type"]) ? "rectilinear" : args["grid_type"]
    Δt = args["dt"]
    time_steps = args["time_steps"]
    warmup_steps = args["warmup_steps"]

    entries = []
    for (topo_name, (Nx, Ny, Nz), FT, dyn_name, adv_name, cls_name, micro_name) in
            Iterators.product(topologies, sizes, float_types, dynamics_names, advections, closures, microphysics_schemes)

        # The name of the matching vanilla result of `run_benchmarks.jl`
        size_str = "$(Nx)x$(Ny)x$(Nz)"
        ft_str = FT == Float32 ? "F32" : FT == Float64 ? "F64" : "BF16"
        grid_suffix = grid_type == "rectilinear" ? "" : "_LatLon"
        substeps_suffix = dyn_name == "compressible_splitexplicit" && args["substeps"] != 12 ? "_N$(args["substeps"])" : ""
        name = "CBL_$(size_str)_$(ft_str)_$(dyn_name)$(substeps_suffix)_$(adv_name)_$(cls_name)_$(micro_name)_$(topo_name)$(grid_suffix)_vanilla"

        println("\n", "-" ^ 70)
        println("Profiling kernels: $name")
        println("-" ^ 70)

        arch = make_backend_arch("vanilla", "GPU")
        model = convective_boundary_layer(arch;
                                          Nx, Ny, Nz,
                                          float_type = FT,
                                          dynamics = make_dynamics(dyn_name, FT; substeps = args["substeps"]),
                                          advection = make_advection(adv_name, FT),
                                          closure = make_closure(cls_name, FT),
                                          microphysics = make_microphysics(micro_name, FT),
                                          topology = make_topology(topo_name),
                                          grid_type = Symbol(grid_type),
                                          simplified = args["simplified"])

        # Compile every kernel before profiling, so the trace holds only steady-state launches
        Δt_FT = FT(Δt)
        many_time_steps!(model, Δt_FT, warmup_steps)
        CUDA.synchronize()

        profile = CUDA.@profile many_time_steps!(model, Δt_FT, time_steps)
        kernels = kernel_times(profile.device)
        total_kernel_time = sum(k -> k.total_time_seconds, kernels; init = 0.0)

        metadata = BenchmarkMetadata(arch)
        push!(entries, (; name,
                        float_type = string(FT),
                        advection = adv_name,
                        closure = cls_name,
                        dynamics = dyn_name,
                        microphysics = micro_name,
                        backend = "vanilla",
                        grid_size = (Nx, Ny, Nz),
                        time_steps,
                        total_kernel_time_seconds = total_kernel_time,
                        kernel_count = length(kernels),
                        top_kernels = first(kernels, TOP_KERNELS),
                        metadata))

        @info "  $(length(kernels)) kernels, $(@sprintf("%.3f", 1e3 * total_kernel_time / time_steps)) ms of GPU time per step"
        for k in first(kernels, TOP_KERNELS)
            @info @sprintf("    %-60s %8.3f ms/step  %5.1f%%  (%d launches, median %.3f ms)",
                           first(k.name, 60), 1e3 * k.total_time_seconds / time_steps,
                           100 * k.total_time_seconds / total_kernel_time, k.instances, 1e3 * k.median_time_seconds)
        end
    end

    return entries
end

function write_kernel_report(filename, entries)
    open(filename, "w") do io
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
    return nothing
end

function main_kernels()
    args = parse_commandline()
    entries = profile_kernels(args)

    output_file = "kernel_profile.json"
    open(output_file, "w") do io
        JSON.json(io, entries; pretty = true)
    end
    write_kernel_report(replace(output_file, ".json" => ".md"), entries)
    println("\nKernel profile saved to: $output_file")

    return entries
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_kernels()
end
