#####
##### Breeze.jl Benchmark Script
#####
##### This script runs the convective boundary layer benchmark case
##### with configurable parameters via command-line arguments.
##### Default device is GPU.
#####
##### Modes:
#####   - benchmark: Quick performance benchmarks (default)
#####   - simulate: Full runs with output for validation
#####
##### Usage (benchmark mode):
#####   julia --project run_benchmarks.jl                          # Default: 64³, GPU, Float32, WENO5
#####   julia --project run_benchmarks.jl --size=128^3             # 128³ grid on GPU
#####   julia --project run_benchmarks.jl --size="64^3, 128^3"     # Multiple sizes
#####
##### Usage (simulate mode):
#####   julia --project run_benchmarks.jl --mode=simulate --size=128^3 --stop_time=2.0
#####   julia --project run_benchmarks.jl --mode=simulate --size=256^3 --stop_time=1.0 --output_interval=5
#####

using ArgParse: @add_arg_table!, ArgParseSettings, parse_args
using BreezeBenchmarks: convective_boundary_layer, benchmark_time_stepping, run_benchmark_simulation, BenchmarkResult
using BreezeBenchmarks: scalar_tendency_problem, benchmark_scalar_tendency
using JSON: JSON
using Oceananigans
using Oceananigans.TurbulenceClosures: SmagorinskyLilly, DynamicSmagorinsky

using Breeze
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization, ExplicitTimeStepping
using Breeze.Microphysics: NonEquilibriumCloudFormation

# Reactant must be loaded before CUDA so that ReactantCUDAExt activates with
# CUDA's kernel compilation hooks correctly registered.
using Reactant: Reactant
using Oceananigans.Architectures: ReactantState
using CUDA: CUDABackend
using AMDGPU: ROCBackend
using Metal: MetalBackend

# Load CloudMicrophysics extension for OneMomentCloudMicrophysics
using CloudMicrophysics: CloudMicrophysics
const CMExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .CMExt: OneMomentCloudMicrophysics

using Printf: @printf
using Dates: DateTime, now, UTC

#####
##### Argument parsing
#####

# Helper functions to refer to the various GPU backends
CUDAGPU() = GPU(CUDABackend())
MetalGPU() = GPU(MetalBackend())
ROCGPU() = GPU(ROCBackend())

function parse_commandline()
    s = ArgParseSettings(
        description = "Run Breeze.jl benchmarks with configurable parameters.",
        version = "0.1.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--mode"
            help = "Mode: 'benchmark' for quick performance tests, 'simulate' for full runs with output, " *
                   "'tendency' to profile a single scalar-tendency WENO kernel (no model; Reactant only)"
            arg_type = String
            default = "benchmark"

        "--size"
            help = "Grid size as NxxNyxNz (e.g., 128x128x128) or N^3 for cubic (e.g., 64^3). " *
                   "Multiple sizes can be specified as comma-separated list."
            arg_type = String
            default = "64^3"

        "--device"
            help = "Device to run on: CPU or GPU"
            arg_type = String
            default = "GPU"

        "--configuration"
            help = "Benchmark configuration: convective_boundary_layer"
            arg_type = String
            default = "convective_boundary_layer"

        "--float_type"
            help = "Floating point type: Float32 or Float64. " *
                   "Multiple types can be specified as comma-separated list."
            arg_type = String
            default = "Float32"

        "--advection"
            help = "Advection scheme: nothing, Centered2, WENO5, WENO9, bounded_WENO5. " *
                   "Multiple schemes can be specified as comma-separated list."
            arg_type = String
            default = "WENO5"

        "--microphysics"
            help = "Microphysics scheme: nothing, SaturationAdjustment, " *
                   "MixedPhaseEquilibrium, WarmPhaseEquilibrium, " *
                   "1M_WarmEquilibrium, 1M_MixedEquilibrium, " *
                   "1M_WarmNonEquilibrium, 1M_MixedNonEquilibrium. " *
                   "Multiple schemes can be specified as comma-separated list."
            arg_type = String
            default = "nothing"

        "--dynamics"
            help = "Dynamics formulation: anelastic, compressible_explicit, compressible_splitexplicit. " *
                   "Multiple can be specified as comma-separated list."
            arg_type = String
            default = "anelastic"

        "--closure"
            help = "Turbulence closure: nothing, SmagorinskyLilly, DynamicSmagorinsky. " *
                   "Multiple closures can be specified as comma-separated list."
            arg_type = String
            default = "nothing"

        "--backend"
            help = "Execution backend: vanilla (eager KA / CUDA) or reactant (compiled via Reactant). " *
                   "Multiple backends can be specified as comma-separated list."
            arg_type = String
            default = "vanilla"

        "--topology"
            help = "Grid topology: PPB (Periodic, Periodic, Bounded) or PBB (Periodic, Bounded, Bounded). " *
                   "Multiple topologies can be specified as comma-separated list."
            arg_type = String
            default = "PPB"

        "--simplified"
            help = "Omit the geostrophic body forcing and field-dependent surface drag BCs " *
                   "from the convective_boundary_layer case. Required for Reactant runs since " *
                   "those forcings do not currently materialize on ReactantState."
            action = :store_true

        "--ad"
            help = "Benchmark the forward+backward (Enzyme reverse-mode) pass through the " *
                   "stepping loop instead of forward only. Reactant backend only."
            action = :store_true

        "--time_steps"
            help = "Number of time steps (benchmark mode only)"
            arg_type = Int
            default = 100

        "--warmup_steps"
            help = "Number of warmup time steps (benchmark mode only)"
            arg_type = Int
            default = 10

        "--dt"
            help = "Time step size in seconds"
            arg_type = Float64
            default = 0.5

        "--stop_time"
            help = "Simulation stop time in hours (simulate mode only)"
            arg_type = Float64
            default = 2.0

        "--output_interval"
            help = "Output interval in minutes (simulate mode only)"
            arg_type = Float64
            default = 10.0

        "--output"
            help = "Output JSON filename for benchmark results"
            arg_type = String
            default = "benchmark_results.json"

        "--output_dir"
            help = "Directory for simulation output files (simulate mode only)"
            arg_type = String
            default = "."

        "--clear"
            help = "Clear existing results file before writing"
            action = :store_true
    end

    return parse_args(s)
end

#####
##### Parsing utilities for comma-separated lists
#####

"""
    parse_list(str)

Parse a comma-separated string into a vector of trimmed strings.
"""
function parse_list(str::AbstractString)
    return [strip(s) for s in split(str, ",")]
end

"""
    parse_size(size_str)

Parse a size string into a tuple (Nx, Ny, Nz).
Formats: "128x128x128" or "64^3" (cubic shorthand).
"""
function parse_size(size_str)
    if occursin("^3", size_str)
        N = parse(Int, replace(size_str, "^3" => ""))
        return (N, N, N)
    end
    parts = split(size_str, "x")
    length(parts) == 3 || error("Invalid size: $size_str. Use NxNyxNz or N^3.")
    return Tuple(parse(Int, p) for p in parts)
end

#####
##### Factory functions to create schemes from names
#####
##### Uses @eval with Symbol to convert strings like "CPU" -> CPU()
#####

# Simple constructors: "CPU" -> CPU(), "Float32" -> Float32
make_architecture(name) = eval(Meta.parse(name))()
make_float_type(name) = @eval $(Symbol(name))

"""
    make_backend_arch(backend, device)

Return the architecture instance to build the model on, given a backend name
("vanilla" or "reactant") and a device name ("CPU" or "GPU"). For Reactant we
also set the default backend so XLA targets the requested device.
"""
function make_backend_arch(backend, device)
    if backend == "vanilla"
        return make_architecture(device)
    elseif backend == "reactant"
        Reactant.set_default_backend(device == "GPU" ? "gpu" : "cpu")
        return ReactantState()
    else
        error("Unknown backend: $backend. Use 'vanilla' or 'reactant'.")
    end
end

"""
    make_topology(name)

Map a short topology label to an Oceananigans topology tuple.
"PPB" -> (Periodic, Periodic, Bounded); "PBB" -> (Periodic, Bounded, Bounded).
"""
function make_topology(name)
    if name == "PPB"
        return (Periodic, Periodic, Bounded)
    elseif name == "PBB"
        return (Periodic, Bounded, Bounded)
    else
        error("Unknown topology: $name. Use 'PPB' or 'PBB'.")
    end
end

# Advection: parse "WENO5" -> WENO(FT; order=5), "Centered2" -> Centered(FT; order=2)
function make_advection(name, FT)
    name == "nothing" && return nothing
    name == "bounded_WENO5" && return WENO(FT; order=5, bounds=(0, Inf))

    # Parse scheme name and order from strings like "WENO5", "Centered2"
    m = match(r"^([A-Za-z]+)(\d+)$", name)
    isnothing(m) && error("Unknown advection: $name. Use WENO5, WENO9, Centered2, or bounded_WENO5.")
    scheme = @eval $(Symbol(m[1]))
    order = parse(Int, m[2])
    return scheme(FT; order)
end

# Closures: "SmagorinskyLilly" -> SmagorinskyLilly(FT)
make_closure(name, FT) = name == "nothing" ? nothing : (@eval $(Symbol(name)))(FT)

# Dynamics: "anelastic", "compressible_explicit", "compressible_splitexplicit"
function make_dynamics(name::AbstractString, FT::Type)
    if name == "anelastic"
        return nothing  # sentinel; convective_boundary_layer handles anelastic by default
    elseif name == "compressible_explicit"
        return CompressibleDynamics(ExplicitTimeStepping())
    elseif name == "compressible_splitexplicit"
        return CompressibleDynamics(SplitExplicitTimeDiscretization(FT; substeps=12))
    else
        error("Unknown dynamics: $name. Use anelastic, compressible_explicit, or compressible_splitexplicit.")
    end
end

# Microphysics: supports 0M saturation adjustment and 1M bulk schemes
function make_microphysics(name, FT=Float32)
    name == "nothing" && return nothing

    # 0M Saturation adjustment schemes
    if name == "SaturationAdjustment"
        return SaturationAdjustment()
    elseif name == "MixedPhaseEquilibrium"
        return SaturationAdjustment(; equilibrium=MixedPhaseEquilibrium())
    elseif name == "WarmPhaseEquilibrium"
        return SaturationAdjustment(; equilibrium=WarmPhaseEquilibrium())

    # 1M schemes with saturation adjustment (equilibrium cloud formation)
    elseif name == "1M_WarmEquilibrium"
        cloud_formation = SaturationAdjustment(; equilibrium=WarmPhaseEquilibrium())
        return OneMomentCloudMicrophysics(FT; cloud_formation)
    elseif name == "1M_MixedEquilibrium"
        cloud_formation = SaturationAdjustment(; equilibrium=MixedPhaseEquilibrium())
        return OneMomentCloudMicrophysics(FT; cloud_formation)

    # 1M schemes with non-equilibrium cloud formation (prognostic cloud condensate)
    elseif name == "1M_WarmNonEquilibrium"
        cloud_formation = NonEquilibriumCloudFormation(nothing, nothing)  # warm-phase only
        return OneMomentCloudMicrophysics(FT; cloud_formation)
    elseif name == "1M_MixedNonEquilibrium"
        cloud_formation = NonEquilibriumCloudFormation(nothing, nothing)  # will get ice from categories
        # For mixed-phase, we need to specify ice formation
        categories = CMExt.one_moment_cloud_microphysics_categories(FT)
        cloud_formation = NonEquilibriumCloudFormation(nothing, categories.cloud_ice)
        return OneMomentCloudMicrophysics(FT; cloud_formation, categories)
    else
        error("Unknown microphysics: $name")
    end
end

#####
##### Main benchmarking logic
#####

function run_benchmarks(args)
    mode = args["mode"]
    device = args["device"]
    configuration = args["configuration"]

    # Parse lists from arguments
    sizes = [parse_size(s) for s in parse_list(args["size"])]
    float_types = [make_float_type(s) for s in parse_list(args["float_type"])]
    dynamics_names = parse_list(args["dynamics"])
    advections = parse_list(args["advection"])
    closures = parse_list(args["closure"])
    microphysics_schemes = parse_list(args["microphysics"])
    backends = parse_list(args["backend"])
    topologies = parse_list(args["topology"])

    # Mode-specific parameters
    Δt = args["dt"]
    time_steps = args["time_steps"]
    warmup_steps = args["warmup_steps"]
    stop_time = args["stop_time"] * 3600  # Convert hours to seconds
    output_interval = args["output_interval"] * 60  # Convert minutes to seconds
    output_dir = args["output_dir"]

    results = []

    println("=" ^ 95)
    println("Breeze.jl Benchmark Suite")
    println("=" ^ 95)
    println("Date: ", now(UTC))
    println("Mode: ", mode)
    println("Device: ", device)
    println("Backends: ", backends)
    println("Topologies: ", topologies)
    println("Sizes: ", sizes)
    println("Float types: ", float_types)
    println("Dynamics: ", dynamics_names)
    println("Advection schemes: ", advections)
    println("Closures: ", closures)
    println("Microphysics: ", microphysics_schemes)
    if mode == "benchmark"
        println("Time steps: ", time_steps, " (warmup: ", warmup_steps, ")")
    else
        println("Stop time: ", args["stop_time"], " hours")
        println("Output interval: ", args["output_interval"], " minutes")
    end
    println("Δt: ", Δt, " s")
    println("=" ^ 95)
    println()

    # Loop over all combinations using Iterators.product
    for (backend_name, topo_name, (Nx, Ny, Nz), FT, dyn_name, adv_name, cls_name, micro_name) in
            Iterators.product(backends, topologies, sizes, float_types, dynamics_names, advections, closures, microphysics_schemes)

        # Set floating point precision so constructors pick up the right default
        Oceananigans.defaults.FloatType = FT

        # Build benchmark name
        size_str = "$(Nx)x$(Ny)x$(Nz)"
        ft_str = FT == Float32 ? "F32" : "F64"
        mode_suffix = args["ad"] ? "_AD" : ""

        # Tendency mode profiles the bare scalar-tendency WENO kernel with no
        # model (hence no dynamics/closure/microphysics). Reactant only, since
        # it compiles the launch into a single XLA program. Handle it up front
        # and skip the model-building path below.
        if mode == "tendency"
            # Dump all MLIR (every compile stage) for the tendency compile so it
            # can be saved as a CI artifact for future inspection (GB-25 pattern).
            if get(ENV, "GITHUB_ACTIONS", "false") == "true"
                Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mkpath(joinpath(@__DIR__, "mlir_dumps"))
                Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
            end

            backend_name == "reactant" ||
                error("tendency mode requires --backend reactant (got $backend_name)")
            m = match(r"^WENO(\d+)$", adv_name)
            isnothing(m) && error("tendency mode supports WENO<order> advection, got $adv_name")
            order = parse(Int, m[1])

            arch = make_backend_arch(backend_name, device)
            topology = make_topology(topo_name)

            # Compile + profile the same kernel both with and without XLA's
            # optimization passes, so we can compare optimized vs unoptimized
            # runtime and MLIR (both dump under DUMP_MLIR_ALWAYS above).
            for optimize in (true, false)
                opt_tag = optimize ? "opt" : "noopt"
                name = "ScalarTendency_$(size_str)_$(ft_str)_$(adv_name)_$(opt_tag)_$(topo_name)_$(backend_name)"
                println("\n", "-" ^ 70)
                println("Running: $name")
                println("-" ^ 70)

                tendency!, tendency_args = scalar_tendency_problem(arch;
                                                                  Nx, Ny, Nz,
                                                                  order,
                                                                  float_type = FT,
                                                                  topology)
                push!(results, benchmark_scalar_tendency(tendency!, tendency_args;
                                                         nrepeat = time_steps,
                                                         name,
                                                         advection = adv_name,
                                                         optimize))
            end
            continue
        end

        name = "CBL_$(size_str)_$(ft_str)_$(dyn_name)_$(adv_name)_$(cls_name)_$(micro_name)_$(topo_name)_$(backend_name)$(mode_suffix)"

        println("\n", "-" ^ 70)
        println("Running: $name")
        println("-" ^ 70)

        # Create arch (backend-aware) and schemes
        arch = make_backend_arch(backend_name, device)
        topology = make_topology(topo_name)
        dynamics = make_dynamics(dyn_name, FT)
        advection = make_advection(adv_name, FT)
        closure = make_closure(cls_name, FT)
        microphysics = make_microphysics(micro_name, FT)

        # Create model based on configuration
        model = if configuration == "convective_boundary_layer"
            convective_boundary_layer(arch;
                                      Nx, Ny, Nz,
                                      float_type = FT,
                                      dynamics,
                                      advection,
                                      closure,
                                      microphysics,
                                      topology,
                                      simplified = args["simplified"],
                                      )
        else
            error("Unknown configuration: $configuration")
        end

        # Run based on mode
        result = if mode == "benchmark"
            benchmark_time_stepping(model;
                                    time_steps,
                                    Δt,
                                    warmup_steps,
                                    name,
                                    verbose=true,
                                    advection=adv_name,
                                    closure=cls_name,
                                    dynamics=dyn_name,
                                    microphysics=micro_name,
                                    backend=backend_name,
                                    ad=args["ad"],
                                    )
        elseif mode == "simulate"
            run_benchmark_simulation(model;
                                     stop_time,
                                     Δt,
                                     output_interval,
                                     output_dir,
                                     name,
                                     verbose=true,
                                     advection=adv_name,
                                     closure=cls_name,
                                     dynamics=dyn_name,
                                     microphysics=micro_name,
                                     )
        else
            error("Unknown mode: $mode. Use 'benchmark', 'simulate', or 'tendency'.")
        end
        push!(results, result)
    end

    return results
end

#####
##### Main entry point
#####

function main()
    args = parse_commandline()
    results = run_benchmarks(args)

    #####
    ##### Summary table
    #####

    println("\n", "=" ^ 105)
    println("BENCHMARK SUMMARY")
    println("=" ^ 105)
    println()

    @printf("%-60s %8s %10s %14s %12s %10s %15s\n",
            "Benchmark", "Float", "Backend", "Grid", "Time/Step", "Steps/s", "Points/s")
    println("-" ^ 130)

    for r in results
        grid_str = "$(r.grid_size[1])×$(r.grid_size[2])×$(r.grid_size[3])"
        backend_str = r isa BenchmarkResult ? r.backend : ""
        @printf("%-60s %8s %10s %14s %10.4f ms %10.2f %15.2e\n",
            r.name,
            r.float_type,
            backend_str,
            grid_str,
            r.time_per_step_seconds * 1000,
            r.steps_per_second,
            r.grid_points_per_second
        )
    end

    println("=" ^ 130)

    #####
    ##### Save results to JSON
    #####

    if !isempty(results)
        output_file = args["output"]
        clear_file = args["clear"]

        # Load existing results or start fresh
        all_entries = if clear_file || !isfile(output_file)
            if clear_file && isfile(output_file)
                println("\nCleared existing results file: $output_file")
            end
            results
        else
            # Read existing file and append
            existing_data = JSON.parse(read(output_file))
            println("\nAppending to existing results file: $output_file")
            vcat(existing_data, results)
        end

        # Write all results to JSON
        open(output_file, "w") do io
            JSON.json(io, all_entries; pretty=true)
        end

        println("Results saved to: $output_file ($(length(results)) new, $(length(all_entries)) total)")

        # Generate markdown report from the full JSON data
        md_file = replace(output_file, ".json" => ".md")
        generate_markdown_report(md_file, JSON.parse(read(output_file)))
        println("Markdown report saved to: $md_file")
    end

    println("Benchmarks completed at ", now(UTC), "Z")
end

"""
Generate a markdown report from benchmark results.
"""
function generate_markdown_report(filename, entries)
    open(filename, "w") do io
        println(io, "# Breeze.jl Benchmark Results")
        println(io)

        # Get metadata from the most recent entry
        if !isempty(entries)
            metadata = entries[end]["metadata"]

            println(io, "## System Information")
            println(io)
            println(io, "| Property | Value |")
            println(io, "|----------|-------|")
            println(io, "| Julia | ", metadata["julia_version"], " |")
            println(io, "| Oceananigans | ", metadata["oceananigans_version"], " |")
            println(io, "| Breeze | ", metadata["breeze_version"], " |")
            println(io, "| Architecture | ", metadata["architecture"], " |")
            println(io, "| CPU | ", metadata["cpu_model"], " |")
            println(io, "| Threads | ", metadata["num_threads"], " |")
            if !isnothing(metadata["gpu_name"])
                println(io, "| GPU | ", metadata["gpu_name"], " |")
                println(io, "| CUDA | ", metadata["cuda_version"], " |")
            end
            println(io, "| Hostname | ", metadata["hostname"], " |")
            println(io)
        end

        println(io, "## Results")
        println(io)
        println(io, "| Benchmark | Float | Backend | Grid | Time/Step (ms) | Steps/s | Points/s | Compile (s) | Timestamp |")
        println(io, "|-----------|-------|---------|------|----------------|---------|----------|-------------|-----------|")

        for entry in entries
            grid = entry["grid_size"]
            grid_str = "$(grid[1])×$(grid[2])×$(grid[3])"
            timestamp = entry["metadata"]["timestamp"]
            backend = get(entry, "backend", "")
            compile_time = get(entry, "compile_time_seconds", 0.0)

            @printf(io, "| `%s` | %s | %s | %s | %.2f | %.2f | %.2e | %.3f | %s |\n",
                    entry["name"],
                    entry["float_type"],
                    backend,
                    grid_str,
                    entry["time_per_step_seconds"] * 1000,
                    entry["steps_per_second"],
                    entry["grid_points_per_second"],
                    compile_time,
                    timestamp)
        end
    end
end

# Run when invoked as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
