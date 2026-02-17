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
using BreezeBenchmarks: convective_boundary_layer, benchmark_time_stepping, run_benchmark_simulation
using JSON: JSON
using Oceananigans
using Oceananigans.TurbulenceClosures: SmagorinskyLilly, DynamicSmagorinsky

using Breeze
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization, ExplicitTimeStepping
using Breeze.Microphysics: NonEquilibriumCloudFormation

# Load CloudMicrophysics extension for OneMomentCloudMicrophysics
using CloudMicrophysics: CloudMicrophysics
const CMExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .CMExt: OneMomentCloudMicrophysics

using Printf: @printf
using Dates: DateTime, now, UTC

#####
##### Argument parsing
#####

function parse_commandline()
    s = ArgParseSettings(
        description = "Run Breeze.jl benchmarks with configurable parameters.",
        version = "0.1.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--mode"
            help = "Mode: 'benchmark' for quick performance tests, 'simulate' for full runs with output"
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
make_architecture(name) = (@eval $(Symbol(name)))()
make_float_type(name) = @eval $(Symbol(name))

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
function make_dynamics(name)
    if name == "anelastic"
        return nothing  # sentinel; convective_boundary_layer handles anelastic by default
    elseif name == "compressible_explicit"
        return CompressibleDynamics(ExplicitTimeStepping())
    elseif name == "compressible_splitexplicit"
        return CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=12))
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
        return SaturationAdjustment(; phase_equilibrium=MixedPhaseEquilibrium())
    elseif name == "WarmPhaseEquilibrium"
        return SaturationAdjustment(; phase_equilibrium=WarmPhaseEquilibrium())

    # 1M schemes with saturation adjustment (equilibrium cloud formation)
    elseif name == "1M_WarmEquilibrium"
        cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium(FT))
        return OneMomentCloudMicrophysics(FT; cloud_formation)
    elseif name == "1M_MixedEquilibrium"
        cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
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
    arch = make_architecture(args["device"])
    configuration = args["configuration"]

    # Parse lists from arguments
    sizes = [parse_size(s) for s in parse_list(args["size"])]
    float_types = [make_float_type(s) for s in parse_list(args["float_type"])]
    dynamics_names = parse_list(args["dynamics"])
    advections = parse_list(args["advection"])
    closures = parse_list(args["closure"])
    microphysics_schemes = parse_list(args["microphysics"])

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
    println("Architecture: ", arch)
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
    for ((Nx, Ny, Nz), FT, dyn_name, adv_name, cls_name, micro_name) in
            Iterators.product(sizes, float_types, dynamics_names, advections, closures, microphysics_schemes)

        # Build benchmark name
        size_str = "$(Nx)x$(Ny)x$(Nz)"
        ft_str = FT == Float32 ? "F32" : "F64"
        name = "CBL_$(size_str)_$(ft_str)_$(dyn_name)_$(adv_name)_$(cls_name)_$(micro_name)"

        println("\n", "-" ^ 70)
        println("Running: $name")
        println("-" ^ 70)

        # Create schemes
        dynamics = make_dynamics(dyn_name)
        advection = make_advection(adv_name, FT)
        closure = make_closure(cls_name, FT)
        microphysics = make_microphysics(micro_name, FT)

        # Create model based on configuration
        if configuration == "convective_boundary_layer"
            model = convective_boundary_layer(arch;
                Nx, Ny, Nz,
                float_type = FT,
                dynamics,
                advection = isnothing(advection) ? WENO(FT; order=5) : advection,
                closure = closure
            )
        else
            error("Unknown configuration: $configuration")
        end

        # Run based on mode
        result = if mode == "benchmark"
            benchmark_time_stepping(model; time_steps, Δt, warmup_steps, name, verbose=true)
        elseif mode == "simulate"
            run_benchmark_simulation(model;
                                     stop_time, Δt, output_interval, output_dir, name, verbose=true
            )
        else
            error("Unknown mode: $mode. Use 'benchmark' or 'simulate'.")
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

    @printf("%-50s %8s %12s %12s %10s %15s\n", "Benchmark", "Float", "Grid", "Time/Step", "Steps/s", "Points/s")
    println("-" ^ 105)

    for r in results
        grid_str = "$(r.grid_size[1])×$(r.grid_size[2])×$(r.grid_size[3])"
        @printf("%-50s %8s %12s %10.4f ms %10.2f %15.2e\n",
            r.name,
            r.float_type,
            grid_str,
            r.time_per_step_seconds * 1000,
            r.steps_per_second,
            r.grid_points_per_second
        )
    end

    println("=" ^ 105)

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
        println(io, "| Benchmark | Float | Grid | Time/Step (ms) | Steps/s | Points/s | Timestamp |")
        println(io, "|-----------|-------|------|----------------|---------|----------|-----------|")

        for entry in entries
            grid = entry["grid_size"]
            grid_str = "$(grid[1])×$(grid[2])×$(grid[3])"
            timestamp = entry["metadata"]["timestamp"]

            @printf(io, "| `%s` | %s | %s | %.2f | %.2f | %.2e | %s |\n",
                    entry["name"],
                    entry["float_type"],
                    grid_str,
                    entry["time_per_step_seconds"] * 1000,
                    entry["steps_per_second"],
                    entry["grid_points_per_second"],
                    timestamp)
        end
    end
end

# Run when invoked as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
