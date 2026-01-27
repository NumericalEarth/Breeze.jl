#####
##### Breeze.jl Benchmark Script
#####
##### This script runs the convective boundary layer benchmark case
##### with configurable parameters via command-line arguments.
#####
##### Usage:
#####   julia --project run_benchmarks.jl --size=16x16x16
#####   julia --project run_benchmarks.jl --size=128x128x128 --device=GPU --float_type=Float32
#####   julia --project run_benchmarks.jl --size="64x64x64, 128x128x64" --advection="WENO5, WENO9"
#####

using Pkg
Pkg.activate(@__DIR__)

using ArgParse
using BreezeBenchmarks
using JSON3
using Oceananigans
using Oceananigans.TurbulenceClosures: SmagorinskyLilly, DynamicSmagorinsky

using Breeze

using Printf
using Dates

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
        "--size"
            help = "Grid size as NxxNyxNz (e.g., 128x128x128) or N^3 for cubic (e.g., 64^3). " *
                   "Multiple sizes can be specified as comma-separated list."
            arg_type = String
            default = "64^3"

        "--device"
            help = "Device to run on: CPU or GPU"
            arg_type = String
            default = "CPU"

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
                   "MixedPhaseEquilibrium, WarmPhaseEquilibrium, BulkMicrophysics. " *
                   "Multiple schemes can be specified as comma-separated list."
            arg_type = String
            default = "nothing"

        "--closure"
            help = "Turbulence closure: nothing, SmagorinskyLilly, DynamicSmagorinsky. " *
                   "Multiple closures can be specified as comma-separated list."
            arg_type = String
            default = "nothing"

        "--time_steps"
            help = "Number of time steps to benchmark"
            arg_type = Int
            default = 100

        "--warmup_steps"
            help = "Number of warmup time steps"
            arg_type = Int
            default = 10

        "--dt"
            help = "Time step size in seconds"
            arg_type = Float64
            default = 0.05

        "--output"
            help = "Output JSON filename for benchmark results"
            arg_type = String
            default = "benchmark_results.json"

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

Supported formats:
- "128x128x128" - explicit dimensions
- "64^3" - shorthand for cubic grid (64x64x64)
"""
function parse_size(size_str::AbstractString)
    # Check for cubic shorthand: N^3
    if occursin("^3", size_str)
        n_str = replace(size_str, "^3" => "")
        N = parse(Int, n_str)
        return (N, N, N)
    end

    # Standard format: NxxNyxNz
    parts = split(size_str, "x")
    if length(parts) != 3
        error("Invalid size format: $size_str. Expected NxxNyxNz (e.g., 128x128x128) or N^3 (e.g., 64^3)")
    end
    return (parse(Int, parts[1]), parse(Int, parts[2]), parse(Int, parts[3]))
end

#####
##### Factory functions to create schemes from names
#####

function make_architecture(name::AbstractString)
    if name == "CPU"
        return CPU()
    elseif name == "GPU"
        return GPU()
    else
        error("Unknown device: $name. Use CPU or GPU.")
    end
end

function make_float_type(name::AbstractString)
    if name == "Float32"
        return Float32
    elseif name == "Float64"
        return Float64
    else
        error("Unknown float type: $name. Use Float32 or Float64.")
    end
end

function make_advection(name::AbstractString, FT)
    if name == "nothing"
        return nothing
    elseif name == "Centered2"
        return Centered(FT; order=2)
    elseif name == "WENO5"
        return WENO(FT; order=5)
    elseif name == "WENO9"
        return WENO(FT; order=9)
    elseif name == "bounded_WENO5"
        return WENO(FT; order=5, bounds=(0, Inf))
    else
        error("Unknown advection scheme: $name. " *
              "Use nothing, Centered2, WENO5, WENO9, or bounded_WENO5.")
    end
end

function make_closure(name::AbstractString, FT)
    if name == "nothing"
        return nothing
    elseif name == "SmagorinskyLilly"
        return SmagorinskyLilly(FT)
    elseif name == "DynamicSmagorinsky"
        return DynamicSmagorinsky(FT)
    else
        error("Unknown closure: $name. " *
              "Use nothing, SmagorinskyLilly, or DynamicSmagorinsky.")
    end
end

function make_microphysics(name::AbstractString)
    if name == "nothing"
        return nothing
    elseif name == "SaturationAdjustment"
        return SaturationAdjustment()
    elseif name == "MixedPhaseEquilibrium"
        return SaturationAdjustment(; phase_equilibrium=MixedPhaseEquilibrium())
    elseif name == "WarmPhaseEquilibrium"
        return SaturationAdjustment(; phase_equilibrium=WarmPhaseEquilibrium())
    elseif name == "BulkMicrophysics"
        return BulkMicrophysics()
    else
        error("Unknown microphysics: $name. " *
              "Use nothing, SaturationAdjustment, MixedPhaseEquilibrium, " *
              "WarmPhaseEquilibrium, or BulkMicrophysics.")
    end
end

#####
##### Main benchmarking logic
#####

function run_benchmarks(args)
    # Parse device (single value)
    arch = make_architecture(args["device"])

    # Parse lists from arguments
    sizes = [parse_size(s) for s in parse_list(args["size"])]
    float_types = [make_float_type(s) for s in parse_list(args["float_type"])]
    advections = parse_list(args["advection"])
    closures = parse_list(args["closure"])
    microphysics_schemes = parse_list(args["microphysics"])

    # Benchmark parameters
    time_steps = args["time_steps"]
    warmup_steps = args["warmup_steps"]
    Δt = args["dt"]

    configuration = args["configuration"]

    results = []

    println("=" ^ 95)
    println("Breeze.jl Benchmark Suite")
    println("=" ^ 95)
    println("Date: ", now())
    println("Architecture: ", arch)
    println("Sizes: ", sizes)
    println("Float types: ", float_types)
    println("Advection schemes: ", advections)
    println("Closures: ", closures)
    println("Microphysics: ", microphysics_schemes)
    println("Time steps: ", time_steps)
    println("Δt: ", Δt, " s")
    println("=" ^ 95)
    println()

    # Loop over all combinations
    for (Nx, Ny, Nz) in sizes
        for FT in float_types
            for adv_name in advections
                for cls_name in closures
                    for micro_name in microphysics_schemes
                        # Build benchmark name
                        size_str = "$(Nx)x$(Ny)x$(Nz)"
                        ft_str = FT == Float32 ? "F32" : "F64"
                        name = "CBL_$(size_str)_$(ft_str)_$(adv_name)_$(cls_name)_$(micro_name)"

                        println("\n", "-" ^ 70)
                        println("Running: $name")
                        println("-" ^ 70)

                        # Create schemes
                        advection = make_advection(adv_name, FT)
                        closure = make_closure(cls_name, FT)
                        microphysics = make_microphysics(micro_name)

                        # Create model based on configuration
                        if configuration == "convective_boundary_layer"
                            model = convective_boundary_layer(arch;
                                Nx, Ny, Nz,
                                float_type = FT,
                                advection = isnothing(advection) ? WENO(FT; order=5) : advection,
                                closure = closure
                            )
                        else
                            error("Unknown configuration: $configuration")
                        end

                        # Run benchmark
                        result = benchmark_time_stepping(model;
                            time_steps, Δt, warmup_steps, name, verbose = true
                        )
                        push!(results, result)
                    end
                end
            end
        end
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
        steps_per_second = 1.0 / r.time_per_step_seconds
        @printf("%-50s %8s %12s %10.4f ms %10.2f %15.2e\n",
            r.name,
            r.float_type,
            grid_str,
            r.time_per_step_seconds * 1000,
            steps_per_second,
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

        # Convert results to JSON-serializable format
        new_entries = [result_to_dict(r) for r in results]

        # Load existing results or start fresh
        if clear_file || !isfile(output_file)
            all_entries = new_entries
            if clear_file && isfile(output_file)
                println("\nCleared existing results file: $output_file")
            end
        else
            # Read existing file and append
            existing_data = open(output_file, "r") do io
                JSON3.read(io)
            end
            all_entries = vcat(existing_data, new_entries)
            println("\nAppending to existing results file: $output_file")
        end

        # Write all results to JSON
        open(output_file, "w") do io
            JSON3.pretty(io, all_entries)
        end

        println("Results saved to: $output_file ($(length(new_entries)) new, $(length(all_entries)) total)")

        # Generate markdown report from the full JSON data
        md_file = replace(output_file, ".json" => ".md")
        generate_markdown_report(md_file, all_entries)
        println("Markdown report saved to: $md_file")
    end

    println("Benchmarks completed at ", now())
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
            # Extract just the date portion for brevity
            date_str = split(timestamp, "T")[1]

            @printf(io, "| %s | %s | %s | %.2f | %.2f | %.2e | %s |\n",
                entry["name"],
                entry["float_type"],
                grid_str,
                entry["time_per_step_seconds"] * 1000,
                entry["steps_per_second"],
                entry["grid_points_per_second"],
                date_str)
        end
    end
end

"""
Convert a BenchmarkResult to a JSON-serializable dictionary.
"""
function result_to_dict(r)
    return Dict(
        "name" => r.name,
        "float_type" => r.float_type,
        "grid_size" => collect(r.grid_size),
        "time_steps" => r.time_steps,
        "dt" => r.Δt,
        "total_time_seconds" => r.total_time_seconds,
        "time_per_step_seconds" => r.time_per_step_seconds,
        "steps_per_second" => 1.0 / r.time_per_step_seconds,
        "grid_points_per_second" => r.grid_points_per_second,
        "metadata" => Dict(
            "julia_version" => r.metadata.julia_version,
            "oceananigans_version" => r.metadata.oceananigans_version,
            "breeze_version" => r.metadata.breeze_version,
            "architecture" => r.metadata.architecture,
            "gpu_name" => r.metadata.gpu_name,
            "cuda_version" => r.metadata.cuda_version,
            "cpu_model" => r.metadata.cpu_model,
            "num_threads" => r.metadata.num_threads,
            "hostname" => r.metadata.hostname,
            "timestamp" => string(r.metadata.timestamp)
        )
    )
end

# Run when invoked as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
