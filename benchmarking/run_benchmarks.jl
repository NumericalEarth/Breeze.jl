#####
##### Breeze.jl Benchmark Script
#####
##### This script runs the convective boundary layer benchmark case
##### with a canonical configuration and systematic variations.
#####

using Pkg
Pkg.activate(@__DIR__)

using BreezeBenchmarks
using Oceananigans
using Oceananigans.TurbulenceClosures: SmagorinskyLilly, DynamicSmagorinsky

using Printf
using Dates

#####
##### Configuration
#####

# Set architecture: CPU() for testing, GPU() for production benchmarks
arch = CPU()

# Resolutions to benchmark
resolutions = [:small, :medium]  # Add :large, :production for comprehensive benchmarks

# Canonical configuration
canonical_float_type = Float32
canonical_advection = :WENO5
canonical_closure = :Nothing

# Benchmark parameters
time_steps = 100
Δt = 0.05  # seconds (from FastEddy paper)
warmup_steps = 10

#####
##### Helper functions to create schemes with correct float type
#####

function make_advection(name::Symbol, FT)
    if name == :Centered2
        return Centered(FT; order=2)
    elseif name == :WENO5
        return WENO(FT; order=5)
    elseif name == :WENO9
        return WENO(FT; order=9)
    else
        error("Unknown advection scheme: $name")
    end
end

function make_closure(name::Symbol, FT)
    if name == :SmagorinskyLilly
        return SmagorinskyLilly(FT)
    elseif name == :DynamicSmagorinsky
        return DynamicSmagorinsky(FT)
    elseif name == :Nothing
        return nothing
    else
        error("Unknown closure: $name")
    end
end

#####
##### Run benchmarks
#####

results = []

println("=" ^ 95)
println("Breeze.jl Benchmark Suite")
println("=" ^ 95)
println("Date: ", now())
println("Architecture: ", arch)
println("Canonical config: $(canonical_advection), $(canonical_closure), $(canonical_float_type)")
println("Time steps: ", time_steps)
println("Δt: ", Δt, " s")
println("=" ^ 95)
println()

for resolution in resolutions
    println("\n", "=" ^ 50)
    println("Resolution: $resolution")
    println("=" ^ 50)

    #####
    ##### 1. Canonical configuration
    #####

    name = "CBL_$(resolution)_canonical"
    println("\n", "-" ^ 70)
    println("Running: $name (WENO5 + Nothing + F32)")
    println("-" ^ 70)

    model = convective_boundary_layer(arch;
        resolution,
        float_type = canonical_float_type,
        advection = make_advection(canonical_advection, canonical_float_type),
        closure = make_closure(canonical_closure, canonical_float_type)
    )

    result = benchmark_time_stepping(model;
        time_steps, Δt, warmup_steps, name, verbose = true
    )
    push!(results, result)

    #####
    ##### 2. Vary float type (F64 vs canonical F32)
    #####

    name = "CBL_$(resolution)_F64"
    println("\n", "-" ^ 70)
    println("Running: $name (vary float type)")
    println("-" ^ 70)

    FT = Float64
    model = convective_boundary_layer(arch;
        resolution,
        float_type = FT,
        advection = make_advection(canonical_advection, FT),
        closure = make_closure(canonical_closure, FT)
    )

    result = benchmark_time_stepping(model;
        time_steps, Δt, warmup_steps, name, verbose = true
    )
    push!(results, result)

    #####
    ##### 3. Vary advection scheme
    #####

    for adv_name in [:Centered2, :WENO9]
        name = "CBL_$(resolution)_$(adv_name)"
        println("\n", "-" ^ 70)
        println("Running: $name (vary advection)")
        println("-" ^ 70)

        model = convective_boundary_layer(arch;
            resolution,
            float_type = canonical_float_type,
            advection = make_advection(adv_name, canonical_float_type),
            closure = make_closure(canonical_closure, canonical_float_type)
        )

        result = benchmark_time_stepping(model;
            time_steps, Δt, warmup_steps, name, verbose = true
        )
        push!(results, result)
    end

    #####
    ##### 4. Vary closure
    #####

    for cls_name in [:SmagorinskyLilly, :DynamicSmagorinsky]
        name = "CBL_$(resolution)_$(cls_name)"
        println("\n", "-" ^ 70)
        println("Running: $name (vary closure)")
        println("-" ^ 70)

        model = convective_boundary_layer(arch;
            resolution,
            float_type = canonical_float_type,
            advection = make_advection(canonical_advection, canonical_float_type),
            closure = make_closure(cls_name, canonical_float_type)
        )

        result = benchmark_time_stepping(model;
            time_steps, Δt, warmup_steps, name, verbose = true
        )
        push!(results, result)
    end
end

#####
##### Summary table
#####

println("\n", "=" ^ 95)
println("BENCHMARK SUMMARY")
println("=" ^ 95)
println()
println("Canonical configuration: WENO5 + Nothing + Float32")
println()

@printf("%-40s %8s %12s %12s %15s\n", "Benchmark", "Float", "Grid", "Time/Step", "Points/s")
println("-" ^ 95)

for r in results
    grid_str = "$(r.grid_size[1])×$(r.grid_size[2])×$(r.grid_size[3])"
    @printf("%-40s %8s %12s %10.4f ms %15.2e\n",
        r.name,
        r.float_type,
        grid_str,
        r.time_per_step_seconds * 1000,
        r.grid_points_per_second
    )
end

println("=" ^ 95)

#####
##### Save results
#####

# Generate filename with timestamp
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
arch_str = arch isa CPU ? "cpu" : "gpu"

# Save binary data
jld2_filename = "benchmark_$(arch_str)_$(timestamp).jld2"
save_benchmark(jld2_filename, results)
println("\nBinary results saved to: $jld2_filename")

# Save markdown report
md_filename = "benchmark_$(arch_str)_$(timestamp).md"
metadata = results[1].metadata

open(md_filename, "w") do io
    println(io, "# Breeze.jl Benchmark Results")
    println(io)
    println(io, "## System Information")
    println(io)
    println(io, "| Property | Value |")
    println(io, "|----------|-------|")
    println(io, "| Date | ", metadata.timestamp, " |")
    println(io, "| Julia | ", metadata.julia_version, " |")
    println(io, "| Oceananigans | ", metadata.oceananigans_version, " |")
    println(io, "| Breeze | ", metadata.breeze_version, " |")
    println(io, "| Architecture | ", metadata.architecture, " |")
    println(io, "| CPU | ", metadata.cpu_model, " |")
    println(io, "| Threads | ", metadata.num_threads, " |")
    if !isnothing(metadata.gpu_name)
        println(io, "| GPU | ", metadata.gpu_name, " |")
        println(io, "| CUDA | ", metadata.cuda_version, " |")
    end
    println(io, "| Hostname | ", metadata.hostname, " |")
    println(io)
    println(io, "## Benchmark Configuration")
    println(io)
    println(io, "- **Canonical:** WENO5 + Nothing + Float32")
    println(io, "- **Time steps:** ", time_steps)
    println(io, "- **Δt:** ", Δt, " s")
    println(io, "- **Warmup steps:** ", warmup_steps)
    println(io)
    println(io, "## Results")
    println(io)
    println(io, "| Benchmark | Float | Grid | Time/Step (ms) | Points/s |")
    println(io, "|-----------|-------|------|----------------|----------|")
    for r in results
        grid_str = "$(r.grid_size[1])×$(r.grid_size[2])×$(r.grid_size[3])"
        @printf(io, "| %s | %s | %s | %.2f | %.2e |\n",
            r.name, r.float_type, grid_str,
            r.time_per_step_seconds * 1000,
            r.grid_points_per_second)
    end
end

println("Markdown report saved to: $md_filename")
println("Benchmarks completed at ", now())
