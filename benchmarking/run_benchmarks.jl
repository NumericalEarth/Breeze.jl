#####
##### Breeze.jl Benchmark Script
#####
##### This script runs the convective boundary layer benchmark case
##### at various resolutions to measure performance.
#####

using Pkg
Pkg.activate(@__DIR__)

using BreezeBenchmarks
using Oceananigans
using Oceananigans.TurbulenceClosures: SmagorinskyLilly

using Printf
using Dates

#####
##### Configuration
#####

# Set architecture: CPU() for testing, GPU() for production benchmarks
arch = CPU()

# Resolutions to benchmark
resolutions = [:small, :medium]  # Add :large, :production for comprehensive benchmarks

# Float types to test
float_types = [
    ("F32", Float32),
    ("F64", Float64),
]

# Advection schemes to test
advection_schemes = [
    ("Centered2", Centered(order=2)),
    ("WENO5", WENO(order=5)),
]

# Closures to test
closures = [
    ("SmagorinskyLilly", SmagorinskyLilly()),
    ("Nothing", nothing),
]

# Benchmark parameters
time_steps = 100
Δt = 0.05  # seconds (from FastEddy paper)
warmup_steps = 10

#####
##### Run benchmarks
#####

results = []

println("=" ^ 95)
println("Breeze.jl Benchmark Suite")
println("=" ^ 95)
println("Date: ", now())
println("Architecture: ", arch)
println("Time steps: ", time_steps)
println("Δt: ", Δt, " s")
println("=" ^ 95)
println()

for resolution in resolutions
    for (ft_name, float_type) in float_types
        for (adv_name, advection) in advection_schemes
            for (cls_name, closure) in closures
                name = "CBL_$(resolution)_$(ft_name)_$(adv_name)_$(cls_name)"

                println("-" ^ 70)
                println("Running: $name")
                println("-" ^ 70)

                model = convective_boundary_layer(arch;
                    resolution,
                    float_type,
                    advection,
                    closure
                )

            result = benchmark_time_stepping(model;
                time_steps,
                Δt,
                warmup_steps,
                name,
                verbose = true
            )

                push!(results, result)
                println()
            end
        end
    end
end

#####
##### Summary table
#####

println("=" ^ 95)
println("BENCHMARK SUMMARY")
println("=" ^ 95)
println()

@printf("%-45s %8s %12s %12s %15s\n", "Benchmark", "Float", "Grid", "Time/Step", "Points/s")
println("-" ^ 95)

for r in results
    grid_str = "$(r.grid_size[1])×$(r.grid_size[2])×$(r.grid_size[3])"
    @printf("%-45s %8s %12s %10.4f ms %15.2e\n",
        r.name,
        r.float_type,
        grid_str,
        r.time_per_step_seconds * 1000,
        r.grid_points_per_second
    )
end

println("=" ^ 95)
println("Benchmarks completed at ", now())
