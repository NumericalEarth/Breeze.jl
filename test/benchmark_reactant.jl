#=
Benchmark: Reactant vs CPU Performance for Breeze AtmosphereModel

This script profiles and benchmarks the Reactant-compiled time stepping
against CPU execution to identify bottlenecks and quantify performance.

## How to Run

```bash
julia --project=test test/benchmark_reactant.jl
```

## What This Measures

1. Compilation time (one-time cost)
2. Per-step execution time (CPU vs Reactant)
3. Component-level timing (if possible)
=#

using Breeze
using CUDA  # Required for Reactant's KernelAbstractions extension
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Advection: Centered
using Oceananigans.Units
using Statistics

println("="^80)
println("Breeze Reactant Performance Benchmark")
println("="^80)
println()

# Configuration - can be overridden via environment
const Nx = parse(Int, get(ENV, "BENCH_NX", "32"))
const Nz = parse(Int, get(ENV, "BENCH_NZ", "32"))
const Lx, Lz = 1000.0, 1000.0
const Δt = 1.0
const WARMUP_STEPS = 3
const BENCHMARK_STEPS = 10

println("Grid size: $Nx × 1 × $Nz (2D with Flat y)")
println("Domain: $Lx × $Lz meters")
println("Time step: $Δt seconds")
println("Warmup steps: $WARMUP_STEPS")
println("Benchmark steps: $BENCHMARK_STEPS")
println()

#####
##### CPU Baseline
#####

println("-"^80)
println("Setting up CPU model...")
println("-"^80)

# For Flat topology, don't specify size or extent in that dimension
cpu_grid = RectilinearGrid(CPU();
    size = (Nx, Nz),
    x = (0, Lx),
    z = (0, Lz),
    topology = (Periodic, Flat, Bounded)
)

cpu_model = AtmosphereModel(cpu_grid; advection = Centered(order=2))

# Initial condition (all 3 args required even for Flat y)
θ₀(x, z) = 300.0 + 0.01 * z
set!(cpu_model; θ = θ₀)

println("CPU model created: $(typeof(cpu_model).name.name)")
println()

# Warmup
println("Warming up CPU model ($WARMUP_STEPS steps)...")
for _ in 1:WARMUP_STEPS
    time_step!(cpu_model, Δt)
end

# Benchmark CPU
println("Benchmarking CPU model ($BENCHMARK_STEPS steps)...")
cpu_times = Float64[]
for i in 1:BENCHMARK_STEPS
    t_start = time_ns()
    time_step!(cpu_model, Δt)
    t_end = time_ns()
    push!(cpu_times, (t_end - t_start) / 1e6)  # Convert to ms
end

cpu_mean = mean(cpu_times)
cpu_std = std(cpu_times)
cpu_min = minimum(cpu_times)
cpu_max = maximum(cpu_times)

println()
println("CPU Results:")
println("  Mean:   $(round(cpu_mean, digits=3)) ms/step")
println("  Std:    $(round(cpu_std, digits=3)) ms")
println("  Min:    $(round(cpu_min, digits=3)) ms")
println("  Max:    $(round(cpu_max, digits=3)) ms")
println()

#####
##### Reactant Model
#####

println("-"^80)
println("Setting up Reactant model...")
println("-"^80)

reactant_grid = RectilinearGrid(ReactantState();
    size = (Nx, Nz),
    x = (0, Lx),
    z = (0, Lz),
    topology = (Periodic, Flat, Bounded)
)

reactant_model = AtmosphereModel(reactant_grid; advection = Centered(order=2))

# Initial condition
set!(reactant_model; θ = θ₀)

println("Reactant model created: $(typeof(reactant_model).name.name)")
println()

# Compile
println("Compiling Reactant time_step!...")
compile_start = time_ns()
compiled_step! = Reactant.@compile time_step!(reactant_model, Δt)
compile_end = time_ns()
compile_time_ms = (compile_end - compile_start) / 1e6

println("Compilation time: $(round(compile_time_ms / 1000, digits=2)) seconds")
println()

# Warmup
println("Warming up Reactant model ($WARMUP_STEPS steps)...")
for _ in 1:WARMUP_STEPS
    compiled_step!(reactant_model, Δt)
end

# Benchmark Reactant
println("Benchmarking Reactant model ($BENCHMARK_STEPS steps)...")
reactant_times = Float64[]
for i in 1:BENCHMARK_STEPS
    t_start = time_ns()
    compiled_step!(reactant_model, Δt)
    t_end = time_ns()
    push!(reactant_times, (t_end - t_start) / 1e6)  # Convert to ms
end

reactant_mean = mean(reactant_times)
reactant_std = std(reactant_times)
reactant_min = minimum(reactant_times)
reactant_max = maximum(reactant_times)

println()
println("Reactant Results:")
println("  Mean:   $(round(reactant_mean, digits=3)) ms/step")
println("  Std:    $(round(reactant_std, digits=3)) ms")
println("  Min:    $(round(reactant_min, digits=3)) ms")
println("  Max:    $(round(reactant_max, digits=3)) ms")
println()

#####
##### Comparison
#####

println("-"^80)
println("COMPARISON")
println("-"^80)

speedup = cpu_mean / reactant_mean
if speedup >= 1.0
    println("Reactant is $(round(speedup, digits=2))× FASTER than CPU")
else
    println("Reactant is $(round(1/speedup, digits=2))× SLOWER than CPU")
end

println()
println("| Metric | CPU | Reactant | Ratio |")
println("|--------|-----|----------|-------|")
println("| Mean (ms) | $(round(cpu_mean, digits=2)) | $(round(reactant_mean, digits=2)) | $(round(speedup, digits=2))× |")
println("| Min (ms) | $(round(cpu_min, digits=2)) | $(round(reactant_min, digits=2)) | $(round(cpu_min/reactant_min, digits=2))× |")
println("| Std (ms) | $(round(cpu_std, digits=2)) | $(round(reactant_std, digits=2)) | - |")
println()

# Amortization analysis
println("-"^80)
println("AMORTIZATION ANALYSIS")
println("-"^80)

# How many steps until compilation cost is amortized?
if reactant_mean < cpu_mean
    time_saved_per_step = cpu_mean - reactant_mean
    steps_to_amortize = ceil(Int, compile_time_ms / time_saved_per_step)
    println("Compilation overhead: $(round(compile_time_ms / 1000, digits=2)) seconds")
    println("Time saved per step: $(round(time_saved_per_step, digits=3)) ms")
    println("Steps to amortize: $steps_to_amortize")
    println()
    println("Reactant is worth it for simulations with > $steps_to_amortize time steps")
else
    println("Reactant is currently slower than CPU for this grid size.")
    println("Consider:")
    println("  - Larger grid sizes (more parallelism)")
    println("  - GPU backend instead of CPU backend")
    println("  - Profiling to identify bottlenecks")
end

println()
println("="^80)
println("Benchmark complete")
println("="^80)

#####
##### Detailed timing breakdown (optional)
#####

println()
println("-"^80)
println("INDIVIDUAL STEP TIMES")
println("-"^80)
println()
println("Step | CPU (ms) | Reactant (ms)")
println("-----|----------|---------------")
for i in 1:BENCHMARK_STEPS
    println("  $i  |  $(round(cpu_times[i], digits=2))   |  $(round(reactant_times[i], digits=2))")
end
