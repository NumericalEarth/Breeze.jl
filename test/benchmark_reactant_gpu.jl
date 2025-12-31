#=
Benchmark: Reactant GPU vs Native CUDA Performance for Breeze AtmosphereModel

This script profiles and benchmarks the Reactant-compiled time stepping (XLA GPU)
against native CUDA execution to identify bottlenecks and quantify performance.

This is the GPU equivalent of benchmark_reactant.jl (which compares CPU vs Reactant CPU).

## Requirements

- **Julia 1.10.x** (Julia 1.12 has MLIR/LLVM lowering issues with KA kernels)
- NVIDIA GPU with CUDA support

## How to Run

```bash
julia +1.10 --project=test test/benchmark_reactant_gpu.jl
```

## Custom Grid Size

```bash
BENCH_NX=128 BENCH_NZ=128 julia +1.10 --project=test test/benchmark_reactant_gpu.jl
```

## What This Measures

1. Compilation time (one-time cost)
2. Per-step execution time (Native CUDA vs Reactant GPU)
3. Component-level timing (if possible)
=#

using Breeze
using CUDA
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState, GPU
using Oceananigans.Advection: Centered
using Oceananigans.Units
using Statistics

# NOTE: We do NOT set backend here. With CUDA loaded, Reactant will automatically
# create GPU ConcreteRArrays. We set GPU backend before @compile to ensure
# compilation targets GPU. The client must match between arrays and compilation target.

# Ensure GPU is available
@assert CUDA.functional() "CUDA not available! This benchmark requires a GPU."

println("="^80)
println("Breeze Reactant GPU Performance Benchmark")
println("="^80)
println()
println("GPU Device: $(CUDA.name(CUDA.device()))")
println()

# Configuration - can be overridden via environment
const Nx = parse(Int, get(ENV, "BENCH_NX", "32"))
const Nz = parse(Int, get(ENV, "BENCH_NZ", "32"))
const Lx, Lz = 1000.0, 1000.0
const Δt_val = 1.0
const WARMUP_STEPS = 3
const BENCHMARK_STEPS = 10

println("Grid size: $Nx × 1 × $Nz (2D with Flat y)")
println("Domain: $Lx × $Lz meters")
println("Time step: $Δt_val seconds")
println("Warmup steps: $WARMUP_STEPS")
println("Benchmark steps: $BENCHMARK_STEPS")
println()

# Initial condition function
θ₀(x, z) = 300.0 + 0.01 * z

#####
##### Reactant GPU Model (must be created BEFORE native CUDA model)
#####
# IMPORTANT: Reactant model must be created first, while Reactant is in CPU mode.
# Creating a GPU() model first can affect Reactant's internal state.

println("-"^80)
println("Setting up Reactant model (CPU mode for construction)...")
println("-"^80)

reactant_grid = RectilinearGrid(ReactantState();
    size = (Nx, Nz),
    halo = (5, 5),  # Match test configuration
    x = (0, Lx),
    z = (0, Lz),
    topology = (Periodic, Flat, Bounded)
)

reactant_model = AtmosphereModel(reactant_grid; advection = Centered(order=2))

# Set initial condition
set!(reactant_model; θ = θ₀)

# Initialize clock (GB-25 pattern)
reactant_model.clock.last_Δt = Δt_val

println("Reactant model created: $(typeof(reactant_model).name.name)")
println()

# Compile with GPU backend
println("Compiling Reactant time_step! for GPU (this may take a few minutes)...")

# Switch to GPU backend for compilation
Reactant.set_default_backend("gpu")
println("Reactant backend set to: gpu (for compilation)")

compile_start = time_ns()
compiled_step! = Reactant.@compile sync=true time_step!(reactant_model, Δt_val)
compile_end = time_ns()
compile_time_ms = (compile_end - compile_start) / 1e6

println("Compilation time: $(round(compile_time_ms / 1000, digits=2)) seconds")
println()

# Helper function to synchronize Reactant model fields
function sync_reactant_model(model)
    # Synchronize on the momentum field data (forces GPU to complete)
    # Use model.momentum.ρu (not model.velocities.ρu - velocities are u, v, w)
    Reactant.synchronize(parent(model.momentum.ρu.data))
end

# Warmup - must synchronize GPU operations
println("Warming up Reactant GPU model ($WARMUP_STEPS steps)...")
for _ in 1:WARMUP_STEPS
    compiled_step!(reactant_model, Δt_val)
end
# Synchronize after warmup (XLA GPU ops are async)
sync_reactant_model(reactant_model)

# Benchmark Reactant GPU - with proper synchronization
println("Benchmarking Reactant GPU model ($BENCHMARK_STEPS steps)...")
reactant_times = Float64[]
for i in 1:BENCHMARK_STEPS
    sync_reactant_model(reactant_model)  # Ensure previous work is done
    t_start = time_ns()
    compiled_step!(reactant_model, Δt_val)
    sync_reactant_model(reactant_model)  # Wait for this step to complete
    t_end = time_ns()
    push!(reactant_times, (t_end - t_start) / 1e6)  # Convert to ms
end

reactant_mean = mean(reactant_times)
reactant_std = std(reactant_times)
reactant_min = minimum(reactant_times)
reactant_max = maximum(reactant_times)

println()
println("Reactant GPU Results:")
println("  Mean:   $(round(reactant_mean, digits=3)) ms/step")
println("  Std:    $(round(reactant_std, digits=3)) ms")
println("  Min:    $(round(reactant_min, digits=3)) ms")
println("  Max:    $(round(reactant_max, digits=3)) ms")
println()

# Clean up Reactant model
reactant_model = nothing
GC.gc()

#####
##### Native CUDA Baseline
#####

println("-"^80)
println("Setting up Native CUDA model...")
println("-"^80)

# For Flat topology, don't specify size or extent in that dimension
gpu_grid = RectilinearGrid(GPU();
    size = (Nx, Nz),
    halo = (5, 5),  # Match test configuration
    x = (0, Lx),
    z = (0, Lz),
    topology = (Periodic, Flat, Bounded)
)

gpu_model = AtmosphereModel(gpu_grid; advection = Centered(order=2))

# Initial condition
set!(gpu_model; θ = θ₀)

println("Native CUDA model created: $(typeof(gpu_model).name.name)")
println()

# Warmup
println("Warming up Native CUDA model ($WARMUP_STEPS steps)...")
for _ in 1:WARMUP_STEPS
    time_step!(gpu_model, Δt_val)
end
CUDA.synchronize()

# Benchmark Native CUDA
println("Benchmarking Native CUDA model ($BENCHMARK_STEPS steps)...")
gpu_times = Float64[]
for i in 1:BENCHMARK_STEPS
    CUDA.synchronize()
    t_start = time_ns()
    time_step!(gpu_model, Δt_val)
    CUDA.synchronize()
    t_end = time_ns()
    push!(gpu_times, (t_end - t_start) / 1e6)  # Convert to ms
end

gpu_mean = mean(gpu_times)
gpu_std = std(gpu_times)
gpu_min = minimum(gpu_times)
gpu_max = maximum(gpu_times)

println()
println("Native CUDA Results:")
println("  Mean:   $(round(gpu_mean, digits=3)) ms/step")
println("  Std:    $(round(gpu_std, digits=3)) ms")
println("  Min:    $(round(gpu_min, digits=3)) ms")
println("  Max:    $(round(gpu_max, digits=3)) ms")
println()

#####
##### Comparison
#####

println("-"^80)
println("COMPARISON: Native CUDA vs Reactant GPU")
println("-"^80)

speedup = gpu_mean / reactant_mean
if speedup >= 1.0
    println("Reactant GPU is $(round(speedup, digits=2))× FASTER than Native CUDA")
else
    println("Reactant GPU is $(round(1/speedup, digits=2))× SLOWER than Native CUDA")
end

println()
println("| Metric | Native CUDA | Reactant GPU | Ratio |")
println("|--------|-------------|--------------|-------|")
println("| Mean (ms) | $(round(gpu_mean, digits=2)) | $(round(reactant_mean, digits=2)) | $(round(speedup, digits=2))× |")
println("| Min (ms) | $(round(gpu_min, digits=2)) | $(round(reactant_min, digits=2)) | $(round(gpu_min/reactant_min, digits=2))× |")
println("| Std (ms) | $(round(gpu_std, digits=2)) | $(round(reactant_std, digits=2)) | - |")
println()

# Amortization analysis
println("-"^80)
println("AMORTIZATION ANALYSIS")
println("-"^80)

# How many steps until compilation cost is amortized?
if reactant_mean < gpu_mean
    time_saved_per_step = gpu_mean - reactant_mean
    steps_to_amortize = ceil(Int, compile_time_ms / time_saved_per_step)
    println("Compilation overhead: $(round(compile_time_ms / 1000, digits=2)) seconds")
    println("Time saved per step: $(round(time_saved_per_step, digits=3)) ms")
    println("Steps to amortize: $steps_to_amortize")
    println()
    println("Reactant GPU is worth it for simulations with > $steps_to_amortize time steps")
else
    println("Reactant GPU is currently slower than Native CUDA.")
    println("This is expected due to workaround overhead in BreezeReactantExt:")
    println("  - Pure-Julia broadcasts instead of optimized CUDA kernels")
    println("  - XLA cannot match cuFFT/cuBLAS optimizations")
    println("Reactant's value is for AD/differentiation, not raw performance.")
    println("See test/REACTANT_ISSUES.md for details.")
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
println("Step | Native CUDA (ms) | Reactant GPU (ms)")
println("-----|------------------|------------------")
for i in 1:BENCHMARK_STEPS
    println("  $i  |  $(round(gpu_times[i], digits=2))   |  $(round(reactant_times[i], digits=2))")
end
