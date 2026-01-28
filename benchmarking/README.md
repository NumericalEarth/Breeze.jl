# Breeze.jl Benchmarks

This directory contains benchmarking tools and canonical configurations for measuring Breeze.jl performance.

## Benchmark Case: Convective Boundary Layer (CBL)

The primary benchmark case is a dry convective boundary layer simulation based on Section 4.2 of
[Sauer & Munoz-Esparza (2020)](https://doi.org/10.1029/2020MS002100), "The FastEddy® Resident-GPU
Accelerated Large-Eddy Simulation Framework".

### Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Domain | 12 × 12 × 3 km | Horizontal × Vertical extent |
| Geostrophic wind | (9, 0) m/s | (Uᵍ, Vᵍ) |
| Latitude | 33.5° N | Coriolis parameter f ≈ 8.0 × 10⁻⁵ s⁻¹ |
| Surface θ | 309 K | Surface potential temperature |
| Surface heat flux | 0.35 K·m/s | Kinematic sensible heat flux |
| Stratification | Neutral below 600 m | dθ/dz = 0.004 K/m above |
| Initial perturbations | ±0.25 K | In lowest 400 m |

### Resolution Presets

| Resolution | Grid Size | Total Points | Purpose |
|------------|-----------|--------------|---------|
| `:small` | 32 × 32 × 32 | 32,768 | Quick tests |
| `:medium` | 64 × 64 × 64 | 262,144 | Development benchmarks |
| `:large` | 128 × 128 × 64 | 1,048,576 | Performance benchmarks |
| `:production` | 600 × 594 × 122 | 43,477,200 | Full case from paper |

## Canonical Configuration

The canonical benchmark configuration is:
- **Float type:** Float32
- **Advection:** WENO5
- **Closure:** None

Benchmarks vary one parameter at a time from this baseline to isolate the impact of each choice.

## Running Benchmarks

### Quick Start

```bash
cd benchmarking
julia --project=. run_benchmarks.jl
```

Results are automatically saved to a timestamped JLD2 file with full system metadata.

### Configuration

Edit `run_benchmarks.jl` to customize:

```julia
# Architecture: CPU() for testing, GPU() for production
arch = CPU()

# Resolutions to test
resolutions = [:small, :medium]

# Canonical configuration
canonical_float_type = Float32
canonical_advection = :WENO5
canonical_closure = :Nothing
```

### Programmatic Usage

```julia
using BreezeBenchmarks
using Oceananigans

# Create a benchmark model
model = convective_boundary_layer(GPU();
    resolution = :medium,
    float_type = Float32,
    advection = WENO(Float32; order=5),
    closure = nothing
)

# Run benchmark (uses many_time_steps! to avoid Simulation overhead)
result = benchmark_time_stepping(model;
    time_steps = 100,
    warmup_steps = 10,
    Δt = 0.05
)

# Save results with full metadata
save_benchmark("my_benchmark.jld2", result)

# Load results later
loaded = load_benchmark("my_benchmark.jld2")
```

### Time Stepping

Benchmarks use `many_time_steps!` which calls `time_step!(model, Δt)` directly in a loop,
avoiding the overhead of `Simulation` and `run!`:

```julia
function many_time_steps!(model, Δt, N=100)
    for _ in 1:N
        time_step!(model, Δt)
    end
end
```

## CPU Benchmark Results

**System:** Apple M3 Max, 6 threads, macOS
**Date:** 2026-01-22
**Julia:** 1.11.4
**Breeze:** 0.3.1
**Oceananigans:** 0.104.2

### Canonical Configuration (WENO5 + Nothing + Float32)

| Resolution | Grid | Time/Step | Points/s |
|------------|------|-----------|----------|
| small | 32³ | 48.5 ms | 6.8×10⁵ |
| medium | 64³ | 108.2 ms | 2.4×10⁶ |

### Float Type Comparison (vs canonical F32)

| Resolution | Float32 | Float64 | F32 Speedup |
|------------|---------|---------|-------------|
| small (32³) | 48.5 ms | 28.0 ms | 0.58× (F64 faster) |
| medium (64³) | 108.2 ms | 182.2 ms | 1.68× |

At small grid sizes, Float64 is faster due to reduced type conversion overhead.
At larger sizes, Float32 becomes faster as memory bandwidth dominates.

### Advection Scheme Comparison (F32, no closure)

| Resolution | Centered2 | WENO5 | WENO9 |
|------------|-----------|-------|-------|
| small (32³) | 44.5 ms | 48.5 ms | 61.4 ms |
| medium (64³) | 64.3 ms | 108.2 ms | 197.3 ms |

WENO schemes are more expensive due to higher-order reconstruction:
- WENO5 is ~1.7× slower than Centered2
- WENO9 is ~3× slower than Centered2

### Closure Comparison (F32, WENO5)

| Resolution | Nothing | SmagorinskyLilly | DynamicSmagorinsky |
|------------|---------|------------------|-------------------|
| small (32³) | 48.5 ms | 53.5 ms (+10%) | 56.5 ms (+16%) |
| medium (64³) | 108.2 ms | 142.0 ms (+31%) | 141.8 ms (+31%) |

Both closures add ~30% overhead at production-relevant grid sizes.
SmagorinskyLilly and DynamicSmagorinsky have similar cost.

## Saved Benchmark Data

Benchmark results are saved to JLD2 files with full metadata:

```julia
result = load_benchmark("benchmark_cpu_2026-01-22_161740.jld2")
result[1].metadata
# BenchmarkMetadata
# ├── julia_version: 1.11.4
# ├── oceananigans_version: 0.104.2
# ├── breeze_version: 0.3.1
# ├── architecture: CPU
# ├── cpu_model: Apple M3 Max
# ├── num_threads: 6
# ├── hostname: ...
# └── timestamp: 2026-01-22T16:17:40
```

For GPU benchmarks, metadata also includes `gpu_name` and `cuda_version`.

## Guidelines for Generating and Reporting Benchmarks

### Before Running Benchmarks

1. **Close other applications** to minimize interference
2. **Use a consistent power state** (plugged in, not in power-saving mode)
3. **Let the system reach thermal equilibrium** before GPU benchmarks

### Benchmark Configuration

1. **Warmup steps**: Always include warmup steps (default: 10) to allow JIT compilation
   and cache warming before timing

2. **Number of time steps**: Use at least 100 time steps for stable measurements

3. **Time step size**: Use Δt = 0.05 s (from FastEddy paper) for consistency

4. **Multiple runs**: For publication-quality results, run benchmarks 3-5 times
   and report median or mean ± standard deviation

### Reporting Results

When reporting benchmark results, the saved metadata includes:

- **Julia version**
- **Oceananigans and Breeze versions**
- **Architecture** (CPU/GPU)
- **GPU name and CUDA version** (if applicable)
- **CPU model**
- **Number of threads**
- **Hostname and timestamp**

### Comparing Results

When comparing benchmarks across systems or versions:

1. **Use the canonical configuration** as baseline
2. **Vary one parameter at a time**
3. **Report relative speedup** rather than absolute times when comparing hardware
4. **Consider statistical significance** for small differences (<10%)

### GPU Benchmarks

For GPU benchmarks:

1. **Use larger problem sizes**: GPUs need sufficient work to overcome launch overhead
   (recommend `:large` or `:production` resolution)
2. **Report GPU memory usage**: Check with `CUDA.memory_status()`
3. **Note GPU temperature**: Throttling affects performance

## File Structure

```
benchmarking/
├── Project.toml                      # Package dependencies
├── README.md                         # This file
├── run_benchmarks.jl                 # Main benchmark script
├── benchmark_cpu_*.jld2              # Saved benchmark results
└── src/
    ├── BreezeBenchmarks.jl           # Module exports and utilities
    └── convective_boundary_layer.jl  # CBL benchmark case
```

## Adding New Benchmark Cases

To add a new benchmark case:

1. Create a new file in `src/` (e.g., `src/your_case.jl`)
2. Define a function that returns an `AtmosphereModel`:
   ```julia
   function your_case(arch = CPU();
                      resolution = :medium,
                      float_type = Float64,
                      advection = WENO(order=5),
                      closure = nothing)
       # ... setup code ...
       return model
   end
   ```
3. Include and export from `BreezeBenchmarks.jl`
4. Document the case in this README

## References

- Sauer, J. A., & Muñoz-Esparza, D. (2020). The FastEddy® resident-GPU accelerated
  large-eddy simulation framework: Model formulation, dynamical-core validation
  and performance benchmarks. *Journal of Advances in Modeling Earth Systems*,
  12, e2020MS002100. https://doi.org/10.1029/2020MS002100
