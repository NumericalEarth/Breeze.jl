# Breeze.jl Benchmarks

This directory contains benchmarking tools for measuring Breeze.jl performance.

## Quick Start

Run benchmarks from the command line:

```bash
cd benchmarking

# Default: GPU benchmark with 64³ grid, Float32, WENO5
julia --project run_benchmarks.jl

# Multiple grid sizes
julia --project run_benchmarks.jl --size="64^3, 128^3, 256x256x128"

# Sweep advection schemes
julia --project run_benchmarks.jl --size=128x128x128 --advection="Centered2, WENO5, WENO9"

# Full configuration sweep
julia --project run_benchmarks.jl \
    --size="64^3, 128^3" \
    --float_type="Float32, Float64" \
    --advection="WENO5, WENO9" \
    --closure="nothing, SmagorinskyLilly"

# Run on CPU instead
julia --project run_benchmarks.jl --device=CPU --size=32^3
```

Results are saved to JSON and a markdown report is automatically generated.

## Modes

The script supports two modes:

### Benchmark Mode (default)

Quick performance benchmarks that run a fixed number of time steps without output.
Used for measuring computational throughput.

```bash
julia --project run_benchmarks.jl --mode=benchmark --size=128^3 --time_steps=100
```

### Simulate Mode

Full simulations that run for a specified duration and save output files.
Used for validation and scientific analysis. Based on the FastEddy CBL case,
which typically runs for 2 hours to reach quasi-steady convective state.

```bash
# Run 2-hour simulation with output every 10 minutes
julia --project run_benchmarks.jl --mode=simulate --size=128^3

# Shorter run for testing
julia --project run_benchmarks.jl --mode=simulate --size=64^3 --stop_time=0.5 --output_interval=5

# Production run
julia --project run_benchmarks.jl --mode=simulate --size=256^3 --stop_time=2.0 --dt=0.5
```

## Command-Line Arguments

### General Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `benchmark` | Mode: `benchmark` or `simulate` |
| `--size` | `64^3` | Grid size. Formats: `NxNyxNz` or `N^3`. Comma-separated for multiple. |
| `--device` | `GPU` | Device: `CPU` or `GPU` |
| `--configuration` | `convective_boundary_layer` | Benchmark case to run |
| `--float_type` | `Float32` | Floating point type: `Float32` or `Float64`. Comma-separated for multiple. |
| `--advection` | `WENO5` | Advection scheme: `nothing`, `Centered2`, `WENO5`, `WENO9`, `bounded_WENO5`. Comma-separated for multiple. |
| `--closure` | `nothing` | Turbulence closure: `nothing`, `SmagorinskyLilly`, `DynamicSmagorinsky`. Comma-separated for multiple. |
| `--microphysics` | `nothing` | Microphysics scheme (see below). Comma-separated for multiple. |
| `--dt` | `0.5` | Time step size in seconds |
| `--output` | `benchmark_results.json` | Output JSON filename for results |
| `--clear` | `false` | Clear existing results file before writing |

### Benchmark Mode Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--time_steps` | `100` | Number of time steps to benchmark |
| `--warmup_steps` | `10` | Number of warmup steps (for JIT compilation) |

### Simulate Mode Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stop_time` | `2.0` | Simulation stop time in hours |
| `--output_interval` | `10.0` | Output interval in minutes |
| `--output_dir` | `.` | Directory for simulation output files |

### Microphysics Schemes

| Scheme | Description |
|--------|-------------|
| `nothing` | Dry dynamics (no moisture) |
| `SaturationAdjustment` | 0M saturation adjustment (default mixed-phase) |
| `WarmPhaseEquilibrium` | 0M warm-phase saturation adjustment |
| `MixedPhaseEquilibrium` | 0M mixed-phase saturation adjustment |
| `1M_WarmEquilibrium` | 1M warm-rain with saturation adjustment |
| `1M_MixedEquilibrium` | 1M mixed-phase with saturation adjustment |
| `1M_WarmNonEquilibrium` | 1M warm-rain with prognostic cloud liquid |
| `1M_MixedNonEquilibrium` | 1M mixed-phase with prognostic cloud liquid/ice |

**0M schemes** use saturation adjustment (equilibrium cloud formation) where cloud
condensate is diagnosed from thermodynamic state. No precipitation.

**1M schemes** add prognostic precipitation (rain, snow) with autoconversion and
accretion processes from CloudMicrophysics.jl:
- **Equilibrium** variants use saturation adjustment for cloud formation
- **NonEquilibrium** variants have prognostic cloud liquid/ice with condensation/evaporation
  tendencies following [Morrison and Grabowski (2008)](https://doi.org/10.1175/2007JAS2491.1)

```bash
# Compare 1M microphysics schemes
julia --project run_benchmarks.jl --size=128^3 \
    --microphysics="nothing, 1M_WarmEquilibrium, 1M_MixedEquilibrium, 1M_WarmNonEquilibrium"
```

### Size Format Examples

```bash
# Cubic grid shorthand
--size=64^3          # 64 × 64 × 64
--size=128^3         # 128 × 128 × 128

# Explicit dimensions
--size=128x128x64    # 128 × 128 × 64
--size=256x256x128   # 256 × 256 × 128

# Multiple sizes (comma-separated, quotes required)
--size="32^3, 64^3, 128^3"
--size="64x64x32, 128x128x64, 256x256x128"
```

### Sweep Examples

```bash
# Compare float types
julia --project run_benchmarks.jl --size=256^3 --float_type="Float32, Float64"

# Compare advection schemes
julia --project run_benchmarks.jl --size=256^3 --advection="Centered2, WENO5, WENO9"

# Compare closures
julia --project run_benchmarks.jl --size=256^3 --closure="nothing, SmagorinskyLilly, DynamicSmagorinsky"

# Compare microphysics schemes
julia --project run_benchmarks.jl --size=128^3 --microphysics="nothing, SaturationAdjustment, MixedPhaseEquilibrium"

# Full factorial design (all combinations)
julia --project run_benchmarks.jl --size=128^3 \
    --float_type="Float32, Float64" \
    --advection="WENO5, WENO9" \
    --closure="nothing, SmagorinskyLilly"
```

## Output Files

Benchmark results are saved in two formats:

1. **JSON file** (`benchmark_results.json`): Machine-readable results that accumulate across runs
2. **Markdown file** (`benchmark_results.md`): Human-readable report auto-generated from JSON

### Appending vs. Clearing Results

By default, new benchmark results are appended to the existing JSON file:

```bash
# These results accumulate in benchmark_results.json
julia --project run_benchmarks.jl --size=64^3
julia --project run_benchmarks.jl --size=128^3
```

Use `--clear` to start fresh:

```bash
julia --project run_benchmarks.jl --size=256^3 --clear
```

Use `--output` to save to a different file:

```bash
julia --project run_benchmarks.jl --output=my_benchmark.json
```

## Benchmark Case: Convective Boundary Layer

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

## Comprehensive GPU Benchmarks

For systematic GPU benchmarking across many configurations, use the dedicated script:

```bash
julia --project run_gpu_benchmarks.jl
```

This script runs a comprehensive suite including:
- Resolution scaling from 128³ to 896³
- Float32 vs Float64 comparison
- All advection schemes (Centered2, WENO5, WENO9)
- All closures (Nothing, SmagorinskyLilly, DynamicSmagorinsky)
- Microphysics schemes (SaturationAdjustment, OneMoment)

### GPU Memory Estimates

| Grid Size | Float32 | Float64 | Notes |
|-----------|---------|---------|-------|
| 128³ | ~0.4 GB | ~0.8 GB | Quick tests |
| 256³ | ~3 GB | ~6 GB | Development |
| 512³ | ~26 GB | ~52 GB | Production |
| 768³ | ~86 GB | ~172 GB | High-end GPU only |
| 896³ | ~136 GB | N/A | Near H200 limit |

## Programmatic Usage

```julia
using BreezeBenchmarks
using Oceananigans

# Create a benchmark model
model = convective_boundary_layer(GPU();
    Nx = 128, Ny = 128, Nz = 128,
    float_type = Float32,
    advection = WENO(Float32; order=5),
    closure = nothing
)

# Run benchmark
result = benchmark_time_stepping(model;
    time_steps = 100,
    warmup_steps = 10,
    Δt = 0.05,
    name = "my_benchmark"
)

# Save results
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

## Benchmark Metadata

Benchmark results include system metadata:

```julia
result.metadata
# BenchmarkMetadata
# ├── julia_version: 1.11.4
# ├── oceananigans_version: 0.104.2
# ├── breeze_version: 0.3.1
# ├── architecture: GPU
# ├── gpu_name: NVIDIA H200
# ├── cuda_version: 12.6
# ├── cpu_model: AMD EPYC
# ├── num_threads: 64
# ├── hostname: ...
# └── timestamp: 2026-01-30T12:00:00
```

## Best Practices

### Before Running Benchmarks

1. **Close other applications** to minimize interference
2. **Use a consistent power state** (plugged in, not in power-saving mode)
3. **Let the system reach thermal equilibrium** before GPU benchmarks

### Benchmark Configuration

1. **Warmup steps**: Always include warmup (default: 10) to allow JIT compilation
2. **Number of time steps**: Use at least 100 for stable measurements
3. **Time step size**: Use Δt = 0.05 s (from FastEddy paper) for consistency
4. **Multiple runs**: For publication-quality results, run 3-5 times and report median

### GPU Recommendations

1. **Use larger problem sizes**: GPUs need sufficient work to overcome launch overhead
   (recommend 256³ or larger for meaningful GPU benchmarks)
2. **Check GPU memory**: Use `CUDA.memory_status()` to verify memory usage
3. **Monitor GPU temperature**: Throttling affects performance

### Comparing Results

1. **Use the canonical configuration** (WENO5, Float32, no closure) as baseline
2. **Vary one parameter at a time** to isolate effects
3. **Report relative speedup** rather than absolute times when comparing hardware
4. **Consider statistical significance** for small differences (<10%)

## File Structure

```
benchmarking/
├── Project.toml                      # Package dependencies
├── README.md                         # This file
├── run_benchmarks.jl                 # Command-line benchmark script
├── run_gpu_benchmarks.jl             # Comprehensive GPU benchmark suite
├── plot_benchmarks.jl                # Visualization utilities
├── benchmark_results.json            # Accumulated results (JSON)
├── benchmark_results.md              # Auto-generated report
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
                      Nx = 64, Ny = 64, Nz = 64,
                      float_type = Float32,
                      advection = WENO(Float32; order=5),
                      closure = nothing)
       # ... setup code ...
       return model
   end
   ```
3. Include and export from `BreezeBenchmarks.jl`
4. Add to `--configuration` options in `run_benchmarks.jl`
5. Document the case in this README

## References

- Sauer, J. A., & Muñoz-Esparza, D. (2020). The FastEddy® resident-GPU accelerated
  large-eddy simulation framework: Model formulation, dynamical-core validation
  and performance benchmarks. *Journal of Advances in Modeling Earth Systems*,
  12, e2020MS002100. https://doi.org/10.1029/2020MS002100
