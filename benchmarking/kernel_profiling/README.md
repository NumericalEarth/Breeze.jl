# Kernel profiling

`run_benchmarks.jl --profile_kernels` reports where the GPU time of a vanilla (CUDA) time step goes, kernel by kernel.
The code is in `kernel_profiling.jl`, which `run_benchmarks.jl` includes.

Profiling runs once every case has been timed and its results saved, in the same Julia process:
- **No timing is affected:** every published number is measured before the profiler starts, so neither the profiler nor any state it leaves behind can affect a timing.
- **No kernel is recompiled:** each vanilla time-stepping case's model is rebuilt from the same configuration and reuses the kernels compiled when the case was timed.
- **Failures are contained:** a profiler failure is logged, and the run still succeeds with its saved results.

For each case, after `--warmup_steps` steps, the next `--time_steps` steps are profiled with CUDA.jl's integrated profiler, `CUDA.@profile`.
That profiler uses CUPTI, which CUDA.jl ships, so no Nsight Systems install is needed.
Launches are grouped by kernel name up to the first `(` or `<`, so a kernel launched for several fields or with several type parameters counts as one.
Memory copies and sets are left out.

For the 10 kernels with the most GPU time, the run records the launch count and the total, median, minimum and maximum GPU time.
It writes them, with the configuration and the run's metadata, to `kernel_profile.json` in the working directory, and writes a Markdown table to `kernel_profile.md`.
Only vanilla time-stepping cases are profiled (benchmark mode, `--device GPU`, without `--ad`).
In a run with `--backend "vanilla, reactant"`, the Reactant cases are timed as usual but not profiled.

```bash
# From benchmarking/: the configuration CI profiles, the Reactant vs vanilla comparison
julia --project run_benchmarks.jl --device GPU --backend "vanilla, reactant" \
    --dynamics compressible_splitexplicit --microphysics nothing --advection WENO5 \
    --size "128x128x32, 256x256x128" --topology PBB \
    --simplified --float_type Float32 --grid_type latitude_longitude \
    --warmup_steps 8 --time_steps 80 --profile_kernels
```

## In CI

A `Benchmarks.yml` matrix entry with `profile_kernels: true` passes `--profile_kernels` to its "Run benchmarks" step.
The tables go to the job summary.
`kernel_profile.json` is uploaded with the entry's `benchmark_results.json`, and the publish step charts each kernel's GPU time per time step (ms/step) on the "Breeze.jl Kernel Times" page.
That page has one figure per configuration, one panel per grid and one line per kernel.
Currently the Reactant vs vanilla entry is the only one with `profile_kernels: true`.

Kernels on the Reactant backend are XLA's fusions rather than Breeze's kernels, so they are not profiled here.
