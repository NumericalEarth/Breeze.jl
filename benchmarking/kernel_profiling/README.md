# Kernel profiling

`profile_kernels.jl` reports where the GPU time of a vanilla (CUDA) time step goes, kernel by kernel.
It builds the convective boundary layer model of `run_benchmarks.jl` from the same command-line arguments, takes `--warmup_steps` steps so every kernel is compiled, and profiles the next `--time_steps` steps with CUDA.jl's integrated profiler, `CUDA.@profile`.
That profiler uses CUPTI, which CUDA.jl ships, so no Nsight Systems install is needed.

Launches are grouped by kernel name up to the first `(` or `<`, so a kernel launched for several fields or with several type parameters counts as one.
Memory copies and sets are left out.
For the 10 kernels with the most GPU time, the script records the launch count and the total, median, minimum and maximum GPU time.
It writes them, with the configuration and the run's metadata, to `kernel_profile.json` in the working directory, and writes a Markdown table to `kernel_profile.md`.

```bash
# From benchmarking/: the configuration CI profiles, the vanilla side of the Reactant vs vanilla comparison
julia --project kernel_profiling/profile_kernels.jl --device GPU \
    --dynamics compressible_splitexplicit --microphysics nothing --advection WENO5 \
    --size "128x128x32, 256x256x128" --topology PBB \
    --simplified --float_type Float32 --grid_type latitude_longitude \
    --warmup_steps 8 --time_steps 80
```

`--backend` is ignored (the run is always vanilla), and the run needs `--device GPU`.
List arguments (`--size`, `--advection`, ...) sweep as in `run_benchmarks.jl`, one profile per configuration.

## In CI

A `Benchmarks.yml` matrix entry with `profile_kernels: true` runs this script with the entry's own arguments after its benchmarks, so the profiler cannot touch their timings.
The tables go to the job summary.
`kernel_profile.json` is uploaded with the entry's `benchmark_results.json`, and the publish step charts each kernel's GPU time per time step (ms/step) on the "Breeze.jl Kernel Times" page.
That page has one figure per configuration, one panel per grid and one line per kernel.
Currently the Reactant vs vanilla entry is the only one with `profile_kernels: true`.

Kernels on the Reactant backend are XLA's fusions rather than Breeze's kernels, so they are not profiled here.
