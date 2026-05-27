# Nsight Systems profiling on Perlmutter

Workflow for collecting Nsight Systems (`nsys`) traces of multi-GPU Breeze
benchmarks under SLURM. Designed to be re-run across branches/optimizations so
the resulting `.nsys-rep` files can be compared side by side.

## Dynamics: split-explicit compressible

The driver uses `CompressibleDynamics(SplitExplicitTimeDiscretization)` —
acoustic substepping, no Poisson solve, fully GPU-distributed-friendly.
Anelastic dynamics use the FFT-based `FourierTridiagonalPoissonSolver`, which
does not yet have a distributed-x implementation in Breeze, so the anelastic
path is single-GPU only.

## Files

- `examples/profile_supercell.jl` — profile-enabled driver. Wraps a single timed
  window in `CUDA.@profile` and adds NVTX ranges (`warmup`, `profile_window`,
  `time_step N`) so timelines are navigable.
- `examples/profile_supercell.sh` — SLURM wrapper. Runs one `nsys` per MPI rank
  with `--capture-range=cudaProfilerApi`, traces CUDA/NVTX/MPI, and writes
  per-rank reports under `examples/nsys_reports/<jobid>/`.

## Submitting a run

```bash
# default: 2 GPUs, 128x128x40 per GPU, 5 warmup + 10 profile steps
NGPUS=2 sbatch examples/profile_supercell.sh

# 4 GPUs, full benchmark grid, 5 profile steps
NGPUS=4 NX_PER_GPU=400 NY=400 NZ=80 PROFILE_STEPS=5 \
  sbatch examples/profile_supercell.sh
```

Environment variables consumed by the script:

| Var | Default | Meaning |
|-----|---------|---------|
| `NGPUS` | `2` | MPI ranks = GPUs (x-decomposition) |
| `FLOAT_TYPE` | `Float32` | `Float32` or `Float64` |
| `NX_PER_GPU` | `128` | Grid points per GPU in x |
| `NY` | `128` | Grid points in y |
| `NZ` | `40` | Grid points in z |
| `WARMUP_STEPS` | `5` | Steps before the profile window opens |
| `PROFILE_STEPS` | `10` | Steps inside `CUDA.@profile` |

The Julia driver also accepts `--dt SECONDS` (default `0.1`) and
`--substeps N` (default `12` — number of acoustic substeps per RK3 stage).

## Why `CUDA.@profile` + `--capture-range=cudaProfilerApi`

Without a capture range, `nsys` would trace from process start — including all
of Julia + CUDA JIT compilation and Oceananigans setup, which dominates a short
run and produces giant trace files. `CUDA.@profile` calls `cudaProfilerStart`
/ `cudaProfilerStop` around the block, and the matching nsys flags tell it to
only record between those calls. The warmup phase (compilation + first kernel
launches) is deliberately outside the profile window.

`NVTX.@range` annotations are picked up via `--trace=...,nvtx` and show as
colored bands on the timeline — useful for landing on a single `time_step!`.

## Multi-rank traces

`nsys` cannot share a single output file across ranks, so the script uses

```
--output=examples/nsys_reports/${SLURM_JOB_ID}/supercell_rank%q{SLURM_PROCID}
```

The `%q{ENV}` token is expanded by `nsys` itself (not by the shell) to the
value of the named environment variable, giving one `.nsys-rep` per rank.

MPI tracing uses `--mpi-impl=mpich` because Perlmutter's Cray MPICH is
ABI-compatible with stock MPICH; OpenMPI builds need `--mpi-impl=openmpi`.

## Viewing reports

Two options:

1. **GUI on a workstation.** Copy reports off Perlmutter and open them in the
   Nsight Systems GUI (`nsys-ui`). The GUI version must be **at least as new**
   as the CLI that produced them (currently 2025.3.x — see the job log).
2. **CLI summaries on Perlmutter.**
   ```bash
   nsys stats examples/nsys_reports/<jobid>/supercell_rank0.nsys-rep
   nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
       examples/nsys_reports/<jobid>/supercell_rank0.nsys-rep
   ```
   Useful pre-canned reports: `cuda_gpu_kern_sum` (kernel time),
   `cuda_gpu_mem_size_sum` (memcpy bytes), `nvtx_sum` (NVTX range timing),
   `mpi_event_sum` (MPI calls).

## Comparing branches

To benchmark an optimization, run the same script on each branch and keep the
report directories alongside each other. The `metric=range_total_time` column
from `nsys stats --report nvtx_sum --filter-include profile_window` is a clean
apples-to-apples comparison.

## Common gotchas

- **Empty trace / "no profile range was entered"** — the `CUDA.@profile` block
  never executed (likely an exception during setup). Check the `.err` file.
- **Huge reports** — drop `--trace=cuda,nvtx,mpi` to just `cuda,nvtx`, or
  shrink `PROFILE_STEPS`. The supplied script already disables CPU sampling
  and context-switch tracing (`--sample=none --cpuctxsw=none`).
- **GPU metrics** — for SM occupancy / DRAM throughput add
  `--gpu-metrics-devices=all --gpu-metrics-frequency=10000`. Not enabled by
  default because it adds noticeable overhead and requires `dcgm` permissions
  that aren't always granted on shared nodes.
- **Module on compute nodes** — the script loads `cudatoolkit` to put `nsys`
  on `PATH`. Override with `NSYS=/path/to/nsys` if you need a specific version.
- **"Importer error … errno 524" when finalizing the report** — nsys cannot
  finalize `.nsys-rep` on `$HOME` (the GPFS/Lustre interaction trips an
  unsupported operation). The script writes streams under
  `$SCRATCH/nsys_reports/<jobid>/` and copies the finalized reports back to
  the repo. If finalize fails on scratch too, the script falls back to
  `QdstrmImporter` to convert `.qdstrm`→`.nsys-rep` post hoc.
- **OpenSSL_jll fails to load** — Julia 1.12.1 on Perlmutter ships libssl 3.3
  but the compute-node libcrypto is older. The script prepends Julia's
  private `lib/julia` to `LD_LIBRARY_PATH` to force the bundled libcrypto.
- **MPI / NVTX missing on the first try** — both need to be direct deps in
  `examples/Project.toml` (already added). `examples/LocalPreferences.toml`
  points MPI.jl at Cray MPICH on Perlmutter; do not delete it.

## Anelastic dynamics on multi-GPU

`AnelasticDynamics` builds a `FourierTridiagonalPoissonSolver` that needs
`poisson_eigenvalues` methods for `Periodic`/`Bounded`/`Flat` topologies. On a
distributed grid partitioned in `x`, the local topology becomes `FullyConnected`
and the solver does not support that yet. For multi-rank profiling, prefer
split-explicit compressible dynamics (the default in `profile_supercell.jl`).
Single-GPU anelastic runs still work via `NGPUS=1`.
