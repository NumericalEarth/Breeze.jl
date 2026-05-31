# Multi-node optimization plan (distributed split-explicit on Perlmutter)

Status of distributed multi-GPU runs and the roadmap to higher scaling efficiency.
Companion to `DISTRIBUTED_TC.md` (the production run) and `optimization/` (the
single-GPU campaign).

## Where we are

Multi-node works and is correct (NCCL+OFI and MPI, validated 1→8 GPU against
analytic truth: reductions, `set!`, Center/XFace/YFace halos, Float32 & Float64).
Two Perlmutter-specific blockers were found and fixed — see `DISTRIBUTED_TC.md`
"Multi-node on Perlmutter":
1. `sanitize_environ!()` after `MPI.Init()` (Cray MPICH malformed-env bug → CUDA.jl
   hang on multi-node).
2. `PLUGIN=2.18.3` AWS-OFI plugin (else NCCL falls back to 10 Gbps TCP, RDMA off).

## Measured weak scaling (124 M cells/GPU = production slab, NCCL+OFI)

| GPUs | ms/step | weak-scaling efficiency (vs 1-GPU) |
|---:|---:|---:|
| 1 | 1498 | 100% (no halo comm) |
| 4 (1 node) | 1904 | 79% |
| 8 (2 nodes) | 2020 | 74% |

**Expected at 60 GPU: ~74%, ~2.0–2.1 s/step** — flat from 8 GPU onward. The halo
exchange is **nearest-neighbor** (2 x-neighbors, independent of rank count); the only
N-dependent cost is the wizard's scalar `max|u|` Allreduce + NaN check (log(N) over
CXI, sub-ms vs a ~2 s step → negligible). Backend comparison at this size:
NCCL+OFI 2020 ms > MPI 2289 ms ≫ NCCL-socket(broken) 6768 ms.

Two separable costs:
- **Per-step efficiency ~74%** — the ~520 ms/step halo overhead (this doc's target).
- **Step count** — Δt≈0.55 s ⇒ ~157k steps per 24 h stage, set by GATE's 50 m surface
  Δz (vertical CFL), *orthogonal* to scaling. Relaxing the near-surface Δz is the
  lever there (see DISTRIBUTED_TC.md).

## Improvements, ranked by payoff

### 1. Phase-5 active-halo substepping (biggest win)

The split-explicit substepper currently **refills halos every acoustic substep
(~36 exchanges/step)** — that is essentially the entire ~520 ms/step overhead.

Fix (from `optimization/OPT_FINAL_REPORT.md` "Multi-GPU outlook"):
- Lift `halo_width` from 0 to `Hx`/`Hy` in `acoustic_kernel_parameters.jl` so the
  substep kernels launch over interior **+ active halos** in one shot, so halos stay
  valid across several substeps without a refill between kernels.
- Plumb `KernelParameters` through `BatchedTridiagonalSolver`'s `solve!` for
  active-halo tridiagonal launches.
- Hoist `configure_kernel` + `convert_to_device` + `GC.@preserve` out of the substep
  loop (per Oceananigans' split-explicit free-surface pattern): ~30 µs × 6 kernels
  × ~220 substeps ≈ 40 ms/run.

Prerequisite already in place: Phase-2C topology-aware operators
(`src/CompressibleEquations/acoustic_operators.jl`).

Expected: cuts ~36 exchanges/step → a handful; efficiency toward **85–90%+**.
Lives in `src/CompressibleEquations/acoustic_substepping.jl`.

### 2. Comm/compute overlap (async halos)

`OceananigansNCCLExt` already supports an `async` halo mode (defer unpack; NCCL runs
on a dedicated `comm_stream` while interior compute proceeds on the default stream,
then `synchronize_communication!` unpacks). The substepper uses the **synchronous**
path today, so the ~520 ms is fully exposed. Wiring async fills + overlapping the
interior tendency computation could hide much of the halo cost behind compute.

### 3. Situational / minor

- **2D partition** `Partition(Rx, Ry)`: squarer tiles vs thin 107-cell x-slabs;
  mainly relevant if pushing well past 60 ranks. Unverified for the substepping path.
- **Fewer acoustic substeps** (currently 12): fewer fills, but trades acoustic
  stability — not free.
- Halo width is pinned at 5 by WENO5; can't shrink without lowering advection order.

## Recommendation

74% is acceptable to run the production campaign now (good for a 1D-partitioned
stencil code crossing nodes). The single highest-value multi-node improvement is
**#1 (active-halo substepping)** — it is also the documented next step of the
single-GPU optimization campaign, and roughly halves the comm overhead. Combined
with relaxing the 50 m surface Δz (step-count lever), it is the realistic path to a
substantially shorter multi-day campaign.
