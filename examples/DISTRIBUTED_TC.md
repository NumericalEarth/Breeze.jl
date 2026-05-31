# Distributed tropical-cyclone rainband run (100 m production)

Multi-GPU version of `examples/tropical_cyclone_with_rainband.jl` (YuDidlake2019).
Same science; distributed over an x-partition with NCCL (MPI fallback), fields
streamed to disk, figures deferred to a post-processing job.

- Driver: `examples/distributed_tropical_cyclone.jl`
- SLURM:  `examples/distributed_tropical_cyclone.sh`

## Why this works multi-GPU at all

The case runs `CompressibleDynamics(SplitExplicitTimeDiscretization)` — acoustic
substepping, **no Poisson solve**. The anelastic FFT/tridiagonal Poisson solver
has no distributed-x implementation in Breeze (see `examples/PROFILING.md`), so
split-explicit is the only distributed-capable path — and it is exactly the path
the single-GPU optimization campaign (`optimization/OPT_FINAL_REPORT.md`, −19%)
tuned. Phase 2C (topology-aware operators, in-loop halo-fill elimination) was
single-GPU-neutral but kept **because each removed halo fill is an avoided
NCCL/MPI exchange** — it pays off here.

## Calibration anchor (this branch, optimized)

From the optimized single-GPU profile (`optimization/`, job 53573262):

| Quantity | Value |
|---|---|
| Grid | 1024 × 1024 × 128 = 134.2 M cells, Float32, 12 substeps |
| GPU memory used | 24.09 GiB on A100-40GB |
| Per-step wall (optimized) | ~1180 ms/step |

→ **~190 bytes/cell** all-in, **~8.8 ns/cell/step**. These two numbers drive the
sizing below.

## Vertical grid

Two options (`--vertical`):

- **`gate`** (default) — SAM `GATE_IDEAL` stretched grid from PR #397, built with
  `PiecewiseStretchedDiscretization(z=[0,1275,5100,18000,27000], Δz=[50,50,100,100,300])`:
  **181 levels**, 50 m at the surface → 100 m through the troposphere → 300 m
  aloft, model top **27 km**, sponge from 19 km. This is the resolution you want
  for a 100 m horizontal LES of a TC — the boundary layer and cloud layer are
  resolved at 50–100 m.
- **`uniform`** — the example's flat grid: `--nz` levels over 25 km
  (Δz ≈ 333 m at Nz = 75), sponge 20–25 km. Cheaper, coarse aloft.

## Production problem size (Δx = 100 m)

YD19 horizontal domain 642 km × 642 km → **Nx = Ny = 6420** at 100 m.
x-partition requires `Ngpus` to divide 6420 (= 2²·3·5·107). Useful divisors:
**20, 30, 60**, then 107, 214, …

| Vertical | Nz | Total cells | Field memory | Min GPUs (A100-40GB) | Per-GPU |
|---|---:|---:|---:|---:|---:|
| **gate** | 181 | 7.46 B | ~1.32 TiB | **60** (15 nodes) | 124 M cells, ~22 GiB |
| uniform | 75 | 3.09 B | ~547 GiB | **20** (5 nodes) | 154 M cells, ~27 GiB |
| uniform | 100 | 4.12 B | ~729 GiB | **30** (≈8 nodes) | 137 M cells, ~24 GiB |

Budget assumes ~32 GiB usable per 40-GB GPU (headroom for x-halos, stepping
temporaries, and the disabled memory pool). On A100-80-GB nodes these counts
roughly halve (GATE fits on ~20).

**Recommended first attempt: GATE grid, 60 GPUs (15 nodes).** This is the memory
minimum for the scientifically appropriate vertical resolution.

## Wall-time estimate (why we calibrate first)

Per-step ≈ (cells/GPU) × 8.8 ns + ~15 % halo exchange. The step **count** is set
by the CFL wizard's Δt at 100 m, which we don't know a priori — hence the
`--benchmark-steps` mode runs the *real* wizard and reports the actual Δt.

Back-of-envelope, GATE grid at 60 GPUs (124 M cells/GPU ≈ 1.26 s/step):

| Δt (wizard) | steps / 24 h sim | wall / 24 h stage |
|---:|---:|---:|
| 1.0 s | 86 400 | ~30 h |
| 0.5 s | 172 800 | ~60 h |

A single 24 h stage at the memory-minimum count **exceeds a 24 h queue window**.
Options to fit:
1. **More GPUs** — to finish a 24 h stage in <24 h wall at Δt≈1 s you need
   per-step ≲1 s → ≲100 M cells/GPU → ~107 GPUs (≈27 nodes) for GATE.
2. **Checkpoint / restart** across jobs (add a `Checkpointer`; not yet wired in).

Note the full YD19 experiment is **three stages** (spinup + control + heated),
so multiply accordingly. The benchmark step removes the guesswork.

## How to run

**1. Calibrate (short, debug queue) — get real ms/step + wizard Δt:**

```bash
NGPUS=60 DX=100 VERTICAL=gate BENCHMARK_STEPS=10 \
  sbatch --qos=debug --time=00:30:00 --nodes=15 \
  examples/distributed_tropical_cyclone.sh
```

Read the projection line in the `.err`/`.out`, then pick the production GPU count.

**2. Production spinup (24 h, 3D output every 1 h):**

```bash
NGPUS=60 DX=100 VERTICAL=gate STOP_TIME=24 OUTPUT_INTERVAL=1 \
  sbatch --qos=regular --time=24:00:00 --nodes=15 \
  examples/distributed_tropical_cyclone.sh
```

**2b. Resume after the wall clock (or a crash):** resubmit the *same* command with
`RESTART=1` — `run!` picks up from the latest `tc_<stage>_checkpoint_*` in `$SCRATCH`.
Chain these to span the multi-day integration across ≤48 h jobs.

**3. Heated stage:** add `HEATING=1`.

Output lands in `examples/output_tc_distributed/tc_<stage>_dx100m_gate_nz181_rank*.jld2`
(one file per rank, each holding its x-slab of `u, v, w, T, ρ`). The figure job
reassembles the slabs and does the azimuthal averaging.

## Driver flags (`distributed_tropical_cyclone.jl`)

| Flag | Default | Meaning |
|---|---|---|
| `--dx` | 100 | horizontal resolution (m); sets Nx=Ny=round(642 km/Δx) |
| `--vertical` | gate | `gate` (181-level stretched, top 27 km) or `uniform` |
| `--nz` | 75 | vertical levels over 25 km (uniform mode only) |
| `--stop-time` | 24 | integration length (hours) |
| `--output-interval` | 1 | 3D snapshot cadence (hours) |
| `--benchmark-steps` | 0 | >0: time N wizard steps, print projection, exit |
| `--warmup-steps` | 3 | steps before the benchmark window |
| `--checkpoint-interval` | 6 | full-state checkpoint cadence (sim-hours) |
| `--restart` | off | pick up from the latest checkpoint in the output dir |
| `--heating` | off | enable MN10 rainband heating (else spinup) |
| `--no-nccl` | (NCCL on) | fall back to plain Cray-MPICH |
| `--float-type` | Float32 | `Float32` or `Float64` |

## Multi-node on Perlmutter — required setup (learned the hard way)

Two things must be right or multi-node runs hang or crawl. Both are handled by
`distributed_tropical_cyclone.sh`; documented here so they aren't lost.

1. **Sanitize the environment after `MPI.Init()`.** Cray MPICH inserts a malformed
   env entry (no `=`) after `MPI_Init` on multi-node `srun`; CUDA.jl chokes on it
   and **every multi-node GPU run hangs** at the first halo exchange (1-node is
   unaffected). Fix: `include("sanitize_environ.jl"); SanitizeEnviron.sanitize_environ!()`
   right after `MPI.Init()` (Oceananigans discussion #5513, romanlee). Already wired
   into the driver.

2. **Give NCCL the AWS-OFI plugin** (`PLUGIN=2.18.3`, default in the `.sh`). Without
   `libnccl-net.so` on `LD_LIBRARY_PATH`, NCCL silently falls back to **10 Gbps TCP
   sockets with GPUDirect RDMA disabled** — inter-node runs ~3.3× slower. With the
   plugin it uses CXI/Slingshot RDMA. `NCCL_jll` is 2.28.3; 2.18.3 is the newest
   NERSC plugin that loads (AWS Libfabric v6). NB: NCCL.jl's error path has an
   `err`-not-defined bug that *masks* plugin failures — check `NCCL_DEBUG=INFO`
   (stdout) for `Using network AWS Libfabric` vs `Using network Socket`.

## Backend & scaling (measured, 124 M cells/GPU = production slab)

Weak scaling, ms/step (`examples/weak_scaling_sweep.sh`):

| GPUs | NCCL (socket, broken) | MPI (Cray-MPICH) | **NCCL + OFI/RDMA** |
|---:|---:|---:|---:|
| 1 (compute) | 1495 | 1500 | 1498 |
| 4 (1 node)  | 6677 | 2201 | **1904** |
| 8 (2 nodes) | 6768 | 2289 | **2020** |

→ **Use NCCL with the OFI plugin** (default): fastest, ~74% weak-scaling efficiency,
+6% across the node boundary. MPI is a solid fallback (`--no-nccl`, 66%). The
60-GPU run is ~2.0 s/step ⇒ a 24 h spinup at Δt≈1 s is ~48 h wall — at the regular
queue limit, so **checkpoint/restart is needed** for the full integration.

## Known gaps / TODO

- **Checkpoint/restart is wired in** (`Checkpointer` every `--checkpoint-interval`
  sim-hours, `cleanup=true`; resume with `RESTART=1`). Still needs a real multi-job
  end-to-end test at scale to confirm distributed pickup.
- Output is per-rank x-slabs; the reassembly/figure job is separate (not in this PR).
- Δz ≈ 333 m at Nz=75; revisit a stretched vertical grid if upper-level wave
  reflection becomes an issue despite the sponge.
