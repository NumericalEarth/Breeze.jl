# Configuring NCCL (and multi-node GPU) on Perlmutter

Hard-won recipe for running distributed Oceananigans/Breeze on NERSC Perlmutter with
`NCCLDistributed` (or plain `Distributed` over Cray MPICH). Every item below cost real
debugging time; the SLURM recipe in `examples/distributed_tropical_cyclone.sh` and
`examples/profile_supercell.sh` bakes these in.

TL;DR: with the right OFI plugin + env, **NCCL is the fastest backend** (CXI/GPUDirect
RDMA). Without it, NCCL silently drops to 10 Gbps TCP and is ~3× slower than MPI. And on
multi-node, *nothing* works (any backend) until you sanitize the environment after
`MPI.Init()`.

---

## 0. The two showstoppers (fix these first)

### (a) Multi-node hang → sanitize the environment after `MPI.Init()`

Cray MPICH inserts a **malformed environment entry** (a string with no `=`) after
`MPI_Init` on multi-node `srun`. CUDA.jl chokes on it and **every multi-node GPU run
hangs at the first device/halo operation**. Single-node runs are unaffected, so this
hides easily and looks like a comms/transport hang.

Symptom in logs (flooding stderr):
```
┌ Warning: malformed environment entry
│   env = ""
```

Fix — call once immediately after `MPI.Init()` / constructing the `Distributed` arch:
```julia
using MPI; MPI.Init()
# Oceananigans ≥ this branch: Oceananigans.DistributedComputations.sanitize_environ!()
# or include the standalone module:
include("sanitize_environ.jl"); SanitizeEnviron.sanitize_environ!()
```
(`examples/sanitize_environ.jl`; also added to Oceananigans as
`DistributedComputations.sanitize_environ!`. Ref: CliMA/Oceananigans.jl discussion #5513.)

### (b) NCCL inter-node is slow → it has no network plugin (uses TCP sockets)

Julia's `NCCL.jl` uses its **own bundled libnccl** (`NCCL_jll`), which on Perlmutter's
Slingshot network needs the **`aws-ofi-nccl` plugin** (`libnccl-net.so`) on
`LD_LIBRARY_PATH`. Without it, NCCL falls back to **TCP sockets with GPUDirect RDMA
disabled, defaulting to 10 Gbps** — inter-node halo exchange is ~3× slower than MPI.

How to tell (NCCL prints to **stdout**, not stderr — see §3):
```
NET/Plugin: Could not find: libnccl-net.so     # BAD: no plugin
Using network Socket
NET/Socket : GPU Direct RDMA Disabled for HCA 'hsn0'
Could not get speed from /sys/.../speed. Defaulting to 10 Gbps.
```
vs. the good case:
```
NET/Plugin: Loaded net plugin AWS Libfabric (v6)
NET/OFI Using aws-ofi-nccl 1.6.0-hcopy ... Selected Provider is cxi
Using network AWS Libfabric
NET/AWS Libfabric : GPU Direct RDMA Enabled for HCA 'cxi0..3'
... via NET/AWS Libfabric/0/GDRDMA/Shared
```

---

## 1. The OFI plugin: match it to `NCCL_jll`

`NCCL_jll` currently provides **NCCL 2.28.3 (cuda 12.9)**. NERSC ships several plugins
under `/global/common/software/nersc9/nccl/<version>/lib/libnccl-net.so`. Match the
**net-plugin API version**, not the exact NCCL version:

| Plugin dir | Result with NCCL_jll 2.28.3 |
|---|---|
| `2.17.1-cuda11` | ❌ fails — masked by NCCL.jl `err` bug (see §3) |
| **`2.18.3`** | ✅ loads as **AWS Libfabric v6**, CXI/RDMA enabled |

So put the 2.18.3 plugin first on `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.18.3/lib:${LD_LIBRARY_PATH}"
```
(Find the plugin: `find /global/common/software/nersc9/nccl -name 'libnccl-net.so'`.
Only `2.15.5`, `2.17.1-cuda11`, `2.18.3` actually ship the plugin; newer dirs are NCCL
builds, not plugins. Re-check `NCCL_jll` version when packages update:
`julia --project=examples -e 'using NCCL; println(NCCL.version())'`.)

## 2. Full environment recipe (SLURM)

```bash
module load julia/1.12.1

# libssl: Julia 1.12.1 ships OpenSSL_jll needing newer symbols than the compute-node libs.
JULIA_PREFIX=$(dirname "$(dirname "$(readlink -f "$(command -v julia)")")")
export LD_LIBRARY_PATH="${JULIA_PREFIX}/lib/julia:${LD_LIBRARY_PATH:-}"

# GPU-aware Cray MPICH (needed by plain Distributed AND by NCCL's MPI-based bootstrap)
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD="${CRAY_MPICH_ROOTDIR:-/opt/cray/pe/mpich/9.0.1}/gtl/lib/libmpi_gtl_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"

# CUDA.jl pool OFF: Cray MPICH cuIpcGetMemHandle rejects pool allocations.
export JULIA_CUDA_MEMORY_POOL=none

# NCCL: AWS-OFI plugin + Slingshot/CXI tuning
export NCCL_SOCKET_IFNAME=hsn                       # Slingshot NICs (bootstrap/socket fallback)
export LD_LIBRARY_PATH="/global/common/software/nersc9/nccl/2.18.3/lib:${LD_LIBRARY_PATH}"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

srun --ntasks=$NGPUS --gpu-bind=none julia --project=examples your_script.jl
```
SLURM: one MPI rank per GPU, `--gpu-bind=none`, `#SBATCH --gpus-per-node=4 --ntasks-per-node=4`.
MPI.jl must point at system Cray MPICH (set once via `examples/LocalPreferences.toml`;
do not delete it).

## 3. Debugging NCCL: it lies quietly

- **NCCL.jl `err`-masking bug:** when an NCCL call errors (e.g. an incompatible plugin),
  NCCL.jl's own error path throws `UndefVarError: err not defined in NCCL.LibNCCL`,
  **hiding the real error**. Don't trust that message — get NCCL's own diagnostics.
- **`NCCL_DEBUG` goes to stdout**, not stderr. To see transport selection:
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,NET
  ```
  Then grep the `.out` (not `.err`) for `Using network …` and `GPU Direct RDMA`.

## 4. What the NCCL extension does (and doesn't)

`OceananigansNCCLExt` (`NCCLDistributed`):
- **Halo exchange** uses NCCL Send/Recv on a dedicated `comm_stream` with CUDA events.
- **Reductions (`Allreduce`) forward to MPI**, not NCCL — so the CFL wizard's global
  `min`/`max` go over Cray MPICH regardless of backend.
- `sync_device!(::NCCLDistributedArchitecture) = nothing` (NCCL is stream-native).
- Bootstrap uses `MPI.Bcast` of the NCCL `UniqueID` — so plain MPI must work first.

## 5. Measured backend performance (124 M cells/GPU, split-explicit, this code)

| GPUs | NCCL **socket** (no plugin) | MPI (Cray-MPICH) | **NCCL + OFI/RDMA** |
|---:|---:|---:|---:|
| 4 (1 node) | 6677 ms/step | 2201 | **1904** |
| 8 (2 nodes) | 6768 ms/step | 2289 | **2020** (~74% weak-scaling eff) |

→ Use **NCCL + the OFI plugin**. MPI is a solid fallback (`--no-nccl`). Socket-fallback
NCCL is a trap — fast to set up, 3× too slow.

## 6. Known open issue

In-model `maximum(abs, ·)` over **face** fields (used by the CFL time-step wizard) returns
flaky/zero global values in the full-model context (correct in isolation — see
`examples/test_nccl_correctness.jl`). Suspected stream-sync interaction. **Workaround:**
run with a fixed `Δt` (`--dt`, wizard off) until fixed; bare-field reductions and the
dynamics are correct.
