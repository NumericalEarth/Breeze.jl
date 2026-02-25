# Investigation: Backward Pass Rank Mismatch (B.6.5) + Buffer Donation + NaN Gradients

**Status:** MIXED (as of 2026-02-23)
**Investigation file:** `cursor-toolchain/rules/domains/differentiability/investigations/backward-pass-rank-mismatch.md`

## Current bugs (collocated here since they all affect BPB and related topologies)

### 1. Rank mismatch (RESOLVED)
The original `dynamic_update_slice` rank mismatch error no longer reproduces as of
Reactant v0.2.226. Backward pass compilation succeeds for all tested topologies
including Bounded.

### 2. Buffer donation error (ACTIVE)
When executing the compiled `grad_loss`, Reactant throws:
```
Donated buffer ... is already marked as donated. Can't donate the same buffer multiple times.
```
Observed on BPB (Bounded, Periodic, Bounded) with nsteps=1 and nsteps=4.
Also observed on PPP in the MedWE. Likely due to `model` and `dmodel` sharing
underlying `ConcretePJRTArray` buffer references after `Enzyme.make_zero`.

### 3. NaN gradients (ACTIVE)
When the buffer donation error does not crash (e.g., earlier runs with nsteps=4 on BPB),
the backward pass returns `loss_val = NaN` and `dθ` filled with NaN. This persists
regardless of timestep size (tested Δt = 0.1, 0.001, 0.00001, 1.0), ruling out CFL
violation. Points to a structural issue in the generated adjoint or checkpointing.

## Test files

| File | Type | Uses |
|------|------|------|
| `test_medwe_bounded_backward.jl` | MedWE | Breeze + Oceananigans + Reactant + Enzyme |
| `test_mwe_rank_mismatch.jl` | MWE | Pure Reactant + Enzyme + KA (no Oceananigans) |

## Topology matrix test

The main test file `Breeze.jl/test/reactant_compressible_dynamics_models.jl` now tests
a full topology matrix:

- **2D:** PPF, BBF
- **3D:** PPP, PBB, PPB, BBB

Each topology is tested with `nsteps=1` (no checkpointing) and `nsteps=9` (checkpointed).

## How to run

```bash
cd Breeze.jl

# Full topology matrix
julia --project test/reactant_compressible_dynamics_models.jl

# MedWE (reproduces buffer donation bug)
julia --project test/backward-pass-rank-mismatch/test_medwe_bounded_backward.jl

# MWE (attempts to isolate to pure Reactant/Enzyme)
julia --project test/backward-pass-rank-mismatch/test_mwe_rank_mismatch.jl
```

MLIR dumps are written to `test/backward-pass-rank-mismatch/mlir_dump/`.

## Synchronized with

- Investigation: `cursor-toolchain/rules/domains/differentiability/investigations/backward-pass-rank-mismatch.md`
- Original test: `Breeze.jl/test/reactant_compressible_dynamics_models.jl`
