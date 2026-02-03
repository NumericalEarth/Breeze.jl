# DelinearizingIndexPassing Segfault Investigation Tests (Midstream)

**Investigation:** DelinearizingIndexPassing Segmentation Fault (B.6.3)  
**Related:** `cursor-toolchain/rules/domains/differentiability/investigations/delinearizing-segfault.md`  
**Synchronized with:** `Manteia.jl/test/delinearize/` (downstream)

## ⚠️ PR Dependency Notice

**This folder contains the primary investigation tests.** The Manteia.jl downstream tests depend on functionality provided by `BreezeReactantExt`, so compilation and gradient computation must be validated here in Breeze.jl first before downstream packages can use it.

When the relevant Breeze PR is merged, downstream packages (Manteia) can then run their own tests.

## Problem Summary

Segmentation faults occur during Reactant/Enzyme autodiff compilation or execution when using periodic topology grids with certain Julia runtime options (ParallelTestRunner, bounds checking). The error messages reference "DelinearizingIndexPassing" which is an MLIR pass related to array index computation.

The issue appears closely linked to:
- Using `ParallelTestRunner`
- Having `--check-bounds=yes` enabled
- `(Periodic, Periodic, Flat)` topology
- `fill_halos!` operations during traced loops

## Current Status

| Component | Status |
|-----------|--------|
| Periodic topology (no ParallelTestRunner) | ⚠️ May work |
| Periodic topology + ParallelTestRunner | ❌ Segfault |
| Periodic topology + check-bounds=yes | ❌ Segfault |
| Root cause identified | ❌ Not yet |

## Role in Package Hierarchy

```
Manteia (downstream) ← awaiting this PR
    ↓
Breeze (midstream) ← YOU ARE HERE (primary investigation)
    ↓
Oceananigans (upstream)
    ↓
Enzyme / KernelAbstractions / Reactant (far upstream)
```

This is the **midstream** test location. BreezeReactantExt provides critical extensions needed for model compilation with Reactant. Once issues are resolved here, fixes propagate downstream to Manteia.

## Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test_delinearize_breeze_medwe.jl` | Full MedWE with Breeze AtmosphereModel | ❌ Fails with --check-bounds=yes |
| `test_delinearize_oceananigans_medwe.jl` | Full MedWE with Oceananigans HydrostaticFreeSurfaceModel | ❌ Fails with --check-bounds=yes |
| `test_delinearize_fill_halos_medwe.jl` | Minimal MedWE focusing on fill_halo_regions! only | ✅ **PASSES** |
| `test_delinearize_timestep_components_medwe.jl` | Progressive test: set! → update_state! → time_step! | Test 1 FAILS |
| `test_delinearize_halo_size_mwe.jl` | **KEY MWE**: Tests halo=1,2,3 - isolates root cause | halo=1 ✅, halo>=2 ❌ |
| `test_delinearize_set_mwe.jl` | set!(model, T=...) - fails because it calls fill_halos internally | ❌ FAILS (halo=3) |
| `test_delinearize_field_ops_mwe.jl` | Progressive: parent()→set!()→update_state!() | Test 3+ fail |
| `test_delinearize_array_ops_mwe.jl` | Pure Reactant array ops: broadcast, view, struct | ✅ All pass |
| `test_delinearize_periodic_indexing_mwe.jl` | Near-MWE with NO Oceananigans - pure Reactant periodic indexing | ⚠️ Scalar indexing issue (not relevant) |

### Key Finding (2026-02-03) - ROOT CAUSE IDENTIFIED

**The issue is `fill_halo_regions!` with `halo >= 2`**

| Halo Size | --check-bounds=no | --check-bounds=yes |
|-----------|-------------------|-------------------|
| `halo=1`  | ✅ PASSES | ✅ PASSES |
| `halo=2`  | ❌ "failed to raise func" | ❌ Segfault |
| `halo=3`  | ❌ "failed to raise func" | ❌ Segfault |

**Why `set!(model, T=field)` fails:** It internally calls `initialization_update_state!` 
which calls `fill_halo_regions!` on tracer fields (with halo=3 typically).

**Root cause:** The periodic halo kernels have loops `for i = 1:H` where H is halo size.
When H>=2, Reactant's MLIR pass manager can't handle the complex affine index expressions
created by loop unrolling. See `fill_halo_regions_periodic.jl`:

```julia
@inbounds for i = 1:H  # ← THIS LOOP IS THE PROBLEM WHEN H >= 2
    parent(c)[i, j, k]     = parent(c)[N+i, j, k]
    parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k]
end
```

**Earlier tests that passed used `halo=1`**, which is why we incorrectly thought fill_halos worked.
- Oceananigans' KernelAbstractions-based halo filling works correctly with Reactant
- The DelinearizingIndexPassing segfault is triggered by something ELSE in the model time-stepping
- The issue is NOT in the periodic boundary halo exchange itself

### Test Hierarchy (from most to least complex)

```
test_delinearize_breeze_medwe.jl          ← Full Breeze model (FAILS with --check-bounds=yes)
test_delinearize_oceananigans_medwe.jl    ← Full Oceananigans model (FAILS with --check-bounds=yes)
test_delinearize_fill_halos_medwe.jl      ← Just fill_halo_regions! (PASSES ✅)
```

### Next Investigation: What's Different in time_step!?

Since fill_halos works in isolation, the culprit must be in the model's `time_step!`:
1. **Tendency computations** - stencil operations in compute_tendencies!
2. **Pressure corrections** - even ExplicitFreeSurface has some operations
3. **Time stepper internals** - SSP RK3 substeps, state storage/restoration
4. **Field operations** - set!, interior access patterns
5. **Kernel parameter dispatch** - workgroup size calculations

## How to Run

Run tests individually using the **test environment** (`--project=test`):

```bash
cd Breeze.jl

# Run with default settings (may work)
julia --project=test test/delinearize/test_delinearize_breeze_medwe.jl

# Run with bounds checking enabled (triggers segfault)
julia --project=test --check-bounds=yes test/delinearize/test_delinearize_breeze_medwe.jl

# Run the component breakdown test to isolate the issue
julia --project=test --check-bounds=yes test/delinearize/test_delinearize_timestep_components_medwe.jl

# Run fill_halos test (should PASS)
julia --project=test --check-bounds=yes test/delinearize/test_delinearize_fill_halos_medwe.jl
```

**Note:** Use `--project=test` (not `--project=.`) to use the test environment defined in `test/Project.toml`.

**Warning:** These tests may segfault. Run them in isolation, not as part of a larger test suite.

## Investigation Notes

### What "DelinearizingIndexPassing" Means

In XLA/MLIR compilation, "delinearizing" refers to converting linear (1D) array indices back to multi-dimensional indices. This is a common operation when:
- Copying data between arrays with different layouts
- Implementing halo exchanges
- Performing stencil operations

The segfault suggests that the index conversion is producing invalid memory addresses.

### Relationship to B.6.4 (Bounded Topology Segfault)

Both B.6.3 (this issue) and B.6.4 involve `fill_halos!` but manifest differently:
- **B.6.3**: Periodic topology + ParallelTestRunner + bounds checking → DelinearizingIndexPassing
- **B.6.4**: Bounded topology + nsteps > 1 → SinkDUS

## Next Steps

1. **Characterize the failure**: Run MedWE with various Julia options
2. **Isolate triggers**: Determine exact combination of flags that cause segfault
3. **Simplify to MWE**: Remove Breeze-specific code while preserving the bug
4. **File upstream issue**: Once we have a pure Reactant+Enzyme MWE
5. **Merge PR**: Once fixed, merge and notify downstream (Manteia)

## Package Version Requirements

Record versions when reporting results:

```julia
@info "Package versions" Breeze=pkgversion(Breeze) Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
```
