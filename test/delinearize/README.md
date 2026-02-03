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
| `test_delinearize_fill_halos_medwe.jl` | Minimal MedWE focusing on fill_halo_regions! only | Active investigation |
| `test_delinearize_periodic_indexing_mwe.jl` | Near-MWE with NO Oceananigans - pure Reactant periodic indexing | Active investigation |

### Test Hierarchy (from most to least complex)

```
test_delinearize_breeze_medwe.jl          ← Full Breeze model (FAILS)
test_delinearize_oceananigans_medwe.jl    ← Full Oceananigans model (FAILS)
test_delinearize_fill_halos_medwe.jl      ← Just fill_halo_regions! (test this)
test_delinearize_periodic_indexing_mwe.jl ← Pure Reactant arrays (test this)
```

If the pure Reactant test fails, the issue is in Reactant's handling of array operations with bounds checking.
If only the fill_halos test fails, the issue is in how Oceananigans implements halo filling.

## How to Run

Run tests individually to isolate the issue:

```bash
# Run with default settings (may work)
julia --project=. test/delinearize/test_delinearize_breeze_medwe.jl

# Run with bounds checking enabled (likely to trigger segfault)
julia --project=. --check-bounds=yes test/delinearize/test_delinearize_breeze_medwe.jl
```

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
