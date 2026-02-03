# DelinearizeIndexingPass Segfault Investigation (B.6.3)

**Investigation:** Segfault with `halo >= 2` + `--check-bounds=yes`  
**Related:** `cursor-toolchain/rules/domains/differentiability/investigations/delinearizing-segfault.md`  
**Synchronized with:** `Manteia.jl/test/delinearize/` (downstream)

## 🔴 ROOT CAUSE IDENTIFIED

**The issue is `halo >= 2` + `--check-bounds=yes`**

| Halo Size | `--check-bounds=no` | `--check-bounds=yes` |
|-----------|---------------------|----------------------|
| `halo=1`  | ✅ Works | ✅ Works |
| `halo≥2`  | ⚠️ May hit B.6.2 (grid size) | ❌ **Segfault (this bug)** |

**Root cause:** Periodic halo kernels have loops with index arithmetic (`for i = 1:H`). When `H > 1`:
- The loop variable `i` appears in index expressions: `i`, `N+i`, `H+i`, `N+H+i`
- With `--check-bounds=yes`, Julia inserts bounds checking for each access
- The combination creates complex MLIR that crashes `DelinearizeIndexingPass`

**Note:** "failed to raise func" with `--check-bounds=no` is typically B.6.2 (grid size issue), not this bug.

**Key MWE:** `test_reactant.jl` (pure KernelAbstractions, no Oceananigans)

## Status: Awaiting Upstream Reactant Fix

Requires Reactant fix to handle bounds checking code with loop-dependent index expressions.

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
| **`test_reactant.jl`** | **UPSTREAM MWE**: Pure KernelAbstractions, no Oceananigans | Use for filing Reactant issue |
| `test_delinearize_halo_size_mwe.jl` | Tests halo=1,2,3 with Oceananigans | halo=1 ✅, halo≥2 + check-bounds ❌ |
| `test_delinearize_breeze_medwe.jl` | Full MedWE with Breeze AtmosphereModel | ❌ Fails with --check-bounds=yes |
| `test_delinearize_oceananigans_medwe.jl` | Full MedWE with Oceananigans | ❌ Fails with --check-bounds=yes |
| `test_delinearize_fill_halos_medwe.jl` | fill_halo_regions! only (with halo=1) | ✅ PASSES |
| `test_delinearize_array_ops_mwe.jl` | Pure Reactant array ops | ✅ All pass |

### Key Finding (2026-02-03) - ROOT CAUSE IDENTIFIED

**The segfault requires both: `halo >= 2` AND `--check-bounds=yes`**

**Why H > 1 + check-bounds causes segfault:**

```julia
@inbounds for i = 1:H
    parent(c)[i, j, k]     = parent(c)[N+i, j, k]     # indices: i, N+i
    parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k]     # indices: N+H+i, H+i
end
```

1. **Loop variable in indices**: `i` appears in 4 index expressions per iteration
2. **Bounds checking code**: Julia inserts `if !(1 <= idx <= size) throw(BoundsError)` for EACH access
3. **Complex conditional MLIR**: Loop-dependent indices inside conditional bounds checks
4. **DelinearizeIndexingPass crash**: Cannot handle the complex affine expressions

**Why `--check-bounds=no` avoids segfault:**
- No bounds checking → simpler MLIR → no crash
- But may still hit B.6.2 ("failed to raise func") if grid is large

### Test Hierarchy (from most to least complex)

```
test_delinearize_breeze_medwe.jl          ← Full Breeze model (FAILS with --check-bounds=yes)
test_delinearize_oceananigans_medwe.jl    ← Full Oceananigans model (FAILS with --check-bounds=yes)
test_delinearize_fill_halos_medwe.jl      ← Just fill_halo_regions! (PASSES ✅)
```

### Resolution Path

1. ✅ Root cause identified: `halo >= 2` in periodic halo kernels
2. **File upstream Reactant issue** with `test_delinearize_halo_size_mwe.jl` as MWE
3. Potential workaround: Manually unroll halo loops in Oceananigans (not ideal)
4. Wait for Reactant fix to handle KA kernels with runtime-dependent loops

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

B.6.3 and B.6.4 likely share the same root cause:
- **B.6.3**: Periodic halo kernels with `halo >= 2` → fail to raise / segfault
- **B.6.4**: Bounded halo kernels may have similar loop structures

Once B.6.3 is fixed upstream in Reactant, B.6.4 may also be resolved.

## Next Steps

1. ✅ **Root cause identified**: `fill_halo_regions!` with `halo >= 2`
2. **File upstream Reactant issue** using `test_delinearize_halo_size_mwe.jl` as MWE
3. Wait for Reactant fix or implement workaround in Oceananigans
4. **Merge PR**: Once fixed, merge and notify downstream (Manteia)

## Package Version Requirements

Record versions when reporting results:

```julia
@info "Package versions" Breeze=pkgversion(Breeze) Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
```
