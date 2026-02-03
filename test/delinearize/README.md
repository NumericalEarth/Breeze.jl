# DelinearizeIndexingPass Segfault Investigation (B.6.3)

**Investigation:** Segfault with `halo >= 2` + `--check-bounds=yes`  
**Related:** `cursor-toolchain/rules/domains/differentiability/investigations/delinearizing-segfault.md`  
**Synchronized with:** `Manteia.jl/test/delinearize/` (downstream)

## 🟡 PARTIAL FIX IN REACTANT v0.2.211

**The issue is `halo >= 2` in KernelAbstractions kernels with loop-dependent indices.**

### Version-Dependent Behavior

| Reactant Version | `halo≥2` Behavior | Notes |
|------------------|-------------------|-------|
| < v0.2.211       | ❌ **Segfault** | Process crashes |
| **v0.2.211+**    | ❌ "failed to raise func" | Clean error, no crash |

**Recommendation:** Use **Reactant v0.2.211+** to get clean error messages instead of segfaults.

### Tested Versions (2026-02-03)
- Reactant: v0.2.211
- Enzyme: v0.13.129
- KernelAbstractions: v0.9.39

### Behavior with Reactant v0.2.211+

| Halo Size | `--check-bounds=no` | `--check-bounds=yes` |
|-----------|---------------------|----------------------|
| `halo=1`  | ✅ Works | ✅ Works |
| `halo≥2`  | ❌ StableHLO shape error | ❌ "failed to raise func" |

**Two different error paths:**
- `--check-bounds=yes`: "failed to raise func" (MLIR affine expressions)
- `--check-bounds=no`: StableHLO `dynamic_update_slice` shape mismatch (workgroup vs array size)

**Root cause:** Periodic halo kernels have loops with index arithmetic (`for i = 1:H`). When `H > 1`:
- The loop variable `i` appears in index expressions: `i`, `N+i`, `H+i`, `N+H+i`
- Complex MLIR affine expressions are generated that the compiler cannot raise

**Key MWE:** `test_reactant.jl` (pure KernelAbstractions, no Oceananigans)

## Status: Awaiting Full Upstream Reactant Fix

Reactant v0.2.211 fixes the segfault (now a clean error). The underlying "failed to raise func" issue still requires an upstream fix to handle loop-dependent index expressions in KernelAbstractions kernels.

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

| File | Purpose | Status (v0.2.211+) |
|------|---------|---------------------|
| **`test_reactant.jl`** | **UPSTREAM MWE**: Pure KernelAbstractions, no Oceananigans | Use for filing Reactant issue |
| `test_delinearize_halo_size_mwe.jl` | Tests halo=1,2,3 with Oceananigans | halo=1 ✅, halo≥2 ❌ (clean error) |
| `test_delinearize_breeze_medwe.jl` | Full MedWE with Breeze AtmosphereModel | ❌ Clean error |
| `test_delinearize_oceananigans_medwe.jl` | Full MedWE with Oceananigans | ❌ Clean error |
| `test_delinearize_fill_halos_medwe.jl` | fill_halo_regions! only (with halo=1) | ✅ PASSES |
| `test_delinearize_array_ops_mwe.jl` | Pure Reactant array ops | ✅ All pass |

### Error Types by Configuration (Reactant v0.2.211+)

| Configuration | `--check-bounds=yes` | `--check-bounds=no` |
|---------------|----------------------|---------------------|
| halo=1, Periodic | ✅ Works | ✅ Works |
| halo=1, Bounded | ⚠️ Untested (B.6.4 may apply) | ⚠️ Untested |
| halo≥2, Periodic | "failed to raise func" | StableHLO shape error |
| halo≥2, Bounded | ❌ Likely fails (B.6.3/B.6.4) | ❌ Likely fails |

### Key Finding (2026-02-03) - ROOT CAUSE IDENTIFIED

**The issue is `halo >= 2` with KernelAbstractions kernels containing loop-dependent indices.**

**Why H > 1 causes compilation failure:**

```julia
@inbounds for i = 1:H
    parent(c)[i, j, k]     = parent(c)[N+i, j, k]     # indices: i, N+i
    parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k]     # indices: N+H+i, H+i
end
```

1. **Loop variable in indices**: `i` appears in 4 index expressions per iteration
2. **Complex MLIR**: Loop-dependent indices generate complex affine/StableHLO expressions
3. **Reactant can't compile**: Two different failure modes depending on `--check-bounds`

**Error paths with Reactant v0.2.211+:**

| `--check-bounds` | Error | Root Cause |
|------------------|-------|------------|
| `yes` | "failed to raise func" | MLIR affine expressions too complex |
| `no` | StableHLO shape error | `dynamic_update_slice` mismatch (workgroup vs array size) |

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

**Minimum Reactant version: v0.2.211** (converts segfault to clean error)

Record versions when reporting results:

```julia
@info "Versions" Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme) KernelAbstractions=pkgversion(KernelAbstractions)
```

### MLIR Debug Output (v0.2.211+)

When compilation fails, Reactant now dumps the MLIR module to a temp file:
```
┌ Error: Compilation failed, MLIR module written to /var/folders/.../module_000_..._post_all_pm.mlir
```

This file contains the complex affine expressions that failed to raise.
