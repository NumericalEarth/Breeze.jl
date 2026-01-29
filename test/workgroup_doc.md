# Reactant `raise=true` and Stencil Kernels: A Complete Guide

## Summary

When compiling Oceananigans kernels with Reactant using `raise=true`, stencil-style kernels that use offset array indexing (e.g., `input[i-1, j]`) fail when combined with **negative offsets** from `KernelParameters`. This document explains:

1. What fails and why
2. The role of `KernelParameters` and `OffsetStaticSize`
3. Why the workaround using positive offsets is correct
4. How this applies to Oceananigans use cases (halos, periodic boundaries)

**Key Finding**: The Reactant `raise` pass cannot simplify complex affine expressions involving `floordiv` when offsets are negative. Using positive offsets avoids this issue while producing correct results.

---

## 1. What Fails and Why

### The Failing Pattern

The following code fails with `raise=true`:

```julia
@kernel function _stencil_kernel!(output, input)
    i, j = @index(Global, NTuple)
    @inbounds output[i, j] = 0.25 * (input[i, j] + input[i-1, j] + input[i, j-1] + input[i-1, j-1])
end

# For Periodic topology with size=11 and halo=3:
ii = -2:14  # 17 points
kp = KernelParameters(ii, jj)  # Creates offset of -3

compiled = Reactant.@compile raise=true compute_stencil!(g, f, grid, kp)  # FAILS
```

### Why It Fails

When Reactant compiles this with `raise=true`, it generates MLIR code with complex affine index expressions:

```mlir
%1 = affine.load %arg1[
    %arg2 + (%arg2 * -17 + %arg3 * 17 + %arg4 * 16 + %arg5 - 1) floordiv 289,
    -%arg2 + %arg3 + %arg4 + (-%arg4 + %arg5 - 1) floordiv 17 - ...,
    ...
]
```

These expressions arise from the combination of:

1. **2D kernel indices** from `@index(Global, NTuple)`
2. **Negative offsets** from `KernelParameters` (e.g., `-3` for range `-2:14`)
3. **Stencil offsets** in the kernel body (`i-1`, `j-1`)
4. **3D OffsetArrays** (Oceananigans Fields have a singleton z-dimension)

The `raise` pass attempts to convert GPU memory accesses into XLA-compatible tensor operations. When the index arithmetic involves `floordiv` with negative offsets, the pass fails to simplify the expressions and aborts.

### What Works

- **Copy kernels** (no stencil offsets) work with any offset sign
- **Stencil kernels with positive offsets** work correctly
- Using `raise=false` bypasses the issue (generates CUDA PTX directly)

---

## 2. Understanding KernelParameters and OffsetStaticSize

### How KernelParameters Works

`KernelParameters` in Oceananigans converts iteration ranges into a worksize and offset:

```julia
KernelParameters(-2:14, -2:14)
# Creates: KernelParameters{(17, 17), (-3, -3)}()
#          worksize = (17, 17)  -- the length of each range
#          offsets = (-3, -3)   -- first(range) - 1
```

### How Offsets Are Applied

The offset is stored in `OffsetStaticSize` and applied in the `expand` function:

```julia
# From kernel_launching.jl
function expand(ndrange::OffsetNDRange{N, S}, groupidx, idx) where {N, S}
    nI = ntuple(Val(N)) do I
        (gidx - 1) * stride + idx.I[I] + S[I]  # S[I] is the offset
    end
    return CartesianIndex(nI)
end
```

When `S[I] = -3`, the kernel index `1` becomes logical index `-2` (since `1 + (-3) = -2`).

### The Problem with Negative Offsets

The expression `idx + offset` where `offset < 0` creates arithmetic that the MLIR raise pass cannot simplify when combined with:
- Workgroup/thread index calculations
- Stencil offset subtractions (`i-1`)
- Multi-dimensional array linearization

---

## 3. Why the Workaround Gives Correct Results

### The Workaround Pattern

Instead of using negative-offset `KernelParameters`, we:

1. Extract parent arrays from Fields (removing OffsetArray wrappers)
2. Use positive worksize and kernel offsets
3. Compute array indices explicitly in the kernel

```julia
# Parent array is 1-based with shape (17, 17)
worksize = (16, 16)       # Iterate 16 points in each dimension
kernel_offsets = (1, 1)   # Start at array index 2

@kernel function _stencil_workaround!(output, input)
    i_base, j_base = @index(Global, NTuple)
    i = i_base + 1  # Maps kernel index 1 → array index 2
    j = j_base + 1
    @inbounds output[i, j] = 0.25 * (input[i, j] + input[i-1, j] + input[i, j-1] + input[i-1, j-1])
end
```

### Why This Is Equivalent (or Better)

Consider a Field with logical indices `-2:14` (axes of the OffsetArray):

| Logical Index | Parent Array Index |
|---------------|-------------------|
| -2 | 1 |
| -1 | 2 |
| 0 | 3 |
| ... | ... |
| 14 | 17 |

The original `KernelParameters(-2:14, ...)` iterates logical indices `-2` to `14`. When the stencil accesses `input[i-1, j]`:
- At `i = -2`: accesses `input[-3, j]` → **OUT OF BOUNDS** (axes are `-2:14`)

The workaround iterates array indices `2` to `17`. When the stencil accesses `input[i-1, j]`:
- At `i = 2`: accesses `input[1, j]` → **VALID** (array is `1:17`)

**The workaround avoids the invalid first iteration and covers the maximum valid range for a stencil kernel.**

### Numerical Equivalence

For any index `i` in the valid range:
- Original: logical index `i`, OffsetArray access
- Workaround: array index `i - offset`, direct array access

Both access the same memory location. The parent array and OffsetArray share the same underlying data; only the indexing convention differs.

---

## 4. Use Cases: Halos and Periodic Boundaries

### Why Oceananigans Uses Negative Offsets

Oceananigans grids support **halo regions** for boundary conditions. For a 1D domain with `N = 11` interior points and `H = 3` halo points:

```
Halo (left)    Interior       Halo (right)
[-2, -1, 0]    [1, ..., 11]   [12, 13, 14]
```

Logical indices span `-H+1` to `N+H`, i.e., `-2:14`.

### Periodic Boundary Conditions

For periodic topologies, halo values wrap around:
- `field[-2]` contains the same value as `field[N-1]` (index 10)
- `field[14]` contains the same value as `field[3]`

Before computing stencils, `fill_halo_regions!` copies interior values to halos, making the periodic wrapping transparent to kernel code.

### What Operations Need Full Halo Coverage?

| Operation | Needs Halos? | Stencil? | Works with raise=true? |
|-----------|--------------|----------|------------------------|
| Fill halo regions | Yes | No (copy) | Yes |
| Interior stencil | No | Yes | Yes (with positive offsets) |
| Halo-aware diagnostic | Yes | Maybe | Depends on stencil pattern |

For stencil operations, you typically only compute in the interior (indices `1:N`) or the interior plus one halo layer (`0:N+1`), never the full halo range, because the outermost halo points lack neighbors for stencil access.

### Practical Guidance

1. **For interior stencils** (most common):
   - Use positive offsets starting at index 2
   - Cover indices `2:N+2H-1` (skipping first row/column for `i-1` access)

2. **For halo-filling kernels** (no stencil offsets):
   - Standard `KernelParameters` with negative offsets works
   - These are copy operations, not stencils, so `raise=true` succeeds

3. **For full-domain diagnostics**:
   - If no stencil access, use standard approach
   - If stencil access needed, use positive offsets and valid range

---

## 5. Implementation Summary

### Files in This Test Suite

| File | Purpose |
|------|---------|
| `workgroup.jl` | Original failing test (documents the issue) |
| `workgroup_mwe.jl` | Minimum working examples with positive offsets |
| `workgroup_full_range.jl` | Full-range stencil test with correctness verification |
| `workgroup_doc.md` | This documentation |

### The Workaround Recipe

```julia
# 1. Extract parent array (before @compile)
input_3d = parent(field.data)
input_2d = reshape(Array(input_3d), sz[1], sz[2])  # For Flat z-topology
input = Reactant.to_rarray(input_2d)

# 2. Use positive worksize and offsets
Nx, Ny = size(input_2d)
worksize = (Nx - 1, Ny - 1)  # Maximum range for stencil with i-1 access
kernel_offsets = (1, 1)      # Start at index 2

# 3. Write kernel with explicit offset handling
@kernel function stencil!(output, input)
    i_base, j_base = @index(Global, NTuple)
    i = i_base + 1  # Apply positive offset
    j = j_base + 1
    @inbounds output[i, j] = input[i, j] + input[i-1, j] + ...
end

# 4. Compile with raise=true
compiled = Reactant.@compile raise=true sync=true compute!(output, input, arch, worksize)
```

---

## 6. Upstream Fix Locations

The issue spans multiple packages in the Julia GPU ecosystem. Here we document where fixes could be implemented, from most fundamental to most application-specific.

### Option A: KernelAbstractions.jl (Recommended)

**Current State**: KernelAbstractions does not have native support for offset indices. Oceananigans implements a custom `OffsetStaticSize` type as a workaround, with an explicit TODO:

```julia
# From Oceananigans/src/Utils/kernel_launching.jl, line 388:
# TODO: when offsets are implemented in KA so that we can call
# `kernel(dev, group, size, offsets)`, remove all of this
```

**Proposed Fix**: Add native offset support to KernelAbstractions with a backend-aware implementation.

```julia
# Hypothetical KA API
@kernel function my_kernel!(output, input)
    i, j = @index(Global, NTuple)  # Returns offset-adjusted indices
    ...
end

# Launch with explicit offsets (new API)
kernel!(backend, workgroup, worksize; offsets=(-3, -3))
```

**Benefits**:
- Single fix benefits all downstream packages (Oceananigans, Breeze, etc.)
- Backend-specific implementations can optimize for each target
- For Reactant backend: generate simple `idx + offset` without `floordiv`
- For CUDA backend: use existing approach or optimize differently
- Oceananigans can remove ~200 lines of custom `OffsetStaticSize` code

**Challenges**:
- Requires coordination with KA maintainers
- Must maintain backwards compatibility
- Each backend extension needs updates

**Priority**: HIGH - This is the most fundamental fix location.

---

### Option B: Reactant.jl / ReactantKernelAbstractionsExt

**Current State**: The `raise` pass in Reactant fails when MLIR affine expressions contain `floordiv` operations with negative offsets. The pass attempts to convert GPU memory accesses to XLA-compatible tensor operations but cannot simplify these expressions.

**Proposed Fix**: Enhance the `raise` pass to recognize and simplify affine patterns from KernelAbstractions.

```mlir
# Current failing pattern:
%1 = affine.load %arg1[
    %arg2 + (%arg2 * -17 + %arg3 * 17 + %arg4 * 16 + %arg5 - 1) floordiv 289,
    ...
]

# Should simplify to:
%1 = affine.load %arg1[%thread_i + %offset_i, %thread_j + %offset_j]
```

**Benefits**:
- No changes needed in user code or Oceananigans
- Existing `KernelParameters` with negative offsets would "just work"
- Improves Reactant's robustness for other complex index patterns

**Challenges**:
- MLIR affine simplification is complex
- May require pattern matching specific to KA-generated code
- Could be fragile if KA changes its index computation

**Alternative**: Instead of fixing the `raise` pass, ReactantKernelAbstractionsExt could intercept offset kernels and rewrite them to use positive offsets internally.

**Priority**: MEDIUM - Good fallback if KA fix is delayed.

---

### Option C: Oceananigans.jl

**Current State**: Oceananigans defines `OffsetStaticSize` and `KernelParameters` in `src/Utils/kernel_launching.jl`. The offset is stored as a type parameter and applied in the `expand` function.

**Proposed Fix**: Add Reactant-specific dispatch that normalizes offsets.

```julia
# New method for ReactantState architecture
@inline function configure_kernel(arch::ReactantState, grid, workspec::KernelParameters{spec, offsets}, 
                                  kernel!, args...; kwargs...) where {spec, offsets}
    # Normalize to positive offsets for Reactant
    normalized_offsets = normalize_offsets_for_reactant(offsets)
    wrapped_kernel = OffsetNormalizingKernel(kernel!, offsets, normalized_offsets)
    # ... launch with normalized offsets
end
```

**Benefits**:
- Targeted fix for Reactant without affecting other backends
- Can be implemented independently of KA changes
- Oceananigans controls its own kernel launching

**Challenges**:
- Adds Reactant-specific code paths to Oceananigans
- Requires wrapping kernels to adjust indices
- May need to modify kernel signatures or use closures

**Priority**: MEDIUM - Good interim solution.

---

### Option D: Breeze.jl Extension (Current Workaround)

**Current State**: The workaround demonstrated in this test suite lives in `ext/BreezeReactantExt/`.

**Implementation**: Helper functions that decompose `KernelParameters` and require explicit offset handling in kernels.

```julia
# From BreezeReactantExt
function decompose_kernel_parameters(::KernelParameters{spec, offsets}) where {spec, offsets}
    return spec, offsets
end

function launch_with_offsets!(arch, grid, kp::KernelParameters, kernel!, args...; kwargs...)
    worksize, offsets = decompose_kernel_parameters(kp)
    all_args = (args..., offsets...)
    launch!(arch, grid, worksize, kernel!, all_args...; kwargs...)
end
```

**Benefits**:
- Immediate solution, no upstream changes needed
- Fully backwards compatible
- Easy to test and iterate

**Challenges**:
- Requires rewriting kernels to accept offset arguments
- Must extract parent arrays before compilation
- Boilerplate in user code

**Priority**: LOW (for long-term) - Use as interim solution while upstream fixes are developed.

---

### Comparison Matrix

| Fix Location | Effort | Impact | Backwards Compat | Timeline |
|--------------|--------|--------|------------------|----------|
| **KernelAbstractions.jl** | High | All KA users | Yes | Long |
| **Reactant.jl** | Medium | Reactant+KA users | Yes | Medium |
| **Oceananigans.jl** | Medium | Oceananigans users | Yes | Medium |
| **Breeze.jl Extension** | Low | Breeze users only | Yes | Immediate |

---

### Recommended Path Forward

1. **Immediate**: Use Breeze extension workaround (documented here)

2. **Short-term**: File issues in:
   - KernelAbstractions.jl: Request native offset support
   - Reactant.jl: Report `raise` pass failure with reproduction case

3. **Medium-term**: Implement Oceananigans.jl fix as interim solution

4. **Long-term**: Migrate to KA native offset support once available

---

## 7. Current Status

The workaround demonstrated here is:
- **Backwards compatible**: Existing code continues to work with `raise=false`
- **Correct**: Produces identical numerical results for valid index ranges
- **Complete**: Covers all stencil patterns that access `i-1`, `j-1`, etc.

For production use, consider wrapping this pattern in helper functions within `BreezeReactantExt` to reduce boilerplate.
