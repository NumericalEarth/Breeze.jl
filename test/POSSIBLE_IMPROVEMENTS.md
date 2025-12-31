# Possible Improvements to BreezeReactantExt

> **Document Purpose**: Guide performance improvements and future development of Breeze's Reactant integration.
>
> **Key Reference**: Compare against GB-25 (`/Users/tobiwan/Documents/aeolus/GB-25`)

---

## Executive Summary

### Current State

Breeze has working Reactant integration with several workarounds in `ext/BreezeReactantExt/BreezeReactantExt.jl`. All 20 tests pass (see `test/reactant_enzyme.jl`).

**Performance (December 2025)**: Reactant is **~1.6-2.9× slower** than native execution due to workaround overhead. Compilation takes ~150-200s (one-time). The value is for AD/differentiation, not raw performance.

### Why Breeze Needs More Workarounds Than GB-25

| Aspect | GB-25 | Breeze |
|--------|-------|--------|
| **Model** | `HydrostaticFreeSurfaceModel` | `AtmosphereModel` |
| **Pressure Solver** | `SplitExplicitFreeSurface` (no FFT) | `FourierTridiagonalPoissonSolver` (FFT-based) |
| **Array Types** | Real only | Real + Complex (spectral) |
| **Reactant Workarounds** | ~3 simple wrappers | ~10 method overrides |

**Root cause**: Reactant's MLIR lowering fails for complex numbers inside KernelAbstractions kernels, but **works** for complex numbers in pure Julia broadcasts.

### Top 3 Recommended Improvements

| Priority | Improvement | Impact | Effort | Status |
|----------|-------------|--------|--------|--------|
| ✅ **1** | Eliminate 2× real tridiagonal split | ~2× faster pressure solve | Low | **DONE** |
| ✅ **2** | Profile and benchmark | Data-driven decisions | Low | **DONE** |
| ✅ **3** | In-place FFT optimization | 1.69× faster FFT operations | Low | **DONE** |
| ⚪ **4** | Reduction caching | Negligible overhead | — | **INVESTIGATED** |
| ⚪ **5** | Grid specialization | XLA optimizes this | — | **INVESTIGATED** |
| 🟢 **6** | Investigate sharded FFT scaling | Multi-GPU readiness | Medium | Pending |

---

## Table of Contents

1. [Background: The Core Problem](#1-background-the-core-problem)
2. [Improvement #1: Eliminate 2× Real Split ✅](#2-improvement-1-eliminate-2-real-split--completed)
3. [Improvement #2: Profile and Benchmark ✅](#3-improvement-2-profile-and-benchmark--completed)
4. [Improvement #3: In-Place FFT Operations ✅](#4-improvement-3-in-place-fft-operations--completed)
5. [Improvement #4: Reduction Caching ⚪](#5-improvement-4-reduction-caching--investigated---low-priority)
6. [Improvement #5: Grid Specialization ⚪](#6-improvement-5-grid-specialization--investigated---no-action-required)
7. [Improvement #6: Parallel Tridiagonal Algorithm](#7-improvement-6-parallel-tridiagonal-algorithm)
8. [Improvement #7: Upstream Reactant Fixes](#8-improvement-7-upstream-reactant-fixes)
9. [Sharding and Distributed Computing](#9-sharding-and-distributed-computing)
10. [GB-25 Patterns to Adopt](#10-gb-25-patterns-to-adopt)
11. [Conclusion](#conclusion)

---

## 1. Background: The Core Problem

### What Works vs What Doesn't

```julia
# ✅ THIS WORKS: Complex numbers in pure Julia broadcasts
sum(real.(complex_array))           # OK
complex_array .* real_array         # OK
ifelse.(condition, complex_a, complex_b)  # OK

# ❌ THIS FAILS: Complex numbers in KernelAbstractions kernels
@kernel function my_kernel!(out, inp)
    i, j, k = @index(Global, NTuple)
    out[i, j, k] = real(inp[i, j, k])  # MLIR error!
end
```

**Error message**:
```
'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got 'complex<f32>'
```

### Verification

Run `test/mwe_reactant_complex_number_mlir.jl` to confirm this behavior.

### Current Workaround Cost Summary

| Workaround | Location (approx lines) | Overhead | Frequency | Status |
|------------|------------------------|----------|-----------|--------|
| ~~2× real tridiagonal solve~~ | ~~415-431~~ | ~~2× compute~~ | ~~9×/iteration~~ | ✅ **ELIMINATED** |
| ~~4 temporary allocations~~ | ~~415-431~~ | ~~GC pressure~~ | ~~Per pressure solve~~ | ✅ **ELIMINATED** |
| ~~FFT copyto! pattern~~ | ~~407-410, 436-438~~ | ~~1.69× slower~~ | ~~Per FFT~~ | ✅ **OPTIMIZED** |
| CPU fallback (construction) | 800-920 | Data transfer | Once/simulation | Active |
| Broadcast source term | 619-679 | Minimal | Per pressure solve | Active |
| Broadcast pressure correction | 714-790 | Minimal | Per RK3 substep | Active |
| Field reduction materialization | 133-134 | 1 alloc/reduce | Per CFL check | Active |

---

## 2. Improvement #1: Eliminate 2× Real Split ✅ COMPLETED

> **Priority**: ✅ DONE  
> **Effort**: Low (code deletion)  
> **Impact**: ~2× faster tridiagonal solve, zero extra allocations
>
> **Implemented**: December 2025 — Thomas algorithm now solves directly on complex arrays

### Current Code (Expensive)

```julia
# ext/BreezeReactantExt/BreezeReactantExt.jl lines 415-431
rhs_re = real.(rhs)           # allocate #1
rhs_im = imag.(rhs)           # allocate #2
ϕ_re = similar(rhs_re)        # allocate #3
ϕ_im = similar(rhs_im)        # allocate #4

_thomas_solve_z_broadcast!(ϕ_re, a, b, c, rhs_re, t, Nz)  # solve #1
_thomas_solve_z_broadcast!(ϕ_im, a, b, c, rhs_im, t, Nz)  # solve #2

ϕ .= complex.(ϕ_re, ϕ_im)     # combine
```

### Why This Can Be Improved

The `_thomas_solve_z_broadcast!` function uses **only broadcasts**, not KA kernels. Therefore, it should work directly on complex arrays.

**Key insight**: The tridiagonal coefficients `a`, `b`, `c` are real (Laplacian eigenvalues). Only `rhs` and `ϕ` are complex. All operations in the Thomas algorithm are broadcast-based.

### Implemented Code

```julia
# ext/BreezeReactantExt/BreezeReactantExt.jl (current implementation)
# Solve tridiagonal system directly on complex arrays
a = solver.batched_tridiagonal_solver.a
b_arr = solver.batched_tridiagonal_solver.b
c = solver.batched_tridiagonal_solver.c
t = solver.batched_tridiagonal_solver.t

ϕ = solver.storage
_thomas_solve_z_broadcast!(ϕ, a, b_arr, c, rhs, t, Nz)  # ONE complex solve
```

### Verification Results ✅

All tests passed (December 2025):
```
Test Summary:        | Pass  Total      Time
Reactant + Enzyme AD |   20     20  15m25.7s
```

### Verification Checklist

- [x] `test/reactant_enzyme.jl` passes (20/20 tests)
- [x] Pressure solve produces correct results
- [x] No new allocations in pressure solve path (4 allocations eliminated)
- [x] No MLIR compilation errors

### Achieved Benefit

| Metric | Before | After |
|--------|--------|-------|
| Tridiagonal solves per pressure solve | 2 | **1** |
| Allocations per pressure solve | 4 | **0** |
| Memory bandwidth | 6× array copies | **0** |

---

## 3. Improvement #2: Profile and Benchmark ✅ COMPLETED

> **Priority**: ✅ DONE  
> **Effort**: Low  
> **Impact**: Data-driven optimization decisions enabled
>
> **Implemented**: December 2025 — Benchmark script created and results collected

### Benchmark Results (December 2025)

Using `test/benchmark_reactant.jl` and `test/benchmark_reactant_gpu.jl` with proper synchronization:

| Comparison | Grid Size | Native (ms/step) | Reactant (ms/step) | Result |
|------------|-----------|------------------|---------------------|--------|
| CPU | 64×64 | 2.5 | 7.3 | **2.9× slower** |
| GPU | 128×128 | 4.1 | 6.6 | **1.6× slower** |

### Key Findings

1. **Reactant is slower than native execution** on both CPU and GPU
2. **Workarounds add overhead**: Pure-Julia broadcasts vs optimized CUDA kernels
3. **Compilation cost**: ~150-200 seconds regardless of grid size (one-time)
4. **Value proposition**: Reactant enables AD/differentiation at the cost of forward performance

### Interpretation

| Observation | Implication |
|-------------|-------------|
| Reactant slower on CPU | Workarounds replace optimized code paths |
| Reactant slower on GPU | XLA cannot match cuFFT/cuBLAS optimizations |
| Consistent slowdown | ~1.6-2.9× overhead from broadcast workarounds |
| High compile time | Only worthwhile for AD, not raw performance |

### Recommendations Based on Benchmarks

1. **For forward simulation**: Use native CPU or GPU — faster execution
2. **For AD/differentiation**: Use Reactant — enables gradient computation
3. **For development**: Use CPU — no compilation overhead
4. **Future optimization**: Upstream Reactant fixes could eliminate workarounds

### Benchmark Script Location

The benchmark script is at `test/benchmark_reactant.jl`. Usage:

```bash
# Default 32×32 grid
julia --project=test test/benchmark_reactant.jl

# Custom grid size
BENCH_NX=128 BENCH_NZ=128 julia --project=test test/benchmark_reactant.jl
```

### Future Work

- [ ] Profile with `Reactant.with_profiler()` to identify bottlenecks
- [ ] Test GPU backend (requires NVIDIA GPU)
- [ ] Investigate compilation caching between Julia sessions
- [ ] Test 3D grids (512×512×64 typical for LES)

---

## 4. Improvement #3: In-Place FFT Operations ✅ COMPLETED

> **Priority**: ✅ DONE  
> **Effort**: Low  
> **Impact**: 1.69× faster FFT operations
>
> **Implemented**: December 2025 — Changed `copyto!` to broadcast assignment

### Previous Code (Slower)

```julia
rhs_fft = AbstractFFTs_mod.fft(rhs, periodic_dims)  # allocates temporary
copyto!(rhs, rhs_fft)                                # copies back
```

### New Code (Faster)

```julia
rhs .= AbstractFFTs_mod.fft(rhs, periodic_dims)  # XLA fuses this!
```

### Investigation Results

Created `test/mwe_reactant_inplace_fft.jl` to investigate FFT patterns:

| Pattern | Description | Time (128×128) |
|---------|-------------|----------------|
| A | `copyto!(A, fft(A))` | 0.092 ms |
| B | `A .= fft(A)` | 0.054 ms |
| C | `return fft(A)` | 0.023 ms |

**Key findings**:
1. `fft!()` does NOT exist for Reactant (pointer conversion fails)
2. Single integer dims fail (`fft(A, 1)`), but tuples work (`fft(A, (1,))`)
3. XLA fuses broadcast assignment, eliminating intermediate allocations
4. Broadcast pattern is **1.69× faster** than `copyto!`

### XLA HLO Analysis

The optimized HLO shows a single fused `stablehlo.fft` operation:
```
func.func @main(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> {
  %0 = stablehlo.fft %arg0, type = FFT, length = [4] : tensor<4x4xcomplex<f64>>
  return %0 : tensor<4x4xcomplex<f64>>
}
```

No intermediate allocation or copy operations — XLA fully optimizes this.

### Code Changes

Updated `ext/BreezeReactantExt/BreezeReactantExt.jl`:
- Line 407-410: Changed forward FFT from `copyto!` to broadcast
- Line 436-438: Changed inverse FFT from `copyto!` to broadcast

### Verification

All 20 tests pass after the change.

---

## 5. Improvement #4: Reduction Caching ⚪ INVESTIGATED - LOW PRIORITY

> **Priority**: ⚪ VERY LOW  
> **Effort**: Medium  
> **Impact**: Negligible after JIT warmup
>
> **Investigated**: December 2025 — Overhead is minimal, not worth the complexity

### Investigation Results

Created `test/mwe_reactant_reduction_caching.jl` to measure Field(op) overhead:

| Call # | Field(op) Time |
|--------|----------------|
| 1st (cold) | 411 ms |
| 2nd+ (warm) | 0.04-0.05 ms |

**Key Findings**:
1. First call is slow due to JIT compilation (~411 ms)
2. Subsequent calls are fast (~0.04 ms) — negligible overhead
3. `minimum(op)` is actually faster than `minimum(precomputed_field)`
4. In Reactant @compile context, XLA optimizes allocations

### Why No Action Required

```
After JIT warmup:
  Field(op) creation: ~0.04 ms
  Compared to:
    FFT operations:   ~0.05 ms (from Improvement #3)
    Time step total:  ~10 ms (from benchmarks)
    
Field(op) overhead: ~0.4% of time step — negligible
```

### Original Proposed Solution (Not Implemented)

```julia
# Would add complexity for negligible benefit
const _REDUCTION_CACHE = Dict{UInt64, Any}()

function _get_scratch_field(op::AbstractOperation)
    # ...caching logic...
end
```

### Conclusion

- Caching adds complexity with uncertain benefit
- JIT warmup handles the overhead naturally
- XLA optimization eliminates allocations in traced code
- **No code changes needed**

---

## 6. Improvement #5: Grid Specialization ⚪ INVESTIGATED - NO ACTION REQUIRED

> **Priority**: ⚪ VERY LOW  
> **Effort**: Low  
> **Impact**: Negligible — XLA optimizes this away
>
> **Investigated**: December 2025 — Runtime checks are effectively free

### Investigation Results

Created `test/mwe_reactant_grid_specialization.jl` to compare patterns:

| Pattern | Time (100k calls) | Per-call |
|---------|-------------------|----------|
| Runtime check | 61.36 ms | 0.61 µs |
| Dispatch | 46.59 ms | 0.47 µs |

**Dispatch is 1.32× faster** in microbenchmarks, BUT:

### Why No Action Required

1. **Absolute overhead is negligible**: 0.14 µs difference per call
2. **XLA eliminates the check**: During @compile, topology is constant
3. **Code clarity**: Runtime checks are self-documenting
4. **Complexity trade-off**: Dispatch requires 2× more method definitions

```
Comparison to actual work:
  Runtime check overhead: 0.61 µs
  FFT operation:          50 µs
  Time step total:        10,000 µs
  
  Overhead: 0.006% of time step — completely negligible
```

### XLA Optimization Insight

When Reactant traces the function:
```julia
topo = topology(grid)     # Returns CONCRETE tuple at trace time
if topo[2] !== Flat       # Becomes CONSTANT check
    error(...)            # Dead code — eliminated by XLA
end
```

The runtime check has **zero overhead** in compiled XLA code.

### Conclusion

- Current runtime checks are clear and effective
- XLA dead-code elimination handles optimization
- Dispatch adds complexity with no meaningful benefit
- **No code changes needed**

---

## 7. Improvement #6: Parallel Tridiagonal Algorithm

> **Priority**: ⚪ VERY LOW (only if profiling shows bottleneck)  
> **Effort**: High  
> **Impact**: Better GPU utilization for shallow grids

### When This Matters

The Thomas algorithm is O(Nz) sequential. For grids with small Nz (e.g., 50-100 levels), this may not fully utilize GPU parallelism.

### Alternative: Cyclic Reduction

```julia
# O(log Nz) parallel steps instead of O(Nz) sequential
function cyclic_reduction_solve!(ϕ, a, b, c, rhs, Nz)
    # Forward reduction: log2(Nz) parallel phases
    for level in 1:ceil(Int, log2(Nz))
        stride = 2^level
        # All k values processed in parallel
        # Each eliminates one variable
    end
    
    # Back substitution: log2(Nz) parallel phases
    for level in ceil(Int, log2(Nz)):-1:1
        # Parallel solve
    end
end
```

### Trade-offs

| Algorithm | Sequential Steps | Best For |
|-----------|------------------|----------|
| Thomas | O(Nz) | CPU, deep grids |
| Cyclic Reduction | O(log Nz) | GPU, shallow grids |

### Recommendation

**Wait for profiling data** before implementing. The Thomas algorithm may be fast enough.

---

## 8. Improvement #7: Upstream Reactant Fixes

> **Priority**: ⚪ MONITORING  
> **Effort**: None (wait for upstream)  
> **Impact**: Would eliminate most workarounds

### When Reactant Fixes Complex Numbers in KA Kernels

We can remove:
1. FFT planning no-ops
2. Thomas algorithm workaround
3. Source term broadcast workaround
4. Pressure correction broadcast workaround

### How to Check

```bash
# Run the MWE periodically
julia --project=test test/mwe_reactant_complex_number_mlir.jl
```

If the "Trigger" section starts passing, upstream is fixed.

### Tracking

- Watch [Reactant.jl issues](https://github.com/EnzymeAD/Reactant.jl/issues)
- Search for "complex" or "llvm.extractvalue"

---

## 9. Sharding and Distributed Computing

> **This section covers multi-GPU/TPU considerations for future Breeze scalability.**

### How GB-25 Does Distributed Computing

GB-25 uses **Reactant's native sharding** (not MPI):

```julia
# GB-25/sharding/simple_sharding_problem.jl
GordonBell25.initialize()  # XLA distributed runtime
ndevices = length(Reactant.devices())

arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)
```

### Oceananigans Sharding Integration

From `OceananigansReactantExt/Grids/sharded_grids.jl`:

```julia
# Data sharded along x and y
xysharding = Sharding.DimsSharding(arch.connectivity, (1, 2), (:x, :y))

# Z replicated across all devices
replicate = Sharding.Replicated(arch.connectivity)
z = sharded_z_direction(grid.z; sharding=replicate)
```

### The FFT Sharding Challenge

Breeze's `FourierTridiagonalPoissonSolver` requires:

```
Physical space → FFT(x,y) → Spectral space → Tridiagonal solve(z) → IFFT(x,y) → Physical space
```

**Problem**: If data is sharded along x or y, FFT requires **all-to-all communication**.

**Why GB-25 avoids this**: Uses `SplitExplicitFreeSurface` — no FFTs, only local stencils.

### Sharding Strategies for Breeze

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **A: Shard x only** | FFT in y is local | Limited parallelism |
| **B: Pencil decomposition** | Transpose before FFT | Communication overhead |
| **C: Alternative solver** | Multigrid/CG | More iterations, no FFT |

### Investigation Checklist

1. **Test basic sharding**
   ```julia
   arch = Distributed(ReactantState(); partition=Partition(2, 1, 1))
   grid = RectilinearGrid(arch; size=(64, 1, 32), ...)
   model = AtmosphereModel(; grid)
   # Does it compile?
   ```

2. **Profile FFT communication**
   ```julia
   Reactant.with_profiler("sharded_profile") do
       compiled_step!(model, Δt)
   end
   # Look for AllToAll or CollectivePermute operations
   ```

3. **Compare scaling**
   ```julia
   # 1 device vs 4 devices — is there speedup?
   ```

### Compatibility Status

| Feature | Status | Notes |
|---------|--------|-------|
| Single-device ReactantState | ✅ Working | All tests pass |
| Distributed(ReactantState) | ⚠️ Untested | Needs investigation |
| Sharded FFT | ❓ Unknown | May require all-to-all |
| Sharded pressure correction | ✅ Should work | Broadcast-based |

---

## 10. GB-25 Patterns to Adopt

### Pattern 1: Wrapper Functions for Tracing

GB-25 wraps functions to ensure proper tracing:

```julia
# GB-25/src/timestepping_utils.jl
function time_step!(model)
    Δt = model.clock.last_Δt + 0  # `+ 0` ensures Δt is traced
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end
```

**Why `+ 0`**: Forces `Δt` to be a new traced value, avoiding issues with accessing struct fields.

### Pattern 2: Separate `first_time_step!` and `loop!`

```julia
# First step may need different initialization
compiled_first! = @compile first_time_step!(model)

# Main loop uses traced iteration
Ninner = ConcreteRNumber(100)
compiled_loop! = @compile loop!(model, Ninner)

# Execution
compiled_first!(model)
compiled_loop!(model, Ninner)
```

### Pattern 3: `@trace` with `track_numbers=false`

```julia
function loop!(model, Ninner)
    Δt = model.clock.last_Δt + 0
    @trace track_numbers=false for _ = 1:Ninner
        time_step!(model, Δt)
    end
end
```

**Why `track_numbers=false`**: Prevents Reactant from trying to trace integer loop indices, which can cause issues.

### Pattern 4: TreeSharding for Nested Structures

```julia
# GB-25/src/sharding_utils.jl
struct TreeSharding{S} <: Sharding.AbstractSharding
    sharding::S
end

# Propagates sharding through nested field access
Base.getproperty(t::TreeSharding, x) = t
```

Useful for complex models like `AtmosphereModel` with deeply nested fields.

---

## Quick Reference: File Locations

### Breeze Extension Code

| Component | File | Lines (approx) |
|-----------|------|----------------|
| Field reductions | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 69-171 |
| CartesianIndex fix | same | 183-187 |
| Thomas algorithm | same | 271-339 |
| Poisson solver | same | 341-451 |
| FFT planning | same | 454-489 |
| RK3 time_step! | same | 521-588 |
| Source term | same | 619-679 |
| Pressure correction | same | 714-790 |
| CPU fallback | same | 801-919 |

### Test Files

| Purpose | File |
|---------|------|
| Main test suite | `test/reactant_enzyme.jl` |
| **CPU benchmark** | `test/benchmark_reactant.jl` |
| **GPU benchmark** | `test/benchmark_reactant_gpu.jl` |
| **In-place FFT MWE** | `test/mwe_reactant_inplace_fft.jl` |
| **Reduction caching MWE** | `test/mwe_reactant_reduction_caching.jl` |
| **Grid specialization MWE** | `test/mwe_reactant_grid_specialization.jl` |
| FFT planning MWE | `test/mwe_reactant_fft_planning.jl` |
| Complex number MWE | `test/mwe_reactant_complex_number_mlir.jl` |
| Solver compile MWE | `test/mwe_reactant_fourier_tridiagonal_solver_compile.jl` |
| Issue tracker | `test/REACTANT_ISSUES.md` |
| Improvement tracker | `test/POSSIBLE_IMPROVEMENTS.md` (this file) |

### GB-25 Reference

| Component | File |
|---------|------|
| Time stepping | `/Users/tobiwan/Documents/aeolus/GB-25/src/timestepping_utils.jl` |
| Sharding utils | `/Users/tobiwan/Documents/aeolus/GB-25/src/sharding_utils.jl` |
| Model setup | `/Users/tobiwan/Documents/aeolus/GB-25/src/baroclinic_instability_model.jl` |
| Sharded run | `/Users/tobiwan/Documents/aeolus/GB-25/sharding/sharded_baroclinic_instability_simulation_run.jl` |

---

## Conclusion

### Completed ✅

1. **Improvement #1** (eliminate 2× real split) — **DONE** (December 2025)
   - Thomas algorithm now solves directly on complex arrays
   - 2× faster tridiagonal solve, 0 allocations (was 4)

2. **Improvement #2** (profile and benchmark) — **DONE** (December 2025)
   - Benchmark scripts: `test/benchmark_reactant.jl` (CPU) and `test/benchmark_reactant_gpu.jl` (GPU)
   - Key finding: Reactant is **~1.6-2.9× slower** than native due to workaround overhead
   - Value proposition: Enables AD/differentiation, not raw performance
   - Compilation time: ~150-200s (one-time cost)

3. **Improvement #3** (in-place FFT) — **DONE** (December 2025)
   - Changed `copyto!(A, fft(A))` to `A .= fft(A)` for better XLA fusion
   - 1.69× faster FFT operations
   - MWE created at `test/mwe_reactant_inplace_fft.jl`

### Investigated - No Action Required ⚪

4. **Improvement #4** (reduction caching) — **INVESTIGATED** (December 2025)
   - Field(op) overhead is ~0.04 ms after JIT warmup — negligible
   - XLA optimizes allocations in traced code
   - MWE created at `test/mwe_reactant_reduction_caching.jl`

5. **Improvement #5** (grid specialization) — **INVESTIGATED** (December 2025)
   - Runtime check overhead is ~0.6 µs — 0.006% of time step
   - XLA dead-code eliminates runtime checks anyway
   - MWE created at `test/mwe_reactant_grid_specialization.jl`

### Short-Term (This Month)

- **Test basic sharding** — verify Breeze works with `Distributed(ReactantState())`
- ✅ **GPU backend tested** — Reactant GPU is ~1.6× slower than native CUDA (workaround overhead)

### Medium-Term (This Quarter)

- **Investigate sharded FFT performance** — is it a bottleneck?
- **Consider alternative solvers** if FFT sharding is problematic

### Long-Term (Monitor)

- **Watch upstream Reactant** — complex number fix would simplify everything
- **Evaluate multigrid** — if FFT scaling is poor at large scale
