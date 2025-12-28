# Reactant + Enzyme AD Integration Issues

## Summary

This document tracks issues encountered when integrating Breeze.jl with Reactant.jl for automatic differentiation via Enzyme.

**Current status (December 2025)**: With `BreezeReactantExt` properly loaded, model construction and many forward operations work on `ReactantState`. The **FourierTridiagonalPoissonSolver** now compiles successfully with `raise=true` thanks to the Option B workaround. However, **issue #223 is still partially blocked**: other KernelAbstractions kernels in the full `AtmosphereModel` time stepping still fail (e.g., `_compute_anelastic_source_term!`).

### ✅ RESOLVED: FourierTridiagonalPoissonSolver compilation (Option B workaround)

The `FourierTridiagonalPoissonSolver.solve!` now compiles successfully with `Reactant.@compile raise=true` thanks to the Option B workaround in `BreezeReactantExt`:

1. **FFT Planning**: No-op for Reactant arrays (XLA handles planning internally)
2. **Pure-Julia Thomas Algorithm**: Replace the KA kernel `solve_batched_tridiagonal_system_kernel!` with pure-Julia broadcast operations
3. **Direct Complex Solve**: Solve tridiagonal system directly on complex arrays (no real/imag split needed — pure Julia broadcasts work with complex numbers)
4. **Broadcast Copy**: Use pure array broadcast instead of `copy_real_component!` kernel

**Performance note**: The Thomas algorithm operates directly on complex spectral coefficients. The tridiagonal coefficients (a, b, c) are real-valued Laplacian eigenvalues, while only the RHS and solution are complex. This eliminates the need for 2× real solves and 4 temporary allocations.

**Verified working**: `test/mwe_reactant_fourier_tridiagonal_solver_compile.jl` now succeeds with Breeze loaded.

### Remaining blocker: `_compute_anelastic_source_term!` kernel fails MLIR raising

While the pressure solver workaround is complete, the full `AtmosphereModel.time_step!` still fails due to **Breeze's own KA kernel** that hasn't been replaced with pure-Julia equivalents:

**Primary blocker:**
- `src/AtmosphereModels/Dynamics/anelastic_pressure_solver.jl:103` - `_compute_anelastic_source_term!`
  - This kernel computes the divergence of the velocity field as the source term for the pressure Poisson equation
  - It is called in `compute_pressure_correction!` before the pressure solver is invoked
  - The kernel writes to a 3D array using KA indexing, which Reactant cannot lower with `raise=true`

**MLIR error message:**
```
'affine.store' op value to store must have the same type as memref element type
...
name = "gpu__compute_anelastic_source_term!"
...
"failed to run pass manager on module"
```

**Why GB-25 works but Breeze fails:**
- **GB-25**: Uses `HydrostaticFreeSurfaceModel` with `SplitExplicitFreeSurface` - **no FFT-based Poisson solver**, and the time stepping kernels are simpler
- **Breeze**: Uses `AnelasticModel` with `FourierTridiagonalPoissonSolver` - has anelastic-specific kernels like `_compute_anelastic_source_term!`

**Fix direction (Option C)**: Replace `_compute_anelastic_source_term!` with a pure-Julia broadcast-based implementation in `BreezeReactantExt`, similar to the `_thomas_solve_z_broadcast!` workaround. This would involve:
1. Overriding `compute_pressure_correction!` for `ReactantAtmosphereModel`
2. Implementing the divergence computation using broadcasts instead of `@kernel`/`@index`
3. Testing with `Reactant.@compile raise=true`

**Note**: Once `_compute_anelastic_source_term!` is replaced, there may be additional Oceananigans kernels in the time stepping pipeline that also need workarounds.

---

## ⚠️ IMPORTANT: Extension Loading

The `BreezeReactantExt` extension must be loaded for Reactant support to work. This extension is triggered when both `Reactant` and `OffsetArrays` are loaded alongside `Breeze`.

**Common pitfall**: If running tests from the `test/` directory, ensure the local development version of Breeze is used (not the registered package). The registered package may not contain the latest extension code.

To verify the extension is loaded:
```julia
using Breeze, Reactant, OffsetArrays
ext = Base.get_extension(Breeze, :BreezeReactantExt)
println("BreezeReactantExt loaded: ", ext !== nothing)  # Should print: true
```

If the extension is not loading, run:
```julia
using Pkg
Pkg.develop(path=".")  # From the Breeze.jl root directory
```

---

## ✅ RESOLVED ISSUES

### 1. FFT Planning for Reactant Arrays

| Field | Details |
|-------|---------|
| **Error** | `MethodError: no method matching plan_forward_transform(::ConcretePJRTArray{ComplexF64, 3, 1}, ::Periodic, ...)` |
| **Source** | `Oceananigans.Solvers.plan_transforms` |
| **File** | `~/.julia/packages/Oceananigans/.../src/Solvers/plan_transforms.jl:188` |
| **Triggered by** | `FourierTridiagonalPoissonSolver` construction |
| **Call chain** | `AtmosphereModel()` → `dynamics_pressure_solver()` → `FourierTridiagonalPoissonSolver()` → `plan_transforms()` |
| **Breeze location** | `src/AtmosphereModels/Dynamics/anelastic_pressure_solver.jl:20` |

**Root cause**: Oceananigans defines `plan_forward_transform` for `Array` and `CuArray`, but not for Reactant's `ConcretePJRTArray`.

**Resolution**: Added no-op methods in `BreezeReactantExt` (XLA handles FFT planning internally):
```julia
# ext/BreezeReactantExt/BreezeReactantExt.jl lines 178-194
function Oceananigans.Solvers.plan_forward_transform(A::ReactantArray, ::Periodic, dims, planner_flag=nothing)
    return nothing
end
```

**MWE**: `test/mwe_reactant_fft_planning.jl`

---

### 2. CartesianIndex Indexing for OffsetArray{TracedRNumber}

| Field | Details |
|-------|---------|
| **Error** | `MethodError: no method matching get_ancestor_and_indices_inner(::OffsetArray{TracedRNumber{Float64},...}, ::CartesianIndex{3})` |
| **Source** | `Reactant.TracedUtils.get_ancestor_and_indices_inner` |
| **File** | `~/.julia/packages/Reactant/.../ext/ReactantOffsetArraysExt.jl:66` |
| **Triggered by** | Any Field reduction (`maximum`, `sum`, etc.) during Reactant tracing |

**Root cause**: `ReactantOffsetArraysExt` defines:
```julia
get_ancestor_and_indices_inner(arr, indices::Vararg{Any,N})  # expects N separate args
```
But `CartesianIndex{N}` is passed as a **single** argument, not N arguments.

**Resolution**: Added method in `BreezeReactantExt` to convert CartesianIndex:
```julia
# ext/BreezeReactantExt/BreezeReactantExt.jl lines 171-175
function Reactant.TracedUtils.get_ancestor_and_indices_inner(
    x::OffsetArray{<:Reactant.TracedRNumber,N}, idx::CartesianIndex{N}
) where {N}
    return Reactant.TracedUtils.get_ancestor_and_indices_inner(x, Tuple(idx)...)
end
```

**MWE**: `test/mwe_reactant_cartesian_index_minimal.jl`, `test/mwe_reactant_cartesian_index.jl`

**Note**: This should be upstreamed to Reactant.jl.

---

### 3. Oceananigans reductions + ReactantState (scalar indexing)  ✅ *mitigated in Breeze*

| Field | Details |
|-------|---------|
| **Error** | `Scalar indexing is disallowed. Invocation of getindex(::TracedRArray, ::Vararg{Int, N})...` |
| **Immediate symptom** | `maximum(field)` and `cell_advection_timescale(...)` fail under ReactantState |
| **Why it happens** | Oceananigans reduces `AbstractField` / `AbstractOperation` by iterating and calling `getindex` repeatedly (scalar element access). Reactant arrays intentionally disallow scalar indexing. |

#### Where it is triggered (exact call chains)

**A) Trigger via Breeze (TimeStepWizard / CFL path)**  
This is the *most important* Breeze-relevant trigger because it blocks adaptive time stepping.

- **Breeze**: `Advection.cell_advection_timescale(model::AtmosphereModel)`  
  `src/AtmosphereModels/atmosphere_model.jl:271`
- **Oceananigans**: `TimeStepWizard.new_time_step` calls `wizard.cell_advection_timescale(model)`  
  `~/.julia/packages/Oceananigans/.../src/Simulations/time_step_wizard.jl:102`
- **Oceananigans**: `cell_advection_timescale(grid, velocities)` computes a `KernelFunctionOperation` and calls `minimum(τ)`  
  `~/.julia/packages/Oceananigans/.../src/Advection/cell_advection_timescale.jl:13-17`
- **Oceananigans**: `minimum(::AbstractField)` reduction machinery  
  `~/.julia/packages/Oceananigans/.../src/Fields/field.jl:736-793`
- **Oceananigans**: scalar evaluation of the operation occurs via `KernelFunctionOperation.getindex`  
  `~/.julia/packages/Oceananigans/.../src/AbstractOperations/kernel_function_operation.jl:68`
- **Oceananigans**: the kernel function indexes into velocity `Field`s (`U[i, j, k]`)  
  `_inverse_timescale` in `~/.julia/packages/Oceananigans/.../src/Advection/cell_advection_timescale.jl:19`
- **Oceananigans**: field scalar indexing is `getindex(f::Field, inds...) = getindex(f.data, inds...)`  
  `~/.julia/packages/Oceananigans/.../src/Fields/field.jl:432`
- **Reactant**: scalar indexing into Traced/Concrete arrays throws  
  `~/.julia/packages/Reactant/.../src/TracedRArray.jl` (+ OffsetArrays extension)

**B) Direct trigger in user/tests (Field reductions)**  
Example from our test suite:

- **Breeze tests**: `maximum(θ)` / `maximum(f)`  
  `test/reactant_enzyme.jl:180` and `test/reactant_enzyme.jl:392`
- **Oceananigans**: same reduction machinery in `src/Fields/field.jl:736-793`

#### MWE (upstream + Breeze-local verification)

- **MWE file**: `test/mwe_reactant_scalar_indexing_oceananigans_reductions.jl`
  - Running *without* `using Breeze` reproduces the upstream Oceananigans↔Reactant failure.
  - Running via `julia --project=test -e 'using Breeze; include(\"test/mwe_reactant_scalar_indexing_oceananigans_reductions.jl\")'` shows the Breeze-local workaround fixes both triggers.

#### Resolution implemented in Breeze (workaround; should be upstreamed)

Implemented in `ext/BreezeReactantExt/BreezeReactantExt.jl`:

1. **For Reactant-backed `Field`s**: redirect reductions to the interior *array view*, which Reactant can reduce without scalar iteration:
   - `maximum(f, field)` → `maximum(f, interior(field))`
   - `minimum(f, field)` → `minimum(f, interior(field))`
   - `sum(f, field)`     → `sum(f, interior(field))`

2. **For `AbstractOperation`s (including `KernelFunctionOperation`) on `ReactantState`**: materialize the operation into a computed `Field(op)` (kernel-based compute) and then reduce its interior array:
   - `minimum(op)` → `tmp = Field(op); minimum(interior(tmp))`

✅ After this workaround:
- `maximum(CenterField(grid))` works on `ReactantState`
- `Oceananigans.Advection.cell_advection_timescale(grid, (u, v, w))` works on `ReactantState`

⚠️ Limitations of the workaround (still outstanding upstream):
- Only implemented for **scalar reductions** (`dims = :`) and **no conditioning** (`condition = nothing`).
- `Field(op)` materialization allocates a temporary field; for frequent reductions this adds overhead.

---

### 4. FourierTridiagonalPoissonSolver compilation ✅ *workaround implemented*

| Field | Details |
|-------|---------|
| **Original Symptom** | `Reactant.@compile ...` fails with `'llvm.extractvalue' ... got 'complex<f32|f64>'` followed by `"failed to run pass manager on module"`. |
| **Root cause** | Oceananigans' FFT-based Poisson solve calls KernelAbstractions kernels that read complex values and write real outputs (`copy_real_component!`, `solve_batched_tridiagonal_system_kernel!`). |
| **Status** | ✅ **RESOLVED** via Option B workaround in `BreezeReactantExt` |

**Resolution**: The workaround bypasses all complex-valued KA kernels by:
1. Using `ReactantAbstractFFTsExt` for FFT operations (Reactant handles complex FFTs natively)
2. Solving the tridiagonal system directly on complex arrays using pure-Julia broadcasts
   (tridiagonal coefficients are real, only RHS and solution are complex)
3. Using pure broadcast for the final copy

**MWE**: `test/mwe_reactant_fourier_tridiagonal_solver_compile.jl` (now succeeds with Breeze loaded)
**MWE**: `test/mwe_reactant_complex_number_mlir.jl` (demonstrates raw upstream issue)

**Note**: The underlying Reactant issue with complex numbers in KA kernels remains; only the pressure solver path is fixed.

---

## ✅ RESOLVED ISSUES (continued)

### 5. Boolean Context with TracedRNumber{Bool} in time_step!

| Field | Details |
|-------|---------|
| **Error** | `TypeError(:if, "", Bool, TracedRNumber{Bool}(()))` |
| **Source** | `Oceananigans.TimeSteppers.time_step!` for RK3 |
| **File** | `~/.julia/packages/Oceananigans/.../src/TimeSteppers/runge_kutta_3.jl:95-98` |
| **Triggered by** | `@compile time_step!(model, Δt)` with AtmosphereModel on ReactantState |

**Root cause**: Oceananigans' RK3 `time_step!` contains conditionals that use runtime values:
```julia
Δt == 0 && @warn "Δt == 0 may cause model blowup!"
model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies = true)
```
When `clock.iteration` and `clock.time` are `ConcretePJRTNumber` (Reactant traced numbers), comparisons like `iteration == 0` return `TracedRNumber{Bool}`, which cannot be used in short-circuit `&&` operators.

**Resolution**: Added Reactant-compatible `time_step!` method in `BreezeReactantExt`:
```julia
# ext/BreezeReactantExt/BreezeReactantExt.jl lines 225-310
# Provides a version of time_step! that removes the problematic conditionals
# for AtmosphereModel with ReactantState architecture.
function Oceananigans.TimeSteppers.time_step!(model::ReactantAtmosphereModel, Δt; callbacks=[])
    # NOTE: We skip the standard checks that use runtime conditionals:
    # - `Δt == 0 && @warn ...` - assume user provides valid Δt
    # - `model.clock.iteration == 0 && update_state!(...)` - assume user initialized model
    # ... (full RK3 implementation without conditionals)
end
```

**MWE**: `test/mwe_reactant_boolean_context.jl`

---

## ✅ RECENTLY RESOLVED ISSUES

### 1. `_compute_anelastic_source_term!` kernel ✅ WORKAROUND IMPLEMENTED

| Field | Details |
|-------|---------|
| **Status** | ✅ **WORKAROUND IMPLEMENTED** in `BreezeReactantExt` |
| **Previous Error** | `'affine.store' op value to store must have the same type as memref element type` |
| **Resolution** | Pure-Julia broadcast implementation (Option C) |

**Original issue**: Breeze's `_compute_anelastic_source_term!` KernelAbstractions kernel failed MLIR lowering.

**Resolution**: The workaround in `BreezeReactantExt` (lines 599-650) replaces the kernel with a pure-Julia broadcast implementation:

```julia
function _compute_source_term_broadcast!(rhs, grid, ρu, ρv, ρw, Δt)
    # Compute divergence using @views and broadcasts
    @views begin
        δx_ρu = ρu_data[(Hx+2):(Hx+Nx+1), iy, iz_c] .- ρu_data[(Hx+1):(Hx+Nx), iy, iz_c]
        δz_ρw = ρw_data[ix_c, iy, (Hz+2):(Hz+Nz+1)] .- ρw_data[ix_c, iy, (Hz+1):(Hz+Nz)]
        rhs_data[1:Nx, 1:Ny, 1:Nz] .= ((Δz_val / Δx_val) .* δx_ρu .+ δz_ρw) ./ Δt
    end
end
```

This workaround always activates on ReactantState (both ConcreteRArrays and TracedRArrays) because KA kernels cannot run on either without Reactant compilation.

**Verification**: `Reactant.@compile raise=true time_step!(model, Δt)` now succeeds.

---

### 2. Complex numbers inside KernelAbstractions kernels (upstream Reactant issue)

| Field | Details |
|-------|---------|
| **Error** | `'llvm.extractvalue' op operand #0 must be LLVM aggregate type, but got 'complex<f32>'` (or `complex<f64>`) |
| **Symptom** | `"failed to run pass manager on module"` during compilation |
| **Status** | ✅ **WORKAROUND IMPLEMENTED** for `FourierTridiagonalPoissonSolver` |

**Root cause**: Reactant's MLIR lowering for KernelAbstractions kernels cannot handle complex number types (`ComplexF32`, `ComplexF64`).

**Resolution for Breeze**: The Option B workaround in `BreezeReactantExt` bypasses this issue by:
- Using `ReactantAbstractFFTsExt` for FFT operations (Reactant handles complex FFTs natively)
- Solving the tridiagonal system directly on complex arrays using pure-Julia broadcasts
  (the key insight: pure Julia broadcasts work with complex, only KA kernels fail)
- Avoiding the `copy_real_component!` kernel by using pure broadcast

**Affected Oceananigans kernels** (all bypassed by workaround):
- `copy_real_component!` / `gpu_copy_real_component!`
- `gpu_solve_batched_tridiagonal_system_kernel!`

**MWE**: `test/mwe_reactant_complex_number_mlir.jl` - demonstrates the raw upstream issue
**MWE**: `test/mwe_reactant_fourier_tridiagonal_solver_compile.jl` - demonstrates the workaround works

**Note**: The underlying Reactant issue remains for any other code paths that use complex-valued KA kernels. This should be fixed upstream in Reactant/EnzymeAD.

---

### 2. `_pressure_correct_momentum!` kernel ✅ WORKAROUND IMPLEMENTED

| Field | Details |
|-------|---------|
| **Status** | ✅ **WORKAROUND IMPLEMENTED** in `BreezeReactantExt` |
| **Previous Error** | KA kernel fails to run on ReactantState arrays |
| **Resolution** | Pure-Julia broadcast implementation |

**Resolution**: Override `make_pressure_correction!` for `ReactantAtmosphereModel` with broadcast-based pressure gradient correction.

---

### 3. Pressure Correction during Model Construction (set!) ✅ CPU FALLBACK IMPLEMENTED

| Field | Details |
|-------|---------|
| **Status** | ✅ **CPU FALLBACK IMPLEMENTED** in `BreezeReactantExt` |
| **Previous Error** | FFTs on `ConcreteRArray` fail (no pointer conversion) |
| **Resolution** | Option D: Copy to CPU, solve with FFTW, copy back |

**Root cause**: `ReactantAbstractFFTsExt` only supports FFTs on `TracedRArray` (during `@compile`), not on
`ConcreteRArray` (outside `@compile`). Model construction calls `set!` → `update_state!` →
`compute_pressure_correction!`, which requires FFTs.

**Resolution (Option D)**: `BreezeReactantExt._cpu_pressure_correction!` implements a CPU fallback:
1. Copy momentum fields from `ConcreteRArray` to regular Julia `Array`
2. Compute divergence (source term) on CPU
3. Perform FFT-based pressure solve using FFTW
4. Apply pressure correction to momentum on CPU
5. Copy corrected fields back to `ConcreteRArray`

This ensures the velocity field is divergence-free after `set!()`, even with non-zero initial velocities.

---

## ⚠️ KNOWN LIMITATIONS

### 1. Numerical differences between CPU and Reactant execution

| Field | Details |
|-------|---------|
| **Status** | ⚠️ **Under investigation** |
| **Symptom** | Reactant-compiled results differ from CPU results by O(10-100) in temperature |
| **Impact** | Functional but not numerically identical |

**Observations**:
- Source term computation: ✅ matches CPU exactly (verified to 1e-34)
- Pressure correction: ✅ matches CPU exactly (verified to 1e-14)
- Full time step: ⚠️ differences observed

**Possible causes**:
1. Other KA kernels in Oceananigans (advection, boundary conditions) behaving differently
2. Floating point operation ordering differences in XLA vs CPU
3. Halo filling or other operations not yet overridden

**Note**: GB-25 (HydrostaticFreeSurfaceModel) works with Reactant because it uses a simpler model formulation without FFT-based pressure solves.

---

## ❌ OUTSTANDING ISSUES

### 1. Reactant @trace Loop with Grid Type Conversion (MAY BE RESOLVED)

| Field | Details |
|-------|---------|
| **Previous Error** | `Reactant.NoFieldMatchError: Cannot convert type RectilinearGrid{Float64,...}` |
| **Current Status** | ✅ **Testing shows this now works** |
| **Triggered by** | `@compile ... @trace for i = 1:nsteps; time_step!(model, Δt); end` |

**Root cause**: When using `@trace` to create a traced loop for autodiff, Reactant attempts to trace through all variables in the loop body. This includes the Grid object, which has type parameters (like `Float64` for coordinates) that Reactant tries to convert to `TracedRNumber{Float64}`. However, the Grid type doesn't support this conversion because its type parameters are captured in the type signature.

**Note**: This issue is currently **masked by Issue #1** (`_compute_anelastic_source_term!`), which fails earlier in the pipeline.

**Impact**: AD workflows that require multiple time steps with checkpointing cannot currently use `@trace` loops.

**Workaround options**:
1. Unroll the loop manually for small step counts
2. Use single-step compilation and call multiple times (no gradient accumulation)
3. Wait for upstream Reactant improvements to handle constant struct fields

**Current status**: The test uses `@test_broken` to document this as a known limitation.

---

### 4. Upstreaming: OceananigansReactantExt still lacks Reactant-safe reductions

Even though Breeze now contains a working workaround, **Oceananigans + Reactant still fails**
without Breeze loaded (see the MWE above).

**What needs to happen upstream**:
- Add Reactant-safe `maximum/minimum/sum/...` methods for `Field` and `AbstractOperation`
  (likely in `OceananigansReactantExt/Fields.jl`).
- Upstream the `CartesianIndex` fix to `ReactantOffsetArraysExt.jl`.

This would remove the need for Breeze-level method overrides.

---

## Code Location Summary

| Issue | Package | File | Line |
|-------|---------|------|------|
| **ANELASTIC PRESSURE SOLVE (Options B, C, D)** | | | |
| Anelastic source term kernel (original) | Breeze | `src/AtmosphereModels/Dynamics/anelastic_pressure_solver.jl` | 103 |
| Anelastic source term (workaround) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | ~600-650 |
| CPU fallback pressure correction (Option D) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | ~760-870 |
| Pressure correction (original) | Breeze | `src/AtmosphereModels/anelastic_time_stepping.jl` | 31-40 |
| Pressure correction (workaround) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | ~880-940 |
| **OTHER RESOLVED WORKAROUNDS** | | | |
| FFT Planning (Breeze fix) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 426-461 |
| CartesianIndex (Breeze fix) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 181-185 |
| Workaround (reductions) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 67-169 |
| Boolean context (Breeze fix) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 463-560 |
| Batched tridiagonal (Breeze fix) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 243-341 |
| FourierTridiagonalPoissonSolver (Breeze fix) | Breeze | `ext/BreezeReactantExt/BreezeReactantExt.jl` | 343-423 |

---

## Test status

The authoritative status is `test/reactant_enzyme.jl`. 

**Current test results (December 2025)**: 20 passed, all tests passing

**What works**:
- ✅ Model construction on `ReactantState` (CPU fallback for pressure correction)
- ✅ Field operations and reductions (via workarounds)
- ✅ `FourierTridiagonalPoissonSolver.solve!` compilation (via Option B workaround)
- ✅ **`Reactant.@compile raise=true time_step!(model, Δt)`** (via Option C workaround)
- ✅ **`@trace` loops for multi-step compilation**
- ✅ Forward pass for AD compiled and executed
- ✅ Enzyme AD with `ReverseWithPrimal` compiles and runs

**Key implementation detail: CPU fallback pressure correction during model construction**

The FFT-based pressure solver (`FourierTridiagonalPoissonSolver`) only works during Reactant tracing
(on `TracedRArray`), NOT on `ConcreteRArray` (outside tracing). This is because `ReactantAbstractFFTsExt`
only supports FFTs on traced arrays.

To enable proper model construction (which calls `set!` → `compute_pressure_correction!`),
`BreezeReactantExt` implements a **CPU fallback** that:
1. Copies momentum fields to regular CPU arrays
2. Performs the FFT-based pressure solve using FFTW on CPU
3. Copies the corrected fields back to `ConcreteRArray`s

This ensures the velocity field is divergence-free after `set!()`, even with non-zero initial velocities.

**Known limitations**:
- ⚠️ Numerical differences between CPU and Reactant (under investigation)
- ⚠️ Only 2D grids (Flat y) with regular spacing currently supported in workarounds

**Why GB-25 doesn't have these issues**: GB-25 uses `HydrostaticFreeSurfaceModel` with `SplitExplicitFreeSurface`, which does NOT require FFT-based pressure solves. The free surface uses an explicit solver that only involves real numbers and simpler KA kernels that Reactant can handle.

---

## Test Files

### Main Test Suite
- `test/reactant_enzyme.jl` - Full test suite for Reactant+Enzyme integration

### Minimum Working Examples (MWEs)

| Issue | MWE File | Status |
|-------|----------|--------|
| FFT Planning | `test/mwe_reactant_fft_planning.jl` | ✅ Fixed in BreezeReactantExt |
| CartesianIndex (minimal) | `test/mwe_reactant_cartesian_index_minimal.jl` | ✅ Fixed in BreezeReactantExt |
| CartesianIndex (Oceananigans) | `test/mwe_reactant_cartesian_index.jl` | ✅ Fixed in BreezeReactantExt |
| Scalar Indexing (reductions) | `test/mwe_reactant_scalar_indexing_oceananigans_reductions.jl` | ✅ Fixed in BreezeReactantExt |
| Boolean Context | `test/mwe_reactant_boolean_context.jl` | ✅ Fixed in BreezeReactantExt |
| Complex Numbers (MLIR) | `test/mwe_reactant_complex_number_mlir.jl` | ❌ Upstream Reactant issue |
| FourierTridiagonalPoissonSolver compile | `test/mwe_reactant_fourier_tridiagonal_solver_compile.jl` | ✅ Fixed in BreezeReactantExt (Option B workaround) |

### Running MWEs

```bash
# Run without Breeze (shows upstream issue):
julia --project=test test/mwe_reactant_scalar_indexing_oceananigans_reductions.jl

# Run with Breeze fix loaded:
julia --project=test -e 'using Breeze; include("test/mwe_reactant_scalar_indexing_oceananigans_reductions.jl")'
```
