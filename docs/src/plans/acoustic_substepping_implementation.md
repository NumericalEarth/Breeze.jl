# Implementation Plan: Acoustic Substepping for CompressibleDynamics

## Overview

Acoustic substepping is a split-explicit time integration method that allows explicit integration of the fully compressible Euler equations without prohibitively small time steps. The approach separates the "slow" physics (advection, diffusion, buoyancy) from the "fast" acoustic modes (pressure gradient, divergence), integrating them on different time scales within a Runge-Kutta framework.

This plan follows the approach used in CM1 ([Bryan and Fritsch, 2002](https://doi.org/10.1175/1520-0493(2002)130<2088:TSMFEM>2.0.CO;2); [Wicker and Skamarock, 2002](https://doi.org/10.1175/1520-0493(2002)130<2088:TSMFEM>2.0.CO;2)).

### Scope

**In scope:**
- Wicker-Skamarock RK3 with acoustic substepping
- Vertically implicit acoustic solver using `BatchedTridiagonalSolver`
- Correct moist sound speed computation
- GPU compatibility via KernelAbstractions.jl

**Out of scope (future work):**
- Open/radiating boundary conditions for acoustic waves
- Terrain-following coordinates

---

## Physical Background

### The Compressible Euler Equations

The fully compressible equations prognosticate density ρ, momentum ρu, and a thermodynamic variable (potential temperature density ρθ or static energy density ρe). The acoustic modes arise from coupling between:

1. **Pressure gradient in momentum equation:** ∂(ρu)/∂t = ... - ∇p
2. **Compression in continuity equation:** ∂ρ/∂t = ... - ρ ∇·u

where the pressure p = ρ Rᵐ T is diagnosed from the equation of state. These terms support acoustic waves that propagate at speed c ≈ 340 m/s (where c² = γᵐ Rᵐ T), requiring very small time steps (Δt ~ 0.1 s for Δx = 100 m) for explicit integration.

### Sound Speed with Moisture

The moist adiabatic sound speed is:

```math
c^2 = \gamma^m R^m T
```

where:
- γᵐ = cₚᵐ / cᵥᵐ is the mixture heat capacity ratio
- Rᵐ = (1 - qᵗ)Rᵈ + qᵛRᵛ is the mixture gas constant
- T is temperature

For accurate acoustic wave propagation, the sound speed must include moisture effects consistently.

### Split-Explicit Time Integration

The Wicker-Skamarock scheme uses a 3-stage Runge-Kutta outer loop with acoustic substepping inside each stage:

| RK Stage | Stage Δt | Acoustic Substeps |
|----------|----------|-------------------|
| nrk = 1 | Δt/3 | nsound/3 |
| nrk = 2 | Δt/2 | nsound/2 |
| nrk = 3 | Δt | nsound |

Within each RK stage:
1. Compute slow tendencies (advection, buoyancy, turbulence) — held fixed during acoustic loop
2. Execute acoustic substep loop for u, v, w, and density ρ (pressure p diagnosed from EOS)
3. Use time-averaged velocities from acoustic loop for scalar advection
4. Advance scalars (θ, moisture) using time-averaged velocities

---

## Implementation Architecture

### New Time Stepper: `AcousticSSPRungeKutta3`

**File:** `src/TimeSteppers/acoustic_ssp_runge_kutta_3.jl`

```julia
"""
Time stepper implementing Wicker-Skamarock RK3 with acoustic substepping
for fully compressible dynamics.

The acoustic substepping uses a forward-backward scheme in the horizontal
and an implicit tridiagonal solve in the vertical for stability.
"""
struct AcousticSSPRungeKutta3{FT, U0, TG, TI, AS} <: AbstractTimeStepper
    # RK3 stage coefficients (same as SSP-RK3)
    α¹ :: FT  # = 1
    α² :: FT  # = 1/4
    α³ :: FT  # = 2/3

    # Storage for state at beginning of time step
    U⁰ :: U0

    # Tendencies
    Gⁿ :: TG

    # Implicit solver (for vertical diffusion, if any)
    implicit_solver :: TI

    # Acoustic substepping infrastructure
    acoustic :: AS
end
```

### Acoustic Substepper Storage

**File:** `src/CompressibleEquations/acoustic_substepping.jl`

See the optimized `AcousticSubstepper` struct in the Core Algorithms section, which includes:
- Precomputed thermodynamic coefficients (ψ = Rᵐ T, c²) for on-the-fly pressure computation
- Time-averaged velocity fields
- Slow tendency storage
- Reference density for divergence damping
- Vertical tridiagonal solver

### Vertical Implicit Solver

Using Oceananigans' `BatchedTridiagonalSolver` for GPU-compatible vertical solves:

```julia
function build_acoustic_vertical_solver(grid)
    # The tridiagonal system couples w and ρ in the vertical
    # Coefficients depend on sound speed and grid metrics

    Nz = size(grid, 3)
    arch = architecture(grid)
    FT = eltype(grid)

    # Preallocate coefficient arrays
    lower_diagonal = zeros(arch, FT, Nz)
    diagonal = zeros(arch, FT, grid.Nx, grid.Ny, Nz)
    upper_diagonal = zeros(arch, FT, Nz)
    scratch = zeros(arch, FT, grid.Nx, grid.Ny, Nz)

    return BatchedTridiagonalSolver(grid;
                                    lower_diagonal,
                                    diagonal,
                                    upper_diagonal,
                                    scratch,
                                    tridiagonal_direction = ZDirection())
end
```

---

## Core Algorithms

### Performance Strategy: On-the-Fly Pressure Computation

**Key insight:** During acoustic substepping, temperature T and thermodynamic properties (Rᵐ, γᵐ) are
**held fixed** - they evolve via slow tendencies outside the acoustic loop.

Since p = ρ Rᵐ T, the pressure gradient splits into fast and slow parts:
```math
\frac{\partial p}{\partial x} = \frac{\partial (\rho R^m T)}{\partial x} =
    \underbrace{R^m T \frac{\partial \rho}{\partial x}}_{\text{fast (acoustic)}} +
    \underbrace{\rho \frac{\partial (R^m T)}{\partial x}}_{\text{slow (buoyancy)}}
```

- **Fast term:** Changes each substep (ρ evolves) → computed on-the-fly
- **Slow term:** Constant during substeps → included in `G_slow`, computed once per RK stage

**Consequence:** We never store pressure during acoustic substepping. Instead:
1. **Precompute once:** Pressure coefficient ψ = Rᵐ T and sound speed c² = γᵐ ψ
2. **Compute on-the-fly:** Fast pressure gradient = ψ ∂ρ/∂x

This follows the pattern from Oceananigans' `SplitExplicitFreeSurface`.

### Acoustic Substepper Storage (Optimized)

```julia
"""
Storage for acoustic substepping. Follows Oceananigans.SplitExplicit patterns
for GPU performance.
"""
struct AcousticSubstepper{N, FT, F3D, F2D, TS}
    # Number of acoustic substeps per full time step
    nsound :: N

    # Implicitness parameters (Crank-Nicolson: α = β = 0.5)
    α :: FT  # Implicit weight
    β :: FT  # Explicit weight (1 - α)

    # Divergence damping coefficient
    kdiv :: FT

    # Precomputed thermodynamic coefficients (computed once per RK stage)
    # ψ = Rᵐ T (pressure coefficient): p = ψ ρ
    # c² = γᵐ ψ (sound speed squared)
    ψ  :: F3D  # CenterField
    c² :: F3D  # CenterField

    # Time-averaged velocities for scalar advection
    ū :: F3D  # XFaceField
    v̄ :: F3D  # YFaceField
    w̄ :: F3D  # ZFaceField

    # Slow tendencies (computed once per RK stage, held fixed during acoustic loop)
    G_slow_ρu :: F3D
    G_slow_ρv :: F3D
    G_slow_ρw :: F3D

    # Reference density at start of acoustic loop (for divergence damping)
    ρ_ref :: F3D

    # Vertical tridiagonal solver for implicit w-ρ coupling
    vertical_solver :: TS
end
```

### Acoustic Substep Loop (Optimized)

Following Oceananigans.SplitExplicit pattern with pre-converted kernel arguments:

```julia
function acoustic_substep_loop!(model, nrk, Δt_rk, nsound)
    substepper = model.timestepper.substepper
    grid = model.grid
    arch = architecture(grid)
    dynamics = model.dynamics

    # Number of substeps for this RK stage
    nloop = acoustic_substeps_per_stage(nrk, nsound)
    Δts = Δt_rk / nloop

    # === PRECOMPUTE PHASE (once per RK stage) ===

    # Compute thermodynamic coefficients: ψ = Rᵐ T, c² = γᵐ ψ
    compute_acoustic_coefficients!(acoustic.ψ, acoustic.c², model)

    # Store density reference for divergence damping
    parent(acoustic.ρ_ref) .= parent(dynamics.density)

    # Initialize time-averaged velocities
    fill!(acoustic.ū, 0)
    fill!(acoustic.v̄, 0)
    fill!(acoustic.w̄, 0)

    # === PRE-CONVERT KERNEL ARGUMENTS (GPU optimization) ===
    # Following Oceananigans.SplitExplicit pattern to minimize latency

    momentum_args = (grid, Δts,
                     dynamics.ρu, dynamics.ρv, dynamics.density,
                     acoustic.ψ, acoustic.G_slow_ρu, acoustic.G_slow_ρv)

    density_args = (grid, Δts,
                    dynamics.density, dynamics.u, dynamics.v, dynamics.w,
                    acoustic.c², acoustic.ρ_ref, acoustic.kdiv,
                    acoustic.ū, acoustic.v̄, acoustic.w̄)

    GC.@preserve momentum_args density_args begin
        converted_momentum_args = convert_to_device(arch, momentum_args)
        converted_density_args = convert_to_device(arch, density_args)

        # === ACOUSTIC SUBSTEP LOOP ===
        for n = 1:nloop
            weight = n == nloop ? 1 / nloop : 1 / nloop  # Averaging weight

            # Horizontal momentum: ∂(ρu)/∂t = -ψ ∂ρ/∂x + G_slow
            acoustic_horizontal_momentum_kernel!(converted_momentum_args...)

            # Vertical implicit solve for ρw and ρ
            acoustic_implicit_vertical_step!(model, Δts, acoustic.α, acoustic.β)

            # Density update + damping + velocity averaging (combined kernel)
            acoustic_density_and_averaging_kernel!(weight, n, nloop, converted_density_args...)
        end
    end

    # Diagnose pressure from final density for slow tendency computation
    compute_pressure_from_eos!(dynamics.pressure, model)

    return nothing
end
```

### Horizontal Momentum Update (Explicit, On-the-Fly Pressure, Topology-Aware)

The fast pressure gradient ψ ∂ρ/∂x is computed on-the-fly using **topology-aware operators**:

```julia
@kernel function _acoustic_horizontal_momentum!(ρu, ρv, grid, Δts, ρ, ψ, G_slow_ρu, G_slow_ρv)
    i, j, k = @index(Global, NTuple)

    # Fast pressure gradient: (∂p/∂x)_fast = ψ ∂ρ/∂x where ψ = Rᵐ T
    # Uses topology-aware operators (∂xTᶠᶜᶜ, ∂yTᶜᶠᶜ) to avoid halo accesses
    @inbounds begin
        # u-component: pressure gradient at (Face, Center, Center)
        ψᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ψ)
        ∂ρ_∂x = ∂xTᶠᶜᶜ(i, j, k, grid, ρ)  # Topology-aware!
        ∂p_∂x_fast = ψᶠᶜᶜ * ∂ρ_∂x

        # Total tendency = -fast pressure gradient + slow tendency
        ρu[i, j, k] += Δts * (-∂p_∂x_fast + G_slow_ρu[i, j, k])

        # v-component: pressure gradient at (Center, Face, Center)
        ψᶜᶠᶜ = ℑyᵃᶠᵃ(i, j, k, grid, ψ)
        ∂ρ_∂y = ∂yTᶜᶠᶜ(i, j, k, grid, ρ)  # Topology-aware!
        ∂p_∂y_fast = ψᶜᶠᶜ * ∂ρ_∂y

        ρv[i, j, k] += Δts * (-∂p_∂y_fast + G_slow_ρv[i, j, k])
    end
end
```

**Performance notes:**
1. `∂xTᶠᶜᶜ` and `∂yTᶜᶠᶜ` are topology-aware operators that encode boundary conditions
   based on grid type (Periodic, Bounded, etc.) - **no halo filling needed between substeps!**
2. The interpolation ψᶠᶜᶜ = ℑxᶠᵃᵃ(i, j, k, grid, ψ) is a simple 2-point average,
   much cheaper than storing/reading a full pressure field.

### Vertical Implicit Solve

The vertical velocity w and density ρ are coupled through the linearized acoustic equations:

```math
\frac{\partial (\rho w)}{\partial t} = -\frac{\partial p}{\partial z} + \text{(slow terms)} \approx -c^2 \frac{\partial \rho}{\partial z} + \text{(slow)}
```
```math
\frac{\partial \rho}{\partial t} = -\rho \frac{\partial w}{\partial z}
```

where c² = γᵐ Rᵐ T is the (moist) sound speed squared and we use the linearized relation ∂p/∂z ≈ c² ∂ρ/∂z.

Discretizing implicitly in the vertical gives a tridiagonal system for w:

```julia
function acoustic_implicit_vertical_step!(model, Δts, α, β)
    # Build tridiagonal coefficients
    compute_vertical_tridiagonal_coefficients!(model, Δts, α)

    # Compute RHS from current state
    compute_vertical_rhs!(model, Δts, α, β)

    # Solve for w^{n+1}
    solve!(model.momentum.ρw, model.timestepper.substepper.vertical_solver, rhs)

    # Update density from new w
    update_density_from_w!(model, Δts, α, β)
end

@kernel function _compute_vertical_tridiagonal_coefficients!(lower, diag, upper,
                                                              grid, c², ρ, Δts, α)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    α² = α * α

    @inbounds for k = 2:Nz
        # Coefficients from linearized w-ρ coupling
        #
        # ρw equation: ∂(ρw)/∂t = -∂p/∂z ≈ -c² ∂ρ/∂z
        # ρ equation:  ∂ρ/∂t = -ρ ∂w/∂z  (vertical compression)
        #
        # Eliminating ρ gives a wave equation for w with tridiagonal structure

        Δz_k = Δzᶜᶜᶠ(i, j, k, grid)
        Δz_km1 = Δzᶜᶜᶠ(i, j, k-1, grid)
        Δz_c = Δzᶜᶜᶜ(i, j, k, grid)

        ρ_k = ρ[i, j, k]
        c²_k = c²[i, j, k]

        # Coupling coefficients from second-order discretization
        # (derived from eliminating ρ between the two equations)
        coeff_upper = α² * Δts² * c²_k / (Δz_k * Δz_c)
        coeff_lower = α² * Δts² * c²_k / (Δz_km1 * Δz_c)

        lower[i, j, k-1] = -coeff_lower
        upper[i, j, k] = -coeff_upper
        diag[i, j, k] = 1 + coeff_lower + coeff_upper
    end
end
```

### Thermodynamic Coefficients (Precomputed Once Per RK Stage)

Compute both the pressure coefficient ψ = Rᵐ T and sound speed c² = γᵐ ψ:

```julia
@kernel function _compute_acoustic_coefficients!(ψ, c², grid, dynamics,
                                                  formulation, microphysics,
                                                  microphysical_fields,
                                                  temperature, constants)
    i, j, k = @index(Global, NTuple)

    # Get moisture
    qᵗ = specific_moisture[i, j, k]

    # Compute moisture fractions (liquid, ice, vapor)
    q = compute_moisture_fractions(i, j, k, grid, microphysics, qᵗ, microphysical_fields)

    # Mixture thermodynamic properties
    Rᵐ = mixture_gas_constant(q, constants)
    cₚᵐ = mixture_heat_capacity(q, constants)
    cᵥᵐ = cₚᵐ - Rᵐ
    γᵐ = cₚᵐ / cᵥᵐ

    # Temperature (already computed in update_state!)
    T = temperature[i, j, k]

    @inbounds begin
        # Pressure coefficient: p = ψ ρ
        ψ[i, j, k] = Rᵐ * T

        # Moist sound speed squared: c² = γᵐ ψ = γᵐ Rᵐ T
        c²[i, j, k] = γᵐ * ψ[i, j, k]
    end
end
```

**Why two fields?**
- `ψ = Rᵐ T`: Used for on-the-fly pressure gradient computation (∂p/∂x = ψ ∂ρ/∂x)
- `c²`: Used for vertical implicit solve and density update from ∂w/∂z

### Combined Density Update + Damping + Velocity Averaging (Topology-Aware)

Following Oceananigans.SplitExplicit, we combine multiple operations into a single kernel
using **topology-aware operators** for divergence:

```julia
@kernel function _acoustic_density_and_averaging!(weight, n, nloop, grid, Δts,
                                                   ρ, u, v, w, c², ρ_ref, kdiv,
                                                   ū, v̄, w̄)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # === Density update from compression ===
        # ∂ρ/∂t = -ρ ∇·u (fast/acoustic part of continuity)
        # Uses topology-aware operators for divergence - no halo filling needed!
        div_u = δxTᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u) * Ax⁻¹ᶜᶜᶜ(i, j, k, grid) +
                δyTᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v) * Ay⁻¹ᶜᶜᶜ(i, j, k, grid) +
                δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, w) * V⁻¹ᶜᶜᶜ(i, j, k, grid)

        ρ[i, j, k] -= Δts * ρ[i, j, k] * div_u

        # === Divergence damping ===
        # Damps spurious acoustic oscillations
        ρ[i, j, k] += kdiv * (ρ[i, j, k] - ρ_ref[i, j, k])

        # === Accumulate time-averaged velocities ===
        # Following SplitExplicit pattern for scalar transport
        ū[i, j, k] += weight * u[i, j, k]
        v̄[i, j, k] += weight * v[i, j, k]
        w̄[i, j, k] += weight * w[i, j, k]
    end
end
```

**Key:** `δxTᶜᵃᵃ` and `δyTᵃᶜᵃ` are topology-aware difference operators that handle:
- **Periodic:** Wrap around at domain boundaries
- **Bounded:** Apply impenetrability (u=0 at walls)
- No need for `fill_halo_regions!` between substeps!

Note: The advection part (-u·∇ρ) is handled in the slow tendency computation outside the acoustic loop.

### Pressure Diagnosis from EOS

Pressure is diagnosed **only at the end of the acoustic loop** (not each substep!) for:
1. Computing slow tendencies for the next RK stage
2. Pressure-based diagnostics/output

```julia
@kernel function _compute_pressure_from_eos!(p, grid, ρ, ψ)
    i, j, k = @index(Global, NTuple)

    # p = ρ ψ where ψ = Rᵐ T (precomputed)
    @inbounds p[i, j, k] = ρ[i, j, k] * ψ[i, j, k]
end
```

---

## File Structure

```
src/
├── TimeSteppers/
│   ├── TimeSteppers.jl                    # Add export
│   ├── ssp_runge_kutta_3.jl               # Existing (anelastic)
│   └── acoustic_ssp_runge_kutta_3.jl      # NEW
│
├── CompressibleEquations/
│   ├── CompressibleEquations.jl           # Add includes
│   ├── compressible_dynamics.jl           # Add AcousticParameters
│   ├── compressible_buoyancy.jl           # Existing
│   ├── compressible_density_tendency.jl   # Existing
│   ├── compressible_time_stepping.jl      # Update
│   │
│   ├── acoustic_substepping.jl            # NEW: main acoustic loop
│   ├── acoustic_coefficients.jl           # NEW: c² (sound speed squared)
│   ├── acoustic_horizontal.jl             # NEW: u,v explicit update
│   └── acoustic_vertical.jl               # NEW: w,ρ implicit solve
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (2-3 days)

**Deliverables:**
1. `AcousticParameters` struct and integration into `CompressibleDynamics`
2. `AcousticSubstepper` storage struct with:
   - Time-averaged velocity fields
   - Slow tendency storage
   - Tridiagonal solver allocation
3. `AcousticSSPRungeKutta3` time stepper skeleton
4. Constructor and initialization

**Files:**
- `src/CompressibleEquations/acoustic_substepping.jl`
- `src/TimeSteppers/acoustic_ssp_runge_kutta_3.jl`
- Modifications to `src/CompressibleEquations/compressible_dynamics.jl`

### Phase 2: Thermodynamic Coefficients (1-2 days)

**Deliverables:**
1. Moist sound speed squared: c² = γᵐ Rᵐ T
2. Pressure equation coefficient: γᵐ p (or equivalently c² ρ)
3. Density interpolation utilities for staggered grid

**Files:**
- `src/CompressibleEquations/acoustic_coefficients.jl`

**Key consideration:** Ensure sound speed includes moisture correctly:
```julia
# Mixture properties
Rᵐ = (1 - qᵗ) * Rᵈ + qᵛ * Rᵛ
cₚᵐ = (1 - qᵗ) * cₚᵈ + qᵛ * cₚᵛ + qˡ * cₚˡ + qⁱ * cₚⁱ
cᵥᵐ = cₚᵐ - Rᵐ
γᵐ = cₚᵐ / cᵥᵐ

# Moist sound speed squared
c² = γᵐ * Rᵐ * T
```

### Phase 3: Horizontal Momentum Update (1-2 days)

**Deliverables:**
1. Explicit pressure gradient kernel for u, v
2. Slow tendency accumulation and storage
3. Time-averaged velocity accumulation

**Files:**
- `src/CompressibleEquations/acoustic_horizontal.jl`

### Phase 4: Vertical Implicit Solve (3-4 days)

**Deliverables:**
1. Tridiagonal coefficient computation kernel
2. RHS computation kernel
3. Integration with `BatchedTridiagonalSolver`
4. Density update from solved w (and pressure diagnosis from EOS)

**Files:**
- `src/CompressibleEquations/acoustic_vertical.jl`

**Key implementation:**
```julia
function build_vertical_acoustic_solver(grid)
    # Use Oceananigans BatchedTridiagonalSolver
    return BatchedTridiagonalSolver(grid;
        lower_diagonal = ...,
        diagonal = ...,
        upper_diagonal = ...,
        tridiagonal_direction = ZDirection())
end
```

### Phase 5: Full Time Stepper Integration (2-3 days)

**Deliverables:**
1. Complete `time_step!` implementation for `AcousticSSPRungeKutta3`
2. Slow tendency computation and caching
3. Scalar advection with time-averaged velocities
4. RK stage management

**Files:**
- `src/TimeSteppers/acoustic_ssp_runge_kutta_3.jl`
- `src/CompressibleEquations/compressible_time_stepping.jl`

### Phase 6: Testing and Validation (3-4 days)

**Unit Tests:**
1. Acoustic coefficient kernels
2. Horizontal momentum update kernel
3. Vertical tridiagonal solve
4. Time-averaged velocity accumulation

**Validation Cases:**
1. **Linear gravity wave**: Verify wave speed matches theory for dry and moist cases
2. **Acoustic pulse**: Verify acoustic wave propagation at correct speed
3. **Rising thermal bubble**: Compare with anelastic solution (should match for small perturbations)
4. **Density current**: Classical benchmark for compressible codes

**Files:**
- `test/test_acoustic_substepping.jl`
- `validation/acoustic_pulse.jl`
- `validation/compressible_gravity_wave.jl`

---

## Key Design Decisions

### 1. Performance Optimizations (Following Oceananigans.SplitExplicit)

Four key optimizations from Oceananigans' split-explicit free surface solver:

**a) On-the-fly pressure gradient computation:**
- **Problem:** Computing and storing pressure each substep is expensive
- **Solution:** Precompute ψ = Rᵐ T once; compute ∂p/∂x = ψ ∂ρ/∂x on-the-fly
- **Savings:** One fewer 3D field read/write per substep (50+ substeps = significant)

**b) Pre-converted kernel arguments:**
```julia
GC.@preserve args begin
    converted_args = convert_to_device(arch, args)
    for n = 1:nloop
        kernel!(converted_args...)  # No conversion overhead
    end
end
```
- **Problem:** GPU kernel launch latency from argument conversion
- **Solution:** Convert once before loop, reuse converted arguments
- **Reference:** Oceananigans `step_split_explicit_free_surface.jl` lines 82-97

**c) Combined kernels:**
- Density update + divergence damping + velocity averaging in one kernel
- Reduces kernel launch overhead (especially for ~50+ substeps)

**d) Topology-aware operators (no halo filling between substeps!):**
- **Problem:** `fill_halo_regions!` between each substep is expensive (GPU sync, memory bandwidth)
- **Solution:** Use topology-aware operators (`∂xTᶠᶜᶜ`, `δxTᶜᵃᵃ`, etc.) that encode boundary conditions
  directly based on grid topology type

The `T` operators (defined in `Oceananigans.Operators.topology_aware_operators.jl`) specialize on:
- `Periodic`: Wrap around at boundaries
- `Bounded`: Apply no-flux/impenetrability conditions
- `LeftConnected`, `RightConnected`: Handle connected (e.g., multi-region) domains

Example from SplitExplicit:
```julia
# Uses ∂xTᶠᶜᶠ instead of ∂xᶠᶜᶠ - no halo access needed!
U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xTᶠᶜᶠ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1])
```

For acoustic substepping, we need:
- `∂xTᶠᶜᶜ(i, j, k, grid, ρ)` - density gradient in x at (Face, Center, Center)
- `∂yTᶜᶠᶜ(i, j, k, grid, ρ)` - density gradient in y at (Center, Face, Center)
- `∂zTᶜᶜᶠ(i, j, k, grid, ρ)` - density gradient in z at (Center, Center, Face)
- `δxTᶜᵃᵃ`, `δyTᵃᶜᵃ`, `δzTᵃᵃᶜ` - for divergence computation

**Note:** Some 3D topology-aware operators (e.g., for the vertical direction with `Bounded` z-topology)
may need to be added to Oceananigans or defined locally. The existing operators in
`topology_aware_operators.jl` focus on 2D barotropic operations; we may need to extend them.

**Savings:** Eliminates ~100+ `fill_halo_regions!` calls per full time step!

### 2. `BatchedTridiagonalSolver` for Vertical Implicit Solve

Oceananigans provides `BatchedTridiagonalSolver` which:
- Works efficiently on GPU via KernelAbstractions
- Solves independent tridiagonal systems for each (i,j) column
- Supports both 1D (constant) and 3D coefficient arrays

Interface:
```julia
solver = BatchedTridiagonalSolver(grid;
    lower_diagonal,   # 1D or 3D array
    diagonal,         # 1D or 3D array
    upper_diagonal,   # 1D or 3D array
    scratch,          # 3D working array
    tridiagonal_direction = ZDirection())

solve!(solution, solver, rhs)  # In-place solve
```

### 3. Step Density (Not Pressure or Exner Function) During Acoustics

We step density ρ during acoustic substeps and diagnose pressure from the equation of state.

**Why CM1 steps Exner function π:**
- Pairs naturally with potential temperature: pressure gradient becomes -cₚ θ ∇π
- Historical convention in NWP

**Why we step density:**
- **Consistency**: ρ is already the prognostic variable in `CompressibleDynamics`
- **No redundancy**: p is always diagnostic from EOS, never stepped directly
- **Simplicity**: One less prognostic variable to manage
- **Physical clarity**: Density represents mass, pressure is derived

**Acoustic equations:**

The full continuity equation splits into advection (slow) and compression (fast):
```math
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \mathbf{u}) = \underbrace{-\mathbf{u} \cdot \nabla \rho}_{\text{advection (slow)}} \underbrace{- \rho \nabla \cdot \mathbf{u}}_{\text{compression (fast)}}
```

During acoustic substeps, we only step the fast (compression) part:
```math
\frac{\partial \rho}{\partial t} = -\rho \nabla \cdot \mathbf{u}
```

Then pressure is diagnosed from EOS:
```math
p = \rho R^m T
```

The momentum equation uses the standard pressure gradient:
```math
\frac{\partial (\rho \mathbf{u})}{\partial t} = \ldots - \nabla p
```

### 4. Time-Averaging for Scalar Transport

Time-averaged velocities from the acoustic loop are used for advecting scalars:

```julia
# Accumulate during acoustic loop
for n = 1:nloop
    # ... acoustic updates ...

    if n < nloop
        ū .+= u
        v̄ .+= v
        w̄ .+= w
    else
        # Final step: include weighting
        ū .= (ū .+ u) ./ nloop
        v̄ .= (v̄ .+ v) ./ nloop
        w̄ .= (w̄ .+ w .* α) ./ nloop  # w uses implicit weight
    end
end
```

### 5. Divergence Damping

Include optional divergence damping to suppress spurious acoustic noise. Since we step density, the damping is applied to ρ:

```julia
ρ_new = ρ_new + kdiv * (ρ_new - ρ_old)
```

Typical values: kdiv ≈ 0.05 - 0.1

Note: This is equivalent to pressure damping since p ∝ ρ via the EOS.

### 6. Future Extensibility

The design accommodates future additions:
- **Open boundaries**: Add radiating boundary condition infrastructure
- **Terrain-following**: Modify pressure gradient and metric terms

---

## Estimated Effort

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Core Infrastructure | 2-3 days |
| 2 | Thermodynamic Coefficients | 1-2 days |
| 3 | Horizontal Momentum Update | 1-2 days |
| 4 | Vertical Implicit Solve | 3-4 days |
| 5 | Full Time Stepper Integration | 2-3 days |
| 6 | Testing and Validation | 3-4 days |
| **Total** | | **12-18 days** |

---

## References

1. **Wicker, L.J. and Skamarock, W.C. (2002)**, "Time-Splitting Methods for Elastic Models Using Forward Time Schemes", *Mon. Wea. Rev.*, 130, 2088-2097.

2. **Klemp, J.B., Skamarock, W.C., and Dudhia, J. (2007)**, "Conservative Split-Explicit Time Integration Methods for the Compressible Nonhydrostatic Equations", *Mon. Wea. Rev.*, 135, 2897-2913.

3. **Bryan, G.H. and Fritsch, J.M. (2002)**, "A Benchmark Simulation for Moist Nonhydrostatic Numerical Models", *Mon. Wea. Rev.*, 130, 2917-2928.

4. **CM1 Numerical Model**, https://www2.mmm.ucar.edu/people/bryan/cm1/
   - `sound.F`: Acoustic substepping implementation
   - `solve2.F`: RK loop structure

5. **Oceananigans.jl Documentation**, https://clima.github.io/OceananigansDocumentation/stable/
   - `BatchedTridiagonalSolver`: GPU-compatible tridiagonal solver

---

## Appendix: CM1 Acoustic Substep Count

From CM1's `sound.F`, the number of acoustic substeps per RK stage:

```fortran
IF( nrkmax.eq.3 )THEN
  if(nrk.eq.1)then
    nloop=nint(float(nsound)/3.0)
    dts=dt/(nloop*3.0)
    if( dts.gt.(dt/nsound) )then
      nloop=nloop+1
      dts=dt/(nloop*3.0)
    endif
  elseif(nrk.eq.2)then
    nloop=0.5*nsound
    dts=dt/nsound
  elseif(nrk.eq.3)then
    nloop=nsound
    dts=dt/nsound
  endif
ENDIF
```

This ensures the acoustic CFL is satisfied while minimizing the number of substeps.
