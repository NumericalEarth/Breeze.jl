# VITS debugging plan

The `VerticallyImplicitTimeStepping` solver blows up immediately: max|v| = 27 m/s
after 200 steps (2 hours) with Δt=40s, growing to 300+ m/s by day 2.5, then NaN.
For comparison, `ExplicitTimeStepping` at Δt=2s gives max|v| = 0.4 m/s at 2 hours.

## Symptoms

- max|v| jumps to 27 m/s within the first few hundred steps (should be ~0)
- max|u| oscillates wildly (27 → 43 → 366 m/s)
- max|w| stays modest (0.3 m/s initially) — the w correction might be OK
- NaN in ρ field at day 2.8

## Architecture

The VITS solver (`vertical_implicit_solver.jl`) does the following each RK stage:

1. **Tendency corrections** (in `dynamics_kernel_functions.jl`):
   - `vertical_acoustic_correction_ρw`: subtracts `(ℂ²/θ) ∂(ρθ)/∂z + ρg` from ρw tendency
   - `vertical_acoustic_correction_ρθ`: subtracts `∂(θ w ρθ)/∂z` from ρθ tendency

2. **Helmholtz solve** (in `_build_ρθ_tridiagonal!`):
   - `[I - (αΔt)² ∂z(ℂ² ∂z)] δρθ = -αΔt ∂z(θ ρw*)`
   - Tridiagonal solve for δρθ

3. **Back-solve** (in `_back_solve_ρw!`):
   - `(ρw)⁺ = (ρw)* - αΔt (ℂ²/θ) ∂(δρθ)/∂z`

## Debugging steps

### Step 1: Δt sensitivity — is it CFL or implementation?

Run VITS with Δt=2s (same as explicit). If it still blows up, the bug is in the
implicit solver itself, not the CFL. If it's stable at Δt=2s but not at Δt=40s,
the solver has a conditional stability limit.

```julia
dynamics = CompressibleDynamics(VerticallyImplicitTimeStepping(); surface_pressure=p₀)
simulation = Simulation(model; Δt=2, stop_iteration=1000)
```

### Step 2: Isolate tendency corrections vs implicit solve

The VITS adds two corrections to the explicit tendencies via
`vertical_acoustic_correction_ρw` and `vertical_acoustic_correction_ρθ`.
These subtract the vertical fast terms so the explicit step doesn't see them.

**Test A**: Run with only the tendency corrections (no Helmholtz solve).
Comment out `vertical_acoustic_implicit_step!` in `ssp_runge_kutta_3.jl`.
This should blow up (CFL violation from missing implicit solve), but if it
blows up *differently* (e.g., much slower), then the Helmholtz solve is
actively injecting energy.

**Test B**: Run with the Helmholtz solve but without tendency corrections.
Comment out the `vertical_acoustic_correction_*` terms in
`dynamics_kernel_functions.jl`. This double-counts the vertical terms
(explicit + implicit), but if it's stable, the corrections are the problem.

### Step 3: Check the tendency corrections

The `_vac_ρw` function subtracts the vertical PGF + buoyancy:

```julia
return (ℂ²ᶠ / θᶠ * δz_ρθ / Δzᶠ + ρᶠ * g) * (k > 1)
```

This should exactly cancel the explicit `z_pressure_gradient + buoyancy_force`
terms. Verify by computing both at initialization (before any time stepping)
and checking they sum to ~zero.

Key question: does `ℂ² / θ * ∂(ρθ)/∂z` equal `∂p/∂z`? This relies on:
```
p = ρ Rᵐ T = ρθ (p/p₀)^κ Rᵐ = ρθ · (function of ρθ)
∂p/∂z ≈ (∂p/∂ρθ) · ∂(ρθ)/∂z = ℂ²/θ · ∂(ρθ)/∂z
```

Check: is the linearization `∂p/∂(ρθ) = ℂ²/θ` correct? Compute both sides
numerically at each grid point and compare.

### Step 4: Check the Helmholtz operator sign

The tridiagonal system is:

```
[I - (αΔt)² ∂z(ℂ² ∂z)] δρθ = RHS
```

The operator `-∂z(ℂ² ∂z)` is **positive definite** (it's a diffusion operator).
So `I + (αΔt)² |∂z(ℂ² ∂z)|` has all eigenvalues > 1, which is unconditionally
stable. If the sign is wrong (`I + (αΔt)² ∂z(ℂ² ∂z)`), eigenvalues can be < 1
and the solve amplifies.

In `_build_ρθ_tridiagonal!`:
```julia
lower[i, j, k] = -Q_lower
upper[i, j, k] = -Q_top
diag[i, j, k]  = 1 + Q_bot + Q_top
```

where `Q = αΔt² ℂ² / (Δzᶠ Δzᶜ) > 0`. This gives a diagonally dominant system
with positive diagonal and negative off-diagonals — correct for `I - α²∂z(ℂ²∂z)`.

**But check**: the sub-diagonal `lower[i,j,k]` is used for row k+1 in
Oceananigans' `BatchedTridiagonalSolver`. Is the convention correct?
Verify by printing the tridiagonal system for a single column and checking
that it matches the expected structure.

### Step 5: Check the RHS

```julia
rhs_field[i, j, k] = -αΔt / Δzᶜ * (θᶠ_top * ρw_top - θᶠ_bot * ρw_bot)
```

This is `-αΔt ∂z(θ ρw*)` at cell centers. For a balanced state with ρw* = 0,
the RHS should be zero and δρθ = 0 (no correction). Verify at initialization.

### Step 6: Check the back-solve

```julia
ρw[i, j, k] = (ρw[i, j, k] - αΔt * ℂ²ᶠ / θᶠ * δz_δρθ) * (k > 1)
```

This updates ρw using the change in ρθ. If δρθ is correct and small, this
should be a small correction. Print δρθ and δz_δρθ after the first solve
to verify they're reasonable.

### Step 7: Verify on 1D column (RectilinearGrid)

Run the same balanced-state test on a 1D column (Nx=Ny=1) with
RectilinearGrid. This eliminates all horizontal operators, metric terms,
and Coriolis — testing only the vertical implicit solve in isolation.

```julia
grid = RectilinearGrid(CPU(); size=(1, 1, 30), extent=(1, 1, 30000))
```

Initialize with the DCMIP temperature/density profile at 45°N.
Run for 100 steps. If it blows up, the bug is in the vertical solver.
If stable, the bug is in the interaction with horizontal terms.

### Step 8: Check `solve!` convention

Oceananigans' `BatchedTridiagonalSolver` may have a specific convention
for `a` (sub-diagonal), `b` (diagonal), `c` (super-diagonal). Verify that
the mapping from `lower`, `diag`, `upper` to the solver's `a`, `b`, `c`
is correct. A swap of sub/super-diagonal would give wrong results.

Check: `solver.a`, `solver.b`, `solver.c` — which is which?

## Quick diagnostic script

```julia
# Run 1 step of VITS on CPU, print all intermediate quantities
dynamics = CompressibleDynamics(VerticallyImplicitTimeStepping(); surface_pressure=p₀)
model = AtmosphereModel(grid; dynamics, coriolis, ...)
set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

# Before first step:
println("Before step:")
println("  max|ρw| = ", maximum(abs, interior(model.momentum.ρw)))
println("  max|ρθ| = ", maximum(abs, interior(model.formulation.potential_temperature_density)))

# After first step:
time_step!(model, 40)

println("After step:")
println("  max|ρw| = ", maximum(abs, interior(model.momentum.ρw)))
println("  max|ρθ| = ", maximum(abs, interior(model.formulation.potential_temperature_density)))
println("  max|v|  = ", maximum(abs, interior(model.velocities.v)))

# Check δρθ magnitude
sc = model.dynamics.vertical_acoustic_solver
δρθ = interior(model.formulation.potential_temperature_density) .- interior(sc.ρθ_scratch)
println("  max|δρθ| = ", maximum(abs, δρθ))
```

## Priority order

1. Step 1 (Δt sensitivity) — 5 minutes, immediately diagnostic
2. Step 7 (1D column) — 10 minutes, isolates vertical solver
3. Step 3 (tendency correction check) — 15 minutes, checks physics
4. Steps 4-6 (operator details) — 30 minutes, checks numerics

## Key finding: interpolation mismatch

The ρθ tendency correction (`_vac_ρθ`) subtracts the vertical flux using
**centered** interpolation (ℑzᵃᵃᶠ):

```julia
(ρθᶠ_top * w_top - ρθᶠ_bot * w_bot) / Δzᶜ
```

But the actual advection operator `div_ρUc` uses **WENO** interpolation for the
same flux. The difference (WENO − centered) is nonzero and gets passed to
the Helmholtz solver as if it were the full vertical flux. This means:

1. The explicit tendency has the WENO vertical flux subtracted AND the
   centered vertical flux re-added (net: WENO − centered = nonzero residual)
2. The Helmholtz solver then "corrects" for a flux that was never fully removed
3. The mismatch acts as a spurious forcing that grows

### Test: does the mismatch explain the blow-up?

Replace WENO with Centered(order=2) in the model to eliminate the
interpolation mismatch. If VITS is then stable, the mismatch is the cause.

```julia
model = AtmosphereModel(grid; dynamics, coriolis, advection=Centered(order=2))
```

### Fix

The correction should use the SAME interpolation as the advection operator.
Options:
1. Use the actual advective flux kernel for the correction (complex)
2. Only subtract the centered part and solve for the centered acoustic mode
   (simpler, but the residual WENO−centered part remains in the explicit step)
3. Don't subtract the vertical advection from ρθ at all — only the vertical
   PGF+buoyancy from ρw. This is what some models do (eg MPAS subtracts
   only the acoustic pressure-velocity coupling, not the advection)
