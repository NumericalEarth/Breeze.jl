# Acoustic substepping code review against MPAS

Review of uncommitted changes in `src/CompressibleEquations/acoustic_substepping.jl`
and `src/CompressibleEquations/compressible_density_tendency.jl` against the MPAS
source code (`mpas_atm_time_integration.F`) and tech note equations.

## Critical issues

### 1. ~~Missing buoyancy in slow w tendency~~ RESOLVED

**Status: Previously flagged, now resolved.**

The `SplitExplicitTimeDiscretization` zeroing dispatches were removed from
`compressible_density_tendency.jl`. The full PGF+buoyancy now flows through the
normal model tendency path (`z_momentum_tendency` → `Gˢρw`) and is converted to
velocity form: `Gˢw = Gˢρw / ρ`.

Note: MPAS computes the slow w tendency in Exner form (line 5905-5907) to ensure
discrete consistency with cofwz in the tridiagonal. The current Breeze approach
uses the conservation-form PGF+buoyancy from the model tendency instead. This may
introduce a discrete mismatch between the slow tendency and the implicit solver.
Monitor for this — if acoustic noise appears, the fix is to compute `tend_w_euler`
in Exner form matching the cofwz discretization.

### 2. ~~Divergence damping formula wrong~~ FIXED

**Status: All three errors corrected in latest revision.**

The kernel now takes a precomputed `coef_div_damp = 2 * smdiv * len_disp / Δτ`,
uses `divΘ = -(rtheta_pp_new - rtheta_pp_old)` (correct negative sign), and
divides by `(θ₁ + θ₂)` (sum, not average). Matches MPAS lines 3044-3057.

## Potential issues (need verification)

### 3. Horizontal flux divergence may have extra Δτ factor

The code computes:
```julia
rs_k = -dts * mass_flux_div
ts_k = -dts * theta_flux_div
```

where `mass_flux_div` uses `ru_p` which already accumulates `ρ × Δτ × (tend + pgf)`
per substep. In MPAS, the flux is (line 2874):
```fortran
flux = sign * dts * dvEdge * ru_p * invArea
```

MPAS's `ru_p` accumulates `dts * tend_ru` per substep, so it has units of
`[momentum_density × time]`. The flux formula then multiplies by another `dts`,
which seems like double-counting. But MPAS works — so either:
(a) the units are different from what I think, or
(b) MPAS's `dts` in the flux line serves a different purpose (converting from
    edge-normal flux to cell-averaged tendency).

**Recommendation**: Verify by running a simple test case and comparing `rs`, `ts`
values between MPAS and Breeze at corresponding grid points.

### 4. ruAvg vs velocity averaging

MPAS accumulates `ruAvg` from `ru_p` (momentum perturbation) and recovers:
```fortran
ruAvg = ru_save + ruAvg / ns
```

Breeze accumulates time-averaged velocities (`ū`) directly from `u, v, w`. These
should be equivalent if the velocity and momentum are consistently related, but the
averaging method should be verified.

## Correct aspects

### Using full stage-frozen Exner for horizontal PGF ✓

The horizontal forward step now computes Π from the EOS pressure:
```julia
Π_i = (pressure[i, j, k] / pˢᵗ)^κ
```

This matches MPAS Eq. 3.25: the stage-frozen FULL Exner `π^t`, not the reference `π₀`.

### Per-stage reset of perturbation variables ✓

All `_pp` and `_p` variables zeroed at stage start. Matches MPAS (line 2850-2860).

### ru_p accumulation pattern ✓

Velocity increment converted to momentum: `ru_p += ρ_face × du`. This is consistent
with MPAS's `ru_p += dts * (tend_ru - pgrad)` since `ρ × du = ρ × Δτ × (Gˢu + pgf)`
= `Δτ × Gˢρu + Δτ × ρ × pgf`.

### Horizontal flux divergence uses area-weighted form ✓

```julia
mass_flux_div = (Ax_e * ru_p[i+1,j,k] - Ax_w * ru_p[i,j,k] + ...) / V
```

Handles LatitudeLongitudeGrid automatically through metric operators.

### θ flux uses face-interpolated θ_m ✓

```julia
θ_e = ℑxᶠᵃᵃ(i + 1, j, k, grid, θ_m)
theta_flux_div = (Ax_e * ru_p[i+1,j,k] * θ_e - ...) / V
```

Matches MPAS's `flux * 0.5*(theta_m(cell1) + theta_m(cell2))`.

## Summary

| Component | Status | MPAS match? |
|-----------|--------|-------------|
| Horizontal forward step | ✓ works | Yes (Eq. 3.25) |
| ru_p/rv_p tracking | ✓ works | Yes |
| Horizontal flux in ts/rs | ✓ implemented | Needs Δτ factor verification |
| Slow w tendency (PGF+buoyancy) | ✓ via Gˢρw | Yes (conservation form, not Exner form) |
| Divergence damping coefficient | ✓ fixed | Yes (coef, sign, denominator all correct) |
| Tridiagonal solve | ✓ unchanged | Previously verified |
| Per-stage reset | ✓ works | Yes |
| Stage-frozen Exner | ✓ works | Yes |

### Remaining concern: discrete consistency of slow PGF

MPAS computes `tend_w_euler` in the SAME Exner form as the cofwz coefficient:
```
tend_w = -c2 * Π * ∂(ρθ_p)/∂z + buoyancy
```

Breeze computes the slow w tendency from the conservation-form model tendency:
```
Gˢw = (1/ρ) * [-∂(p-p_ref)/∂z + g(ρ-ρ_ref) + advection + Coriolis + ...]
```

The conservation-form PGF `∂p/∂z` and the Exner-form PGF `c2 Π ∂(ρθ)/∂z` are
analytically equivalent but discretely different (different stencil/interpolation).
If the slow tendency's PGF doesn't match the acoustic solver's cofwz, the mismatch
creates a spurious source/sink at each substep. This may appear as acoustic noise
proportional to Δτ. Watch for it in test results.
