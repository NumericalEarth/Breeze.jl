# Acoustic substepping code review against MPAS

Review of uncommitted changes in `src/CompressibleEquations/acoustic_substepping.jl`
and `src/CompressibleEquations/compressible_density_tendency.jl` against the MPAS
source code (`mpas_atm_time_integration.F`) and tech note equations.

## Critical issues

### 1. Missing buoyancy in slow w tendency

**Status: BROKEN — model has no vertical PGF or gravity wave forcing.**

The `compressible_density_tendency.jl` change zeroes the explicit vertical PGF and
buoyancy for `SplitExplicitTimeDiscretization`:
```julia
@inline AtmosphereModels.explicit_z_pressure_gradient(i, j, k, grid,
        d::CompressibleDynamics{<:SplitExplicitTimeDiscretization}) = zero(grid)
@inline AtmosphereModels.explicit_buoyancy_forceᶜᶜᶠ(i, j, k, grid,
        d::CompressibleDynamics{<:SplitExplicitTimeDiscretization}, args...) = zero(grid)
```

The comment says: *"The Exner-form PGF is added in `_convert_slow_tendencies!`
using the SAME discretization as the acoustic loop's cofwz coefficient."*

But in `_convert_slow_tendencies!`, the entire buoyancy computation was **deleted**:
```julia
# BEFORE (removed):
b = -cᵖᵈ * θᵥᶠ * δz_πᵣ / Δzᶠ - g
Gˢw[i, j, k] = (Gˢρw[i, j, k] / ρᶜᶜᶠ + b) * (k > 1)

# AFTER:
Gˢw[i, j, k] = (Gˢρw[i, j, k] / ρᶜᶜᶠ) * (k > 1)
```

So `Gˢw` has NO vertical PGF and NO buoyancy. The model cannot support gravity waves
or maintain hydrostatic balance.

**MPAS reference** (line 5905-5907):
```fortran
tend_w_euler(k) = tend_w_euler(k) - cqw(k) * (
    rdzu(k) * (pp(k) - pp(k-1))                    ! perturbation Exner PGF
  - (fzm(k)*dpdz(k) + fzp(k)*dpdz(k-1)))          ! buoyancy
```

where:
- `pp` = perturbation pressure = `zz * Rd * (exner * rtheta_p + rtheta_base * (exner - exner_base))`
- `dpdz` = buoyancy = `-g * (rb * qtot + rr_save * (1 + qtot))`

The slow w tendency MUST include both the perturbation Exner PGF and the buoyancy.
These should be computed in the same Exner-function form as `cofwz` in the tridiagonal
to avoid discrete inconsistency between the slow tendency and the implicit solver.

**Fix**: Add the Exner-form PGF + buoyancy back to `_convert_slow_tendencies!`:
```julia
# MPAS tend_w_euler: Exner-form PGF + buoyancy
# PGF: -c2 * Π_face * ∂(ρθ_p)/∂z * cqw    [matches cofwz discretization]
# Buoyancy: g * (θᵥ - θ_base) / θ_base     [or equivalently from dpdz]
```

### 2. Divergence damping formula is wrong

**Status: Three errors in the coefficient.**

The implementation:
```julia
ru_p[i, j, k] += smdiv * Δx_u * (δΘ_i - δΘ_im1) / (Δx_u * θ_u_safe) * ...
```

simplifies to `smdiv * (δΘ_i - δΘ_im1) / θ_u` — the `Δx` cancels.

**MPAS reference** (lines 3044-3057):
```fortran
coef_divdamp = 2.0 * smdiv * config_len_disp * rdts
divCell1 = -(rtheta_pp(k,cell1) - rtheta_pp_old(k,cell1))
divCell2 = -(rtheta_pp(k,cell2) - rtheta_pp_old(k,cell2))
ru_p(k,iEdge) = ru_p(k,iEdge) + coef_divdamp * (divCell2 - divCell1) &
                / (theta_m(k,cell1) + theta_m(k,cell2))
```

**Error 1: Missing scale factor.** MPAS coefficient is `2 × smdiv × Δx × (1/Δτ)`.
The Breeze code has just `smdiv`. Missing `2 Δx / Δτ`.

**Error 2: Wrong sign.** MPAS uses `divCell = -(Θ_new - Θ_old)` (negative of the
change), giving damping proportional to `-δ_τΘ''`. Breeze uses `+(Θ_new - Θ_old)`.
The sign must be negative to provide damping (not amplification).

**Error 3: Denominator off by factor 2.** MPAS divides by `(θ₁ + θ₂)` (sum).
Breeze computes `θ_face = (θ₁ + θ₂)/2` (average) then divides by that, so the
effective denominator is `(θ₁ + θ₂)/2`, which is 2× too small.

**Fix**: The correct discrete formula is:
```julia
coef = 2 * smdiv * Δx / Δτ
δΘ_i = -(rtheta_pp_new[i,j,k] - rtheta_pp_old[i,j,k])    # negative sign!
δΘ_im1 = -(rtheta_pp_new[i-1,j,k] - rtheta_pp_old[i-1,j,k])
θ_sum = θ_m[i,j,k] + θ_m[i-1,j,k]                        # sum, not average
ru_p[i,j,k] += coef * (δΘ_i - δΘ_im1) / θ_sum
```

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
| Horizontal flux in ts/rs | ✓ implemented | Needs Δτ verification |
| Slow w tendency (PGF+buoyancy) | **✗ MISSING** | No — must add back |
| Divergence damping coefficient | **✗ WRONG** | No — 3 errors |
| Tridiagonal solve | ✓ unchanged | Previously verified |
| Per-stage reset | ✓ works | Yes |
| Stage-frozen Exner | ✓ works | Yes |
