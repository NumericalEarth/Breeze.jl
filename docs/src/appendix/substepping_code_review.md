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

## Open issue: horizontal PGF discrete mismatch

**The slow tendency `Gˢu` includes the FULL EOS horizontal PGF `∂p_EOS/∂x`.**
This is NOT zeroed for `SplitExplicitTimeDiscretization` (only vertical PGF is zeroed).

The acoustic horizontal PGF uses the Exner linearization:
```julia
pgrad_u = c2 * Π_face * ∂(rtheta_pp)/∂x
```

In MPAS, `tend_u_euler` uses the SAME linearized `pp` for horizontal PGF (line 5383):
```fortran
tend_u_euler = -cqu * (pp(cell2) - pp(cell1)) * invDcEdge / (0.5*(zz2+zz1))
```

Both slow and acoustic horizontal PGFs use the SAME linearization → no mismatch.

In Breeze: slow horizontal PGF is `∂p_EOS/∂x` (nonlinear), acoustic is
`c2*Π*∂(rtheta_pp)/∂x` (linear). The mismatch is O(|ρθ'|²/ρθ²) per substep.
For the baroclinic wave with 60K gradient: ~4% per substep, ~24% accumulated
over 6 substeps.

**Fix**: Also zero `x_pressure_gradient` and `y_pressure_gradient` for
`SplitExplicitTimeDiscretization`, and compute the linearized horizontal `pp`
gradient in `_convert_slow_tendencies!`.

## Additional minor issues

1. **`ρθ_base` derived on-the-fly**: Computed as `pᵣ/(Rᵈ*Π_base)` instead of
   stored directly. Float32 roundoff may matter. MPAS stores `rtheta_base`.

2. **Stale comment on line ~1025**: "The slow tendency Gˢw already includes the
   full PGF+gravity" — wrong now that PGF+buoyancy are zeroed and re-added in
   Exner form.

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

### CRITICAL: Discrete consistency of slow PGF with cofwz

**This is the most important remaining issue.**

MPAS is designed so the slow tendency and acoustic solver use **discretely identical**
pressure gradient formulations. Specifically:

**MPAS slow w tendency** (line 5905-5907):
```fortran
tend_w_euler(k) = -cqw(k) * (rdzu(k) * (pp(k) - pp(k-1)) - fzm(k)*dpdz(k) - fzp(k)*dpdz(k-1))
```

where `pp` is the **linearized** perturbation pressure (line 6985-6988):
```fortran
pressure_p(k) = zz * rgas * (exner(k)*rtheta_p(k) + rtheta_base(k)*(exner(k)-exner_base(k)))
```

This is NOT `p_EOS - p_base`. It is a specific linearization of the EOS that is
**discretely consistent** with cofwz's `c2 * Π * ∂(ρθ)/∂z`.

**MPAS acoustic cofwz**:
```fortran
cofwz(k) = dtseps * c2 * zz_face * rdzu * cqw * pi_face
```

The slow PGF `rdzu * (pp(k) - pp(k-1))` and the acoustic correction `cofwz * ∂(ρθ_pp)/∂z`
use the SAME discrete operator. Their sum gives the correct total pressure gradient.
The slow part captures the stage-frozen PGF; the acoustic part corrects for the
ρθ perturbation accumulated during substeps.

**Breeze current state** — slow tendency uses conservation-form PGF:
```julia
Gˢw = Gˢρw / ρ   where Gˢρw includes -∂p_EOS/∂z + g(ρ-ρ_ref)
```

The EOS PGF `∂p/∂z = ∂/∂z[p₀(Rᵈρθ/p₀)^γ]` is discretely different from the
Exner-form PGF `c2 Π ∂(ρθ)/∂z`. The difference is O(Δz²) and creates a spurious
forcing at every substep. Over N substeps per stage, this accumulates to O(N Δτ Δz²).

**The same issue applies to:**
1. **Horizontal slow PGF**: Gˢu includes `∂p/∂x` from EOS, but acoustic horizontal
   PGF uses `c2 Π ∂(rtheta_pp)/∂x`. Discrete mismatch.
2. **Buoyancy**: Gˢw includes `-g(ρ-ρ_ref)/ρ`, but cofwr uses `g/2`. Different
   discrete forms.

**The fix (matching MPAS architecture):**

Restore the `SplitExplicitTimeDiscretization` zeroing dispatches for PGF and buoyancy
in `compressible_density_tendency.jl`, then compute the Exner-form PGF+buoyancy in
`_convert_slow_tendencies!`:

```julia
# In _convert_slow_tendencies!:
# Vertical PGF in Exner form (matches cofwz discretization):
rcv = Rᵈ / (cₚ - Rᵈ)
c2 = cₚ * rcv
Π_face = ℑzᵃᵃᶠ(i, j, k, grid, Π)  # stage-frozen full Exner
ρθ_p = model_ρθ - ρθ_base  # perturbation ρθ from reference
∂z_ρθ_p = (ρθ_p[i,j,k] - ρθ_p[i,j,k-1]) / Δzᶠ
pgf_exner = -c2 * Π_face * ∂z_ρθ_p

# Buoyancy (MPAS dpdz form):
# dpdz = -g * (ρ_base * qtot + ρ_p * (1 + qtot))
# For dry: dpdz = -g * ρ_p = -g * (ρ - ρ_base)
buoyancy = g * (ρ - ρ_base) / ρ  # velocity form

Gˢw[i,j,k] = (Gˢρw[i,j,k]/ρ + pgf_exner/ρ_face + buoyancy) * (k > 1)
```

Wait — actually the momentum tendency Gˢρw already includes advection + Coriolis
but NOT PGF+buoyancy (if zeroed by dispatch). So:

```julia
Gˢw = (Gˢρw/ρ) + exner_PGF/ρ + buoyancy   [PGF+buoyancy added in Exner form]
```

This ensures the slow PGF matches cofwz's discretization exactly.
