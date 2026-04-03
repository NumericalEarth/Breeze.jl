# Acoustic substepping: discrepancy tracker

Systematic catalogue of all differences between Breeze's implementation and
MPAS source code. Updated 2026-04-04 after Phase 1 fixes.

## Fixed

### #1 — w recovery missing ρw⁰ ✓
**Lines**: 1318-1324, 1767-1769
**Was**: `w = rw_p / ρ_recovered` (perturbation only, loses initial ρw)
**Now**: `w = (ρw⁰ + rw_p) / ρ_recovered` using U⁰[4]
**MPAS ref**: line 3331-3334, `w = rw / rho_edge`

### #2 — fzm/fzp interpolation weights swapped ✓
**Lines**: 457-458, 704-705
**Was**: `fzm = Δz_above / total` (wrong for MPAS convention)
**Now**: `fzm = Δz_below / total` (matches MPAS `fzm(k) = dzw(k-1)/(dzw(k-1)+dzw(k))`)
**Effect on uniform grid**: None (both = 0.5)

### #6 — Horizontal PGF incorrectly zeroed for SplitExplicit ✓
**File**: `compressible_density_tendency.jl` lines 18-27
**Was**: `x_pressure_gradient` and `y_pressure_gradient` returned zero for SplitExplicit
**Now**: Zeroing removed; horizontal PGF flows through from dynamics kernel using
current-state pressure. Matches MPAS which recomputes pp (horizontal PGF) at every stage.
Only VERTICAL PGF+buoyancy are zeroed (added back via linearized pp in `_convert_slow_tendencies!`).

### #11 — Missing halo fills between substeps ✓
**Lines**: 1740-1745
**Added**: `fill_halo_regions!` for rtheta_pp, rho_pp, ru_p, rv_p after each substep.
**MPAS ref**: `exchange_halo_group` calls at lines 1279-1322.

## Outstanding numerical discrepancies

### #3 — epssm = 0.2 vs MPAS default 0.1
**Line**: 1686
**Code**: `epssm = FT(2 * ω - 1)` where `ω = forward_weight = 0.6`
→ `epssm = 0.2`
**MPAS**: `config_epssm = 0.1`
**Effect**: Stronger off-centering → more implicit damping of vertical acoustic modes.
`dtseps = 0.5 * Δτ * 1.2` (Breeze) vs `0.5 * Δτ * 1.1` (MPAS).
`resm = 0.667` (Breeze) vs `0.818` (MPAS).
All tridiagonal coefficients scale with dtseps, so the implicit solve is ~9% stronger.
**Fix**: Set `forward_weight = 0.55` (gives epssm=0.1) or pass epssm directly.

### #15 — len_disp uses minimum grid spacing instead of nominal resolution
**Lines**: 1734-1736
**Code**: `len_disp = min(minimum_xspacing(grid), minimum_yspacing(grid))`
**MPAS**: `config_len_disp` = nominal grid resolution (e.g., 60000 for 60km mesh)
**Effect on LatitudeLongitudeGrid**: `minimum_xspacing` is at the poles (very small
after polar filter), making `coef_div_damp = 2 * smdiv * len_disp / Δτ` much smaller
than MPAS intends. Divergence damping is too weak near the equator and about right
near the poles — opposite of the intended uniform damping rate.
**Fix**: Pass nominal resolution as a parameter (e.g., from `SplitExplicitTimeDiscretization`).

### #4 — Per-stage reset vs MPAS cross-stage accumulation
**Lines**: 1648-1656
**Code**: All perturbation variables (rho_pp, rtheta_pp, rw_p, ru_p, rv_p) zeroed
at the start of EVERY RK stage.
**MPAS**: Perturbations reset only at stage 1 (`small_step == 1` at line 2850).
At stages 2-3, perturbations carry over from the previous stage.
**Effect**: The WS-RK3 scheme advances `φ* = φⁿ + (Δt/3)·R`, `φ** = φⁿ + (Δt/2)·R`,
`φⁿ⁺¹ = φⁿ + Δt·R`. Each stage starts from φⁿ (not from the previous stage).
Per-stage reset is consistent with this — the perturbations represent the change
from φⁿ within each stage. MPAS's cross-stage accumulation may be an optimization
that exploits continuity between stages. Breeze's per-stage reset is safe but
less efficient (recomputes from scratch each stage).
**Status**: Design choice, not a bug. But worth testing both approaches.

### #10 — Time-averaged horizontal velocities (uAvg/vAvg) not accumulated
**Lines**: 1642, missing accumulation in substep loop
**Code**: `ū.u` and `ū.v` are zeroed but never accumulated during substeps.
Only `ū.w` (wwAvg) is accumulated (pre-solve and post-solve halves).
**MPAS**: `ruAvg` accumulated each substep (line 2786), recovered as
`ruAvg = ru_save + ruAvg/ns` (line 3368).
**Effect**: Time-averaged horizontal velocities unavailable for scalar transport.
For the baroclinic wave test (no scalars), this doesn't matter. For moist
simulations with scalar transport, this is required.
**Status**: Not yet needed for current tests but required for production.

## Code quality issues (not affecting correctness)

### #5 — `if k > 1` in kernel instead of `ifelse`
**Line**: 688 in `_convert_slow_tendencies!`
**Rule**: AGENTS.md requires `ifelse` for GPU compatibility.
**Risk**: May fail on GPU. Works on CPU.

### #7 — Unused kernel arguments
- `_mpas_acoustic_substep!` (line 1106): `ρu⁰, ρv⁰, ρ⁰` never read
- `_mpas_recovery_wsrk3!` (line 1300): `θᵥ, Gˢρχ, Gˢρ, Δt_stage` never read
**Risk**: Wasted memory transfer to GPU, confusing code.

### #8 — alpha_tri/gamma_tri struct fields unused
**Lines**: 506-509
**Code**: Initialized to zero during coefficient computation. The substep kernel
uses `gamma_scratch` (aliased to a different field) instead.
**Risk**: Wasted GPU memory.

### #12 — Inconsistent interpolation operators
**Code**: `_convert_rw_p_to_w!` uses topology-aware `ℑzTᵃᵃᶠ` (line 1316).
`_reset_velocities_to_U0!` uses standard `ℑzᵃᵃᶠ` (line 1793).
**Risk**: Different behavior at boundaries for non-periodic z topology.

### #13 — No division-by-zero guards in `_reset_velocities_to_U0!`
**Lines**: 1787-1795
**Risk**: NaN if ρ = 0 at a face. Unlikely in practice.

## Not applicable / correct as-is

### #9 — Rayleigh damping (Step 8 in algorithm doc)
Intentionally skipped. Not needed for tests without absorbing layer.

### #14 — coftz at Nz+1 boundary
Set to zero (line 477). Correct for impenetrable top BC since rw_p[Nz+1] = 0.

### Tridiagonal b_k uses coftz_k twice
Verified against MPAS source (line 2340-2342): MPAS also uses `coftz(k)` in both
positions of the cofwz contribution to b_k. NOT a bug.

## Phase 2 analysis (2026-04-04)

### N=2 works at Δt=600s, N=6 crashes at all Δt

The bug amplifies per substep. Verified that the following are NOT the cause:
- Off-centering sign is correct (forward-biased = damping) ✓
- dtseps scales correctly with Δτ (smaller Δτ → smaller corrections) ✓
- Divergence damping coefficient scales correctly (coef ∝ 1/Δτ) ✓
- No scratch field aliasing between ts_scratch, rs_scratch, rtheta_pp_old ✓
- Total accumulated ru_p over the full stage is the same for N=2 and N=6 ✓
- Tridiagonal is rebuilt with correct dtseps each substep ✓

### Outstanding discrepancies that could affect N=6 stability

**#3 (epssm=0.2)**: Stronger off-centering should increase damping. But the
interaction between stronger implicit weighting and the acoustic coupling
through rtheta_pp might behave non-monotonically. Worth testing with epssm=0.1.

**#15 (len_disp=minimum_xspacing)**: If the grid has non-uniform spacing,
the divergence damping coefficient varies across the domain. For a
LatitudeLongitudeGrid with polar filter at 60°, the minimum spacing is at
the poles. With `len_disp = min(Δx_min, Δy_min)`, the damping is calibrated
for the smallest cells. At lower latitudes where cells are larger, the
damping is too WEAK relative to MPAS (which uses the nominal resolution).
Insufficient damping at lower latitudes could allow acoustic modes to grow.

### Recommendation
Test with epssm=0.1 and len_disp = nominal grid resolution (not minimum)
to eliminate discrepancies #3 and #15 before further debugging.
