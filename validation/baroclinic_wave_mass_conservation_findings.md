# Mass conservation of the compressible baroclinic wave on a LatitudeLongitudeGrid

**Date:** 2026-07-03.
**Question:** is the DCMIP2016 baroclinic wave (`examples/baroclinic_wave_run.jl` configuration:
`CompressibleDynamics(SplitExplicitTimeDiscretization())`, 1° lat-lon ±75°, 32 levels, WENO5,
inviscid) mass conserving? If not, where in the domain is mass gained/lost?

**Answer: no — total mass GROWS**, at ≈2×10⁻⁴ (relative) per day at Δt = 12 min, decelerating
as the initial state adjusts. All measurements below are Float64 sums of ρ·V with exact
spherical-shell cell volumes (cross-checked against cell-centre-cosine weights — identical, so
this is creation, not redistribution).

## Established facts

1. **Magnitude/time behaviour** (`baroclinic_wave_mass_conservation.jl`, 4 days):
   ΔM/M₀ = +8.39×10⁻⁴ over 4 days; daily increments decay 3.4 → 1.2×10⁻⁴/day —
   adjustment-driven, not a steady leak. M₀ = 4.987×10¹⁸ kg (physically correct for the shell).
2. **Deterministic**: F32 and F64 give the same drift to 6 significant figures. Not roundoff.
3. **Independent of advection scheme and of the wave** (`baroclinic_wave_mass_inspect.jl`):
   WENO5 vs Centered(4), with or without the DCMIP perturbation — per-step mass creation
   identical to ~5 digits (+3.50×10⁻⁶ on step 1). The *balanced zonal state alone* produces it,
   from the very first step, zonally uniformly (deviation from zonal mean = 0.000 in the
   unperturbed run).
4. **Where in the domain**: fractional gain is largest near the **south wall** (rows 65–75°S)
   and decays smoothly northward — a ~100:1 south/north band asymmetry over 4 days despite the
   hemispherically symmetric basic state; vertically bottom-heavy ∝ ρ̄ (a roughly
   height-uniform *fractional* source, hotspot at the lowest level, ≈66°S).
5. **The walls are exactly closed** (`baroclinic_wave_mass_wall_flux.jl`): ρv ≡ 0 (to the last
   bit) on both ±75° faces and ρw ≡ 0 on z = 0, H at every checked step, while mass still grows
   +3.5×10⁻⁶/step. With closed boundaries any telescoping flux-form update conserves ΣρV to
   roundoff ⇒ the source is **interior, in the density update of the dynamical core**, not a
   boundary flux and not the advection operator.
6. **Scaling** (`mass_dt_probe.jl`, same simulated hour):
   - per-step creation ∝ **Δt²** (Δt 720→360 s: 3.50×10⁻⁶ → 0.898×10⁻⁶ per step, ratio 3.9),
     i.e. the drift per simulated time is **first order in Δt**;
   - essentially independent of the acoustic substep count (acoustic_cfl 0.5 → 0.25: −3%);
   - independent of the Crank–Nicolson forward weight (ω = 0.65 → 1.0: unchanged).

## Root cause (resolved by stage instrumentation, 2026-07-03)

Stage/phase instrumentation (`mass_stage_instrumentation.jl`) and the explicit-time-stepping
control (`mass_explicit_probe.jl`) close the case:

7. **The explicit path conserves exactly**: `CompressibleDynamics(ExplicitTimeStepping())`
   (SSP-RK3, Δt = 1 s, 60 steps) holds ΣρV to ±1×10⁻¹⁴, zero-mean — pure roundoff.
8. **∫Gρ dV = 0 to ~10⁻¹⁸** under both volume weightings at every RK stage — the flux-form
   slow density tendency telescopes perfectly; advection divergence and grid metrics are clean.
9. **All creation happens inside `acoustic_rk3_substep_loop!`** — every other phase
   (`prepare_acoustic_cache!`, scalar substeps, `update_state!`) leaves Σρ unchanged. Per-stage
   gains (Δt = 720 s): +4.30×10⁻⁷, +4.92×10⁻⁷, +2.58×10⁻⁶ (sum = the observed
   +3.50×10⁻⁶/step), and each stage's gain equals Σρ′ at stage end.
10. **The smoking gun — a spurious wall-face tendency, integrated by the substep loop.** The
    slow y-momentum tendency `Gⁿρv` is **nonzero on the south wall v-face** (max ≈ 5.3×10⁻⁴,
    which is ρ·|f·u| at 74.5°S — the Coriolis force) and **exactly zero on the north wall
    face**: the tendency kernel's `:xyz` launch covers j = 1…Nφ, which *includes* the south
    wall face (j=1) but *never reaches* the north one (j=Nφ+1). An impenetrable wall should
    have zero tendency on both. Inside the substep loop, `_explicit_horizontal_step!`
    integrates it: post-loop wall `ρv′` = 0.128 / 0.192 / 0.385 kg m⁻² s⁻¹ in stages 1/2/3 —
    **exactly Nτ·Δτ·Gⁿρv(wall)** (6/9/18 substeps × 40 s × 5.3×10⁻⁴). The predictor's
    horizontal divergence then pumps mass through the south wall face every substep; the
    linearly-growing flux time-integrates to ∝ Gv·(βΔt)², reproducing the per-stage gains and
    the measured Δt² scaling. Stage-end recovery + BC application zero the *prognostic* wall
    ρv, hiding the evidence (which is why the closed-wall check on the final state passed).

**Why explicit conserves and split-explicit doesn't**: the explicit path commits the same
wall-face tendency, but its wall ρv is zeroed by the no-penetration BC in `update_state!`
*before* the next divergence is evaluated. The substep loop takes divergences of ρv′
internally, and the perturbation fields carry only zero-*gradient* halos — never a
zero-*value* wall condition. (`apply_open_boundary_relaxation!` acknowledges the
per-substep-BC principle for *open* boundaries; the impermeable-wall case was missed.)

**Fix (implemented 2026-07-03, option b):** `enforce_wall_impenetrability!` in
`src/CompressibleEquations/acoustic_substepping.jl` zeroes the wall-normal momentum
perturbations (ρu′ on Bounded-x faces i=1, Nx+1; ρv′ on Bounded-y faces j=1, Ny+1; open-BC
sides excluded) twice per acoustic substep — after the explicit horizontal step (before the
predictor's divergence) and after the divergence damping — the WRF/MPAS per-substep boundary
enforcement convention, mirroring the existing `apply_open_boundary_relaxation!`.

**Fix verified:** per-step creation +3.50×10⁻⁶ → **+5.6×10⁻¹⁵**; the 4-day drift
+8.39×10⁻⁴ → **+4.2×10⁻¹⁵** (≈1×10⁻¹⁵/day, non-accumulating, matching the explicit control) —
eleven orders of magnitude. Wall-face ρv′ ≡ 0 through every stage.

The south-heavy/bottom-heavy spatial pattern is thus fully explained: the source is the
south-wall row (j=1 launch-bounds asymmetry), ∝ ρ̄·f·u at the wall (largest at low levels),
spreading northward by the adjustment circulation.

Practical consequences: ≈+2×10⁻⁴/day at Δt = 720 s (∝ Δt² per step); +0.4–0.6% total mass by
day 30 (surface-pressure-equivalent ≈ +4–6 hPa). Until fixed, budget-based diagnostics on
split-explicit lat-lon runs carry this bias; `ExplicitTimeStepping` is a mass-conserving (if
expensive) alternative.

## Scripts

| script | what it does |
|--------|--------------|
| `baroclinic_wave_mass_conservation.jl` | 4-day drift + per-row/per-level localization |
| `baroclinic_wave_mass_inspect.jl` | A/B/C: perturbation on/off, WENO vs Centered; per-step increments; 2-D source maps; zonal-uniformity metric |
| `baroclinic_wave_mass_wall_flux.jl` | closed-wall check: wall-face ρv/ρw and integrated wall fluxes vs per-step ΔM |
| `mass_dt_probe.jl` (+ `mass_probe_common.jl`) | Δt, acoustic-substep, and CN-weight scaling of the source |
| `mass_stage_instrumentation.jl` | replicates the acoustic-RK3 `time_step!` phase by phase: Σρ/Σρᴸ/Σρ′, ∫Gρ dV, and wall-face Gⁿρv & ρv′ probes (the root-cause finder) |
| `mass_explicit_probe.jl` | `ExplicitTimeStepping` control — conserves to ±1e-14 (roundoff) |

All run with `julia --project=examples validation/<script>.jl` on the GPU; diagnostics are
computed in Float64 on the CPU regardless of the model float type.
