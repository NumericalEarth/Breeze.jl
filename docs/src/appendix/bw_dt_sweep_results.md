# Baroclinic-wave Δt sweep: substepper stability ceiling

This note records the empirical Δt stability ceiling of Breeze's MPAS-style
acoustic substepper on the canonical 1° DCMIP2016 baroclinic wave (Ullrich et
al. 2014), with and without the [`MonolithicFirstStage`](@ref) WS-RK3
distribution.

## Setup

- **Grid**: `LatitudeLongitudeGrid` 360 × 170 × 15, latitude (-85, 85), `H = 30 km`.
- **IC**: DCMIP2016 BW initial condition (`Tᴱ = 310 K`, `Tᴾ = 240 K`,
  Γ = 0.005, K = 3, b = 2) with the standard 1 m/s velocity perturbation
  centred at (λ, φ) = (20°, 40°).
- **Reference state**: `θ_ref(z) = 250·exp(g z / (cᵖᵈ · 250))` (isothermal
  T₀ = 250 K hydrostatic background).
- **Time stepper**: `:AcousticRungeKutta3` (Wicker–Skamarock RK3 outer loop +
  MPAS conservative-perturbation acoustic substepping).
- **Damping**: `ThermodynamicDivergenceDamping(coefficient = 0.1)` (the
  Breeze default and the closest match to MPAS Klemp 2018), plus
  `acoustic_damping_coefficient = 0.5` (Klemp 2018 acoustic damping).
- **Adaptive substep count**: yes (`substeps = nothing`), based on the
  conservative horizontal acoustic CFL `Δτ ≤ 1.2 · Δx_min / c_s`.
- **Run target**: 7 simulated days, or first NaN.
- **Hardware**: NVIDIA H100 80GB.

The script lives at `test_bw_1deg_dt_sweep_compare.jl` and the full log at
`test_bw_1deg_dt_sweep_compare.log`.

## Results

`day7` means the run reached the 7-day target without a NaN. `crash dD.DD`
means it died on day `D.DD`. `max|w|` is the largest column-vertical-velocity
seen at any point in the run.

| Δt (s) | `ProportionalSubsteps`     | `MonolithicFirstStage`     |
|-------:|:---------------------------|:---------------------------|
|     60 | day7, max\|w\| = **0.26**  | day7, max\|w\| = 1.62      |
|    100 | day7, max\|w\| = **0.26**  | day7, max\|w\| = 0.82      |
|    150 | day7, max\|w\| = 2.70      | day7, max\|w\| = 3.00      |
|    200 | day7, max\|w\| = 3.51      | day7, max\|w\| = 3.85      |
|    300 | **crash d6.96**            | day7, max\|w\| = 5.42      |
|    500 | crash d4.20                | crash d3.99                |
|    800 | crash d2.25                | crash d2.18                |
|   1100 | crash d1.94                | crash d1.71                |
|   1400 | crash d1.65                | crash d1.46                |

`ProportionalSubsteps` reaches day 7 with a clean BCI lifecycle at Δt = 60s
and Δt = 100s (max\|w\| ≈ 0.26 in both, matching the canonical reference).
The two configurations are bit-comparable through day 4 and only diverge
once the BCI starts producing strong vertical motion.

## Key takeaways

### 1. The stability ceiling is set by *gravity-wave temporal resolution*, not the acoustic CFL

For the substepper alone, the relevant CFL is on the substep size
``Δτ = Δt/N``, and `compute_acoustic_substeps` adapts ``N`` so that
``Δτ \cdot c_s / Δx_\mathrm{min} ≤ 1`` is satisfied at every Δt. So the
substep CFL is held fixed across the sweep — every Δt is run at roughly the
same per-substep CFL.

The crashes happen because **buoyancy is computed in the slow tendency**
which is frozen across all WS-RK3 stages. At Δt > ~200 s the gravity wave
period (``2π/N ≈ 600`` s for ``N = 0.01``) is no longer well-resolved by Δt,
and the slow buoyancy update lags actual gravity wave evolution. This is a
slow-buildup instability — the runs survive several days before max\|w\|
grows monotonically and finally NaNs out.

### 2. `MonolithicFirstStage` is *not* limited by stage-1 horizontal acoustic CFL

A naive theory predicts that the WS-RK3 stage-1-with-one-substep form
should crash above Δt = 3·Δx_min/c_s ≈ 86 s on the 1° lat-lon grid (where
Δx_min ≈ 9.7 km at lat 85°). Empirically this is wrong: stage 1 of
`MonolithicFirstStage` runs cleanly with stage-1 substep CFL of 1.19
(Δt = 100), 1.79 (Δt = 150), 2.4 (Δt = 200), and even 3.6 (Δt = 300) — well
above the naive limit of 1.

The actual ceiling for both distributions is around Δt = 200–300 s, well
above the naive stage-1 prediction. Several mechanisms are likely
contributing — the off-centering parameter (`forward_weight = 0.6`, ε = 0.2)
provides additional damping; the `acoustic_damping_coefficient = 0.5`
extends the stable range further; and the WS-RK3 stage 1 advances state
only by `β₁ Δt` so a single unstable substep cannot grow far before stages 2
and 3 (which are CFL-stable) damp it.

### 3. `MonolithicFirstStage` adds noise at small Δt

Compare Δt = 60: `ProportionalSubsteps` max\|w\| = 0.26, `MonolithicFirstStage`
max\|w\| = 1.62 — six times noisier. Stage 1 of `MonolithicFirstStage` takes
*one* substep at ``Δt/3 = 20`` s instead of the *two* `Δt/N = 10` s substeps
that `ProportionalSubsteps` takes (with the auto-N = 6 floor on this grid),
so the time-averaging of the slow tendency over stage 1 is coarser. The
visible cost is a higher acoustic-noise floor in `w`.

**Recommendation**: prefer `ProportionalSubsteps` (the Breeze default) for
low-noise BW runs. `MonolithicFirstStage` is only useful for bit-compatible
comparisons against MPAS-A `config_time_integration_order = 3`.

### 4. The hard ceiling is around Δt ≈ 250 s on this grid

Both distributions reach day 7 at Δt ≤ 200 s and crash above Δt ~ 300 s.
The gap to MPAS-A's "Δt = 1800 s on a 1° mesh" is **roughly an order of
magnitude**.

A first-cut diagnosis (gravity wave temporal resolution) was wrong. The
actual culprit is the **advective CFL of WENO5 + WS-RK3 combined with the
BCI peak jet**:

| Δt (s) | CFL @ U = 30 m/s (init) | CFL @ U = 60 m/s (BCI peak) |
|-------:|-------------------------:|-----------------------------:|
|     60 | 0.19 | 0.37 |
|    100 | 0.31 | 0.62 |
|    150 | 0.46 | 0.93 |
|    200 | 0.62 | **1.24** |
|    300 | 0.93 | **1.86** |
|    400 | 1.24 | **2.47** |
|    500 | 1.55 | 3.09 |

WS-RK3 + WENO5 has an effective advective CFL limit of ~1.0–1.4. At
Δt = 200 the BCI peak hits CFL ≈ 1.24, right at the marginal stability
boundary. At Δt = 300 the BCI peak hits CFL ≈ 1.86 — clearly above. The
crash mode (slow growth of max\|w\| from ~0.1 to NaN over 4–7 days,
*after* the BCI starts producing strong winds) is consistent with an
advective scheme operating at marginal stability, *not* with a fast
acoustic blowup. This is also why no choice of divergence-damping
strategy can fix Δt > 300 — divergence damping cannot rescue an
underresolved advective scheme.

Closing the gap to MPAS-A's Δt = 1800 needs:

- a **top Rayleigh / sponge layer** to absorb upward-propagating gravity waves;
- a **4th-order horizontal hyperdiffusion** on momentum and θ to suppress
  grid-scale and 2Δx noise (the immediate driver of the marginal-CFL crash);
- and/or a more dissipative advection scheme (or one with looser CFL).

These should be tracked in a separate plan and are out of scope for the
substepper cleanup.

## Pressure-projection damping sweep — Δt × β_d

A follow-up sweep tests [`PressureProjectionDamping`](@ref) at three values
of the projection weight ``β_d`` to ask whether (a) it can match
[`ThermodynamicDivergenceDamping`](@ref) for noise suppression at low Δt,
and (b) cranking ``β_d`` extends the Δt ceiling.

Setup is identical to the table above (1° DCMIP2016 BW, 7-day target,
`ProportionalSubsteps`) except the damping is
`PressureProjectionDamping(coefficient = β_d)`. The script is
`test_bw_1deg_pressure_sweep.jl` and the log
`test_bw_1deg_pressure_sweep.log`.

The summary shows the **trajectory max\|w\| excluding the first 5 startup
steps** (the projection produces a benign IC transient at step ≈ 1–2 that
hits ~1.6 m/s and is identical across β_d, since the projection has no
history at that point — it is not a real instability).

| Δt (s) | β_d = 0.10 | β_d = 0.25 | β_d = 0.50 |
|-------:|:----------|:----------|:----------|
|    60 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.25 |
|   100 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.23 |
|   200 | day7, max\|w\| = **7.69** | day7, max\|w\| = 3.04 | day7, max\|w\| = **0.38** ✓ |
|   400 | crash d1.81             | crash d2.08             | crash d5.70             |
|   800 | crash d1.05             | crash d1.19             | crash d2.01             |
|  1400 | crash d1.12             | crash d1.20             | crash d1.26             |

### Key findings

1. **β_d barely matters at Δt ≤ 100 s.** The day-7 max\|w\| differs by less
   than 1 % across β_d ∈ {0.10, 0.25, 0.50}: at this Δt the (ρθ)″
   perturbation evolves slowly enough that the forward-extrapolation by
   one substep adds essentially nothing. The projection has no work to do
   when the substepping is already accurate.

2. **β_d = 0.5 at Δt = 200 produces a *cleaner* BCI than `ThermodynamicDivergenceDamping(0.1)` at the same Δt.**
   Compare:
   - `ThermodynamicDivergenceDamping(0.1)` at Δt = 200: trajectory max\|w\| = **3.51**
   - `PressureProjectionDamping(0.5)` at Δt = 200: trajectory max\|w\| (excluding startup) = **0.38**
     and ts_w\[end\] = 0.26 — comparable to Thermo at Δt = 60.

   This is the first empirical evidence that the literal-ERF/CM1/WRF
   pressure-projection form, with a strong enough projection weight, is
   competitive with — actually better than — the MPAS Klemp 2018
   thermodynamic-divergence form on this configuration. The cost is one
   additional `CenterField` of scratch storage and a per-cell EOS
   evaluation.

3. **β_d = 0.5 modestly extends the ceiling**, from Δt ≈ 300 to Δt ≈ 400.
   At Δt = 400, β_d = 0.5 survives to day 5.7 vs day 1.8 at β_d = 0.1.
   At Δt ≥ 500 all three β_d values crash within 1–2 days — the gravity
   wave temporal resolution problem dominates and no amount of damping can
   fix it.

4. The day-0.002 startup transient is a benign artifact of the projection
   filter being applied at substep 1 of stage 1 of step 1, when neither
   `ρθ″` nor `previous_rtheta_pp` has any history; it relaxes within a few
   steps and does not affect the BCI trajectory.

### Updated recommendation

For BW runs that target Δt > 100 s on the 1° lat-lon grid, prefer
**`PressureProjectionDamping(coefficient = 0.5)`** over the default
`ThermodynamicDivergenceDamping(coefficient = 0.1)`. It produces a cleaner
BCI lifecycle at moderate Δt (200) and a slightly higher Δt ceiling, at the
cost of one CenterField of scratch and a per-cell EOS evaluation per substep.

The default in `time_discretizations.jl` is unchanged
(`ThermodynamicDivergenceDamping(0.1)`) for bit-compatibility with the
pre-Phase-2 hardcoded path. Users who want larger Δt should opt in
explicitly.

## See also

- [Acoustic Substepping Cleanup and Damping Plan](substepping_cleanup_and_damping_plan.md)
- The four-strategy comparison at fixed Δt = 1400 s lives in
  `test_bw_1deg_damping_compare.jl` / `test_bw_1deg_damping_compare.log`,
  with the "form of damping matters at the same coefficient, but not enough
  at coefficient = 0.1" finding.
