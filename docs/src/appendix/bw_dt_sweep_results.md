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
magnitude**, and it is *not* a substepper-tuning problem. As laid out in
the [substepping cleanup plan](substepping_cleanup_and_damping_plan.md)
"out of scope" section, closing that gap requires the two damping
mechanisms MPAS-A has and Breeze does not:

- a **top Rayleigh / sponge layer** to absorb upward-propagating gravity waves;
- a **4th-order horizontal hyperdiffusion** on momentum and θ to suppress
  grid-scale and 2Δx noise.

These should be tracked in a separate plan.

## See also

- [Acoustic Substepping Cleanup and Damping Plan](substepping_cleanup_and_damping_plan.md)
- The four-strategy comparison at fixed Δt = 1400 s lives in
  `test_bw_1deg_damping_compare.jl` / `test_bw_1deg_damping_compare.log`,
  with the "form of damping matters at the same coefficient, but not enough"
  finding.
