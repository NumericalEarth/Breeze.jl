# Terrain-Following Production Validation Completion Audit

Status: incomplete as of 2026-05-22. Latest production gate:
`pass=16 present=23 fail=26 missing=0 blocked=5`.

Objective:

- Generate quantitative production-validation metrics for terrain-following
  compressible dynamics and acoustic substepping following
  `terrain_following_production_validation_metrics_plan.md`.
- Only long, production-validation artifacts can satisfy completion. Smoke and
  diagnostic artifacts are excluded.

## Checklist

| Requirement | Evidence | Status |
|---|---|---:|
| Schar 6 h CM1 explicit reference at matched production grid | `cm1_schar_400x200_periodic_theta300_reference/` and gate row for matched 400x200 grid | pass |
| Schar 6 h Breeze explicit and substepper production artifacts | `terrain_schar_6h_400x200_production_explicit/`, `terrain_schar_6h_400x200_production_substepper/` | pass |
| Schar explicit-vs-CM1 below-sponge 1% metrics | `schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_production_1pct_metrics_state_metrics.csv`; includes plan-required `u`, `w`, `θ`, and pressure fields and fails all field pass flags | fail |
| Schar substepper-vs-explicit below-sponge 1% metrics | `schar_6h_400x200_substepper_vs_explicit_production_1pct_metrics_state_metrics.csv`; includes plan-required `u`, `w`, `θ`, and pressure fields and fails at least `u`, `w`, `θ`, pressure, and drag gates | fail |
| Schar production movie or plot | `schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie/schar_cm1_vs_breeze_substepper_raw_error.mp4`, `schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie/schar_cm1_vs_breeze_substepper_field_error.mp4`, and production PPMs | pass |
| Schar three-resolution 6 h convergence campaign | `terrain_schar_6h_substepper_convergence_production/` | pass |
| Schar low-amplitude analytical linear-wave production artifacts | `linear_mountain_wave_production_400x200_6h_gpu/` and `linear_mountain_wave_explicit_production_400x200_6h_gpu/` include production metrics and final-state PPM plots | present |
| Schar low-amplitude analytical linear-wave 1% metrics | Substepper and explicit controls are stable but both fail analytical `w` gates after excluding boundary faces; substepper below-sponge `w_relative_l2_error = 1.6310697784`, explicit below-sponge `w_relative_l2_error = 1.8079439240` | fail |
| Schar low-amplitude substepper-vs-explicit Tier-1 metrics | `linear_mountain_wave_substepper_vs_explicit_400x200_6h_gpu/schar_substepper_vs_explicit_state_metrics.csv`; exact coordinate parity, but below-sponge `w_relative_l2_error = 0.2267029486`, pressure `relative_l2_error = 0.9280320257`, drag relative error `0.6306652106` | fail |
| Schar low-amplitude exact-wtilde substepper metrics | `linear_mountain_wave_production_400x200_6h_gpu_wtilde/linear_mountain_wave_state_metrics.csv`; stable, but below-sponge `w_tilde_relative_l2_error = 0.8352048912`, `w_tilde_relative_linf_error = 0.3833304476`, and `w_tilde_pattern_correlation = 0.7344266758` | fail |
| Schar low-amplitude exact-wtilde explicit-control metrics | `linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde/linear_mountain_wave_state_metrics.csv`; stable, but below-sponge `w_tilde_relative_l2_error = 0.9243801650`, `w_tilde_relative_linf_error = 0.4364041061`, and `w_tilde_pattern_correlation = 0.6997871856` | fail |
| Schar matched outer-dt production discriminator | Current-branch refresh `schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/schar_substepper_vs_explicit_state_metrics.csv`; exact coordinate parity, improved relative to the `dt = 2 s` no-damping/no-sponge candidate, but below-sponge `w_relative_l2_error = 0.1232164948`, `w_relative_linf_error = 0.0745803850`, `w_normalized_rmse = 0.0159684898`, pressure relative L2 `0.6617009969`, and drag relative error `0.4057695816` still fail 1% | fail |
| Schar previous-horizontal-divergence matched-dt production discriminator | `schar_substepper_vs_explicit_tier1_6h_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/schar_substepper_vs_explicit_state_metrics.csv`; exact coordinate parity, but below-sponge `w_relative_l2_error = 0.1232164948`, `w_relative_linf_error = 0.0745803850`, `w_normalized_rmse = 0.0159684898`, pressure relative L2 `0.6617009969`, and drag relative error `0.4057695816` still fail 1% and match the baseline failure | fail |
| Schar forward-weight 0.60 matched-dt production discriminator | `schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_forward0p6_grid/schar_substepper_vs_explicit_state_metrics.csv`; exact coordinate parity, but below-sponge `w_relative_l2_error = 0.1150234361`, `w_relative_linf_error = 0.0702176911`, `w_normalized_rmse = 0.0149066939`, pressure relative L2 `0.6416904611`, and drag relative error `0.3984645180` still fail 1% | fail |
| Schar first-substep-PGF matched-dt production discriminator | `schar_substepper_vs_explicit_tier1_6h_dt0p35_first_substep_pgf_no_damping_no_upper_sponge_grid/schar_substepper_vs_explicit_state_metrics.csv`; exact coordinate parity, but below-sponge `w_relative_l2_error = 0.0994969834`, `w_relative_linf_error = 0.0605755313`, `w_normalized_rmse = 0.0128945120`, pressure relative L2 `0.3980671346`, and drag relative error `0.1232760180` still fail 1% | fail |
| Schar first-substep-PGF plus forward-weight 0.60 matched-dt production discriminator | `schar_substepper_vs_explicit_tier1_6h_dt0p35_first_substep_pgf_forward0p6_no_damping_no_upper_sponge_grid/schar_substepper_vs_explicit_state_metrics.csv`; exact coordinate parity and best Schar production discriminator so far, but below-sponge `w_relative_l2_error = 0.0863024939`, `w_relative_linf_error = 0.0538500941`, `w_normalized_rmse = 0.0111845455`, pressure relative L2 `0.3343898585`, and drag relative error `0.0906279405` still fail 1% | fail |
| Schar 2 s operator-budget blocker baseline | `schar_2s_operator_budget_blocker_summary.csv` consolidates the early-time PGF, buoyancy, and advection blocker rows. `schar_2s_cm1_budget_closure_summary.csv` verifies CM1's own emitted u-budget closes to `(u₂-u₀)/Δt` at relative L2 `2.582710086e-8`, while `schar_2s_breeze_acoustic_increment_vs_cm1_ub_pgrad_summary.csv` shows Breeze's simple outer-step acoustic increment still fails against CM1 `ub_pgrad` at relative L2 `1.181135702`. Stable explicit Breeze `dt = 0.1 s` face-`u` increment also fails against CM1 face-`u` increment at relative L2 `1.324056471`; the CM1-terrain/CM1-constants rerun remains at relative L2 `1.339791739`. Active substepper pressure diagnostics self-close to roundoff but still fail against CM1 acoustic `ppd`; the horizontal channel dominates (`relative_l2 = 1.317131151`), pressure-vs-Exner, one-face shifts, sign flip, ungated first-substep pressure, post-recovery pressure replay, and nonlinear acoustic-state pressure reconstruction are ruled out. The post-recovery replay improves only to `relative_l2 = 1.174397371` against total CM1 acoustic `ppd`; nonlinear pressure follows the bad ungated branch (`relative_l2 = 1.363666148`) and remains diagnostic evidence only. | present |
| Complex mountain production manifest | `complex_mountain_doernbrack_production_manifest.md` declares 120x120x150 Doernbrack, 6 h, matched CM1/Breeze grid | pass |
| Complex mountain Breeze substepper production artifact | `complex_mountain_production_substepper/complex_mountain_state_slice.csv` and metrics/summary/time series | pass |
| Complex mountain Breeze explicit production artifact | `complex_mountain_production_explicit/complex_mountain_state_slice.csv`, metrics, summary, time series, and snapshots; run reached 21600 s | pass |
| Complex mountain CM1 reference artifact | `complex_mountain_production_cm1_reference/complex_mountain_state_slice.csv` and metrics converted from `cm1out_000037.nc` at `t = 21600 s` | pass |
| Complex mountain explicit-vs-CM1 metrics | `complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv`; below-sponge `w_relative_l2_error = 3.777`, pressure `relative_l2_error = 7.605`, drag signs do not match | fail |
| Complex mountain substepper-vs-explicit metrics | `complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv`; below-sponge `w_relative_l2_error = 0.764`, pressure `relative_l2_error = 0.827`, drag-x relative error `0.722` | fail |
| Complex mountain production movie | `complex_mountain_cm1_vs_breeze_substepper_movie/complex_mountain_cm1_vs_breeze_substepper.mp4`, 37 frames from 6 h production outputs | pass |
| Askervein production LES metrics/plot | `askervein_les_production/` exists and is labeled production_validation, but it is only a 1 s artifact without declared spin-up/averaging | present |
| Askervein accepted explicit feasible-window metrics | `askervein_explicit_substepper_compare_production/` exists, but the files are the old `2026-05-17` `1 s` artifact. A `2026-05-22` refresh attempt, Slurm job `1150`, tried `96 x 72 x 32`, `60 s`, CPU explicit/substepper and failed in the explicit half at step `134`, `t = 6.7 s`, in `temperature_and_pressure` with a negative base for exponentiation. Smaller-`dt` GPU feasibility runs reached `60 s` at `dt = 0.01 s`: job `1155` on Gaussian terrain failed vertical velocity (`w` full-domain rel L2 `0.170`, `w_tilde` rel L2 `0.283`), and job `1157` on ERF terrain also failed vertical velocity (`w` full-domain rel L2 `0.199`, `w_tilde` rel L2 `0.275`). Pressure-gradient-forced Askervein high bracket job `1156` completed at `192 x 192 x 64`, `300 s`, but `a = 0.04`, `0.06`, and `0.08 m s^-2` all produced `NaN` RS mast speed and `NaN` max FSR error. Lower bracket job `1158` also saturated at `a = 0.025`, `0.03`, and `0.035 m s^-2`, all with `reference_speed_model = NaN` and `max_abs_fsr_error = NaN`. Long finite-response job `1159` failed in diagnostic frame writing before metrics were emitted; the NaN-safe frame patch was applied and resubmitted as `1164`. The uncapped `1164` `a = 0.02 m s^-2` point completed at `256 x 256 x 96`, `2400 s`, with 24 full-size frames but still `reference_speed_model = NaN` and `max_abs_fsr_error = NaN`; `1164` was cancelled before the `0.025` point and capped retry `1165` was queued with `ASKER_CASE_MAX_DT=0.05`. Capped long retry `1165` completed `a = 0.02` and `0.025 m s^-2` at `256 x 256 x 96`, `2400 s` with finite but underdriven diagnostic mast metrics (`reference_speed_model = 2.256985683648537`, `max_abs_fsr_error = 6.555834500603011`; and `reference_speed_model = 2.5369400008976233`, `max_abs_fsr_error = 6.69566015642208`). Capped long retry `1186` completed `a = 0.10 m s^-2` at `256 x 256 x 96`, `2400 s` with finite but worse diagnostic mast metrics (`reference_speed_model = 5.253831533722133`, `max_abs_fsr_error = 10.139617785288996`, `max_abs_tke_error = 733.4619859674443`). Capped low-grid job `1166` completed `a = 0.02 m s^-2` at `192 x 192 x 64`, `300 s` with finite but inaccurate diagnostic mast metrics (`reference_speed_model = 4.643822590370479`, `max_abs_fsr_error = 1.409831851518315`) and then had its duplicate `a = 0.025 m s^-2` second leg cancelled. Capped single-leg jobs `1169`, `1172`, and `1174` completed `a = 0.025`, `0.045`, and `0.10 m s^-2` at `192 x 192 x 64`, `300 s` with finite but inaccurate diagnostic mast metrics (`reference_speed_model = 4.935863244558616`, `max_abs_fsr_error = 1.4516840207284545`; `reference_speed_model = 5.989918904395469`, `max_abs_fsr_error = 1.5342592003584854`; and `reference_speed_model = 8.30920752092283`, `max_abs_fsr_error = 1.3397411630772242`). No accepted production-window artifact exists. | blocked |
| Askervein WEMEP named-mast reference diagnostic | `askervein_wemep_reference/askervein_les_production_askervein_wemep_mast_metrics.csv`; matched RS/HT/CP, maximum FSR absolute error `0.387`, diagnostic only because Breeze artifact is idealized | present |
| Askervein accepted external reference / spin-up / averaging window | WEMEP/Zenodo reference files downloaded under `askervein_wemep_reference/`; `askervein_coordinate_faithful_production_manifest.md` records the WEMEP/ERF target geometry and reference scope; diagnostic ERF-terrain and WEMEP-mast plumbing smokes exist, including `96 x 72 x 32` explicit-vs-substepper diagnostics passing all 1% rows through `1.2 s` across full, near-terrain, centerline, lee-side, and hilltop regions and failing by `1.25 s` on `w_tilde` relative L∞, but production boundary conditions, spin-up/averaging, and accepted explicit feasible window remain undeclared | blocked |
| Required robustness metric schema | `terrain_following_production_validation_metric_schema_audit.md`; completed Schar and complex-mountain Breeze production metrics/time series now contain direct robustness fields, including CFL, acoustic CFL, finite-value counts, mass drift, bottom-normal velocity, high-k near-terrain energy, reflection fraction, and walltime-per-step | pass |

## 2026-05-19 Update: Combined Schar Setup Candidate

The missing combined Schar diagnostic was run after the earlier half-cell
terrain and prognostic-sponge candidates:

```text
validation_output/substepper/run_schar_400x200_gpu_cm1_terrain_prognostic_sponge_candidate.batch
```

Completed Slurm job:

```text
1053, gpu-dev, 2026-05-19
```

Configuration:

- `400 x 200`, `6 h`;
- `SCHAR_TERRAIN_INTERPRETATION=cm1`;
- prognostic sponge on;
- acoustic `UpperSponge` off;
- explicit `dt = 0.35 s`;
- substepper outer `dt = 2 s`.

Artifacts:

```text
validation_output/substepper/schar_cm1_terrain_prognostic_sponge_candidate_summary.md
validation_output/substepper/schar_production_validation_audit.md
validation_output/substepper/schar_pressure_reference_diagnosis.md
```

Result: still fails the active 1% production-validation contract.

Explicit-vs-CM1, below sponge:

- `w_normalized_rmse = 0.05753429306`
- `w_pattern_correlation = 0.6383547138`
- `p_relative_l2_error = 0.9877399439`
- `mountain_drag_relative_error = 1.721840766`

Substepper-vs-explicit, below sponge:

- `u_relative_l2_error = 0.0008602379329`
- `w_normalized_rmse = 0.009093622115`
- `w_relative_l2_error = 0.09752908177`
- `p_relative_l2_error = 0.4031520614`
- `mountain_drag_relative_error = 0.1011601771`

Pressure diagnosis:

- Removing the horizontal mean pressure perturbation at each vertical level
  cuts explicit-vs-CM1 full-domain `p'` relative L2 from `0.9940837434` to
  `0.5228791694`, but this remains far outside the 1% target.
- Substepper-vs-explicit pressure error does not improve under this
  x-demeaning (`0.4010089745` raw to `0.4989245643` x-demeaned full-domain
  relative L2).

Interpretation: CM1 terrain placement and upper-layer prognostic damping are
real setup corrections, but the remaining Schar pressure and wave-field gap is
not just a reference-column offset and not only a missing-artifact problem.

## 2026-05-19 Update: Schar Pressure-Gradient Stencil Check

Ran an explicit-only 400×200, 6 h candidate with the same CM1 terrain
interpretation and prognostic sponge, but with
`SCHAR_PRESSURE_GRADIENT_STENCIL=outside`:

```text
validation_output/substepper/run_schar_400x200_gpu_cm1_terrain_prognostic_sponge_outside_pgf_explicit_candidate.batch
```

Completed Slurm job:

```text
1055, gpu-dev, 2026-05-19
```

Summary:

```text
validation_output/substepper/schar_outside_pgf_candidate_summary.md
```

Result: the outside-stencil candidate still fails the 1% explicit-vs-CM1 gate:

- `w_normalized_rmse = 0.05739379042`
- `w_pattern_correlation = 0.6390117184`
- `p_relative_l2_error = 0.9876719330`
- `mountain_drag_relative_error = 1.720912066`

Outside-vs-inside Breeze explicit is small by comparison:

- `w_normalized_rmse = 0.001435565080`
- `p_relative_l2_error = 0.003254700596`
- `mountain_drag_relative_error = 0.001286571244`

Interpretation: the remaining Schar CM1 gap is not explained by the
inside-vs-outside terrain pressure-gradient stencil choice.

The production gate was rerun after this update:

```sh
julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl
```

Result:

```text
earlier gate result: pass=16 present=16 fail=15 missing=0 blocked=5
```

The command exits nonzero because the validation contract remains incomplete.

## 2026-05-20 Update: Terrain Interpretation Interface and CM1 `kdiv`

The Schar terrain placement discrepancy now has an explicit Breeze interface:

```julia
follow_terrain!(grid, h; terrain_interpretation = FaceSampledTerrain())
```

`GridFittedTerrain()` remains the default cell-center interpretation.
`FaceSampledTerrain()` evaluates function-valued topography at the upper
horizontal face of each non-flat cell, matching the half-cell CM1 interpretation
used by the Schar validation scripts. Targeted smokes passed:

```text
FaceSampledTerrain smoke passed
terrain interpretation docs names exported
```

The production gate was rerun after this interface work:

```sh
julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl
```

Result:

```text
earlier gate result: pass=16 present=16 fail=15 missing=0 blocked=5
```

The gate remains incomplete. This historical count is superseded by the latest
gate result recorded above: `pass=16 present=23 fail=26 missing=0 blocked=5`.

CM1 source inspection identifies the next Schar operator mismatch. In the
official NCAR/CM1 source, `psolver = 3` applies `kdiv` by first updating pressure
perturbation and then using a damped pressure field:

```text
ppd(i,j,k) = pp3d(i,j,k)
pp3d(i,j,k) = pp3d(i,j,k) - tem * div / thv0(i,j,k)
dum1(i,j,k) = kdiv * (pp3d(i,j,k) - ppd(i,j,k))
ppd(i,j,k) = pp3d(i,j,k) + dum1(i,j,k)
```

This is not algebraically the same as Breeze's current
`ThermalDivergenceDamping`, which applies a post-substep horizontal momentum
correction from the `(ρθ)'` tendency. The Schar validation scripts can disable
that Breeze damping or tune its coefficient, but they do not yet implement a
CM1-style pressure-increment damping path.

CM1's vertical Rayleigh damper also differs from the candidate Breeze sponge
setup in where it is applied:

- CM1 `irdamp = 1` damps `u`, `v`, `w`, and `θ` tendencies above `zd = 20 km`
  with a `0.5 * (1 - cos(...))` ramp and `rdalpha = 1 / 300 s⁻¹`.
- The best matched Breeze candidate used external prognostic damping to mimic
  that slow-tendency behavior and disabled the acoustic `UpperSponge`.

Interpretation: the terrain sampling discrepancy has an interface and targeted
tests, but the Schar 1% blocker remains.

An experimental CM1-style pressure-increment damping path was then prototyped
and tested in a `600 s`, `400 x 200` discriminator, but it was not kept in
source because it did not improve the validation enough to justify adding a new
public damping strategy to this PR.

Artifacts:

```text
validation_output/substepper/terrain_schar_600s_400x200_substepper_cm1_pressure_increment_discriminator/
validation_output/substepper/schar_600s_400x200_substepper_pressure_increment_vs_cm1_periodic_theta300_state_metrics.csv
validation_output/substepper/schar_600s_400x200_substepper_pressure_increment_vs_explicit_cm1_terrain_prognostic_sponge_state_metrics.csv
```

Pressure-increment substepper-vs-CM1 at `600 s`, below sponge:

- `u_relative_l2_error = 0.01006377416`
- `w_relative_l2_error = 1.927417558`
- `θ_relative_l2_error = 2.651204359`
- `p_relative_l2_error = 1.792064315`
- `mountain_drag_relative_error = 1.478028659`

Pressure-increment substepper-vs-explicit at `600 s`, below sponge:

- `u_relative_l2_error = 0.003059422885`
- `w_relative_l2_error = 0.2331986559`
- `θ_relative_l2_error = 0.01451900473`
- `p_relative_l2_error = 0.6469242896`
- `mountain_drag_relative_error = 0.03266989277`

Result: all 1% field hooks still fail. The prototype modestly improves some
substepper-vs-explicit rows relative to the default thermal damping, but it
worsens substepper-vs-CM1 pressure relative to the default 600 s discriminator.
This rules out the simple linearized pressure-increment filter as the Schar
fix. The next Schar target should be pressure/reference-state or Rayleigh
damping placement, not another blind `kdiv` coefficient sweep.

Additional `600 s` setup discriminator:

```text
validation_output/substepper/schar_600s_cm1_constants_discriminator_summary.md
```

This tested whether CM1's dry-air constants (`Rd = 287.04`, `cp = 1005.7`)
explain the early Schar state-level mismatch. They do not. Initial-condition
comparison with CM1 terrain interpretation shows very small geometry,
potential-temperature, and velocity differences, plus a `10.3 Pa` maximum
absolute hydrostatic pressure difference. Running both Breeze explicit and
substepper with CM1 constants leaves the `600 s` metrics essentially unchanged.

CM1-constants explicit-vs-CM1 at `600 s`, below sponge:

- `w_relative_l2_error = 1.994880230`
- `p_relative_l2_error = 2.818088612`
- `mountain_drag_relative_error = 1.494476900`

CM1-constants substepper-vs-CM1 at `600 s`, below sponge:

- `w_relative_l2_error = 1.906884706`
- `p_relative_l2_error = 1.454004061`
- `mountain_drag_relative_error = 1.454448741`

CM1-constants substepper-vs-explicit at `600 s`, below sponge:

- `w_relative_l2_error = 0.2550219082`
- `p_relative_l2_error = 0.7591503671`
- `mountain_drag_relative_error = 0.08095051372`

Interpretation: CM1/Breeze dry-air constant parity is useful documentation, but
it is not a Schar fix. The next Schar target remains Rayleigh damping placement
or a deeper source-term/operator comparison.

## 2026-05-20 Update: Schar CM1-Style Exner Pressure-Gradient Diagnostic

Ran a `400 x 200`, `600 s` GPU discriminator using the `FaceSampledTerrain()`
interface, CM1-rate prognostic sponge, and a validation-only
`SCHAR_PRESSURE_GRADIENT_STENCIL=cm1_exner` prototype in
`terrain_schar_mountain_wave_explicit_validation.jl`.

Artifacts:

```text
validation_output/substepper/terrain_schar_600s_400x200_explicit_cm1_exner_operator_discriminator/
validation_output/substepper/schar_600s_breeze_explicit_cm1_exner_exact_operator_budget_vs_cm1_summary.md
validation_output/substepper/schar_600s_400x200_explicit_cm1_exner_vs_cm1_periodic_theta300_state_metrics.csv
validation_output/substepper/schar_600s_400x200_explicit_cm1_exner_vs_cm1_periodic_theta300_state_summary.txt
validation_output/substepper/schar_cm1_exner_600s_gpu-1061.log
```

Result: still fails the active 1% production-validation contract and should
not be promoted to public source as-is.

Below-sponge explicit-vs-CM1 at `600 s`:

- `u_relative_l2_error = 1.064697974e-02`
- `w_relative_l2_error = 1.994119807`
- `θ'_relative_l2_error = 2.656085046`
- `p'_relative_l2_error = 2.817149674`
- `mountain_drag_relative_error = 1.494087191`

Live Breeze operator-budget comparison against CM1 at `600 s`:

- `ub_pgrad` relative L2 `12.74054722`
- `wb_pgrad` relative L2 `11.85731404`
- `wb_buoy` relative L2 `4.41445510`

Interpretation: the Schar blocker remains unresolved. This negative
discriminator rules out the current simple CM1-style Exner pressure-gradient
prototype as a fix and keeps the goal incomplete.

## 2026-05-20 Update: Schar Early-Time Operator-Budget Diagnostic

Ran a `2 s`, `400 x 200` CM1 budget diagnostic and a matched Breeze explicit
operator-budget diagnostic to determine whether the operator mismatch is
already present before the `600 s` state divergence.

Artifacts:

```text
validation_output/substepper/cm1_schar_400x200_periodic_theta300_budget_2s_reference/
validation_output/substepper/terrain_schar_2s_400x200_explicit_operator_budget_diagnostic/
validation_output/substepper/schar_2s_breeze_explicit_exact_operator_budget_vs_cm1_summary.md
validation_output/substepper/schar_2s_400x200_explicit_vs_cm1_periodic_theta300_state_summary.txt
validation_output/substepper/schar_2s_breeze_cm1_style_operator_budget_vs_cm1_summary.md
validation_output/substepper/schar_2s_cm1_state_cm1_style_operator_budget_vs_cm1_summary.md
validation_output/substepper/schar_2s_budget_and_state_diagnosis.md
validation_output/substepper/schar_0_to_2s_breeze_explicit_vs_cm1_state_delta_metrics.csv
validation_output/substepper/schar_0_to_2s_breeze_explicit_vs_cm1_state_delta_metrics_summary.md
validation_output/substepper/schar_0_to_2s_breeze_explicit_dt2_vs_cm1_state_delta_metrics.csv
validation_output/substepper/schar_0_to_2s_breeze_explicit_dt2_vs_cm1_state_delta_metrics_summary.md
```

Below-sponge exact operator-budget comparison at `2 s`:

- `ub_pgrad` relative L2 `1.352884684`
- `wb_pgrad` relative L2 `1.566359834`
- `wb_buoy` relative L2 `759.0814827`
- `ub_pgrad` pattern correlation `0.6893866486`
- `wb_pgrad` pattern correlation `0.2385745030`
- combined `wb_pgrad + wb_buoy` relative L2 `1.583056028`
- combined `wb_pgrad + wb_buoy` pattern correlation `0.2082062112`

The `wb_buoy` relative error is inflated by CM1's near-zero `2 s` buoyancy
budget (`1.104348048e-04` max abs), but the absolute Breeze-vs-CM1 difference
is still nontrivial (`4.604738673e-02` max abs).

Below-sponge Breeze explicit vs CM1 state comparison at `2 s`:

- `u_relative_l2_error = 2.966589680e-03`
- `w_relative_l2_error = 0.3594733174`
- `p'_relative_l2_error = 0.9389786848`
- `mountain_drag_relative_error = 3.866197207e-02`

State-delta comparison from initialized state to `2 s`:

- `u` delta relative L2 `1.464901222`
- `w_center` delta relative L2 `0.2411136857`
- `theta_perturbation` delta relative L2 `6.662845586`
- `pressure_perturbation` delta relative L2 `0.3631769546`

Matching Breeze explicit to `SCHAR_DT=2` does not improve the early comparison:

- `w_center` delta relative L2 worsens to `3.687142155`
- `pressure_perturbation` delta relative L2 worsens to `11.56600544`
- direct `2 s` `p'_relative_l2_error` worsens to `11.59411650`
- direct `2 s` mountain-drag relative error worsens to `17.31686255`

CM1 source inspection also found that `ub_pgrad`/`wb_pgrad` are diagnosed from
velocity changes over the model step after saving the pre-pressure-gradient
tendencies, while Breeze's current operator-budget CSV is an instantaneous
live-operator evaluation. The comparison is still a useful discriminator, but
not a final apples-to-apples source-term proof.

The CM1-style Exner formula check at `2 s` confirms the convention issue:
even on the CM1 output state, the formula gives `relative_l2_error =
1.060727576` and pattern correlation `0.8231754156` against CM1 `ub_pgrad`.

CM1's own budget closure at `2 s` was then checked directly:

```text
validation_output/substepper/schar_2s_cm1_budget_closure_summary.md
validation_output/substepper/schar_2s_cm1_budget_closure_summary.csv
```

Below the sponge, the sum of CM1 `ub_*` terms matches `(u₂-u₀)/Δt` with
relative L2 `2.582710086e-8`, and residual-inferred PGF matches emitted
`ub_pgrad` with relative L2 `2.582295417e-8`. This confirms CM1's emitted
`ub_pgrad` is a valid early-time reference budget even though a pointwise
output-state PGF formula does not reconstruct it.

A Breeze outer-step acoustic-increment diagnostic was also tested:

```text
validation_output/substepper/schar_2s_breeze_acoustic_increment_vs_cm1_ub_pgrad_summary.md
validation_output/substepper/schar_2s_breeze_acoustic_increment_vs_cm1_ub_pgrad_summary.csv
```

It remains far from CM1 `ub_pgrad`: below-sponge relative L2 `1.181135702`,
normalized RMSE `0.01871781586`, and pattern correlation `0.7197870843`.
This rules out the simple Breeze outer-step increment view as a sufficient
reproduction of CM1's in-step pressure/acoustic budget convention.

Interpretation: the Schar mismatch is already present at `2 s`, so it is not a
late-time sponge/reflection artifact. The next Schar work should target
initial/source-term parity, especially the vertical pressure-gradient/buoyancy
split over terrain and a closer reproduction of CM1's in-step budget
conventions.

Follow-up Lin finite-volume PGF discriminator:

```text
validation_output/substepper/terrain_schar_2s_400x200_explicit_weno9_lin_pgf_operator_budget_diagnostic/
validation_output/substepper/schar_2s_weno9_lin_pgf_operator_budget_vs_cm1_summary.md
```

This tested an experimental Lin-style finite-volume terrain pressure-gradient
stencil in the Schar validation path. It is a failed diagnostic, not a
production-validation artifact: the run produced NaNs by `t = 2 s`
(`nan_count = 372392`), and the live operator-budget comparison still failed
badly (`ub_pgrad` relative L2 `1.575561862`, `wb_pgrad` relative L2
`1.529498472`). The experimental public stencil was removed from source to keep
the PR minimal and avoid retaining a broken unused API.

Additional CM1 `csound` discriminator:

```text
validation_output/substepper/schar_600s_cm1_csound347_discriminator_summary.md
```

This tested whether CM1's `csound = 300 m/s` is a mismatch against Breeze's
physical compressible EOS sound speed, approximately `347 m/s` near `θ = 300 K`.
A short CM1 run with `csound = 347.0 m/s` completed to `600 s`.

Result: CM1 `csound=300` and CM1 `csound=347` are identical to the current
state-slice metric precision at `600 s`:

- CM1-vs-CM1 `w_relative_l2_error = 0`
- CM1-vs-CM1 `p_relative_l2_error = 0`
- CM1-vs-CM1 `mountain_drag_relative_error = 0`

Breeze-vs-CM1 `csound=347` errors are unchanged:

- explicit-vs-CM1 `w_relative_l2_error = 1.995760207`
- explicit-vs-CM1 `p_relative_l2_error = 2.816347394`
- substepper-vs-CM1 `w_relative_l2_error = 1.907534093`
- substepper-vs-CM1 `p_relative_l2_error = 1.452524639`

Interpretation: CM1 `csound` is not the missing early Schar fix. The remaining
gap points back to terrain/source-term operator details, advection/numerical
diffusion, or an unisolated output/conversion convention.

## Current Gate

Latest command:

```sh
julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl
```

Latest result:

```text
production validation gate: pass=16 present=23 fail=26 missing=0 blocked=5
```

The gate remains incomplete. Completion is blocked by:

- measured 1% accuracy failures in Schar and complex mountain;
- the best matched outer-`dt = 0.35 s` Schar production discriminator now
  combines first-substep PGF with `forward_weight = 0.60`; it improves the
  Schar Tier-1 metrics but still fails the 1% gates;
- measured analytical and substepper-vs-explicit 1% failures in the
  low-amplitude Schar linear-wave production control;
- Askervein's missing coordinate-faithful production definition;
- Askervein's missing accepted explicit-feasible production window;
- Askervein's missing production LES, reference-comparison, spin-up/averaging,
  and explicit-window output contract;
- final-manifest incompleteness caused by the unresolved Askervein production
  artifacts.

The Askervein reference files are downloaded, a coordinate-faithful production
manifest exists, and diagnostic ERF-terrain/WEMEP-mast plumbing exists. A
`96 x 72 x 32`, `1.2 s` ERF-terrain explicit-vs-substepper diagnostic passes
all 1% rows across full, near-terrain, centerline, lee-side, and hilltop
regions, while `1.25 s` fails on `w_tilde` relative L∞. A newer target-grid
`300 x 300 x 18`, `1.2 s` GPU diagnostic proves the ERF grid and terrain can
run, but narrowly fails vertical metrics (`w_relative_l2_error = 0.01037718669`,
`w_tilde_relative_l2_error = 0.01157095641`). The validation workflow still
lacks production boundary conditions, declared production spin-up/averaging
window, and an accepted explicit-feasible production window.

The robustness metric schema for completed Schar and complex-mountain
production runs is now direct rather than proxy-derived, and the finite-value /
bottom-normal-velocity rows pass for those completed runs. Plan-required
saved-time reference comparison coverage now exists for the completed Schar and
complex-mountain artifacts, but those saved-time explicit-vs-CM1 rows fail the
1% gate. The remaining blockers are accuracy failures plus the unresolved
Askervein production definition/window.

## 2026-05-19 Update: Schar 600 s State Discriminator

A short production-grid diagnostic was added to determine whether the Schar gap
is already present at the first CM1 saved time:

```text
validation_output/substepper/schar_600s_state_discriminator_summary.md
```

This diagnostic is not a production-validation artifact because it runs only
`600 s`, below the required Schar `6 h` minimum. It is useful blocker evidence:
the state-level gap is immediate enough that another blind 6 h rerun is not the
right next step.

Below-sponge `600 s` explicit-vs-CM1:

- `u_relative_l2_error = 0.01065042654`
- `w_relative_l2_error = 1.995760207`
- `theta_relative_l2_error = 2.657010018`
- `p_relative_l2_error = 2.816347394`
- `mountain_drag_relative_error = 1.494173246`

Below-sponge `600 s` substepper-vs-CM1:

- `u_relative_l2_error = 0.009897861734`
- `w_relative_l2_error = 1.907534093`
- `theta_relative_l2_error = 2.649673372`
- `p_relative_l2_error = 1.452524639`
- `mountain_drag_relative_error = 1.454436023`

Below-sponge `600 s` substepper-vs-explicit:

- `u_relative_l2_error = 0.003539572831`
- `w_relative_l2_error = 0.2551554125`
- `theta_relative_l2_error = 0.01716495356`
- `p_relative_l2_error = 0.7595737236`
- `mountain_drag_relative_error = 0.08041152263`

Interpretation: the Schar mismatch is a full-state early-response problem, not
only a late-time reflection or movie-field artifact. The next Schar work should
use this 600 s window to test CM1-like acoustic filtering/damping and
pressure/reference-state conventions before another long production rerun.

Follow-up 600 s discriminator:

```text
validation_output/substepper/schar_600s_forward_weight_0p8_discriminator_summary.md
```

This tested `SCHAR_FORWARD_WEIGHT=0.8`, motivated by CM1's `alph = 0.60`.
It does not close the gap:

- substepper-vs-CM1 `w_relative_l2_error`: `1.907534093` at `ω = 0.65` to
  `1.902723632` at `ω = 0.8`;
- substepper-vs-CM1 `p_relative_l2_error`: `1.452524639` to `1.431653863`;
- substepper-vs-explicit `w_relative_l2_error`: `0.2551554125` to
  `0.2584556665`;
- substepper-vs-explicit `p_relative_l2_error`: `0.7595737236` to
  `0.7652366867`.

Interpretation: simple acoustic off-centering parity is not the missing Schar
fix. The next target remains pressure/reference-state convention or CM1 `kdiv`
operator equivalence.

Additional 600 s discriminator:

```text
validation_output/substepper/schar_600s_no_divergence_damping_discriminator_summary.md
```

This tested `SCHAR_DIVERGENCE_DAMPING=none`. It improves
substepper-vs-explicit but not enough for the 1% gate:

- substepper-vs-explicit `w_relative_l2_error`: `0.2551554125` to
  `0.2255918258`;
- substepper-vs-explicit `theta_relative_l2_error`: `0.01716495356` to
  `0.01335220350`;
- substepper-vs-explicit `p_relative_l2_error`: `0.7595737236` to
  `0.6281521535`;
- substepper-vs-explicit `mountain_drag_relative_error`: `0.08041152263` to
  `0.02789799007`.

But it worsens substepper-vs-CM1 pressure:

- substepper-vs-CM1 `p_relative_l2_error`: `1.452524639` to `1.853423408`.

Interpretation: thermal divergence damping contributes to the Breeze
substepper-vs-explicit Schar gap, but removing it does not solve the CM1
cross-model validation gap and is still not a passing production candidate.

Additional 600 s discriminator:

```text
validation_output/substepper/schar_600s_weno9_discriminator_summary.md
```

This tested CM1-like WENO-9 advection with the current Schar CM1-terrain and
prognostic-sponge setup. It does not close the 1% gate:

- explicit WENO-9 vs CM1 `w_relative_l2_error = 1.988231344`,
  `p_relative_l2_error = 2.802184993`, drag error `1.630462675`;
- substepper WENO-9 vs CM1 `w_relative_l2_error = 1.900304759`,
  `p_relative_l2_error = 1.452945670`, drag error `1.593968337`;
- substepper WENO-9 vs explicit WENO-9 `w_relative_l2_error = 0.2451021382`,
  `p_relative_l2_error = 0.7572594937`, drag error `0.05788501042`.

Interpretation: advection-order parity with CM1 is not the missing Schar fix.
The source-level terrain interpretation option remains useful, but the next
Schar blocker is more likely terrain/source-term operator details or an
unisolated CM1 output/conversion convention.

Conversion-convention discriminator:

```text
validation_output/substepper/schar_cm1_conversion_convention_diagnosis.md
```

This checked the `t = 600 s` CM1 frame used by the early state discriminator:

- CM1 `zs` matches analytic Schar terrain evaluated at `xh + dx/2` to roundoff
  (`max abs = 4.58e-5 m`) and not at `xh` (`max abs = 48.9 m`);
- CM1 `uinterp` and `winterp` match face-averaged `u`/`w` exactly;
- CM1 `thpert` matches `th - th0` to single-precision roundoff
  (`max abs = 1.53e-5 K`).

Interpretation: the current CM1 converter is not the source of the large 600 s
state mismatch through these simple field-location or perturbation
conventions. Remaining Schar work should focus on terrain/source-term operator
parity.

CM1 budget-operator artifact:

```text
validation_output/substepper/schar_cm1_budget_operator_diagnosis.md
```

This ran the same `400 x 200`, `600 s`, periodic Schar CM1 setup with
`output_ubudget`, `output_vbudget`, and `output_wbudget` enabled. The budget run
is bit-identical to the existing CM1 `600 s` state for `uinterp`, `winterp`,
`thpert`, and `prs`, so the diagnostic output does not perturb the baseline.

New full-field CM1 operator diagnostics are available:

- `ub_pgrad` max abs `2.874486707e-02`, rms `1.388462142e-03`;
- `wb_pgrad` max abs `1.976923086e-02`, rms `2.379304165e-03`;
- `wb_buoy` max abs `1.871932484e-02`, rms `4.700960255e-04`.

Interpretation: the next Schar completion blocker can now be attacked with a
direct Breeze-vs-CM1 operator comparison at `600 s`, instead of another
full-run discriminator.

Approximate Breeze-vs-CM1 pressure-gradient operator comparison:

```text
validation_output/substepper/compare_schar_breeze_cm1_pgrad_budget.jl
validation_output/substepper/schar_600s_breeze_explicit_vs_cm1_pgrad_budget_summary.md
```

This reconstructs a Breeze explicit horizontal pressure-gradient acceleration
from the saved `600 s` state slice and compares it to CM1's `ub_pgrad` budget
field over below-sponge interior u-faces. It is an operator-screening
diagnostic, not a production artifact and not an exact in-kernel Breeze budget.

Result:

- `relative_l2_error = 5.213824307`;
- `normalized_rmse = 1.838029210`;
- `pattern_correlation = -0.03806730697`;
- Breeze reconstructed max abs `7.471639080e-02`;
- CM1 `ub_pgrad` max abs `3.482475877e-03`.

Interpretation: this is the strongest current lead for the Schar blocker. The
next step should be an exact Breeze in-model pressure-gradient / buoyancy
budget diagnostic, followed by a targeted operator fix if the exact diagnostic
confirms the reconstructed mismatch.

Exact Breeze-vs-CM1 operator-budget comparison:

```text
validation_output/substepper/compare_schar_breeze_cm1_exact_operator_budget.jl
validation_output/substepper/schar_600s_breeze_explicit_exact_operator_budget_vs_cm1_summary.md
```

The Schar explicit validation script now has optional validation-only operator
output via `SCHAR_WRITE_OPERATOR_BUDGET=true`, writing exact live-model
`ub_pgrad`, `wb_pgrad`, and `wb_buoy` accelerations. A `400 x 200`, `600 s`,
GPU explicit rerun with this flag produced state fields bit-identical to the
previous explicit discriminator for `u`, `w`, `theta_perturbation`, and
`pressure_perturbation`, so the operator output does not perturb the state.

Below-sponge exact operator comparison against CM1 budget output:

- `ub_pgrad` relative L2 `12.72824860`, normalized RMSE `0.5488917869`,
  pattern correlation `0.1162468955`;
- `wb_pgrad` relative L2 `11.84904428`, normalized RMSE `0.6856449520`,
  pattern correlation `0.1250048315`;
- `wb_buoy` relative L2 `4.413985282`, normalized RMSE `0.1356355540`,
  pattern correlation `0.2132844621`.

Interpretation: the Schar gap is now localized to terrain pressure-gradient /
buoyancy operator parity, especially the pressure-gradient terms. This is a
diagnostic artifact and does not satisfy the production validation goal, but it
identifies the next targeted implementation experiment.

CM1-style pressure-gradient formula discriminator:

```text
validation_output/substepper/compare_schar_breeze_cm1_style_operator_budget.jl
validation_output/substepper/schar_600s_breeze_cm1_style_operator_budget_vs_cm1_summary.md
validation_output/substepper/schar_600s_cm1_pgrad_formula_validation.md
```

Evaluating a CM1-style Exner pressure-gradient formula on the saved Breeze
`600 s` state does not match CM1 `ub_pgrad`:

- relative L2 `9.989748504`;
- normalized RMSE `0.4307969683`;
- pattern correlation `-0.001144230603`.

The same formula evaluated on the CM1 `600 s` state reproduces CM1's own
`ub_pgrad` well enough for a diagnostic:

- relative L2 `0.1134128818`;
- normalized RMSE `0.004890806372`;
- pattern correlation `0.9935849841`.

Interpretation: the formula/indexing is credible, but the mismatch is not a
simple post-hoc pressure-vs-Exner conversion of Breeze output. The next
targeted experiment must rerun Breeze with an opt-in CM1-style terrain
pressure-gradient / buoyancy operator and then compare the exact operator
budget again before any long production validation rerun.

## 2026-05-20 Update: Schar 2 s Momentum-Advection Budget

The Schar operator-budget diagnostic was extended to include Breeze's exact
total momentum-advection accelerations:

```text
validation_output/substepper/terrain_schar_2s_400x200_explicit_operator_budget_adv_diagnostic/
validation_output/substepper/schar_2s_breeze_explicit_exact_operator_budget_with_advection_vs_cm1_summary.md
```

These rows compare Breeze's `-momentum_flux_divergence / ρ` against CM1's
`*_hadv + *_vadv` budget terms at `2 s`, below sponge:

- `ub_adv` relative L2 `11.36898998`, normalized RMSE `0.1263347457`,
  pattern correlation `0.3563706622`;
- `wb_adv` relative L2 `5.575643183`, normalized RMSE `0.06425630750`,
  pattern correlation `-0.01979790422`.

The pressure-gradient and buoyancy rows remain far outside the 1% gate at the
same time:

- `ub_pgrad` relative L2 `1.352884684`;
- `wb_pgrad` relative L2 `1.566359834`;
- `wb_buoy` relative L2 `759.0814827`.

Interpretation: the Schar blocker is broader than pressure-gradient formula
parity. The next Schar implementation discriminator should target
terrain-coordinate momentum-budget parity across advection, pressure-gradient,
and buoyancy/source-term conventions before spending another long production
run.

## 2026-05-20 Update: Schar θ-Offset Convention Check

CM1 advects `sadv = (th0 - th0r) + thpert` with `th0r = 300 K`. Breeze advects
diagnostic θ through prognostic `ρθ`. This convention difference was tested as
a possible explanation for the early θ′ mismatch:

```text
validation_output/substepper/diagnose_schar_theta_offset_convention.jl
validation_output/substepper/schar_2s_theta_offset_convention_diagnosis.md
```

The diagnostic reconstructs dry density from the `0 s` and `2 s` state slices
and applies the two possible first-order `±300 Δρ/ρ` corrections to Breeze's
θ′ delta. The correction does not help:

- raw Breeze θ′ delta relative L2 `6.662845586`;
- Breeze θ′ delta plus `300 Δρ/ρ` relative L2 `209.4573699`;
- Breeze θ′ delta minus `300 Δρ/ρ` relative L2 `206.3556228`.

Interpretation: the CM1 `θ - 300 K` advection convention is not the primary
Schar blocker. This remains diagnostic-only evidence and does not satisfy any
production validation gate.

## 2026-05-20 Update: Schar Split-Advection Budget

The Schar operator budget now splits Breeze momentum advection into horizontal
and vertical pieces so the rows can be compared directly with CM1's
`*_hadv` and `*_vadv` budgets:

```text
validation_output/substepper/terrain_schar_2s_400x200_explicit_operator_budget_split_adv_diagnostic/
validation_output/substepper/schar_2s_breeze_explicit_exact_operator_budget_split_advection_vs_cm1_summary.md
```

Below-sponge `2 s` results:

- `ub_hadv` relative L2 `9.841472398`, pattern correlation `0.5094853488`;
- `ub_vadv` relative L2 `114.2449497`, pattern correlation `-0.03849451545`;
- `wb_hadv` relative L2 `5.608731512`, pattern correlation `-0.01993244941`;
- `wb_vadv` relative L2 `1.880021143`, pattern correlation `0.1939855348`.

Interpretation: the momentum-advection mismatch is not only a total-vs-split
budget-label artifact. The largest normalized discrepancy is `ub_vadv`, where
CM1's vertical advection of x-momentum is tiny but Breeze's terrain-coordinate
vertical transport contributes substantially. This is diagnostic-only evidence
and does not satisfy the production validation gate.

CM1 source inspection supports this lead:

- `solve2.F:533` constructs terrain `rru` with an `rgzu` metric factor;
- `solve2.F:555-565` constructs terrain `rrw` as `rf0 * w3d` plus a
  slope-weighted neighboring `rru`/`rrv` contribution;
- `solve2.F:885-890` passes those metric mass fluxes into `advu`, `advv`, and
  `advw`.

Breeze currently constructs `ρw̃ = ρw - slope_x * ρu - slope_y * ρv` and then
uses Oceananigans area/volume operators. The next diagnostic should compute a
CM1-style terrain vertical mass flux from the Breeze state and compare budget
terms before making any source-level dynamics change.

A WENO-9 control was run for the same split-budget diagnostic:

```text
validation_output/substepper/terrain_schar_2s_400x200_explicit_weno9_operator_budget_split_adv_diagnostic/
validation_output/substepper/schar_2s_breeze_explicit_weno9_operator_budget_split_advection_vs_cm1_summary.md
```

The mismatch does not improve:

- `ub_hadv` relative L2 `11.12416618`;
- `ub_vadv` relative L2 `126.9171154`;
- `wb_hadv` relative L2 `6.284670403`;
- `wb_vadv` relative L2 `2.013449596`.

Interpretation: advection order is not the explanation for the split-budget
gap. The next diagnostic should still target CM1-style terrain mass-flux
construction.

The CM1-style terrain vertical mass-flux hypothesis was then tested directly:

```text
validation_output/substepper/compare_schar_vertical_mass_flux.jl
validation_output/substepper/schar_2s_weno9_breeze_vs_cm1_like_vertical_mass_flux_diagnosis.md
validation_output/substepper/terrain_schar_2s_400x200_explicit_weno9_operator_budget_cm1_mass_flux_diagnostic/
```

The Schar operator budget now emits Breeze's live `ρw̃` and a CM1-like `rrw`
reconstructed from the same Breeze state. At `2 s`, below sponge:

- `ρw̃` vs CM1-like `rrw` relative L2 `2.952386152e-4`;
- normalized RMSE `1.565306739e-5`;
- pattern correlation `0.9999999564`;
- Breeze max abs `1.885864434`;
- CM1-like max abs `1.885802421`.

Interpretation: the vertical mass-flux construction itself is not the large
`ub_vadv` mismatch. Remaining Schar operator parity work should inspect flux
discretization, CM1 budget-term convention and density normalization in
`advu`/`advw`, plus the already-large pressure-gradient and buoyancy budget
differences. This is diagnostic-only evidence and does not satisfy the
production validation gate.

The CM1 velocity-form budget convention was then tested. CM1's `ud_vadv`
contains a mass-divergence correction, `u * ∂z(rrw)`, before density/metric
normalization. Breeze diagnostic rows were added:

```text
ub_vadv_velocity_form
wb_vadv_velocity_form
validation_output/substepper/schar_2s_weno9_operator_budget_velocity_form_vs_cm1_summary.md
```

At `2 s`, WENO-9, below sponge:

- `ub_vadv` relative L2 `126.9171154`;
- `ub_vadv_velocity_form` relative L2 `14.14802568`;
- `wb_vadv` relative L2 `2.013449596`;
- `wb_vadv_velocity_form` relative L2 `2.546511647`.

Interpretation: conservative-vs-velocity-form budget convention explains much
of the `ub_vadv` amplitude discrepancy, but the corrected row still fails by a
large margin and has poor pattern correlation. This remains diagnostic-only and
does not satisfy the production validation gate.

The CM1 `gzu / rho0_face` normalization was then tested explicitly:

```text
ub_vadv_velocity_form_cm1_norm
validation_output/substepper/schar_2s_weno9_operator_budget_cm1_norm_vs_cm1_summary.md
```

At `2 s`, WENO-9, below sponge:

- `ub_vadv_velocity_form` relative L2 `14.14802568`;
- `ub_vadv_velocity_form_cm1_norm` relative L2 `14.18970804`.

Interpretation: CM1-style normalization does not close the remaining
`ub_vadv` gap. The remaining advection mismatch is likely in the exact
`vadv_flx9` interpolation or CM1 diagnostic-budget convention, while
pressure-gradient and buoyancy remain larger Schar blockers.

A conservative RK split-increment diagnostic was also tested for the
`400 x 200`, `2 s`, `dt = 0.1 s`, CM1-terrain/CM1-constants explicit setup:

```text
validation_output/substepper/terrain_schar_2s_400x200_explicit_dt0p1_cm1terrain_cm1constants_rk_split_conservative_diagnostic_cpu/
validation_output/substepper/schar_2s_breeze_cm1_rk_split_conservative_increment_budget_summary.md
```

This does not satisfy any completion criterion. Breeze RK split total vs
Breeze actual face-`u` increment now closes to relative L2
`1.850618453e-13`, but the internally valid pressure split still fails against
CM1 `ub_pgrad` at relative L2 `1.339146674`, and the total remains the same
failed Breeze-vs-CM1 face-`u` increment comparison. A matched outside-stencil
control also closes Breeze's own update but gives pressure relative L2
`1.341576719` against CM1 `ub_pgrad`, so the pressure-gradient stencil variant
is not the explanation.

An instrumented CM1 acoustic-PGF run was then completed:

```text
validation_output/substepper/cm1_schar_400x200_periodic_theta300_budget_2s_acoustic_pgrad_instrumented/
validation_output/substepper/cm1_schar_acoustic_pgrad_increment_validation.md
```

The accumulated CM1 acoustic `ppd` reconstruction of emitted `ub_pgrad` has
relative L2 `9.707927905e-6`, normalized RMSE `1.538444792e-7`, and pattern
correlation `0.999999999953`. This misses the strict predeclared `1e-6`
self-validation threshold but leaves max absolute residual `8.614479157e-7`.
The component split shows the acoustic `ppd` term dominates the CM1 first-step
`u` pressure budget; the terrain modification RMS is only `1.390774279e-7`.
This practical closure is good enough for the short discriminator chain, but
the artifact remains diagnostic-only and does not satisfy any production-run
completion gate.

A direct Breeze pressure-split comparison against the instrumented CM1
components was then run:

```text
validation_output/substepper/schar_2s_breeze_rk_pressure_vs_cm1_acoustic_components_summary.md
```

Breeze RK pressure vs CM1 acoustic `ppd` has relative L2 `1.339147038` and
pattern correlation `0.6162401561`, indistinguishable from the comparison
against CM1 emitted `ub_pgrad` (`relative_l2 = 1.339146674`). The Schär
pressure blocker is therefore in the acoustic-PGF channel itself, not in CM1's
terrain-modification labeling or Breeze's pressure/nonpressure bookkeeping.

The active Breeze acoustic substepper path was then instrumented for a matching
`2 s`, `400 x 200`, CM1-terrain/CM1-constants diagnostic with
`forward_weight = 0.60` and divergence damping disabled:

```text
validation_output/substepper/terrain_schar_2s_400x200_substepper_fw0p60_nodamp_cm1terrain_cm1constants_acoustic_pressure_budget_cpu/
validation_output/substepper/schar_2s_breeze_substep_pressure_vs_cm1_acoustic_components_summary.md
```

The diagnostic accumulates the final acoustic RK stage's horizontal pressure
increment. Breeze substepper pressure vs CM1 acoustic `ppd` has relative L2
`1.315442204`, normalized RMSE `0.02084610066`, and pattern correlation
`0.6215743620`. Component comparisons show that neither Breeze piece maps
directly to CM1 acoustic `ppd`: frozen pressure alone has relative L2
`1.355279402`, while perturbation pressure alone has relative L2
`1.143469936` and negative pattern correlation `-0.7212579097`. This remains
diagnostic-only and does not satisfy the production validation contract.
