# Terrain-Following Production Validation Gap Summary

Status: quantitative validation generated; acceptance incomplete.

Latest gate:

```text
production validation gate: pass=16 present=23 fail=26 missing=0 blocked=5
```

Gate report:

- `validation_output/substepper/terrain_following_production_validation_gate_report.md`
- `validation_output/substepper/terrain_following_production_validation_gate_report.csv`

Time-resolved comparison audit:

- `validation_output/substepper/terrain_following_time_resolved_comparison_audit.md`

Completion audit:

- `validation_output/substepper/terrain_following_production_validation_completion_audit.md`

## Schar 6 h 400x200

Production artifacts exist for CM1 explicit, Breeze explicit, Breeze
substepper, production plots/movie, and 6 h three-resolution convergence.

Primary failures:

- explicit-vs-CM1 below-sponge:
  - `u_relative_l2_error = 0.026636791805590348`
  - `u_relative_linf_error = 0.16389643821208222`
  - `w_relative_l2_error = 1.7549539691829341`
  - `w_normalized_rmse = 0.11204348858250109`
  - `w_pattern_correlation = 0.43100073886205914`
  - `mountain_drag_relative_error = 3.3597789530483224`
- substepper-vs-explicit below-sponge:
  - `u_relative_l2_error = 0.004411935075567736`
  - `u_relative_linf_error = 0.015416752531768444`
  - `w_relative_l2_error = 0.22638797656312776`
  - `w_normalized_rmse = 0.029339205804194135`
  - `w_pattern_correlation = 0.9754128236898442`
  - `mountain_drag_relative_error = 0.6269783052655318`
- matched `dt = 0.35 s`, no-damping, no-upper-sponge
  substepper-vs-explicit discriminator, current-branch refresh in
  `schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/`:
  - `w_relative_l2_error = 0.12321649479555473`
  - `w_relative_linf_error = 0.07458038500593407`
  - `w_normalized_rmse = 0.01596848982070456`
  - `w_pattern_correlation = 0.9923917588419493`
  - `w_projection_amplitude_error = 0.01798208545616331`
  - `pressure_relative_l2_error = 0.6617009969436629`
  - `mountain_drag_relative_error = 0.405769581615587`
  - coordinate parity passes exactly, but the 1% Tier-1 gate still fails.
- previous-horizontal-divergence matched `dt = 0.35 s`, no-damping,
  no-upper-sponge discriminator in
  `schar_substepper_vs_explicit_tier1_6h_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/`:
  - `w_relative_l2_error = 0.12321649479541662`
  - `w_relative_linf_error = 0.0745803849951862`
  - `w_normalized_rmse = 0.01596848982068666`
  - `w_pattern_correlation = 0.9923917588419646`
  - `w_projection_amplitude_error = 0.01798208545535651`
  - `pressure_relative_l2_error = 0.6617009969440094`
  - `mountain_drag_relative_error = 0.4057695816151736`
  - coordinate parity passes exactly, but the result is effectively identical
    to the matched-`dt` baseline and fails the 1% Tier-1 gate.
- forward-weight `0.60` matched `dt = 0.35 s`, no-damping, no-upper-sponge
  discriminator in
  `schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_forward0p6_grid/`:
  - `w_relative_l2_error = 0.1150234361357348`
  - `w_relative_linf_error = 0.07021769106471422`
  - `w_normalized_rmse = 0.014906693881558191`
  - `w_pattern_correlation = 0.9933650234561142`
  - `w_projection_amplitude_error = 0.014951846662962565`
  - `pressure_relative_l2_error = 0.6416904610778428`
  - `mountain_drag_relative_error = 0.3984645180251729`
  - coordinate parity passes exactly, but the 1% Tier-1 gate still fails.
- first-substep-PGF matched `dt = 0.35 s`, no-damping, no-upper-sponge
  discriminator in
  `schar_substepper_vs_explicit_tier1_6h_dt0p35_first_substep_pgf_no_damping_no_upper_sponge_grid/`:
  - `w_relative_l2_error = 0.0994969833538295`
  - `w_relative_linf_error = 0.06057553130893131`
  - `w_normalized_rmse = 0.012894511960534663`
  - `w_pattern_correlation = 0.9950837531073307`
  - `w_projection_amplitude_error = 0.018036852913514867`
  - `pressure_relative_l2_error = 0.39806713459847487`
  - `mountain_drag_relative_error = 0.12327601804709179`
  - coordinate parity passes exactly, and pressure/drag improve materially,
    but the 1% Tier-1 gate still fails.
- first-substep-PGF plus forward-weight `0.60` matched `dt = 0.35 s`,
  no-damping, no-upper-sponge discriminator in
  `schar_substepper_vs_explicit_tier1_6h_dt0p35_first_substep_pgf_forward0p6_no_damping_no_upper_sponge_grid/`:
  - `w_relative_l2_error = 0.086302493892431`
  - `w_relative_linf_error = 0.05385009410824209`
  - `w_normalized_rmse = 0.01118454552297831`
  - `w_pattern_correlation = 0.9963036281082229`
  - `w_projection_amplitude_error = 0.01504397990572448`
  - `pressure_relative_l2_error = 0.3343898584973857`
  - `mountain_drag_relative_error = 0.09062794054993754`
  - coordinate parity passes exactly, and this is the best Schär production
    discriminator so far, but the 1% Tier-1 gate still fails.
- projection-amplitude gates:
  - explicit-vs-CM1 below-sponge `projection_amplitude_error = 0.17606602967141627`
  - substepper-vs-explicit below-sponge `projection_amplitude_error = 0.09475538706521736`
  - explicit-vs-CM1 near-terrain `projection_amplitude_error = 0.17505443644277563`
  - substepper-vs-explicit near-terrain `projection_amplitude_error = 0.03938473645174334`
- near-terrain diagnostic:
  - explicit-vs-CM1 `w_relative_l2_error = 1.1949691854558966`
  - substepper-vs-explicit `w_relative_l2_error = 0.13382617379916242`
  - substepper-vs-explicit `u` passes the 1% field gate near terrain, but
    `w`, `θ`, pressure, and drag still fail.
- saved-time scalar substepper-vs-explicit diagnostic:
  - rows: `407`
  - failing 1% rows: `294`
  - worst relative error: `41.565943649924705` for `mountain_drag` at `10200 s`
- saved-time field substepper-vs-explicit diagnostic:
  - rows: `74`
  - failing 1% rows: `72`
  - worst relative L2 error: `0.9579903914564901` for
    `pressure_perturbation` at `10798.200000006214 s`
- saved-time CM1-vs-Breeze raw-error movie frame diagnostic:
  - matched frames: `37`
  - maximum time offset: `0 s`
  - worst maximum absolute `w` error: `3.5978230` at `20400 s`
- saved-time CM1-vs-Breeze `w` and pressure frame diagnostic:
  - matched frames: `37`
  - maximum time offset: `0 s`
  - worst maximum absolute `w` error: `3.5884120` at `20400 s`
  - worst `w_relative_l2_error`: `2.8738358` at `21600 s`
  - worst maximum absolute pressure error: `328.80732` at `0 s`
  - worst `pressure_relative_l2_error`: `2.3224117e11` at `0 s`

Useful diagnostic note:

- `terrain_schar_prognostic_sponge_diagnosis.md` shows CM1-like prognostic
  sponge damping improves 6 h CM1 comparison but does not close the 1% gate.
- The refreshed 6 h outside-pressure-gradient-stencil discriminator on the
  current branch also does not close the CM1 gap. Explicit outside-PGF vs CM1
  below-sponge still reports `w_relative_l2_error = 1.085535602`,
  `w_normalized_rmse = 0.05739379042`, `w_pattern_correlation = 0.6390117184`,
  `p_relative_l2_error = 0.9876719330`, and
  `mountain_drag_relative_error = 1.720912066`. Outside-PGF vs inside-PGF
  Breeze explicit is much smaller (`w_relative_l2_error = 0.01539643305`,
  pressure relative L2 `0.003254700596`, drag best-convention error
  `0.001286571244`), so the stencil choice is not the Schär explicit-vs-CM1
  closure path.

## Complex Mountain 6 h 120x120x150

Production artifacts now exist for CM1 explicit, Breeze explicit, Breeze
substepper, explicit-vs-CM1, substepper-vs-explicit, and a 37-frame movie.

Movie:

- `complex_mountain_cm1_vs_breeze_substepper_movie/complex_mountain_cm1_vs_breeze_substepper.mp4`

Primary failures:

- explicit-vs-CM1 below-sponge:
  - `u_relative_l2_error = 0.11624580774050561`
  - `v_relative_l2_error = 1.6843374389067955`
  - `w_relative_l2_error = 3.7768447403611067`
  - `theta_relative_l2_error = 1.7658682588906576`
  - `pressure_relative_l2_error = 7.60472948572488`
  - pressure drag signs do not match CM1.
- substepper-vs-explicit below-sponge:
  - `u_relative_l2_error = 0.019977664187591452`
  - `v_relative_l2_error = 0.39023566036109475`
  - `w_relative_l2_error = 0.7640921528867319`
  - `theta_relative_l2_error = 0.22295525778125608`
  - `pressure_relative_l2_error = 0.8266102318858068`
  - drag signs match, but drag magnitudes fail: x error `0.7218918533`,
    y error `0.8263035360`.
- projection-amplitude gates:
  - explicit-vs-CM1 below-sponge `projection_amplitude_error = 0.94392136146878`
  - explicit-vs-CM1 near-terrain `projection_amplitude_error = 1.1206564592811683`
  - substepper-vs-explicit below-sponge `projection_amplitude_error = 0.6706262595274035`
  - substepper-vs-explicit near-terrain `projection_amplitude_error = 0.38668445790027184`
- saved-time scalar substepper-vs-explicit diagnostic:
  - rows: `481`
  - failing 1% rows: `419`
  - worst relative error: `14.85330164373352` for `pressure_drag_y` at `10800 s`
- saved-time field substepper-vs-explicit diagnostic:
  - rows: `148`
  - failing 1% rows: `144`
  - worst relative L2 error: `1.0313818112185893` for
    `pressure_perturbation` at `18000 s`
- saved-time CM1-vs-Breeze frame diagnostic:
  - matched frames: `37`
  - maximum time offset: `0 s`
  - worst maximum absolute `w` error: `1.4010414` at `21600 s`
  - worst maximum absolute pressure error: `122.86421` at `0 s`
  - worst `w_relative_l2_error`: `12.012647` at frame `3`
  - worst `pressure_relative_l2_error`: `2.0827813e11` at `0 s`

Interpretation:

- The complex case confirms the split-explicit gap is not Schar-specific.
- The explicit-vs-CM1 gap is larger than the substepper-vs-explicit gap, so
  there are both cross-model setup/physics/stencil differences and Breeze
  substepping differences to separate.

## Askervein

Current Breeze artifacts:

- `askervein_les_production/` exists but is only a 1 s idealized artifact.
- `askervein_explicit_substepper_compare_production/` exists but vertical
  velocity and `w_tilde` fail 1%.

Reference data:

- WEMEP/Zenodo files are downloaded under `askervein_wemep_reference/`.
- `askervein_wemep_reference_manifest.md` records source and scope.
- `askervein_coordinate_faithful_production_manifest.md` records the local
  WEMEP/ERF target setup: `6 km x 6 km`, `300 x 300 x 18`, `20 m`
  horizontal spacing, `257 x 257` terrain file, and `72` validation rows.
- Named-mast diagnostic:
  `askervein_wemep_reference/askervein_les_production_askervein_wemep_mast_metrics.csv`

Named-mast diagnostic result:

- matched masts: `RS`, `HT`, `CP`
- maximum fractional-speed-up absolute error: `0.38745059611643906`
- one-percent pass: `false`

Remaining blocker:

- Breeze now has diagnostic plumbing for ERF terrain and WEMEP mast parsing,
  but does not yet run a production coordinate-faithful WEMEP Askervein case
  with the required boundary conditions and averaging workflow.
- ERF-terrain `96 x 72 x 32` explicit-vs-substepper diagnostics pass all 1%
  rows through `1.2 s`; the largest full-domain relative L2 error at `1.2 s`
  is `0.00205` for `w`. The refreshed `1.2 s` diagnostic includes
  `full_domain`, `near_terrain`, `centerline_slice`, `lee_side_box`, and
  `hilltop_box`; the worst relative L∞ row is full-domain `w_tilde = 0.00970`.
  The strict gate fails by `1.25 s` on
  `w_tilde_relative_linf_error = 0.01009`, and the `5 s` diagnostic fails with
  `w_relative_linf_error = 0.02677` and `w_tilde_relative_linf_error =
  0.03680`. This is a useful early-time bracket but not a production runtime.
- The refreshed `300 x 300 x 18`, `1.2 s` ERF-terrain GPU diagnostic confirms
  the target horizontal-resolution path still runs. Full-domain `w` narrowly
  misses strict 1% (`relative_l2_error = 0.010377`, `relative_linf_error =
  0.009760`), full-domain `w_tilde` misses (`relative_l2_error = 0.011571`,
  `relative_linf_error = 0.012351`), and near-terrain/hilltop `w` passes while
  `w_tilde` remains just above the strict Linf gate. This is diagnostic only
  because it is a `1.2 s` explicit window.
- No accepted production spin-up/averaging window is declared.
- No accepted explicit-feasible window is declared beyond the measured
  instability bracket: 5 s completes but fails vertical velocity; 6.5 s is near
  blow-up; 20 s and 60 s fail in the explicit thermodynamic state.
- A 2026-05-22 CPU refresh of the nominal `96 x 72 x 32`, `60 s`
  explicit-vs-substepper production-window artifact failed in the explicit half
  at step `134`, `t = 6.7 s`, in `temperature_and_pressure` after the
  thermodynamic state became invalid. The existing
  `askervein_explicit_substepper_compare_production/` files are therefore still
  the old `1 s` artifact and do not satisfy the plan. A smaller-`dt` GPU
  feasibility diagnostic, job `1155`, reached `60 s` at `dt = 0.01 s` on the
  Gaussian comparison setup, but it remains diagnostic and fails vertical
  velocity badly: full-domain `w_relative_l2_error = 0.17003`,
  `w_tilde_relative_l2_error = 0.28331`, near-terrain
  `w_relative_l2_error = 0.09942`, and hilltop `w_relative_l2_error =
  0.03747`. This establishes a stable explicit-feasible diagnostic window, not
  an accepted production Askervein window.
- The same smaller-`dt` 60 s window was then run with ERF terrain at
  `96 x 72 x 32`. It also completed, so explicit feasibility is not the
  immediate blocker for this reduced ERF diagnostic. The vertical metrics still
  fail by large margins: full-domain `w_relative_l2_error = 0.19931`,
  `w_tilde_relative_l2_error = 0.27501`, near-terrain
  `w_relative_l2_error = 0.12165`, and hilltop `w_relative_l2_error =
  0.02568`. This reinforces the shared substepper-vs-explicit vertical-velocity
  blocker rather than resolving Askervein production validation.
- A pressure-gradient-forced Askervein bracket at `192 x 192 x 64`, `300 s`
  found that the high accelerations `0.04`, `0.06`, and `0.08 m s^-2` all
  saturate to `NaN` mast diagnostics: model RS speed and maximum FSR error are
  `NaN` in each `1156` artifact. This rules out those forcing magnitudes as
  validation candidates and reframes them as a stability-threshold diagnostic.
- The pressure-gradient diagnostics that were active at `2026-05-22T01:41Z`
  have finished or been superseded: `1158` completed the lower
  `0.025`/`0.03`/`0.035 m s^-2` bracket with non-finite mast diagnostics, and
  `1159` failed during movie-frame writing before metrics were emitted. They
  remain diagnostic-only and are superseded by the capped retries below.
- The first lower-bracket point, `a = 0.025 m s^-2`, also saturated: job
  `1158` wrote finite metadata but `reference_speed_model = NaN` and
  `max_abs_fsr_error = NaN` with `19` samples over `300 s`. This pushes the
  finite-response threshold for the `192 x 192 x 64`, `300 s` setup below
  `0.025 m s^-2` or indicates that the forcing/spin-up strategy needs revision.
- The next lower-bracket point, `a = 0.03 m s^-2`, also saturated with
  `reference_speed_model = NaN` and `max_abs_fsr_error = NaN` after `19`
  samples over `300 s`. Thus `0.025` and `0.03 m s^-2` are both unusable
  validation forcings at `192 x 192 x 64`.
- The full lower bracket is now complete. The final `a = 0.035 m s^-2` point
  also saturated with `reference_speed_model = NaN` and `max_abs_fsr_error =
  NaN` after `19` samples over `300 s`. All tested `192 x 192 x 64`
  pressure-gradient accelerations from `0.025` through `0.08 m s^-2` are
  diagnostic failures.
- The first long finite-response attempt, job `1159`, failed in diagnostic
  frame writing before metrics were emitted because the frame writer attempted
  to convert `NaN` to `UInt8`. The validation script now renders non-finite
  frame values as magenta and ignores them when choosing the color limit.
  The same long diagnostic was resubmitted as job `1164`; it passed the
  previous frame-writing failure point and wrote full-size frame artifacts, but
  still produced non-finite mast metrics.
- The resubmitted long `a = 0.02 m s^-2` point completed at `256 x 256 x 96`
  and `2400 s`, but still produced `reference_speed_model = NaN` and
  `max_abs_fsr_error = NaN` after `26` samples over a `1395 s` averaging
  window. It wrote `24` full-size `w_tilde_xz` frames, so this is no longer an
  artifact-writing problem; the longer pressure-gradient Askervein setup itself
  remains non-finite in the mast diagnostics.
- Because the uncapped run used `dt_mean = 0.257 s` and `dt_max = 0.262 s`,
  while the earlier finite `a = 0.02`, `900 s` precedent used `max_dt =
  0.05 s`, the uncapped `1164` job was cancelled before its `a = 0.025` case
  completed. The long batch now sets `ASKER_CASE_MAX_DT=0.05`,
  `ASKER_CASE_DT_UPDATE_INTERVAL=10`, and `ASKER_CASE_SLICE_LIMIT=0.75`; the
  capped retry is job `1165`.
- To use available parallel GPU capacity while `1165` runs, job `1166`
  tested capped `a = 0.02 m s^-2` on `gpu-dev`. Its planned `a = 0.025`
  second leg was cancelled after the first summary because job `1169` already
  completed the same capped `a = 0.025 m s^-2` configuration. The single-leg
  capped `a = 0.025 m s^-2` diagnostic was relaunched as job `1169` on
  `gpu-prod-st-gpu-prod-2` after equivalent job `1168` stayed in dynamic-node
  `CONFIGURING`.
- Job `1166` finished the exact capped low-grid `a = 0.02 m s^-2` point at
  `192 x 192 x 64`, `300 s`: `reference_speed_model =
  4.643822590370479`, `max_abs_fsr_error = 1.409831851518315`, and
  `118` samples. At fixed `a = 0.02`, this is roughly `2.06×` the long
  `256 x 256 x 96` RS speed but still fails the spatial mast metrics badly,
  reinforcing that the pressure-gradient response is strongly grid-dependent.
- Job `1169` finished at `192 x 192 x 64`, `300 s`, with the `0.05 s` max-dt
  cap and finite mast metrics: `reference_speed_model =
  4.935863244558616`, `max_abs_fsr_error = 1.4516840207284545`, and
  `118` samples. This confirms the cap fixes the low-grid `a = 0.025`
  NaN-mast failure mode, but the result remains diagnostic-only and inaccurate:
  `production_validation = false`, `production_average = false`, and the
  reference speed/FSR errors are far outside any accepted validation threshold.
- Job `1172` extended the capped low-grid bracket to `a = 0.045 m s^-2` and
  also stayed finite: `reference_speed_model = 5.989918904395469`,
  `max_abs_fsr_error = 1.5342592003584854`, and `118` samples. The higher
  acceleration improves RS speed but worsens the worst FSR error, so pressure
  gradient amplitude alone is not an accepted validation path.
- Job `1174` extended the capped low-grid bracket to `a = 0.10 m s^-2` and
  nearly matched the single RS reference speed (`reference_speed_model =
  8.30920752092283` versus `8.895`). The spatial comparison still fails badly:
  `max_abs_fsr_error = 1.3397411630772242`, and
  `max_abs_tke_error = 125.94100845453049`. This confirms that pressure
  gradient amplitude can tune one reference speed, but it does not solve the
  Askervein mast/transect validation mismatch.
- The long capped `256 x 256 x 96`, `2400 s`, `a = 0.02 m s^-2` retry
  completed as job `1165`'s first point with finite diagnostics and
  `121` `w_tilde_xz` frames. The cap fixed the full-window NaN failure
  (`dt_mean = 0.049967729174907875`, `dt_max = 0.05`), but the flow remains
  severely underdriven: `reference_speed_model = 2.256985683648537`,
  `max_abs_fsr_error = 6.555834500603011`, and `max_abs_tke_error =
  62.146123787742106`. Job `1165` is continuing with the capped
  `a = 0.025 m s^-2` long point.
- The long capped `256 x 256 x 96`, `2400 s`, `a = 0.025 m s^-2` point also
  completed with finite diagnostics and `121` `w_tilde_xz` frames. Increasing
  the long-run pressure-gradient acceleration from `0.02` to `0.025 m s^-2`
  only raises `reference_speed_model` from `2.256985683648537` to
  `2.5369400008976233` and worsens `max_abs_fsr_error` from
  `6.555834500603011` to `6.69566015642208`; `max_abs_tke_error` also worsens
  to `79.20465786249468`. Thus the capped long-run PG-amplitude bracket is
  finite but does not recover the Askervein field validation.
- The long capped `256 x 256 x 96`, `2400 s`, `a = 0.10 m s^-2` point
  completed as job `1186` with finite diagnostics and `121` `w_tilde_xz`
  frames. It raises `reference_speed_model` to `5.253831533722133 m s^-1`,
  still below the observed `8.895 m s^-1`, and worsens the spatial mast
  comparison: `max_abs_fsr_error = 10.139617785288996` and
  `max_abs_tke_error = 733.4619859674443`. This confirms that simply raising
  constant pressure-gradient forcing does not recover Askervein field
  validation at production-like resolution and averaging.

## Next Goal Inputs

The next goal should be a fix/debug goal, not a metrics-generation goal.
Suggested scope:

1. Separate explicit-vs-CM1 setup differences from Breeze dycore differences
   using Schar and complex mountain before tuning substepping.
2. Investigate the shared vertical-velocity/pressure/drag failures in
   substepper-vs-explicit for Schar, complex mountain, and Askervein.
3. Decide whether Askervein should become a coordinate-faithful WEMEP
   production case in this branch, or remain a diagnostic until the dycore
   mountain-wave failures are reduced.
