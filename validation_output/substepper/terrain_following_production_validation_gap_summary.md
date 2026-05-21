# Terrain-Following Production Validation Gap Summary

Status: quantitative validation generated; acceptance incomplete.

Latest gate:

```text
production validation gate: pass=16 present=21 fail=21 missing=0 blocked=5
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
  substepper-vs-explicit discriminator:
  - `w_relative_l2_error = 0.12321649479560344`
  - `w_relative_linf_error = 0.07458038498963351`
  - `w_normalized_rmse = 0.015968489820710877`
  - `w_pattern_correlation = 0.9923917588419416`
  - `w_projection_amplitude_error = 0.01798208545593083`
  - `pressure_relative_l2_error = 0.6617009969437023`
  - `mountain_drag_relative_error = 0.4057695816160268`
  - coordinate parity passes exactly, but the 1% Tier-1 gate still fails.
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
- No accepted production spin-up/averaging window is declared.
- No accepted explicit-feasible window is declared beyond the measured
  instability bracket: 5 s completes but fails vertical velocity; 6.5 s is near
  blow-up; 20 s and 60 s fail in the explicit thermodynamic state.

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
