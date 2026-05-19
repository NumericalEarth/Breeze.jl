# Terrain-Following Production Validation Completion Checklist

Status: incomplete as of the latest production gate run.

Authoritative plan:

- `validation_output/substepper/terrain_following_production_validation_metrics_plan.md`

Authoritative gate:

- command: `julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl`
- latest result: `pass=16 present=16 fail=14 missing=0 blocked=5`
- report: `validation_output/substepper/terrain_following_production_validation_gate_report.md`
- csv: `validation_output/substepper/terrain_following_production_validation_gate_report.csv`

## Objective Restatement

Generate quantitative production-validation evidence for terrain-following
compressible dynamics and acoustic substepping. Completion requires production
or validation-grade runs, not smoke tests, for Schär, a complex mountain case,
and Askervein Hill.

The required accuracy targets are:

1. Breeze explicit stepping must match CM1 explicit stepping to 1% for each
   production validation case with a CM1 reference.
2. Breeze acoustic substepping must match Breeze explicit stepping to 1% for
   each production validation case where explicit stepping is feasible.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---:|
| Schär CM1 explicit reference resolution matches the Breeze production grid | `cm1_schar_400x200_periodic_theta300_reference/cm1_config.txt` reports `nx=400`, `nz=200`, matching `terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_summary.txt` with `Nx, Nz = 400, 200`. | pass |
| Strict 1% Breeze explicit vs CM1 explicit comparison for Schär below sponge | `schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_production_1pct_metrics_state_metrics.csv` reports `u_l2=0.026636791805590348`, `u_linf=0.16389643821208222`, `w_l2=1.7549539691829341`, `w_linf=1.0840695503670112`, `w_rmse=0.11204348858250109`, `w_amp=0.046317958947867166`, `w_corr=0.43100073886205914`, `phase=1.499999999055035e-8`, `drag=3.3597789530483224`, and field pass flags false. | fail |
| Strict 1% Breeze substepper vs Breeze explicit comparison for Schär below sponge | `schar_6h_400x200_substepper_vs_explicit_production_1pct_metrics_state_metrics.csv` reports `u_l2=0.004411935075567736`, `u_linf=0.015416752531768444`, `w_l2=0.22638797656312776`, `w_linf=0.1415003325381681`, `w_rmse=0.029339205804194135`, `w_amp=0.00025591835720307815`, `w_corr=0.9754128236898442`, `phase=0.0`, `drag=0.6269783052655318`, and field pass flags false. | fail |
| Strict 1% Breeze explicit vs CM1 projection-amplitude comparison for Schär below sponge | `schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_projection_metrics.csv` reports below-sponge `projection_amplitude_error=0.17606602967141627`, near-terrain `projection_amplitude_error=0.17505443644277563`, `best_shift_cells=0`, and below-sponge `best_shift_projection_amplitude_error=0.17606602967141627`. | fail |
| Strict 1% Breeze substepper vs Breeze explicit projection-amplitude comparison for Schär below sponge | `schar_6h_400x200_substepper_vs_explicit_projection_metrics.csv` reports below-sponge `projection_amplitude_error=0.09475538706521736`, near-terrain `projection_amplitude_error=0.03938473645174334`, `best_shift_cells=0`, and below-sponge `best_shift_projection_amplitude_error=0.09475538706521736`. | fail |
| Schär matched Breeze pair failure is diagnosed by region and time series | `terrain_schar_400x200_substepper_vs_explicit_diagnostics.csv` and `terrain_schar_400x200_substepper_vs_explicit_diagnosis.md` report that `w` amplitude and phase are close, but below-sponge `w_relative_l2_error=0.226387976563`, `p_relative_l2_error=0.9198464428`, drag relative error is `0.626978305267`, and final below-sponge energy relative error is `0.148336011227`. | present |
| Schär production runtime is long enough | 400x200 Schär artifacts reach 6 h, the minimum production validation time. | pass |
| Schär and complex-mountain production time-series cadence is sufficient | Schär explicit, Schär substepper, complex explicit, and complex substepper time-series all reach `21600 s`; maximum saved-time gap is no larger than `600 s` within tolerance. | pass |
| Schär and complex-mountain production time-series metric contract is sufficient | The production gate now checks that Schär explicit, Schär substepper, complex explicit, and complex substepper time-series include `time`, `maximum_w`, `maximum_u`, `maximum_pressure_perturbation`, `domain_kinetic_energy`, `mass_relative_drift`, `maximum_cfl`, `maximum_acoustic_cfl`, `high_k_energy_fraction_near_terrain`, `reflection_energy_fraction_above_sponge_start`, plus applicable drag metrics (`mountain_drag` for Schär and `pressure_drag_x/y` for complex mountain). | pass |
| Schär production-grid explicit artifact exists | `terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_summary.txt`, `terrain_schar_mountain_wave_state_slice.csv`, and `terrain_schar_mountain_wave_energy_timeseries.csv`. | pass |
| Schär production-grid substepper artifact exists | `terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_summary.txt`, `terrain_schar_mountain_wave_state_slice.csv`, and `terrain_schar_mountain_wave_energy_timeseries.csv`. | pass |
| Schär human-inspectable production plot or movie exists | `terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_w_comparison.ppm`, `terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_w_comparison.ppm`, `schar_cm1_vs_breeze_substepper_movie/schar_cm1_vs_breeze_substepper_side_by_side.mp4`, `schar_cm1_periodic_vs_breeze_substepper_movie/schar_cm1_vs_breeze_substepper_side_by_side.mp4`, `schar_cm1_periodic_theta300_vs_breeze_substepper_movie/schar_cm1_vs_breeze_substepper_side_by_side.mp4`, `schar_cm1_periodic_theta300_vs_breeze_substepper_rendered_error_movie/schar_cm1_vs_breeze_substepper_rendered_error_side_by_side.mp4`, `schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie/schar_cm1_vs_breeze_substepper_raw_error.mp4`, and `schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie/schar_cm1_vs_breeze_substepper_field_error.mp4`. The field-error movie uses 6 h production-cadence saved `w` and pressure perturbation fields; strict pass/fail still comes from CSV metrics. | pass |
| Schär three-resolution production convergence campaign exists | `terrain_schar_6h_substepper_convergence_production/terrain_schar_grid_convergence_summary.txt`, `terrain_schar_grid_convergence_metrics.csv`, and `terrain_schar_grid_convergence_state_metrics.csv`. | pass |
| Schär full-domain diagnostic metrics are present | `schar_6h_400x200_explicit_vs_cm1_400x200_full_domain_production_1pct_metrics_state_metrics.csv` and `schar_6h_400x200_substepper_vs_explicit_full_domain_production_1pct_metrics_state_metrics.csv`. These are diagnostic only and do not satisfy the below-sponge 1% gate. | present |
| Schär near-terrain diagnostic metrics are present | `schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_near_terrain_production_1pct_metrics_state_metrics.csv` reports explicit-vs-CM1 `w_l2=1.1949691854558966`; `schar_6h_400x200_substepper_vs_explicit_near_terrain_production_1pct_metrics_state_metrics.csv` reports substepper-vs-explicit `w_l2=0.13382617379916242`. | present |
| Schär saved-time scalar substepper-vs-explicit comparison metrics are present | `schar_6h_400x200_substepper_vs_explicit_timeseries_metrics.csv` has `407` rows, `294` failing 1% rows, and worst relative error `41.565943649924705` for `mountain_drag` at `10200 s`. | present |
| Schär saved-time field substepper-vs-explicit comparison metrics are present | `schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics.csv` has `74` rows over saved production field snapshots. `72` rows fail the 1% gate; worst `relative_l2_error = 0.9579903914564901` for `pressure_perturbation` at `10798.200000006214 s`. | fail |
| Schär saved-time CM1-vs-Breeze raw-error frame metrics are present | `schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie/frame_pairs.csv` has `37` matched frames, zero time offset, and worst maximum absolute `w` error `3.5978230` at CM1 time `20400 s`. | present |
| Schär saved-time CM1-vs-Breeze `w` and pressure frame error metrics are present | `schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie/frame_pairs.csv` has `37` matched frames, zero time offset, worst maximum absolute `w` error `3.5884120` at CM1 time `20400 s`, worst `w_relative_l2_error=2.8738358` at `21600 s`, worst maximum absolute pressure error `328.80732` at `0 s`, and worst `pressure_relative_l2_error=2.3224117e11` at `0 s`. | present |
| Schär saved-time explicit-vs-CM1 comparison metrics are present | `schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv` has `74` rows over `37` matched frames and two fields. All rows fail the 1% gate; worst `relative_l2_error=2.3224117e11` for pressure at `0 s`. | present |
| Complex mountain terrain, grid, runtime, and CM1 reference source are specified before running | `complex_mountain_doernbrack_production_manifest.md` selects the CM1-native Doernbrack `itern=3` hill, declares the local CM1 executable, grid `120x120x150`, `6 h` runtime, periodic lateral boundaries matched to Breeze, comparison regions, required artifacts, and 1% pass criteria. | pass |
| Complex mountain production CM1, Breeze explicit, Breeze substepper, and comparison metrics exist | `complex_mountain_production_cm1_reference/complex_mountain_state_slice.csv`, `complex_mountain_production_explicit/complex_mountain_state_slice.csv`, `complex_mountain_production_substepper/complex_mountain_state_slice.csv`, `complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv`, and `complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv` are present. The comparison metrics fail the 1% accuracy gate. | pass |
| Complex mountain explicit-vs-CM1 1% comparison | `complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv` reports `w_relative_l2_error=3.7768447403611067`, `pressure_relative_l2_error=7.60472948572488`, and pressure-drag signs do not match CM1. | fail |
| Complex mountain substepper-vs-explicit 1% comparison | `complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv` reports `w_relative_l2_error=0.7640921516428109`, `pressure_relative_l2_error=0.8266102311331819`, and drag magnitudes fail despite matching signs. | fail |
| Complex mountain projection-amplitude comparisons | `complex_mountain_explicit_vs_cm1_projection_metrics.csv` reports below-sponge `projection_amplitude_error=0.94392136146878`; `complex_mountain_substepper_vs_explicit_projection_metrics.csv` reports below-sponge `projection_amplitude_error=0.6706262595274035`. | fail |
| Completed-case comparison metrics cover required regions | Schär comparison coverage is split across below-sponge, full-domain, and near-terrain production metric files. Complex mountain comparison CSVs include `below_sponge`, `full_domain`, `near_terrain`, `centerline_slice`, `lee_side_box`, and `hilltop_box` regions for both explicit-vs-CM1 and substepper-vs-explicit comparisons. | pass |
| Complex mountain saved-time scalar substepper-vs-explicit comparison metrics are present | `complex_mountain_substepper_vs_explicit_timeseries_metrics.csv` has `481` rows, `419` failing 1% rows, and worst relative error `14.85330164373352` for `pressure_drag_y` at `10800 s`. | present |
| Complex mountain saved-time field substepper-vs-explicit comparison metrics are present | `complex_mountain_substepper_vs_explicit_field_timeseries_metrics.csv` has `148` centerline-slice field rows over `37` saved production times. `144` rows fail the 1% gate; worst `relative_l2_error = 1.0313818112185893` for `pressure_perturbation` at `18000 s`. | fail |
| Complex mountain saved-time CM1-vs-Breeze frame error metrics are present | `complex_mountain_cm1_vs_breeze_substepper_movie/frame_pairs.csv` has `37` matched frames, zero time offset, worst maximum absolute `w` error `1.4010414` at `21600 s`, worst `w_relative_l2_error=12.012647` at frame `3`, worst maximum absolute pressure error `122.86421` at `0 s`, and worst `pressure_relative_l2_error=2.0827813e11` at `0 s`. | present |
| Complex mountain saved-time explicit-vs-CM1 comparison metrics are present | `complex_mountain_explicit_vs_cm1_timeseries_metrics.csv` has `148` rows over `37` matched frames and four centerline-slice fields. All rows fail the 1% gate; worst `relative_l2_error=2.0827813e11` for pressure at `0 s`. | present |
| Complex mountain production movies exist | `complex_mountain_cm1_vs_breeze_explicit_movie/complex_mountain_cm1_vs_breeze_substepper.mp4` and `complex_mountain_cm1_vs_breeze_substepper_movie/complex_mountain_cm1_vs_breeze_substepper.mp4`, each with 37 frames from 6 h production output. | pass |
| Askervein accepted production explicit-vs-substepper window exists and passes 1% metrics | `askervein_explicit_substepper_compare_production/askervein_explicit_substepper_summary.txt` and `askervein_explicit_substepper_metrics.csv` exist, but the summary warns that the configuration is diagnostic unless rerun with production resolution/runtime/class. The metrics also do not pass the 1% gate. | blocked |
| Askervein explicit-vs-substepper 1% failure is diagnosed by field and region | `askervein_explicit_substepper_1pct_failure_diagnosis.md` records that horizontal `u`, `v`, and `speed` pass the 1% gate in the 1 s production artifact, while `w` and `w_tilde` fail in both full-domain and near-terrain regions. Near terrain, `w_tilde_relative_l2_error=0.09595874686964813` and `w_tilde_relative_linf_error=0.14038778245263928`. Diagnostic probes show 5 s completes but still fails vertical velocity, 6.5 s is near blow-up with order-one errors, and 20 s / 60 s attempts fail in explicit `temperature_and_pressure`. | present |
| Askervein diagnostic ERF-terrain explicit-window brackets include required 3D regions | The refreshed `1.2 s` pass-side and `1.25 s` fail-side GPU diagnostics emit `full_domain`, `near_terrain`, `centerline_slice`, `lee_side_box`, and `hilltop_box`. | present |
| Askervein diagnostic substepper LES artifact exists | `askervein_les_production/askervein_les_summary.txt`, `askervein_les_metrics.csv`, and `askervein_les_speedup_w.ppm` exist, but the summary says accepted production reference data are still required before the artifact can satisfy the full Askervein LES validation contract. | present |
| Askervein production grid, stable explicit window, spin-up, averaging window, and accepted reference are specified | The ERF canonical Askervein setup exists at `/shared/home/greg/ERF/Exec/CanonicalTests/Real_Terrain/Askervein/`; `askervein_coordinate_faithful_production_manifest.md` records the `6 km x 6 km`, `300 x 300 x 18`, `20 m` target setup and WEMEP reference scope. Diagnostic ERF-terrain/WEMEP-mast plumbing smokes exist, including `96 x 72 x 32` LES and explicit-vs-substepper runs with `production_resolution = true`; explicit-vs-substepper diagnostics pass all 1% rows at `0.001 s`, `0.01 s`, `0.1 s`, `1 s`, and `1.2 s`, then fail by `1.25 s` on `w_tilde` relative L∞. Production boundary conditions, spin-up/averaging, runtime, and an accepted explicit-feasible window remain unresolved. | blocked |
| Askervein required production LES, reference-comparison, and explicit-window outputs exist | `terrain_following_production_validation_metric_schema_audit.md` documents that Askervein is missing production LES time series, production spin-up/averaging metrics, accepted all-transect observational/reference comparison, coordinate-faithful WEMEP transect/profile comparison files, and accepted explicit-window output. | blocked |
| Smoke or diagnostic artifacts are excluded from completion | The production gate counts only `pass` production-validation checks toward completion and labels `present`, `missing`, `fail`, and `blocked` separately. | pass |
| Final validation artifact manifest satisfies the plan contract | `terrain_following_production_validation_artifact_manifest.md` includes the required manifest schema and records Schär and complex-mountain production coverage, but Askervein final-manifest fields remain incomplete because no coordinate-faithful production run/window has been accepted. | blocked |
| Completed production artifacts have concrete manifest provenance | `terrain_following_production_validation_artifact_manifest.md` records concrete command-source or command, backend/machine, grid/runtime, metrics, summary, time-series, plot/movie, and status evidence for the completed Schär and complex-mountain production artifacts and comparison products. | pass |
| Final markdown summary reports 1% pass/fail status | `terrain_following_production_validation_1pct_summary.md` summarizes Schär, complex mountain, and Askervein against the explicit-vs-CM1, substepper-vs-explicit, saved-time, projection, and blocked-artifact portions of the active 1% contract. | pass |
| Production resolution adequacy is documented before accepting production artifacts | Schär now records the `400x200`, `dx = 500 m`, `dz = 150 m`, terrain wavelength, sponge-base, active-cell, and three-resolution-convergence rationale in `terrain_following_production_validation_artifact_manifest.md`. Complex mountain records the `120x120x150`, `dx = dy = 1000 m`, `dz = 200 m`, hill half-width, active-cell, and feasible-CM1-grid rationale in `complex_mountain_doernbrack_production_manifest.md`. `askervein_coordinate_faithful_production_manifest.md` records the ERF/WEMEP target setup and remaining production gaps, but no accepted production-resolution Askervein validation artifact exists. | blocked |
| Completed Schär and complex-mountain production runs have finite values and no bottom-normal leakage | The gate checks the production time series and finds `nan_count = 0`, `inf_count = 0`, and `bottom_normal_velocity_max_abs = 0` for Schär explicit, Schär substepper, complex explicit, and complex substepper. | pass |

## Current Blockers

The validation goal is not complete.

- Schär has long matched `400x200` production artifacts and a matched CM1
  reference, but final-state and saved-time required 1% comparisons still
  fail.
- Complex mountain has complete production artifacts, comparisons, and a
  regenerated movie, but final-state and saved-time required 1% comparisons
  still fail.
- Askervein has diagnostic artifacts with stale `production_validation` flags,
  but the explicit-vs-substepper summary itself warns that the default
  configuration is diagnostic. The current window is only `1 s`, the
  vertical-velocity and terrain-following contravariant-velocity metrics fail
  the 1% gate, and the accepted explicit-feasible window still has to be
  declared.
- Askervein full validation has WEMEP/ERF geometry, reference scope, and
  diagnostic terrain/mast parsing recorded, but still needs production boundary
  conditions plus declared spin-up and averaging windows.
- Askervein required output coverage remains incomplete: production LES time
  series, spin-up/averaging metrics, all-transect reference comparison,
  coordinate-faithful WEMEP transect/profile comparison, and accepted
  explicit-window output are still missing.
- Resolution adequacy is documented for the completed Schär and
  complex-mountain production cases, but remains blocked for Askervein because
  no accepted production-resolution validation artifact exists.
- Plan-required saved-time reference comparison coverage now exists for the
  completed Schar and complex-mountain artifacts, but the saved-time
  explicit-vs-CM1 rows fail the 1% gate.

## Next Required Inputs or Work

To continue productively, the next non-code input is the accepted Askervein
production definition: coordinate-faithful reference comparison, spin-up and
averaging windows, and an explicit-feasible comparison window. The local ERF
Askervein setup is available for geometry and forcing metadata.

The next code or physics work is to diagnose the existing Schär,
complex-mountain, and Askervein 1% failures. Those failures are measured; they
are not missing-artifact issues. For Schär, the reference-resolution issue is
now closed; the remaining issue is field/drag disagreement under the active 1%
production contract.
