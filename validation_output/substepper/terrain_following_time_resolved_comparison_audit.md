# Terrain-Following Time-Resolved Comparison Audit

Status: partial coverage; validation incomplete.

The production validation plan requires every production run to emit a
time-series metrics file with comparison errors against reference at every saved
reference time. Final-state comparison metrics are not sufficient.

Saved-time substepper-vs-explicit field metrics were generated from existing
production snapshots with:

- `validation_output/substepper/substepper_explicit_field_timeseries_metrics.jl`

## Current Coverage

| Case | Pair | Saved-time evidence | Status |
|---|---|---|---:|
| Schar | substepper vs Breeze explicit | `schar_6h_400x200_substepper_vs_explicit_timeseries_metrics.csv` has saved-time scalar comparison metrics. | present |
| Schar | substepper vs Breeze explicit | `schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics.csv` has saved-time `w` and pressure perturbation field metrics from production field snapshots. | present |
| Schar | Breeze substepper vs CM1 explicit | `schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie/frame_pairs.csv` and `schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie/frame_pairs.csv` have 37 matched saved frames and `w`/pressure error metrics. | present |
| Schar | Breeze explicit vs CM1 explicit | `schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv` has 37 matched saved frames, `w` and pressure rows, and 74 rows. The metrics fail the 1% gate for all rows, but the saved-time coverage is now present. | present |
| Complex mountain | substepper vs Breeze explicit | `complex_mountain_substepper_vs_explicit_timeseries_metrics.csv` has saved-time scalar comparison metrics. | present |
| Complex mountain | substepper vs Breeze explicit | `complex_mountain_substepper_vs_explicit_field_timeseries_metrics.csv` has saved-time centerline-slice field metrics from production snapshots. | present |
| Complex mountain | Breeze substepper vs CM1 explicit | `complex_mountain_cm1_vs_breeze_substepper_movie/frame_pairs.csv` has 37 matched saved frames and `w`/pressure error metrics. | present |
| Complex mountain | Breeze explicit vs CM1 explicit | `complex_mountain_explicit_vs_cm1_timeseries_metrics.csv` has 37 matched saved frames, 4 centerline-slice fields, and 148 rows. The metrics fail the 1% gate for all rows, but the saved-time coverage is now present. | present |
| Askervein | production LES vs WEMEP/reference | No coordinate-faithful production LES exists, so saved-time reference comparison metrics are blocked. | blocked |
| Askervein | explicit-window substepper vs explicit | ERF-terrain short-window diagnostic brackets exist, but no accepted production explicit-feasible window exists. | blocked |

## Required Follow-Up

- Schar saved-time explicit-vs-CM1 metrics are now present in
  `schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv`, generated from
  the GPU field-snapshot rerun
  `terrain_schar_6h_400x200_production_explicit_field_snapshots/`.
- Complex-mountain saved-time explicit-vs-CM1 centerline-slice metrics are now
  present in `complex_mountain_explicit_vs_cm1_timeseries_metrics.csv`. A
  fuller 3D saved-time comparison would require saved 3D Breeze fields, but the
  current production snapshots are centerline slices.
- Do not count Askervein saved-time comparison coverage until the
  coordinate-faithful production run and accepted explicit-feasible window are
  defined.

The saved-time comparison coverage now exists for the completed Schar and
complex mountain production artifacts. These metrics do not change the
accuracy conclusion: the saved-time explicit-vs-CM1 rows fail the 1% gate.
