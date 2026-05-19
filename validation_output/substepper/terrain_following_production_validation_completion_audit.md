# Terrain-Following Production Validation Completion Audit

Status: incomplete as of 2026-05-19 13:45 UTC.

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
| Complex mountain production manifest | `complex_mountain_doernbrack_production_manifest.md` declares 120x120x150 Doernbrack, 6 h, matched CM1/Breeze grid | pass |
| Complex mountain Breeze substepper production artifact | `complex_mountain_production_substepper/complex_mountain_state_slice.csv` and metrics/summary/time series | pass |
| Complex mountain Breeze explicit production artifact | `complex_mountain_production_explicit/complex_mountain_state_slice.csv`, metrics, summary, time series, and snapshots; run reached 21600 s | pass |
| Complex mountain CM1 reference artifact | `complex_mountain_production_cm1_reference/complex_mountain_state_slice.csv` and metrics converted from `cm1out_000037.nc` at `t = 21600 s` | pass |
| Complex mountain explicit-vs-CM1 metrics | `complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv`; below-sponge `w_relative_l2_error = 3.777`, pressure `relative_l2_error = 7.605`, drag signs do not match | fail |
| Complex mountain substepper-vs-explicit metrics | `complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv`; below-sponge `w_relative_l2_error = 0.764`, pressure `relative_l2_error = 0.827`, drag-x relative error `0.722` | fail |
| Complex mountain production movie | `complex_mountain_cm1_vs_breeze_substepper_movie/complex_mountain_cm1_vs_breeze_substepper.mp4`, 37 frames from 6 h production outputs | pass |
| Askervein production LES metrics/plot | `askervein_les_production/` exists and is labeled production_validation, but it is only a 1 s artifact without declared spin-up/averaging | present |
| Askervein accepted explicit feasible-window metrics | `askervein_explicit_substepper_compare_production/` exists, but the summary warns that the default configuration is diagnostic unless explicitly rerun with production resolution/runtime/class. Vertical velocity and `w_tilde` also fail 1%. | blocked |
| Askervein WEMEP named-mast reference diagnostic | `askervein_wemep_reference/askervein_les_production_askervein_wemep_mast_metrics.csv`; matched RS/HT/CP, maximum FSR absolute error `0.387`, diagnostic only because Breeze artifact is idealized | present |
| Askervein accepted external reference / spin-up / averaging window | WEMEP/Zenodo reference files downloaded under `askervein_wemep_reference/`; `askervein_coordinate_faithful_production_manifest.md` records the WEMEP/ERF target geometry and reference scope; diagnostic ERF-terrain and WEMEP-mast plumbing smokes exist, including `96 x 72 x 32` explicit-vs-substepper diagnostics passing all 1% rows through `1.2 s` across full, near-terrain, centerline, lee-side, and hilltop regions and failing by `1.25 s` on `w_tilde` relative L∞, but production boundary conditions, spin-up/averaging, and accepted explicit feasible window remain undeclared | blocked |
| Required robustness metric schema | `terrain_following_production_validation_metric_schema_audit.md`; completed Schar and complex-mountain Breeze production metrics/time series now contain direct robustness fields, including CFL, acoustic CFL, finite-value counts, mass drift, bottom-normal velocity, high-k near-terrain energy, reflection fraction, and walltime-per-step | pass |

## Current Gate

Latest command:

```sh
julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl
```

Latest result:

```text
production validation gate: pass=16 present=16 fail=14 missing=0 blocked=5
```

The gate remains incomplete. Completion is blocked by:

- measured 1% accuracy failures in Schar and complex mountain;
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
regions, while `1.25 s` fails on `w_tilde` relative L∞. However, the validation
workflow still lacks production boundary conditions, declared production
spin-up/averaging window, and an accepted explicit-feasible production window.

The robustness metric schema for completed Schar and complex-mountain
production runs is now direct rather than proxy-derived, and the finite-value /
bottom-normal-velocity rows pass for those completed runs. Plan-required
saved-time reference comparison coverage now exists for the completed Schar and
complex-mountain artifacts, but those saved-time explicit-vs-CM1 rows fail the
1% gate. The remaining blockers are accuracy failures plus the unresolved
Askervein production definition/window.
