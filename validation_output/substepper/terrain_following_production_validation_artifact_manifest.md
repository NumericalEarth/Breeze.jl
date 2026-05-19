# Terrain-Following Production Validation Artifact Manifest

Status: generated artifacts complete enough to quantify the current gaps;
validation acceptance is incomplete.

Git commit:

- `0022ac7acdc224a165ff4b63bcf096992b5303d7`
- `9b00422` is the current local HEAD used for the latest Askervein
  ERF-terrain GPU diagnostic brackets and the Schar field-snapshot movie
  artifact. Earlier Schar and complex-mountain artifacts were produced before
  this test-only commit; their artifact paths and gate evidence remain
  unchanged.

Latest gate:

```sh
julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl
```

Latest result:

```text
production validation gate: pass=16 present=16 fail=14 missing=0 blocked=5
```

Smoke and diagnostic artifacts are excluded from completion. Rows marked
`present` are useful evidence but do not satisfy the 1% production-validation
contract.

Metric schema audit:

- `validation_output/substepper/terrain_following_production_validation_metric_schema_audit.md`

Time-resolved comparison audit:

- `validation_output/substepper/terrain_following_time_resolved_comparison_audit.md`

Saved-time substepper-vs-explicit field metrics were generated from existing
production snapshots with:

- `validation_output/substepper/substepper_explicit_field_timeseries_metrics.jl`

The completed Schar and complex-mountain production Breeze artifacts now emit
the required direct robustness fields in their metrics and time-series files:
`maximum_cfl`, `maximum_acoustic_cfl`, `nan_count`, `inf_count`,
`mass_relative_drift`, `bottom_normal_velocity_max_abs`,
`high_k_energy_fraction_near_terrain`,
`reflection_energy_fraction_above_sponge_start`, and `walltime_per_step`.
The production time-series cadence gate confirms Schar explicit, Schar
substepper, complex explicit, and complex substepper all reach `21600 s` with
maximum saved-time gaps no larger than `600 s` within tolerance.
The finite-state/no-normal-flow robustness gate confirms those completed runs
have `nan_count = 0`, `inf_count = 0`, and
`bottom_normal_velocity_max_abs = 0`.
Askervein remains blocked because the current artifacts are diagnostic rather
than a coordinate-faithful production validation run.
Saved-time reference comparison coverage is now present for the completed
Schar and complex-mountain production artifacts. The saved-time metrics do not
pass the 1% accuracy gate; they expose the same explicit-vs-CM1 gaps as the
final-state metrics.

Legacy derived robustness metrics:

- `validation_output/substepper/derive_state_slice_robustness_metrics.jl`
- `validation_output/substepper/terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_derived_robustness_metrics.csv`
- `validation_output/substepper/terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_derived_robustness_metrics.csv`
- `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_derived_robustness_metrics.csv`
- `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_derived_robustness_metrics.csv`

These derived files were used before the direct production reruns. They recover
finite-value counts, high-k proxies, bottom-velocity proxies, walltime-per-step,
and simple CFL proxies from saved state slices, but the current gate relies on
the direct fields in the Schar and complex-mountain production outputs.

Retrofitted-script smoke checks:

- `validation_output/substepper/schar_robustness_smoke_explicit/terrain_schar_mountain_wave_metrics.csv`
- `validation_output/substepper/schar_robustness_smoke_substepper/terrain_schar_mountain_wave_metrics.csv`
- `validation_output/substepper/complex_mountain_robustness_smoke_substepper/complex_mountain_metrics.csv`

These smoke artifacts verify that the scripts now write `maximum_cfl`,
`maximum_acoustic_cfl`, `nan_count`, `inf_count`, `mass_relative_drift`, and
`bottom_normal_velocity_max_abs` directly. The production reruns now carry those
direct fields as validation evidence; the smoke artifacts remain diagnostic.

## Schar Mountain Wave

### CM1 Explicit Reference

- Artifact class: `production_validation`.
- Command source:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference.batch`
- Run log:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference-898.log`
- Backend/machine: CM1 reference generated locally, CPU-style CM1 executable.
- Grid/runtime: `nx = 400`, `nz = 200`, `dx = 500 m`, `dz = 150 m`, `6 h`.
- Reference path:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/`
- Config:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/cm1_config.txt`
- Metrics:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/external_schar_400x200_periodic_theta300_reference_metrics.csv`
- State slice:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/external_schar_400x200_periodic_theta300_reference_state_slice.csv`
- Raw final frame:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_reference/cm1out_000037.nc`
- Summary path: no standalone summary file; `cm1_config.txt`, the batch file,
  and the run log are the recorded provenance for this CM1 reference.
- Status: `pass` for artifact presence and matched-grid reference.

### Breeze Explicit

- Artifact class: `production_validation`.
- Command source:
  `validation_output/substepper/run_schar_400x200_explicit_dt0p35_production.batch`
- Backend/machine: Breeze CPU production run.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 0.35 s`, `6 h`.
- Metrics:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_metrics.csv`
- Summary:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_summary.txt`
- State slice:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_state_slice.csv`
- Time series:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_energy_timeseries.csv`
- Plot:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_w_comparison.ppm`
- Status: `pass` for artifact presence.

### Breeze Explicit Field-Snapshot Rerun

- Artifact class: `production_validation`.
- Purpose: production-cadence saved `w` and pressure perturbation fields for
  explicit-vs-CM1 saved-time metrics.
- Command:
  `SCHAR_ARCH=gpu SCHAR_OUTPUT_DIR=validation_output/substepper/terrain_schar_6h_400x200_production_explicit_field_snapshots SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=0.35 SCHAR_MAKE_MOVIE=true SCHAR_MOVIE_INTERVAL_SECONDS=600 SCHAR_WRITE_FIELD_SNAPSHOT_CSVS=true SCHAR_WRITE_ENERGY_TIMESERIES=false julia --project=test --color=no validation_output/substepper/terrain_schar_mountain_wave_explicit_validation.jl`
- Backend/machine: Breeze GPU production field-snapshot rerun on visible
  `NVIDIA H100 80GB HBM3`; `CUDA` loaded through the `test` project.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 0.35 s`, `6 h`, snapshots every
  `600 s`.
- Metrics:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit_field_snapshots/terrain_schar_mountain_wave_metrics.csv`
- Summary:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit_field_snapshots/terrain_schar_mountain_wave_summary.txt`
- Field snapshots:
  `validation_output/substepper/terrain_schar_6h_400x200_production_explicit_field_snapshots/terrain_schar_mountain_wave_field_snapshot_csvs/`
- Field-error movie:
  `validation_output/substepper/schar_cm1_periodic_theta300_vs_breeze_explicit_field_error_movie/schar_cm1_vs_breeze_substepper_field_error.mp4`
- Saved-time metrics:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv`
- Status: `pass` for saved-time coverage; strict 1% pass/fail remains attached
  to the comparison metrics.

### Breeze Acoustic Substepper

- Artifact class: `production_validation`.
- Command source:
  `validation_output/substepper/run_schar_400x200_production_validation.batch`
- Backend/machine: Breeze CPU production run.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 2.0 s`, `6 h`.
- Metrics:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_metrics.csv`
- Summary:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_summary.txt`
- State slice:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_state_slice.csv`
- Time series:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_energy_timeseries.csv`
- Plot:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_w_comparison.ppm`
- Status: `pass` for artifact presence.

### Breeze Acoustic Substepper Field-Snapshot Rerun

- Artifact class: `production_validation`.
- Purpose: production-cadence saved `w` and pressure perturbation fields for
  the CM1/Breeze/error movie required by the validation plan.
- Command:
  `SCHAR_ARCH=gpu SCHAR_OUTPUT_DIR=validation_output/substepper/terrain_schar_6h_400x200_production_substepper_field_snapshots SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=2 SCHAR_MAKE_MOVIE=true SCHAR_MOVIE_INTERVAL_SECONDS=600 SCHAR_WRITE_FIELD_SNAPSHOT_CSVS=true SCHAR_WRITE_W_SNAPSHOT_CSVS=false SCHAR_WRITE_ENERGY_TIMESERIES=false julia --project=test --color=no validation_output/substepper/terrain_schar_mountain_wave_validation.jl`
- Git commit: `9b00422`.
- Backend/machine: Breeze GPU production field-snapshot rerun on visible
  `NVIDIA H100 80GB HBM3`; `CUDA` loaded through the `test` project.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 2.0 s`, `6 h`, snapshots every
  `600 s`.
- Metrics:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper_field_snapshots/terrain_schar_mountain_wave_metrics.csv`
- Summary:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper_field_snapshots/terrain_schar_mountain_wave_summary.txt`
- State slice:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper_field_snapshots/terrain_schar_mountain_wave_state_slice.csv`
- Field snapshots:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper_field_snapshots/terrain_schar_mountain_wave_field_snapshot_csvs/`
- Snapshot cadence evidence:
  `37` field snapshots plus `snapshot_times.csv`, covering `0 s` through
  `21600 s` at `600 s` cadence.
- Plot:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper_field_snapshots/terrain_schar_mountain_wave_w_comparison.ppm`
- Status: `pass` for field-snapshot artifact presence; strict 1% pass/fail
  remains attached to the comparison metrics below.

### Schar Comparisons

- Resolution adequacy: the production/reference grid is `400x200` over
  `Lx = 200 km` and `Lz = 30 km`, giving `dx = 500 m` and `dz = 150 m`
  in both CM1 and Breeze. The Schar terrain uses `h0 = 250 m`, envelope
  half-width `a = 5 km`, and oscillatory wavelength `lambda = 4 km`; the
  grid therefore resolves the wavelength with 8 points per lambda and the
  envelope half-width with 10 points per a. The sponge base is `20 km`,
  leaving about 133 active vertical cells below the sponge. A separate
  three-resolution convergence campaign reaches the same `6 h` runtime.
- Explicit vs CM1:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_production_1pct_metrics_state_metrics.csv`
- Explicit vs CM1 summary:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_production_1pct_metrics_state_summary.txt`
- Substepper vs explicit:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_production_1pct_metrics_state_metrics.csv`
- Substepper vs explicit summary:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_production_1pct_metrics_state_summary.txt`
- Explicit vs CM1 projection metrics:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_projection_metrics.csv`
- Explicit vs CM1 saved-time metrics:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_timeseries_metrics.csv`
- Explicit vs CM1 saved-time summary:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_timeseries_metrics_summary.txt`
- Substepper vs explicit projection metrics:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_projection_metrics.csv`
- Substepper vs explicit saved-time scalar comparison metrics:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_timeseries_metrics.csv`
- Substepper vs explicit saved-time field comparison metrics:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics.csv`
- Substepper vs explicit saved-time field comparison summary:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_field_timeseries_metrics_summary.txt`
- Explicit vs CM1 near-terrain diagnostic metrics:
  `validation_output/substepper/schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_near_terrain_production_1pct_metrics_state_metrics.csv`
- Substepper vs explicit near-terrain diagnostic metrics:
  `validation_output/substepper/schar_6h_400x200_substepper_vs_explicit_near_terrain_production_1pct_metrics_state_metrics.csv`
- Saved-time CM1-vs-Breeze raw-error frame metrics:
  `validation_output/substepper/schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie/frame_pairs.csv`
- Saved-time CM1-vs-Breeze `w` and pressure frame error metrics:
  `validation_output/substepper/schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie/frame_pairs.csv`
- Movie:
  `validation_output/substepper/schar_cm1_periodic_theta300_vs_breeze_substepper_raw_error_movie/schar_cm1_vs_breeze_substepper_raw_error.mp4`
- Field-error movie:
  `validation_output/substepper/schar_cm1_periodic_theta300_vs_breeze_substepper_field_error_movie/schar_cm1_vs_breeze_substepper_field_error.mp4`
- Three-resolution campaign:
  `validation_output/substepper/terrain_schar_6h_substepper_convergence_production/`
- Status: `fail`.
- Evidence:
  - explicit-vs-CM1 below-sponge `u_relative_l2_error = 0.026636791805590348`
  - explicit-vs-CM1 below-sponge `w_relative_l2_error = 1.7549539691829341`
  - explicit-vs-CM1 below-sponge `w_normalized_rmse = 0.11204348858250109`
  - explicit-vs-CM1 `mountain_drag_relative_error = 3.3597789530483224`
  - substepper-vs-explicit below-sponge `u_relative_l2_error = 0.004411935075567736`
  - substepper-vs-explicit below-sponge `w_relative_l2_error = 0.22638797656312776`
  - substepper-vs-explicit below-sponge `w_normalized_rmse = 0.029339205804194135`
  - substepper-vs-explicit `mountain_drag_relative_error = 0.6269783052655318`
  - near-terrain explicit-vs-CM1 `w_relative_l2_error = 1.1949691854558966`
  - near-terrain substepper-vs-explicit `w_relative_l2_error = 0.13382617379916242`
  - explicit-vs-CM1 below-sponge `projection_amplitude_error = 0.17606602967141627`
  - substepper-vs-explicit below-sponge `projection_amplitude_error = 0.09475538706521736`
  - explicit-vs-CM1 near-terrain `projection_amplitude_error = 0.17505443644277563`
  - substepper-vs-explicit near-terrain `projection_amplitude_error = 0.03938473645174334`
  - saved-time substepper-vs-explicit scalar comparison has `407` rows,
    `294` failing 1% rows, and worst relative error `41.565943649924705` for
    `mountain_drag` at `10200 s`.
  - saved-time substepper-vs-explicit field comparison has `74` rows,
    `72` failing 1% rows, and worst relative L2 error
    `0.9579903914564901` for `pressure_perturbation` at
    `10798.200000006214 s`.
  - saved-time CM1-vs-Breeze raw-error frame metrics have `37` matched frames,
    zero time offset, and worst maximum absolute `w` error `3.5978230` at
    frame `35`, CM1 time `20400 s`.
  - saved-time CM1-vs-Breeze `w` and pressure frame metrics have `37`
    matched frames, zero time offset, worst maximum absolute `w` error
    `3.5884120` at `20400 s`, worst `w_relative_l2_error = 2.8738358`
    at `21600 s`, worst maximum absolute pressure error `328.80732` at
    `0 s`, and worst `pressure_relative_l2_error = 2.3224117e11` at `0 s`.
  - saved-time explicit-vs-CM1 metrics have `74` rows over `37` matched
    frames and two fields; all rows fail the 1% gate, with worst
    `relative_l2_error = 2.3224117e11` for pressure at `0 s`.

## Complex Mountain: CM1 Doernbrack Itern=3

### Production Declaration

- Manifest:
  `validation_output/substepper/complex_mountain_doernbrack_production_manifest.md`
- Artifact class: `production_validation`.
- Selected case: CM1-native Doernbrack-style 3D hill, `itern = 3`.
- Grid/runtime: `120x120x150`, `dx = dy = 1000 m`, `dz = 200 m`, `6 h`.
- Resolution note: 20 horizontal points per 20 km hill half-width; 100 active
  vertical cells below the sponge; 120x120x150 declared before comparison
  after 240x240x150 proved infeasible for local serial CM1.
- Status: `pass` for manifest.

### CM1 Explicit Reference

- Artifact class: `production_validation`.
- Command source:
  `validation_output/substepper/cm1_complex_mountain_doernbrack_production.batch`
- Config:
  `validation_output/substepper/complex_mountain_production_cm1_reference/cm1_config.txt`
- Backend/machine: local CM1 on `gpu-prod` allocation; CM1 run used CPU-style
  executable with `OMP_NUM_THREADS=8`.
- Grid/runtime: `nx = 120`, `ny = 120`, `nz = 150`, `6 h`.
- Raw final frame:
  `validation_output/substepper/complex_mountain_production_cm1_reference/cm1out_000037.nc`
- Converted metrics:
  `validation_output/substepper/complex_mountain_production_cm1_reference/complex_mountain_metrics.csv`
- Converted state slice:
  `validation_output/substepper/complex_mountain_production_cm1_reference/complex_mountain_state_slice.csv`
- Summary path: no standalone summary file; `cm1_config.txt`, the batch file,
  and converted metrics are the recorded provenance for this CM1 reference.
- Status: `pass` for artifact presence.

### Breeze Explicit

- Artifact class: `production_validation`.
- Command source:
  `validation_output/substepper/run_complex_mountain_doernbrack_breeze_production.batch`
- Backend/machine: Breeze CPU production run on `cpu-large`.
- Grid/runtime: `Nx = 120`, `Ny = 120`, `Nz = 150`, `dt = 0.25 s`, `6 h`.
- Metrics:
  `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_metrics.csv`
- Summary:
  `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_summary.txt`
- State slice:
  `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_state_slice.csv`
- Time series:
  `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_timeseries.csv`
- Centerline snapshots:
  `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_centerline_snapshots.csv`
- Plot:
  `validation_output/substepper/complex_mountain_production_explicit/complex_mountain_centerline_w.ppm`
- Status: `pass` for artifact presence.

### Breeze Acoustic Substepper

- Artifact class: `production_validation`.
- Command source:
  `validation_output/substepper/run_complex_mountain_doernbrack_breeze_production.batch`
- Backend/machine: Breeze CPU production run on `cpu-large`.
- Grid/runtime: `Nx = 120`, `Ny = 120`, `Nz = 150`, `dt = 2.0 s`, `6 h`.
- Metrics:
  `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_metrics.csv`
- Summary:
  `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_summary.txt`
- State slice:
  `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_state_slice.csv`
- Time series:
  `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_timeseries.csv`
- Centerline snapshots:
  `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_centerline_snapshots.csv`
- Plot:
  `validation_output/substepper/complex_mountain_production_substepper/complex_mountain_centerline_w.ppm`
- Status: `pass` for artifact presence.

### Complex Mountain Comparisons

- Explicit vs CM1:
  `validation_output/substepper/complex_mountain_explicit_vs_cm1_production_1pct_metrics.csv`
- Explicit vs CM1 summary:
  `validation_output/substepper/complex_mountain_explicit_vs_cm1_production_1pct_metrics_summary.txt`
- Explicit vs CM1 saved-time centerline metrics:
  `validation_output/substepper/complex_mountain_explicit_vs_cm1_timeseries_metrics.csv`
- Explicit vs CM1 saved-time centerline summary:
  `validation_output/substepper/complex_mountain_explicit_vs_cm1_timeseries_metrics_summary.txt`
- Substepper vs explicit:
  `validation_output/substepper/complex_mountain_substepper_vs_explicit_production_1pct_metrics.csv`
- Substepper vs explicit summary:
  `validation_output/substepper/complex_mountain_substepper_vs_explicit_production_1pct_metrics_summary.txt`
- Explicit vs CM1 projection metrics:
  `validation_output/substepper/complex_mountain_explicit_vs_cm1_projection_metrics.csv`
- Substepper vs explicit projection metrics:
  `validation_output/substepper/complex_mountain_substepper_vs_explicit_projection_metrics.csv`
- Substepper vs explicit saved-time scalar comparison metrics:
  `validation_output/substepper/complex_mountain_substepper_vs_explicit_timeseries_metrics.csv`
- Substepper vs explicit saved-time field comparison metrics:
  `validation_output/substepper/complex_mountain_substepper_vs_explicit_field_timeseries_metrics.csv`
- Substepper vs explicit saved-time field comparison summary:
  `validation_output/substepper/complex_mountain_substepper_vs_explicit_field_timeseries_metrics_summary.txt`
- Saved-time CM1-vs-Breeze frame error metrics:
  `validation_output/substepper/complex_mountain_cm1_vs_breeze_substepper_movie/frame_pairs.csv`
- Explicit-vs-CM1 movie:
  `validation_output/substepper/complex_mountain_cm1_vs_breeze_explicit_movie/complex_mountain_cm1_vs_breeze_substepper.mp4`
- Substepper-vs-CM1 movie:
  `validation_output/substepper/complex_mountain_cm1_vs_breeze_substepper_movie/complex_mountain_cm1_vs_breeze_substepper.mp4`
- Movie frames: 37 production frames.
- Status: `fail`.
- Evidence:
  - explicit-vs-CM1 below-sponge `w_relative_l2_error = 3.7768447403611067`
  - explicit-vs-CM1 below-sponge `pressure_relative_l2_error = 7.60472948572488`
  - explicit-vs-CM1 pressure-drag signs do not match CM1
  - substepper-vs-explicit below-sponge `w_relative_l2_error = 0.7640921528867319`
  - substepper-vs-explicit below-sponge `pressure_relative_l2_error = 0.8266102318858068`
  - substepper-vs-explicit drag signs match, but x/y drag relative errors are
    `0.7218918533` and `0.8263035360`
  - explicit-vs-CM1 below-sponge `projection_amplitude_error = 0.94392136146878`
  - substepper-vs-explicit below-sponge `projection_amplitude_error = 0.6706262595274035`
  - saved-time substepper-vs-explicit scalar comparison has `481` rows,
    `419` failing 1% rows, and worst relative error `14.85330164373352` for
    `pressure_drag_y` at `10800 s`.
  - saved-time substepper-vs-explicit field comparison has `148` rows,
    `144` failing 1% rows, and worst relative L2 error
    `1.0313818112185893` for `pressure_perturbation` at `18000 s`.
  - saved-time explicit-vs-CM1 centerline comparison has `148` rows over
    `37` matched frames and four fields; all rows fail the 1% gate, with worst
    `relative_l2_error = 2.0827813e11` for pressure at `0 s`.
  - saved-time CM1-vs-Breeze frame pairing has `37` frames, zero time offset,
    worst maximum absolute `w` error `1.4010414` at `21600 s`, and worst
    maximum absolute pressure error `122.86421` at `0 s`.
  - the same frame-pair file now records relative errors at every saved frame;
    worst `w_relative_l2_error = 12.012647` at frame `3`, and worst
    `pressure_relative_l2_error = 2.0827813e11` at `0 s`.

## Askervein

### Breeze Idealized LES Artifact

- Artifact class: `production_validation` in the local summary, but
  completion status is only `present`.
- Reason it is not accepted: it is a 1 s idealized artifact and does not have
  a declared production spin-up/averaging window or coordinate-faithful WEMEP
  reference comparison.
- Coordinate-faithful production manifest:
  `validation_output/substepper/askervein_coordinate_faithful_production_manifest.md`
- Command source:
  `validation_output/substepper/askervein_les_smoke_validation.jl`
- Backend/machine: Breeze CPU run.
- Grid/runtime: `Nx = 96`, `Ny = 72`, `Nz = 32`, `dt = 0.05 s`, `1 s`.
- Metrics:
  `validation_output/substepper/askervein_les_production/askervein_les_metrics.csv`
- Summary:
  `validation_output/substepper/askervein_les_production/askervein_les_summary.txt`
- Masts:
  `validation_output/substepper/askervein_les_production/askervein_les_masts.csv`
- Spectra:
  `validation_output/substepper/askervein_les_production/askervein_les_spectra.csv`
- Plot:
  `validation_output/substepper/askervein_les_production/askervein_les_speedup_w.ppm`
- Status: `present`, not pass.

### Breeze Explicit-Feasible Window

- Artifact class: `production_validation` in the local summary, but the
  summary also warns that the default configuration is diagnostic unless
  explicitly rerun with production resolution, runtime, and artifact class.
- Command source:
  `validation_output/substepper/run_askervein_explicit_substepper_production_window.batch`
- Backend/machine: Breeze CPU run.
- Grid/runtime: `Nx = 96`, `Ny = 72`, `Nz = 32`, `dt = 0.05 s`, `1 s`.
- Metrics:
  `validation_output/substepper/askervein_explicit_substepper_compare_production/askervein_explicit_substepper_metrics.csv`
- Summary:
  `validation_output/substepper/askervein_explicit_substepper_compare_production/askervein_explicit_substepper_summary.txt`
- Vertical error profile:
  `validation_output/substepper/askervein_explicit_substepper_compare_production/askervein_vertical_velocity_error_profile.csv`
- Vertical error extrema:
  `validation_output/substepper/askervein_explicit_substepper_compare_production/askervein_vertical_velocity_error_extrema.csv`
- Status: `blocked`.
- Evidence:
  - horizontal velocity/speed rows pass the 1% gate.
  - vertical velocity and `w_tilde` fail the 1% gate.
  - 5 s diagnostic completes but fails vertical velocity.
  - 6.5 s diagnostic is near blow-up.
  - 20 s and 60 s attempts fail in the explicit thermodynamic state.
  - no accepted explicit-feasible production window has been declared.

### Coordinate-Faithful ERF-Terrain Explicit Brackets

- Artifact class: `diagnostic`.
- Reason excluded from completion: these use ERF Askervein terrain at the
  current comparison-grid resolution, but they are short explicit/substepper
  brackets rather than a production LES with accepted boundary conditions,
  spin-up, averaging, and reference comparison.
- Script:
  `validation_output/substepper/askervein_explicit_substepper_compare.jl`
- Relevant command pattern:
  `ASKER_COMPARE_TERRAIN_SOURCE=erf ASKER_COMPARE_NX=96 ASKER_COMPARE_NY=72 ASKER_COMPARE_NZ=32 ASKER_COMPARE_DT=0.001 ASKER_COMPARE_ARCH=gpu julia --project=test --color=no validation_output/substepper/askervein_explicit_substepper_compare.jl`
- Git commit: `9b00422`.
- Backend/machine: Breeze GPU diagnostic run on `gpu-prod-st-gpu-prod-1`,
  H100 visible through CUDA in the `test` project environment.
- Grid: `Nx = 96`, `Ny = 72`, `Nz = 32`.
- Timestep: `dt = 0.001 s`.
- Reference dataset:
  `/shared/home/greg/ERF/Exec/CanonicalTests/Real_Terrain/Askervein/askervein.txt`
- Passing explicit/substepper brackets:
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1s_gpu_diagnostic/askervein_explicit_substepper_metrics.csv`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1p2s_gpu_diagnostic/askervein_explicit_substepper_metrics.csv`
- Failing explicit/substepper brackets:
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1p25s_gpu_diagnostic/askervein_explicit_substepper_metrics.csv`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1p5s_gpu_diagnostic/askervein_explicit_substepper_metrics.csv`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_2s_gpu_diagnostic/askervein_explicit_substepper_metrics.csv`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_5s_gpu_diagnostic/askervein_explicit_substepper_metrics.csv`
- Summaries:
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1s_gpu_diagnostic/askervein_explicit_substepper_summary.txt`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1p2s_gpu_diagnostic/askervein_explicit_substepper_summary.txt`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1p25s_gpu_diagnostic/askervein_explicit_substepper_summary.txt`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_1p5s_gpu_diagnostic/askervein_explicit_substepper_summary.txt`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_2s_gpu_diagnostic/askervein_explicit_substepper_summary.txt`
  - `validation_output/substepper/askervein_erf_terrain_explicit_substepper_96x72x32_5s_gpu_diagnostic/askervein_explicit_substepper_summary.txt`
- Pass/fail status:
  - `1.0 s`: pass, all rows satisfy the strict 1% gate.
  - `1.2 s`: pass, all rows satisfy the strict 1% gate; worst full-domain
    relative L2 is `w = 0.00205`, and `w_tilde_relative_linf_error =
    0.00970`. This refreshed diagnostic includes `full_domain`,
    `near_terrain`, `centerline_slice`, `lee_side_box`, and `hilltop_box`.
  - `1.25 s`: fail, `w_tilde_relative_linf_error = 0.01009`.
  - `1.5 s`: fail, `w_tilde_relative_linf_error = 0.01220`.
  - `2.0 s`: fail, `w_relative_linf_error = 0.01175` and
    `w_tilde_relative_linf_error = 0.01636`.
  - `5.0 s`: fail, `w_relative_linf_error = 0.02677` and
    `w_tilde_relative_linf_error = 0.03680`.
- Plot/movie path: none for these diagnostic brackets; completion still relies
  on the existing Askervein plot for the idealized LES artifact and requires a
  future production LES movie or plot sequence from coordinate-faithful output.

### WEMEP Reference Data

- Source: https://wemep.readthedocs.io/en/latest/windconditions/benchmarks/askervein.html
- Source dataset: https://zenodo.org/records/4095052
- DOI: `10.5281/zenodo.4095052`
- Manifest:
  `validation_output/substepper/askervein_wemep_reference_manifest.md`
- Downloaded files:
  - `validation_output/substepper/askervein_wemep_reference/askervein_elevation-roughness.map`
  - `validation_output/substepper/askervein_wemep_reference/askervein_inlet1.txt`
  - `validation_output/substepper/askervein_wemep_reference/askervein_sensor1.txt`
  - `validation_output/substepper/askervein_wemep_reference/askervein_validation1.txt`
- Named-mast diagnostic script:
  `validation_output/substepper/askervein_wemep_mast_compare.jl`
- Named-mast metrics:
  `validation_output/substepper/askervein_wemep_reference/askervein_les_production_askervein_wemep_mast_metrics.csv`
- Named-mast summary:
  `validation_output/substepper/askervein_wemep_reference/askervein_les_production_askervein_wemep_mast_summary.txt`
- Status: `present`, not pass.
- Evidence:
  - matched masts: `RS`, `HT`, `CP`
  - maximum fractional-speed-up absolute error: `0.38745059611643906`
  - one-percent pass: `false`
  - diagnostic only because the Breeze artifact is idealized and not a
    coordinate-faithful WEMEP production run.

## Overall Status

The metrics-generation objective has produced quantitative evidence for all
three required case families, but the production-validation acceptance target
is not achieved:

- Schar has complete production artifacts and fails 1% comparisons.
- Complex mountain has complete production artifacts, comparisons, and movie,
  and fails 1% comparisons.
- Askervein has reference data and diagnostic comparisons, but remains blocked
  on coordinate-faithful Breeze/WEMEP comparison plumbing, accepted spin-up and
  averaging windows, and an accepted explicit-feasible comparison window. ERF
  terrain explicit/substepper diagnostics now quantify a strict 1% window:
  pass through `1.2 s`, fail by `1.25 s`, but these are diagnostic brackets,
  not production-validation artifacts. The coordinate-faithful production
  manifest now records the WEMEP/ERF target geometry and reference scope, but
  no Breeze production run consumes the full production setup yet.

## Manifest Contract Audit

The plan requires the final validation manifest to list, for each case and run:
run command, git commit, machine/backend, grid and timestep, physical runtime,
artifact class, metrics path, summary path, movie or plot path, reference
dataset path, and pass/fail status for each 1% metric.

Reference dataset path coverage is recorded in each reference subsection: CM1
output directories for Schar and complex mountain, ERF terrain files for the
Askervein diagnostic brackets, and WEMEP/Zenodo files for the Askervein
observational reference.

Current status:

- Schar: manifest coverage is complete for the production artifacts and the
  field-snapshot movie rerun. Older CPU production runs are tied to the global
  artifact commit and command-source batch files rather than per-run inline
  command strings.
- Complex mountain: manifest coverage is complete for the completed production
  artifacts, CM1 reference, comparisons, projection metrics, and movie. Older
  CPU production runs are tied to the global artifact commit and command-source
  batch files rather than per-run inline command strings.
- Askervein: manifest coverage is intentionally incomplete. The listed LES and
  explicit-window artifacts are diagnostic/blocker evidence, not accepted
  production validation. The missing final-manifest fields are the
  coordinate-faithful Breeze production run command, accepted production
  runtime, spin-up and averaging window, production movie or plot sequence from
  coordinate-faithful output, accepted reference comparison path, and pass/fail
  status for a declared explicit-feasible production window.

Smoke and diagnostic artifacts are explicitly excluded from completion.
