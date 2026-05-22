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
- `3ea9984` is the current local HEAD at the latest audit update. The
  additional source-facing change since `9b00422` is the
  `FaceSampledTerrain()` / `GridFittedTerrain()` terrain-interpretation
  interface plus removal of the failed experimental Lin finite-volume PGF
  public surface. Existing production artifacts predate this local source
  update unless explicitly noted.

Latest gate:

```sh
julia --project=. --color=no validation_output/substepper/terrain_following_production_validation_gate.jl
```

Latest result:

```text
production validation gate: pass=16 present=23 fail=23 missing=0 blocked=5
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

### Breeze Acoustic Substepper Matched Outer-Dt Discriminator

- Artifact class: `production_validation`.
- Purpose: production-length Schär Tier-1 discriminator with the substepper
  outer step matched to the explicit `dt = 0.35 s`, plus grid terrain, no
  divergence damping, and no acoustic upper sponge.
- Command source:
  `validation_output/substepper/run_schar_400x200_substepper_dt0p35_no_damping_no_upper_sponge_grid.batch`
- Run log:
  `validation_output/substepper/schar_sub_dt0p35_grid_nodamp_nosponge-1089.log`
- Current-branch refresh log:
  `validation_output/substepper/schar_sub_dt0p35_grid_nodamp_nosponge-1133.log`
- Backend/machine: Breeze GPU production run on `gpu-dev`; current-branch
  refresh on `gpu-prod`.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 0.35 s`, `6 h`.
- Metrics:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/terrain_schar_mountain_wave_metrics.csv`
- Summary:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/terrain_schar_mountain_wave_summary.txt`
- State slice:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/terrain_schar_mountain_wave_state_slice.csv`
- Time series:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/terrain_schar_mountain_wave_energy_timeseries.csv`
- Tier-1 comparison metrics:
  `validation_output/substepper/schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/schar_substepper_vs_explicit_state_metrics.csv`
- Tier-1 comparison summary:
  `validation_output/substepper/schar_substepper_vs_explicit_tier1_6h_dt0p35_no_damping_no_upper_sponge_grid_current_gpu_prod_1133/schar_substepper_vs_explicit_summary.txt`
- Status: `fail` for 1% Tier-1 accuracy. Coordinate parity is exact, but
  below-sponge `w_relative_linf_error = 0.0745803850`,
  `w_relative_l2_error = 0.1232164948`, `w_normalized_rmse = 0.0159684898`,
  pressure relative L2 is `0.6617009969`, and mountain-drag relative error is
  `0.4057695816`.

### Breeze Acoustic Substepper Previous-Horizontal-Divergence Discriminator

- Artifact class: `production_validation`.
- Purpose: production-length Schär Tier-1 discriminator for the short-window
  previous-horizontal-divergence timing candidate.
- Command source:
  `validation_output/substepper/run_schar_400x200_substepper_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid.batch`
- Run log:
  `validation_output/substepper/schar_prev_hdiv_dt035-1185.log`
- Backend/machine: Breeze GPU production run on `gpu-dev-st-gpu-dev-1`.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 0.35 s`, `6 h`.
- Timing: `SCHAR_HORIZONTAL_DIVERGENCE_TIMING=previous`.
- Metrics:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/terrain_schar_mountain_wave_metrics.csv`
- Summary:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/terrain_schar_mountain_wave_summary.txt`
- State slice:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/terrain_schar_mountain_wave_state_slice.csv`
- Time series:
  `validation_output/substepper/terrain_schar_6h_400x200_substepper_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/terrain_schar_mountain_wave_energy_timeseries.csv`
- Tier-1 comparison metrics:
  `validation_output/substepper/schar_substepper_vs_explicit_tier1_6h_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/schar_substepper_vs_explicit_state_metrics.csv`
- Tier-1 comparison summary:
  `validation_output/substepper/schar_substepper_vs_explicit_tier1_6h_dt0p35_previous_hdiv_no_damping_no_upper_sponge_grid/schar_substepper_vs_explicit_summary.txt`
- Status: `fail` for 1% Tier-1 accuracy. Coordinate parity is exact, but
  below-sponge `w_relative_linf_error = 0.0745803849951862`,
  `w_relative_l2_error = 0.12321649479541662`,
  `w_normalized_rmse = 0.01596848982068666`, pressure relative L2 is
  `0.6617009969440094`, and mountain-drag relative error is
  `0.4057695816151736`.

### Breeze Low-Amplitude Linear Mountain Wave

- Artifact class: `production_validation`.
- Purpose: CM1-free low-amplitude Schär mountain-wave comparison against the
  built-in analytical linear-wave reference, plus a low-amplitude
  substepper-vs-explicit Tier-1 comparison.
- Command:
  `LINEAR_MOUNTAIN_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu SCHAR_ARCH=gpu SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=2 SCHAR_H0=25 SCHAR_TERRAIN_INTERPRETATION=grid SCHAR_PRESSURE_GRADIENT_STENCIL=inside SCHAR_WRITE_ENERGY_TIMESERIES=true SCHAR_ENERGY_INTERVAL_SECONDS=600 SCHAR_MAKE_MOVIE=false julia --project=examples --color=no validation_output/substepper/linear_mountain_wave_validation.jl`
- Backend/machine: Breeze GPU production run on visible `NVIDIA H100 80GB HBM3`.
- Grid/runtime: `Nx = 400`, `Nz = 200`, `dt = 2.0 s`, `6 h`, `h0 = 25 m`.
- Metrics:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu/linear_mountain_wave_state_metrics.csv`
- Scalar metrics:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu/linear_mountain_wave_scalar_metrics.csv`
- Summary:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu/linear_mountain_wave_summary.md`
- Final-state plot:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu/linear_mountain_wave_w_comparison.ppm`
- State slice:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu/terrain_schar_mountain_wave_state_slice.csv`
- Time series:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu/terrain_schar_mountain_wave_energy_timeseries.csv`
- Status: `fail` for analytical `w` accuracy. The run is stable
  (`nan_count = 0`, `inf_count = 0`), but below-sponge `w`, excluding boundary
  faces, has relative L2 `1.6310697784`, relative L∞ `0.4566711505`,
  normalized RMSE `0.0971352861`, pattern correlation `0.4962314148`,
  maximum-amplitude error `0.0634351093`, projection amplitude error
  `0.0687402754`, best-shift projection amplitude error `0.0687402754`, and
  best shift `0` cells.
- Explicit-control artifact:
  `validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu/`
- Explicit-control command:
  `SCHAR_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu SCHAR_ARCH=gpu SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=0.35 SCHAR_H0=25 SCHAR_TERRAIN_INTERPRETATION=grid SCHAR_PRESSURE_GRADIENT_STENCIL=inside SCHAR_WRITE_ENERGY_TIMESERIES=true SCHAR_ENERGY_INTERVAL_SECONDS=600 SCHAR_MAKE_MOVIE=false julia --project=examples --color=no validation_output/substepper/terrain_schar_mountain_wave_explicit_validation.jl`
- Explicit-control metrics:
  `validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu/linear_mountain_wave_state_metrics.csv`
- Explicit-control final-state plot:
  `validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu/linear_mountain_wave_w_comparison.ppm`
- Explicit-control status: `fail` for analytical `w` accuracy. The run is
  stable (`nan_count = 0`, `inf_count = 0`), but below-sponge `w`, excluding
  boundary faces, has relative L2 `1.8079439240`, relative L∞ `0.5198932319`,
  normalized RMSE `0.1076686924`, pattern correlation `0.4605970664`,
  maximum-amplitude error `0.0658505614`, projection amplitude error
  `0.0634816092`, best-shift projection amplitude error `0.0634816092`, and
  best shift `0` cells.
- Substepper-vs-explicit metrics:
  `validation_output/substepper/linear_mountain_wave_substepper_vs_explicit_400x200_6h_gpu/schar_substepper_vs_explicit_state_metrics.csv`
- Substepper-vs-explicit summary:
  `validation_output/substepper/linear_mountain_wave_substepper_vs_explicit_400x200_6h_gpu/schar_substepper_vs_explicit_summary.txt`
- Convention audit:
  `validation_output/substepper/linear_mountain_wave_convention_audit.md`
  records that excluding boundary faces removes a misleading analytical
  `relative_linf_error = 1.0`, and that an approximate contravariant `w̃`
  comparison improves but does not rescue the analytical gate.
- Substepper-vs-explicit status: `fail`. Coordinate parity is exact, but
  below-sponge `w` has relative L∞ `0.1583762049`, relative L2
  `0.2267029486`, normalized RMSE `0.0309254096`, pattern correlation
  `0.9757036753`, projection amplitude error `0.1004766690`, and
  mountain-drag relative error `0.6306652106`.

#### Exact-`w̃` Substepper Rerun

- Artifact class: `production_validation`.
- Purpose: production-length low-amplitude Schär substepper rerun with exact
  `w_tilde`, `w_tilde_linear`, and `w_tilde_error` columns in the final
  `w_slice`.
- Command:
  `LINEAR_MOUNTAIN_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu_wtilde SCHAR_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu_wtilde SCHAR_ARCH=gpu SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=2 SCHAR_H0=25 SCHAR_TERRAIN_INTERPRETATION=grid SCHAR_PRESSURE_GRADIENT_STENCIL=inside SCHAR_WRITE_ENERGY_TIMESERIES=true SCHAR_ENERGY_INTERVAL_SECONDS=600 SCHAR_MAKE_MOVIE=false JULIA_NUM_THREADS=1 julia --project=examples --color=no validation_output/substepper/linear_mountain_wave_validation.jl`
- Metrics:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu_wtilde/linear_mountain_wave_state_metrics.csv`
- State slice:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu_wtilde/terrain_schar_mountain_wave_w_slice.csv`
- Final-state plot:
  `validation_output/substepper/linear_mountain_wave_production_400x200_6h_gpu_wtilde/linear_mountain_wave_w_comparison.ppm`
- Status: stable (`nan_count = 0`, `inf_count = 0`) but fails analytical
  exact-`w̃` accuracy. Below-sponge `w_tilde` relative L2 `0.8352048912`,
  relative L∞ `0.3833304476`, normalized RMSE `0.0821621241`, pattern
  correlation `0.7344266758`, projection amplitude error `0.1033396488`, and
  best shift `0` cells.

#### Exact-`w̃` Explicit-Control Rerun

- Artifact class: `production_validation`.
- Purpose: production-length low-amplitude Schär explicit-control rerun with
  exact `w_tilde`, `w_tilde_linear`, and `w_tilde_error` columns in the final
  `w_slice`.
- Command:
  `SCHAR_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde SCHAR_ARCH=gpu SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=0.35 SCHAR_H0=25 SCHAR_TERRAIN_INTERPRETATION=grid SCHAR_PRESSURE_GRADIENT_STENCIL=inside SCHAR_WRITE_ENERGY_TIMESERIES=true SCHAR_ENERGY_INTERVAL_SECONDS=600 SCHAR_MAKE_MOVIE=false JULIA_NUM_THREADS=1 julia --project=examples --color=no validation_output/substepper/terrain_schar_mountain_wave_explicit_validation.jl`
- Postprocessing command:
  `LINEAR_MOUNTAIN_REUSE_OUTPUT=true LINEAR_MOUNTAIN_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde SCHAR_OUTPUT_DIR=validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde SCHAR_ARCH=gpu SCHAR_NX=400 SCHAR_NZ=200 SCHAR_STOP_SECONDS=21600 SCHAR_DT=0.35 SCHAR_H0=25 LINEAR_MOUNTAIN_MODEL_LABEL=breeze_explicit JULIA_NUM_THREADS=1 julia --project=examples --color=no validation_output/substepper/linear_mountain_wave_validation.jl`
- Metrics:
  `validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde/linear_mountain_wave_state_metrics.csv`
- State slice:
  `validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde/terrain_schar_mountain_wave_w_slice.csv`
- Final-state plot:
  `validation_output/substepper/linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde/linear_mountain_wave_w_comparison.ppm`
- Status: stable (`nan_count = 0`, `inf_count = 0`) but fails analytical
  exact-`w̃` accuracy. Below-sponge `w_tilde` relative L2 `0.9243801650`,
  relative L∞ `0.4364041061`, normalized RMSE `0.0909346181`, pattern
  correlation `0.6997871856`, projection amplitude error `0.1009619580`, and
  best shift `0` cells.

### Schär Early-Time Operator-Budget Diagnostics

- Artifact class: `diagnostic`.
- Purpose: localize the Schär production-validation blocker before launching
  more 6 h production reruns.
- Summary:
  `validation_output/substepper/schar_2s_operator_budget_blocker_summary.md`
- Machine-readable rows:
  `validation_output/substepper/schar_2s_operator_budget_blocker_summary.csv`
- CM1 PGF formula validation:
  `validation_output/substepper/schar_cm1_pgrad_budget_source_audit.md`,
  `validation_output/substepper/schar_2s_cm1_pgrad_formula_validation.md`
  `validation_output/substepper/schar_2s_cm1_pgrad_time_centering_diagnosis.md`
  and `validation_output/substepper/schar_600s_cm1_pgrad_formula_validation.md`
- CM1 u-budget closure:
  `validation_output/substepper/schar_2s_cm1_budget_closure_summary.md`
  and
  `validation_output/substepper/schar_2s_cm1_budget_closure_summary.csv`
- Breeze acoustic-increment diagnostic:
  `validation_output/substepper/schar_2s_breeze_acoustic_increment_vs_cm1_ub_pgrad_summary.md`
  and
  `validation_output/substepper/schar_2s_breeze_acoustic_increment_vs_cm1_ub_pgrad_summary.csv`
- Breeze stable-explicit face-u increment diagnostic:
  `validation_output/substepper/schar_2s_breeze_dt0p1_cm1_u_increment_closure_summary.md`
  and
  `validation_output/substepper/schar_2s_breeze_dt0p1_cm1_u_increment_closure_summary.csv`
- Breeze stable-explicit face-u increment diagnostics with CM1 terrain/constants:
  `validation_output/substepper/schar_2s_breeze_dt0p1_cm1terrain_cm1_u_increment_closure_summary.md`,
  `validation_output/substepper/schar_2s_breeze_dt0p1_cm1terrain_cm1_u_increment_closure_summary.csv`,
  `validation_output/substepper/schar_2s_breeze_dt0p1_cm1terrain_cm1constants_cm1_u_increment_closure_summary.md`,
  and
  `validation_output/substepper/schar_2s_breeze_dt0p1_cm1terrain_cm1constants_cm1_u_increment_closure_summary.csv`
- Status: the `2 s`, `400 x 200`, below-sponge budget comparison fails the
  1% target by large margins. Current blocker rows include `ub_pgrad`
  relative L2 `1.352884684`, `wb_pgrad` relative L2 `1.566359834`,
  `wb_buoy` relative L2 `759.0814827`, centered `ub_vadv` relative L2
  `114.2449497`, and best tested velocity-form `ub_vadv` relative L2
  `14.14802568`. The postprocessed CM1 horizontal PGF formula validates at
  `600 s` but not at the `2 s` blocker time. A simple midpoint test does not
  fix the `2 s` pattern mismatch. Source inspection shows CM1's emitted
  `ub_pgrad` is an acoustic-step velocity increment budget, so `2 s` PGF
  implementation experiments should compare to CM1's emitted budget field or
  reproduce that increment convention. CM1's own emitted u-budget closes
  against its saved state increment at relative L2 `2.582710086e-8`, and
  reconstructs emitted `ub_pgrad` by residual at relative L2 `2.582295417e-8`.
  A first Breeze outer-step
  acoustic-increment diagnostic still fails against CM1 `ub_pgrad` with
  relative L2 `1.181135702` and pattern correlation `0.7197870843`, so the
  exact CM1 in-step budget convention remains unresolved. A stable explicit
  Breeze `dt = 0.1 s` face-`u` increment diagnostic also fails against CM1's
  actual face-`u` increment with relative L2 `1.324056471` and pattern
  correlation `0.6295060182`. Rerunning with CM1 terrain interpretation and
  CM1 dry constants does not improve it: relative L2 `1.339791739` and pattern
  correlation `0.6172172957`.

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
- Substepper vs explicit TEST E/F production discriminator:
  `validation_output/substepper/schar_substepper_vs_explicit_tier1_6h_no_damping_no_upper_sponge_grid_schema_refresh_coordcheck/schar_substepper_vs_explicit_state_metrics.csv`
- Substepper vs explicit TEST E/F production discriminator summary:
  `validation_output/substepper/schar_substepper_vs_explicit_tier1_6h_no_damping_no_upper_sponge_grid_schema_refresh_coordcheck/schar_substepper_vs_explicit_summary.txt`
- TEST E/F production substepper artifact:
  `validation_output/substepper/terrain_schar_6h_400x200_production_substepper_no_damping_no_upper_sponge_grid/`
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
- Failed Lin finite-volume PGF diagnostic:
  `validation_output/substepper/terrain_schar_2s_400x200_explicit_weno9_lin_pgf_operator_budget_diagnostic/`
- Failed Lin finite-volume PGF operator-budget summary:
  `validation_output/substepper/schar_2s_weno9_lin_pgf_operator_budget_vs_cm1_summary.md`
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
  - TEST E/F no-damping/no-upper-sponge production discriminator has matched
    grid coordinates (`maximum |Δx| = 0`, `maximum |Δz| = 0`) and is stable
    (`nan_count = 0`, `inf_count = 0`), but still fails:
    below-sponge `w_relative_linf_error = 0.1244488499`,
    `w_relative_l2_error = 0.1970133428`,
    `w_normalized_rmse = 0.0255323410`,
    `w_pattern_correlation = 0.9808452322`,
    `w_projection_amplitude_error = 0.0546232719`, and
    `mountain_drag_relative_error = 0.5949801384`.
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
  - the Lin finite-volume PGF discriminator is diagnostic-only and failed:
    the `2 s`, `400 x 200`, WENO-9 run produced `nan_count = 372392`, while
    operator-budget comparison still reported `ub_pgrad` relative L2
    `1.575561862` and `wb_pgrad` relative L2 `1.529498472`. The public
    `LinFiniteVolume` API was removed rather than retained as unused broken
    source.

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

### Pressure-Gradient Askervein Diagnostics

- Artifact class: `diagnostic`.
- Reason excluded from completion: these runs probe the forcing/stability
  envelope for a coordinate-faithful Askervein LES, but they do not yet define
  an accepted production spin-up, averaging window, reference-comparison path,
  or explicit-feasible production window.
- Script:
  `validation_output/substepper/askervein_neutral_les_case.jl`
- Completed high bracket command source:
  `validation_output/substepper/run_askervein_pressure_gradient_bracket_gpu.batch`
- Completed high bracket artifacts:
  - `validation_output/substepper/askervein_pressure_gradient_bracket_cuda_192x192x64_300s_a0p04/`
  - `validation_output/substepper/askervein_pressure_gradient_bracket_cuda_192x192x64_300s_a0p06/`
  - `validation_output/substepper/askervein_pressure_gradient_bracket_cuda_192x192x64_300s_a0p08/`
- Completed high bracket status: fail/diagnostic only. Accelerations `0.04`,
  `0.06`, and `0.08 m s^-2` all produced `NaN` model RS mast speed and `NaN`
  maximum fractional-speed-up error at `192 x 192 x 64`, `300 s`.
- Completed lower-bracket point:
  `validation_output/substepper/askervein_pressure_gradient_low_bracket_cuda_192x192x64_300s_a0p025/`
- Completed lower-bracket point status: fail/diagnostic only.
  `a = 0.025 m s^-2` also produced `NaN` model RS mast speed and `NaN`
  maximum fractional-speed-up error with `19` samples over `300 s`.
- Completed lower-bracket point:
  `validation_output/substepper/askervein_pressure_gradient_low_bracket_cuda_192x192x64_300s_a0p03/`
- Completed lower-bracket point status: fail/diagnostic only.
  `a = 0.03 m s^-2` also produced `NaN` model RS mast speed and `NaN`
  maximum fractional-speed-up error with `19` samples over `300 s`.
- Completed lower-bracket point:
  `validation_output/substepper/askervein_pressure_gradient_low_bracket_cuda_192x192x64_300s_a0p035/`
- Completed lower-bracket point status: fail/diagnostic only.
  `a = 0.035 m s^-2` also produced `NaN` model RS mast speed and `NaN`
  maximum fractional-speed-up error with `19` samples over `300 s`.
- Active lower bracket command source:
  `validation_output/substepper/run_askervein_pressure_gradient_low_bracket_gpu.batch`
- Active long finite-response command source:
  `validation_output/substepper/run_askervein_pressure_gradient_finite_long_gpu_prod.batch`
- Completed long finite-response point:
  `validation_output/substepper/askervein_pressure_gradient_finite_long_cuda_256x256x96_2400s_a0p02/`
- Completed long finite-response status: fail/diagnostic only.
  The uncapped `a = 0.02 m s^-2` attempt at `256 x 256 x 96`, `2400 s`,
  wrote `24` full-size `w_tilde_xz` frames but still produced `NaN` model RS
  mast speed and `NaN` maximum fractional-speed-up error after `26` samples.
- Capped long finite-response retry command source:
  `validation_output/substepper/run_askervein_pressure_gradient_finite_long_gpu_prod.batch`
- Completed capped long finite-response point:
  `validation_output/substepper/askervein_pressure_gradient_finite_long_cuda_256x256x96_2400s_a0p02/`
- Completed capped long finite-response status: fail/diagnostic only.
  Job `1165` completed the capped `a = 0.02 m s^-2` point at
  `256 x 256 x 96`, `2400 s`, after adding `ASKER_CASE_MAX_DT=0.05`,
  `ASKER_CASE_DT_UPDATE_INTERVAL=10`, and `ASKER_CASE_SLICE_LIMIT=0.75`.
  It wrote `121` full-size `w_tilde_xz` frames and finite mast metrics
  (`reference_speed_model = 2.256985683648537`,
  `max_abs_fsr_error = 6.555834500603011`, `samples = 220`,
  `average_seconds = 2201.512515079789`), proving the cap fixes the long
  NaN-mast failure mode but does not produce an accepted reference comparison.
- Completed capped long finite-response point:
  `validation_output/substepper/askervein_pressure_gradient_finite_long_cuda_256x256x96_2400s_a0p025/`
- Completed capped long finite-response status:
  fail/diagnostic only. Job `1165` completed the capped
  `a = 0.025 m s^-2` point at `256 x 256 x 96`, `2400 s`. It wrote `121`
  full-size `w_tilde_xz` frames and finite mast metrics
  (`reference_speed_model = 2.5369400008976233`,
  `max_abs_fsr_error = 6.69566015642208`, `samples = 220`,
  `average_seconds = 2201.512515079789`), but the reference comparison is
  still far outside acceptance.
- Completed capped long finite-response point:
  `validation_output/substepper/askervein_pressure_gradient_finite_long_cuda_256x256x96_2400s_a0p100/`
- Completed capped long finite-response status:
  fail/diagnostic only. Job `1186` ran on `gpu-prod-st-gpu-prod-1` and tested
  `a = 0.10 m s^-2` at `256 x 256 x 96`, `2400 s`, with
  `ASKER_CASE_MAX_DT=0.05`. It wrote `121` full-size `w_tilde_xz` frames and
  finite mast metrics (`reference_speed_model = 5.253831533722133`,
  `max_abs_fsr_error = 10.139617785288996`, `max_abs_tke_error =
  733.4619859674443`, `samples = 220`, `average_seconds =
  2201.512515079789`), showing that higher PG improves one RS speed but
  worsens the production-resolution mast field.
- Short capped low-grid retry command source:
  `validation_output/substepper/run_askervein_pressure_gradient_capped_low_gpu_dev.batch`
- Completed short capped low-grid retry:
  `validation_output/substepper/askervein_pressure_gradient_capped_low_cuda_192x192x64_300s_a0p02/`
- Completed short capped low-grid retry status:
  fail/diagnostic only. Job `1166` completed `a = 0.02 m s^-2` at
  `192 x 192 x 64`, `300 s`, with `ASKER_CASE_MAX_DT=0.05`. It produced
  finite mast metrics (`reference_speed_model = 4.643822590370479`,
  `max_abs_fsr_error = 1.409831851518315`, `samples = 118`) but remains
  inaccurate and not accepted validation. The planned `a = 0.025 m s^-2`
  second leg was cancelled after this summary because job `1169` already
  completed the same capped configuration.
- Completed parallel single-leg capped low-grid retry:
  `validation_output/substepper/askervein_pressure_gradient_capped_single_cuda_192x192x64_300s_a0p025/`
- Completed parallel single-leg capped low-grid retry status:
  fail/diagnostic only. Job `1169` ran on `gpu-prod-st-gpu-prod-2` and tested
  only `a = 0.025 m s^-2` at `192 x 192 x 64`, `300 s`, with
  `ASKER_CASE_MAX_DT=0.05`. It produced finite mast metrics
  (`reference_speed_model = 4.935863244558616`,
  `max_abs_fsr_error = 1.4516840207284545`, `samples = 118`), proving the
  cap fixes the low-grid NaN-mast failure mode for this point, but it is still
  not accepted validation (`production_validation = false`,
  `production_average = false`, fringe forcing). Job `1168` was an equivalent
  dynamic-`gpu-dev` submission but was cancelled after staying in
  `CONFIGURING` with no log output.
- Completed capped `a = 0.045 m s^-2` low-grid retry:
  `validation_output/substepper/askervein_pressure_gradient_capped_single_cuda_192x192x64_300s_a0p045/`
- Completed capped `a = 0.045 m s^-2` low-grid retry status:
  fail/diagnostic only. Job `1172` ran on `gpu-prod-st-gpu-prod-2` at
  `192 x 192 x 64`, `300 s`, with `ASKER_CASE_MAX_DT=0.05`. It stayed finite
  (`reference_speed_model = 5.989918904395469`,
  `max_abs_fsr_error = 1.5342592003584854`, `samples = 118`) but remains far
  outside the reference-comparison target and is not accepted validation.
- Completed capped `a = 0.10 m s^-2` low-grid retry:
  `validation_output/substepper/askervein_pressure_gradient_capped_single_cuda_192x192x64_300s_a0p100/`
- Completed capped `a = 0.10 m s^-2` low-grid retry status:
  fail/diagnostic only. Job `1174` ran on `gpu-prod-st-gpu-prod-2` at
  `192 x 192 x 64`, `300 s`, with `ASKER_CASE_MAX_DT=0.05`. It nearly matches
  the single RS reference speed (`reference_speed_model = 8.30920752092283`
  versus `reference_speed_observed = 8.895`) but still has order-one spatial
  errors (`max_abs_fsr_error = 1.3397411630772242`,
  `max_abs_tke_error = 125.94100845453049`) and is not accepted validation.
- Failed long finite-response attempt: job `1159` failed in diagnostic
  movie-frame writing before metrics were emitted because a non-finite slice
  value reached `UInt8(NaN)`.
- Diagnostic script patch: `color_triplet` now maps non-finite values to
  magenta, and frame color limits ignore non-finite values.
- Diagnostic status as of `2026-05-22T07:25Z`: lower bracket job `1158`
  completed on `gpu-dev`; all lower-bracket points from `0.025` to
  `0.035 m s^-2` have non-finite mast metrics. Uncapped long finite-response
  job `1164` completed the `0.02 m s^-2` point with non-finite mast metrics
  and was cancelled before finishing `0.025 m s^-2`. Capped long retry `1165`
  completed the `0.02 m s^-2` point with finite but underdriven diagnostic
  mast/reference metrics. Capped long retry `1165` also completed the
  `0.025 m s^-2` point with finite but still inaccurate diagnostic
  mast/reference metrics. Capped low-grid retry `1166` (`a = 0.02`) and
  single-leg capped retries `1169` (`a = 0.025`), `1172` (`a = 0.045`), and
  `1174` (`a = 0.10`) completed with finite but inaccurate diagnostic
  mast/reference metrics. Long capped retry `1186` (`a = 0.10`) also completed
  with finite but worse diagnostic mast/reference metrics. No accepted
  pressure-gradient validation metrics are recorded yet.

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
  field-snapshot movie rerun. The latest field-snapshot movie refresh is job
  `1163`, completed on `gpu-prod` at `2026-05-22T02:02Z` in
  `validation_output/substepper/terrain_schar_fine_6h_movie_gpu/`, with
  `400 x 200`, `6 h`, `dt = 2 s`, `nan_count = 0`, `inf_count = 0`, and
  `38` raw `w` snapshots. Older CPU production runs are tied to the global
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

## Additional Schär 2 s Diagnostic Evidence

- Artifact:
  `validation_output/substepper/terrain_schar_2s_400x200_explicit_dt0p1_cm1terrain_cm1constants_rk_split_conservative_diagnostic_cpu/`
- Command class: explicit Schär, `400 x 200`, `2 s`, `dt = 0.1 s`, CM1 terrain
  interpretation, CM1 dry constants, CPU backend, RK split-increment diagnostic
  enabled.
- Metrics:
  `validation_output/substepper/schar_2s_breeze_cm1_rk_split_conservative_increment_budget_summary.csv`
- Summary:
  `validation_output/substepper/schar_2s_breeze_cm1_rk_split_conservative_increment_budget_summary.md`
- Status: diagnostic-only, not pass.
- Evidence:
  - RK split total vs Breeze actual face-`u` increment relative L2:
    `1.850618453e-13`;
  - RK pressure split vs CM1 `ub_pgrad` relative L2: `1.339146674`;
  - Breeze actual face-`u` increment vs CM1 face-`u` increment remains
    `1.339791739`.
- Interpretation: the conservative split diagnostic closes Breeze's own update
  to roundoff and is internally interpretable, but it still fails the CM1
  comparison by order one.
- Outside-stencil control:
  `validation_output/substepper/schar_2s_breeze_cm1_rk_split_conservative_outside_pgf_increment_budget_summary.md`
  also closes Breeze's own update to roundoff but gives pressure relative L2
  `1.341576719` against CM1 `ub_pgrad`, slightly worse than the inside-stencil
  diagnostic.
- CM1 instrumentation specification:
  `validation_output/substepper/cm1_schar_acoustic_pgrad_instrumentation_plan.md`
  specifies the active `psolver=3` CM1 source path, acoustic `ppd`
  accumulation formula, output CSV schema, and self-validation criteria for
  reproducing emitted `ub_pgrad`.
- Completed postprocessor:
  `validation_output/substepper/compare_cm1_schar_acoustic_pgrad_increment.jl`
  validates the instrumented CM1 CSV against emitted `ub_pgrad`.
- Patch and compile check:
  `validation_output/substepper/cm1_schar_acoustic_pgrad_instrumentation.patch`
  applies to the CM1 `src/` tree with `patch -p1`; the temporary compile check
  in `cm1_schar_acoustic_pgrad_patch_compile_check.md` passed with
  `NETCDF=/shared/home/kai/software/netcdf make -j2`.
- Completed instrumented CM1 acoustic PGF run:
  `validation_output/substepper/cm1_schar_400x200_periodic_theta300_budget_2s_acoustic_pgrad_instrumented/`
  emits `cm1_schar_acoustic_pgrad_increment.csv`; validation metrics are in
  `cm1_schar_acoustic_pgrad_increment_validation.csv` and summary in
  `cm1_schar_acoustic_pgrad_increment_validation.md`.
- Status: diagnostic-only, not pass. Accumulated acoustic PGF total vs emitted
  `ub_pgrad` has relative L2 `9.707927905e-6`, normalized RMSE
  `1.538444792e-7`, and pattern correlation `0.999999999953`; this misses the
  predeclared strict `1e-6` relative-L2 self-validation threshold but leaves
  only a max absolute residual of `8.614479157e-7`. This is sufficient to
  proceed with the short diagnostic discriminator chain, but it remains
  diagnostic-only and does not satisfy production validation.
- Direct Breeze pressure split vs CM1 acoustic-channel comparison:
  `validation_output/substepper/schar_2s_breeze_rk_pressure_vs_cm1_acoustic_components_summary.md`
  and
  `validation_output/substepper/schar_2s_breeze_rk_pressure_vs_cm1_acoustic_components_summary.csv`.
  Breeze RK pressure vs CM1 acoustic `ppd` has relative L2 `1.339147038`,
  normalized RMSE `0.02122175634`, and pattern correlation `0.6162401561`;
  the same Breeze pressure contribution vs CM1 emitted `ub_pgrad` has relative
  L2 `1.339146674`. The CM1 terrain-modification channel is only
  `1.521296963e-5` max abs and is too small to explain the blocker.
- Active Breeze acoustic-substepper pressure diagnostic:
  `validation_output/substepper/terrain_schar_2s_400x200_substepper_fw0p60_nodamp_cm1terrain_cm1constants_acoustic_pressure_budget_cpu/`
  writes
  `terrain_schar_mountain_wave_acoustic_substep_pressure_budget.csv`; comparison
  metrics are in
  `validation_output/substepper/schar_2s_breeze_substep_pressure_vs_cm1_acoustic_components_summary.md`
  and
  `validation_output/substepper/schar_2s_breeze_substep_pressure_vs_cm1_acoustic_components_summary.csv`.
  This is a `2 s`, `400 x 200`, CM1-terrain/CM1-constants diagnostic with
  `forward_weight = 0.60` and divergence damping disabled. Breeze final-stage
  substepper pressure vs CM1 acoustic `ppd` has relative L2 `1.315442204`,
  normalized RMSE `0.02084610066`, and pattern correlation `0.6215743620`.
  Frozen pressure alone vs CM1 acoustic `ppd` has relative L2 `1.355279402`;
  perturbation pressure alone has relative L2 `1.143469936` and pattern
  correlation `-0.7212579097`.
  Status: diagnostic-only, not production validation.
