# Terrain-Following Production Validation Metric Schema Audit

Status: Schar and complex-mountain production schema complete; Askervein
blocked/incomplete.

This audit checks the active metrics artifacts against the metric families
required by `terrain_following_production_validation_metrics_plan.md`. It is
separate from the 1% accuracy gate: a run can have comparison metrics and still
miss required robustness or time-series fields.

Update: the Breeze Schar explicit, Schar substepper, and complex-mountain
production artifacts were rerun after the robustness retrofit and now emit the
required direct robustness metrics. The older derived proxy files remain useful
for provenance but are no longer the acceptance evidence for those cases.

## Required Production Robustness Metrics

The plan requires every production run to emit:

- `maximum_cfl`
- `maximum_acoustic_cfl`
- `nan_count`
- `inf_count`
- `mass_relative_drift`
- `bottom_normal_velocity_max_abs`
- `high_k_energy_fraction_near_terrain`
- `reflection_energy_fraction_above_sponge_start`
- `walltime_per_step`

Retrofitted scripts:

- `terrain_schar_mountain_wave_explicit_validation.jl`
- `terrain_schar_mountain_wave_validation.jl`
- `complex_mountain_doernbrack_validation.jl`

Smoke checks:

- `schar_robustness_smoke_explicit/terrain_schar_mountain_wave_metrics.csv`
- `schar_robustness_smoke_substepper/terrain_schar_mountain_wave_metrics.csv`
- `complex_mountain_robustness_smoke_substepper/complex_mountain_metrics.csv`

## Schar 6 h 400x200

Checked files:

- `terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_metrics.csv`
- `terrain_schar_6h_400x200_production_explicit/terrain_schar_mountain_wave_energy_timeseries.csv`
- `terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_metrics.csv`
- `terrain_schar_6h_400x200_production_substepper/terrain_schar_mountain_wave_energy_timeseries.csv`

Present:

- final scalar metrics: `normalized_rmse`, `amplitude_error`, `maximum_w`,
  `maximum_reference_w`, `mean_momentum_flux`, `mountain_drag`,
  `maximum_cfl`, `maximum_acoustic_cfl`, `nan_count`, `inf_count`,
  `mass_relative_drift`, `bottom_normal_velocity_max_abs`,
  `high_k_energy_fraction_near_terrain`, `walltime_per_step`,
  `wall_clock_seconds`, `wall_clock_seconds_per_simulated_hour`, `stable_dt`
- time-series metrics: `time`, `lower_energy`, `below_sponge_energy`,
  `sponge_energy`, `below_sponge_to_lower_energy`, `sponge_to_lower_energy`,
  `reflection_energy_fraction_above_sponge_start`, `maximum_cfl`,
  `maximum_acoustic_cfl`, `nan_count`, `inf_count`,
  `mass_relative_drift`, `bottom_normal_velocity_max_abs`,
  `high_k_energy_fraction_near_terrain`, `walltime_per_step`

Schema status: `complete` for the completed Schar production Breeze runs.

## Schar Low-Amplitude Linear Wave 6 h 400x200

Checked files:

- `linear_mountain_wave_production_400x200_6h_gpu/linear_mountain_wave_state_metrics.csv`
- `linear_mountain_wave_production_400x200_6h_gpu/linear_mountain_wave_scalar_metrics.csv`
- `linear_mountain_wave_production_400x200_6h_gpu/terrain_schar_mountain_wave_energy_timeseries.csv`
- `linear_mountain_wave_explicit_production_400x200_6h_gpu/linear_mountain_wave_state_metrics.csv`
- `linear_mountain_wave_explicit_production_400x200_6h_gpu/linear_mountain_wave_scalar_metrics.csv`
- `linear_mountain_wave_explicit_production_400x200_6h_gpu/terrain_schar_mountain_wave_energy_timeseries.csv`
- `linear_mountain_wave_substepper_vs_explicit_400x200_6h_gpu/schar_substepper_vs_explicit_state_metrics.csv`

Present:

- analytical comparison rows for `below_sponge` and `full_domain`;
- wave metrics: `relative_l2_error`, `relative_linf_error`,
  `normalized_rmse`, `pattern_correlation`, `maximum_amplitude_error`,
  `projection_amplitude_error`, `best_shift_projection_amplitude_error`,
  `best_shift_cells`, and `phase_error_wavelengths`;
- robustness row with finite-value counts, stable timestep, mass drift,
  high-k near-terrain energy, bottom-normal velocity, and CFL values;
- final-state human-inspectable PPM plots for explicit and substepper runs.

Schema status: `complete` for the low-amplitude linear-wave production
artifacts. Accuracy status is `fail`; see the production gate report.

## Complex Mountain 6 h 120x120x150

Checked files:

- `complex_mountain_production_explicit/complex_mountain_metrics.csv`
- `complex_mountain_production_explicit/complex_mountain_timeseries.csv`
- `complex_mountain_production_substepper/complex_mountain_metrics.csv`
- `complex_mountain_production_substepper/complex_mountain_timeseries.csv`
- `complex_mountain_production_cm1_reference/complex_mountain_metrics.csv`

Present:

- final scalar metrics: `maximum_w`, `maximum_u`, `maximum_v`,
  `maximum_pressure_perturbation`, `domain_kinetic_energy`,
  `below_sponge_w_energy`, `sponge_w_energy`,
  `reflection_energy_fraction_above_sponge_start`, `pressure_drag_x`,
  `pressure_drag_y`, `maximum_cfl`, `maximum_acoustic_cfl`, `nan_count`,
  `inf_count`, `mass_relative_drift`, `bottom_normal_velocity_max_abs`,
  `high_k_energy_fraction_near_terrain`, `walltime_per_step`,
  `wall_clock_seconds`,
  `wall_clock_seconds_per_simulated_hour`, `stable_dt`
- time-series metrics: `time`, `maximum_w`, `maximum_u`, `maximum_v`,
  `maximum_pressure_perturbation`, `domain_kinetic_energy`,
  `below_sponge_w_energy`, `sponge_w_energy`,
  `reflection_energy_fraction_above_sponge_start`, `pressure_drag_x`,
  `pressure_drag_y`, `maximum_cfl`, `maximum_acoustic_cfl`, `nan_count`,
  `inf_count`, `mass_relative_drift`, `bottom_normal_velocity_max_abs`,
  `high_k_energy_fraction_near_terrain`, `walltime_per_step`
- CM1 converted metrics: `maximum_w`, `pressure_drag_x`, `pressure_drag_y`,
  `time`

Schema status: `complete` for the completed complex-mountain production Breeze
runs.

## Time-Series Cadence

The production gate now verifies that the completed Schar and complex-mountain
Breeze production time series reach the declared `6 h` validation time with
maximum saved-time gaps no larger than `600 s` within tolerance:

- Schar explicit: final time `21600 s`, maximum gap `599.9 s`.
- Schar substepper: final time `21600 s`, maximum gap `600 s`.
- Complex explicit: final time `21600 s`, maximum gap `600 s`.
- Complex substepper: final time `21600 s`, maximum gap `600 s`.

The production gate also verifies that those time series contain the plan's
required robustness, reflection, high-k, and drag fields:

- common fields: `time`, `maximum_w`, `maximum_u`,
  `maximum_pressure_perturbation`, `domain_kinetic_energy`,
  `mass_relative_drift`, `maximum_cfl`, `maximum_acoustic_cfl`,
  `high_k_energy_fraction_near_terrain`,
  `reflection_energy_fraction_above_sponge_start`
- Schar drag field: `mountain_drag`
- complex-mountain drag fields: `pressure_drag_x`, `pressure_drag_y`

## Askervein

Checked files:

- `askervein_les_production/askervein_les_metrics.csv`
- `askervein_explicit_substepper_compare_production/askervein_explicit_substepper_metrics.csv`
- `askervein_wemep_reference/askervein_les_production_askervein_wemep_mast_metrics.csv`

Present:

- LES scalar metrics: `maximum_horizontal_cfl`,
  `maximum_contravariant_vertical_cfl`, mast speed-up/direction diagnostics,
  high-k ratios for speed-up and `w_tilde`, spectra/plot paths
- explicit-vs-substepper field comparison metrics for full-domain and
  near-terrain regions; the refreshed ERF-terrain `1.2 s` GPU diagnostic also
  emits `centerline_slice`, `lee_side_box`, and `hilltop_box`
- WEMEP named-mast fractional-speed-up diagnostic for overlapping named masts

Missing or not named as required:

- production spin-up and averaging-window metrics
- production LES time series
- accepted observational/reference comparison over all required transects
- `maximum_cfl` and `maximum_acoustic_cfl` under the plan's exact names
- `nan_count`
- `inf_count`
- `mass_relative_drift`
- `bottom_normal_velocity_max_abs`
- `walltime_per_step`
- coordinate-faithful WEMEP transect/profile comparison files from a
  production LES

Schema status: `blocked/incomplete`.

This is not just a column-name cleanup. A field-comparable Askervein
production artifact still needs production setup capability that is not present
in the current validation scripts: precursor or recycling inflow, rough-wall /
MOST-compatible surface treatment, declared spin-up, and declared averaging
windows. Until that workflow exists, any Askervein file with the required
metric names would still be diagnostic rather than production-validation
evidence.

## Comparison Metric Coverage

The comparison CSVs for Schar, complex mountain, and Askervein generally
include:

- `relative_l2_error`
- `relative_linf_error`
- `bias`
- `normalized_rmse`
- `maximum_amplitude_error`
- `maximum_absolute_error`
- `pattern_correlation`
- `one_percent_pass`

Coverage gaps:

- Schar wave-specific phase metrics exist in the Schar comparison files.
  Projection metrics are emitted by separate Schar projection post-processing
  files, including below-sponge, full-domain, and near-terrain regions:
  `schar_6h_400x200_explicit_vs_cm1_400x200_periodic_theta300_projection_metrics.csv`
  and `schar_6h_400x200_substepper_vs_explicit_projection_metrics.csv`.
- Complex mountain is treated as a terrain case rather than a wave-phase
  benchmark, so it has projection-amplitude diagnostics rather than
  Schar-style phase-shift diagnostics. The production projection files are
  `complex_mountain_explicit_vs_cm1_projection_metrics.csv` and
  `complex_mountain_substepper_vs_explicit_projection_metrics.csv`; they cover
  below-sponge, full-domain, near-terrain, centerline-slice, lee-side-box, and
  hilltop-box regions.
- Drag diagnostics exist for Schar and complex mountain comparisons, but
  convention-best drag is not present for the older Schar production files.
- Saved-time CM1-vs-Breeze frame-pair diagnostics now exist for Schar and
  complex mountain. The Schar field-error frame-pair file records relative
  L2/Linf and maximum-absolute errors for `w` and pressure at every matched
  saved frame. The complex-mountain frame-pair file records the same error
  families for `w` and pressure over the 37 matched production frames.

The production gate now explicitly verifies completed-case region coverage:
Schar has below-sponge, full-domain, and near-terrain metric files for both
production comparison pairs; complex mountain has `below_sponge`,
`full_domain`, `near_terrain`, `centerline_slice`, `lee_side_box`, and
`hilltop_box` rows in both production comparison CSVs.

Additional Schar Tier-1 discriminator coverage now exists in
`schar_substepper_vs_explicit_tier1_6h_no_damping_no_upper_sponge_grid_schema_refresh_coordcheck/`.
That comparator output includes below-sponge and full-domain field rows with
the plan-required L2, L∞, bias, max-normalized RMSE, pattern-correlation,
amplitude, phase-shift, projection-amplitude, and best-shift projection
metrics. It also emits coordinate-parity rows for `coordinate_x` and
`coordinate_z` and refuses to compare state slices whose physical coordinates
do not match. This discriminator is schema-complete for the Tier-1 Schar
question, but it still fails the 1% accuracy gate.

## Conclusion

The validation goal has produced enough quantitative evidence to expose the
current accuracy gaps. The direct robustness metric schema is complete for the
completed Schar and complex-mountain production Breeze runs. The remaining
schema gap is Askervein, because the available Breeze artifacts are diagnostic
and do not yet define a coordinate-faithful production run, accepted spin-up and
averaging window, or explicit-feasible comparison window.
