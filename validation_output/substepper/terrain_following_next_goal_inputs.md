# Terrain-Following Validation: Next Goal Inputs

Status: metrics-generation goal has produced quantitative evidence, but
production validation is incomplete.

Latest production gate:

```text
production validation gate: pass=16 present=16 fail=14 missing=0 blocked=5
```

## Why The Current Goal Is Not Complete

The current production-validation metrics plan requires:

1. Breeze explicit stepping to match CM1 explicit stepping to 1% for each
   production validation case with a CM1 reference.
2. Breeze acoustic substepping to match Breeze explicit stepping to 1% for
   each production validation case where explicit stepping is feasible.
3. Full production or validation-grade runs, not smoke tests or reduced
   runtime diagnostics.

The Schar and complex-mountain evidence is now broad enough to diagnose the
gap: production runs, saved-time scalar metrics, saved-time field metrics,
movies/plots, required comparison regions, robustness metrics, resolution
notes, and manifest provenance are present. The metrics fail the 1% gates.

Askervein remains blocked because no accepted coordinate-faithful production
workflow or explicit-feasible production window exists.

## Required Full-Run Validation Standard

No smoke test, short diagnostic, or reduced-grid run can satisfy the next
goal. A claimed pass requires:

- production or validation-grade resolution declared before running;
- full required physical runtime;
- machine-readable metrics from the same production output;
- saved-time comparison metrics at every saved production/reference time;
- human-inspectable movie or plot from the same production output;
- explicit `production_validation` artifact class;
- pass/fail summary against the 1% contract.

For Schar, no run shorter than `6 h` can satisfy validation.

For complex mountain, the declared production runtime is `6 h`; any alternate
runtime or resolution must be declared before running and justified by the
case physics.

For Askervein, the full LES validation must include spin-up and an averaging
window; the explicit-vs-substepper window may be shorter only if it is
declared before running and uses the accepted production grid or a documented
infeasibility exception.

## Tighter Accuracy Metrics

Field metrics must pass in the primary comparison region and be reported in
the full domain:

- `relative_l2_error <= 0.01`
- `relative_linf_error <= 0.01`
- `normalized_rmse <= 0.01`
- `maximum_amplitude_error <= 0.01`
- `pattern_correlation >= 0.99`

Wave and projection metrics:

- `phase_error_wavelengths <= 0.01`
- `projection_amplitude_error <= 0.01`
- `best_shift_projection_amplitude_error <= 0.01`

Drag metrics:

- drag sign convention must be explicitly labeled;
- `mountain_drag_relative_error <= 0.01` or pressure-drag component relative
  errors `<= 0.01`;
- sign must match the declared convention.

Saved-time metrics:

- final-state pass is insufficient;
- every saved production/reference frame must satisfy the 1% field gate;
- scalar time-series comparisons must satisfy the 1% gate at every saved time.

Robustness metrics:

- `nan_count = 0`
- `inf_count = 0`
- `bottom_normal_velocity_max_abs` within precision tolerance
- no late-time monotonic growth in high-k near-terrain energy
- no unresolved late-time reflection growth above the sponge start

## Schar Remaining Work

Current production evidence:

- CM1 explicit, Breeze explicit, and Breeze acoustic substepper production
  runs exist at matched `400x200` resolution.
- Production runtime reaches `6 h`.
- Saved-time scalar and field metrics exist.
- Movies/plots and required region metrics exist.
- Three-resolution convergence evidence exists.

Current failures:

- explicit-vs-CM1 below-sponge field/phase/drag gate fails:
  `u_relative_l2_error = 0.026636791805590348`,
  `w_relative_l2_error = 1.7549539691829341`,
  `w_normalized_rmse = 0.11204348858250109`,
  `w_pattern_correlation = 0.43100073886205914`,
  `mountain_drag_relative_error = 3.3597789530483224`.
- substepper-vs-explicit below-sponge field/phase/drag gate fails:
  `u_relative_linf_error = 0.015416752531768444`,
  `w_relative_l2_error = 0.22638797656312776`,
  `w_normalized_rmse = 0.029339205804194135`,
  `w_pattern_correlation = 0.9754128236898442`,
  `mountain_drag_relative_error = 0.6269783052655318`.
- saved-time substepper-vs-explicit scalar gate fails:
  `294 / 407` rows fail; worst relative error is
  `41.565943649924705` for `mountain_drag`.
- saved-time substepper-vs-explicit field gate fails:
  `72 / 74` rows fail; worst `relative_l2_error = 0.9579903914564901`
  for `pressure_perturbation`.
- saved-time explicit-vs-CM1 field gate fails:
  `74 / 74` rows fail; worst `relative_l2_error = 2.3224117e11`
  for pressure.

Next work:

- diagnose and fix the explicit-vs-CM1 field/pressure/drag mismatch;
- diagnose and fix the remaining substepper-vs-explicit pressure/vertical
  velocity/drag mismatch;
- rerun full `6 h` production explicit and substepper cases after fixes;
- regenerate saved-time scalar metrics, saved-time field metrics, movies, and
  gate report.

## Complex Mountain Remaining Work

Current production evidence:

- CM1 explicit, Breeze explicit, and Breeze acoustic substepper production
  runs exist at `120x120x150`.
- Production runtime reaches `6 h`.
- Required 3D comparison regions are present:
  `below_sponge`, `full_domain`, `near_terrain`, `centerline_slice`,
  `lee_side_box`, and `hilltop_box`.
- Saved-time scalar and field metrics exist.
- Explicit and substepper CM1 comparison movies exist.

Current failures:

- explicit-vs-CM1 below-sponge field/drag gate fails:
  `u_relative_l2_error = 0.11624580774050561`,
  `v_relative_l2_error = 1.6843374389067955`,
  `w_relative_l2_error = 3.7768447403611067`,
  `theta_relative_l2_error = 1.7658682588906576`,
  `pressure_relative_l2_error = 7.60472948572488`,
  and pressure-drag signs do not match CM1.
- substepper-vs-explicit below-sponge field/drag gate fails:
  `u_relative_l2_error = 0.019977663988256778`,
  `v_relative_l2_error = 0.3902356616733189`,
  `w_relative_l2_error = 0.7640921516428109`,
  `theta_relative_l2_error = 0.2229552644588148`,
  `pressure_relative_l2_error = 0.8266102311331819`.
- saved-time substepper-vs-explicit scalar gate fails:
  `419 / 481` rows fail; worst relative error is
  `14.85330164373352` for `pressure_drag_y`.
- saved-time substepper-vs-explicit field gate fails:
  `144 / 148` rows fail; worst `relative_l2_error = 1.0313818112185893`
  for `pressure_perturbation`.
- saved-time explicit-vs-CM1 field gate fails:
  `148 / 148` rows fail; worst `relative_l2_error = 2.082781302869588e11`
  for pressure.

Next work:

- separate CM1 setup/physics differences from Breeze dycore differences;
- diagnose pressure and drag sign/magnitude disagreement;
- diagnose substepper-vs-explicit pressure and vertical-velocity drift;
- rerun full `6 h` production explicit and substepper cases after fixes;
- regenerate saved-time scalar metrics, saved-time field metrics, movies, and
  gate report.

## Askervein Remaining Work

Current evidence:

- WEMEP/Zenodo files are present.
- ERF terrain and WEMEP mast parsing have diagnostic coverage.
- ERF-terrain explicit-vs-substepper diagnostic brackets include required 3D
  regions and pass through `1.2 s`, then fail by `1.25 s`.

Current blockers:

- no accepted coordinate-faithful Breeze production LES exists;
- no declared production boundary-condition workflow exists;
- no declared spin-up and averaging windows exist;
- no production LES time series exists;
- no accepted all-transect WEMEP reference comparison exists;
- no accepted explicit-feasible production comparison window exists.

Next work:

- define the coordinate-faithful production setup:
  WEMEP/ERF terrain, forcing, boundary conditions, roughness/surface layer,
  spin-up, averaging window, runtime, and sampling plan;
- implement the Breeze production workflow if it is in scope;
- run full production LES after the setup is accepted;
- run the explicit-feasible comparison window on the accepted grid or document
  a justified infeasibility exception;
- generate WEMEP transect/profile metrics, production plots or movies, and
  pass/fail summaries.

## Suggested Next Goal

Fix and validate terrain-following production accuracy so that Schar and
complex mountain pass 1% explicit-vs-CM1 and substepper-vs-explicit metrics
over full production runs, while defining or implementing the coordinate-
faithful Askervein production validation workflow with quantitative pass/fail
gates.
