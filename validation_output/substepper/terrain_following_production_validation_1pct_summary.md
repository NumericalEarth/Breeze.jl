# Terrain-Following Production Validation 1% Summary

Status: incomplete. Quantitative metrics have been generated for Schar and
complex mountain production artifacts, and diagnostic/blocker metrics have
been generated for Askervein. The active 1% accuracy contract is not satisfied.

Latest gate:

```text
production validation gate: pass=16 present=21 fail=21 missing=0 blocked=5
```

## Contract

The production-validation plan requires:

- Breeze explicit vs CM1 explicit to pass 1% field, wave/projection, and drag
  gates where a CM1 reference exists.
- Breeze acoustic substepper vs Breeze explicit to pass the same 1% gates where
  explicit stepping is feasible.
- Smoke and diagnostic artifacts to be excluded from completion.

## Schar 6 h 400x200

Production artifact coverage: complete for CM1 explicit, Breeze explicit,
Breeze acoustic substepper, saved-time metrics, movies/plots, and
three-resolution convergence.

1% status: fail.

- explicit-vs-CM1 below-sponge field/phase/drag gate: fail.
  `u_relative_l2_error = 0.026636791805590348`,
  `w_relative_l2_error = 1.7549539691829341`,
  `w_normalized_rmse = 0.11204348858250109`,
  `w_pattern_correlation = 0.43100073886205914`,
  `mountain_drag_relative_error = 3.3597789530483224`.
- substepper-vs-explicit below-sponge field/phase/drag gate: fail.
  `u_relative_linf_error = 0.015416752531768444`,
  `w_relative_l2_error = 0.22638797656312776`,
  `w_normalized_rmse = 0.029339205804194135`,
  `w_pattern_correlation = 0.9754128236898442`,
  `mountain_drag_relative_error = 0.6269783052655318`.
- explicit-vs-CM1 projection-amplitude gate: fail.
  Below-sponge `projection_amplitude_error = 0.17606602967141627`.
- substepper-vs-explicit projection-amplitude gate: fail.
  Below-sponge `projection_amplitude_error = 0.09475538706521736`.
- saved-time explicit-vs-CM1 field gate: fail.
  `74 / 74` rows fail; worst `relative_l2_error = 2.3224117e11` for
  pressure at `0 s`.
- saved-time substepper-vs-explicit scalar gate: fail.
  `294 / 407` rows fail; worst relative error `41.565943649924705` for
  `mountain_drag` at `10200 s`.
- saved-time substepper-vs-explicit field gate: fail.
  `72 / 74` rows fail; worst `relative_l2_error = 0.9579903914564901` for
  `pressure_perturbation` at `10798.200000006214 s`.
- diagnostic Lin finite-volume pressure-gradient discriminator: fail and
  excluded from source. The `2 s`, `400 x 200`, WENO-9 diagnostic produced
  `nan_count = 372392`; the operator-budget summary still reported
  `ub_pgrad` relative L2 `1.575561862` and `wb_pgrad` relative L2
  `1.529498472`.
- early-time acoustic pressure-history discriminators: diagnostic only. Active
  substepper pressure self-closes to roundoff, but the dominant horizontal
  channel still fails against CM1 acoustic `ppd` (`relative_l2 = 1.317131151`).
  Pressure-vs-Exner, one-face shifts, sign flip, ungated first-substep
  pressure, post-recovery pressure replay, and nonlinear acoustic-state
  pressure reconstruction do not close the gap. The nonlinear replay follows
  the bad ungated branch (`relative_l2 = 1.363666148` total,
  `1.364949064` horizontal).
- matched outer-`dt` discriminator: fail. Completed Slurm job `1089` ran
  `400 x 200`, `6 h`, `SCHAR_DT = 0.35`, grid terrain, no divergence damping,
  and no acoustic upper sponge. It improves the no-damping/no-sponge
  `dt = 2 s` production result but still fails the 1% gate:
  below-sponge `w_relative_linf_error = 0.0745803850`,
  `w_relative_l2_error = 0.1232164948`,
  `w_normalized_rmse = 0.0159684898`,
  `w_projection_amplitude_error = 0.0179820855`, pressure relative L2
  `0.6617009969`, and mountain-drag relative error `0.4057695816`.

## Schar Low-Amplitude Linear Wave 6 h 400x200

Production artifact coverage: complete for Breeze substepper and Breeze
explicit control, including machine-readable linear metrics and final-state
PPM plots.

1% status: fail.

- substepper-vs-analytical-linear-wave gate: fail.
  Below-sponge, excluding boundary faces, `w_relative_l2_error = 1.6310697784`,
  `w_relative_linf_error = 0.4566711505`,
  `w_normalized_rmse = 0.0971352861`,
  `w_pattern_correlation = 0.4962314148`,
  `projection_amplitude_error = 0.0687402754`, and best shift is `0` cells.
  Robustness passes with `nan_count = 0` and `inf_count = 0`.
- explicit-vs-analytical-linear-wave control: fail.
  Below-sponge, excluding boundary faces, `w_relative_l2_error = 1.8079439240`,
  `w_relative_linf_error = 0.5198932319`,
  `w_normalized_rmse = 0.1076686924`,
  `w_pattern_correlation = 0.4605970664`,
  `projection_amplitude_error = 0.0634816092`, and best shift is `0` cells.
  This indicates the analytical-reference convention is not yet a standalone
  pass/fail oracle for the finite-domain run.
- substepper-vs-explicit Tier-1 low-amplitude gate: fail.
  Below-sponge `w_relative_l2_error = 0.2267029486`,
  `w_relative_linf_error = 0.1583762049`,
  `w_normalized_rmse = 0.0309254096`,
  `w_pattern_correlation = 0.9757036753`,
  `w_projection_amplitude_error = 0.1004766690`,
  `pressure_relative_l2_error = 0.9280320257`, and
  `mountain_drag_relative_error = 0.6306652106`.
- exact-`w̃` substepper-vs-analytical-linear-wave production gate: fail.
  The regenerated `400 x 200`, `6 h`, GPU substepper artifact
  `linear_mountain_wave_production_400x200_6h_gpu_wtilde/` includes exact
  `w_tilde` columns and `field = w_tilde` rows. Below-sponge
  `w_tilde_relative_l2_error = 0.8352048912`,
  `w_tilde_relative_linf_error = 0.3833304476`,
  `w_tilde_normalized_rmse = 0.0821621241`,
  `w_tilde_pattern_correlation = 0.7344266758`, and
  `w_tilde_projection_amplitude_error = 0.1033396488`.
- exact-`w̃` explicit-control-vs-analytical-linear-wave production gate: fail.
  The regenerated `400 x 200`, `6 h`, GPU explicit artifact
  `linear_mountain_wave_explicit_production_400x200_6h_gpu_wtilde/` includes
  exact `w_tilde` columns and `field = w_tilde` rows. Below-sponge
  `w_tilde_relative_l2_error = 0.9243801650`,
  `w_tilde_relative_linf_error = 0.4364041061`,
  `w_tilde_normalized_rmse = 0.0909346181`,
  `w_tilde_pattern_correlation = 0.6997871856`, and
  `w_tilde_projection_amplitude_error = 0.1009619580`.

## Complex Mountain 6 h 120x120x150

Production artifact coverage: complete for CM1 explicit, Breeze explicit,
Breeze acoustic substepper, saved-time metrics, and explicit/substepper movies.

1% status: fail.

- explicit-vs-CM1 below-sponge field/drag gate: fail.
  `u_relative_l2_error = 0.11624580774050561`,
  `v_relative_l2_error = 1.6843374389067955`,
  `w_relative_l2_error = 3.7768447403611067`,
  `theta_relative_l2_error = 1.7658682588906576`,
  `pressure_relative_l2_error = 7.60472948572488`,
  and pressure-drag signs do not match CM1.
- substepper-vs-explicit below-sponge field/drag gate: fail.
  `u_relative_l2_error = 0.019977663988256778`,
  `v_relative_l2_error = 0.3902356616733189`,
  `w_relative_l2_error = 0.7640921516428109`,
  `theta_relative_l2_error = 0.2229552644588148`,
  `pressure_relative_l2_error = 0.8266102311331819`.
- explicit-vs-CM1 projection-amplitude gate: fail.
  Below-sponge `projection_amplitude_error = 0.94392136146878`.
- substepper-vs-explicit projection-amplitude gate: fail.
  Below-sponge `projection_amplitude_error = 0.6706262595274035`.
- saved-time explicit-vs-CM1 field gate: fail.
  `148 / 148` rows fail; worst `relative_l2_error = 2.082781302869588e11`
  for pressure at `0 s`.
- saved-time substepper-vs-explicit scalar gate: fail.
  `419 / 481` rows fail; worst relative error `14.85330164373352` for
  `pressure_drag_y` at `10800 s`.
- saved-time substepper-vs-explicit field gate: fail.
  `144 / 148` rows fail; worst `relative_l2_error = 1.0313818112185893` for
  `pressure_perturbation` at `18000 s`.

## Askervein Hill

Production artifact coverage: blocked.

1% status: blocked.

- WEMEP/ERF reference files and diagnostic named-mast comparisons are present.
- ERF-terrain explicit-vs-substepper diagnostic brackets pass through `1.2 s`
  and fail by `1.25 s` due to `w_tilde_relative_linf_error`.
- No accepted coordinate-faithful production LES exists with declared
  boundary conditions, spin-up, averaging window, production runtime, and
  reference comparison.
- A field-comparable Askervein LES also needs missing model setup capability:
  precursor or recycling inflow, rough-wall/MOST-compatible surface treatment,
  and a declared spin-up/averaging workflow.
- No accepted production explicit-feasible window exists.

## Conclusion

The metrics-generation goal has enough quantitative evidence to identify the
current failures, but the production-validation contract is not complete:
Schar and complex mountain fail the 1% gates, and Askervein remains blocked by
missing production-definition and production-output requirements.
