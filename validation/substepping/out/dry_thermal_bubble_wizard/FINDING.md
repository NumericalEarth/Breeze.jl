# Thermal bubble: the ring artifact is a discretization error that scales with outer Δt

## Summary

At `t = 400 s` the dry thermal bubble run under `PressureProjectionDamping(0.5) +
forward_weight = 0.8 + wizard cfl = 0.3` shows a localized "ring" of high `w` at the
plume top that does **not** appear in either the anelastic reference or a fully
explicit compressible run at Δt = 0.1 s. This is not real physics — it is a
truncation error in the substepper that grows with the outer Δt.

Peak vertical velocity at `t = 400 s`:

| run                                              | max \|w\| at t=400 s |
|--------------------------------------------------|---------------------:|
| anelastic                                        |        29.5 m/s     |
| **compressible explicit, Δt = 0.1 s (ground truth)** |    **29.6 m/s** |
| substepper, wizard cfl = 0.3 (avg Δt ≈ 0.5–1 s)      |     **38.4 m/s** |
| substepper, Δt = 0.1 fixed                       |        30.0 m/s     |
| substepper, Δt = 0.25 N = 12 fixed               |        30 m/s (matches) |

Anelastic and explicit agree to 0.3 %. Explicit is the correct reference. The
substepper at Δt = 0.1 reproduces explicit within 1.5 %; at Δt = 0.25 it also
matches. Only the wizard-cfl=0.3 configuration, which lets Δt drift to
~0.5–1.0 s during the plume's active phase, produces the ring.

## What the 4-panel stills show

See `frame4_t0300.png`, `frame4_t0400.png`, `frame4_t0500.png` (side-by-side
anelastic / explicit / wizard substepper / tight Δt=0.1 substepper).

- t = 300 s: all four agree — clean rising plume.
- t = 400 s: anelastic, explicit, and tight-Δt substepper show a smooth mushroom
  cap at `z ≈ 6–7 km`. The wizard substepper shows the **same** mushroom cap
  **plus** a localized high-amplitude ring at `(x = 0, z ≈ 9 km)`.
- t = 500 s: anelastic, explicit, and tight-Δt substepper agree. The wizard
  substepper still has its ring artifact and downstream noise.

The peak-`|w|`(t) curve in `peak_w_3way.png` shows this crisply: explicit,
anelastic, and both tight-Δt substepper variants lie on top of one another;
only the wizard-cfl=0.3 line spikes.

## Hypothesis for the failure mode

The substepper's slow vertical tendency `tend_w_euler = -∂_z p_frozen - g(ρ⁰ − ρ_base)`
is computed once per outer step from the **outer-step-start state `U⁰`** (in
`_convert_slow_tendencies!`, `acoustic_substepping.jl:636`). This value is held
frozen across all three WS-RK3 stages and across every acoustic substep within
each stage. MPAS does this same freeze — but MPAS's calibration is for
broad, slowly-evolving flows (BCI-scale). A compact thermal bubble with
`Δθ = 10 K` in a nearly neutral atmosphere hits `|w| ≈ 30 m/s` over `Δz = 78 m`
in the plume's leading edge, which means `ρ` and `p` at a given grid point
change by O(10 %) per outer Δt at the bubble top. Freezing `tend_w_euler` at
`U⁰` across the whole outer step therefore carries an O(Δt) error that is
much larger than the scheme's nominal order.

Evidence for this being the dominant error:

1. The error amplitude depends on the *outer* Δt, not on the substep size.
   `Δt = 0.25 s` with `N = 12` (Δτ = 0.02 s acoustic substeps) works cleanly,
   while wizard-cfl=0.3 (Δt ≈ 0.5 s) with `N = 6` (Δτ = 0.08 s) fails. Same
   substep CFL, very different outer-step accuracy.

2. The ring appears at the plume leading edge, where `∂_t ρ` and `∂_t p` are
   largest — exactly where the frozen `tend_w_euler` is most wrong.

3. Damping strategy and substep count do not remove the ring. An earlier
   sweep (REPORT.md §3.A) confirmed the same insensitivity.

## Recommended fix

Re-evaluate `tend_w_euler` (and, probably, `tend_u_euler`) at the current RK
stage state instead of freezing it at `U⁰`. This is what WRF does (see
`docs/src/appendix/bw_dt_sweep_results.md` for MPAS/CM1 discussion); the
MPAS-style freeze is a tuning choice made for their benign operational
configurations, not an inherent requirement of the Klemp-Skamarock-Dudhia
scheme. The cost is one extra EOS/pressure evaluation per stage, which is
negligible compared to the substep loop.

A quick sanity check: after the fix the wizard-cfl=0.3 run should match explicit
and anelastic at t = 400 s. If it still shows a ring, a secondary contributor
is `freeze_outer_step_state!` (freezes `model.dynamics.pressure` across all 3
RK stages); that one may also need to be re-evaluated per stage.

## Implications for the other validation cases

The BOMEX / RICO / neutral_ABL "NaN at iter ≤ 100" failures in REPORT.md §3.B
are very likely the same bug scaled up: the frozen vertical PGF+buoyancy can't
track a 3D turbulent field where local `ρ` and `p` change as fast as in the
bubble. The fix above should also unblock those cases.
