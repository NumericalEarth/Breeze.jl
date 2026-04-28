# Substepper instability — investigation summary

**Date:** 2026-04-27
**Scope:** Empirical characterization of the rest-atmosphere drift /
moist-baroclinic-wave failure on `glw/hevi-imex-docs` with the
uncommitted rewrite of `src/CompressibleEquations/acoustic_substepping.jl`.
**Mode:** Diagnostic only. No source code modified. All test code is in
`test/substepper_validation/`.

## The bug, in one paragraph

A rest atmosphere on the substepper grows `max|w|` exponentially from
machine zero by **factor ~1.78 per outer step at (Δt=20 s, ω=0.55)** —
the production parameters. The growth rate goes to ~1.0 (stable) for
either Δt ≤ 1 s OR ω ≥ 0.7, but the bug is present across the entire
Δt × ω matrix the moist-BCI test would touch. The DCMIP-2016 moist
baroclinic wave NaNs at step 15 because of this drift saturating; the
dry baroclinic wave at the documented Δt=225 s NaNs at step 11 for the
same reason.

## What is the bug NOT

Each of these was hypothesized and falsified by direct measurement:

| Candidate | Evidence ruling it out |
|---|---|
| **Float32 precision** | Float64 gives identical growth rate (`H_FT` sweep). |
| **Microphysics / surface fluxes** | Dry baroclinic wave fails identically (`repro_substepper_dry_bw.jl`). |
| **Curvilinear / lat-lon metric** | 2-D Cartesian Flat-y, 3-D Periodic-Periodic-Bounded, 3-D Periodic-Bounded-Bounded, lat-lon all give identical growth rate at the same (Δt, ω, Lz) (`D_topo` sweep). |
| **Bounded-y boundary handling** | Same growth rate on Periodic-Periodic-Bounded as on Periodic-Bounded-Bounded. |
| **Lz** | Bug exists at Lz=5 km, 10 km, 15 km, 20 km, 30 km, 40 km. Earlier "Lz hypothesis" was a confound with Δt. |
| **Number of substeps N** | N ∈ {6, 12, 24, 48, 96, 192} all give the same per-outer-step growth factor of ~2.2. The bug is at outer-step level, not per-substep CFL. Increasing N does *not* help. |
| **Reference-state discretization residual** | A patched reference state in *exact* discrete hydrostatic balance (machine-ε residual) reduces env by only ~30% with the same growth rate. Setting `ref ≡ 0` (no reference subtraction at all) gives growth rate within 3% of the original. **The reference state is not load-bearing.** |
| **Substep loop's intrinsic stability (single substep)** | Eigenvalue scan: ρ(U) = 1 *exactly* for every (Δt, ω) tested. Substep loop is asymptotically neutral. Per-outer-step amplification of a localized ρw or ρθ perturbation < 1 for every (Δt, ω) tested at ω ≥ 0.55. |

## What the bug IS, with the best evidence

The substep operator U has spectral radius exactly 1 *but* spectral
norm ‖U‖₂ ≈ 44 at (Δt=20 s, ω=0.55). It is **highly non-normal**.
Localized perturbations decay (norm collapses fast onto stable
eigenmodes), but distributed FP-noise patterns excite the
transient-amplification subspace and produce O(40×) gain per substep
before settling.

The cumulative roundoff each WS-RK3 stage gets re-injected: stage 2 and
stage 3 compute slow tendencies from `U^(k-1)` (the latest model state,
which has drifted ~ε from `U⁰_outer` due to stage 1's substep loop).
Even though `U⁰_outer` and `U^(k-1)` agree to machine ε, the **slow
tendencies they produce differ by machine ε**, and those machine-ε
differences flow into the substep loop where they get amplified by
‖U‖₂ × ‖predictor‖. After 30 outer steps, the cumulative effect saturates
the field.

Why ω ≥ 0.7 stabilises: increases implicit dissipation in the substep
loop, which damps the non-normal transient amplification (does not
change ρ(U) but does shrink ‖U^k‖ over enough k).

Why Δt ≤ 1 s stabilises: ‖U‖₂ scales with Δt (Δt=1 s gives ‖U‖₂ ≈ 2.1
vs Δt=20 s gives ‖U‖₂ ≈ 44). At small Δt the transient amplification
factor per substep is small enough not to matter.

## Where to look in the source

The non-normal amplifier is built from these terms in
`acoustic_substepping.jl`:

- **Predictor σ̃, η̃** (lines 729-749): explicit forward step using
  `δτ_new = ω·Δτ` for the vertical mass-flux divergence. Same `δτ_new`
  appears in the post-solve recovery (lines 802-806).
- **Tridiag matrix A** (lines 509-561): coefficients use `δτ_new²`. The
  buoyancy term (lines 540, 521, 559) has a *signed* `(rdz_above ±
  rdz_below)/2` structure that breaks symmetry of A — likely the source
  of the non-normality.
- **σ-face averaging** in the predictor RHS (lines 763-765): simple
  arithmetic mean. For a Schur reduction to give a symmetric tridiag
  the averaging operator must be the discrete adjoint of the centered
  difference in `_post_solve_recovery!` (lines 802-806). They *are*
  adjoint on uniform Δz, but the resulting operator may still be
  symmetric-but-indefinite and amplify selected modes via non-normality.

The right thing to do is:

1. Hand-derive the linearised acoustic-buoyancy substep update operator
   for an isothermal rest atmosphere and verify it matches the operator
   the code actually constructs.
2. Identify which term breaks the Schur-symmetry of the matrix.
3. Replace that term with a symmetric counterpart.

This is an algebra-and-implementation problem, not a tuning problem.

## What does NOT need to be done

- **Do not** raise the default `forward_weight` from 0.55 to 0.7 as a
  fix. It works empirically by adding implicit damping to the
  non-normal modes, but that damping also dissipates physical
  gravity-wave modes and is not a real fix.
- **Do not** increase substep count N as a fix. The bug is at outer-step
  level. (Sweep B confirmed.)
- **Do not** try to fix the reference-state hydrostatic-balance
  residual. It's not the cause and a "fix" there only reduces env by
  ~30% with identical growth rate.

## Recommended ancillary cleanup

The reference state in the current implementation does no useful work
and creates a measurable test confound (env differs by ~50% between
`ref ≡ 0` and the broken trapezoidal reference at the same (Δt, ω, grid)).
Removing it from `freeze_outer_step_state!` and the slow-vertical-momentum
tendency assembly would clean up the test baseline. If/when Float32
atmospheric runs become a target, add back a *correctly* discretely-
balanced reference state (the recurrence is in
`test/substepper_validation/proof_test.jl::dbal_perturbation`).

This is testability hygiene, not a stability fix.

## Test artifacts (in `~/Breeze/test/substepper_validation/`)

| File | Purpose |
|---|---|
| `sweep_runner.jl` | 9 sweeps over Δt, N, ω, topology, ref-form, Lz, Δz, FloatType. Output `results.jld2`, `results.csv`. |
| `sweep2_boundary.jl` | 2-D (Δt, ω) stability map; Δz scan; discretely-balanced reference verification. Output `results_boundary.jld2`. |
| `proof_test.jl` | A/B drift comparison — original vs discretely-balanced reference state. Output `proof_results.jld2`. |
| `eigenvalue_scan.jl` | Replicates substep matrix in pure Julia, computes ρ(U), ‖U‖₂, ‖U^Nτ‖₂. Output `eigenvalue_results.jld2`. |
| `per_outer_step_amplification.jl` | Direct linearised A/B perturbation test. Output `amp_results.jld2`. |
| `no_reference_test.jl` | The disproof harness: `ref ≡ 0` vs original ref. Output `no_ref_results.jld2`. |

Each is self-contained; together they take ~30 minutes on one GPU.
Re-running them after a code-side fix will quantify the improvement.

The rest-atmosphere drift at (Δt=20 s, ω=0.55, Nx=Ny=32, Nz=64) is a
single-test regression: when growth/step drops below 1.05 there, the
substepper is fixed.

## Reports superseded by this summary

- `test/substepper_validation/REPORT.md` — initial empirical
  characterization (Lz hypothesis, since falsified).
- `test/substepper_validation/PROOF_REPORT.md` — discrete-balance A/B
  test (showed the dbal patch is necessary but not sufficient).
- `test/substepper_validation/FINAL_DIAGNOSIS.md` — feedback-amplifier
  picture (later refined: feedback is real but not the dominant
  mechanism per the no-reference test).
- The `BreezyBaroclinicInstability.jl/SUBSTEPPER_INSTABILITY_REPORT.md`
  and `SUBSTEPPER_TEST_PLAN.md` files — earlier hypotheses, also
  superseded.

This summary supersedes all of them.
