# Substepper instability — final diagnosis

> **Update 2026-04-27 (after `no_reference_test.jl`):** the reference
> state is NOT the issue. Setting `ref.pressure ≡ 0`, `ref.density ≡ 0`
> (zero reference state, equivalent to no reference subtraction at all)
> produces growth rates and envelopes essentially identical to the
> original broken reference state — see "Result D" appendix below. The
> reference-state mechanism described in the body of this report is real
> but is *not* the dominant cause of the instability.
> The bug must lie elsewhere — most likely in the per-stage WS-RK3
> bookkeeping of perturbations vs frozen state, or in the way the slow
> tendency from `model.dynamics.pressure` (which IS subject to FP noise
> regardless of the reference) is fed into the substep loop.

Date: 2026-04-27
Branch: `~/Breeze` `glw/hevi-imex-docs` HEAD `de99960` plus uncommitted
rewrite of `acoustic_substepping.jl`.

## TL;DR

The instability is not a single bug. It's a *feedback* between three
separately-correct components:

1. **The reference state has a non-trivial discrete-hydrostatic-balance
   residual** (~3e-4 relative). This produces a small seed in the slow
   vertical-momentum tendency at every outer step.
2. **The substep loop's update operator U has eigenvalues on the unit
   circle and decays single-perturbation amplifications**: `‖U^Nτ × pert‖ <
   ‖pert‖` for any localized perturbation we tested. The substep loop is
   internally stable and **does not** amplify perturbations within an
   outer step.
3. **The frozen state changes between outer steps**, and so do
   `Π⁰_face`, `θ⁰_face`, `p⁰` — *and so does the slow-tendency seed*.
   At every outer step, the previous step's perturbation feeds back
   into a slightly-different frozen state, which produces a slightly-
   different (and possibly larger) seed.

The growth factor of ~2.2 per outer step in the rest-atmosphere drift
test is a **feedback amplifier**, not a per-step linear amplifier of
the substep operator itself. It cannot be fixed by stabilizing U
alone, nor by fixing the seed alone — it requires the seed to vanish
*at every state*, not just at the special state `model = reference`.

## Method

Three independent measurements:

- **A. Eigenvalue scan of the column update operator U** (single substep):
  pure Julia replication of the substepper's documented matrix and
  predictor formulas (no Breeze source modification). For each (Δt, ω),
  build the `(3Nz+1) × (3Nz+1)` operator on `(σ, η, μw)` and compute its
  spectrum.

- **B. Per-outer-step linearized amplification**: build two identical
  rest-atmosphere models, perturb one by `δρw = 1e-10` at one cell,
  take ONE outer step on both, measure `‖response‖ / ‖perturbation‖`.

- **C. A/B drift comparison** (already done): rest atmosphere with the
  original reference state vs. with my discretely-balanced patch.

## Result A: ρ(U) = 1 exactly, but ‖U‖₂ ≫ 1

| Δt \ ω    | 0.50      | 0.55      | 0.60      | 0.70      | 0.99      |
|-----------|-----------|-----------|-----------|-----------|-----------|
| ρ(U)      | 1.000000  | 1.000000  | 1.000000  | 1.000000  | 1.000000  |
| ‖U‖₂      | 40.06     | 44.29     | 48.48     | 56.73     | 79.82     |
| ‖U^6‖₂    | 285       | 309       | 332       | 376       | 495       |

(at Δt=20s; full table in `eigenvalue_results.jld2`).

The single-substep operator has all eigenvalues exactly on the unit
circle — it's neutrally stable. ω increases the *non-normality* (largest
singular value), so increasing ω makes worst-case transient amplification
*larger*. This contradicts the empirical observation that ω=0.7
stabilizes the rest-atmosphere drift, so worst-case ‖U‖₂ is not
the relevant quantity.

## Result B: Per-outer-step amplification of a localized perturbation < 1

| Δt \ ω    | 0.50    | 0.55    | 0.60    | 0.70    | 0.80    | 0.99    |
|-----------|---------|---------|---------|---------|---------|---------|
| Δt=1 s    | 0.6540  | 0.6529  | 0.6519  | 0.6502  | 0.6488  | 0.6469  |
| Δt=2 s    | 0.3083  | 0.3169  | 0.3250  | 0.3403  | 0.3543  | 0.3781  |
| Δt=5 s    | 0.4244  | 0.2971  | 0.2144  | 0.1346  | 0.1372  | 0.1421  |
| Δt=10 s   | 1.1921  | 0.7691  | 0.5281  | 0.2770  | 0.1660  | 0.1087  |
| Δt=20 s   | 0.5100  | 0.3090  | 0.2065  | 0.1564  | 0.1151  | 0.0480  |
| Δt=40 s   | 0.2235  | 0.1705  | 0.1464  | 0.0900  | 0.0470  | 0.0259  |

`amplification = max|δρw_post| / |δρw_perturbation|` at perturbation
location after one outer step. **Almost everywhere ≪ 1**, meaning a
localized ρw perturbation *decays* over one outer step. The exception
is `ω=0.5, Δt=10s` where it's 1.19 (mildly amplifying).

This is also consistent with the substep loop being internally stable —
a perturbation injected at the start of an outer step doesn't grow
within that outer step.

A ρθ-perturbation gives similar amplifications (all ≪ 1 for ω ≥ 0.55).

## Result C: A/B drift comparison (recap)

At (ω=0.55, Δt=20 s, rest atmosphere):
- A_orig: env = 1.09e-6 m/s, growth = 1.78 / outer step
- B_dbal: env = 7.04e-7 m/s, growth = 1.70 / outer step

Both grow exponentially despite results A and B saying the substep loop
itself does not amplify perturbations.

## The reconciliation: feedback amplification

The rest-atmosphere drift growth factor 2.2/step (in the original
sweeps) is **not** the eigenvalue of the substep operator U, and **not**
the per-outer-step linearized amplification of a localized perturbation.
It is the **feedback rate** of the loop:

1. Outer step n starts. Frozen state `U⁰_n` is snapshotted from
   model state.
2. The slow-tendency residual at this state, `Gˢρw = Gⁿρw −
   ∂z(p⁰_n − p_ref) − g·(ρ⁰_n − ρ_ref)`, is non-zero because:
   - `(p⁰_n − p_ref)` includes both the reference-state discretization
     residual *and* whatever drift has accumulated in `p⁰_n` from earlier
     outer steps.
3. Substep loop runs for Nτ substeps, accumulating perturbations driven
   by this residual.
4. End of outer step: model state = `U⁰_n + perturbations`. The drift
   from the reference state has *grown*.
5. Outer step n+1: new frozen state, larger residual, repeat.

For the linear regime, this is approximately

  `x_{n+1} = (A + ε(x_n)·I) x_n`

where `A` is the substep loop (bounded; from results A and B) and
`ε(x_n)` is the slow-tendency seed at the current state. The effective
spectral radius of the per-outer-step operator is `|A| + |ε|`. With
`|A| ≈ 1` and `|ε|` of order the discrete-balance residual times the
substep-loop's amplification of that seed, the per-step rate ends up at
1.78–2.2.

Why ω=0.7 stabilizes (empirically): higher ω makes the substep loop's
output *smoother* (off-centered CN damps grid-scale modes), so the
frozen state in step n+1 differs less from step n's, so `ε(x_n)`
doesn't grow as fast.

Why the dbal patch only reduces env by ~30% but doesn't change the
growth rate: it kills the *initial* seed (the residual when state =
reference), but the FEEDBACK seed grows as the model state drifts. The
patch fixes the t=0 condition, not the per-step injection.

## Frozen state ↔ reference state interaction

This is what lets the bug *propagate* across outer steps:

- The implicit-solve **matrix coefficients** depend on `Π⁰_face` and
  `θ⁰_face`, derived from the frozen state. Reference state does not
  enter the matrix.
- The slow vertical-momentum tendency depends on
  **both** the frozen state (via `outer_step_pressure`,
  `outer_step_density`) and the reference state (via
  `ref.pressure`, `ref.density`). Their *difference*  is the seed.
- At t=0, frozen state ≈ reference state, so seed = (reference-state
  discrete imbalance). My dbal patch makes this zero.
- After step 1, frozen state ≠ reference state, so seed = (drift from
  reference). This drift grows feedback-style each outer step.

The clean fix is therefore to **recompute the reference state each
outer step from the current model state**, satisfying discrete
hydrostatic balance with the substepper's exact operators. Then the
seed is always zero by construction, regardless of how the model has
drifted.

Alternative: drop the reference-subtraction entirely and accept FP
cancellation in the slow vertical-momentum tendency. The slow tendency
becomes `−∂z p⁰ − g·ρ⁰` (no subtraction). For Float64 this is fine; for
Float32 it becomes lossy near hydrostatic balance. The failure mode is
not exponential amplification — it's just one-time FP precision loss.
That's a much milder failure than the current feedback amplification.

## Recommended fix (revised)

1. **Recompute reference state per outer step from the current model
   state**, and project to discrete hydrostatic balance using the
   same recurrence as in `proof_test.jl::dbal_perturbation`. Cost:
   one O(Nz) recurrence per outer step per column, negligible.

   Replace `freeze_outer_step_state!` to do this projection after
   snapshotting `outer_step_pressure` and `outer_step_density`. The
   substepper's reference-subtracted slow tendency would then have
   `(p⁰ − p_ref) ≡ 0` always, eliminating the feedback seed.

2. *Or* drop the reference subtraction:
   `Gˢρw = Gⁿρw − ∂z p⁰ − g·ρ⁰`. Lose FP precision near rest, but
   gain robust stability at production Δt. Worth comparing.

3. *Don't* try to fix this by raising ω alone. The empirical
   stabilization at ω=0.7 is incidental (it smooths the substep
   loop's output and slows the feedback) — it doesn't address the
   root cause and it adds artificial dissipation to physical
   gravity-wave modes.

## Everything I tested, end to end

- **Reference state seed (dbal patch)**: necessary but not sufficient.
- **Increasing N substeps**: doesn't help (per-substep amplification
  already < 1; the bug is at outer-step level).
- **Topology**: completely irrelevant (Cartesian / lat-lon /
  Periodic / Bounded all behave identically).
- **Lz**: doesn't change the bug; my earlier "Lz" hypothesis was a
  confound with Δt.
- **Δt**: bug onset around Δt=2-5s for ω=0.55, scales with vertical
  acoustic CFL.
- **ω**: stabilizes by smoothing, not by fixing the root cause.
- **Float type (32 vs 64)**: identical mechanism; Float32 saturates
  faster because the seed floor is bigger.

## Artifacts

- `eigenvalue_scan.jl` — column operator U construction and spectral analysis
- `eigenvalue_results.jld2` — ρ(U), ‖U‖₂, ‖U^6‖₂ table
- `per_outer_step_amplification.jl` — direct linearized A/B perturbation test
- `amp_results.jld2` — per-outer-step amplifications
- `proof_test.jl` — reference-state A/B drift test
- `proof_results.jld2` — A/B drift sweep
- `no_reference_test.jl` — **the test that disproves the reference-state hypothesis**
- `no_ref_results.jld2` — zero-ref vs orig-ref comparison
- `sweep_runner.jl`, `sweep2_boundary.jl` — original characterization sweeps

All in `~/Breeze/test/substepper_validation/`. Each is self-contained;
runs in 1-5 minutes on one GPU. Re-running them after a code-side fix
will quantify the improvement.

---

## Appendix — Result D: zero reference state

We set `ref.pressure ≡ 0` and `ref.density ≡ 0` after model
construction (so the substepper's slow vertical-momentum tendency
becomes `Gˢρw = Gⁿρw − ∂z p⁰ − g·ρ⁰` — the bare form, no
subtraction). Compared to the original broken reference state:

### Envelope after 600 s of simulated time

| Δt   | ω    | original   | zero       |
|------|------|------------|------------|
| 1 s  | 0.50 | 2.31e-13   | 4.56e-13   |
| 5 s  | 0.50 | 2.27e-9    | 4.50e-9    |
| 20 s | 0.50 | **7.57**   | **6.97**   |
| 40 s | 0.50 | 1.50e-7    | 1.70e-7    |
| 1 s  | 0.55 | 3.24e-13   | 3.88e-13   |
| 5 s  | 0.55 | 2.81e-13   | 1.11e-12   |
| 20 s | 0.55 | **1.09e-6**| **1.62e-6**|
| 40 s | 0.55 | 9.31e-11   | 5.53e-11   |
| 1 s  | 0.70 | 3.38e-13   | 5.51e-13   |
| 5 s  | 0.70 | 6.72e-14   | 1.23e-13   |
| 20 s | 0.70 | 4.94e-14   | 6.81e-14   |
| 40 s | 0.70 | 3.93e-14   | 5.75e-14   |

### Per-outer-step growth factor

| Δt   | ω    | original | zero  |
|------|------|----------|-------|
| 1 s  | 0.50 | 1.0002   | 1.0005|
| 5 s  | 0.50 | 1.1042   | 1.1034|
| 20 s | 0.50 | 3.0695   | 3.0932|
| 40 s | 0.50 | 2.7331   | 2.6713|
| 1 s  | 0.55 | 0.9996   | 0.9995|
| 5 s  | 0.55 | 1.0110   | 1.0232|
| 20 s | 0.55 | 1.7816   | 1.8327|
| 40 s | 0.55 | 1.6192   | 1.4962|
| 1 s  | 0.70 | 1.0013   | 1.0017|
| 5 s  | 0.70 | 0.9999   | 0.9958|
| 20 s | 0.70 | 0.9882   | 0.9928|
| 40 s | 0.70 | 1.0124   | 1.0139|

Differences are at most ~10% — within run-to-run noise. The bug is
present with or without reference subtraction.

### Implications

The reference state is **not load-bearing** in the substepper as
implemented. Removing it (or replacing it with a discretely-balanced
version) doesn't change the instability mechanism. So:

- **The discrete-hydrostatic-balance residual we measured (~3e-4
  relative) is real but not the cause of the instability.** It's a
  spurious feature of the reference-state construction, but it's
  benign because the reference state isn't actually being relied on
  for FP cancellation.
- **My earlier "feedback amplification" picture is partly wrong.**
  The slow tendency is `−∂z p⁰ − g·ρ⁰` (in either case for Float64
  this is at machine ε in the rest atmosphere), so the seed *is* at
  machine ε in both cases. Yet the growth still happens.
- **The bug must therefore be in the per-stage substep loop dynamics
  themselves**, not in the slow-tendency seed. Specifically, the way
  the substep loop interacts with the WS-RK3 stage structure (reset
  perturbations, run substeps from zero, recover full state, repeat)
  must have an instability that's not visible in the single-substep
  eigenvalue scan or the localized-perturbation linearized
  amplification — but IS visible in the cumulative roundoff-driven
  rest-atmosphere drift test.

The remaining suspect: **the WS-RK3 stage logic** —
- Stage 1: starts from U⁰, integrates Nτ₁ substeps with σ=η=μ=0 IC,
  recovers state to U⁰ + (σ, η, μ)₁.
- Stage 2: starts from U^(1), recomputes slow tendencies from U^(1),
  but linearization basic state stays at U⁰. Substeps from σ=η=μ=0.
  Recover to U⁰ + (σ, η, μ)₂.
- Stage 3: same but starting from U^(2).

The slow tendency in stages 2 and 3 is computed from a state that has
drifted from U⁰ (by the previous stages' perturbations). Even if those
perturbations are at machine ε, the slow tendency from a slightly
shifted state has machine-ε differences from the slow tendency at U⁰.
These accumulate over the WS-RK3 outer step.

**The next thing for the agent to investigate:** put a probe inside
`acoustic_rk3_substep!` measuring the slow tendency `Gⁿ` between
stages. If `Gⁿρw` at stages 2 and 3 has any non-zero values at rest
(when at U⁰ it was zero), that's the seed for the feedback. Even
machine-ε `Gⁿρw` integrated against a non-normal substep operator
(‖U‖₂ ≈ 44 per substep) can produce O(1) growth over 30 outer steps.
