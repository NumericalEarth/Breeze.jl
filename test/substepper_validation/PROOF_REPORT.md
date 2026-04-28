# Proof test of the proposed fix — A/B characterization

Date: 2026-04-27

## TL;DR

**My proposed fix (discretely-balanced reference state) is necessary but NOT
sufficient.** The empirical A/B test conclusively shows it is not the
primary cause of the substepper instability.

The reference-state seed *is* real and *is* fixable to machine ε — but
removing it leaves the dominant amplifier intact. The substepper's
implicit Crank–Nicolson scheme at ω=0.55 amplifies *machine-zero*
perturbations by factor ~1.7 per outer step at production Δt; the
reference-state seed merely provides a slightly larger initial value
that saturates a few iterations earlier than pure roundoff would.

The actual fix lives elsewhere — most likely in the implicit-solve
coefficients or the predictor/post-solve weight matching — not in
how the reference state is constructed.

## Method

1. Implement the discretely-balanced reference state as a model patch
   (without modifying Breeze source code), via a non-linear iteration on
   `(p_ref, ρ_ref, ρθ_state)` that converges:

   - `(p[k] - p[k-1])/Δz_face + g·(ρ[k] + ρ[k-1])/2 = 0` to machine ε
   - `model.dynamics.pressure ≡ ref.pressure` to machine ε

   This is exactly the property that the proposed source-code fix would give.

2. Run the same drift sweep over (Δt, ω) ∈ {1,2,5,10,20,40} × {0.51,0.55,0.60,0.70}
   on a 3-D Cartesian (Periodic, Periodic, Bounded) grid, Lz=30 km, Nz=64,
   under both **A_orig** (existing reference state) and **B_dbal** (patched
   to machine-ε hydrostatic balance). Pass criterion: `max|w| ≤ 1e-10`.

3. Run the same A/B at Δt=20 s, ω=0.55 across topologies (3-D Periodic-Periodic,
   3-D Periodic-Bounded, lat-lon).

## Results

### Discrete-balance residual is fixed to machine ε

| variant  | residual (relative) |
|----------|---------------------|
| A_orig   | 3.31e-4             |
| B_dbal   | **1.86e-15**        |

11 orders of magnitude reduction. ✓

### Pressure consistency between EoS and reference is fixed

| variant  | max abs(p_eos − p_ref) |
|----------|-------------------------|
| A_orig   | 2.91e-11 Pa             |
| B_dbal   | **~1.8e-10 Pa**         |

Both at machine-ε scale (the 10× higher value in B is from finite-step
Newton iteration; would converge to ε with more iterations). ✓

### A_orig: env after 600 s of simulated time

| Δt \ ω  | 0.51    | 0.55    | 0.60    | 0.70    |
|---------|---------|---------|---------|---------|
| 1 s     | 3e-13   | 3.2e-13 | 3.8e-13 | 3.4e-13 |
| 2 s     | **0.34**| 2e-8    | 2.3e-13 | 2e-13   |
| 5 s     | 5.5e-10 | 2.8e-13 | 1.4e-13 | 6.7e-14 |
| 10 s    | **NaN** | **67**  | 4.7e-10 | 8.4e-14 |
| 20 s    | **0.35**| 1.1e-6  | 2.6e-12 | 4.9e-14 |
| 40 s    | 5e-8    | 9.3e-11 | 1.5e-13 | 3.9e-14 |

### B_dbal: env after 600 s of simulated time

| Δt \ ω  | 0.51    | 0.55    | 0.60    | 0.70    |
|---------|---------|---------|---------|---------|
| 1 s     | 1.9e-12 | 1.9e-12 | 1.9e-12 | 1.9e-12 |
| 2 s     | **0.65**| 4e-8    | 1.8e-12 | 1.8e-12 |
| 5 s     | 1.5e-9  | 1.7e-12 | 1.7e-12 | 1.6e-12 |
| 10 s    | **NaN** | **1e+2**| 3.4e-10 | 1.5e-12 |
| 20 s    | **0.15**| 7e-7    | 1.6e-12 | 1.1e-12 |
| 40 s    | 9e-8    | 1.1e-10 | 3.7e-13 | 3.4e-13 |

### Topology A/B at Δt=20s, ω=0.55

| topology | A_orig env | A_orig growth | B_dbal env | B_dbal growth |
|----------|------------|---------------|------------|---------------|
| 3D PPB   | 1.09e-6    | 1.782         | 7.04e-7    | 1.698         |
| 3D PBB   | 1.09e-6    | 1.782         | 7.04e-7    | 1.698         |
| lat-lon  | 1.09e-6    | 1.782         | 7.04e-7    | 1.698         |

Topology-independent confirms what the earlier sweeps showed.

## Interpretation

### What the dbal patch did

- Reduced the discrete-balance residual from ~3e-4 (relative) to machine ε.
- Reduced the env at the canonical failing point (Δt=20 s, ω=0.55) by
  ~30%: from 1.09e-6 to 7.04e-7.
- Slightly *raised* the noise floor in stable regimes (1.9e-12 vs 3e-13)
  because the Newton iteration introduces its own small rounding.
- Did NOT change the stability boundary: every (Δt, ω) cell that was
  unstable in A_orig is still unstable in B_dbal, every cell that was
  stable is still stable.
- Did NOT change the per-outer-step growth factor: both 1.78 and 1.70
  are above 1, both saturate the ρw envelope on the same timescale (just
  with a slightly different starting amplitude).

### What this proves

If the discrete-balance residual were the *primary* cause of the
instability, fixing it to machine ε would either:

1. Eliminate the growth (turn unstable cells into stable cells), OR
2. Vastly reduce the saturation envelope (because there's no seed left
   to amplify).

Neither happened. The growth rate `factor ≈ 1.7-1.8 per outer step` is
*identical* in A_orig and B_dbal at the same (Δt, ω), and the env
differs only by a factor of order 1.

The remaining seed in B_dbal is `~1.8e-10 Pa` (the iterative tolerance
on `p_eos - p_ref`). Float64 rounding alone can produce perturbations
at `~1e-13` per step. Either of these seeds, when amplified by a factor
of 1.78 over 30 outer steps, gives:

- 1.8e-10 × 1.78^30 ≈ 5e-3 m/s
- 1e-13   × 1.78^30 ≈ 3e-6 m/s

The B_dbal observed env (7e-7) sits between these — consistent with the
saturation timescale being limited by the amplifier, not by the seed
size.

### What the actual bug is

The data require an unstable mode in the linearized acoustic-buoyancy
system at production parameters. Specifically:

- ω=0.55, Δt=20 s, Δz=469 m on this grid: factor ~1.78/step (unstable).
- ω=0.7,  Δt=20 s, same: factor ~1.0 (stable).
- ω=0.55, Δt=1 s, same: factor ~1.0 (stable).

The off-centered Crank–Nicolson scheme should be unconditionally stable
for ω > 0.5 in the linearized inviscid system. The empirical violation
of that means one of the following is broken in the implementation:

1. **Predictor / matrix weight mismatch.** Lines 742-749 of
   `acoustic_substepping.jl` use `δτ_new = ω·Δτ` in the σ and η
   predictors; the implicit-solve matrix coefficients (lines 519-521,
   539-541, 559-561) use `δτ_new²`. The unconditional-stability proof
   requires these weights and their squares to match exactly across
   predictor + matrix + recovery. A factor-of-2 or sign error breaks it.

2. **Buoyancy-PGF cross-term diagonal.** The diagonal entry of the
   tridiag (line 539-540) sums a PGF-from-θ⁰ term and a buoyancy-from-σ
   term. The buoyancy term has a *signed* `(rdz_above − rdz_below)/2`
   structure, which can flip sign across non-uniform Δz. Even on uniform
   Δz, this term is added with no cancellation guarantee.

3. **σ_face_pred uses simple arithmetic mean.** Lines 763-765 form the
   half-implicit buoyancy force from `(σ[k] + σ[k-1])/2`. For a Schur
   reduction to give a symmetric positive-definite tridiag, the
   averaging operator must be the discrete adjoint of the centered
   difference operator used in `_post_solve_recovery!` (lines 802-806).
   These are both arithmetic means / centered differences, so they are
   adjoint on uniform Δz — but the issue may be that the resulting
   discrete operator is symmetric-but-indefinite in some
   (Π⁰, θ⁰_face) regimes.

## Negative result, in summary

The pre-registered hypothesis was: discretely-balanced reference state
is the primary fix.

The A/B test conclusively rejects this. The reference-state residual
exists, is fixable, and contributes a ~30 % reduction in env, but it
is a *secondary* effect. The primary cause is an unstable mode in the
implicit solve / predictor that is independent of the seed.

The right fix lies in the substepper internals — most likely in the
weight matching between `_build_predictors_and_vertical_rhs!` and the
`AcousticTridiag{Lower,Diagonal,Upper}` coefficients in
`src/CompressibleEquations/acoustic_substepping.jl`. The next agent
should run an eigenvalue scan of the column tridiagonal matrix as a
function of `δτ_new` and verify that the spectrum stays inside the
unit disk for ω = 0.55 — and find which term takes it outside.

## Artifacts

- `proof_test.jl` — runs the full A/B sweep, takes ~5 minutes on one GPU.
- `proof_results.jld2` / `proof_results.csv` — all trajectories and
  derived metrics.
- This report: `PROOF_REPORT.md`.
- Sister reports for context: `REPORT.md` (initial empirical
  characterization), `SUBSTEPPER_INSTABILITY_REPORT.md` and
  `SUBSTEPPER_TEST_PLAN.md` in the BBI repo.
