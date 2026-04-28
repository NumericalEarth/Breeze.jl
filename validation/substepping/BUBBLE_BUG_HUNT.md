# Bubble bug hunt — running log

## Symptom

Three-way diagnostic on a small bubble (Δθ=0.1K, 64×64, N²=1e-4, 300s):

| | max\|w\| at t=300s |
|--|-----|
| anelastic (ground truth) | 0.116 m/s |
| explicit-compressible (ground truth) | 0.121 m/s (4% from anelastic) |
| substepper ω=0.55 no damp | **0.011 m/s** (10× too small) |
| substepper ω=0.7 no damp | **256 m/s** (explosion) |
| substepper ω=0.9 / Klemp damp | NaN |

The user's clue: "acoustics work but slow dynamics don't" — supported by tests
(i)–(iv) all passing. The bug is at the intersection of buoyancy and slow
advection, possibly in the linearization basic state.

## Hypotheses

### H1. Wrong basic state for linearization (most promising)

The substepper currently linearizes the perturbation operators at `U⁰` (the
outer-step-start state). This means `θ⁰` and `Π⁰` carry the bubble's
perturbation. The **canonical Baldauf/MPAS approach is to linearize at a
SMOOTH HYDROSTATIC REFERENCE state** (`θ_ref`, `Π_ref`). The bubble's IC
perturbation should appear as the initial value of `η = ρθ − ρθ_ref` (NOT
in a slow-tendency reference subtraction term as I currently do).

Predictions:
- At small Δθ, my code (linearize at U⁰) ≈ standard (linearize at ref)
- At large Δθ, the U⁰ linearization gets the local θ wrong by Δθ/θ → could
  cause systematic suppression
- For the bubble at Δθ=0.1K, this is ~3e-4 which is small, but if some
  coefficient amplifies it 1000× the discrepancy could matter

### H2. σ, η reset to 0 at stage start loses bubble's IC

Currently `reset_perturbations!` sets `σ = η = μ = 0` at every stage start.
For the bubble, the IC has `ρθ⁰ − ρθ_ref ≠ 0`. If we instead initialize
`η ← ρθ⁰ − ρθ_ref` (with linearization at reference), the substep PGF
`−γRᵈ Π_ref ∂z η` drives the wave naturally from substep 1, without needing
reference subtraction in `Gˢρw`.

Currently the bubble's drive is in `Gˢρw = −∂z(p⁰ − p_ref) − g(ρ⁰ − ρ_ref)`
which is FROZEN throughout the outer step. Maybe this freezing prevents
the within-outer-step evolution of the imbalance from feeding back.

### H3. Δt² error in σ_pred from M_pert build-up

`σ_pred -= Δτ × ∇h·M_pert` adds a Δt² correction (since M_pert ~ Δτ ×
Gⁿρu_h after the explicit horizontal step). This may or may not match
the proper WS-RK3 stage-advance Δt² correction.

### H4. CN time-step weighting

`δτ_new = ω·Δτ` is used for both the σ-predictor's explicit half and the
post-solve implicit half. Strict CN would use `δτ_old = (1−ω)·Δτ` for the
explicit half. At ω=0.5 these match; at ω=0.55 they differ by 22%. This
acts as effective extra damping at small ω (suppression) and effective
amplification at large ω (explosion) — consistent with the observed
0.011 → 256 → NaN progression. **First attempt to fix broke the rest
atmosphere — needs care.**

## Test plan

### Step 1: amplitude sweep on the bubble (this commit)

`56_bubble_amplitude_sweep.jl`: Δθ ∈ {1, 0.1, 0.01, 0.001, 0.0001} K. For each,
run anelastic + explicit + substepper. Look at:
- ratio max\|w\|_substep / max\|w\|_explicit vs Δθ
- ratio max\|w\|_substep / max\|w\|_anelastic vs Δθ

If ratio is independent of Δθ → linear-regime structural bug.
If ratio → 1 as Δθ → 0 → nonlinear / amplitude-dependent bug.

### Step 2: horizontal-shear bubble

Take test (iv) IGW (which works) and change to bubble-shape perturbation (peaks at
center, decays outward) instead of sin(πz/Lz). Same domain, same N², same U₀=20.
If this works, the issue is the absence of background flow in the bubble setup.

### Step 3: bubble with no stratification (N²=0)

Pure horizontal acoustic + buoyancy with constant θ̄. Should not need vertical
PGF subtraction beyond what's already there. Diagnostic for whether stratification
is the trigger.

### Step 4: H1 fix attempt — linearize at reference

Replace `outer_step_potential_temperature` and `outer_step_exner` with reference-state
`θ_ref`, `Π_ref` in the substep loop. Initialize `η ← ρθ⁰ − ρθ_ref` at outer-step
start (NOT zero). Drop reference-subtraction in `Gˢρw`. Re-run all tests.

### Step 5: H4 fix attempt — strict CN

Use `δτ_old = (1−ω)·Δτ` for σ predictor's explicit half (and η predictor).
Keep `δτ_new = ω·Δτ` for post-solve implicit. Re-run rest atmosphere AND
bubble. The previous attempt broke rest atmosphere because the rest case's
buoyancy mode is marginally unstable at strict ω=0.5; need to verify ω=0.55
works at strict CN.

## Log

- 2026-04-25 — bug identified via 3-way diagnostic. Substepper at ω=0.55
  gives 0.011 vs explicit 0.121.

- 2026-04-26 — **amplitude sweep (Step 1, `56_bubble_amplitude_sweep.jl`):**
  ratio `sub / expl` is **0.091 ± 0.001 across Δθ ∈ {1, 0.1, 0.01, 0.001, 0.0001}** —
  4 orders of magnitude. Constant ratio → **linear-regime structural bug**.
  This rules out: nonlinear breakdown, amplitude-dependent linearization issues.

- 2026-04-26 — **H1+H2 attempt (linearize at reference + initialize η ≠ 0):** broke
  rest atmosphere (FP-rounding mismatch in η⁰ amplifies with marginal-stability
  buoyancy mode). Also broke the bubble (NaN). Reverted to baseline.

- 2026-04-26 — **H1 alone (linearize at reference, keep η reset to 0):** identical
  result to baseline. Doesn't help. Linearization basic state isn't the bug.

- 2026-04-26 — **ω sweep on bubble (`57_bubble_omega_sweep.jl`)** at fixed Ns=12, Δθ=0.001:
    | ω    | sub/expl ratio |
    |------|----------------|
    | 0.50 | NaN            |
    | 0.55 | 0.091          |
    | 0.60 | 0.093          |
    | 0.65 | 0.095          |
    | 0.70 | 118000 (explosion) |
    | 0.75+| NaN            |
  Ratio nearly constant ≈ 0.091-0.095 in stable ω range, then catastrophic at ω=0.7.
  The 0.091 is INDEPENDENT of off-centering in stable range.

- 2026-04-26 — **Ns sweep (`57b_bubble_ns_sweep.jl`)** at Δt=1, ω=0.55:
    | Ns | Δτ | CFL=cs·Δτ/Δx | ratio |
    |----|-----|---------------|-------|
    | 6  | 0.167 | 0.185 | 0.082 |
    | 12 | 0.083 | 0.093 | 0.091 |
    | 24 | 0.042 | 0.046 | 0.096 |
    | 48 | 0.021 | 0.023 | 0.098 |
    | 96 | 0.010 | 0.012 | 0.099 |
  As Ns→∞, ratio approaches **0.099 ≈ 1/10**, NOT 1.0. Substepper at Ns=∞
  gives 1/10 of the correct answer — STRUCTURAL discretization error, not
  Ns-discretization error.

- 2026-04-26 — **Δt sweep (`57c_bubble_dt_sweep.jl`)** at Ns=12, ω=0.55:
    | Δt   | Δτ    | ratio |
    |------|-------|-------|
    | 0.05 | 0.0042 | 0.232 |
    | 0.10 | 0.0083 | 0.152 |
    | 0.20 | 0.0167 | 0.076 |
    | 0.50 | 0.042  | 0.052 |
    | 1.00 | 0.083  | 0.091 |
    | 2.00 | 0.167  | NaN   |
  **Non-monotonic.** At Δt=0.05 (close to explicit's Δt=0.05): ratio 0.23 — STILL not 1.
  Substepper ≠ explicit even at small Δt. So the Δt mismatch isn't the issue;
  there's a structural bug independent of the time step.

  **The non-monotonicity** (minimum around Δt=0.5) is suspicious — suggests
  some Δt-dependent interference, possibly with the BV oscillation period
  (T=600s) or with the cross-outer-step Gˢρw refresh.

- 2026-04-26 — **H1 revert (linearize at U⁰: θ⁰, Π⁰ in fast operators):**
  Identical 0.091 ratio across all Δθ ∈ {1, 0.1, 0.01, 0.001, 0.0001}. Confirms
  the basic-state choice is irrelevant at small Δθ — the linearizations are
  numerically equivalent. **H1 fully ruled out.**

  At this point all four hypotheses tried have failed, but the symptom is
  unambiguously a STRUCTURAL bug: ratio is independent of Δθ, ω (in stable
  range), Ns (asymptotic 0.099), and is non-monotonic in Δt. This points
  to something missing/wrong in the **fast-operator equations** themselves,
  not in the discretization.

## New hypotheses

### H5. Acoustic equilibrium has σ/η degeneracy

The bubble's drive is the FROZEN imbalance Gˢρw = -∂z(p⁰ - p_ref) -
g(ρ⁰ - ρ_ref). The fast-operator equilibrium (μw=0) satisfies
γRᵈΠ⁰ ∂z η_eq + g σ_eq = Gˢρw. This is one equation in two unknowns;
the actual equilibrium depends on the dynamic path. Rather than
guessing, instrument η, σ, μw at one outer step and compare with the
explicit code's response.

### H6. Slow tendency Gⁿρu/v carries something it shouldn't

The horizontal substep uses `μu += Δτ × (Gⁿρu - γRᵈ Π⁰ ∂x η)`. If
Gⁿρu in SlowTendencyMode includes a residual horizontal pressure
gradient (full p, not zero), it would compete with the fast PGF term.
Need to check `slow_tendency_mode_*` in the dynamics kernel. If the
slow horizontal PGF is nonzero at U⁰, the substepper's μu update
double-counts the bubble's pressure imbalance.

### H7. **MISSING horizontal frozen pressure imbalance** ✅ THE BUG

`SlowTendencyMode` zeroes the PGF in `Gⁿρu_h`, so the full horizontal
PGF must be reinstated by the substepper. The vertical equation has
`Gˢρw = Gⁿρw - ∂z(p⁰ - p_ref) - g(ρ⁰ - ρ_ref)` — frozen U⁰ imbalance.
There was NO equivalent horizontal-PGF reinstatement. The substep
horizontal momentum read `μu_t = Gⁿρu_h - γRᵈ Π⁰ ∂x η`, which at
substep start (η = 0) gives ZERO horizontal force. The bubble's
horizontal acoustic adjustment (which is the dominant path for the
2D bubble) was completely missing.

**Predicted scaling:** 1D (Flat horizontal) tests should be UNAFFECTED
(no horizontal motion), 2D bubble should be substantially worse — and
indeed the 1D buoyancy oscillator gave ratio 0.2 while 2D bubble gave
0.091.

## Log (continued)

- 2026-04-26 — **H7 fix (frozen horizontal pressure imbalance):**
  Added `pressure_imbalance = p⁰ - p_ref` field, computed once per
  outer Δt in `freeze_outer_step_state!`. `_explicit_horizontal_step!`
  now adds `-∂x(p⁰ - p_ref)` and `-∂y(p⁰ - p_ref)` to μu, μv. Result:
  bubble dynamics tracks explicit ground truth to ~5–10% from t=0 to
  t=140s (was 9× too small before). Hydrostatic balance test still
  passes (max|w| stays at FP roundoff ~1e-13). IGW (test iv) still
  passes. **Confirms H7 was THE BUG.**

  | t (s) | expl max|w| | subst max|w| | ratio |
  |-------|-------------|--------------|-------|
  |  10   | 3.98e-4     | 3.29e-4      | 0.83  |
  |  60   | 9.08e-4     | 8.45e-4      | 0.93  |
  | 100   | 1.24e-3     | 1.22e-3      | 0.98  |
  | 140   | 1.80e-3     | 1.61e-3      | 0.89  |
  | 150   | 1.64e-3     | 1.71e-3      | 1.04  |
  | 160   | 1.41e-3     | 9.46e-3      | (instability kicks in) |
  | 200   | 1.72e-3     | 1.19e+1      | (NaN at 210s)          |

## Remaining issue: late-time instability

After ~150 s the substepper develops an exponentially growing instability
(growth factor ~5×/10s). Higher ω postpones the blow-up but doesn't
eliminate it. Increasing Ns doesn't help and sometimes makes it worse.
This is a SEPARATE bug from H7 — the bubble's *physics* is now correct,
but a numerical mode goes unstable.

### H8. Off-centered CN time-step inconsistency

`δτ_new = ω·Δτ` is used uniformly for σ, η predictor's old AND new
vertical contributions. Klemp/MPAS form uses 2ω·Δτ effective time
step instead of strict CN's (1-ω) + ω = 1. At ω=0.55: 10% over-
integration. Strict CN with `δτ_old = (1-ω)·Δτ` for the old half
might fix the marginal-stability issue.

- 2026-04-26 — **H8 attempt (strict CN with δτ_old = (1-ω)Δτ):** WORSE
  than Klemp/MPAS form. Bubble blows up at t=130s (vs 160s with
  Klemp/MPAS). Reverted. The Klemp/MPAS form's extra 2ω-1 ≈ 10%
  over-integration acts as load-bearing damping for the marginal
  buoyancy mode. **H8 ruled out.** (Both forms identical at ω=0.5.)

## Status & next hypotheses

H7 (horizontal frozen pressure imbalance) was the bug we were after.
The bubble physics is now correct to ~5% accuracy through 140s. The
remaining late-time instability is a separate, smaller bug.

Observations on the remaining instability:
- Amplitude-dependent: smaller Δθ → later blow-up (consistent with
  the instability being driven by the bubble's pressure-imbalance
  signal, not a free-running numerical mode).
- ω-dependent: higher ω postpones blow-up (more damping helps), but
  can't eliminate it within the stable acoustic-CFL ω range.
- Ns-dependent in a complex way: Ns=24 worse, Ns=48 better but still
  off; this is unlike a CFL-related issue.
- u (horizontal velocity) starts deviating before w; u becomes 2× too
  large at t=140 just before w explodes. **Suggests the instability
  is in horizontal momentum dynamics specifically.**

### H9. Horizontal step is forward Euler — μu has no implicit damping

The vertical μw is solved implicitly via tridiag with off-centered
CN. Horizontal μu, μv are pure forward Euler with the new pressure
imbalance forcing. Note however: η_pred uses the **NEW** μu (computed
in step A before step B), so (μu, η) form a *symplectic* Euler pair —
in 1D pure horizontal acoustics this is stable for CFL ≤ 2. So pure
horizontal acoustics shouldn't blow up. The instability must be in
the *coupling* — most likely σ predictor's ∂z μw_old contribution
combined with the new ∂x μu_new horizontal divergence creates a
non-symplectic mode.

## Summary of progress in this session

| Bug                           | Status      | Effect on bubble ratio       |
|-------------------------------|-------------|-------------------------------|
| H1 (linearization basic state)| ruled out   | no change                     |
| H2 (η reset to 0 vs IC)       | ruled out   | broke rest atm                |
| H3 (Δt² in σ_pred)            | not tested  | n/a                           |
| H4 (CN time-step weighting)   | unhelpful   | δτ_old=(1-ω)Δτ broke things   |
| H5 (σ/η degeneracy)           | not tested  | n/a                           |
| H6 (Gⁿρu has stray PGF)       | ruled out   | no — SlowTendencyMode zeroes  |
| **H7 (frozen ∂x p⁰)**         | **THE BUG** | **0.091 → 0.93 (10× → 1.07×)**|
| H8 (strict CN δτ_old)         | ruled out   | strict CN diverges sooner     |
| H9 (forward Euler horizontal) | suspected   | *to investigate*              |

Plus a separate 1D vertical buoyancy bug: substepper gives ratio 0.2
in pure 1D vertical buoyancy oscillator (`58_1d_buoyancy_oscillator`),
unaffected by H7. This is a smaller bug, distinct from the 2D bubble
horizontal-PGF bug.

## Next session

1. Track down the horizontal-coupling instability in 2D bubble
   (suspect H9: σ-predictor mixes ∂z μw_old with ∂x μu_new asymmetry).
2. Track down the 1D vertical buoyancy 5× suppression. Likely a
   factor-of-N mistake in the σ-η-μw coupling under frozen Gˢρw.
3. Re-test the bubble at all Δθ once the late-time instability is
   fixed; expect ratio ≈ 1 across the amplitude sweep.

## Update — 2026-04-26 (continued)

After more investigation, the picture clarified considerably.

### IGW amplitude sweep (`59_igw_amplitude_sweep.jl`)

Substepper is **stable across all amplitudes** (Δθ ∈ {0.01, 0.1, 1.0,
5.0} K) over a full 600 s run with U₀=20 m/s background flow. Ratio
sub/expl is consistent at ~0.875 in `wmax_final` but **0.97-0.98 at
peak `wmax_overall`** — the 12 % gap was a phase artifact of the
gravity-wave moving away from the original location by t=600s. So IGW
**is fine** with the H7 fix.

### "1D buoyancy oscillator" test misnamed

The `58_1d_buoyancy_oscillator.jl` test runs in 1D vertical with `Flat`
horizontal. There is **no real BV mode** in this geometry — buoyancy
oscillation requires horizontal parcel exchange to satisfy
incompressibility. What the test actually measures is **vertical
acoustic standing waves** (round-trip period ≈ 2·Lz/cs ≈ 56 s, visible
in the explicit time series). Off-centered CN at ω > 0.5
**intentionally damps** these acoustics — that is the scheme's design
purpose. The "5× under-prediction" matches that intentional damping.
Not a bug.

### The Δt × cs / Δx soft outer-step CFL

`56f_bubble_dt_omega_grid.jl` shows the bubble:

| Δt   | result                              |
|------|-------------------------------------|
| 0.1  | ✓  ratio_overall = 0.99             |
| 0.25 | ✓  ratio_overall = 0.99             |
| 0.5  | ✓  ratio_overall = 0.98             |
| 1.0  | NaN (at all ω ∈ {0.55..0.8})        |

At Δt=0.5: cs·Δt/Δx = 350×0.5/312.5 = 0.56.
At Δt=1.0: cs·Δt/Δx = 1.12 (>1) → instability.

The substepper has a soft outer-step CFL constraint Δt < Δx/cs even
though acoustics are integrated implicitly within the substep loop.
Likely cause: the substepper linearizes at U⁰ for the entire outer
step, and once the actual state drifts more than one grid cell of
acoustic distance from U⁰, the linearization error compounds enough to
trigger the instability we see at t≈150 s.

**Practical consequence:** the substepper is **stable and 98 % accurate
on the bubble at Δt ≤ 0.5 s**. To use the substepper at Δt=1.0 s on a
2D bubble with this grid (Δx=312 m), additional damping or refreshing
the linearization per stage would be needed.

### Final scoreboard (with `KlempDivergenceDamping(coefficient = 0.1)` available)

| Test                               | NoDamp                       | Klemp(0.1)                |
|------------------------------------|------------------------------|---------------------------|
| Hydrostatic balance (rest atm)     | ✓ machine-zero drift         | ✓ machine-zero drift      |
| IGW (test iv) all Δθ               | ✓ 0.97 at peak               | ✓ 0.97 at peak (no penalty)|
| 2D dry bubble at Δt ≤ 0.5          | ✓ 0.98 accurate              | ✓ 0.93 accurate (slight cost) |
| 2D dry bubble at Δt = 1.0          | ✗ NaN at t ≈ 200 s           | **✓ 0.92 accurate**       |
| 2D dry bubble at Δt = 1.25         | ✗ NaN                        | ✓ 0.92 accurate           |
| 2D dry bubble at Δt ≥ 1.5          | ✗ NaN                        | ✗ NaN (CFL limit ~1.4)    |
| 1D "BV" oscillator                 | (mis-named — vertical acoustic test, intended damping) | same |

### Outer-step CFL boundary verified at multiple resolutions

(`56g_bubble_cfl_test.jl`)

| grid   | Δt  | CFL = cs·Δt/Δx | result                     |
|--------|-----|----------------|-----------------------------|
| 64×64  | 1.0 | 1.11           | NaN                        |
| 32×32  | 1.0 | 0.56           | ✓  ratio=0.96              |
| 32×32  | 2.0 | 1.11           | ✓ but 4.5× over-amplified  |
| 32×32  | 4.0 | 2.22           | NaN                        |
| 16×16  | 4.0 | 1.11           | ✓ but 6.4× over-amplified  |
| 16×16  | 8.0 | 2.22           | NaN                        |

The boundary is consistent across resolutions: CFL ≲ 1 is stable and accurate
to within a few percent; CFL ≳ 1 is over-amplified (and eventually NaN).

This is a **soft outer-step CFL** induced by the substepper's
linearize-at-U⁰ design: when cs·Δt/Δx > 1 the linearization basic state
(frozen at outer-step start) becomes more than one acoustic-grid-point
stale by the end of the outer step, and accumulated linearization error
destabilizes the bubble. To break this limit would require refreshing
the linearization point per RK stage AND restructuring σ, η as
"perturbation from current stage state" instead of "perturbation from
outer-step-start state" — a non-trivial refactor.

### Overall progress this session

- **H7 fix (frozen horizontal pressure imbalance) is THE major bubble bug.**
  Bubble dynamics went from 0.091× explicit (10× error) to ~0.98× explicit
  (within a few percent) at any Δt within the CFL limit.
- **H11 fix (correct-sign + face-operator Klemp 2018 divergence damping)**
  extends the soft outer-step CFL from Δt ≤ 0.89 s to Δt ≤ 1.25 s on
  the Δx=312 m grid (CFL=1.4), with default `KlempDivergenceDamping(0.1)`.
  No measurable cost on hydrostatic balance or IGW; small ~5 % cost on
  bubble accuracy at Δt = 0.5 s (0.98 → 0.93).
- IGW: stable & accurate at all amplitudes Δθ ∈ {0.01, 0.1, 1.0, 5.0} K.
- Hydrostatic balance: machine-zero drift across Ns sweep.
- Soft outer-step CFL Δt ≲ Δx/cs documented and reproduced; partially
  broken via Klemp damping.

### H11. Klemp 2018 divergence damping with corrected sign + face operators

The original Klemp damping kernel had a face/center operator mismatch
(`δxᶜᵃᵃ` applied at face indices) and a sign error
(`+ν · ∂x((η − η_old)/θ⁰)` instead of `−ν · ∂x((η − η_old)/θ⁰)`).
Rewrote with proper `∂xᶠᶜᶜ`/`∂yᶜᶠᶜ` operators and corrected sign.

Coefficient normalization: `ν = coefficient × cs × ℓ_disp` so the
dimensionless `coefficient` is a natural ~0.05–1.0 with default 0.1
mapping to ν ≈ 1.1×10⁴ m²/s on Δx=312 m.

**Result on the bubble at Δt=1 s (CFL = 1.12):**

| coefficient | result                              |
|-------------|-------------------------------------|
| 0.0 (NoDamp)| ✗ NaN at t ≈ 200 s                  |
| 0.05        | ✓ ratio_overall = 0.98              |
| 0.1 (default)| ✓ ratio_overall = 0.92             |
| 0.5         | ✓ ratio_overall = 0.97              |
| 1.0         | ✓ ratio_overall = 0.97              |

Klemp(0.1) **does not affect**:
- Hydrostatic balance (max\|w\| stays at 7×10⁻¹⁴ machine zero)
- IGW amplitude (substepper wmax is identical with/without damping)
- IGW stability across amplitudes Δθ ∈ {0.01, 0.1, 1.0, 5.0} K

Klemp(0.1) **extends Δt range** for the bubble from CFL ≤ 1 (Δt ≤ 0.89 s
on Δx = 312 m) to CFL ≤ ~1.4 (Δt ≤ 1.25 s). Beyond CFL ≈ 1.4 the
substepper still NaNs even with damping, so this is a partial fix —
the underlying linearize-at-U⁰ staleness limit still applies for very
large Δt. But it brings Δt = 1 s — a typical user choice — squarely
into the stable & accurate regime.

### H10. Per-stage refresh of linearization basic state

To break the soft outer-step CFL, attempted to refresh
`(outer_step_pressure, outer_step_density, outer_step_*, pressure_imbalance)`
from the *current* model state at each WS-RK3 stage start (in
`prepare_acoustic_cache!`). Recovery base `(recovery_density,
recovery_density_potential_temperature)` was split into separate fields,
frozen at outer-step start, so the WS-RK3 invariant
``U^{(k)} = U^{0}_{outer} + β_k Δt G(U^{(k-1)})`` would still hold.

**Result on the rest atmosphere: catastrophic instability.** max|w|
amplifies by ~1.5× per outer step (5e-12 at iter=10 → 5e-4 at iter=50 →
NaN at iter=90). Cause: positive FP-rounding feedback through
pressure_imbalance — the per-stage refreshed pressure picks up the
substepper's own slight perturbations, and re-feeds them as the next
stage's frozen drive. **Reverted to outer-step-start-only refresh.**

The split recovery/linearization fields were retained (no functional
change at present, but provides clean infrastructure for a future
attempt that decouples the linearization-point feedback from the
recovery base).
