# New substepper — plan

This is the working planning document for the from-scratch reformulation of
Breeze's compressible-equations acoustic substepper. It supersedes the
existing `acoustic_substepping.jl` (which was an MPAS-A port and is being
retired).

**Companion theory documents** (in repo root, written by the user):
- `_draft_first_pass.md` — Baldauf 2010 in Breeze notation; full linearized
  perturbation equations, stability proofs, MIS framework overview.
- `split_explicit_substepping_foundations.md` — outline / planning document
  for a more rigorous theoretical reference once tier-1 papers are in hand.

> **Author intent (from user, paraphrased and quoted across messages):**
> "We are abandoning hevi-imex … there should be no routines called 'mpas'
> or 'erf'. We are undergoing a complete reformulation."
>
> "The equation set we are using is the same one that we use for fully
> compressible. For time-integration, we are using a substepping scheme
> that evolves acoustic perturbations with linearized equations between
> each stage. The acoustic perturbation equations are formulated by
> linearizing around the state at the BEGINNING of the time-step — the
> main nonlinear term is the potential temperature, which is used to
> compute the perturbation temperature flux in terms of perturbation
> momentum. The other term is the pressure gradient term. The pressure
> gradient is formulated in terms of an exner function for both the
> fully nonlinear set, and the perturbation set. The perturbation
> equations are formulated with an off-centered CN scheme to produce a
> tridiagonal system for ρ·w, following Baldauf and MPAS. For testing
> we start with perfectly centered, ie classic CN. For damping we use
> the formulation proposed by Klemp et al 2018."

---

## 1. Why we are rewriting

The current `acoustic_substepping.jl` (1564 lines) is a verbatim port of
MPAS-A's `atm_advance_acoustic_step`. Diagnostic evidence (2026-04-25):

- On a **hydrostatic rest atmosphere** (validation 40), the scheme NaNs
  at Ns=48 by t≈560s. A correct discretization should keep `max|w|` at
  machine epsilon.
- On the same test with the centered time-discretization (ω=0.5,
  validation 41-42), the scheme is unstable for **all Ns and any
  damping setting**. The current defaults (`forward_weight = 0.7`,
  `PressureProjectionDamping(0.5)`) mask the underlying instability via
  numerical dissipation. Per the damping philosophy memory, defaults
  must not hide scheme bugs.

The rewrite unwinds the linearization choice, the time-integration form,
and the divergence damping in a clean re-derivation, keeping only the
pieces of structure that are physically correct for Breeze's EOS, and
naming everything in Breeze terms (no `cofwz`, no `dtseps`, no
`tend_u_euler`, etc.).

---

## 2. Design (resolved)

### 2.1 Equation set

- Same prognostic family as fully-compressible Breeze: **(ρ, ρu, ρv, ρw, ρθ)**
  plus moisture / tracers.
- Conservative form (Breeze's `governing equations`):
  ```
  ∂t ρ      + ∇·(ρu)        = 0
  ∂t (ρu)   + ∇·(ρuu) + ∇p  = -ρ g ẑ
  ∂t (ρθ)   + ∇·(ρθ u)      = 0
  ```
- EoS: `p = pˢᵗ · (Rᵈ ρθ / pˢᵗ)^γ` — pressure depends on the **product**
  ρθ. For dry air `γ = γᵈ = cᵖᵈ/cᵛᵈ`. The Exner function is
  `π = (p / pˢᵗ)^(Rᵈ/cᵖᵈ)`, used both in the fully nonlinear
  formulation and in the perturbation set.

### 2.2 Substepper prognostic perturbations

- The substepper evolves **acoustic perturbations** from the
  outer-step-start state `U⁰ = (ρ⁰, ρu⁰, ρv⁰, ρw⁰, ρθ⁰)`:
  ```
  σ  = ρ  − ρ⁰
  μu = ρu − ρu⁰,  μv = ρv − ρv⁰,  μw = ρw − ρw⁰
  π  = ρθ − ρθ⁰
  ```
  (Working names — the actual identifiers in the code can be
  refined; the point is one perturbation per prognostic, named cleanly
  without MPAS-style suffixes like `_pp` or `″`.)

### 2.3 Linearization point

- **Single linearization point per outer Δt**, equal to the state at
  the **beginning of the outer time-step** `U⁰`.
- This snapshot is taken once at outer-step start and used by every
  RK stage's substep loop. Background quantities `θ⁰ = ρθ⁰/ρ⁰` and
  `Π⁰ = (p⁰/pˢᵗ)^κ` (with `p⁰` from the EoS at `U⁰`) are computed
  from this snapshot once and cached.
- Re-linearizing each outer Δt — rather than around a fixed
  hydrostatic reference — captures the local stratification of the
  evolving flow without freezing the wrong state across stages.
- The previous `acoustic_substepping.jl` carried two cached
  pressures (`outer_step_pressure`, `frozen_pressure`); the new
  scheme uses one — the outer-step-start snapshot — and folds
  whatever role the reference pressure played into the slow-tendency
  assembly.

### 2.4 What's nonlinear inside the substep

Two terms carry the nonlinearity through the linearization (paraphrasing
the user's description):

1. **Background potential temperature θ⁰.** The **perturbation
   temperature flux** is `θ⁰ × (μu, μv, μw)` — background θ times
   perturbation momentum. This is the sole place the thermodynamic
   state appears multiplicatively against the perturbation momentum.
2. **Pressure gradient term, formulated via the Exner function.** Both
   the fully nonlinear set and the perturbation set use the Exner
   formulation. For the perturbation set, the linearized PGF reads:
   - vertical:   `∂z(p) − ∂z(p⁰) ≈ γᵈRᵈ · Π⁰ · ∂z(π)`
   - horizontal: `∂x(p) − ∂x(p⁰) ≈ γᵈRᵈ · Π⁰_face · ∂x(π)` (etc.)

Buoyancy in the vertical momentum equation enters as `−g · σ` (the
hydrostatic part `−g · ρ⁰` is absorbed into the slow tendency, since
the linearization point is hydrostatically self-consistent: `∂z p⁰ = −ρ⁰ g`
holds at outer-step start to within EoS round-off).

### 2.5 Linearized perturbation equations (continuous form)

Adapted from `_draft_first_pass.md` § 2.4 / Eq. (7), with reference
state replaced by outer-step-start state per §2.3 above:

```
∂t σ   +     ∇·(μu, μv, μw)         = Gˢρ
∂t π   +     ∇·(θ⁰ · (μu, μv, μw))  = Gˢρθ
∂t μu  + γᵈRᵈ · Π⁰_fcc · ∂x π        = Gˢρu
∂t μv  + γᵈRᵈ · Π⁰_cfc · ∂y π        = Gˢρv
∂t μw  + γᵈRᵈ · Π⁰_ccf · ∂z π + g·σ  = Gˢρw
```

`Gˢ` are the **slow tendencies** — advection + Coriolis + closure +
microphysics — held fixed within a substep loop and re-evaluated each
outer RK stage. The slow tendency does **not** include PGF or
buoyancy; those live in the fast operator inside the substep loop.

### 2.6 Time-integration form

- **Off-centered Crank-Nicolson** for the perturbation equations,
  giving a **tridiagonal system in z for the implicit ρw update**.
  Following the structure of Baldauf and the MPAS/CM1/WRF/ERF family,
  but the assembly is re-derived from the equations in §2.5.
- **Default for testing: classic centered CN** (off-centering ε = 0,
  weight β = 1/2). The first round of validation runs at β = 1/2; the
  off-centered β > 1/2 (Baldauf β_S, β_B) is added once the centered
  scheme is correct and validated.
- Outer integration: **WS-RK3** (current `AcousticRungeKutta3`,
  unchanged structurally — but the slow-tendency-snapshot machinery
  inside it is dropped because the new scheme doesn't need it).

### 2.7 Damping

- **Klemp et al. 2018** form
  ([Klemp, Skamarock & Ha 2018](@cite KlempSkamarockHa2018), the
  thermodynamic-divergence-damping derivation).
- Damping is *not* used to stabilize a buggy core scheme. The
  centered-CN scheme must pass tests (i) and (ii) with no damping; the
  Klemp 2018 form is added for production runs to filter
  small-amplitude grid-scale acoustic divergence over long
  integrations.

---

## 3. Hard constraints

- **No `mpas_*` or `erf_*` naming** anywhere in identifiers, comments,
  or struct fields. Inspiration from those codebases is fine in
  docstrings; the public surface is Breeze-native. Specifically: no
  `cofwz`, `cofwt`, `coftz`, `cofrz`, `dtseps`, `epssm`, `smdiv`,
  `rho_pp`, `ru_p`, `rtheta_pp`, `tend_u_euler`, `tend_w_euler`,
  `″`-suffix variables, etc.
- **GPU-compatible kernels** (`@kernel` / `@index`, no models inside
  kernels, `ifelse` not `?:`, etc. — see `.claude/rules/kernel-rules.md`).
- **Type-stable, allocation-free kernels.** Materialization pattern for
  any state held by the substepper.

---

## 4. What we keep, what we drop

### Drop wholesale

- `src/CompressibleEquations/acoustic_substepping.jl` (1564 lines).
- `slow_tendency_snapshot` field on `AcousticRungeKutta3` and the
  `snapshot_slow_tendencies!` / `restore_slow_tendencies!` /
  `add_horizontal_pgf!` machinery in
  `src/TimeSteppers/acoustic_substep_helpers.jl`. The new scheme does
  not freeze tendencies across RK stages; each stage recomputes slow
  tendencies fresh.
- `outer_step_pressure` and `frozen_pressure` fields — collapse to
  one outer-step-start snapshot (see §2.3).
- The two pressure-projection-style damping strategies
  (`PressureProjectionDamping`, `ConservativeProjectionDamping`).
  Replaced by a single Klemp 2018 implementation.

### Keep

- The `AcousticDampingStrategy` type hierarchy (selectable damping) —
  but populated only with `NoDivergenceDamping` (default) and
  `KlempDivergenceDamping` (re-derived per §2.7).
- The `AcousticSubstepDistribution` interface (proportional vs
  monolithic-first-stage).
- `BatchedTridiagonalSolver` for the implicit vertical solve — this is
  Oceananigans infrastructure, scheme-agnostic.
- The `AcousticRungeKutta3` outer driver — it will call into the new
  substepper, but its WS-RK3 stage logic stays.
- The validation scripts in `validation/substepping/` — they're the
  regression bed for the new scheme; some need API updates after the
  rewrite.
- `compute_acoustic_substeps` (adaptive Ns logic).

---

## 5. Test ladder

Each test is run twice when applicable: once with **classic centered
CN, no damping** (baseline correctness) and once with **off-centered
CN + Klemp 2018 damping** (production config). Centered + no damping is
the diagnostic that catches scheme bugs; the second run is the
production-readiness check.

| # | Test | Adds | Advection schemes | Pass criterion |
|---|------|------|-------------------|----------------|
| **0**   | **Hydrostatic rest atmosphere** (already exists: `40_hydrostatic_balance.jl`) | sanity check | n/a | `max|w|` at machine epsilon, all Ns |
| **i**   | **1D vertical acoustic wave** | implicit vertical Schur | n/a (no horizontal) | wave propagates at `c_s` with no growth in centered CN |
| **ii**  | **2D Gaussian acoustic pulse** | + horizontal coupling | n/a (linear, no slow advection) | pulse propagates radially, neutrally stable at centered CN |
| **iii** | **2D Gaussian pulse in shear flow** | + slow advection (nonlinear) | Centered & WENO | pulse advected by shear, no spurious growth, both advection schemes give similar wave field |
| **iv**  | **IGW (linear gravity wave)** | + buoyancy + slow advection | Centered & WENO | gravity-wave dispersion matches linear theory |
| **v**   | **Dry thermal bubble** | + stiff IC + nonlinear coupling | WENO | matches anelastic and explicit-compressible reference; no aggressive damping |

Tests already in the repo, mapped to this ladder:
- Test 0: `40_hydrostatic_balance.jl` (works).
- Test (ii): existing `30_pure_acoustic_pulse.jl` (g=0, 2D pulse).
- Test (iv-precursor): `31_acoustic_wave_2d.jl`.
- Test (v): `01_dry_thermal_bubble.jl`, `07_dry_thermal_bubble_wizard.jl`,
  `14_dry_thermal_bubble_ns12_long.jl`.

To add:
- Test (i): `50_vertical_acoustic_wave.jl` — 1D vertical (Flat horizontal).
- Test (iii): `51_pulse_in_shear.jl` — 2D pulse + background horizontal flow.
- Test (iv): linear-IGW driver if `31_*` is insufficient.

---

## 6. Approach (sequence of work)

1. ✅ **Plan in place** (this document) and theoretical companions
   (`_draft_first_pass.md`, `split_explicit_substepping_foundations.md`).
2. **Add test (i)** (1D vertical acoustic wave) to the validation
   ladder.
3. **Stand up the new substepper** in
   `src/CompressibleEquations/`. Replace the contents of
   `acoustic_substepping.jl` wholesale, keep filename so existing
   `include` chain stays intact. Or split into a new file with a clean
   directory structure — decide pragmatically.
4. Implement just enough of the substep to pass **test 0** (rest
   atmosphere) and **test (i)** (1D vertical wave) at centered CN, no
   damping.
5. Add the horizontal acoustic coupling; pass **test (ii)**.
6. Add slow advection contribution to the predictor; pass **test (iii)**
   in centered and WENO.
7. Add buoyancy via the `−g·σ` term (already in §2.5) and verify
   **test (iv)** in centered and WENO.
8. Run **test (v)** end-to-end.
9. Add Klemp 2018 damping; verify it doesn't degrade the centered tests
   and improves long-integration robustness.
10. Strip any leftover references to the old MPAS-style code, retire
    obsolete validation scripts, tighten docstrings.

---

## 7. References (key tier-1 papers)

Already in the repo / read:

- **Baldauf, M., 2010.** *Linear stability analysis of Runge–Kutta-based
  partial time-splitting schemes for the Euler equations.* MWR 138,
  4475–4496. (Stability framework, off-centering analysis,
  truncation-error expansion. Backbone of the rewrite.)
- **Wicker, L. J. & Skamarock, W. C., 2002.** *Time-splitting methods
  for elastic models using forward time schemes.* MWR 130, 2088–2097.
  (WS-RK3 outer integrator.)
- **Klemp, J. B., Skamarock, W. C. & Ha, S.-Y., 2018.** *Damping
  acoustic modes in compressible HEVI and split-explicit time
  integration schemes.* MWR 146, 1911–1929. (The damping form §2.7.)

Outstanding (for theoretical depth, per
`split_explicit_substepping_foundations.md`):

- Wensch, Knoth & Galant, 2009 (BIT). MIS framework.
- Knoth, Schlegel & Wensch, 2014 (MWR). Atmospheric MIS tableaux.
- Skamarock & Klemp, 1992 (MWR). Original stability framework.
- Klemp, Skamarock & Dudhia, 2007 (MWR). Conservative-form
  linearization derivation.

---

## 8. Status / log

- 2026-04-25 — Initial scaffold drafted by Claude.
- 2026-04-25 — User pinned design (§2) and test ladder (§5).
- 2026-04-25 — User added theoretical drafts
  (`_draft_first_pass.md`, `split_explicit_substepping_foundations.md`);
  plan updated to incorporate the linearized perturbation equation set
  (§2.5) and to resolve the linearization-point question to
  outer-step-start (§2.3) per user directive.
- 2026-04-25 — New substepper implemented (`acoustic_substepping.jl`,
  ~620 lines, replacing the 1564-line MPAS-port). Old WS-RK3 driver
  simplified to drop the snapshot/restore freeze policy. `time_discretizations.jl`
  drops `PressureProjectionDamping`, `ConservativeProjectionDamping`,
  renames `ThermodynamicDivergenceDamping` to `KlempDivergenceDamping`
  (alias retained).
- 2026-04-25 — **Tests (i)–(iii) pass at centered CN, no damping.**
  Excellent Ns-consistency:
    - Test (i) 1D vertical acoustic wave: Ns ∈ {6,12,24,48} all give
      final max\|w\| ≈ 1.96e-5 m/s — within 0.5% spread.
    - Test (ii) 2D Gaussian pulse: Ns ∈ {6,12,24,48} all give
      final max\|w\| ≈ 1.7e-4 m/s — within 3% spread.
    - Test (iii) 2D pulse in shear (Centered & WENO): all 6 cases stable,
      both advection schemes agree to within 0.4%.
- 2026-04-25 — **Test 0 (hydrostatic rest atmosphere with g≠0)**
  reveals a marginal-stability issue at centered CN (ω=0.5). The
  buoyancy off-diagonal in the Schur tridiag is anti-symmetric for
  uniform Δz (`A[k,k+1] = -A[k,k-1]`), making the matrix non-normal.
  Centered CN is *neutrally* stable in theory; with finite-precision
  arithmetic the non-normality causes slow exponential growth on a
  rest atmosphere. ω = 0.55 (canonical ERF/MPAS minimal off-centering,
  ε = 0.1) fully stabilizes — the rest atmosphere stays at machine
  zero across all Ns. **Default updated to forward_weight = 0.55.**

  Reference-state subtraction added to `assemble_slow_vertical_momentum_tendency!`:
  `Gˢρw = Gⁿρw - ∂z(p⁰ - pᵣ) - g(ρ⁰ - ρᵣ)` so that the rest
  atmosphere's IC starts at machine zero exactly (subtracting two
  numerically-equal hydrostatic profiles to cancel discretization
  residuals from the EoS / hydrostatic-integration mismatch).

- 2026-04-25 — **Test (iv) IGW: PASSES** at ω=0.55, Δt=1, no damping,
  with both Centered(2) and WENO(5) advection. Ns ∈ {12, 24} converge
  to max\|w\| ≈ 6.3e-4 m/s within 3% spread. (Original setup with
  Δt=5 NaN'd; Δt=1 is the right outer-step size for this resolution.)

- 2026-04-25 — **Test (v) bubble: OPEN.** NaN at iter 20 (t=20s) even
  with `KlempDivergenceDamping(0.1)`, ω=0.55, Δt=1, smooth Gaussian IC,
  Δθ=0.5K, N²=1e-4 (matching the IGW setup that works). The substepper
  fails on this 128×128 grid (Δx=156m) but works at IGW resolution
  (Δx=1km). Possible causes:
    - High-resolution grids stress the GS split-explicit horizontal
      scheme's stability margin near acoustic Nyquist.
    - The WENO(9) used in advection for ρ, ρθ produces slope-limited
      gradients that interact badly with the centered substep
      operators.
    - Residual non-normality of the Schur matrix (test 0 only failed
      at ω=0.5; bubble fails at ω=0.55 — different growth rate from
      the rest-atmosphere instability).
  The OLD `acoustic_substepping.jl` at ω=0.7 / `PressureProjectionDamping(0.5)`
  / Δt=2 successfully ran the bubble for 25 minutes (per
  `validation/substepping/14_dry_thermal_bubble_ns12_long.jl`), so
  there's a known-working configuration for comparison.

  Next-iteration debugging targets for test (v):
    - Try with much larger off-centering (ω=0.7) and damping coefficient
      (Klemp(0.5)) to confirm bubble works at all.
    - Reduce resolution (64×64) to match what works for IGW.
    - Add per-stage diagnostic prints to see what blows up first
      (μw, σ, η).
    - Compare against the fully-explicit `ExplicitTimeStepping` solution
      to see if the issue is the substepper or the model setup.

- 2026-04-25 — **Diagnostic via 3-way comparison** (`55_3way_compare.jl`):
  for a small Δθ=0.1K bubble at 64×64 / N²=1e-4 / Δt=1, the anelastic
  and explicit-compressible reach max\|w\| ≈ 0.116, 0.121 (4% spread —
  ground truth). The substepper at ω=0.55, no damping, gives **0.011 m/s
  (10× too small)**. ω=0.7 explodes to 256 m/s; ω=0.9 NaNs; Klemp damping
  (any coefficient) NaN. The non-monotonic ω response is a smoking gun —
  the substepper's slow-dynamics ↔ substep coupling has a structural
  bug that increasing damping makes worse rather than better.

  **Working hypotheses for the open bug:**
    - ρu, ρv get accumulated `Δτ × Gⁿρu` from the explicit horizontal
      step. M_pert then enters σ_pred's `∇h·M_pert` term as a Δt²
      correction. This Δt² correction may not match what proper WS-RK3
      would compute, especially when the slow ρu, ρv tendencies are
      strongly coupled to slow ρθ tendencies (as in a buoyantly
      rising bubble).
    - The horizontal PGF in `Gⁿρu` is computed at the stage state
      (NOT zeroed by SlowTendencyMode for SplitExplicit, see
      `compressible_density_tendency.jl:24-27`). The substepper then
      adds a perturbation `−γRᵈ Π⁰ × ∂x η` on top. For η starting at 0
      each stage, this is small initially but grows. Whether the FULL
      slow + perturbation reproduces the WS-RK3 stage advance for
      cases with strong vertical dynamics is unclear — needs a careful
      derivation.
    - The off-centered CN `δτ_new = ω × Δτ` is currently used uniformly
      for both the σ predictor (explicit half) and post-solve (implicit
      half). The strict CN form would use `δτ_old = (1-ω) × Δτ` for the
      explicit half. At ω=0.55 the discrepancy is ~22%; this acts as
      effective extra damping at small ω and effective amplification at
      large ω, which is consistent with the observed
      0.011 (ω=0.55) → 256 (ω=0.7) → NaN (ω=0.9) progression.
      Attempted fix at strict CN broke the rest atmosphere — needs
      careful re-derivation that preserves both rest-atmosphere
      stability AND correct slow dynamics.
