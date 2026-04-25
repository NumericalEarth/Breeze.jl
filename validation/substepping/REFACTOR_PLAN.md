# Substepper refactor plan

## Diagnosis

Current scheme is **inconsistent**: more substeps `Ns` produce more spurious noise rather than converging to the analytic operator. Baldauf 2010 (MWR, "Linear Stability Analysis of Runge-Kutta-Based Partial Time-Splitting Schemes for the Euler Equations") proves the RK3WS scheme converges as `n_s → ∞` at second order in `ΔT` (eq 14), so any divergence with `Ns` is an implementation bug, not a property of the scheme.

The architectural problem: Breeze splits PGF + buoyancy across slow and fast operators, with a **stage-1 frozen** linearization point and a **perturbation-form** substepper. That isn't what Baldauf's analysis (or ERF) assumes:

- **Baldauf eq (20)**: `∂q/∂t = P_A q + P_S q + P_B q + P_D q` where `P_A` = advection (slow), `P_S` = sound, `P_B` = buoyancy, `P_D` = damping (all fast).
- **Slow operator carries advection only.** Sound, buoyancy, and damping all live in the fast operator.
- **The fast operator is linearized around a time-independent hydrostatic base state** `(ρᵣ(z), pᵣ(z), Πᵣ(z))`, not around the current prognostic state or a stage-frozen state.

Mismatch between current Breeze and the Baldauf framework:

| Term | Baldauf | Current Breeze |
|---|---|---|
| Advection of ρ, ρu, ρv, ρw, ρθ | slow operator P_A | slow `Gⁿ.*` (per-stage) ✓ |
| Coriolis, closure, BCs | slow operator P_A | slow `Gⁿ.*` (per-stage) ✓ |
| Horizontal PGF | fast operator P_S | snapshotted at stage 1, frozen across stages ✗ |
| Vertical PGF | fast operator P_S | `tend_w_euler` from frozen U⁰, frozen across stages ✗ |
| Buoyancy | fast operator P_B | `tend_w_euler`'s g·(ρ−ρᵣ) from frozen U⁰ ✗ |
| Acoustic damping | fast operator P_D | applied per substep, OK ✓ |
| Linearization of EOS | hydrostatic reference | frozen-at-stage-1 prognostic state ✗ |

The user's framing: "frozen always means PER STAGE (over substeps), not across stages." Baldauf's framework is even stronger: the *fast operator's linearization point* should be the **time-independent reference state**, so the only thing that changes between stages is the slow tendency (which is just advection).

## Target architecture

**Slow operator**: advection + Coriolis + closure + boundary terms only. Recomputed every RK stage from the current stage state. No PGF, no buoyancy, no snapshots, no restores.

**Fast operator**: full PGF + full buoyancy + acoustic damping. Linearized around the time-independent hydrostatic reference state `(ρᵣ(z), pᵣ(z))`. Updates the prognostic ρ, ρu, ρv, ρw, ρθ directly (no perturbation form). Slow tendency × Δτ added at every substep.

**Freeze policy**: nothing frozen across stages. Within a stage's substep loop, the slow tendency is "frozen" in the sense it isn't recomputed during the substeps, but that's just the standard split-explicit construction (see Baldauf eq 9).

## Concrete refactor steps

### Phase 1 — slow tendency cleanup (preparatory)

1. **Drop horizontal PGF freeze.** Remove `snapshot_horizontal_pgf!`, `add_horizontal_pgf!`, the `slow_tendency_snapshot` field on the timestepper, and the `:snapshot` / `:restore` machinery. The slow tendency for `Gⁿ.ρu`, `Gⁿ.ρv` will keep using `SlowTendencyMode` (zeros PGF) so the slow tendency contains only advection + Coriolis + closure.

2. **Remove `convert_slow_tendencies!` entirely.** PGF and buoyancy will no longer be computed as a slow forcing fed into the substepper. Drop `Gˢρw_total` field on the substepper.

3. **Switch all stages to `:none` freeze policy** (or remove the `freeze` keyword altogether — it's unused after the snapshot/restore machinery is gone).

### Phase 2 — fast operator carries PGF and buoyancy

4. **Inside the substep loop, compute the full horizontal PGF every substep** at the *current* state (which evolves substep-by-substep within the substep loop). Replace the existing `_mpas_horizontal_forward!` kernel that reads `ρθ_for_pgf` (perturbation) with one that reads the current `ρθ` and `ρ` directly.

5. **Inside the substep loop, compute the full vertical PGF + buoyancy every substep** at the current state. The Schur tridiagonal still implicitly couples ρw and ρθ vertically, but with the fast operator now being "the full thing" — not a perturbation around a frozen state.

6. **Linearize the tridiagonal coefficients around the hydrostatic reference state** `(ρᵣ(z), pᵣ(z), Πᵣ(z))` — these are time-independent. Drop `frozen_pressure`. The substepper's `acoustic_pgf_coefficient` and `buoyancy_linearization_coefficient` will use the reference state instead of `frozen_pressure`.

### Phase 3 — drop perturbation form

7. **Substepper updates full prognostic ρ, ρu, ρv, ρw, ρθ directly.** Remove the perturbation fields `ρ″, ρu″, ρv″, ρw″, ρθ″` and the predictor fields `ρ″_predictor, ρθ″_predictor`. The substepper's BatchedTridiagonalSolver works on `ρw` (or its tendency) directly.

8. **Remove the per-stage perturbation reset** (`fill!(parent(ρu″), 0)` etc.) and the velocity reset to U⁰ at stage start (`_reset_velocities_to_U0!`). The substepper integrates the full state; WS-RK3 stage update is `q^{n,m+1} = q^n + γ_m R(q^{n,m})` where `R` is the full RHS.

9. **Reformulate `_explicit_ρw″_face_update`** as an update to ρw directly, including the full PGF and buoyancy at the current substep state.

10. **Reformulate `ρθ″_predictor` and `ρ″_predictor`** as direct ρθ, ρ tendencies (full advective transport including the substep's mass flux contribution).

### Phase 4 — damping strategy adjustments

11. **Damping operates on the prognostic ρθ, ρ, ρu, ρv, ρw** (not on perturbations). The PressureProjectionDamping forward-projects `(ρθ − ρθ_old_substep)` rather than `(ρθ″ − ρθ″_old)`. The Klemp-Skamarock-Ha 3D divergence damping stays as is — it already operates on the prognostic ρu, ρv with the `δθᵥ_*` proxy.

### Phase 5 — notation cleanup

12. **Replace MPAS-style cryptic names** with descriptive ones:
    - `cofrz` → `inv_Δzᶜᶜᶜ_Δτᵋ` (or inline as `Δτᵋ * rdz`)
    - `rdzw_above`, `rdzw_below` → `inv_Δzᶜ_above`, `inv_Δzᶜ_below`
    - `pgf_coeff` → `Π_face_γRᵈ_Δτᵋ_per_Δzᶠ` (or compute inline; the operator is `γRᵈ Πᶠ ∂z`)
    - Drop `Jθ` shorthand in favor of explicit `θᵥ_face × Δτᵋ` where it appears.
13. **Use Oceananigans operator names**: `ℑzᵃᵃᶠ` for arithmetic-mean center-to-face, `δzᵃᵃᶠ` for vertical centered difference, etc. Already partially done after the `acoustic_pgf_coefficient` refactor.
14. **Remove `_safe` defensive guards** from interior fields that are never zero in practice — keep them only on quantities that actually need them at boundaries (e.g. `θᵥ_at_face` already has the correct boundary behavior).
15. **Group long argument lists** into named tuples or use `@unpack` patterns — current `_build_acoustic_rhs!` takes ~20 arguments, mostly cryptically named.

## Risk and validation

- **Tests will break** during this refactor — the perturbation-form prognostics (ρ″ etc.) are part of the substepper struct that some tests check. We will need to update tests as we go.
- **Validation suite**:
  - `22_acoustic_pulse.jl` (g=0, small pulse): should remain stable at all Ns, wmax should match across Ns to ~5%
  - `25_centered_ns_sweep.jl` (bubble, Centered(2), Ns sweep): wmax should converge as Ns increases (currently diverges)
  - `13_bubble_substeps_sweep.jl` (bubble, WENO-9, fixed Δt=1, Ns sweep): currently crashes at Ns≥24 — should not crash if scheme is consistent
  - `06_neutral_abl.jl` and other moist test cases: should still run

## Order of operations

1. Phase 1 first (drop snapshot/restore, drop `convert_slow_tendencies!`). This will break the substepper temporarily because its `Gˢρw_total` will be empty. Acceptable because phase 2 fixes it.
2. Phase 2 (fast operator carries PGF/buoyancy at current state). This makes the substepper functional again, even though it's still using perturbation form.
3. Phase 3 (drop perturbation form). Biggest change. After this, run the validation suite.
4. Phase 4 (damping adjustments) — hopefully minimal because the existing damping kernels are already structured around Δτᵋ-time differences.
5. Phase 5 (notation cleanup) — done in passing as we touch each file, plus a final sweep.

## Notes / open questions

- The Schur tridiagonal in MPAS/ERF style implicitly couples ρw and ρθ for stability. We retain that — the implicit solve is the "fast vertical acoustic" treatment that lets `Δτ` exceed the vertical acoustic CFL. The change is what state the linearization uses (reference, not frozen prognostic).
- Conservative-perturbation mass flux: MPAS's design uses `θᵥ × ρ_pp` to keep the perturbation form mass-conservative. Without perturbation form we go back to standard `θᵥ × ρu` directly — should still be conservative.
- The forward-weight off-centering `ω = 2β + 1` is independent of all this and stays.
