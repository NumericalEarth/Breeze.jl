# Substepping cleanup & bug hunt — running notes

Ongoing engineering notes while we clean up `acoustic_substepping.jl` and hunt a
stability bug visible in the dry thermal bubble (validation case 01 and similar).
See `REPORT.md` for the validation evidence.

## Cleanup done on this branch

- Renamed MPAS-idiom identifiers to Breeze math notation (`_pp → ″`, `_face →
  ᶠᶜᶜ/ᶜᶠᶜ/ᶜᶜᶠ`, `pgrad → ∂x_p″ / ∂y_p″ / ∂z_p′`, `rtheta_pp → ρθ″`, `ru_p →
  ρu″`, etc.). Also renamed the misleadingly-named `c²` (actually `γRᵈ`, not the
  acoustic-speed squared) and `cₚ` (actually the dry-air value `cᵖᵈ`).
- Distinguished the two perturbation kinds: `″` = acoustic deviation from
  outer-step-start state (`ρθ″, ρu″, ρw″, ρ″`), `′` = deviation from a
  hydrostatic reference state (`p′ = p_frozen − p_ref` in
  `_convert_slow_tendencies!`).
- Dropped the precomputed `moist_rθ_ratio` CenterField; now computed on the
  fly via `_moist_rθ_ratio` if we need it at all (see next entry).
- **Removed the Xiao et al. 2015 moist correction factor ζ = Rᵐ/Rᵈ entirely.**
  Rationale: for dry air ζ ≡ 1 so no change; for moist air the factor is an
  MPAS-convention correction assuming `θ` is the *dry* potential temperature,
  which does not match Breeze's `LiquidIcePotentialTemperatureFormulation` EOS
  (see "Open: moist acoustic PGF" below). Removing it leaves the code
  dry-correct and makes the remaining moist discrepancy an explicit open
  problem rather than a buried approximation.

## Open: moist acoustic PGF needs its own derivation

Breeze's `LiquidIcePotentialTemperatureFormulation` is *not* MPAS's dry `θ`
(nor MPAS's moist `θ_m = θ_d · (1 + 0.608 qᵛ)`). It's the liquid-ice potential
temperature `θ_li`, and the EOS (`compressible_time_stepping.jl:128-152`) is

```
T = θ_li^γₘ · (ρ Rᵐ / pˢᵗ)^(γₘ − 1) + (ℒˡ qˡ + ℒⁱ qⁱ)/cᵖᵐ       γₘ = cᵖᵐ/cᵛᵐ
p = ρ Rᵐ T
```

Consequences for the acoustic PGF:

1. **Unsaturated** (qˡ=qⁱ=0): `p/pˢᵗ = (ρ Rᵐ θ_li / pˢᵗ)^γₘ` — uses **moist**
   γₘ. MPAS's substepper assumes dry γ_d. The Xiao factor fixes Rᵈ ↦ Rᵐ
   inside the base but does **not** fix the exponent. The linearized PGF
   `γRᵈ · Π · ∂x(ρθ_m)` (with `Π = (Rᵈ ρθ_m/pˢᵗ)^(Rᵈ/cᵛᵈ)`) is wrong at
   O(qᵛ · (γₘ − γ_d)).
2. **Saturated** (qˡ>0 or qⁱ>0): the latent-heat offset
   `(ℒˡ qˡ + ℒⁱ qⁱ)/cᵖᵐ` in T means ∂x T has a contribution from ∂x(qˡ,qⁱ)
   that is absent from the current linearization. This is likely what the
   `46df021` moist-BW fix papered over in the *slow* vertical PGF by using
   `p′ = p_frozen − p_ref` directly; the *fast* horizontal PGF still has this
   hole.

**TODO**: Derive the acoustic horizontal PGF directly from Breeze's EOS rather
than borrowing MPAS's form. For unsaturated air the correct form is roughly
`γₘ Rᵐ · Πₘ · ∂x(ρ θ_li)` with `Πₘ = (p/pˢᵗ)^(Rᵐ/cᵖᵐ)`. For saturated air a
latent-heat term needs to be added or absorbed into a different thermodynamic
variable.

## Open: dry thermal bubble failure mode

The REPORT shows fixed-Δt=2s crashes at iter 6 with `ρ_min ≈ −0.06` and
`max|w| ≈ 725 m/s`. Δt=0.25s and adaptive-CFL runs complete. This is
consistent with a genuine split-explicit CFL limit (≥8× tighter than
anelastic), but we want to confirm it's not a code bug. Candidates we haven't
ruled out:

1. **Initial state imbalance.** The driver does `set!(model; θ=θᵢ, ρ=ref.density)`
   — ρ set from the hydrostatic reference built against background θᵇᵍ only, but
   θ has the 10K bubble on top. The initial EOS pressure is therefore out of
   hydrostatic balance at the bubble. This launches a real acoustic response
   at t=0. Not a bug per se, but its amplitude interacts with #2.
2. **Density recovery across RK stages.** `_mpas_recovery_wsrk3!` writes
   `ρ = ρ⁰ + ρ″`. The snapshot/restore of `Gⁿ.ρ` (freeze at stage 1, restore
   at 2/3) is subtle; a miswrite would let ρ drift freely inside the substep
   loop and go negative within a few outer steps.
3. **Bounded-grid horizontal PGF.** Case 02 (Bounded/Flat/Bounded) crashes
   worse than case 01 (Periodic/Flat/Bounded). That hints at a boundary-face
   handling issue in `_mpas_horizontal_forward!`, the `on_x_boundary` masks,
   or the halo fills on `ρu″`.
4. **Stage-2/3 snapshot interactions.** `convert_slow_tendencies!` reads
   `model.dynamics.pressure` live at each stage, while `substepper.frozen_pressure`
   is snapshotted once per outer step. The `p′ = p_frozen − pᵣ` term thus uses
   two different time levels. The code treats this as intentional (match MPAS),
   but worth re-verifying under the renamed notation.

### Post-cleanup smoke test (2026-04-24)

Dry thermal bubble, Δt = 2.0 s, `PressureProjectionDamping(0.1)`:

```
step 1: wmax = 2.71, ρmin(interior) = 0.4418
step 2: wmax = 2.56, ρmin           = 0.4418
step 3: wmax = 1.97, ρmin           = 0.4418
step 4: wmax = 1.61, ρmin           = 0.4418
step 5: wmax = 3.07, ρmin           = 0.4418
step 6: DomainError with -12247.07  ← same as REPORT.md
```

Interior `ρmin` is stable at ~0.44 (top-of-domain density), yet step 6 hits a
pow of a negative number. Since all healthy fields flow `ρ`, `p`, `θ` through
`_exner_from_p` or `acoustic_pgf_coefficient` (both evaluating `(p/pˢᵗ)^κ`),
the offender is almost certainly `substepper.frozen_pressure` or an
intermediate computation on a **halo** cell (interior was queried in the test,
but halos aren't). Next: dump the pressure field at step 5 and see which cell
goes negative at step 6.

Note: the cleanup (Xiao removal + u/v momentum-tendency refactor) preserves
the failure mode exactly — so the bug isn't a regression introduced by any of
these edits.

## Next: u/v velocity-tendency round-trip (comes before the ρw question)

The dynamics kernel produces momentum tendencies `Gⁿ.ρu`, `Gⁿ.ρv`. The
substepper's `_convert_slow_tendencies!` does a divide-by-ρ round-trip:
`Gˢu = Gⁿ.ρu / ρ_face`, stored in `slow_tendencies.velocity.u`, which the
horizontal forward kernel then multiplies back by `ρ_face` to update `ρu″`.
This is wasted work and an extra 3D field. Plan: drop
`slow_tendencies.velocity.u/v` from the struct, pass `Gⁿ.ρu/Gⁿ.ρv` directly
into `_mpas_horizontal_forward!`, divide by ρ_face inside the kernel once
for the u-update. Leave `Gˢw` alone for now — it assembles the linearized
PGF+buoyancy and is not a pure divide.

## ERF-style recovery refactor — done 2026-04-24

Switched the substep loop to ERF's momentum-only recovery
(https://erf-model.github.io/ERF/theory/NumericalBackground.html, "Acoustic
Sub-stepping" section in the bundled docs):

- `_mpas_horizontal_forward!` no longer touches `u`, `v` — only `ρu″`, `ρv″`.
- Added `_erf_recover_momentum_and_velocity!` that does
  `ρu_new = ρu⁰ + ρu″`, `u_new = ρu_new / ρ_new_face` in a single pass after
  the substep loop. Replaces the old pair `_convert_ρw″_to_w!` +
  `_recover_momentum!` which multiplied the substep-updated velocities back
  by the new density.

At the linearized level ERF ≡ WRF ≡ MPAS. The numerical difference between
the two recovery arithmetics is `O(ρ″/ρ)` ~ 1%. Empirically confirmed: both
schemes crash at step 6 in the dry thermal bubble at Δt=2s, with nearly
identical first-4-step wmax and different DomainError values at the blow-up
instant. The ERF-style recovery is cleaner code; not responsible for the
blow-up.

### Smoke-test comparison (2026-04-24, dry thermal bubble, Δt=2s)

| Step | MPAS wmax | ERF wmax | ERF umax |
|------|----------:|---------:|---------:|
| 1    |      2.71 |     2.71 |     2.78 |
| 2    |      2.56 |     2.56 |     2.70 |
| 3    |      1.97 |     1.97 |     2.49 |
| 4    |      1.61 |     1.61 |     4.27 |
| 5    |      3.07 |     2.96 |    15.29 |
| 6    |     crash |    crash |        — |

### Bug-hunt implication

The root cause is **upstream of the recovery**. Candidates to inspect next:

- Slow-tendency assembly (`_convert_slow_tendencies!`) — especially the
  `p′ = p_frozen − p_ref` vertical PGF at cells near the hot bubble.
- Column kernel's explicit ρw″ predictor (`_build_acoustic_rhs!`).
- Divergence damping (projection filter sign / coefficient under strong
  acoustic response).
- Initial-state imbalance at the bubble interface that the damping can't
  absorb within a Δt=2s step.

Alternative interpretation: this is a genuine CFL limit of the scheme. At
Δt=0.25s the case runs fine (per REPORT §2). The REPORT already concludes
the substepper needs a tighter outer CFL than anelastic.

### Pre-existing fix caught during refactor

`model.dynamics.reference_state.standard_pressure` at acoustic_substepping.jl:1399
was crashing for `reference_state === nothing`. Replaced with
`model.dynamics.standard_pressure`, which is always present. This was a
latent bug unrelated to the bubble failure.

## Instrumented step-6 crash (2026-04-24)

Full field bounds (interior + halos) across the first 6 outer steps of the
dry thermal bubble, ERF-style recovery, `PressureProjectionDamping(0.1)`,
Δt = 2 s:

```
step 5:  interior ρ∈[0.4418, 1.1754]  p∈[25889, 101424]  T∈[204, 320]  ρθ∈[132.7, 352.0]
         halos:   max|u|=15.29  max|w|=2.96
step 6:  CRASH — DomainError with -13756
         interior ρ∈[-0.0359, 1.6742]  T∈[-12841, 9102]
         halos:   max|u|=57.78   max|w|=13346
```

Conclusions:

1. **`ρθ` is healthy throughout** (`[132.7, 352.0]` even post-crash). The
   thermodynamic scalar is not what fails.
2. **`ρ` is what goes bad** — it becomes negative at one or more cells at
   step 6. Since the EOS computes `T_dry = θ^γ · (ρ Rᵐ/pˢᵗ)^(γ−1)` with
   `θ = ρθ/ρ`, a negative ρ produces a giant negative θ and `θ^γ` → DomainError.
3. **Advective CFL is not the cause.** At step 5, `max|u| ≈ 15.3`, `max|w| ≈ 3`
   give CFL_x ≈ 0.20, CFL_z ≈ 0.08 — well within the 0.7–1.0 envelope that
   Skamarock & Klemp (2008), Baldauf (2010), Klemp-Skamarock-Dudhia (2007)
   document for WS time-splitting.
4. **This is a real bug, not a scheme limitation.** The published stability
   analyses say this CFL should be safe.

### Root-cause hypotheses, ranked

1. **Horizontal projection filter may be weak / mis-signed.** Damping
   `coefficient = 0.1` is the WRF/ERF canonical value, but the dry thermal
   bubble's IC (warm, weakly stratified, sharp cone) produces a strong
   horizontal acoustic response that the filter isn't containing. Possible
   sign or conversion-factor error in `_pressure_projection_filter!`.
2. **Horizontal PGF coefficient off.** `γRᵈ · Π · ∂ₓρθ″` — the dry-γ form is
   correct in principle; could there be a missing moisture-free `(Rᵐ/Rᵈ)`
   normalization that WRF carries even for "dry" runs? Compare the exact
   coefficient formula in ERF/WRF source.
3. **ρθ″ halo staleness.** The loop driver fills ρθ″ halos twice per substep
   (before damping and again after). Verify the order against WRF reference:
   if the pre-PGF read at the next substep sees stale halos on ρθ″, we get
   spurious gradients that feed back into ρu″.

### Per-substep instrumentation & damping-variant matrix (2026-04-24)

Temporarily added env-gated `@info` printing inside the substep loop to
watch per-substep max of each perturbation field. The signal was
unambiguous: a two-substep acoustic oscillation between ρu″ and ρθ″
that amplifies across stages.

**Step 5 stage 3 (6 substeps, PressureProjectionDamping(0.1))**:
```
substep 1/6: |ρu″|= 0.94  |ρθ″|=10.94
substep 2/6: |ρu″|=19.49  |ρθ″|= 2.60   ← 20× jump
substep 3/6: |ρu″|=14.63  |ρθ″|= 9.86
substep 4/6: |ρu″|= 1.26  |ρθ″|= 2.39
substep 5/6: |ρu″|= 6.75  |ρθ″|= 6.26
substep 6/6: |ρu″|=16.42  |ρθ″|= 3.43
```

**Step 6 stage 3 (same config)**:
```
substep 1/6: |ρu″|=  5.33  |ρθ″|= 93.57
substep 2/6: |ρu″|=162.61  |ρθ″|=104.47   ← runaway
```

Then I swept damping strategy and substep count:

| Damping                              | N   | Result                              |
| ------------------------------------ | --- | ----------------------------------- |
| `NoDivergenceDamping`                | 12  | crash @ step 7 (damping is mandatory) |
| `PressureProjectionDamping(0.1)`     | 6   | crash @ step 6 (original failure)    |
| `PressureProjectionDamping(0.1)`     | 12  | crash @ step 9                       |
| `PressureProjectionDamping(0.1)`     | 24  | crash @ step 8                       |
| `PressureProjectionDamping(0.5)`     | 12  | runs, max\|u\| rebounds by step 15   |
| `ThermodynamicDivergenceDamping(0.1)`| 12  | **runs cleanly**, max\|u\| decays    |

### Diagnosis

1. The instability is a real horizontal-acoustic amplification of the
   2Δτ mode. `NoDivergenceDamping` confirms: damping is load-bearing.
2. Increasing the substep count alone (N=12, N=24) does not help much at
   `PressureProjectionDamping(0.1)`. So it is not a pure acoustic-CFL
   problem.
3. **The two damping implementations are NOT on the same scale at equal
   coefficient.**
   - `ThermodynamicDivergenceDamping(β)` applies a momentum correction
     `ρu″ += 2β·Δx/Δτ · δx(δτρθ″) / (2θ_edge)` — has an effective
     horizontal diffusivity `ν ≈ 2β·Δx²/Δτ`. For β=0.1, Δx=156 m,
     Δτ=0.167 s, that is ν ≈ 29000 m²/s — a large, scale-aware damping.
   - `PressureProjectionDamping(β)` multiplies only a fractional forward
     projection of the PGF source by β. No length/time scale enters the
     coefficient — it is a dimensionless number.
   - At nominal β=0.1, `ThermodynamicDivergenceDamping` is effectively
     much stronger on this grid.
4. Klemp 2007 documents β=0.1 as the canonical PressureProjection
   coefficient for synoptic-scale WRF runs. For a sharp-IC
   high-resolution case (Δθ=10 K cone in a 10 km bounded box), this
   appears insufficient; Breeze's default of β=0.5 bounds it but is also
   marginal (rebounds from step 12 onward).

### Conclusion for the bug hunt

The failure is a **numerical-stability issue specific to stiff ICs**,
not a Breeze-introduced code bug. The projection-damping formula matches
ERF's docs and linearizes to the same filter as
`ConservativeProjectionDamping`. But β=0.1 is simply too weak for this
stress test.

**Recommendations for users of the substepper on stiff ICs:**
1. Default `PressureProjectionDamping(coefficient=0.5)` (already the
   Breeze default in `time_discretizations.jl`) is a reasonable choice
   for BCI-lifecycle-scale problems.
2. For LES-scale sharp ICs (like the dry thermal bubble validation),
   prefer `ThermodynamicDivergenceDamping(coefficient=0.1)` — the
   momentum-correction form with a length-scale-aware coefficient.
3. Adaptive CFL outer stepping (as in REPORT §2.07) is still the
   pragmatic answer for production runs.

**Candidate for a true code-level fix:** consider normalizing
`PressureProjectionDamping`'s coefficient so that β=0.1 gives a damping
on the same effective scale as the Klemp 2018 form. This would make the
two strategies interchangeable at the user-facing coefficient level.
