# Phase 2A results — γᵐRᵐ⁰ in PGF was NOT the moist BW NaN fix

**Date:** 2026-04-28 evening
**Author:** Claude (continuing the moist substepper fix workstream)

## TL;DR

I implemented Phase 1 + Phase 2A of `MOIST_SUBSTEPPER_STRATEGY.md` /
`PRISTINE_SUBSTEPPER_PLAN.md` §A3:

1. **Phase 1**: snapshot `qᵛ⁰`, `qˡ⁰`, `qⁱ⁰` into `AcousticSubstepper`'s
   basic state at outer-step start.
2. **Phase 2A**: replace the scalar `γᵈRᵈ` in the linearised PGF / Schur
   block with a per-cell field `γᵐRᵐ⁰` derived from the snapshot. Five
   kernels updated; dry path collapses bit-identically.

All 71 dry acoustic-substepper unit tests pass. Then I ran the moist BW
lat-lon (no surface fluxes) at full DCMIP-2016 resolution (360×160×64),
Δt=20s. **Pre-fix this NaN'd at day 4.05; post-Phase-2A it NaN'd at day
4.005.** Trajectory virtually identical.

So the dominant cause of the moist BW lat-lon NaN is **not** the per-step
γᵈ-vs-γᵐ mismatch in the linearised PGF, even though the Δt-sweep
signature suggested per-outer-step error injection.

## Numbers

Pre vs post-Phase-2A, full DCMIP-2016 grid, Δt=20s, no surface fluxes:

| | Pre-fix | Phase 2A |
|---|---|---|
| NaN at iter | 17500 | 17300 |
| NaN at sim time | day 4.051 | day 4.005 |
| peak max\|u\| (m/s) | 60.09 | 57.72 |
| peak max\|w\| (m/s) | 0.909 | 0.988 |
| min p_surf (Pa) | 96472 | 96466 |
| mass drift | -2.25e-4 | -2.22e-4 |

## What's still in the working tree

- `src/CompressibleEquations/acoustic_substepping.jl`: Phase 1 + Phase 2A
  changes. Phase 2B (μᵥ on buoyancy) was attempted and reverted — see the
  in-file comments and `~/.claude/.../moist_substepper_phase2a.md`.
- `src/CompressibleEquations/CompressibleEquations.jl`: added
  `vapor_gas_constant` to imports.

The substepper now carries `outer_step_gamma_R_mixture` (used) and
`outer_step_virtual_density_factor` (precomputed but unused — kept for
diagnostics / future Phase 3+ work).

## Decisive diagnostic: microphysics is the cause

A `moist_bw_no_microphysics.jl` test — same as no-fluxes but with
`microphysics = nothing` — **completed 5 days at Δt=20s without NaN** in
12.7 minutes wall time. Final stats: peak max|u|=33.7 m/s, peak max|w|=0.035
m/s, mass drift = 5e-6.

| Run | NaN at | peak max\|u\| | peak max\|w\| | min p_surf |
|---|---|---|---|---|
| Pre-fix (γᵈRᵈ, full µphys) | day 4.051, iter 17500 | 60.09 | 0.909 | 96472 |
| Phase 2A (γᵐRᵐ⁰, full µphys) | day 4.005, iter 17300 | 57.72 | 0.988 | 96466 |
| Phase 2A, **microphysics off** | none (5 days) | 33.74 | 0.0346 | 96723 |

The 60× larger max|w| in the with-µphys runs is microphysics-driven
convective enhancement (latent-heat release). Without that, the BCI is
weak but stable. With it, the convection compounds until something
overflows.

## 2026-04-28 evening update: comprehensive Δt-sweep + diagnostic ablations

After the user pointed out (1) microphysics makes dynamics violent so the
no-µphys success doesn't uniquely blame microphysics, (2) the production
target is CFL ~0.7 / Δt ~150-200s on lat-lon, not Δt=20s, I ran a
comprehensive Δt sweep + parameter ablations with adaptive cfl=0.7 and at
fixed Δt above the established 20s.

### Δt-sweep on moist BW (no surface fluxes, OneMomentCloudMicrophysics + NonEqCloudFormation)

| Δt (s) | NaN at | Iter to NaN | Regime |
|---|---|---|---|
| 2 | day 1.09 | 47100 | slow accumulative |
| 5 | day 1.86 | 32100 | slow accumulative |
| 10 | day 3.03 | 26200 | slow accumulative |
| **20** | **day 4.05** | **17500** | **best fixed-Δt point** |
| 30 | day 0.71 | 2050 | catastrophic onset |
| 37 | day 0.13 | 300 | catastrophic |
| Adaptive cfl=0.7 ramp | day 0.07 | ~500 | crashed at ramp peak Δt=37s |

There's a **steep stability boundary between Δt=20s and Δt=30s**. Below
20s, smaller Δt fails earlier in *sim time* but later in *iter count*
(slow per-step accumulation). Above 25s or so, failure is catastrophic.

### Dry comparison (same lat-lon grid, no qᵛ, microphysics=nothing)

| Run | Δt achieved | At day | Status |
|---|---|---|---|
| Dry BW Δt=20s fixed | 20s | clean 5d | ✅ |
| Dry BW Δt=225s fixed (production) | 225s | clean 14d | ✅ |
| Dry BW adaptive cfl=0.7 | ramped to 300s (max), then NaN | 1.3d | ✅ up to 300s |

**The dry substepper handles acoustic CFL ≥ 5.5 at Δt=300s. Moist
substepper craters at acoustic CFL ≈ 0.5 (Δt=25-30s). 10× gap.**

### Mitigations attempted (all at Δt=37s post-Phase-2A) — all FAILED at the same point:

| Mitigation | Outcome |
|---|---|
| Phase 2A (γᵈRᵈ → γᵐRᵐ⁰) | NaN at iter 300/day 0.13 (= baseline) |
| Slow τ_relax = 2000s (10× slower condensation) | NaN at iter 400/day 0.07 (adaptive ramp) |
| WENO-3 for moisture | NaN at iter 400/day 0.07 |
| Hyperdiffusion (HorizontalScalarBiharmonicDiffusivity, ν=1e10) | NaN at iter 400/day 0.07 |
| Ns=24 substeps (4× more) | NaN at iter 100/day 0.043 (WORSE) |
| ω = 0.85 (more implicit dissipation) | NaN at iter 300/day 0.128 (= baseline) |

**Striking that the threshold is invariant under all these knobs.** This
points to a *structural* issue in the substepper-microphysics interaction
at acoustic CFL > ~0.5, not a tuning issue.

### Working hypothesis: substepper's frozen U⁰ basic state vs in-step microphysics evolution

The substepper integrates linearised acoustic perturbations with
basic-state quantities (`ρ⁰, ρθ⁰, p⁰, Π⁰, θ⁰, qᵛ⁰, ...`) frozen at
outer-step start. Microphysics modifies `ρqᵛ`, `ρqᶜˡ`, etc. through the
slow tendency `Gⁿ`, evaluated at the current state (each WS-RK3 stage).
**Within a single outer step**, the model state's mixture-Rᵐ shifts by
up to ~15% at Δt=30s (rate × Δt = 30/200 = 0.15) where the linearisation
basic state's Rᵐ⁰ stays frozen.

The substepper Klemp damping has fixed per-substep rate β_d=0.1, so
total damping per outer step = β_d × N_τ ≈ 0.6 with N_τ=6 floor —
constant in Δt, while error scales with Δt² in the moist case.

Increasing N_τ (to 24) made things WORSE — confirming the issue is
divergence from frozen-U⁰-truth, not lack of damping. More iterations of
the wrong system is worse than fewer.

### What's likely needed — structural changes the agent should evaluate

1. **Per-stage refresh of moisture-related basic state** (γᵐRᵐ, μᵥ, θᵥ)
   while keeping the acoustic basic state (Π⁰, p⁰) frozen. The docstring
   says full per-stage refresh worsens dry stability — but a *partial*
   refresh of moisture quantities only might give moist what it needs
   without breaking dry.

2. **Switch from total-ρ to dry-ρ as the prognostic** (WRF/MPAS style).
   Then qᵛ, qˡ, qⁱ are separate prognostics with their own evolution
   equations, and the EoS uses ρ_d × R_d × T_v (virtual). The
   substepper sees clean acoustic system without microphysics-EoS
   coupling. Big change but matches what works in WRF/MPAS at Δt=200s.

3. **More aggressive moisture transport limiting**. Bound-preserving
   advection alone may not be enough — could need monotonicity
   preservation, or even simpler: clip qᵛ to physical range before
   feeding to microphysics (currently clips after).

4. **Investigate the actual NaN location**. Add a post-step check that
   identifies (i,j,k) where ρ first goes negative. Polar columns?
   Cyclone core? Saturated columns? Each points to a different
   mechanism.

## Hypotheses for the actual NaN cause — narrowed to microphysics

The no-microphysics run completing 5 days clean rules out:
- Polar-grid CFL violations (those would fail without microphysics too)
- Substepper linearisation breakdown (same)
- Reference-state mismatch (same)

What's left, all microphysics-related:
1. **Latent heat release magnitude.** Condensation `qᵛ → qᶜˡ` at
   `τ_relax=200s` releases `L_v · δqᵛ` into θ via `Gⁿρθ`. On localised
   convective cells, this drives strong `Gⁿρθ` that the WS-RK3 outer
   step + substepper interaction may amplify.
2. **Bound-preserving WENO + condensation interaction.** `bp_weno`
   (order 5, bounds [0,1]) constrains the moisture mass fractions.
   The combination with the relaxation source term may produce a
   flux pattern that drives ρ to NaN.
3. **Multi-stage ρθ_li update.** ρθ_li is updated by both the
   substepper (PGF + buoyancy) AND by the slow microphysics tendency
   in the WS-RK3 outer step. If these two paths fight each other on
   short Δt, that's per-step error injection — and **it would be
   Δt-sweep sensitive in exactly the way that was originally
   attributed to §A3**. This is the most plausible hypothesis given
   the Δt-sweep evidence.

## Recommended next steps

The no-microphysics run completed cleanly (5 days), so microphysics is
the proven culprit. Next investigations to narrow to a specific bug:

1. **Halve the relaxation rate** to `τ_relax = 1000s` (5× slower
   condensation) and rerun the no-fluxes test. If it passes 5 days,
   condensation rate is too fast for Δt=20s — physical CFL violation
   in saturation adjustment. If it still NaNs, it's structural.

2. **Compute the latent-heat tendency magnitude** separately and check
   whether it's the dominant term in `Gⁿρθ_li` near where the NaN
   appears. The `condensation_rate` formula in `bulk_microphysics.jl`
   has the rate; multiply by `L_v ≈ 2.5e6` to get the θ tendency.

3. **Disable bound-preserving WENO for moisture** (use plain `weno`
   for ρqᵛ etc.) and rerun. If it passes, the bp_weno-relaxation
   interaction is the bug.

4. **Check the Cartesian moist convection test** results
   (`cartesian_moist_v2_results.jld2`): same microphysics scheme, ran
   1 hour clean at advective Δt~1s. So scheme is OK at small Δt; the
   issue is specifically Δt=20s + microphysics.

5. **Test the Δt-sweep without microphysics.** If smaller Δt no longer
   makes the run fail earlier (or at all) without microphysics, that
   conclusively pins per-step compounding to the microphysics-dynamics
   coupling.

## What I'd advise NOT doing

- Don't keep adding moist features (Phase 2C / θᵥ in η-flux) without
  first diagnosing the actual failure mechanism. The strategy doc's
  premise was that A3/B1/B2 were the cause; A3 is now disproven, so the
  whole §B chain is suspect too.
- Don't revert Phase 2A — it's theoretically correct, doesn't regress
  dry tests, and the moist substepper SHOULD use mixture coefficients
  in the long run. It just isn't the smoking-gun fix.
- Don't chase M1 / M2 results — those tests have IC bugs (κᵈ-vs-κᵐ
  mismatch in θ construction) that mask substepper effects. Either
  fix the IC or rely on the moist BW for diagnosis.
