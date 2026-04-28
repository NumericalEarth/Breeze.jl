# Moist baroclinic wave example — future plan

**Status:** the dry baroclinic-wave example (`examples/baroclinic_wave.jl`)
is in. The moist version will be added in a follow-up PR alongside (or
after) a switch to a dry-density formulation. This document captures the
context, evidence, and remaining work.

## Why moist BW is deferred

The current Breeze substepper handles the **dry** BCI cleanly at the
production target Δt (CFL ≈ 0.7), but cannot reach the same Δt with
microphysics on. The Δt-sweep evidence (post-Phase-2A, full DCMIP-2016
no-fluxes setup, 1° grid, 360×160×64):

| Δt (s) | NaN at | Iter to NaN |
|---|---|---|
| 2 | 1.09 d | 47100 |
| 5 | 1.86 d | 32100 |
| 10 | 3.03 d | 26200 |
| **20** | **4.05 d** | **17500 (best)** |
| 30 | 0.71 d | 2050 |
| 37 | 0.13 d | 300 |

Stability cliff between Δt=20s and Δt=30s. At 2° resolution, the moist
case is **less** stable than at 1° (NaN day 2.34 at Δt=20s vs 4.05 at
1°), so coarsening doesn't help.

For the same grid the dry adaptive cfl=0.7 ramps to Δt=300s before NaN
(vs ~37s for moist). The moist substepper has a 10× tighter CFL margin
than the dry substepper.

**The moist BW at Δt=20s is NOT viable as a 5–15-day example** — it
NaNs at day 4. We can run partial integrations (e.g., 3 days at Δt=20s
on the 1° grid, ~1 hour wall) but that misses the cyclogenesis peak and
isn't suitable as a documentation example.

## What surface-level fixes were tried (all negative)

All at the same iter ~400 / Δt=37s ramp peak adaptive cfl=0.7:

| Knob | Outcome |
|---|---|
| Phase 2A (γᵐRᵐ⁰ in PGF/Schur) | no change |
| τ_relax 200s → 2000s (10× slower condensation) | no change |
| WENO order 5 → 3 for moisture | no change |
| Hyperdiffusion (constant ν=1e10 biharmonic) | no change |
| Substep count 6 → 24 | WORSE |
| Forward weight ω 0.65 → 0.85 | no change |
| Resolution 1° → 0.5° | no change |
| Resolution 1° → 2° | WORSE (moist) |

The threshold is **invariant** under all surface-level tuning. This
points to a structural mismatch between the substepper's frozen-`U⁰`
linearization and microphysics-driven evolution of moisture mass
fractions inside an outer step.

## Plan: switch to dry-density formulation

WRF/MPAS achieve Δt = 200 s on similar lat-lon grids with the same kind
of mixed-phase microphysics. Their formulation differs from Breeze's in
one structural way: they prognose **dry** density `ρ_d` (or
hydrostatic-pressure-based mass coordinate), not total density `ρ`. The
EoS is `p = ρ_d R_d T_v` with virtual temperature `T_v`. The acoustic
substepper sees a clean linearised system that is decoupled from
microphysics-driven mass shifts (because `ρ_d` doesn't change when
condensation moves water between species).

### Phase A — formulation switch (large refactor)

1. Change `LiquidIcePotentialTemperatureFormulation` (or add a new
   `DryDensityLiquidIcePotentialTemperatureFormulation`) so that:
   - the prognostic density is `ρ_d`, not total `ρ`
   - the EoS computes `p = ρ_d R_d T_v`
   - all advection / continuity is in terms of `ρ_d` and the moisture
     mass fractions are advected by `(ρ_d u_i)`
2. Update the substepper to use `ρ_d, ρ_dθ_li` as basic state. The
   substepper no longer needs the moisture-aware `γᵐRᵐ⁰` because it
   sees `R_d` in the linearised PGF (the moist correction is already
   absorbed into `T_v` in the EoS).
3. Microphysics: track moisture mass fractions `q^v, q^l, q^i, q^r,
   q^s` directly (per unit `ρ_d`, not per unit `ρ_total`). Their slow
   tendencies look the same as today, but feed into a different EoS.
4. Surface fluxes: `BulkVaporFlux` etc. need to translate between the
   two density conventions. Should be a thin wrapper.
5. Buoyancy: `g · ρ_d` (no virtual factor needed since `T_v` is in the
   EoS). The reference state is dry-hydrostatic by definition.

### Phase B — reuse Phase 1+2A infrastructure

The Phase 1 moisture-snapshot machinery
(`outer_step_vapor_mass_fraction` etc., `snapshot_moist_basic_state!`)
stays useful: even in the dry-density formulation, the substepper may
want to use moisture-aware `T_v⁰` per cell to avoid a per-step EoS
re-evaluation. The Phase 2A `γᵐRᵐ⁰` field is essentially **unused** in
the dry-density formulation (collapses to constant `γ_d R_d`); could be
removed or kept as a no-op safety field.

### Phase C — moist BW example

Once Phase A lands and dry tests still pass:

1. Verify moist BW lat-lon at Δt = 60–100 s (start small, ramp up
   only if stable). Target Δt = 200 s adaptive cfl=0.7.
2. Add a **moist** section to `examples/baroclinic_wave.jl` that runs
   sequentially after the dry section. Both should complete in <30
   minutes on one GPU (Δt ≈ 200s × 14 days = 6048 outer steps at 2°
   ~ 3 minutes, projected).
3. Add a `moist_bw` regression test (M-tier acceptance from
   `MOIST_SUBSTEPPER_STRATEGY.md`):
   - M9: lat-lon 14d Δt=200s, no NaN, max|w|<5 m/s
   - M11: full physics with surface fluxes
   - M12: Δt-sweep regression (smaller Δt should NOT fail earlier)

## Open questions for the dry-density switch

- Does Breeze have other formulations that depend on total-`ρ` as the
  natural prognostic? (e.g., `StaticEnergyThermodynamics` — needs a
  parallel dry-density variant or stays on `ρ_total` as a separate
  branch)
- How does the dry-density formulation interact with the
  `AnelasticDynamics` path? Anelastic already uses a dry reference
  density — likely no impact, but worth confirming.
- Surface fluxes: `BulkVaporFlux` provides `ρ_t · q_t` to the surface.
  Need to convert this to a `ρ_d · q_d` flux in the new formulation.

## Pre-work to make merging this PR safe

The current PR (Phase 1 + 2A + diagnostic study + dry BW example
update) should merge cleanly with the following caveats noted in the
PR description:

- Phase 2A's `γᵐRᵐ⁰` and `μᵥ⁰` fields are **infrastructure for the
  future moist substepper**; in current code they are read by the
  five PGF kernels (γᵐRᵐ⁰) but reduce bit-identically to the dry
  constant. The `μᵥ⁰` field is allocated but unread.
- The diagnostic study showed Phase 2A's `γᵐRᵐ⁰` swap does NOT
  unlock larger Δt for moist BW. The fix lives in Phase A above.
- The dry BW example is at 2° / Δt=450s / 14d; that runs in ~2.2
  minutes on one GPU. Production dry runs at 1° / Δt=225s also work
  (Phase 4.5 verified, 14 days clean).
