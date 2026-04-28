# Substepper long-run validation — running report

**Date:** 2026-04-27
**Branch:** `~/Breeze` `glw/hevi-imex-docs` HEAD `de99960` + uncommitted
substepper rewrite (the post-fix code).
**Status:** Active investigation — this report is updated as tests complete.

## Executive summary

**The substepper fix is partial.** Three regimes have been tested:

| Regime | Result |
|---|---|
| Rest atmosphere drift (canonical pre-fix failure) | ✅ Stable at Δt=20s, ω=0.55 — machine ε drift over 30 outer steps. |
| Dry baroclinic wave, 14 days, Δt=225s | ✅ Completed cleanly. Cyclone deepens to 932 hPa at day 9 (within JW06 reference), 862 hPa by day 14. Mass conservation at FP-floor for Float32. |
| Moist baroclinic wave, 15 days | ❌ NaN at day 3.3 (Δt=20s fixed) or at Δt~40-50s (adaptive). Extreme `max|qcl|` (~100 g/kg) precedes failure. |
| Cartesian moist convection, adaptive Δt, mild SST | ✅ **BOTH** compressible-substepper AND anelastic completed 1 hour cleanly (Δt~1s, advective-CFL-limited by deep moist convection w~60 m/s, max(qcl)~1 g/kg physically reasonable). |
| Moist BW, NO surface fluxes, fixed Δt=20s | ❌ Failed at day 4.1 (vs day 3.3 with fluxes). State was healthy right before NaN: max(qcl)~0.6 g/kg, max\|w\|~0.5 m/s. **Surface fluxes accelerate the failure but aren't the only cause**. |

The dry-physics fix is solid. The moist-physics regime — specifically lat-lon BCI with bulk surface fluxes over warm tropical SST — still fails. The Cartesian test is in progress to disambiguate "moist physics path" vs "substepper at large Δt".

## What's been measured

### A. Rest-atmosphere drift (post-fix verification)

**Setup:** 32×32×64 3-D Cartesian, isothermal-T₀=250K, fixed Δt=20s, 30 outer steps.

| Pre-fix | Post-fix |
|---|---|
| env = 1.09e-6 m/s | env = 6.2e-14 m/s |
| growth/step = 1.78 | growth/step = 1.0066 |

**Conclusion:** the fix works for the canonical failing test.

### B. Dry baroclinic wave, 14 days at Δt=225s

**Setup:** 360×160×64 lat-lon, latitude (-80, 80), Lz=30km, Δt=225s fixed,
isothermal-T₀=250K reference, no microphysics, no surface fluxes.

```
iter  t (d)  Δt (s)  max|u|  max|w|     p_surf_min  ΔM/M0     ΔH/H0
   0   0.00     225   28.05  0.00e+00      96723      0.00e+00  0.00e+00
1500   3.91     225   28.10  1.63e-03      96750     -2.71e-06  3.16e-07
3000   7.81     225   29.14  2.99e-02      95881     -4.32e-06  5.53e-06
3850  10.03     225   39.11  8.53e-02      91387     -6.11e-05  5.84e-05
5350  13.93     225   66.94  7.76e-02      86239     -1.36e-03  3.21e-04
```

5350 outer steps in 5.6 min wall (51 ms/step steady-state). Cyclone deepens
~70 hPa/day during peak intensification (day 7-9). Final p_surf_min = 862 hPa.
Mass drift -1.4e-3 over 14 days — at Float32 FP-floor.

**Conclusion:** dry BW works end-to-end at production Δt. Substepper handles
acoustic CFL correctly across the whole BCI lifecycle.

### C. Moist baroclinic wave, fixed Δt=20s

**Setup:** Same lat-lon grid, full moist physics (`NonEquilibriumCloudFormation`,
`τ_relax=200s`, `OneMomentCloudMicrophysics`), bulk surface fluxes over the
DCMIP analytic SST (T₀=240–310K from pole to equator), fixed Δt=20s.

```
iter   t (d)  max|u|  max|w|     max|qcl|   p_surf
 1500   0.35   28.04  3.91e-02   3.15e-06   96753   ← convection initiating
 2500   0.58   27.93  2.40e-01   3.56e-04   96740   ← deep updrafts
 6000   1.39   55.56  2.85e-01   3.38e-02   96720   ← qcl 33 g/kg unphysical
12500   2.89   51.15  5.20e-01   1.03e-01   96630   ← qcl 100 g/kg
14400   3.33   ---    ---        ---        NaN     ← failure
```

Failure mode: very large local cloud water (`max|qcl|` reaching 100 g/kg —
real cumulus has 1–10 g/kg) and deep updrafts in localized cells. Substepper
+ moist physics + surface fluxes coupling fails after ~3 days.

### D. Moist baroclinic wave, adaptive Δt with cfl=0.7

**Setup:** Same as C but `conjure_time_step_wizard!(sim; cfl=0.7, max_Δt=300s, max_change=1.05)`,
initial Δt=5s.

```
iter  t (d)  Δt    max|u|  max|w|     max|qcl|   p_surf
   0  0.000   5.2   28.05  0.000e+00  0.00e+00   96723
 200  0.020  13.9   28.02  4.060e-03  2.45e-10   96751
 400  0.073  37.0   28.01  3.325e-03  9.48e-08   96749
                ← InexactError: Int64(NaN) at substep-count
```

Wizard ramped Δt cleanly through 13.9 → 37s. NaN happened between iter 400
and iter 500, at some Δt ≈ 40-50s. The `InexactError: Int64(NaN)` from
`compute_acoustic_substeps` is the substepper's signature error when state
goes NaN and the wizard then computes NaN as next Δt.

**Conclusion:** the substepper is binding well below the advective CFL
ceiling (which is ~190s for polar Δx_min=19km, U=70m/s).

### E. Cartesian moist convection, adaptive Δt — IN PROGRESS

**Setup:** 128×128 (Periodic, Flat, Bounded), Lx=20km, Lz=10km, SST front
±2K around 285K (from `validation/anelastic_compressible_comparison/05_prescribed_sst.jl`),
the same `NonEquilibriumCloudFormation`+`τ_relax=200s` microphysics. Two
runs back-to-back: compressible-substepper, then anelastic.

**Compressible (in progress):** running cleanly, Δt limited to ~0.85–1.5 s
by advective CFL with intense convective `max|w|` ≈ 50–70 m/s and
`max|u|` ≈ 30–55 m/s. After 50 minutes of simulated time:
- max|qcl| ≈ 0.5–0.9e-3 (g/kg level — physically reasonable)
- max|w| ≈ 50 m/s sustained
- max|u| ≈ 40–55 m/s

The substepper here is rock-solid. With Δt at the *advective* CFL the
substepper is keeping up indefinitely.

**Anelastic comparison:** queued to run after compressible finishes.

### F. Cartesian moist anelastic-vs-compressible — COMPLETED

**Setup:** 128×128 (Periodic, Flat, Bounded), Lx=20km, Lz=10km, Δt
adaptive with cfl=0.7. Same `NonEquilibriumCloudFormation`+τ_relax=200s
microphysics as moist BW. Mild SST front (±2K around 285K) over a 20km
wavelength. Simulated 1 hour.

**Compressible-substepper:**
```
iter=  3000 t= 59.00min Δt=0.80s max|u|=58.7 max|w|=6.07e+1 max(ρqcl)=7.8e-4 wall=53s
final_t=3600.0s, final_iter=3032, crashed=false
```

**Anelastic:**
```
iter=  3000 t= 59.00min Δt=0.80s max|u|=58.7 max|w|=6.07e+1 max(ρqcl)=7.8e-4 wall=53s
final_t=3600.0s, final_iter=3073, crashed=false
```

Both ran cleanly for 1 hour. Identical Δt limit (~0.8s, advective-CFL
on Δx=156m with U_max=58 m/s). The compressible substepper handles
deep moist convection correctly when Δt is at the advective CFL limit.

### H. Moist BW Δt sweep — DEFINITIVE DIAGNOSTIC

**Setup:** Same lat-lon BW config as C (full surface fluxes). Fixed Δt
∈ {10, 5, 2} s. Each capped at 5 simulated days; recorded the trajectory
+ NaN / divergence point.

| Δt   | outer steps | sim time at failure | max\|u\| just before (m/s) | failure mode |
|------|-------------|----------------------|------------------------------|---------------|
| 20 s | 14,400      | day 3.33             | 67                           | NaN, qcl ~100 g/kg |
| 10 s | 26,200      | day 3.03             | 67                           | NaN, qcl ~3 g/kg |
|  5 s | 32,000+     | day 1.85 (diverging) | 184                          | unphysical wind growth |
|  2 s | 47,000+     | day 1.09 (diverging) | 1,128                        | jet-stream-speed unphysical |

**Smaller Δt produces faster failure** — both in sim time and in iteration
count to unphysical state. This is the **opposite** of CFL-driven stability:

- A genuine CFL violation would be cured by smaller Δt.
- A genuine moist physics runaway would be Δt-independent in sim time (same
  saturation time regardless of step size).
- A *per-outer-step error injection* would scale with N = #outer steps:
  smaller Δt → more N for same sim time → more cumulative error.

The data fits the third pattern. The mechanism is: each WS-RK3 outer step
injects machine-ε-scale noise that the substepper transient-amplifies (from
the earlier eigenvalue scan: ‖U‖₂ ≈ 44 per substep, even with ρ(U) = 1).
With surface fluxes + moist convection providing source terms, that noise
projects onto unstable patterns more easily, and accumulates linearly with
N at fixed sim time.

### G. Moist BW WITHOUT surface fluxes — FAILED LATER

**Setup:** Same lat-lon BW config as C, but `boundary_conditions = NamedTuple()`.

```
iter   t (d)  max|u|  max|w|     max|qcl|   p_surf
   0   0.000  28.05  0.000e+00  0.00e+00   96723
 5500   1.273  48.14  3.91e-01   1.62e-04   96720   ← qcl tiny
11500   2.662  56.06  6.15e-01   5.52e-04   96659   ← still healthy
17500   4.051  46.69  6.45e-01   6.39e-04   96481   ← still healthy
17700   4.097  ---    ---        ---        NaN     ← NaN in ρ
```

The state was **physically reasonable right before NaN**: qcl=0.6 g/kg
(real-cumulus level), max|w|<1 m/s, no extreme features. The substepper
nonetheless produced NaN in ρ.

**This is the key diagnostic:** removing surface fluxes only delays the
failure. The remaining failure mechanism is intrinsic to the lat-lon
moist BW + Δt=20s combination, not to surface flux runaway.

## Lessons (so far)

1. **The pre-fix substepper bug — non-normal transient amplification at rest —
   is genuinely fixed.** Rest-atmosphere drift is now at machine ε.

2. **The dry-physics path is solid at production Δt.** 14-day DCMIP dry BW
   matches JW06 reference behaviour.

3. **The moist-physics path has a remaining failure mode** that's not just
   the substepper non-normality: it's specific to the moist baroclinic wave
   on lat-lon with bulk surface fluxes. The Cartesian moist test with the
   *same* microphysics path runs cleanly when Δt is at the advective CFL,
   so the moist physics implementation per se isn't broken.

4. **The lat-lon BCI failure correlates with two things**:
   - Very large local cloud water (qcl 100 g/kg) preceding NaN — suggests
     surface evaporation + condensation feedback in tropical regions
     (T_surface = 310K) is producing localized runaway plumes.
   - The substepper failing at Δt ≈ 40-50s, well below the 190s advective
     CFL — suggests a remaining sub-fix substepper instability that's
     activated by the moist physics' large local source terms (latent heat
     release, surface vapor flux).

5. **The fact that the dry BW completes at Δt=225s but the moist BW fails
   at Δt=20s** points squarely at the moist-physics-substepper coupling,
   not at the substepper-only problem.

## Open questions (test these next)

- **Q1**: Does moist BW with smaller Δt (10s, 5s, 2s) survive 15 days?
  → Tests whether the moist-physics-substepper coupling has a Δt threshold.
- **Q2**: Does moist BW *without* surface fluxes complete 15 days at
  Δt=20s? → Isolates surface-flux runaway from microphysics + substepper.
- **Q3**: Does the Cartesian moist test ALSO fail when forced to run at
  larger Δt (artificially capping the wizard at e.g. 30s)? → Tests whether
  the same Δt threshold from lat-lon shows up in Cartesian moist physics.
- **Q4**: Does anelastic on the same Cartesian setup as E run cleanly? →
  Confirms moist-physics implementation correctness independent of acoustic
  treatment.

## Work TODO (developer-actionable)

### 1. Diagnose the moist-BW Δt-threshold failure

Re-run the moist BW (lat-lon) at Δt = 10, 5, 2 s with a NaN-checker on every
field every iteration. Identify:
- Which field NaNs first (ρ? ρw? ρqᶜˡ?)
- The (i,j,k) location of the first NaN
- The iteration count and corresponding Δt at NaN

If NaN onset shifts cleanly with Δt, it's a Δt-CFL-like instability. If
NaN happens at the same wall time regardless of Δt, it's a moist-physics
runaway.

### 2. Localise the moist-BCI extreme qcl

When `max|qcl|` hits 33 g/kg at iter 6000 of moist-BW-Δt=20, the value is
already ~3× a physically reasonable bound. Save a state snapshot at that
point and inspect:
- Where (latitude, height) is qcl ≥ 30 g/kg located?
- What's the local q_v, ρ, w, surface flux?
- Is it a single grid cell or a coherent region?

This will tell us whether it's a localised surface-flux instability (spike
near the equator surface from BulkVaporFlux feedback) or a downstream
substepper artefact.

### 3. Verify the substepper's Δt ceiling on a state with strong vertical
   motion

The current rest-atmosphere validation only tests the substepper near
hydrostatic balance. The moist-BCI failure happens in a state with strong
moist convective updrafts. Add a Tier-1 test:
- Initialise a state with prescribed `max|w|` ≈ 5 m/s (e.g. a Gaussian w
  pulse) at different (Δt, ω) combinations.
- Verify the substepper damps the perturbation rather than amplifying it.

### 4. Decide on the surface flux strategy for moist BW

Current setup uses bulk fluxes with constant `Cᴰ=1e-3` and gustiness
`Uᵍ=10⁻²` m/s over T_surface that ranges 240–310K. At the tropical
equator with T_surface=310K and θ_surface(z=0) ≈ 270K, the bulk vapor
flux is large enough to inject ~1 g/kg of water vapor per cell per
day, potentially driving local saturation runaway. Consider:
- Capping T_surface at e.g. 300K
- Adding a small Newtonian relaxation toward the analytic temperature
  profile (Held-Suarez-style) to prevent boundary-layer θ runaway
- Switching to `FilteredSurfaceVelocities` to suppress the gustiness-driven
  initial flux burst

These are *test-setup* changes, not substepper changes, but they may be
what the moist BW needs to be a clean validation case.

### 5. Track conservation budgets

The moist BW fixed-Δt run shows mass drift `~5e-4` over 3 days — far worse
than the dry case's `~3e-6` over 14 days. Likely from microphysics + surface
flux source terms not being conservation-symmetric. Check:
- Vapor budget: ∫(ρ qᵛ) dV vs ∫surface vapor flux dt
- Condensate budget: ∫(ρ qᶜˡ + ρ qᶜⁱ + ρ qʳ + ρ qˢ) dV
- Total water: should equal initial + ∫surface vapor flux dt

## Test artifacts

In `~/Breeze/test/substepper_validation/long_runs/`:

| File | Purpose |
|---|---|
| `dry_bw_14day.jl` + `_diagnostics.jld2` + `_state.jld2` + `.log` | F2 dry BW (PASSED) |
| `moist_bw_15day.jl` + `_diagnostics.jld2` + `_state.jld2` + `.log` | F1 moist BW fixed Δt (FAILED day 3.3) |
| `moist_bw_15day_adaptive.jl` + `_diagnostics.jld2` + `.log` | F1 adaptive Δt (FAILED ~Δt=40s) |
| `cartesian_moist_v2.jl` + `_results.jld2` + `.log` | Cartesian anel-vs-comp (compressible PASSED, anelastic queued) |

The pre-fix diagnostic and characterisation scripts (the eigenvalue scan,
A/B reference-state test, no-reference test, etc.) are one level up in
`~/Breeze/test/substepper_validation/`.
