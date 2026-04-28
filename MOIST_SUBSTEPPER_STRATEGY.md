# Making the acoustic substepper moist-aware — strategy and test hierarchy

**Date:** 2026-04-28
**Status (2026-04-28 evening): Phase 1 + 2A LANDED but DID NOT FIX moist BW NaN.**
🔴 No-fluxes moist BW lat-lon at Δt=20s NaN'd at day 4.005 post-Phase-2A
vs day 4.05 pre-fix — essentially the same failure point. **A3 (`γᵈRᵈ →
γᵐRᵐ⁰`) is theoretically correct and stays landed (no dry regression),
but it is NOT the dominant cause of the moist BW lat-lon NaN.** The
Δt-sweep signature (smaller Δt → earlier failure) must come from a
different mechanism — see memory `moist_substepper_phase2a.md` for the
full pre/post comparison and candidate hypotheses (microphysics
coupling, polar-grid CFL, linearisation breakdown at deep cyclones).
- Phase 1 (snapshot moisture into basic state, no behaviour change): ✅
- Phase 2A (`γᵈRᵈ → γᵐRᵐ⁰` in linearised PGF + Schur): ✅, bit-identical dry collapse, all 71 dry tests pass — but did not fix moist BW NaN.
- Phase 2B (`g·μᵥ⁰·ρ′` in buoyancy): tried and reverted. **In Breeze's
  conservation form with `ρ` = total density, the momentum equation has
  `-g·ρ` (not `-g·ρv`).** μᵥ-on-buoyancy applies only in WRF/MPAS-style
  formulations that prognose dry density. Empirically Phase 2B made M1
  envelope worse (1.6e-1 vs 9.5e-2 pre-fix). The μᵥ⁰ field is still
  precomputed and stored in the substepper for future / diagnostic use.
- Phase 2C (θᵥ⁰ in η-flux): not done — same WRF-formulation question.
  Breeze prognoses `ρθ_li`, so the linearised flux uses `θ_li⁰`.
- A1 / A2 (moist `cs` in damping coef + substep count): skipped — moist
  cs differs from dry by < 0.5% at qᵛ=10g/kg, well within the 2× safety
  margin already in place.

**Open**: M1 / M2 ICs have a κᵈ-vs-κᵐ inconsistency (build θ with κᵈ
but model EoS uses κᵐ) that creates a ~0.06% IC-level pressure
mismatch dominating the small γᵐ-vs-γᵈ correction Phase 2A makes. They
cannot probe A3 cleanly until rebuilt. The definitive A3 test is the
moist BW lat-lon — Δt=20s no-fluxes pre-fix NaN'd at day 4.05; running
now to compare.

**Context:** Long-run validation post-substepper-fix shows the dry baroclinic
wave completes 14 days at Δt=225s but the moist baroclinic wave on a
LatitudeLongitudeGrid fails after 3-4 days at Δt=20s. The Cartesian moist
test (small box, deep convection) at advective-CFL Δt~1s succeeds; the
lat-lon moist BW at Δt=20s fails even *without* surface fluxes. The
Δt-sweep diagnostic shows that **smaller Δt makes failure earlier in sim
time** — opposite of CFL behaviour, characteristic of per-outer-step error
injection that compounds when N (number of outer steps) is large.

**Relation to existing planning artefacts.** This document operationalises
the items already enumerated in `validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md`
under "Phase 3 — Moist physics in the substepper":

- **A3 🔴** — `γᵈRᵈ` everywhere in the linearised PGF and Schur block (matrix
  lines 519, 539, 558; horizontal explicit lines 685, 690; sound force line
  760). Replaced by `γᵐRᵐ⁰(i,j,k)` here.
- **B1 🔴** — No moisture in the basic state. Adds
  `outer_step_vapor_mass_fraction` (and condensed phases) to the substepper.
- **B2 🔴** — Buoyancy `g·ρ′` uses dry density perturbation. Replaced by
  virtual-density perturbation `g·μᵥ⁰·ρ′` here.
- **A1 🟠** — Damping coefficient uses `cs = sqrt(γᵈRᵈ·300)` (line 364);
  becomes field-aware moist `cs(i,j,k)` once mixture quantities are
  snapshotted.
- **A2 🟠** — `compute_acoustic_substeps` likewise uses dry uniform-T
  `cs`; switched to moist `cs_max` from the basic state.

The PRISTINE plan estimates Phase 3 at 1–2 weeks; the test hierarchy
(M0–M12) below is the acceptance gate for that work.

The remaining hypothesis (per the developer agent's note): the substepper
is not moisture-aware. It linearises the acoustic system using *dry*
gas constants (γᵈ, Rᵈ) and the dry potential temperature, while the
prognostic state is the moist ρθ_li. The linearisation error scales with
mixture-vs-dry gas-constant ratio, surface humidity, and condensate; it
gets re-injected at every outer step.

This document lays out the modifications and a hierarchy of tests for
isolating, fixing, and validating the moist substepper.

## What "moist-aware substepper" means

In a dry atmosphere the linearised acoustic-buoyancy system uses

```
∂t μw = −γᵈ Rᵈ Π⁰ ∂z η − g σ + Gˢρw
∂t η  = −∇·(θ⁰ μ) + Gˢρθ
∂t σ  = −∇·μ + Gˢρ
```

with `γᵈ = cpᵈ/cvᵈ`, `Π⁰ = (p⁰/pˢᵗ)^κ`, `κ = Rᵈ/cpᵈ`, all dry. The
linearised pressure perturbation is `p_pert ≈ γᵈ Rᵈ Π⁰ η`, which derives
from the dry ideal-gas EoS `p = ρ Rᵈ T = ρ Rᵈ θ (p/pˢᵗ)^κ`.

For moist air the EoS is

```
p = ρ Rᵐ T,    Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ
```

with `qᵈ = 1 − qᵛ − qˡ − qⁱ` and `Rᵛ ≈ 461.5 J/(kg·K) ≈ 1.61 Rᵈ`. Mixture
heat capacity `cpᵐ = qᵈ cpᵈ + qᵛ cpᵛ + qˡ cpˡ + qⁱ cpⁱ` (commonly only qᵈ
and qᵛ contribute meaningfully). Speed of sound

```
cs² = γᵐ Rᵐ T,    γᵐ = cpᵐ/cvᵐ
```

is roughly 0.5–1 % faster than dry value at qᵛ≈10 g/kg. The buoyancy
term should use *virtual* density

```
ρv = ρ (1 + 0.608 qᵛ − qˡ − qⁱ)
```

so positive qᵛ reduces ρv (warm-moist parcel rises) and condensate
increases ρv (loading).

WRF / MPAS / ERF handle this by:

1. Computing `Rᵐ`, `cpᵐ`, `γᵐ` per cell from the prognostic moisture
   state, snapshotted at outer-step start as part of the linearisation
   basic state.
2. Replacing `γᵈ Rᵈ Π⁰` with `γᵐ Rᵐ Πᵐ⁰` in the linearised acoustic
   pressure-gradient term, where `Πᵐ⁰ = (p⁰/pˢᵗ)^κᵐ` and `κᵐ = Rᵐ/cpᵐ`.
3. Using virtual potential temperature `θᵥ = θ (1 + 0.608 qᵛ − qˡ − qⁱ)`
   in the linearised flux `−∇·(θᵥ⁰ μ)` for the η equation, so the
   linearised vertical-velocity-pressure coupling has the right buoyancy.
4. Using virtual density `ρv` for the buoyancy term `−g σᵥ`.

The reference for production codes:
- *WRF Technical Note (Skamarock et al. 2008/2019)* — describes how the
  ARW substepper handles moisture: virtual θ in the predictor, mixture
  gas constants in the implicit solve.
- *Klemp & Wilhelmson 1978*, *Wicker & Skamarock 2002* — original split-
  explicit derivations.
- *Klemp, Skamarock & Ha 2018* — divergence damping in HEVI/split-explicit
  schemes; companion paper that includes moisture handling.
- *Park et al. 2013 / 2019* — MPAS moist BCI validation, documents the
  Δt limits achievable when the substepper is properly moist-aware.

## Hypothesis (testable)

**Primary:** the lat-lon moist BW failure is caused by the substepper using
dry gas constants in the linearised acoustic system, while the
prognostic state is moist. The error per outer step scales with local
qᵛ; the lat-lon polar-equator gradient (qᵛ ranging from 1e-12 to 1.8e-2)
makes this error spatially structured, projecting onto Rossby and
gravity-wave modes that grow per outer step.

**Secondary:** the Cartesian moist test passes because (a) Δt is at the
advective CFL ~1s — much smaller, so per-outer-step error has less time
to accumulate per simulated day; (b) the qᵛ in the Cartesian test is
spatially uniform (no large gradients to project on dynamical modes).

**Predicted outcome of the fix:** with mixture-aware Rᵐ, γᵐ, and virtual
θᵥ in the substepper's linearisation, the moist BW lat-lon should
achieve advective-CFL Δt (~190 s), completing 15 days cleanly.

## Test hierarchy (M-tier)

Each test is a stand-alone Julia script. Pass criterion is quantitative.
Order is bottom-up: lower-tier failures invalidate higher-tier results.

### M0. Dry rest atmosphere — already passing

`test/substepper_validation/sweep_runner.jl`. Pass: max\|w\|≤1e-10 m/s
over 30 outer steps at Δt=20s. Re-run after every substepper change.

### M1. Moist rest atmosphere

**New.** Same as M0 but with `qᵛ` set to a small uniform profile (e.g. 1
g/kg). State is hydrostatically balanced with virtual temperature; θ
profile chosen so analytic ∂z p + g ρ_v = 0 holds.

**Pass:** max\|w\| ≤ 1e-10 m/s over 30 outer steps at Δt=20s, on a 32×32×64
3-D Cartesian box. With dry-substepper this is expected to drift; with
mixture-aware it should stay at machine ε.

### M2. Moist rest atmosphere with strong qᵛ gradient

**New.** A horizontally varying qᵛ(x) (e.g. 0 → 15 g/kg cosine, like the
moist BW meridional profile). Hydrostatic. At rest.

**Pass:** max\|w\| ≤ 1e-10 m/s. Tests whether the spatial mixture-gas-
constant gradient creates a spurious horizontal pressure gradient force.

### M3. Acoustic pulse in moist atmosphere

**New.** Small pressure pulse on uniform `qᵛ = 10 g/kg` background.
Propagation at speed `cs_moist = √(γᵐ Rᵐ T)`.

**Pass:** measured arrival time at a probe matches `cs_moist` within 1%.
With dry substepper, expect ~0.5% slower (uses cs_dry = √(γᵈ Rᵈ T)) which
is a clear signature.

### M4. Moist 2-D Cartesian inertia-gravity wave (Skamarock-Klemp 1994 moist analog)

**New.** Linear gravity-wave propagation on a moist hydrostatic
background. Compare numerical θ′ vs analytic moist linear-wave dispersion
relation.

**Pass:** L2 error in θ′ ≤ 5%.

### M5. Moist thermal bubble — anelastic vs compressible (Cartesian)

**New** but maps to existing `validation/anelastic_compressible_comparison/07_cloudy_thermal_bubble.jl`
template. Bubble of warm moist air rises, condenses, evolves. Both
solvers should agree to 5% on max\|w\|, max\|qcl\|, bubble centroid
height after 600s.

**Pass:** the matching criterion the developer agent already uses.

### M6. Cartesian moist convection adaptive Δt — already passing

`test/substepper_validation/long_runs/cartesian_moist_v2.jl`. Both
anelastic and compressible-substepper run 1 hour cleanly at advective-CFL
Δt~1s. Re-run as a regression after every fix.

### M7. Cartesian moist BW analog at LARGE Δt

**New.** A 2-D Cartesian box (Lx=20000km, Lz=30km, Δx≈100km matching the
1° equatorial spacing), with the DCMIP-2016 BW initial conditions
projected onto (x, z). Δt=225s, 14 days.

**Pass:** completes 14 days, no NaN, max\|w\| < 5 m/s.

This separates "curvilinear" from "Δt + moisture" — if it passes, the
remaining bug is curvilinear-specific.

### M8. Lat-lon dry BW — already passing

`test/substepper_validation/long_runs/dry_bw_14day.jl`. 14 days at Δt=225s.

### M9. Lat-lon moist BW NO surface fluxes, NO microphysics

**New.** Same as the lat-lon BW but with `qᵛ` initialised from the DCMIP
profile (1.8 g/kg max) and microphysics OFF (`microphysics = nothing`).
The qᵛ field is just a passive tracer that affects ρv. Δt=20s.

**Pass:** completes 15 days, no NaN.

This isolates "moisture in the linearised acoustic system" from
"microphysics integrated externally" and from "surface fluxes". It's the
cleanest single-knob test of moist-substepper-correctness.

### M10. Lat-lon moist BW microphysics ON, surface fluxes OFF

**Already failed at day 4.1 with current code** (no_surface_fluxes test).
Re-run as a regression once the moist-aware fix is applied. **Expected
to pass** post-fix.

### M11. Lat-lon moist BW microphysics ON, surface fluxes ON

**The full validation case**, currently failing at day 3.3.

**Pass:** completes 15 days at Δt=20s OR achieves Δt~150-190s under
adaptive stepping. Cyclone deepens to <970 hPa by day 9, qcl stays
< 5 g/kg.

### M12. Lat-lon moist BW Δt sweep regression

Re-run M11 at Δt ∈ {20, 10, 5, 2} s post-fix and verify that **smaller
Δt no longer fails earlier**. Specifically: same NaN-free behaviour,
same physical max\|u\| trajectory, same surface-pressure deepening at
day 9 ± 5 hPa.

This is the *signature* test that the per-outer-step error injection has
been removed.

## Predicted failure progression with current (dry-only) substepper

If our hypothesis is right:

| Test  | Current code |
|-------|--------------|
| M0    | ✅ pass |
| M1    | ❌ fail (drift from dry-vs-moist hydrostatic mismatch) |
| M2    | ❌ fail more (gradient amplifies) |
| M3    | ❌ wrong propagation speed by 0.5–1% |
| M4    | ❌ wrong dispersion relation |
| M5    | ❌ disagrees with anelastic (the developer agent likely sees this in the existing 07_cloudy_thermal_bubble) |
| M6    | ✅ pass (Δt small) |
| M7    | ❓ likely fail (moist + large-Δt) |
| M8    | ✅ pass (dry) |
| M9    | ❌ fail (the cleanest moist-only test) |
| M10   | ❌ fail (current data: day 4.1) |
| M11   | ❌ fail (current data: day 3.3) |
| M12   | smaller Δt → faster failure (current data confirms) |

After moisture-aware fix: M0 through M12 all pass, with M11 reaching
advective-CFL Δt and clean cyclogenesis.

## Implementation plan

### Phase 1 — instrument the existing substepper (no behaviour change)

In `src/CompressibleEquations/acoustic_substepping.jl`:

1. Add a sentinel `@assert maximum(qᵛ_in_state) == 0` at the top of
   `freeze_outer_step_state!`. Run M9 — confirm assertion fires (proving
   the substepper is currently silently using dry math on moist state).
   *If M9 fires this assertion, that alone is the bug.*

2. Compute `Rᵐ`, `cpᵐ` per cell from the current state and store as
   substepper-internal fields:
   - `outer_step_R_mixture :: CenterField`
   - `outer_step_cp_mixture :: CenterField`
   - `outer_step_kappa_mixture :: CenterField` (= Rᵐ/cpᵐ)
   - `outer_step_gamma_mixture :: CenterField` (= cpᵐ/cvᵐ)
   - `outer_step_virtual_theta :: CenterField` (= θ_li × (1 + 0.608 qᵛ − qˡ − qⁱ))
   - `outer_step_virtual_density :: CenterField`

3. Refresh these in `freeze_outer_step_state!` from the current
   model state.

### Phase 2 — switch the linearisation to use mixture quantities

In `_explicit_horizontal_step!`:

Replace `γRᵈ * Π⁰_x * ∂x_η` with `γᵐ_face × Rᵐ_face × Πᵐ_x × ∂x_η`, with
`γᵐ_face` and `Rᵐ_face` interpolated from cell-center values.

In `assemble_slow_vertical_momentum_tendency!`:

Replace `g * (ρ⁰[k] − ρᵣ[k])` with `g * (ρv⁰[k] − ρv_ref[k])` — virtual
density.

In the implicit-solve coefficients (`AcousticTridiag{Lower, Diagonal,
Upper}`):

Replace `γRᵈ` with `γᵐ_face × Rᵐ_face` and `θ⁰_face` with `θᵥ⁰_face`
(virtual θ).

In the η flux `_theta_face_x_flux`:

Use `θᵥ⁰` (virtual potential temperature) so the linearised η-flux
matches the moist EoS.

### Phase 3 — verify with the M-tier hierarchy

Run M0 first (regression of the dry rest test). Then M1, M2 in order.
Each must pass before moving to higher-tier. Document Δt-sweep regression
in M12 at the end.

### Phase 4 — adaptive-Δt regression

The existing `moist_bw_15day_adaptive.jl` should now reach an outer Δt
near the advective CFL (~150–190 s on the 1° lat-lon grid). Document
the achieved Δt envelope in the report.

## Notes on Breeze's specific timestepper that may add considerations beyond WRF

1. **Reference state**: Breeze's `ReferenceState` is dry. For moist-aware
   substepper the reference state's contributions to the slow vertical-
   momentum tendency (`−∂z(p⁰ − p_ref) − g·(ρ⁰ − ρ_ref)`) should be
   evaluated as the *virtual* analogues. Per the no-reference test
   (`no_reference_test.jl`), the reference state is currently not load-
   bearing — but if it's reactivated during the moist-aware refactor,
   it must be rebuilt as a *virtual hydrostatic* reference, not a dry
   one.

2. **`LiquidIcePotentialTemperatureFormulation`**: this prognostic uses
   `θ_li`, which is conserved through phase changes. In the substepper's
   linearisation, `θ⁰` should be `θᵥ⁰` (virtual θ), not θ_li⁰. The
   distinction matters when condensate (qˡ, qⁱ) is present.

3. **Microphysics-substepper coupling**: Breeze integrates microphysics
   in `Gⁿρθ_li`, `Gⁿρqᵛ`, etc. (the slow tendencies). The substepper
   doesn't see the microphysics directly. This is correct, but means
   the moisture-aware mixture quantities `Rᵐ`, `γᵐ`, `θᵥ` must be
   computed at **outer-step start** (frozen at U⁰), not refreshed per
   substep. Refreshing per substep risks the same FP-rounding feedback
   that broke the per-stage refresh of the linearisation basic state.

4. **WS-RK3 stage interaction**: the model state evolves between stages
   (microphysics + dynamics increments). The substepper's frozen
   `Rᵐ_outer_step` etc. become slightly inconsistent with `U^(k-1)`'s
   moist state by stages 2 and 3. This is the same per-stage drift we
   identified pre-fix; the fix kept linearisation at U⁰ and tolerated
   the drift. The moist-aware version should do the same — `Rᵐ`, `γᵐ`,
   `θᵥ` are frozen at U⁰_outer for the entire outer step.

5. **Precision of κᵐ**: `κᵐ = Rᵐ/cpᵐ` differs from `κᵈ = 0.2855` by
   ~ 0.01–0.02 in the tropics. The Exner function `Πᵐ = (p/pˢᵗ)^κᵐ`
   varies with z; storing `κᵐ` per cell (or computing it on the fly
   inside the kernel from frozen Rᵐ, cpᵐ) is cheaper than recomputing
   the moist Π each substep.

## Acceptance criteria for the fix

The substepper is "moist-aware enough" when:

1. **M0 still passes** (no regression on dry rest).
2. **M1, M2 pass** at machine ε (moist rest is bit-quiet).
3. **M3 passes** (moist sound speed is correct within 1%).
4. **M9, M10, M11 all pass** at Δt=20s for 15 sim days.
5. **M11 reaches advective-CFL Δt under adaptive stepping** (target
   Δt_max ≥ 100s, ideally ≥ 150s on the 1° grid).
6. **M12 has no Δt-direction reversal** — smaller Δt should *not* fail
   earlier in sim time.
7. The full-physics F1 case (`moist_bw_15day_adaptive.jl`) runs to 15
   days with cyclone deepening to <970 hPa by day 9 and max\|qcl\| < 5
   g/kg throughout.

## Out-of-scope (do not bundle into this fix)

- Replacing the reference state with a discretely-balanced moist version.
  *(Reference state is currently not load-bearing; do this only if M1
  fails after the moist-aware switch.)*
- Increasing `forward_weight` (ω). Empirically that masks symptoms by
  adding implicit dissipation, not by fixing the underlying linearisation
  error.
- Microphysics subcycling. Current `τ_relax = 200 s` makes microphysics
  non-stiff for any reasonable Δt; subcycling adds machinery without a
  needed stability benefit.

## Deliverables

When the fix lands, the following should appear in
`test/substepper_validation/long_runs/`:

| File | Purpose |
|---|---|
| `m1_moist_rest.jl` + diagnostics | M1 |
| `m2_moist_rest_qv_gradient.jl` | M2 |
| `m3_moist_acoustic_pulse.jl` | M3 |
| `m4_moist_igw.jl` | M4 |
| `m7_cartesian_moist_bw_analog.jl` | M7 |
| `m9_lat_lon_moist_no_microphysics.jl` | M9 |
| `dry_bw_14day.jl` (regression) | M8 / M0 |
| `moist_bw_15day.jl` (regression) | M11 |
| `moist_bw_15day_adaptive.jl` (regression) | M11 + adaptive |
| `moist_bw_dt_sweep.jl` (regression) | M12 |
| `cartesian_moist_v2.jl` (regression) | M6 |

Each ≤ 5 minutes wall on one GPU after the fix; full hierarchy < 1 hour.

## References

(Order reflects relevance to the immediate fix.)

- *WRF Technical Note* (Skamarock et al. 2008, 2019 update): operational
  description of how ARW splits the moist compressible system. The
  closest published recipe.
- *Park, S.-H., Skamarock, W. C., et al. (2013 / 2019 MPAS).
  "Evaluation of Global Atmospheric Solvers Using Extensions of the
  Jablonowski and Williamson Baroclinic Wave Test Case"*: MPAS' moist
  BCI validation — directly comparable to our M11.
- *Klemp, Skamarock & Ha 2018, MWR 146, 1911–1929.* HEVI / split-explicit
  acoustic damping. Includes the moist mixture-gas-constant treatment
  used in operational MPAS.
- *Wicker & Skamarock 2002, MWR 130, 2088–2097.* Original WS-RK3.
- *Klemp, Skamarock & Dudhia 2007, MWR 135, 2897–2913.* Conservative
  split-explicit; the closest formal derivation of the linearised
  acoustic system the substepper integrates.
- *Klemp & Wilhelmson 1978, JAS 35, 1070–1096.* Split-explicit foundations.

The user mentioned a 2015 reference; if it's a specific paper not above
(e.g. a Park et al. 2015 follow-up), the recipe is qualitatively
unchanged — virtual θ, mixture R and γ, virtual density. Track down the
exact citation when implementing.
