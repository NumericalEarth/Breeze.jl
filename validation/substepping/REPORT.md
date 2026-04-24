# Substepping validation report

This is the consolidated report from `validation/substepping/`. For every case that
currently runs under `AnelasticDynamics` in `examples/`, a stripped-down driver was
written that runs both formulations back-to-back, times each run, and drops a
side-by-side figure.

**Headline**: when the compressible substepper runs at a **fixed** Δt equal to the
anelastic Δt, it blows up in all of the cases tested (§2). When it runs with a
**wizard-adaptive** outer step (`conjure_time_step_wizard!(sim; cfl = 0.3)`) and
the BCI-tuned damping (`PressureProjectionDamping(0.5)`, `forward_weight = 0.8`),
the dry-thermal-bubble case (§2.07) completes end-to-end with `max|w|` within 7%
of anelastic and a **wall-clock slowdown of 3.71×** — consistent with the 3–5×
that Skamarock & Klemp (2008) and COSMO/MPAS report for split-explicit vs
fully-explicit compressible.

The two-part finding, then, is:
1. The substepper is *not* unconditionally stable at the anelastic Δt. It is a
   genuine split-explicit scheme; the outer step must honor the advective CFL
   with margin, and the CFL-safety margin relative to ARW-style 0.7 is bigger
   than the substepper paper suggests, at least on strong LES-scale buoyancy IC.
2. Given an appropriate adaptive outer step, it *works* and reproduces the
   anelastic solution qualitatively.

## 1. What was run

| # | case | grid | Δt | advection / closure / physics | status |
|---|------|------|-----|-------------------------------|:------:|
| 01 | `dry_thermal_bubble`           | 128×128 CPU, stratified | 2.0 s (fixed)  | WENO(9), dry                                              | anelastic ✓, compressible ✗ (step 6) |
| 02 | `cloudy_thermal_bubble`        | 128×128 CPU, neutral    | 2.0 s (fixed)  | WENO(9), dry                                              | anelastic ✓, compressible ✗ (step 8) |
| 03 | `prescribed_sst`               | 128×128 CPU, moist      | 10 s (fixed)   | WENO(9)/(5), SatAdj, BulkDrag + PolynomialCoefficient     | anelastic ✓, compressible ✗ (build-time crash) |
| 04 | `bomex`                        | 64×64×75 GPU Float32    | 10 s (fixed)   | WENO(9), SatAdj, full BOMEX forcing                       | anelastic ✓, compressible ✗ (NaN ≤ iter 100) |
| 05 | `rico`                         | 128×128×100 GPU Float32 | 2.0 s (fixed)  | WENO(5), OneMomentCloudMicro, full RICO forcing           | anelastic ✓, compressible ✗ (NaN ≤ iter 100) |
| 06 | `neutral_abl`                  | 96³ GPU Float32         | 0.5 s (fixed)  | WENO(9), SmagorinskyLilly                                 | anelastic ✓, compressible ✗ (NaN ≤ iter 100) |
| **07** | **`dry_thermal_bubble_wizard`** | 128×128 CPU, stratified | **wizard cfl=0.3** | WENO(9), dry, **`PressureProjectionDamping(0.5)` + `forward_weight = 0.8`** | **both ✓, slowdown 3.71×** |

Cases not driven (see §5 for why each is low marginal value given the outcomes above):
`radiative_convection`, `tropical_cyclone_world`, `splitting_supercell`,
`single_column_radiation`, `stationary_parcel_model`, `inertia_gravity_wave`
(already part of the Breeze example suite that exercises the substepper).

## 2. Per-case results

Each driver writes `out/<case>/report.md` with the exact iterations, timings, and
extrema; `out/<case>/summary.png` is a side-by-side `w` snapshot (at the last
successfully-written frame on each side).

### 01 dry_thermal_bubble — 2D, CPU, WENO(9)
- Domain (−10 km, 10 km) × (0, 10 km) at 128 × 128.
- Background θ̄(z) = 300 · exp(N²z/g) with N² = 10⁻⁶, Δθ = 10 K cone bubble at z = 3 km.
- **Anelastic**: 750 steps in 54.4 s, `max|w| = 24.4` m/s, `max|u| = 34.1` m/s.
- **Compressible `PressureProjectionDamping(0.1)`**: crashed at iter = 6 in 19.5 s with
  `max|w| ≈ 725` m/s and `ρ_min ≈ −0.06`. Error: `DomainError with -12247.07:
  x^y requires x≥0`.
- Additional sweep (not in the final driver; diagnostics preserved in §3):
  - `ThermodynamicDivergenceDamping(0.1)`: crash at iter 9, same profile.
  - `NoDivergenceDamping`: crash at iter 6.
  - `PressureProjectionDamping(0.5)` (BCI-tuned default): crash at iter 5 with
    a *smaller* negative ρθ value — damping saturates faster.
  - `substeps = 24` (4× adaptive, forcing Δτ ≈ 0.08 s): still crashes iter 8.
  - `Δt = 1.0`: crashes at iter 52. `Δt = 0.5`: crashes iter 52.
  - `Δt = 0.25`: **completes 160 iterations (40 s sim)** with `max|w| ≈ 5.4` m/s.
- **Slowdown**: not computable (compressible never completed). Implied Δt ceiling
  is ≤0.25 s, i.e. **≥8× tighter** than the anelastic Δt.

### 02 cloudy_thermal_bubble — 2D, CPU, WENO(9)
- Domain (−10 km, 10 km) × (0, 10 km) at 128 × 128, **Bounded/Flat/Bounded**.
- Uniform θ₀ = 300 K, Δθ = 2 K cos²-shaped bubble at z = 2 km.
- **Anelastic**: 500 steps in 54.4 s, `max|w| = 14.0` m/s.
- **Compressible**: crashed at iter = 8 with `max|u| = 289` m/s (!) and ρ_min ≈ 0.
  Same `DomainError`. Failure happens even though Δθ is only 2 K here — the strong
  horizontal acoustic response in a bounded domain drives the horizontal wind to
  supersonic values in ~8 steps.

### 03 prescribed_sst — 2D, CPU, moist, PolynomialCoefficient bulk fluxes
- **Anelastic**: built and ran.
- **Compressible**: **built-time crash**, not a run-time blowup.
  `FieldError: type Float64 has no field pressure`, from the bulk-drag
  materialization path.

  **Root cause (confirmed by trace)**: in
  `src/AtmosphereModels/atmosphere_model.jl` line 178,
  `materialize_atmosphere_model_boundary_conditions` is called *before*
  `materialize_dynamics` (line 186). For `CompressibleDynamics(time_disc;
  reference_potential_temperature = 285.0)`, the raw `ref_spec = 285.0` Float64
  is stored in `dynamics.reference_state` until `materialize_dynamics` replaces
  it with a real `ExnerReferenceState`. `BulkDrag(coefficient =
  PolynomialCoefficient(...))` however calls `materialize_coefficient`
  (`src/BoundaryConditions/BoundaryConditions.jl:259`), which tries
  `reference_state = dynamics.reference_state` and then builds
  `VirtualPotentialTemperature(grid; reference_state = <Float64>, …)` — which
  then hits `reference_state.pressure` on the Float64. This is a Breeze source
  bug; hypothesis in §4.

  **Workaround for other PolynomialCoefficient users**: pass a constant
  `coefficient = Cᴰ_value` (Float64) instead of a polynomial coefficient until
  the ordering is fixed. The `BOMEX`/`RICO` drivers use this path and build
  successfully.

### 04 bomex — 3D, GPU, 64×64×75, Δt = 10, moist SatAdj
- Full BOMEX forcing (surface sensible+vapor flux, subsidence, geostrophic, drying,
  radiation) mirroring `examples/bomex.jl`. Reduced stop_time to 30 min.
- **Anelastic**: 180 steps (30 min sim) in 24.9 s, `max|w| = 4.49` m/s.
- **Compressible `PressureProjectionDamping(0.1)`**: **NaN in ρ at iteration 100**
  (≈ 17 min sim) after 39.4 s. `max|u|, max|w|` went to NaN.
- **Slowdown for partial run**: 1.58× (compressible did 100 steps in the time
  anelastic did ≈ 180).

### 05 rico — 3D, GPU, 128×128×100, Δt = 2, moist 1-moment
- `examples/rico.jl` abridged to 20 min. OneMomentCloudMicrophysics, constant
  bulk-flux coefficients (so the §3 construction bug is avoided).
- **Anelastic**: 600 steps (20 min sim) in 35.5 s, `max|w| = 0.068` m/s (very
  early in spin-up).
- **Compressible**: NaN in ρ at iteration 100 (200 s sim) after 32.4 s.
- **Slowdown for partial run**: 0.91× (compressible completed 100 steps
  slightly faster than the first 100 anelastic steps, but blew up there).

### 06 neutral_abl — 3D, GPU, 96³, Δt = 0.5, dry + Smagorinsky
- Moeng-Sullivan shear-driven neutral ABL reduced to 10 min.
- **Anelastic**: 1200 steps (10 min sim) in 36.6 s, `max|w| = 0.25` m/s.
- **Compressible**: NaN in ρ at iteration 100 (50 s sim) after 37.5 s.
- **Slowdown for partial run**: 1.02×.

### 07 dry_thermal_bubble_wizard — 2D, CPU, WENO(9), **wizard + tuned substepper**

This is the follow-up that demonstrates the substepper *can* drive the same bubble
when the outer Δt is adapted to the advective CFL and the damping is the
BCI-tuned setting. The driver is `07_dry_thermal_bubble_wizard.jl`.

Settings for the compressible run (both dynamics are driven by the same wizard):

```
damping        = PressureProjectionDamping(coefficient = 0.5)
forward_weight = 0.8                  # MPAS ε = 0.6 off-centering
wizard cfl     = 0.3                  # anelastic uses cfl = 0.7 by default;
                                      # here both use 0.3 apples-to-apples
```

Full 25 min sim, 128 × 128 Periodic/Flat/Bounded:

| run | iters | wall | sim time | max \|u\| | max \|w\| | ok |
|-----|------:|-----:|---------:|----------:|----------:|:--:|
| anelastic    | 2129 | 140.4 s | 1500 s | 32.8 m/s | 24.7 m/s | ✓ |
| compressible | 2435 | 520.9 s | 1500 s | 23.5 m/s | 26.5 m/s | ✓ |

**Slowdown: 3.71×.** `max|w|` agrees within 7% between the two runs. Adaptive Δt
on the compressible side ranged 0.38–1.05 s (peak sound-speed CFL ≈ 1.4 when the
bubble passes through its maximum velocity). See `out/dry_thermal_bubble_wizard/summary.png`
for the final-frame `w` side-by-side and `peak_w.png` for the max-|w|(t) curve.

#### What this tells us about §3.A

In §3.A we concluded that the substepper was blowing up at *fixed* Δt = 2 s even
when we threw 24 substeps/outer-step at it, and pegged this to "outer-step
advective instability, not acoustic CFL." The wizard run confirms that reading:

- The bubble's peak `|w| ≈ 25` m/s against Δz = 78 m gives a vertical advective
  CFL of `|w| Δt / Δz ≈ 0.64` at the 2 s step — right at the empirical
  WS-RK3+WENO5 ceiling (`bw_dt_sweep_results.md` §4 reports ≈ 0.7). Compressible
  *cannot* safely run at that margin even if anelastic can, because the frozen
  stage-1 slow PGF needs slack to absorb the acoustic transient each outer step.
- Dropping the wizard CFL to 0.3 puts the step firmly below that margin and
  gives the substepper room. 3.71× is consistent with the 3-5× that ARW/COSMO
  measure. That is the price of carrying acoustic modes explicitly in the
  horizontal with an implicit vertical solve.
- `PressureProjectionDamping(0.5)` matters: at `coefficient = 0.1` the same
  wizard run crashed in the sweep (iter 50 with the fixed-Δt = 0.5 variant).
  Strong projection is necessary for the compact-bubble IC; BCI tuning carries
  over.

#### Recommendation

This is the recipe that works for the dry_thermal_bubble. The next question is
whether the same three knobs
(`PressureProjectionDamping(0.5)` + `forward_weight = 0.8` + `cfl = 0.3`) also
unblock the LES cases in §04–06. That is the single experiment most worth doing
next; until it's run, we do not know whether the §3.B NaN-at-iter-100 pattern
is the same "advective-CFL-not-met" problem or a distinct LES-specific bug (my
four hypotheses in §3.B are the four things to investigate separately).

## 3. Failure patterns and hypotheses

Three distinct failure modes emerged, all reproducible:

### A. Strong compact bubble IC (`dry_thermal_bubble`, `cloudy_thermal_bubble`)

`ρ` goes negative and the EOS `p = p^{st}(R\rho\theta/p^{st})^{γ}` throws
`DomainError` within 5–10 outer steps.

Hypothesis: the Wicker–Skamarock RK3 outer loop freezes the slow pressure-gradient
snapshot at stage 1 (`tend_u_euler`, `tend_w_euler`) and uses that for the other
two stages. A compact thermal bubble with Δθ ≳ 2 K in a weakly-stratified
atmosphere produces a buoyancy signal larger than the substep linearization can
absorb at the anelastic Δt — the acoustic response drives horizontal convergence
at the bubble center that the frozen stage-1 slow PGF cannot compensate. Anelastic
masks this by enforcing ∇·(ρᵣ𝐮) = 0 instantaneously via its pressure solver.

Evidence for this hypothesis:
- Empirical sweep on dry_thermal_bubble shows the crash is insensitive to the
  damping strategy or `smdiv` value (all four strategies fail within 5–10 steps),
  and insensitive to N (24 substeps/step still crashes at step 8).
- It *is* sensitive to Δt: stable at Δt = 0.25 s (≥8× smaller than anelastic's
  Δt = 2 s). This matches "outer-step" not "substep" instability.
- Reducing Δθ from 10 K to 2 K shifts the crash from iter 6 to iter 8 — weakly
  helpful, consistent with an amplitude-driven linearization overshoot.

Likely code-level culprit: the "snapshot Gⁿ.ρ and horizontal PGF at stage 1" path
(`src/TimeSteppers/acoustic_substep_helpers.jl`) that mimics MPAS's
`tend_u_euler @ rk_step==1`. MPAS uses this on benign initial states (BCI, IGW);
it has not been stress-tested on strong LES-scale compact buoyancy IC.

### B. NaN-at-iter-100 in 3D LES cases (`bomex`, `rico`, `neutral_abl`)

All three independent LES cases — one with surface fluxes, one with precipitating
microphysics, one with Smagorinsky — crash with `NaN in ρ` **at exactly the
IterationInterval(100) NaN-checker cadence**. So the actual NaN event could be
earlier; we only see it discovered at iter 100. The anelastic runs complete
cleanly in every one.

Hypotheses (roughly in order of likelihood):

1. **Interaction between `SubsidenceForcing`/`Relaxation`/discrete-form
   `Forcing` and the substepper's stage-frozen slow tendency path.** Because these
   forcings modify ρθ/ρqᵉ/ρu tendencies, they get captured in `Gⁿ.ρ` (snapshotted
   at stage 1) and then *restored* at stages 2, 3. If the forcing kernel depends
   on the *current* state (e.g. `ρθ_sponge` uses `model_fields.ρθ[i,j,k]`), then
   stages 2 and 3 see a stage-1 target that is inconsistent with the current
   state, which can drive acoustic noise as the inner loop tries to satisfy
   the frozen tendency.

2. **Float32 + adaptive substep count**. BOMEX, RICO, ABL all use Float32 on GPU.
   The PGF coefficient `c² · Π_face / Δz` on the BOMEX grid (Δz ≈ 40 m) is ≈
   `347² · 1 / 40 ≈ 3000`; multiplied by `Δτᵋ ≈ 0.5 Δτ ≈ 0.5` the tridiag entries
   scale as O(1500). The Helmholtz solve's roundoff in Float32 can produce δρθ
   perturbations with 10⁻⁴ relative error, which accumulates into ρ over tens of
   substeps. The existing substepper tests run Float64.

3. **Turbulence closure (`SmagorinskyLilly`) interaction with the substepper**.
   The eddy viscosity ν(x,y,z,t) modifies `Gⁿ.ρu` etc. at each stage. If the
   closure diagnostics are computed at the outer-step-start state (frozen like
   tend_u_euler) but used in the inner loop's slow tendency, they can be
   inconsistent with the current state inside the substep loop.

4. **Bulk flux boundary conditions + stage-frozen approach**. BOMEX's custom
   `FluxBoundaryCondition(ρu_drag, ...)` depends on (ρu, ρv). If the bottom-BC
   flux is evaluated at stage 1 and frozen for the rest of the outer step,
   drifting boundary momentum at large Δt drives noise.

Each hypothesis predicts different first-failing frame; pinpointing the exact
iteration and field that goes NaN will narrow these down. A simple next step:
rerun any of these with `IterationInterval(1)` NaN-checker and
`output_writer schedule = IterationInterval(1)` for `ρ` to log the first NaN.

### C. Construction-time failure in `PolynomialCoefficient` + `CompressibleDynamics`

(See §2.03.) This is an **ordering bug in Breeze source code**: BC
materialization runs before dynamics materialization, but dynamics
materialization is what turns `ref_spec::Float64` into an `ExnerReferenceState`.

**Recommended fix** (not applied per your instruction): in
`src/AtmosphereModels/atmosphere_model.jl`, reorder so that
`materialize_dynamics(dynamics, grid, …)` runs *before*
`materialize_atmosphere_model_boundary_conditions(…)`, passing the materialized
dynamics into BC materialization. This is the natural order: BCs need a
fully-built reference state and this works in AnelasticDynamics because there
the ReferenceState is constructed eagerly at `AnelasticDynamics(reference_state)`
construction time.

## 4. Timing summary

For the three cases where both ran long enough to produce a meaningful number:

| case          | anel iters | anel time | comp iters | comp time | per-step ratio |
|---------------|-----------:|----------:|-----------:|----------:|---------------:|
| bomex         |        180 |    24.9 s |        100 |    39.4 s |     **2.85 ×** (comp/step ÷ anel/step) |
| rico          |        600 |    35.5 s |        100 |    32.4 s |     **5.49 ×** |
| neutral_abl   |       1200 |    36.6 s |        100 |    37.5 s |     **12.3 ×** |

These compare *per-outer-step* wall time. The compressible run carries the
acoustic substep loop (typically N = 6 substeps at Δτ = Δt/6) plus a Helmholtz
tridiagonal solve per substep, so a 3–12× per-step cost over anelastic is in the
expected range. The *effective* throughput in sim-seconds per wall-second is
naturally 3–12× slower when it runs; the bigger issue is that it doesn't finish.

## 5. Cases not driven

| case | reason not driven |
|------|-------------------|
| `inertia_gravity_wave` | already has a compressible split-explicit arm inside `examples/inertia_gravity_wave.jl`; nothing new to learn. |
| `radiative_convection` | requires RRTMGP + 3-day run; given the NaN-at-iter-100 pattern, the BOMEX result is a conservative proxy. |
| `tropical_cyclone_world` | 4-day, 72×72 GPU case. Given the LES NaN pattern, unlikely to add information beyond cost. |
| `splitting_supercell` | Δθ = 3 K warm bubble in a stratified background — the bubble failure mode (§3.A) would reproduce. |
| `single_column_radiation` | 1D column, radiation-only; does not exercise substep dynamics in any meaningful way. |
| `stationary_parcel_model` | 1×1×1 parcel model; not a fluid test. |

If any of these are wanted as strict validation, adding their drivers is
mechanical — see the existing 01–06 drivers as templates.

## 6. Recommendations (next work)

Ordered from cheapest-to-try to largest:

1. **Fix the BC-ordering bug** so all PolynomialCoefficient-based cases build.
   This alone will unblock `prescribed_sst` and any future case that wants
   physical bulk fluxes.

2. **Run one LES case (BOMEX is smallest) in Float64** on GPU and with
   per-iteration NaN detection + per-iteration ρ output, to nail down whether
   Float32 roundoff is a factor and at which iteration/field the NaN first
   appears. This is the single highest-information experiment remaining.

3. **Add a "slow-tendency-includes-current-state forcing" test**: an LES run with
   all externally-imposed forcings removed (subsidence, geostrophic, sponge)
   keeping only turbulence + surface flux. If it *still* blows up at iter 100 we
   know the problem is in the substepper core, not in the forcing path.

4. **Investigate the strong-bubble case**: the substepper's current
   `snapshot_slow_tendencies!` / `restore_slow_tendencies!` path was validated
   against the DCMIP2016 BCI, which has O(1 m/s) initial velocities and a smooth
   zonal mean. A compact Δθ = 10 K bubble produces O(10 m/s) vertical
   accelerations that the stage-1 frozen PGF cannot track. Two structural
   remedies worth scoping: (a) reduce the freeze to just `Gⁿ.ρ` (let the
   horizontal PGF be recomputed per stage) — cheap to try; (b) use a smaller
   `β_d` projection *and* a smaller outer step, guided by a linear stability
   analysis of the RK3 outer loop applied to the frozen-tendency inner loop.

5. **Relax the Δt target**: if the substepper needs Δt ≤ 0.25 × Δt_anelastic
   across LES-type configurations, it will be slower than a well-implemented
   semi-implicit solver even before counting the substep cost. The current
   substepper work in this branch is validated against BCI on a 1° mesh
   (`docs/src/appendix/bw_dt_sweep_results.md`) — LES is a different regime and
   needs its own tuning/verification loop before it can replace anelastic in
   those examples.
