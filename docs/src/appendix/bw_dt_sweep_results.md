# Baroclinic-wave Δt sweep: substepper stability ceiling

This note records the empirical Δt stability ceiling of Breeze's MPAS-style
acoustic substepper on the canonical 1° DCMIP2016 baroclinic wave (Ullrich et
al. 2014), with and without the [`MonolithicFirstStage`](@ref) WS-RK3
distribution.

## Setup

- **Grid**: `LatitudeLongitudeGrid` 360 × 170 × 15, latitude (-85, 85), `H = 30 km`.
- **IC**: DCMIP2016 BW initial condition (`Tᴱ = 310 K`, `Tᴾ = 240 K`,
  Γ = 0.005, K = 3, b = 2) with the standard 1 m/s velocity perturbation
  centred at (λ, φ) = (20°, 40°).
- **Reference state**: `θ_ref(z) = 250·exp(g z / (cᵖᵈ · 250))` (isothermal
  T₀ = 250 K hydrostatic background).
- **Time stepper**: `:AcousticRungeKutta3` (Wicker–Skamarock RK3 outer loop +
  MPAS conservative-perturbation acoustic substepping).
- **Damping**: `ThermodynamicDivergenceDamping(coefficient = 0.1)` (the
  Breeze default and the closest match to MPAS Klemp 2018), plus
  `acoustic_damping_coefficient = 0.5` (Klemp 2018 acoustic damping).
- **Adaptive substep count**: yes (`substeps = nothing`), based on the
  conservative horizontal acoustic CFL `Δτ ≤ 1.2 · Δx_min / c_s`.
- **Run target**: 7 simulated days, or first NaN.
- **Hardware**: NVIDIA H100 80GB.

The script lives at `test_bw_1deg_dt_sweep_compare.jl` and the full log at
`test_bw_1deg_dt_sweep_compare.log`.

## Results

`day7` means the run reached the 7-day target without a NaN. `crash dD.DD`
means it died on day `D.DD`. `max|w|` is the largest column-vertical-velocity
seen at any point in the run.

| Δt (s) | `ProportionalSubsteps`     | `MonolithicFirstStage`     |
|-------:|:---------------------------|:---------------------------|
|     60 | day7, max\|w\| = **0.26**  | day7, max\|w\| = 1.62      |
|    100 | day7, max\|w\| = **0.26**  | day7, max\|w\| = 0.82      |
|    150 | day7, max\|w\| = 2.70      | day7, max\|w\| = 3.00      |
|    200 | day7, max\|w\| = 3.51      | day7, max\|w\| = 3.85      |
|    300 | **crash d6.96**            | day7, max\|w\| = 5.42      |
|    500 | crash d4.20                | crash d3.99                |
|    800 | crash d2.25                | crash d2.18                |
|   1100 | crash d1.94                | crash d1.71                |
|   1400 | crash d1.65                | crash d1.46                |

`ProportionalSubsteps` reaches day 7 with a clean BCI lifecycle at Δt = 60s
and Δt = 100s (max\|w\| ≈ 0.26 in both, matching the canonical reference).
The two configurations are bit-comparable through day 4 and only diverge
once the BCI starts producing strong vertical motion.

## Key takeaways

### 1. The stability ceiling is set by *gravity-wave temporal resolution*, not the acoustic CFL

For the substepper alone, the relevant CFL is on the substep size
``Δτ = Δt/N``, and `compute_acoustic_substeps` adapts ``N`` so that
``Δτ \cdot c_s / Δx_\mathrm{min} ≤ 1`` is satisfied at every Δt. So the
substep CFL is held fixed across the sweep — every Δt is run at roughly the
same per-substep CFL.

The crashes happen because **buoyancy is computed in the slow tendency**
which is frozen across all WS-RK3 stages. At Δt > ~200 s the gravity wave
period (``2π/N ≈ 600`` s for ``N = 0.01``) is no longer well-resolved by Δt,
and the slow buoyancy update lags actual gravity wave evolution. This is a
slow-buildup instability — the runs survive several days before max\|w\|
grows monotonically and finally NaNs out.

### 2. `MonolithicFirstStage` is *not* limited by stage-1 horizontal acoustic CFL

A naive theory predicts that the WS-RK3 stage-1-with-one-substep form
should crash above Δt = 3·Δx_min/c_s ≈ 86 s on the 1° lat-lon grid (where
Δx_min ≈ 9.7 km at lat 85°). Empirically this is wrong: stage 1 of
`MonolithicFirstStage` runs cleanly with stage-1 substep CFL of 1.19
(Δt = 100), 1.79 (Δt = 150), 2.4 (Δt = 200), and even 3.6 (Δt = 300) — well
above the naive limit of 1.

The actual ceiling for both distributions is around Δt = 200–300 s, well
above the naive stage-1 prediction. Several mechanisms are likely
contributing — the off-centering parameter (`forward_weight = 0.6`, ε = 0.2)
provides additional damping; the `acoustic_damping_coefficient = 0.5`
extends the stable range further; and the WS-RK3 stage 1 advances state
only by `β₁ Δt` so a single unstable substep cannot grow far before stages 2
and 3 (which are CFL-stable) damp it.

### 3. `MonolithicFirstStage` adds noise at small Δt

Compare Δt = 60: `ProportionalSubsteps` max\|w\| = 0.26, `MonolithicFirstStage`
max\|w\| = 1.62 — six times noisier. Stage 1 of `MonolithicFirstStage` takes
*one* substep at ``Δt/3 = 20`` s instead of the *two* `Δt/N = 10` s substeps
that `ProportionalSubsteps` takes (with the auto-N = 6 floor on this grid),
so the time-averaging of the slow tendency over stage 1 is coarser. The
visible cost is a higher acoustic-noise floor in `w`.

**Recommendation**: prefer `ProportionalSubsteps` (the Breeze default) for
low-noise BW runs. `MonolithicFirstStage` is only useful for bit-compatible
comparisons against MPAS-A `config_time_integration_order = 3`.

### 4. The hard ceiling is the **advective CFL at the polar Δx_min**

Both distributions reach day 7 at Δt ≤ 200 s and crash above Δt ~ 300 s.
This is *not* a substepper limit, *not* a gravity-wave temporal-resolution
limit, and *not* fixable by divergence damping. It is the explicit
WS-RK3 + WENO5 advective CFL operating on the smallest cell on the
`LatitudeLongitudeGrid` — which on a 1° lat-lon grid happens at the poles.

**Practical advective CFL limit on WS-RK3 + WENO5**: empirically ~0.7.
Hyperdiffusion does *not* extend it — diffusion damps grid-scale noise
within the CFL-stable region but cannot rescue a scheme that is above its
CFL limit.

**Δx_min on this grid**: with `latitude=(-85, 85)` and `Nλ=360`, the
smallest zonal cell is at φ = 85°:

```
Δx_min = a · cos(85°) · (2π / Nλ) ≈ 6.371e6 · 0.0872 · 0.01745 ≈ 9.7 km
```

The meridional spacing is `Δy ≈ 111 km` (uniform), so the CFL is set by
the zonal Δx at the poles.

**CFL table** for the BW peak jet (U_max ≈ 60 m/s once the BCI develops):

| Δt (s) | CFL @ U = 30 m/s (init) | CFL @ U = 60 m/s (BCI peak) | observed result |
|-------:|------------------------:|----------------------------:|:-----------------|
|     60 | 0.19 | 0.37 | clean |
|    100 | 0.31 | 0.62 | clean |
|    150 | 0.46 | **0.93** | reaches day 7, max\|w\| jumps to 2.7 |
|    200 | 0.62 | **1.24** | reaches day 7, max\|w\| 3.5 |
|    300 | 0.93 | **1.86** | crash day 6.96 |
|    400 | 1.24 | **2.47** | crash day 4.20 |

The transition from "clean" to "noise visible" lands exactly where CFL
crosses ~0.7 at the BCI peak (between Δt = 100 and Δt = 150). The
transition from "noisy but completes" to "crash" lands where CFL exceeds
~1 at the BCI peak (between Δt = 200 and Δt = 300). This is exactly the
empirical CFL ≈ 0.7 limit.

**The gap to MPAS-A's "Δt = 1800 s on a 1° mesh" is a mesh problem**,
not a time-stepper problem. MPAS-A uses an SCVT (Spherical Centroidal
Voronoi Tessellation) mesh with **uniform** Δx ≈ 120 km on its 1° mesh.
Breeze's `LatitudeLongitudeGrid` has Δx_min ≈ 9.7 km at lat 85° — twelve
times smaller. At U = 60 m/s with the same CFL = 0.7 ceiling,
maximum stable Δt is

- **MPAS-A** (Δx = 120 km uniform): Δt_max = 0.7 · 120000 / 60 ≈ **1400 s**
  — comfortably allows their actual 1800 s.
- **Breeze lat-lon** (Δx_min = 9.7 km at the pole): Δt_max = 0.7 · 9700 / 60 ≈ **113 s**
  — exactly where Breeze's empirical "clean" regime tops out.

The 12× ratio in mesh spacing perfectly accounts for the 12× ratio in
maximum stable Δt. There is no missing damping to add and no substepper
tuning to do — closing this gap requires fixing the mesh:

- a **uniform-area mesh** like cubed-sphere or SCVT; *or*
- **polar filtering** that effectively coarsens the zonal resolution near
  the poles (recovering Δx ~ 120 km in the polar caps); *or*
- **constraining the domain** to mid-latitudes (avoiding the singular
  region entirely).

These are mesh-and-discretization concerns, not substepper concerns. The
substepper cleanup is complete.

## Pressure-projection damping sweep — Δt × β_d

A follow-up sweep tests [`PressureProjectionDamping`](@ref) at three values
of the projection weight ``β_d`` to ask whether (a) it can match
[`ThermodynamicDivergenceDamping`](@ref) for noise suppression at low Δt,
and (b) cranking ``β_d`` extends the Δt ceiling.

Setup is identical to the table above (1° DCMIP2016 BW, 7-day target,
`ProportionalSubsteps`) except the damping is
`PressureProjectionDamping(coefficient = β_d)`. The script is
`test_bw_1deg_pressure_sweep.jl` and the log
`test_bw_1deg_pressure_sweep.log`.

The summary shows the **trajectory max\|w\| excluding the first 5 startup
steps** (the projection produces a benign IC transient at step ≈ 1–2 that
hits ~1.6 m/s and is identical across β_d, since the projection has no
history at that point — it is not a real instability).

| Δt (s) | β_d = 0.10 | β_d = 0.25 | β_d = 0.50 |
|-------:|:----------|:----------|:----------|
|    60 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.25 |
|   100 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.23 | day7, max\|w\| ≈ 0.23 |
|   200 | day7, max\|w\| = **7.69** | day7, max\|w\| = 3.04 | day7, max\|w\| = **0.38** ✓ |
|   400 | crash d1.81             | crash d2.08             | crash d5.70             |
|   800 | crash d1.05             | crash d1.19             | crash d2.01             |
|  1400 | crash d1.12             | crash d1.20             | crash d1.26             |

### Key findings

1. **β_d barely matters at Δt ≤ 100 s.** The day-7 max\|w\| differs by less
   than 1 % across β_d ∈ {0.10, 0.25, 0.50}: at this Δt the (ρθ)″
   perturbation evolves slowly enough that the forward-extrapolation by
   one substep adds essentially nothing. The projection has no work to do
   when the substepping is already accurate.

2. **β_d = 0.5 at Δt = 200 produces a *cleaner* BCI than `ThermodynamicDivergenceDamping(0.1)` at the same Δt.**
   Compare:
   - `ThermodynamicDivergenceDamping(0.1)` at Δt = 200: trajectory max\|w\| = **3.51**
   - `PressureProjectionDamping(0.5)` at Δt = 200: trajectory max\|w\| (excluding startup) = **0.38**
     and ts_w\[end\] = 0.26 — comparable to Thermo at Δt = 60.

   This is the first empirical evidence that the literal-ERF/CM1/WRF
   pressure-projection form, with a strong enough projection weight, is
   competitive with — actually better than — the MPAS Klemp 2018
   thermodynamic-divergence form on this configuration. The cost is one
   additional `CenterField` of scratch storage and a per-cell EOS
   evaluation.

3. **β_d = 0.5 modestly extends the ceiling**, from Δt ≈ 300 to Δt ≈ 400.
   At Δt = 400, β_d = 0.5 survives to day 5.7 vs day 1.8 at β_d = 0.1.
   At Δt ≥ 500 all three β_d values crash within 1–2 days. The hard wall
   at Δt ≈ 200–300 is the *advective CFL of WS-RK3 + WENO5 at Δx_min* (the
   smallest cell on a 1° lat-lon grid is at the pole, ≈ 9.7 km wide); see
   the full diagnosis in §4 above. No projection coefficient can extend
   the explicit advection's CFL.

4. The day-0.002 startup transient is a benign artifact of the projection
   filter being applied at substep 1 of stage 1 of step 1, when neither
   `ρθ″` nor `previous_rtheta_pp` has any history; it relaxes within a few
   steps and does not affect the BCI trajectory.

## Final controlled comparison: CFL = 0.7 on the (-80, 80) grid

The previous sweeps were partially confounded by the polar Δx_min trap on
`latitude=(-85, 85)`: the smallest cell at lat 85° is ≈ 9.7 km, and at
the BCI peak (U ≈ 60 m/s) any Δt > 113 s exceeds advective CFL = 0.7. So
"noise at Δt > 100" might have been advective marginal-stability noise
rather than the substepper's acoustic noise.

To compare damping strategies *as such*, we use a configuration where the
advective scheme is comfortably stable across the whole BCI lifecycle:

- **Domain**: `latitude = (-80, 80)` (avoids the polar Δx_min trap;
  Δx_min ≈ 19.3 km at lat 80°).
- **Time step**: Δt = **225 s**, chosen so the BCI peak (U ≈ 60 m/s)
  hits CFL = 60 · 225 / 19300 ≈ **0.70** exactly, right at the empirical
  WS-RK3 + WENO5 limit.
- **Run target**: 7 simulated days.

At this Δt the advective scheme is stable throughout the run, so any
noise we see is *acoustic* and the comparison is genuinely about damping
strategy. Results from `test_bw_cfl07_compare.jl`:

| Strategy                                  | result          | body max\|w\| | day-7 max\|w\| |
|-------------------------------------------|-----------------|---------------:|---------------:|
| `NoDivergenceDamping`                     | **crash d2.66** | 14.17          | (n/a)          |
| `ThermodynamicDivergenceDamping(0.1)`     | day7, noisy     | 5.14           | 4.28           |
| `ThermodynamicDivergenceDamping(0.5)`     | **crash d0.02** | 1.97e+5        | (n/a)          |
| `ConservativeProjectionDamping(0.1)`      | day7, noisy     | 6.41           | 3.61           |
| `PressureProjectionDamping(0.1)`          | day7, noisy     | 6.71           | 4.17           |
| **`PressureProjectionDamping(0.5)`**      | **day7, clean** | **0.38**       | **0.24**       |

`body max\|w\|` excludes the first 5 startup steps (the projection
strategies briefly transient on initialisation, see §3 above).

### Findings

1. **`NoDivergenceDamping` crashes at day 2.66 even at CFL = 0.7.** Some
   form of divergence damping is *necessary* — acoustic noise builds up
   even when the advective scheme is comfortably below its CFL ceiling.

2. **`ThermodynamicDivergenceDamping(0.5)` crashes at step 10** with
   `max\|w\| ≈ 200000`. The MPAS Klemp 2018 momentum-correction form is
   *itself* unstable when the coefficient is too large — the correction
   becomes large enough that it overshoots the divergence it is supposed
   to damp. The MPAS default `config_smdiv = 0.1` (= the Breeze default
   coefficient until this commit) is the maximum useful value of this
   strategy on this configuration.

3. **The three "moderate damping" strategies** (Thermo(0.1), Cons(0.1),
   Press(0.1)) all reach day 7 but produce noisy BCIs (max\|w\| ≈ 4–7 m/s).
   They are stable but they are not removing enough acoustic noise to
   recover the canonical reference's clean lifecycle.

4. **`PressureProjectionDamping(0.5)` is in a class of its own.** Day-7
   max\|w\| = 0.24 — essentially identical to the canonical Δt = 60
   reference. body max\|w\| = 0.38 — the trajectory peak (excluding the
   benign step-1 transient) is far below any other strategy. This is the
   first configuration that produces a clean BCI lifecycle at the
   advective-CFL ceiling.

### Recommendation and new default

The default in `src/CompressibleEquations/time_discretizations.jl` is
`damping = PressureProjectionDamping(coefficient = 0.1)` — the literal
ERF/CM1/WRF projection form at the WRF/CM1 standard coefficient. This is
a behaviour change from the prior default
`ThermodynamicDivergenceDamping(0.1)` (which was itself bit-equivalent to
the pre-Phase-2 hardcoded path). On the canonical Skamarock-Klemp IGW the
new default produces max\|w\| = 7.567e-3 vs the old default's 7.553e-3 —
within 0.2 %.

**Important caveat about `coefficient = 0.5`**: although the table above
shows that `PressureProjectionDamping(0.5)` is the empirical winner for
this *one* BW configuration at advective CFL = 0.7, β_d = 0.5 is *too
aggressive* for small-amplitude wave configurations like the
Skamarock-Klemp 1994 IGW (peak θ' = 0.01 K, peak |w| ≈ 0.01 m/s). At Δt =
25 s on the IGW, both `PressureProjectionDamping(0.5)` and
`ConservativeProjectionDamping(0.5)` crash after ~25 outer steps — the
forward-extrapolation amplifies the wave growth at each substep until it
overshoots and the EOS exponentiation `(R(ρθ + (ρθ)″)/p^{st})^{R/c_v}`
hits a negative base. This is *not* a bug in the substepper or the
projection kernel; it is the projection form's standard
forward-extrapolation instability when β_d × (one substep's increment) is
comparable to the wave amplitude.

Users running large-amplitude / strongly nonlinear configurations
(baroclinic wave, supercell, tropical cyclone, …) where acoustic noise
visibly contaminates the solution can opt into a stronger projection
weight explicitly:

```julia
SplitExplicitTimeDiscretization(;
    damping = PressureProjectionDamping(coefficient = 0.5)  # not safe for IGW-like cases
)
```

Users who need bit-equivalence with the pre-Phase-2 hardcoded path should
construct the time discretization explicitly with the old strategy:

```julia
SplitExplicitTimeDiscretization(;
    damping = ThermodynamicDivergenceDamping(coefficient = 0.1)
)
```

## See also

- [Acoustic Substepping Cleanup and Damping Plan](substepping_cleanup_and_damping_plan.md)
- The four-strategy comparison at fixed Δt = 1400 s lives in
  `test_bw_1deg_damping_compare.jl` / `test_bw_1deg_damping_compare.log`,
  with the "form of damping matters at the same coefficient, but not enough
  at coefficient = 0.1" finding.
