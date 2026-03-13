# P3 Microphysics Implementation Review: Breeze.jl vs Fortran P3

**Date:** 2026-03-12
**Branch:** glw/p3 (commit a9017cb)
**Reference:** Fortran P3 v5.5.0 (reference_cld1d.f90)
**Reviewers:** 6 parallel NWP review agents (ice properties, nucleation/deposition, collection/melting, rain/fall speed, interface/bulk kernel, quadrature/sixth moment)

---

## Executive Summary

The Breeze P3 implementation is structurally sound with correct physics for the core
ice particle properties, size distribution, transport properties, and many process rates.
However, comparison against the Fortran P3 v5.5.0 reference reveals **1 critical**,
**9 high**, **14 medium**, and **9 low** severity issues. The most impactful are:
missing rain number evaporation, a spurious `qi >= qr` gate on rain-ice riming,
missing `Sc^(1/3)/sqrt(nu)` in tabulated deposition, and no aggregate sink limiting.

---

## CRITICAL Issues

### C1. No sum-of-sinks limiting in tendency formulation

**Files:** `process_rates.jl` (compute_p3_process_rates, tendency_* functions)

`compute_p3_process_rates` computes all process rates independently and returns them.
The tendency assembly functions sum rates without checking that the total mass removed
from any species exceeds what is available. For example, cloud liquid is consumed by
autoconversion + accretion + cloud_riming + cloud_freezing + cloud_homogeneous. If
these independently-computed rates each assume full cloud liquid is available, their
sum can exceed `qcl` in a single timestep.

**What Fortran does:** Applies rates *sequentially* within a single column timestep,
updating state after each process group. Before applying any tendencies, Fortran
computes total sources and sinks for each species. If sinks exceed sources + available
mass, ALL sink rates are rescaled by `ratio = sources/sinks`. This is done for rain,
each ice category, and vapor (lines 3590-3630).

**Impact:** Without in-step limiting, the explicit time integrator can produce negative
mixing ratios. The per-process `1/tau` limiters (e.g., `max_sublim = -qi/tau_dep`)
provide some protection, but the *sum* of all sink terms for a single species is not
limited.

**Fix:** Before assembling final tendencies, compute total sink rate for each species
and scale all competing sinks proportionally if their sum exceeds available mass / safety
timescale.

---

## HIGH Issues

### H1. Tabulated ice deposition missing `Sc^(1/3)/sqrt(nu)` correction

**Files:** `process_rates.jl:62-69`, `quadrature.jl:510-515`

The tabulated deposition dispatch path returns raw table values:
```julia
return vent(log_m, Ff, Fl) + vent_e(log_m, Ff, Fl)
```

The table stores the enhanced ventilation term as `0.44 * sqrt(V*D)` (quadrature.jl:514),
but the physics requires `0.44 * Sc^(1/3) * sqrt(V*D/nu)`. The comment at
quadrature.jl:511 explicitly states: "At runtime, the deposition rate multiplies by
Sc^(1/3)/sqrt(nu)." However, this runtime multiplication is NOT applied in the tabulated
dispatch path.

The analytical path correctly includes it (process_rates.jl:82-84):
```julia
Sc = nu / max(D_v, FT(1e-30))
Re_term = sqrt(V * D_mean / nu)
f_v = FT(0.65) + FT(0.44) * cbrt(Sc) * Re_term
```

The memory notes that "Deposition tables give 0.27x of analytical rate at mature ice"
-- this missing correction is very likely a root cause of that discrepancy.

**Impact:** Systematically underestimates tabulated deposition rate. The correction
factor `Sc^(1/3)/sqrt(nu)` ranges from ~0.5 to ~1.0 at typical mixed-phase cloud
conditions (T~250-273K, P~50000-80000Pa).

**Fix:** Apply `Sc^(1/3)/sqrt(nu)` to the `vent_e` table result at runtime, or bake
the reference-condition correction into the table and apply a T,P-dependent adjustment.

---

### H2. Missing rain number evaporation tendency

**Files:** `process_rates.jl:585-607`

The rain number tendency `tendency_rho_nr` includes self-collection, breakup,
autoconversion, melting, shedding, riming, and freezing -- but there is NO explicit
evaporation contribution to rain number.

**What Fortran does:** Rain evaporation removes rain number proportionally:
```fortran
nr_evap = nr * (evap_rate / qr)
```

**Impact:** Without an evaporation number tendency, rain evaporation reduces mass but
not number. Mean rain drop mass decreases monotonically during evaporation, producing
unrealistically many tiny drops. These tiny drops evaporate too quickly, creating a
positive feedback that significantly affects low-level rain profiles and surface
precipitation.

**Fix:** Add proportional number removal: `n_evap = (nr/qr) * evap_rate` to
`tendency_rho_nr`.

---

### H3. Spurious `qi >= qr` condition in rain_riming_rate

**Files:** `collection_rates.jl:217`

```julia
ice_dominant = qi_eff >= qr_eff
```

When `qr > qi`, `rain_riming_rate` returns zero. The comment references "Mizuno et al.
1990" for this condition.

**What Fortran does (lines 2725-2731):** NO `qi >= qr` condition whatsoever. Rain-ice
collection rate `qrcol` is computed whenever `qitot >= qsmall .and. qr >= qsmall .and.
T <= 273.15`. There is no branch based on which mixing ratio is larger.

**Impact:** In rain-rich environments (common during early precipitation development),
Breeze suppresses rain-ice collection entirely. This artificially reduces ice growth
by riming.

**Fix:** Remove the `ice_dominant` condition. Rain-ice collection should proceed
whenever both species are present and T <= T0.

---

### H4. Missing cloud-water collection above freezing (qcshd pathway)

**Files:** `collection_rates.jl:132`

`cloud_riming_rate` enforces `below_freezing = T < T0` and returns zero when `T >= T0`.

**What Fortran does (lines 2697-2710):** Two separate paths:
- `T <= 273.15`: `qccol` -- collected cloud water freezes onto ice (riming)
- `T > 273.15`: `qcshd` -- collected cloud water is shed as rain drops, using the
  same collection kernel `rhofaci * f1pr04 * qc * eci * rho * ni`. Also produces
  rain number `ncshdc = qcshd * 1.923e6` (1mm shed drops).

**Impact:** Above freezing, melting ice particles in Fortran still sweep up cloud
droplets and shed them as rain. Breeze misses this pathway entirely:
1. Cloud water persists too long above the melting level
2. Rain production from shedding of collected cloud water is missing
3. The rain number source from shedding (ncshdc) is missing

**Fix:** Create a `cloud_shedding_rate` function for `T > T0` using the same collection
kernel but routing mass to rain instead of ice.

---

### H5. Tendency cache freezes microphysics across RK stages

**Files:** `p3_interface.jl:212-245`

Process rates are computed and cached during `update_microphysical_auxiliaries!`, which
runs at the beginning of the timestep. The `grid_microphysical_tendency` overrides
(lines 418-446) read cached values as tendencies for the current sub-step.

**What Fortran does:** P3_MAIN is called as a complete subroutine that takes the current
state, computes all processes sequentially with in-step limiting, updates state
directly, and returns the modified state (Lie splitting).

**Impact:** For SSPRK3 with 3 stages, microphysics tendencies do not respond to state
changes from other stages. Effectively reduces microphysics to first-order time accuracy.
Likely acceptable for most applications since microphysics is typically the slowest
process, but it differs fundamentally from the Fortran approach.

---

### H6. Missing lambda_r bounds for rain

**Files:** `terminal_velocities.jl:56-58`, `rain_process_rates.jl:236-239`

**What Fortran does:**
```fortran
lamr = (pi*rhow*nr/qr)**0.333333
lamr = min(lamr, lamr_max)
lamr = max(lamr, lamr_min)
```

In the Breeze code, `lambda_r` is computed on-the-fly without any clamping. The only
protection is `max(m_mean, 1e-15)` and clamping `nr >= 1`. While the kin1d driver
applies Nr constraints externally, the core library functions do not.

**Impact:** Extreme `qr/Nr` ratios produce unphysically small or large `lambda_r`,
leading to out-of-range table lookups, unphysical fall speeds, or numerical instability.

**Fix:** Add `lambda_r` clamping matching Fortran bounds (typically `lamr_min ~ 125`,
`lamr_max ~ 50000`), or equivalently clamp mean rain drop mass.

---

### H7. Rain density correction exponent 0.5 vs Fortran 0.54

**Files:** `terminal_velocities.jl:43-44, 110-111`

```julia
rho_correction = (rho_0 / rho)^FT(0.5)
```

The Fortran P3 uses `rhofacr = (rhosur/rho)**0.54` -- the same 0.54 exponent as for
ice. The rain_quadrature.jl docstring (line 86) says "Apply (rho_0/rho)^0.54 at the
call site if needed," contradicting the actual 0.5 used.

**Impact:** At upper-tropospheric density (~0.4 kg/m^3), the difference is 3-5% in
rain terminal velocity, affecting sedimentation and all V-dependent processes.

**Fix:** Use 0.54 to match Fortran P3, or document why 0.5 is a deliberate choice.

---

### H8. Rain ventilation factor inconsistency between tabulated and mean-mass paths

**Files:** `rain_quadrature.jl:267-269` vs `rain_process_rates.jl:273-276`

Tabulated path: `f_v = 0.78 + 0.32 * sqrt(Re)` (no Sc^(1/3))
Mean-mass path: `f_v = 0.78 + 0.32 * Sc^(1/3) * sqrt(Re)` (includes Sc^(1/3))

At standard conditions Sc^(1/3) ~ 0.88, giving ~12% inconsistency in the
Reynolds-dependent term when switching between tabulated and non-tabulated modes.

**Fix:** Make the two paths consistent. Either use Fortran convention (Sc baked into
coefficients, ~0.308 instead of 0.32) everywhere, or include explicit Sc^(1/3) in both.

---

### H9. SixthMomentDeposition1 tabulated but never used

**Files:** `process_rates.jl:806-833`

In `_tabulated_z_tendency`, the Z-tendency for deposition uses only the constant-term
ventilation integral (`SixthMomentDeposition`). The enhanced-ventilation integral
(`sixth.deposition1`, corresponding to `SixthMomentDeposition1`) is tabulated in
`IceSixthMoment` but never looked up or added to `z_dep`.

**What Fortran does:** Computes Z-deposition using both m6dep and m6dep1:
```fortran
Z_dep = (m6dep + Sc^(1/3)/sqrt(nu) * m6dep1) * thermodynamic_factor
```

**Impact:** Three-moment ice simulations will underestimate radar reflectivity growth
during deposition, particularly for larger particles where the Reynolds-dependent
ventilation enhancement dominates.

**Fix:** Add `sixth.deposition1(log_m, Ff, Fl)` to the `z_dep` computation with
appropriate Schmidt number weighting.

---

## MEDIUM Issues

### M1. Missing inverse rain-ice collection (rain dominates)

**Files:** `collection_rates.jl:217`

When `qr > qi`, Fortran reverses collection: rain collects ice, transferring ice mass
to rain. Breeze shuts off rain-ice interaction entirely. This affects melting-layer
and warm-rain regions where rain dominates. Ice particles should get absorbed into rain
drops.

---

### M2. No post-step clipping in P3 interface

**Files:** `p3_interface.jl`

The P3 interface provides tendencies but includes no post-step clipping of negative
mixing ratios. Must be handled by the AtmosphereModel framework. If negative mixing
ratios propagate, they cause NaN (e.g., in `safe_divide`, `log10` for table lookup,
`cbrt` for diameter computation).

---

### M3. Missing melt-densification of rime toward 917 kg/m^3

**Files:** `process_rates.jl:681-722`

Breeze does proportional removal during melting (rime fraction and density unchanged).

Fortran applies density-preserving densification (lines 3841-3844):
```fortran
birim = qirim / (tmp1 + (917 - tmp1)*qimlt*dt/tmp2)
```

This makes rime density approach 917 kg/m^3 during melting (low-density parts melt
first, remaining ice becomes denser). Affects fall speed, cross-section, and particle
shape during the melting transition.

---

### M4. Missing wet growth calculation and shedding

**Files:** `collection_rates.jl`

Fortran computes wet growth rate `qwgrth` from heat balance (lines 2873-2894). When
collection rate exceeds wet growth rate, excess is shed as rain (nrshdr) and ice
densifies to maximum. Breeze relies on the Milbrandt et al. 2025 liquid fraction
framework instead (different mechanism). Without wet growth shedding, Breeze can
overestimate rime mass in convective environments.

---

### M5. Rime volume source density mismatch for rain riming

**Files:** `process_rates.jl:714`

Fortran uses `rho_rimeMax = 900` for rain rime density and Cober-List computed density
only for cloud droplet riming. Breeze uses the Cober-List computed density for both
cloud AND rain riming, and uses 1000 kg/m^3 (water density) for freezing processes
(Fortran uses 900).

---

### M6. Condensation/evaporation timescale asymmetry

**Files:** `process_rates.jl:178-195`

Condensation timescale is `Gamma * tau_cl` (Gamma > 1 from psychrometric correction),
but evaporation limit uses `tau_cl` without the Gamma correction:
```julia
S_cond_min = -max(0, qcl) / tau_cl
```

Evaporation is limited ~50-100% more aggressively than condensation, potentially
causing a wet bias over many timesteps.

---

### M7. Melting vapor term: fixed rho_vs vs pressure-dependent

**Files:** `melting_rates.jl:68-73`

Breeze uses `e_s0/(R_v*T0) = 611/(461.5*273.15) ~ 4.85e-3 kg/m^3` for saturation
vapor density at T0. Fortran uses `rho * qsat0(T0, P)` which varies with pressure.
At P=50000 Pa: Fortran gives ~5.38e-3 kg/m^3 (~10% higher). The sensible heat term
dominates melting, so net impact on total melting rate is ~5%.

---

### M8. Rain mu always 0 (exponential PSD)

**Files:** `rain_process_rates.jl:236`, `rain_quadrature.jl`

Fortran diagnoses `mu_r` from DSD parameters (Milbrandt & Yau 2005), typically 0-10.
Breeze always uses exponential (mu_r=0), giving more extreme drop sizes (more very
small and very large drops). This overestimates evaporation of small drops and
underestimates large drop fall speeds.

---

### M9. Hallett-Mossop uses Gaussian vs Fortran's linear

**Files:** `ice_nucleation_rates.jl:363`

Breeze uses Gaussian `exp(-((T-268.15)/2.5)^2)` peaked at -5C (symmetric).
Fortran uses linear ramp from 1.0 at -8C to 0.0 at -3C.
- At -5C: Breeze=1.0, Fortran=0.4 (2.5x more in Breeze)
- At -7C: Breeze=0.53, Fortran=0.8 (1.5x less in Breeze)

The Gaussian is arguably more physical (Hallett & Mossop 1974 peak near -5C) but
breaks Fortran parity.

---

### M10. Table resolution coarser than Fortran

**Files:** `tabulation.jl:685-701`

Default: 50 mass points, 4 rime fraction points, 4 liquid fraction points.
Fortran: ~300 mass points, ~10 rime fraction bins.

With only 4 points in rime fraction (0, 0.33, 0.67, 1.0), trilinear interpolation
has limited accuracy for intermediate rime fractions.

---

### M11. Water density inconsistency (997 vs 1000 kg/m^3)

**Files:** `rain_quadrature.jl:105` vs `process_rate_parameters.jl:170`

Rain table evaluators hardcode `rho_w = 997`, rest of code and Fortran use 1000.
0.3% inconsistency in mass-diameter relationship; rain tabulated velocities computed
with slightly different drop masses.

---

### M12. Sixth moment integrals completely unvalidated

**Files:** `validate_lookup_table.jl:138-205`

The validation script compares 14 ice integrals against Fortran reference but does NOT
validate any of the 9 sixth moment integrals (m6rime, m6dep, m6dep1, m6mlt1, m6mlt2,
m6agg, m6shd, m6sub, m6sub1). Combined with H9, the three-moment ice Z-tendency could
have systematic errors that go undetected.

---

### M13. Rain evaporation missing density correction on fall speed

**Files:** `rain_process_rates.jl:268-271`

Mean-mass evaporation path uses `V = ar * D^br` without `(rho_0/rho)^0.54` density
correction. At low air density (high altitude), actual fall speed is 1.5-2x higher,
increasing Re and ventilation by 20-40% for large drops. Underestimates evaporation
at altitude.

---

### M14. No sedimentation substepping

**Files:** `p3_interface.jl:474`

Fortran substeps sedimentation to satisfy CFL `V_t * dt_sub / dz < 1`. Breeze handles
sedimentation via advection velocities on fields, relying on the model timestep being
stable. For fine vertical grids (dz < 100m) or large timesteps, instability is possible.
Noted as remaining work in MEMORY.md.

---

## LOW Issues

### L1. safe_divide threshold is very small (1e-30)

**File:** `process_rates.jl:42`

For Float64, `1e-30` is well below machine precision, meaning division by numbers as
small as 1e-30 is allowed, producing values up to 1e+30. Using `eps(typeof(a))` would
be more robust.

---

### L2. R_d hardcoded as 287.0 in transport_properties

**File:** `transport_properties.jl:54`

Hardcoded rather than read from a constants module. Potential inconsistency with
Breeze.Thermodynamics R_d value. The Fortran P3 also uses 287.0.

---

### L3. Cooper nucleation relaxation (60s) vs Fortran instantaneous

**File:** `ice_nucleation_rates.jl:51`

Breeze uses `N_nuc = (N_eq - ni) / tau_nuc` with tau_nuc=60s. Fortran applies
instantaneously per timestep: `N_nuc = max(0, (N_ice - ni)) / dt`. The net effect
over a 10s timestep is nearly identical since ni reaches N_eq within dt regardless.

---

### L4. Nucleation 5% supersaturation threshold vs Fortran 0%

**File:** `process_rate_parameters.jl:226`

Breeze requires Si > 0.05 (5% ice supersaturation) for nucleation activation.
Fortran checks `qv > qv_sat_ice` (any supersaturation). Conservative but not
unreasonable; prevents nucleation from rounding noise.

---

### L5. Ice fall speed mass_weight_factor = 1.9 (correct is ~1.79)

**File:** `terminal_velocities.jl:236-244`

For exponential PSD with b_V=0.41: Gamma(4.41)/Gamma(4) = 10.72/6 ~ 1.787, not 1.9.
~6% overestimate of ice terminal velocity in the analytical (non-tabulated) path.
When tables are available, this fallback is not used.

---

### L6. RainProperties stores unused coefficients (4854, 1.0)

**File:** `rain_properties.jl`

`RainProperties` has `fall_speed_coefficient = 4854` and `fall_speed_exponent = 1`,
which are NOT the Fortran P3 rain parameters (ar=842, br=0.8). The actual code reads
from `prp.rain_fall_speed_coefficient = 842`. Confusing and could be a latent bug if
referenced.

---

### L7. Table range [-15, -5] narrower than Fortran [-17.3, -5.3]

**File:** `tabulation.jl:685-701`

Julia table spans mass 10^-15 to 10^-5 kg. Fortran spans ~5e-18 to 4.7e-6 kg. Very
small freshly-nucleated particles below 10^-15 kg will clamp to boundary values.

---

### L8. No cache invalidation mechanism

**File:** `p3_interface.jl:418-446`

If `update_microphysical_auxiliaries!` is not called before `grid_microphysical_tendency`,
stale values will be read. Oceananigans framework guarantees update order, so this is
a defensive coding concern.

---

### L9. Rime density Cober-List variant (deliberate upgrade)

**File:** `collection_rates.jl:296-339`

Breeze uses Cober & List (1993) with Stokes impact parameter. Fortran uses a different
Cober & List variant with temperature dependence via `Ri = -0.5e6 * D_c * V * /Tc`.
Documented as deliberate physics upgrade.

---

## Confirmed CORRECT

The following aspects were verified to match the Fortran P3 implementation:

| Item | Description |
|------|-------------|
| Cooper formula | `N = 5 * exp(0.304*(T0-T))`, c_nuc=5.0/m^3 |
| Barklie-Gokhale parameters | aimm=0.65, bimm=2.0 (NOT Bigg 1953) |
| Nucleated ice mass | mi0 = 4pi/3 * 900 * (1e-6)^3 (rho=900, NOT 917) |
| Capacitance convention | capm=D (cap=1), 2pi prefactor (Fortran convention) |
| Transport properties | D_v, mu, K_a, nu all match Fortran v5.5.0 exactly |
| Collection efficiencies | eci=0.5, eri=1.0 |
| Eii temperature ramp | 0.001 at T<253.15K, linear to 0.3 at T>=273.15K |
| Eii_fact limiter | 1.0 for Fr<0.6, ramp to 0 at Fr=0.9 |
| Self-collection factor | 1/2 for aggregation (upper triangle) |
| Melting number ratio | nmltratio = 1.0 |
| Total water conservation | Analytic tendency sum = 0 (verified algebraically) |
| Thermodynamic resistance | Mason (1971) A+B terms dimensionally correct |
| Saturation vapor pressure | Correct inversion from q_vs |
| Analytical ventilation | Sc^(1/3) included for both ice and rain |
| Homogeneous freezing | Correct with mass-number consistency cap |
| Rain self-collection | k_rr = 5.78, matches Fortran |
| Rain breakup | Three-piece Seifert-Beheng (2006) Eq. 13, correct structure |
| Autoconversion | KK2000 formula, correct rescaling to within <1% |

---

## Priority Fix Order

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | H2: Missing rain number evaporation | Small (1 line) | High - PSD degradation |
| 2 | H3: Remove qi>=qr condition | Small (1 line) | High - suppresses riming |
| 3 | C1: Sum-of-sinks limiting | Medium | High - production stability |
| 4 | H1: Tabulated deposition Sc/nu | Small | High - explains 0.27x discrepancy |
| 5 | H4: Above-freezing cloud collection | Medium | Medium - missing rain source |
| 6 | H6: Lambda_r bounds | Small | Medium - numerical safety |
| 7 | H7: Density correction exponent | Small (1 char) | Medium - 3-5% velocity |
| 8 | H8: Rain ventilation consistency | Small | Medium - 12% inconsistency |
| 9 | M3: Melt-densification | Medium | Medium - melting layer fidelity |
| 10 | M4: Wet growth/shedding | Large | Medium - convective riming |

---

## Architectural Differences (Not Bugs)

These are fundamental design differences between Breeze and Fortran P3:

1. **Method of lines vs operator splitting:** Breeze computes tendencies for the RK
   integrator. Fortran applies processes sequentially within a single call. Both are
   valid approaches with different stability/accuracy tradeoffs.

2. **Prescribed vs prognostic Nc:** Breeze prescribes cloud droplet number. Fortran
   carries it as prognostic. The homogeneous freezing N_hom cap compensates.

3. **Relaxation condensation vs prognostic supersaturation:** Breeze uses
   relaxation-to-saturation. Fortran uses prognostic `ssat`. Both are well-established
   approaches (Tao et al. 1989, Grabowski 2006).

4. **Predicted liquid fraction vs wet growth:** Breeze uses Milbrandt et al. (2025)
   liquid fraction framework. Fortran uses classical wet growth/shedding. These are
   complementary physical frameworks.
