# P3 Microphysics — Comprehensive Code Review

**Date:** 2026-03-12
**Branch:** `glw/p3`
**Reviewers:** NWP physics, Julia code quality, Oceananigans architecture (parallel agents)

---

## Summary

Three parallel review agents examined the full P3 implementation (~30 source files).
**Recommendation: NEEDS WORK** — 5 critical, 10 high, 13 medium, and 5 low issues identified.

The physics errors (C1, H1, H2, H3) directly affect process rates and will manifest in
validation discrepancies. The architecture issues (C4, H5) block GPU deployment.

---

## Provenance — Fortran-inherited vs Breeze-specific

Each issue is tagged **[Breeze]**, **[Fortran]**, or **[Hybrid]** to indicate whether
the root cause originates from our Julia implementation, was inherited from the Fortran
P3 reference code, or is a mixture of both.

**Breeze-specific (25):** C1, C4, C5, H1, H2, H3, H5, H6, H7, H8, H9, H10,
M1, M2, M3, M4, M5, M7, M8, M10, M11, M12, M13, L2, L3, L4, L5

**Fortran-inherited (3):** M6, M9, L1

**Hybrid (5):** C2, C3, H4

The most consequential physics bugs (C1, H2, H3) are all translation errors introduced
during porting — they do not exist in the Fortran reference. The Fortran-inherited issues
are hardcoded constants that are self-consistent within the Fortran codebase; the
inconsistency only arises because we partially modernized some paths (e.g. condensation
uses Breeze thermodynamics) while leaving others with Fortran literal values.

---

## CRITICAL Issues (5)

### C1 — Wrong saturation vapor pressure in rain evaporation **[Breeze]**

**File:** `rain_process_rates.jl:209`

The saturation vapor pressure `e_s` is computed as:

```julia
e_s = rho * max(q_vs, FT(1e-30)) * R_v * T
```

This uses the ideal gas relation `e = rho_v * R_v * T`, which gives the ACTUAL vapor
pressure from the current vapor density, not the SATURATION vapor pressure. The correct
inversion of `q_vs = epsilon * e_s / (P - (1 - epsilon) * e_s)` is:

```julia
e_s = P * q_vs / (epsilon + q_vs * (1 - epsilon))
```

This is already done correctly in the ice deposition path (`process_rates.jl:250-253`).
The current formula overstates `e_s` by a factor that depends on conditions, changing
the thermodynamic resistance term `B = R_v * T / (e_s * D_v)` and thus the evaporation rate.

**Impact:** Substantial error in rain evaporation rate.
**Provenance:** Translation error. Fortran P3 uses `polysvp1` (polynomial fit) for saturation
vapor pressure, not the ideal gas relation. The ice deposition path ports the correct inversion;
the rain path does not.
**Fix:** Use `e_s = P * q_vs / (epsilon + q_vs * (1 - epsilon))` as in the ice deposition path.

---

### C2 — Inconsistent latent heats across processes **[Hybrid]**

**Files:** `process_rates.jl:241`, `melting_rates.jl:57`, `rain_process_rates.jl:201`

Condensation uses the T-dependent `liquid_latent_heat(T, constants)` from Breeze
thermodynamics, but deposition, melting, and rain evaporation use hardcoded constants:

| Process | Value used | Source |
|---------|-----------|--------|
| Condensation | `liquid_latent_heat(T, constants)` | Breeze thermodynamics (T-dependent) |
| Deposition | `L_s = 2.835e6` | Hardcoded constant |
| Melting | `L_f = 3.34e5`, `L_v = 2.5e6` | Hardcoded constants |
| Rain evaporation | `L_v = 2.5e6` | Hardcoded constant |

This creates physically inconsistent energy budgets between processes.

**Impact:** Inconsistent thermodynamic closure.
**Provenance:** Fortran P3 also uses hardcoded constants — but consistently everywhere.
The inconsistency is ours: we upgraded the condensation path to use Breeze T-dependent
thermodynamics while leaving deposition, melting, and rain evaporation with Fortran literals.
**Fix:** Pass the `constants` argument through to all process rate functions and use
`vapor_gas_constant(constants)`, `liquid_latent_heat(T, constants)`, etc. consistently.

---

### C3 — Hardcoded density mismatch: 900 vs 917 **[Hybrid]**

**File:** `process_rates.jl:698-699`

```julia
rho_water = FT(1000)    # should read p3.water_density
rho_rim_hom = FT(900)   # differs from ProcessRateParameters.pure_ice_density = 917
```

The homogeneous freezing rime density 900 kg/m^3 is the Fortran P3 convention
(nucleated ice uses rho=900, not 917), but it is hardcoded as a magic number rather
than read from any parameter struct. Meanwhile `pure_ice_density = 917` in
`ProcessRateParameters` represents a different value for a related quantity.

**Impact:** Silent physical inconsistency between tendency function and scheme parameters.
**Provenance:** The value 900 is the correct Fortran convention (`mi0` uses `rho=900`).
The issue is that we hardcode it as a magic number rather than reading from a parameter
struct, and 917 in `ProcessRateParameters` represents a different physical quantity.
**Fix:** Add `homogeneous_ice_density` to `ProcessRateParameters` (default 900) and
replace `rho_water` with `p3.water_density`.

---

### C4 — Missing `@inline` on `tendency_rho_bf` **[Breeze]**

**File:** `process_rates.jl:692`

All nine other `tendency_*` functions are correctly marked `@inline`, but `tendency_rho_bf`
is not. This function is called from the KernelAbstractions kernel context in
`update_microphysical_auxiliaries!`.

**Impact:** Potential function call overhead on GPU; may prevent kernel compilation.
**Fix:** Add `@inline` annotation. One-line fix.

---

### C5 — Union-typed closure in table construction **[Breeze]**

**File:** `tabulation.jl:149-153`

```julia
if isnan(mu_override)
    closure = P3Closure(FT)
else
    closure = FixedShapeParameter(mu_override)
end
```

This produces `Union{P3Closure{FT}, FixedShapeParameter{FT}}`, causing dynamic dispatch
through `distribution_parameters -> solve_lambda -> shape_parameter`. The code
acknowledges the issue ("the union type from the if/else is intentional") but does not
resolve it.

**Impact:** Type instability propagates through the entire table construction path.
**Fix:** Parameterize `P3IntegralEvaluator` on the closure type instead of using
`NaN` as a sentinel. Accept `closure = P3Closure(FT)` as a keyword argument.

---

## HIGH Issues (10)

### H1 — Missing Schmidt number in ventilation factor **[Breeze]**

**Files:** `process_rates.jl:71-83`, `rain_process_rates.jl:269-271`

The analytical ventilation factor computes:

```
f_v = 0.65 + 0.44 * sqrt(V * D / nu)
```

The correct Hall & Pruppacher (1976) formula is:

```
f_v = 0.65 + 0.44 * Sc^(1/3) * Re^(1/2)
```

where `Sc = nu / D_v` (Schmidt number). For air at 273 K, `Sc^(1/3) ~ 0.88`, so
deposition rates are underestimated by ~12%. Rain evaporation (coefficients 0.78, 0.32)
is similarly affected but with smaller relative impact (~5-10%).

**Provenance:** Our analytical fallback formula. Fortran P3 sidesteps this entirely by
using PSD-integrated tabulated ventilation factors, never the analytical formula directly.
**Fix:** Compute `Sc = nu / D_v` from the already-available transport properties and
multiply the enhanced ventilation term by `Sc^(1/3)`.

---

### H2 — Rain density correction exponent 0.54 should be 0.5 **[Breeze]**

**File:** `terminal_velocities.jl:44`

The rain terminal velocity uses `(rho_0 / rho)^0.54`, but the correct exponent for
rain is 0.5 (Foote & du Toit 1969). The 0.54 is from Heymsfield et al. (2006) for
ice particles. Fortran P3 uses 0.54 for ice (`rhofaci`) and 0.5 for rain (`rhofacr`).

**Impact:** ~2-4% error in rain terminal velocity at mid-tropospheric densities,
propagating through evaporation, self-collection, and sedimentation.
**Provenance:** Translation error. Fortran P3 defines `rhofaci = (rho_0/rho)^0.54` for
ice and `rhofacr = (rho_0/rho)^0.5` for rain. We applied the ice exponent to rain.
**Fix:** Use exponent 0.5 for rain fall speed density correction.

---

### H3 — Spurious `-1/T` in condensation Clausius-Clapeyron derivative **[Breeze]**

**File:** `process_rates.jl:183`

```julia
dq_vs_dT = q_vs * (L / (R_v * T^2) - 1 / T)
```

The standard Clausius-Clapeyron derivative gives `dq_vs/dT = q_vs * L / (R_v * T^2)`.
The `-1/T` term does not appear in any standard derivation. At T=273K, this introduces
a ~5% error in the psychrometric correction factor Gamma_l, making condensation
slightly too fast.

**Provenance:** Our derivation error. Fortran P3 uses a saturation adjustment scheme,
not an explicit CC derivative for condensation. This formula is unique to our implementation.
**Fix:** Remove the `-1/T` term, or derive the exact correction from
`q_vs = epsilon * e_s / (P - (1-epsilon) * e_s)` which gives an additional
`(1 + (1-epsilon)/epsilon * q_vs)` multiplicative factor, not a `-1/T` additive term.

---

### H4 — Hardcoded R_v, R_d scattered across files **[Hybrid]**

**Files:** `process_rates.jl:239-240`, `rain_process_rates.jl:200`, `melting_rates.jl:59`,
`transport_properties.jl:54`

Multiple files independently define `R_v = 461.5` and `R_d = 287.0` instead of using
`vapor_gas_constant(constants)` and `dry_air_gas_constant(constants)`. The condensation
rate already uses the centralized constants, making the hardcoded values inconsistent.

**Provenance:** Fortran P3 hardcodes these everywhere — that's its convention. The issue
is that we have a centralized constants system and only use it in condensation, creating
inconsistency. The values themselves are correct.
**Fix:** Thread the `constants` argument through to all process rate functions.

---

### H5 — Missing `Adapt.adapt_structure` on container structs **[Breeze]**

**Files:** `ice_properties.jl`, `p3_scheme.jl`, `ice_fall_speed.jl`, `ice_deposition.jl`,
`ice_bulk_properties.jl`, `ice_collection.jl`, `ice_sixth_moment.jl`,
`ice_lambda_limiter.jl`, `ice_rain_collection.jl`, `rain_properties.jl`

Only the leaf types `TabulatedFunction3D` and `TabulatedFunction1D` define
`Adapt.adapt_structure`. All wrapping structs (`IceFallSpeed`, `IceDeposition`,
`IceProperties`, `RainProperties`, `PredictedParticlePropertiesMicrophysics`) do not.
GPU transfer of a tabulated P3 scheme will fail or silently use CPU arrays.

**Fix:** Add `Adapt.adapt_structure` to every struct in the containment chain, following
the pattern already established for `TabulatedFunction3D`.

---

### H6 — Triple redundant `air_transport_properties(T, P)` per grid point **[Breeze]**

**Files:** `process_rates.jl`, `melting_rates.jl`, `rain_process_rates.jl`

`air_transport_properties(T, P)` is called independently in deposition, melting, and
rain evaporation with identical `(T, P)` arguments. Each call computes `T^1.81` and
`T^1.5` transcendentals.

**Provenance:** Fortran computes transport properties once at the top of its column loop.
Our structural decomposition into separate functions recomputes them independently.
**Fix:** Compute transport properties once in `compute_p3_process_rates` and pass
the result through.

---

### H7 — Triple redundant mean mass and density correction in terminal velocity **[Breeze]**

**File:** `terminal_velocities.jl:163-333`

Mass-weighted, number-weighted, and reflectivity-weighted ice terminal velocities each
independently compute `m_bar = qi / ni` and `rho_correction = (rho_0 / rho)^0.54`.
All three are called sequentially from `update_microphysical_auxiliaries!`.

**Fix:** Pre-compute `m_bar` and `rho_correction` once and pass to all three functions,
or combine into a single function returning a NamedTuple.

---

### H8 — `::Any` fallback dispatch instead of concrete types **[Breeze]**

**Files:** `process_rates.jl`, `rain_process_rates.jl`, `terminal_velocities.jl`

Analytical fallback methods use `::Any` as the dispatch type for the "non-tabulated"
path. This catches erroneous types silently rather than throwing `MethodError`.

**Fix:** Replace `::Any` with `::Nothing` or the appropriate abstract integral subtype
from `integral_types.jl`.

---

### H9 — Wrong collection rate law in `inter_category_collection` **[Breeze]**

**File:** `multi_ice_category.jl:159-173`

Uses `sqrt(qi_1 * qi_2)` and `sqrt(ni_1 * ni_2)` (geometric mean, scales as N)
instead of the physically correct `n_1 * n_2` (product, scales as N^2). Also uses
`>` instead of `ifelse` (kernel-incompatible).

**Provenance:** Our implementation. Fortran multi-category P3 uses the correct product
form `n_1 * n_2 * kernel`. The geometric-mean approximation was introduced here.
**Fix:** Use product form `n_1 * n_2 * kernel` and replace `>` with `ifelse`.

---

### H10 — Fragile regime selection logic **[Breeze]**

**Files:** `lambda_solver.jl:497-509`, `quadrature.jl:328-339`

The nested `ifelse` chain for selecting ice mass regime coefficients uses a
priority-ordering that differs in direction between `lambda_solver.jl` (coarse-to-fine)
and `quadrature.jl` (fine-to-coarse). Both produce the same result, but the
inconsistency makes correctness verification difficult.

**Fix:** Unify the ordering convention, or extract into a shared
`select_regime_coefficients` function used by both sites.

---

## MEDIUM Issues (13)

### M1 — `cloud_riming_number_rate` units mismatch **[Breeze]**

**File:** `collection_rates.jl:169-174`

Returns `N_c / q_cl * riming_rate` which has units [1/m^3/s], but the struct documents
[1/kg/s]. Currently harmless because the value is stored but not consumed by any
tendency function.

### M2 — `safe_divide` threshold too small for Float32 **[Breeze]**

**File:** `process_rates.jl:39-43`

Uses `eps(FT)` as zero threshold. For Float32, `eps ~ 1.2e-7`, which is too loose for
physical quantities and may produce very large intermediate values.

### M3 — No rate-level vapor budget limiter **[Breeze]**

**File:** `process_rates.jl:407-426`

Combined deposition + condensation can exceed available vapor per timestep. Relies
entirely on the time integrator for positivity. Fortran P3 also lacks an explicit
rate-level limiter but uses a post-hoc saturation adjustment that partially compensates.

### M4 — Autoconversion threshold differs from KK2000 and Fortran **[Breeze]**

**File:** `rain_process_rates.jl:25`

Subtracts `autoconversion_threshold = 1e-4 kg/kg` before computing KK2000 rate.
Neither the original paper nor Fortran P3 use a threshold. Our addition.

### M5 — Rime density parameterization differs from Fortran **[Breeze]**

**File:** `collection_rates.jl:288-348`

Uses Cober & List (1993) with Stokes parameter K. Fortran P3 uses a simpler
temperature-dependent formula. Deliberate upgrade — better physics but breaks
Fortran parity.

### M6 — Prescribed Nc never depleted **[Fortran]**

**File:** `process_rates.jl:523-531`

Known limitation. Cloud droplet number is not reduced by riming, freezing, or
autoconversion. Artificially suppresses autoconversion when cloud fraction is low.
Fortran P3 prescribed-Nc mode has the same limitation.

### M7 — Duplicate collection efficiency parameter **[Breeze]**

**Files:** `ice_collection.jl:51` (default 0.1), `process_rate_parameters.jl:207` (default 0.5)

`IceCollection.ice_cloud_collection_efficiency` appears unused; the riming rate reads
from `ProcessRateParameters.cloud_ice_collection_efficiency` instead.

### M8 — Dead `CloudDropletProperties.autoconversion_threshold` **[Breeze]**

**File:** `cloud_droplet_properties.jl:13-17`

Field exists but `rain_autoconversion_rate` reads from `ProcessRateParameters` instead.
Could mislead users.

### M9 — Hardcoded breakup constants **[Fortran]**

**File:** `rain_process_rates.jl:139-140`

`D_th = 0.35e-3` and `k_br = 1000` from SB2006 are inline instead of in
`ProcessRateParameters`. Fortran P3 also hardcodes these values inline.

### M10 — Silent `kwargs...` in lambda solver **[Breeze]**

**File:** `lambda_solver.jl:789,849,1044,1129`

Accepted but never forwarded — misspelled keyword arguments are silently ignored.

### M11 — Symbol dispatch in `tabulate` **[Breeze]**

**File:** `tabulation.jl:971-1131`

`if property == :rain` chain instead of `Val`-dispatch. Prevents compile-time
specialization.

### M12 — Plain `julia` blocks in docstrings **[Breeze]**

**Files:** `p3_scheme.jl`, `tabulation.jl`, `multi_ice_category.jl`

Seven instances use `` ```julia `` instead of the required `` ```jldoctest ``.
Untested examples can silently rot.

### M13 — Possible sublimation double-counting in Z tendency **[Breeze]**

**File:** `process_rates.jl:812-814`

`z_dep * rates.deposition` (negative during sublimation) is added at line 806, then
`z_sub * abs(rates.deposition)` is added again at line 814. This may double-count
the sublimation contribution to reflectivity tendency. Needs cross-check with
Milbrandt et al. (2021) Table 1.

---

## LOW Issues (5)

| # | Issue | File | Provenance |
|---|-------|------|------------|
| L1 | KK2000 accretion exponent on `q_r` only (matches Fortran, differs from paper) | `rain_process_rates.jl:54-64` | **[Fortran]** |
| L2 | Fixed `mass_weight_factor = 1.9` in analytical ice fall speed fallback | `terminal_velocities.jl:237` | **[Breeze]** |
| L3 | O(n^2) aggregation double integral during table construction (CPU-only) | `tabulation.jl:221-256` | **[Breeze]** |
| L4 | `TabulatedFunction3D.func` stored but unused at runtime | `tabulation.jl:346-369` | **[Breeze]** |
| L5 | Wrong citation key for Heymsfield (2006) | `ice_fall_speed.jl:33` | **[Breeze]** |

---

## Mass Conservation Verification

Algebraic verification of total water conservation across all tendency functions:

```
tendency_qv + tendency_qcl + tendency_qr + tendency_qi + tendency_qwi = 0
```

All source/sink terms cancel pairwise. Splintering mass is correctly handled:
subtracted from rime mass (qf) but not from total ice (qi), representing a conversion
from rimed to unrimed ice. **Conservation is satisfied analytically.**

An automated conservation test should be added.

---

## Fix Status

### Fixed (this PR)

| # | Issue | Fix |
|---|-------|-----|
| **C1** | Wrong `e_s` in rain evaporation | Use `P * q_vs / (ε + q_vs(1-ε))` inversion |
| **C3** | Hardcoded density 900/1000 | Read from `prp.liquid_water_density`, `prp.pure_ice_density` |
| **C4** | Missing `@inline` on `tendency_rho_bf` | Already fixed in prior commit |
| **H1** | Missing Schmidt number `Sc^(1/3)` | Added to both ice deposition and rain evaporation ventilation |
| **H2** | Rain density correction exponent 0.54 | Changed to 0.5 for rain (Foote & du Toit 1969) |
| **H3** | Spurious `-1/T` in CC derivative | Removed; `dq_vs/dT = q_vs * L/(R_v T²)` |
| **H6** | Triple redundant transport properties | Compute once in `compute_p3_process_rates`, pass through |
| **H7** | Triple redundant mean mass / ρ-correction | Added `ice_terminal_velocities` combined function |
| **H8** | `::Any` fallback dispatch | Replaced with `::AbstractFallSpeedIntegral`, etc. |
| **H9** | Wrong inter-category collection (√ vs product) | Changed to `qi₁ * qi₂` and `ni₁ * ni₂` product form |
| **H10** | Fragile regime selection ordering | Unified ordering, cross-reference comments |
| **M1** | Units mismatch documentation | Fixed docstring |
| **M2** | `safe_divide` threshold for Float32 | Changed `eps(FT)` to `FT(1e-30)` |
| **M4** | Non-standard autoconversion threshold | Removed; KK2000 with no threshold |
| **M5** | Rime density parameterization | Added documentation noting deliberate Cober & List upgrade |
| **M7** | Duplicate collection efficiency | Added deprecation comment |
| **M8** | Dead autoconversion_threshold field | Added deprecation comment |
| **M10** | Silent `kwargs...` | Removed from 4 lambda solver signatures |
| **M13** | Z sublimation possible double-counting | Confirmed correct; added clarifying comment |
| **L2** | Magic `mass_weight_factor = 1.9` | Added derivation comment |
| **L5** | Wrong citation key | Fixed Heymsfield (2006) reference |

### Deferred (next PR)

| # | Issue | Reason |
|---|-------|--------|
| **C2** | Inconsistent latent heats (Hybrid) | Requires threading `constants` through all process rates |
| **C5** | Union-typed closure | Requires struct field addition to `P3IntegralEvaluator` |
| **H4** | Hardcoded R_v, R_d (Hybrid) | Same scope as C2 — unified constants pass-through |
| **H5** | Missing `Adapt.adapt_structure` | 10 struct definitions; separate GPU-readiness PR |
| **M3** | No rate-level vapor budget limiter | Requires careful physics analysis |
| **M11** | Symbol dispatch in `tabulate` | `Val`-dispatch refactor; low urgency |
| **M12** | Plain `julia` docstring blocks | 11 blocks need `jldoctest` conversion with output |
| **L3** | O(n²) aggregation table construction | CPU-only; low urgency |
| **L4** | Unused `TabulatedFunction3D.func` | Low impact |

### Not applicable (Fortran-inherited, no fix needed)

| # | Issue | Status |
|---|-------|--------|
| **M6** | Prescribed Nc never depleted | Shared limitation with Fortran prescribed-Nc mode |
| **M9** | Hardcoded breakup constants | Matches Fortran; documented |
| **L1** | KK2000 accretion exponent | Matches Fortran convention |
