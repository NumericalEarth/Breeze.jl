# P3 Fortran-Julia Parity Audit

Comparison of the Fortran P3 v5.5.0 (`~/Aeolus/P3-microphysics/src/`) against the Julia
implementation in `src/Microphysics/PredictedParticleProperties/`.

Date: 2026-04-09 (updated 2026-04-09)
Branch: `glw/p3`

---

## Resolutions

**False positives** (verified correct after detailed analysis):

- **C3 (HM splintering rate)**: `splintering_rate = 3.5e8` already includes the kg→g
  conversion: `3.5e5 splinters/g × 1000 g/kg = 3.5e8 splinters/kg`. No fix needed.

- **M1 (PSD-integrated vs mean-mass immersion freezing)**: The monodisperse + PSD
  correction approach is analytically identical to the Fortran PSD-integrated formula.
  The correction factor `C(μ) = Γ(μ+7)Γ(μ+1)/Γ(μ+4)²` recovers the exact PSD integral.
  For number rate, `C_N = 1` is analytically correct. No fix needed.

- **M2 (splintering mass not subtracted from riming)**: For nCat = 1, the Fortran
  subtracts splintering from riming (`qccol -= dum2`) then adds it back as `qcmul` to
  the same category. The net effect cancels exactly. The Julia comment at
  `process_rates.jl:1318-1320` is correct. No fix needed.

**Fixed items**:
- **C4**: Documented dry PSD issue in melting integrands (`quadrature.jl`)
- **C5**: Implemented `ccn_activation_rate` in `process_rates.jl`
- **C6**: Implemented `rain_condensation_rate` in `rain_process_rates.jl`
- **M7**: Documented normalization factor K = 1 vs K = 2 in `integral_types.jl`
- **M8**: Fixed ice-rain collection integrands: removed spurious 100 μm threshold,
  added proper M6 Jacobian, documented as single-integral approximations
- **M9**: Added `RayleighReflectivity` type and integrand with `|K_w|² = 0.1892`
- **L8**: Fixed: homogeneous freezing Z now uses `μ_cloud` (from `diagnose_cloud_dsd`)
  instead of `zero(FT)`, matching Fortran `mu_i_new = mu_c(i,k)`
- **L9**: Verified: Fortran-generated lookup tables provide the correct double-integral
  `m6agg` values at runtime. The single-integral Wisner fallback is only used in unit
  tests without tables. Corrected misleading comments in `quadrature.jl`.
- **L10**: Fixed: removed spurious 100 μm threshold from `IceRainMassCollection` and
  `IceRainNumberCollection` integrands (applies only to cloud-riming, not ice-rain)

---

## Critical Inconsistencies (High Impact)

### C1. Semi-Analytic Coupled Saturation Adjustment -- Missing

**Fortran** (`microphy_p3.f90:3612-3743`): Couples ice deposition, cloud condensation, and
rain evaporation through a shared relaxation timescale `xx` with exponential decay
`1 - exp(-xx * dt)`. The forcing term `aaa` is driven by the vertical-motion cooling rate
`w = -cp/g * (T - T_old)/dt`, and accounts for the previous timestep's vapor `qv_old`.
This prevents overshooting during rapid phase changes and naturally captures the Bergeron
effect (ice growth at the expense of liquid).

**Julia** (`process_rates.jl:496-571`): Uses independent single-species
ventilation-enhanced deposition formulas:

```julia
dm_dt = 2pi * C_fv * (S_i - 1) / (Gamma_i * thermodynamic_factor)
```

No coupling between ice/liquid/rain phase changes. No exponential relaxation. No Bergeron
forcing term.

**Impact**: Julia can overshoot thermodynamically and does not capture the Bergeron effect.

### C2. Saturation-Adjustment Limiting -- Missing

**Fortran** (`microphy_p3.f90:3991-4058`): After computing all process rates, limits total
condensation and deposition to saturation-adjustment ceilings:

```fortran
qcon_satadj = (Qv_cld - dumqvs) / (1. + xxlv**2 * dumqvs / (cp*rv*T**2)) * i_dt * SCF
qdep_satadj = (qv_tmp - dumqvi) / (1. + xxls**2 * dumqvi / (cp*rv*t_tmp**2)) * i_dt * SCF
```

Then rescales individual rates proportionally if total exceeds the ceiling.

**Julia**: Only does sink-limiting (preventing negative mixing ratios via
`sink_limiting_factor`). No thermodynamic ceiling on source rates.

**Impact**: Condensation, deposition, and nucleation rates can overshoot what is
thermodynamically consistent.

### C3. Hallett-Mossop Splintering Rate -- 1000x Too Small

**Fortran** (`microphy_p3.f90:3551`):

```fortran
dum1 = 35.e+4 * qccol(iice) * dum * 1000.  ! 1000 converts kg to g
```

The `* 1000` converts kg to g because the rate coefficient (3.5e8 splinters per gram of
rime) is specified per gram.

**Julia** (`ice_nucleation_rates.jl:365-402`):

```julia
cloud_number = efficiency * c_splinter * cloud_riming_eff
```

with `c_splinter = 3.5e8`. Missing the `* 1000` kg-to-g conversion.

**Impact**: Julia's secondary ice production rate is **1000x too small**.

### C4. Melting Integrals Use Wrong PSD (Wet vs Dry)

**Fortran** (`create_p3_lookupTable_1.f90:1985-2006`): Melting ventilation integrals use the
**dry ice PSD** parameters `(n0d, mu_id, lamd)`:

```fortran
sum1 = sum1 + capm * fac1 * n0d * d1**(mu_id) * exp(-lamd*d1) * dd
```

**Julia** (`quadrature.jl:811-845`): Uses the **full mixed-phase PSD** via
`size_distribution(D, state)`, which returns `N0 * D^mu * exp(-lam*D)` from the wet (total)
PSD.

**Impact**: When liquid fraction Fl > 0, Julia overestimates melting by including liquid water
in the distribution shape. For Fl = 0, the two PSDs are identical.

### C5. CCN Activation -- Stubbed to Zero

**Fortran** (`microphy_p3.f90:3930-3965`): Computes cloud droplet activation from CCN
(`qcnuc`, `ncnuc`) with two-moment cloud, and 1-moment activation with prescribed `nccnst`.

**Julia** (`process_rates.jl:1086`): Stubbed to `zero(FT)`.

**Impact**: Missing cloud mass source from CCN activation.

### C6. Rain Condensation -- Missing

**Fortran** (`microphy_p3.f90:3680-3701`): `qrcon` (condensation onto rain drops) is computed
through the semi-analytic framework and can be positive in supersaturated conditions.

**Julia**: The `rain_condensation` field exists in `P3ProcessRates` but is always zero
(`process_rates.jl:1086`). Rain can only grow through autoconversion, accretion, and melting.

### C7. SCPF (Subgrid Cloud/Precipitation Fraction) -- Absent

**Fortran**: All process rates are modified by sub-grid cloud/precipitation fraction factors
(`SCF`, `iSCF`, `SPF`, `iSPF`, `SPF_clr`, `Qv_cld`, `Qv_clr`) throughout `p3_main`. These
partition the grid cell into cloudy and clear sub-fractions.

**Julia**: All computations use grid-mean values with no sub-grid fraction adjustment.

**Impact**: Systematic difference whenever cloud fraction < 1.

---

## Significant Inconsistencies (Medium Impact)

### M1. PSD-Integrated vs Mean-Mass Immersion Freezing

**Fortran** (`microphy_p3.f90:3436-3496`): Uses gamma-PSD-resolved integrals:

```fortran
Q_nuc = cons6 * cdist1 * gamma(7. + mu_c) * exp(aimm*(T0-T)) * (1/lamc)^6
N_nuc = cons5 * cdist1 * gamma(mu_c + 4.) * exp(aimm*(T0-T)) * (1/lamc)^3
```

**Julia** (`ice_nucleation_rates.jl:88-137`): Uses mean-mass approach with a post-hoc PSD
correction factor:

```julia
prob_per_s = bimm * V_drop * exp(aimm * dT)
Q_frz = q_cl_eff * psd_correction * prob_per_s
```

**Impact**: Quantitative differences in freezing rates due to different PSD treatment.

### M2. Splintering Mass Not Subtracted from Riming

**Fortran** (`microphy_p3.f90:3553-3559`):

```fortran
qccol(iice) = qccol(iice) - dum2  ! subtract splintering crystal mass from rime
```

Fortran subtracts the splintering crystal mass from rime mass transfer, ensuring mass
conservation.

**Julia** (`process_rates.jl:1054-1056`): Caps splintering mass to riming mass but does not
subtract it from riming. For nCat = 1 this is a no-op (same category), but for nCat > 1
it is a mass conservation issue.

### M3. Ice-Ice Collection Lookup Table -- Missing

**Fortran** (`microphy_p3.f90:6510-9093`): Full `proc_from_LUT_ii` and
`find_lookupTable_indices_2` for multi-category ice-ice collection.

**Julia** (`multi_ice_category.jl:155-163`): Explicitly marked
"PLACEHOLDER: dimensionally incorrect heuristic scaling".

### M4. `icecat_destination` -- Missing

**Fortran** (`microphy_p3.f90:10075-10172`): Routes nucleated/frozen ice to the correct
category based on closest mean diameter, with `deltaD_init` thresholds
(175--999 um depending on nCat).

**Julia**: No equivalent. Multi-category ice (nCat > 1) cannot correctly route new ice to
the appropriate category.

### M5. Rain Self-Process 2D Lookup Table -- Different Approach

**Fortran** (`microphy_p3.f90:10469-10512`): Piecewise non-uniform 2D grid indexed by
`(mu_r, lamr)` for rain self-collection, breakup, and evaporation ventilation integrals.

**Julia**: Generates rain 1D tables from quadrature with uniform spacing and a different
parameterization.

### M6. Saturation Vapor Pressure Formula Differs

**Fortran** (`microphy_p3.f90:9878-9947`): Flatau et al. (1992) polynomial fit with
Goff-Gratch fallback below 195.8 K (ice) or 202.0 K (liquid).

**Julia**: Uses Clausius-Clapeyron or Tetens from the Breeze thermodynamics module.

**Impact**: All saturation-dependent rates (condensation, evaporation, deposition,
sublimation, nucleation) differ slightly.

### M7. Sixth Moment Normalization Factors (1 vs 2) in Table Generation

**Fortran** (`create_p3_lookupTable_1.f90:2130-2134`):

```fortran
m6dep  = 1./mom3**2 * sum3 - 2.*mom6/mom3**3 * sum5   ! deposition: factor 2
m6sub  = 1./mom3**2 * sum3 -    mom6/mom3**3 * sum5   ! sublimation: factor 1
m6mlt1 = 1./mom3**2 * sum5 -    mom6/mom3**3 * sum7   ! melting: factor 1
```

The coefficient on the `mom6/mom3^3` correction term differs by process: factor 2 for
deposition/riming/shedding, factor 1 for sublimation/melting.

**Julia**: Stores identical raw integrands for deposition and sublimation
(`SixthMomentDeposition` = `SixthMomentSublimation`). The different factors must be applied
during tabulation or at runtime. If Julia ever regenerates tables natively, this distinction
must be handled correctly.

### M8. Ice-Rain Collection Single-Integral Integrands Are Wrong

**Fortran** (`create_p3_lookupTable_1.f90:1820-1830`): Double integral over both ice and rain
PSDs with gravitational collection kernel:

```fortran
(sqrt(area_ice) + sqrt(pi*0.25*d2**2))**2 * abs(V_ice - V_rain)
```

**Julia** (`quadrature.jl:1282-1296`): Single integral over ice PSD only:

```julia
return ifelse(D < FT(100e-6), zero(FT), V * A * m * Np)
```

Missing: rain PSD integration, rain cross-section, differential fall speed.

**Impact**: Currently only used in unit tests (runtime reads Fortran tables), but tests are
validating wrong physics.

### M9. Missing Rayleigh-Scattering Reflectivity (refl2)

**Fortran** (`create_p3_lookupTable_1.f90:1227-1268`): Computes `refl2` using 0.1892
prefactor, `rayleigh_soak_wetice` with Maxwell-Garnett mixing for partially liquid particles.

**Julia** (`quadrature.jl:931-938`): Only implements simple reflectivity
`(6/(pi*917))^2 * m^2 * Np` without the Rayleigh scattering or complex refractive index
mixing.

---

## Minor Inconsistencies (Low Impact)

### L1. Physical Constants

| Constant            | Fortran   | Julia      | Diff   |
|---------------------|-----------|------------|--------|
| g                   | 9.816     | 9.81       | 0.06%  |
| R_d                 | 287.15    | ~287.00    | 0.05%  |
| R_v                 | 461.51    | ~461.53    | 0.004% |
| epsilon = R_d / R_v | 0.622     | ~0.6218    | 0.03%  |
| rho_ice (small sph) | 900       | 917 (default in `IceSizeDistributionState`) | 1.9% |
| L_v at 273.15 K     | ~2.501e6  | 2.5e6      | 0.04%  |

Note: `IceMassPowerLaw.ice_density` correctly uses 900 (`lambda_solver.jl:63`). The 917
default only affects standalone `IceSizeDistributionState` construction.

### L2. Mixing Ratio vs Specific Humidity

**Fortran** `qv_sat` (`microphy_p3.f90:10756-10787`): Returns saturation **mixing ratio**
`w = epsilon * e / (p - e)`.

**Julia**: Returns **specific humidity** `q = epsilon * e / p` (approximately). The two
diverge at high moisture values (tropical near-surface conditions).

### L3. Splintering Diameter Threshold for nCat > 1

**Fortran** (`microphy_p3.f90:3516`): `Dmin_HM = 1000e-6` for nCat > 1.

**Julia** (`process_rate_parameters.jl:282`): Always `250e-6`.

**Impact**: For multi-category simulations, Julia applies HM splintering to much smaller
particles than Fortran intends.

### L4. `impose_max_Ni` -- Hard Clamp vs Relaxation Sink

**Fortran** (`microphy_p3.f90:10734`): Instant hard clamp
`nitot = min(nitot, max_Ni / rho)`.

**Julia** (`process_rates.jl:792`): Relaxation sink
`ni_lim = clamp_positive(ni - N_max/rho) / tau`. Softer; takes one timestep to drain.

### L5. Autoconversion -- Only KK2000 Implemented

**Fortran**: Supports `autoAccr_param` = 1 (SB2001), 2 (KK2000), 3 (Kogan2013).

**Julia**: Only KK2000.

### L6. `check_values` Runtime Validation -- Absent

**Fortran** (`microphy_p3.f90:10791-10958`): Validates all prognostics against bounds
(T: 173--323 K, Q: 0--60e-3, N: 0--1e20, Z: 0--10). Checks for NaN, sign consistency.

**Julia**: No runtime validation. By design for GPU compatibility, but invalid states
propagate silently.

### L7. Initial Saturation Adjustment (First Timestep) -- Missing

**Fortran** (`microphy_p3.f90:3973-3981`): At `it <= 1`, performs a saturation adjustment
to remove any initial supersaturation.

**Julia**: No special first-timestep handling.

### L8. Homogeneous Freezing Z Tendency Uses mu = 0 Instead of mu_c

**Fortran** (`microphy_p3.f90:4720`): For cloud homogeneous freezing Z:
`mu_i_new = mu_c(i,k)`.

**Julia** (`process_rates.jl:1412`): Simplified path uses `zero(FT)` for all Group 2
processes. The detailed path (`process_rates.jl:1585`) correctly uses mu_c.

### L9. Aggregation Z -- Single Integral vs Double Integral

**Fortran** (`create_p3_lookupTable_1.f90:1415-1470`): Aggregation Z is a double integral
with three-term normalization:

```fortran
m6agg = mom6/mom3**2 * sum1 + 1/mom3**2 * (sum2+sum3) - 2*mom6/mom3**3 * (sum4+sum5)
```

**Julia** (`quadrature.jl:1211-1216`): Single-integral Wisner (1972) approximation
`D^6 * V * A * Np^2`. A comment notes that `evaluate_quadrature` is specialized for the
full double integral during tabulation. Verify this specialization is complete.

### L10. Spurious 100 um Threshold in Ice-Rain Integrands

**Fortran** (`create_p3_lookupTable_1.f90:1740-1873`): The ice-rain **double integral** has
no 100 um minimum ice diameter threshold (that threshold applies only to cloud-riming).

**Julia** (`quadrature.jl:1282, 1291`): `IceRainMassCollection` and
`IceRainNumberCollection` apply `ifelse(D < 100e-6, zero(FT), ...)`. This would exclude
small ice particles that Fortran includes, if these integrands were used for table generation.

### L11. Reference Air Density

**Fortran**: `rho_ref = 60000 / (287.15 * 253.15) = 0.8254`.

**Julia** (`quadrature.jl:20`): `P3_REF_RHO = 60000 / (287.00 * 253.15) = 0.8258`.

Propagates into Best number and fall speeds. Practically negligible (~0.05%).

---

## Verified Matches (No Issues Found)

The following were confirmed consistent between Fortran and Julia:

- Mass-dimension relationships (4 regimes) and area-dimension with mass-based interpolation
- Fall speed (Mitchell & Heymsfield 2005): Best number, drag coefficients, velocity formula
- Rain fall speed: piecewise power law (same constants, same size thresholds)
- Liquid fraction blending of fall speed: `V = Fl * V_rain + (1-Fl) * V_ice`
- Capacitance (sphere/non-sphere regimes and mass-weighted interpolation)
- Ventilation factor constants (0.65/0.44 ice, 0.78/0.28 rain)
- Shedding threshold (D >= 9 mm) and riming threshold (D >= 100 um)
- Lambda limiter constants (Dm_max1 = 5 mm, Dm_max2 = 20 mm, Dm_min = 2 um)
- MH2005 drag parameters (delta_0 = 5.83, C_0 = 0.6, C_1, C_2)
- Slinn collection efficiency (all constants and formula structure)
- Deposited ice density (algebraic Julia vs iterative Fortran converge to same result)
- Diagnostic mu-lambda relationship (Field et al. 2007 and large-particle diagnostic)
- Shedding exponent (bb = 3)
- Reference conditions (T = 253.15 K, P = 60000 Pa)
- LUT multilinear interpolation (index clamping, fractional weights)
- `proc_from_LUT_main2mom`, `proc_from_LUT_main3mom`
- `proc_from_LUT_ir2mom`, `proc_from_LUT_ir3mom` (including reversed rain-lambda axis)
- `proc_from_LUT_3` (Table 3, mu_i / rho_i)
- `find_lookupTable_indices_1a` / `1b` / `1c` / `3a` formulas
- Rime density index piecewise transform
- `calc_bulkRhoRime` logic
- Wet growth formula structure (2pi / L_f placement on latent term only)
- Melting latent heat term sign convention
- Rain number from autoconversion (`cons3 = 1 / m_25um`)
- Table dimension ordering (different storage layout, correct coordinate mapping)
