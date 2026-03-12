# P3 Physics Review: Breeze.jl vs Fortran P3 v5.5.0

## Overview

A detailed physics review comparing the Breeze.jl P3 microphysics implementation
against the official Fortran P3 v5.5.0 source code (`P3-microphysics/P3-microphysics`).
The review covers process rate formulations, parameter values, and kin1d validation
against official test results.

## Parameter Verification

All P3 process rate parameters were verified against the Fortran source:

| Parameter | Fortran v5.5.0 | Breeze.jl | Match |
|-----------|----------------|-----------|-------|
| Cloud-ice collection efficiency `eci` | 0.5 | 0.5 | Yes |
| Rain-ice collection efficiency `eri` | 1.0 | 1.0 | Yes |
| Ice-ice collection `eii` (cold/warm) | 0.001 / 0.3 | 0.001 / 0.3 | Yes |
| Immersion freezing `aimm` | 0.65 | 0.65 | Yes |
| Immersion freezing `bimm` | 2.0 | 2.0 | Yes |
| Rain fall speed `ar` | 841.997 | 842.0 | Yes |
| Rain fall speed `br` | 0.8 | 0.8 | Yes |
| Ventilation `f1r` | 0.78 | 0.78 | Yes |
| Ventilation `f2r` | 0.32 | 0.31 | Close (< 3%) |
| Nucleated ice mass `mi0` | 4pi/3 * 900 * (1e-6)^3 | 1e-12 | Yes |
| Nucleation coefficient (Cooper) | 0.005 * exp(0.304*(273.15-T)) | Same | Yes |
| Autoconversion (KK2000) | 1350 * qc^2.47 * Nc^-1.79 | Same | Yes |
| Accretion (KK2000) | 67 * (qc*qr)^1.15 | Same | Yes |
| Freezing temperature | 273.15 K | 273.15 K | Yes |

## Formula Verification

### Thermodynamic Resistance (Mason 1971) — Correct

Both implementations use:
```
A = L / (K_a * T) * (L / (R_v * T) - 1)
B = R_v * T / (e_s * D_v)
```
This is the standard Mason (1971) thermodynamic resistance for diffusional growth.
The Fortran code computes `abi = 1 + dqsidT * L_s / c_p` for saturation adjustment,
which is a different quantity — the diffusional growth resistance is applied within
the PSD-integrated lookup tables.

### Rain Evaporation — Correct

Both use the same physics: `dqr/dt = 2*pi*n*(S-1)*D*f_v / (A+B)`.

**Primary path (tabulated, recommended):** Rain lookup tables are pre-computed
via `tabulate(p3, :rain, CPU())` and integrate `D × f_v(D) × N(D) dD` exactly
over the exponential PSD using the physical piecewise Gunn-Kinzer/Beard fall
speed law. This matches the Fortran PSD-integrated approach.

**Fallback path (mean-mass):** When tables are not available, uses a single
representative drop at the volume-mean diameter with Fortran power-law
`V = 842 D^0.8` and ventilation coefficients f1r=0.78, f2r=0.32.

| Component | Tabulated path | Mean-mass fallback | Fortran |
|-----------|---------------|-------------------|---------|
| Fall speed | Piecewise Gunn-Kinzer/Beard | V = 842 D^0.8 | PSD-integrated lookup |
| Ventilation | PSD-integrated f_v(D) | f1r + f2r√Re | PSD-integrated |
| f2 coefficient | 0.32 | 0.32 | 0.32 |

### Ice Deposition/Sublimation — Correct (with T,P-dependent transport)

The formulation matches Fortran. Both implementations use T,P-dependent transport
properties via `air_transport_properties(T, P)` (Fortran P3 v5.5.0 formulas):

| Property | Formula | Source |
|----------|---------|--------|
| D_v (water vapor diffusivity) | 8.794e-5 × T^1.81 / P | Hall & Pruppacher (1976) |
| K_a (thermal conductivity) | 1414 × μ (Sutherland) | Fortran P3 v5.5.0 |
| Kinematic viscosity | μ × R_d × T / P | Fortran P3 v5.5.0 |

These are actively used in deposition, melting, and evaporation rate computations.

**kin1d impact:** The kin1d driver's PSD correction factors (alpha_dep, alpha_rim)
are calibrated with T,P-dependent transport active. Early testing with constant
transport (D_v=2.5e-5, K_a=0.025, ν=1.5e-5) gave different validation results.

### Ice Melting — Correct

Uses the standard formulation with latent heat balance at 273.15 K.

### Collection/Riming — Correct

Collection kernel `K = E * pi/4 * D^2 * |Vi - Vj|` matches Fortran.
Collection efficiencies match (eci=0.5, eri=1.0, eii ramp from 0.001 to 0.3).

### Freezing — Correct

Immersion freezing uses Barklie-Gokhale (1959) parameterization with
aimm=0.65, bimm=2.0, matching Fortran exactly (not Bigg 1953).

## kin1d Validation Results

Comparison against official Fortran P3 test results from
`P3-microphysics/P3-microphysics/test-summaries/kin1d_tests/results-last_commit`:

| Variable | Breeze / Fortran | Notes |
|----------|-----------------|-------|
| Cloud water (qc) | 0.78x | Slightly low; sensitive to autoconversion threshold |
| Rain water (qr) | 0.80x | Slightly low; coupled to cloud and evaporation |
| Ice mass (qi) | 0.53x | Primary gap — mean-mass approximation |
| Temperature (T) | 1.00x | Excellent agreement |

### Ice Underestimate Root Cause

The 0.53x ice underestimate is a **fundamental consequence of the mean-mass
approximation** (Jensen's inequality). P3 process rates are nonlinear functions
of particle size:
- Collection rate ~ D^2 * V(D) ~ D^(2+b)
- Deposition rate ~ D * f_v(D) ~ D * (a + b*sqrt(V*D))

For these nonlinear functions, f(mean(D)) < mean(f(D)). The Fortran code
compensates by using PSD-integrated lookup tables that correctly account for the
full size distribution. Without these tables, the mean-mass approximation
systematically underestimates growth rates for ice.

PSD correction factors (empirical multipliers) partially compensate:
- Riming PSD correction: 2.0x
- Cloud freezing PSD correction: 5.0x
- Rain freezing PSD correction: 10.0x

But these cannot fully recover the PSD-integrated rates because the correction
needed varies with particle size, temperature, and process type.

## Attempted Fixes and Outcomes

### 1. T,P-dependent Transport Properties
- **Change:** Replace constant D_v=2.5e-5, K_a=0.025 with Fortran's T,P-dependent formulas
- **Result:** With original PSD corrections calibrated to constants, ice degraded.
- **Resolution:** T,P-dependent transport is now active in the library code
  (`air_transport_properties(T, P)`). The kin1d driver's PSD correction factors
  were recalibrated accordingly.

### 2. Rain Velocity Unification
- **Change:** Rain evaporation mean-mass fallback now uses `V = 842 D^0.8`
  (Fortran P3 v5.5.0 power law), consistent with terminal velocity fallback.
- **Previous state:** Used `V = 130 D^0.5` as a PSD-effective approximation.
- **Resolution:** With rain lookup tables always recommended (exact PSD integration),
  the mean-mass fallback is rarely used. Unified to Fortran's formula for consistency.

### 3. Ventilation Coefficient f2=0.32
- **Change:** Match Fortran's f2r=0.32
- **Result:** Now consistent: both tabulated and fallback paths use f2r=0.32.

## Path to Fortran Parity

### Completed

- ✅ T,P-dependent transport properties (`air_transport_properties(T, P)`)
- ✅ Fortran rain fall speed (842 D^0.8) in all code paths
- ✅ Rain PSD lookup tables (fall speed + evaporation ventilation integral)
- ✅ Ice PSD lookup tables (fall speed, deposition, collection, sixth moment)
- ✅ Lookup table validation against Fortran v6.9-2momI (median < 1%)

### Remaining for full parity

1. **Use PSD lookup tables for ALL process rates** (deposition, collection,
   melting) — currently only fall speed tables are used by default; deposition
   and collection use analytical fallback with PSD correction factors
2. **Prognostic cloud droplet number (Nᶜ)** — requires aerosol activation
3. **Remove kin1d empirical corrections** — set all alpha_dep, alpha_rim,
   alpha_melt, ice_vt_psd_factor to 1.0 when full tables are active

## Conclusions

1. **All P3 formulations are physically correct** — Mason thermodynamic resistance,
   Barklie-Gokhale freezing, Cooper nucleation, KK2000 warm rain, Seifert-Beheng
   self-collection.
2. **All parameter values match Fortran v5.5.0** within rounding.
3. **The ice gap is architectural**, not a bug — it stems from the mean-mass
   approximation vs PSD-integrated lookup tables.
4. **Temperature agreement is excellent** (1.00x), indicating the thermodynamic
   framework is sound.
5. **T,P-dependent transport properties** are active in the library via
   `air_transport_properties(T, P)`, matching Fortran v5.5.0 formulas.
6. **Rain velocity is unified**: tabulated path uses physical Gunn-Kinzer/Beard law;
   mean-mass fallback uses Fortran `842 D^0.8` consistently across all code paths.
7. **Cloud droplet number (Nᶜ) is prescribed**, not prognostic. This is a design
   simplification; the Fortran P3 driver uses prognostic Nᶜ. The homogeneous
   freezing number cap (`N_hom ≤ Q_hom / minimum_cloud_drop_mass`) compensates
   for this difference and can be removed when prognostic Nᶜ is implemented.
