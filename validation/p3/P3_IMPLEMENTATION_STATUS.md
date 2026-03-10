# P3 Microphysics Implementation Status

This document summarizes the current state of the Predicted Particle Properties (P3)
microphysics implementation in Breeze.jl and what remains to reach parity with the
reference Fortran implementation in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics).

## Overview

P3 is a bulk microphysics scheme that uses a **single ice category** with continuously
predicted properties, rather than discrete categories like cloud ice, snow, graupel,
and hail. The implementation follows:

- **Morrison & Milbrandt (2015a)** - Original P3 scheme
- **Milbrandt et al. (2021)** - Three-moment ice (mass, number, reflectivity)
- **Milbrandt et al. (2025)** - Predicted liquid fraction

## Current Implementation Status

### ✅ Complete

#### Core Infrastructure

| Component | Description | Status |
|-----------|-------------|--------|
| **Prognostic fields** | 9 fields: `ρqᶜˡ`, `ρqʳ`, `ρnʳ`, `ρqⁱ`, `ρnⁱ`, `ρqᶠ`, `ρbᶠ`, `ρzⁱ`, `ρqʷⁱ` | ✅ |
| **AtmosphereModel integration** | `microphysics` interface for P3 as a drop-in scheme | ✅ |
| **Field materialization** | Creates all prognostic and diagnostic fields | ✅ |
| **Moisture fraction computation** | `compute_moisture_fractions` for thermodynamics | ✅ |
| **State update** | `update_microphysical_fields!` computes vapor as residual | ✅ |

#### Size Distribution & Lambda Solver

| Component | Description | Status |
|-----------|-------------|--------|
| **Gamma distribution** | `IceSizeDistributionState` with N₀, μ, λ | ✅ |
| **Two-moment closure** | μ-λ relationship from Field et al. (2007) | ✅ |
| **P3Closure** | Updated μ-λ including large-particle diagnostic (matching Fortran table gen) | ✅ |
| **FixedShapeParameter** | Constant μ=0 for exponential PSD (Fortran Table 1 convention) | ✅ |
| **Three-moment closure** | μ from Z/N constraint, no empirical closure | ✅ |
| **Lambda solver** | Secant method for two-moment and three-moment | ✅ |
| **Mass-diameter relation** | Piecewise power law (4 regimes) | ✅ |
| **Regime thresholds** | `ice_regime_thresholds` for small/aggregate/graupel/partial | ✅ |
| **Distribution parameters** | `distribution_parameters(L, N, ...)` and `(L, N, Z, ...)` | ✅ |

#### Quadrature & Tabulation

| Component | Description | Status |
|-----------|-------------|--------|
| **Chebyshev-Gauss quadrature** | Numerical integration over size distribution | ✅ |
| **Domain transformation** | x ∈ [-1, 1] → D ∈ [0, ∞) | ✅ |
| **Integral type hierarchy** | Abstract types for all integral categories | ✅ |
| **Integrand functions** | Fall speed, deposition, bulk, collection, sixth moment, λ-limiter | ✅ |
| **TabulatedFunction3D** | 3D trilinear interpolation wrapper | ✅ |
| **Full tabulation** | `tabulate(p3, CPU())` generates all ice + rain lookup tables | ✅ |
| **GPU tabulation** | `tabulate(p3, GPU())` transfers tables to GPU | ✅ |
| **Fortran Table 1 validation** | Julia quadrature vs Fortran v6.9-2momI (median <1% error) | ✅ |
| **Table dispatch** | Process rates auto-dispatch to table or analytical path via type system | ✅ |

#### Ice Property Containers

| Component | Description | Status |
|-----------|-------------|--------|
| **IceProperties** | Top-level container for all ice computations | ✅ |
| **IceFallSpeed** | Number/mass/reflectivity-weighted fall speed integrals | ✅ |
| **IceDeposition** | Ventilation integrals for vapor diffusion | ✅ |
| **IceBulkProperties** | Mean diameter, density, reflectivity | ✅ |
| **IceCollection** | Aggregation, rain collection integrals | ✅ |
| **IceSixthMoment** | Z-tendency integrals for rime, deposition, melt, etc. | ✅ |
| **IceLambdaLimiter** | Constraints on λ bounds | ✅ |
| **IceRainCollection** | Ice-rain collection integrals | ✅ |
| **RainProperties** | Rain integral types | ✅ |
| **CloudDropletProperties** | Cloud droplet parameters | ✅ |

#### Process Rates (All Phases)

| Process | Function | Status |
|---------|----------|--------|
| **Rain autoconversion** | `rain_autoconversion_rate` (KK2000) | ✅ |
| **Rain accretion** | `rain_accretion_rate` (KK2000 Eq. 33: `qc × qr^α`) | ✅ |
| **Rain self-collection** | `rain_self_collection_rate` (SB2001/2006) | ✅ |
| **Rain breakup** | `rain_breakup_rate` (SB2006 Eq. 13) | ✅ |
| **Rain evaporation** | `rain_evaporation_rate` | ✅ |
| **Ice deposition/sublimation** | `ventilation_enhanced_deposition` (MM15a Eq. 30) | ✅ |
| **Partitioned melting** | Partial (stays on ice) / complete (sheds to rain) | ✅ |
| **Ice aggregation** | `ice_aggregation_rate`, T-dependent Eii, rime-fraction limiter | ✅ |
| **Cloud riming** | `cloud_riming_rate`, full collection equation `E × ρ × ni × <AV>` | ✅ |
| **Rain riming** | `rain_riming_rate`, full collection equation | ✅ |
| **Shedding** | `shedding_rate` + number tendency | ✅ |
| **Refreezing** | `refreezing_rate` (liquid coating on ice) | ✅ |
| **Rime density** | `rime_density` (T/V-dependent) | ✅ |
| **Deposition nucleation** | `deposition_nucleation_rate` (Cooper 1986) | ✅ |
| **Immersion freezing (cloud)** | `immersion_freezing_cloud_rate` (Barklie-Gokhale 1959) | ✅ |
| **Immersion freezing (rain)** | `immersion_freezing_rain_rate` (Barklie-Gokhale 1959) | ✅ |
| **Rime splintering** | `rime_splintering_rate` (Hallett-Mossop 1974) | ✅ |

#### Terminal Velocities & Sedimentation

| Component | Description | Status |
|-----------|-------------|--------|
| **Mitchell-Heymsfield (2005)** | Best-number fall speed for quadrature | ✅ |
| **Rain terminal velocity** | Mass-weighted and number-weighted (power law) | ✅ |
| **Ice terminal velocity** | Mass/number/reflectivity-weighted (analytical + table dispatch) | ✅ |
| **`microphysical_velocities`** | Interface for Oceananigans tracer advection | ✅ |

#### Sixth Moment (Z) Tendencies

| Process | Status |
|---------|--------|
| Z from deposition/sublimation | ✅ |
| Z from nucleation | ✅ |
| Z from riming | ✅ |
| Z from aggregation | ✅ |
| Z from melting | ✅ |

### ⚠️ Partial / Simplified

| Component | Description | Status |
|-----------|-------------|--------|
| **Terminal velocity (process rates)** | Simplified power law (not Best-number); used in analytical path | ⚠️ |
| **Capacitance** | Simple sphere (cap=1)/plate (cap=0.48) formula | ⚠️ |
| **Transport properties** | Hardcoded `K_a=2.5e-2`, `D_v=2.5e-5`; T,P-dependent utility exists but not used | ⚠️ |

### ❌ Not Implemented

| Component | Description | Status |
|-----------|-------------|--------|
| **Contact freezing** | Meyers et al. (1992) | ❌ |
| **Condensation-freezing** | Additional nucleation mode for T < -35°C | ❌ |
| **Cloud droplet activation** | Aerosol → CCN | ❌ |
| **Table 2 (rain integrals)** | Rain property lookup tables (analytical fallback used) | ✅ (added) |
| **Multiple ice categories** | Full Part III (MultiIceCategory framework exists) | ⚠️ |
| **Sedimentation substepping** | Operator-split time stepping for CFL stability | ❌ |
| **Diagnostics** | Radar reflectivity, precipitation rate output | ❌ |

---

## Validation

### Lookup Table Validation (vs Fortran P3 v6.9-2momI)

Tables generated with `FixedShapeParameter(μ=0)` to match Fortran Table 1 convention.
Tested over Qnorm = 5–31, unrimed ice, |Fortran| > 1×10⁻¹⁰:

| Integral | Median error | P90 | Max |
|----------|-------------|-----|-----|
| uns (Vₙ) | 0.5% | 2.5% | 3.3% |
| ums (Vₘ) | 0.7% | 2.2% | 5.6% |
| vdep | 0.2% | 2.0% | 2.5% |
| eff | 0.6% | 2.1% | 3.0% |
| dmm | 0.0% | 0.2% | 0.6% |
| nrwat | 0.3% | 2.7% | 5.5% |

Rimed entries: 5–17% median (acceptable; regime boundary differences).

### kin1d Validation (KOUN sounding, nk=41, dt=10 s, 90 min)

Reference: Fortran P3 v5.5.0, nCat=1, trplMomIce=true, liqFrac=true.

**Temporal maxima ratios (Breeze / Fortran):**

| Field | Fortran max | Breeze max | Ratio | Assessment |
|-------|-------------|------------|-------|------------|
| Cloud liquid | 4.17 g/kg | ~4.05 g/kg | ~0.97× | Excellent |
| Rain | 5.47 g/kg | ~5.04 g/kg | ~0.92× | Good |
| Ice | 12.30 g/kg | ~5.40 g/kg | ~0.44× | Underproduced |
| Temperature | 30.6 °C | ~30.6 °C | ~1.00× | Excellent |

**Time evolution (selected times):**

| Time | Cloud ratio | Rain ratio | Ice ratio | Notes |
|------|-------------|-----------|-----------|-------|
| t=30 min | ~1.00× | ~0.97× | ~0.04× | Ice 25× too low; warm rain good |
| t=40 min | ~0.78× | ~0.80× | ~0.41× | Ice timing lag visible |
| t=60 min | — | ~8.55× | ~0.41× | Rain excess driven by ice deficit |

**Active PSD correction factors in kin1d driver:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `nucleation_coefficient` | 15.0 /m³ | 3× Cooper (1986) to approximate missing contact/condensation-freezing |
| `riming_psd_correction` | 5.0 | Mean-mass riming underestimates PSD-integrated collection kernel |
| `alpha_dep` (peak) | 2.0 | Mean-mass deposition overestimated D by 2× → f_v 3.7× too high; net boost needed |
| `alpha_dep` (floor) | 0.5 | Level-dependent below ice production peak |
| `alpha_rim` | 0.5 / 0.2 | Level-dependent riming correction |
| Nr constraint | clamp m_r to [1.4e-8, 5e-6] kg | Prevent Nr explosion from self-collection without breakup |

**Root causes of ice underproduction (~0.44×):**

1. **Missing nucleation modes**: Fortran P3 includes contact and condensation-freezing in addition
   to Cooper (1986) deposition freezing. These together produce 10–100× more ice particles at
   T < −35°C. Julia uses 3× Cooper coefficient as a proxy.

2. **Mean-mass approximation mismatch**: The analytical deposition path uses `ρ_eff = 100 kg/m³`
   to compute an effective diameter `D_mean ≈ 212 μm`. The actual distribution (P3Closure: μ ≈ 1
   at D_mvd ≈ 101 μm) has a number-mean diameter of ~111 μm. Because `C × f_v ∝ D^1.7`, this
   2× size gap causes the analytical path to overestimate per-particle deposition by ~3.7×. The
   `alpha_dep = 2.0` boost partially compensates.

3. **Full deposition tables are NOT the fix**: Full PSD-integrated tables give **0.27×** of the
   analytical rate (they are physically more correct, but the analytical was already overestimating).
   Switching to full deposition tables would reduce ice from ~0.44× to ~0.12×. The `alpha_dep`
   correction compensates for the analytical's inflated D_mean, not for missing PSD tail effects.

**Known limitations (inherent to 2-moment mean-mass approximation):**

- Ice peak altitude ~1.5 km lower than Fortran (no size-sorting; all ice falls at mean-mass velocity)
- Ice profile below peak flatter than Fortran (PSD broadening not captured without Z moment)
- Rain excess at mid-levels driven by ice underproduction (less riming consumption of rain)
- At T < −40°C: nucleation is Sᵢ-limited (vapor depletes before Cooper equilibrium); raising
  `nucleation_maximum_concentration` causes ni explosions and worsens validation

---

## Roadmap to Parity

### Phase 1: Core Process Rates ✅ COMPLETE
### Phase 2: Ice-Specific Processes ✅ COMPLETE
### Phase 3: Sedimentation ✅ COMPLETE
### Phase 4: Validation Driver ✅ COMPLETE

`kin1d_driver.jl` runs the full 41-level, 90-minute kinematic column test and compares
against Fortran P3 v5.5.0 reference output. Validation ratio ice ~0.44×; best achievable
with current 2-moment + mean-mass + Cooper-only nucleation.

### Phase 5: Lookup Tables ✅ COMPLETE

Julia-native lookup tables are implemented via `tabulate(p3, CPU())` and validated
against the Fortran P3 v6.9-2momI reference table. Tables integrate over the full
particle size distribution using Chebyshev-Gauss quadrature.

**Validation (unrimed, Qnorm 5–31):** median < 1%, P90 < 3%, max < 6%.

**Key Fortran convention fixes implemented:**
- Capacitance: `C_sphere = D` (`capm = cap × D`); rate equations use `2π × capm`
- Aggregation: factor 0.5 for self-collection matching Fortran upper-triangle sum
- nrwat: D ≥ 100 μm threshold matching Fortran

**Fall speed dispatch:** `kin1d_driver.jl` uses tabulated fall speeds via
`tabulate(p3, :ice_fall_speed, CPU())`. Deposition, riming, and aggregation use the
analytical fallback with PSD correction factors (switching to full tables requires
adding proper nucleation modes first).

### Phase 6: Remaining Work

| Item | Priority | Description |
|------|----------|-------------|
| **Contact + condensation-freezing nucleation** | High | Needed to close the ~2–3× ice deficit. Meyers (1992) contact freezing; condensation-freezing for T < −35°C |
| **T,P-dependent transport properties** | Medium | Switch from constant `K_a`, `D_v` to Sutherland/polynomial fits. Requires re-tuning PSD corrections. Utility `air_transport_properties(T, P)` already exists. |
| **3-moment Z in kin1d driver** | Medium | Currently Z is prognostic but driver doesn't exploit it for μ diagnosis; full 3-moment closure would improve ice particle size distribution |
| **Table 2 (rain integrals)** | Low | Added but kin1d uses analytical rain evap + Vt path with PSD boost factors |
| **Bulk tendency kernel** | Performance | 10× redundancy: each of 9 P3 fields calls `compute_p3_process_rates` independently |
| **Sedimentation substepping** | Stability | CFL constraint for large Δt; may be needed for production LES runs |
| **3D LES cases** | Validation | BOMEX with ice, deep convection |

---

## Code Organization

```
src/Microphysics/PredictedParticleProperties/
├── PredictedParticleProperties.jl  # Module definition, exports
├── p3_scheme.jl                    # PredictedParticlePropertiesMicrophysics type
├── p3_interface.jl                 # AtmosphereModel integration
├── process_rates.jl                # Table-dispatch helpers + compute_p3_process_rates
├── process_rate_parameters.jl      # All tunable process rate parameters
├── rain_process_rates.jl           # Rain autoconversion, accretion, evaporation
├── melting_rates.jl                # Ice melting (partitioned: partial/complete)
├── collection_rates.jl             # Ice riming and aggregation rates
├── ice_nucleation_rates.jl         # Deposition nucleation, immersion/rain freezing
├── terminal_velocities.jl          # Rain/ice terminal velocity computation
├── integral_types.jl               # Abstract integral type hierarchy
├── size_distribution.jl            # Gamma distribution, regime thresholds
├── lambda_solver.jl                # Two/three-moment λ, μ solvers + closures
├── quadrature.jl                   # Chebyshev-Gauss integration + integrand functions
├── tabulation.jl                   # Lookup table infrastructure (TabulatedFunction3D)
├── ice_properties.jl               # IceProperties container
├── ice_fall_speed.jl               # Fall speed integral types
├── ice_deposition.jl               # Ventilation integral types
├── ice_bulk_properties.jl          # Bulk property integral types
├── ice_collection.jl               # Collection integral types
├── ice_sixth_moment.jl             # Z-tendency integral types
├── ice_lambda_limiter.jl           # λ constraint integral types
├── ice_rain_collection.jl          # Ice-rain collection integrals
├── rain_properties.jl              # Rain integral types
├── cloud_droplet_properties.jl     # Cloud droplet properties
└── multi_ice_category.jl           # Multi-category ice framework
```

---

## References

1. Morrison, H., and J. A. Milbrandt, 2015a: Parameterization of cloud microphysics
   based on the prediction of bulk ice particle properties. Part I. *J. Atmos. Sci.*, **72**, 287–311.

2. Milbrandt, J. A., H. Morrison, D. T. Dawson II, and M. Paukert, 2021: A triple-moment
   representation of ice in the Predicted Particle Properties (P3) microphysics scheme.
   *J. Atmos. Sci.*, **78**, 439–458.

3. Milbrandt, J. A., H. Morrison, A. Ackerman, and H. Jäkel, 2025: Predicted liquid
   fraction on ice particles in the P3 microphysics scheme. *J. Atmos. Sci.* (submitted).

4. Field, P. R., et al., 2007: Snow size distribution parameterization for midlatitude
   and tropical ice clouds. *J. Atmos. Sci.*, **64**, 4346–4365.

5. Khairoutdinov, M., and Y. Kogan, 2000: A new cloud physics parameterization in a
   large-eddy simulation model of marine stratocumulus. *Mon. Wea. Rev.*, **128**, 229–243.

6. Seifert, A., and K. D. Beheng, 2001: A double-moment parameterization for simulating
   autoconversion, accretion and self-collection. *Atmos. Res.*, **59–60**, 265–281.

7. Seifert, A., and K. D. Beheng, 2006: A two-moment cloud microphysics parameterization
   for mixed-phase clouds. *Meteorol. Atmos. Phys.*, **92**, 45–66.

8. Barklie, R. H. D., and N. R. Gokhale, 1959: The freezing of supercooled water drops.
   *Sci. Rep. MW-30*, Stormy Weather Group, McGill University.

9. Cooper, W. A., 1986: Ice initiation in natural clouds. *Precipitation Enhancement—A
   Scientific Challenge*, Meteor. Monogr., No. 43, Amer. Meteor. Soc., 29–32.

10. Hallett, J., and S. C. Mossop, 1974: Production of secondary ice particles during the
    riming process. *Nature*, **249**, 26–28.

11. Mitchell, D. L., and A. J. Heymsfield, 2005: Refinements in the treatment of ice particle
    terminal velocities, highlighting aggregates. *J. Atmos. Sci.*, **62**, 1637–1644.
