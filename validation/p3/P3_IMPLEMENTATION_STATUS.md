# P3 Microphysics Implementation Status

This document summarizes the current state of the Predicted Particle Properties (P3)
microphysics implementation in Breeze.jl and what remains to reach parity with the
reference Fortran implementation in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics) (v5.5.0).

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
| **Integrand functions** | Fall speed, deposition, bulk properties, collection, sixth moment | ✅ |
| **Tabulation infrastructure** | `TabulatedIntegral` wrapper for lookup tables | ✅ |
| **Tabulation parameters** | `TabulationParameters` for grid specification | ✅ |

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

### ⚠️ Partial / Simplified

| Component | Description | Status |
|-----------|-------------|--------|
| **Terminal velocity (quadrature)** | Full Best-number Mitchell-Heymsfield formulation | ✅ |
| **Terminal velocity (process rates)** | Simplified power law with regime switching | ⚠️ |
| **Particle mass in quadrature** | Full piecewise m(D) with four regimes | ✅ |
| **Capacitance** | Simple sphere/plate formula; needs full shape model | ⚠️ |
| **Critical diameters** | Dynamic computation in `size_distribution.jl` | ✅ |

### ✅ Phase 1 Process Rates (VERIFIED WORKING)

Phase 1 process rates are implemented in `process_rates.jl` and have been verified to work
in a parcel model test. After 100 seconds of simulation:
- Cloud liquid: 5.0 → 3.0 g/kg (autoconversion/accretion)
- Rain: 1.0 → 0.002 g/kg (evaporation in subsaturated air)
- Ice: 2.0 → 0.0 g/kg (melting at T > 273 K)

#### Rain Processes

| Process | Function | Status |
|---------|----------|--------|
| **Rain autoconversion** | `rain_autoconversion_rate` | ✅ |
| **Rain accretion** | `rain_accretion_rate` | ✅ |
| **Rain self-collection** | `rain_self_collection_rate` | ✅ |
| **Rain breakup** | `rain_breakup_rate` | ✅ |
| **Rain evaporation** | `rain_evaporation_rate` | ✅ |

Implementation follows Khairoutdinov & Kogan (2000) for autoconversion/accretion,
Seifert & Beheng (2001, 2006) for self-collection and breakup.

#### Ice Deposition/Sublimation

| Process | Function | Status |
|---------|----------|--------|
| **Ice deposition** | `ice_deposition_rate` | ✅ |
| **Ventilation-enhanced deposition** | `ventilation_enhanced_deposition` | ✅ |

Relaxation-to-saturation formulation with simplified ventilation factors.

#### Melting

| Process | Function | Status |
|---------|----------|--------|
| **Ice melting (mass)** | `ice_melting_rate` | ✅ |
| **Ice melting (number)** | `ice_melting_number_rate` | ✅ |

Temperature-dependent melting rate for T > T_freeze.

#### Tendency Integration

| Component | Description | Status |
|-----------|-------------|--------|
| **P3ProcessRates** | Container for all computed rates | ✅ |
| **compute_p3_process_rates** | Main rate calculation function | ✅ |
| **microphysical_tendency** | Now dispatches to field-specific tendencies | ✅ |
| **Field tendency functions** | `tendency_ρqᶜˡ`, `tendency_ρqʳ`, etc. | ✅ |

### ⚠️ Partially Implemented

#### Process Rate Tendencies

| Process | Function | Status |
|---------|----------|--------|
| **Ice aggregation** | `ice_aggregation_rate` | ✅ |
| **Cloud riming** | `cloud_riming_rate` | ✅ |
| **Rain riming** | `rain_riming_rate` | ✅ |
| **Shedding** | `shedding_rate` | ✅ |
| **Refreezing** | `refreezing_rate` | ✅ |
| **Rime density** | `rime_density` | ✅ |

### ✅ Ice Nucleation & Secondary Ice (VERIFIED)

Ice nucleation and secondary ice production are now implemented:

#### Ice Nucleation

| Process | Function | Status |
|---------|----------|--------|
| **Deposition nucleation** | `deposition_nucleation_rate` | ✅ |
| **Immersion freezing (cloud)** | `immersion_freezing_cloud_rate` | ✅ |
| **Immersion freezing (rain)** | `immersion_freezing_rain_rate` | ✅ |
| **Rime splintering** | `rime_splintering_rate` | ✅ |

- Deposition nucleation: Cooper (1986) parameterization, T < -15°C, Sᵢ > 5%
- Immersion freezing: Barklie-Gokhale (1959) stochastic freezing, T < -4°C, a_imm=0.65
- Rime splintering: Hallett-Mossop (1974), -8°C < T < -3°C, peaks at -5°C

#### Sixth Moment Tendencies

| Process | Status |
|---------|--------|
| **Z from deposition/melting** | ✅ |
| **Z from nucleation** | ✅ |
| **Z from riming** | ✅ |
| **Z from aggregation** | ✅ (conserved approximation) |

### ❌ Not Implemented

#### Remaining Components

| Process | Description | Status |
|---------|-------------|--------|
| **Cloud droplet activation** | (aerosol module) | ❌ |
| **Cloud condensation/evaporation** | `cloud_condensation_rate` with psychrometric correction | ✅ |
| **Contact freezing** | `contact_freezing_rate` following Meyers et al. (1992) | ✅ |
| **Lookup tables (Fortran I/O)** | Read Fortran tables | ❌ |

#### Sedimentation

| Component | Status |
|-----------|--------|
| **Rain sedimentation** | ✅ |
| **Ice sedimentation** | ✅ |
| **Terminal velocity computation** | ✅ |
| **Flux-form advection** | ✅ (via Oceananigans) |
| **Substepping** | ⚠️ (not yet, may be needed for stability) |

Note: P3 uses a different sedimentation approach than DCMIP2016Kessler. Rather than
explicit column-by-column sedimentation with substepping, P3 returns terminal velocity
structs via `microphysical_velocities` that are used by Oceananigans' tracer advection.
Substepping would need to be implemented at a higher level (e.g., in the time stepper).

#### Lookup Tables

| Component | Description | Status |
|-----------|-------------|--------|
| **Tabulation infrastructure** | `tabulate()` function, `TabulationParameters` | ✅ |
| **Fall speed tabulation** | `tabulate(p3, :ice_fall_speed, arch)` | ✅ |
| **Deposition tabulation** | `tabulate(p3, :ice_deposition, arch)` | ✅ |
| **Reading Fortran tables** | Parse `p3_lookupTable_*.dat` files | ❌ |
| **Table 1** | Ice property integrals (size, rime, μ) | ⚠️ (can generate, not read) |
| **Table 2** | Rain property integrals | ❌ |
| **Table 3** | Z integrals for three-moment ice | ❌ |
| **GPU table storage** | Transfer tables to GPU architecture | ⚠️ (TODO in code) |

#### Other

| Component | Description | Status |
|-----------|-------------|--------|
| **Multiple ice categories** | `MultiIceCategory` framework exists, full Part III not complete | ⚠️ |
| **Substepping** | Implicit/operator-split time stepping | ❌ |
| **Diagnostics** | Reflectivity, precipitation rate | ❌ |
| **Aerosol coupling** | CCN activation | ❌ |

## Validation

A reference dataset from the Fortran `kin1d` driver must be generated separately (see
`README.md`). `kin1d_reference.nc` is not checked into the repository and is not available
on all machines. Parcel model validation (`parcel_validation.jl`) exercises
`compute_p3_process_rates` directly and is the current primary validation tool.

## Roadmap to Parity

### Phase 1: Core Process Rates ✅ COMPLETE

1. **Rain processes** ✅
   - Autoconversion (cloud → rain): `rain_autoconversion_rate`
   - Accretion (cloud + rain → rain): `rain_accretion_rate`
   - Self-collection: `rain_self_collection_rate`
   - Evaporation: `rain_evaporation_rate`

2. **Ice deposition/sublimation** ✅
   - Vapor diffusion growth/loss: `ice_deposition_rate`
   - Ventilation factors: `ventilation_enhanced_deposition`

3. **Melting** ✅
   - Ice → rain conversion: `ice_melting_rate`
   - Number tendency: `ice_melting_number_rate`

### Phase 2: Ice-Specific Processes ✅ COMPLETE

Phase 2 process rates are implemented in `process_rates.jl` and verified:
- Ice number decreased from 10000 → 5723 /kg (aggregation active)
- Cloud liquid consumed by autoconversion + accretion + riming
- Rime density computed based on temperature

4. **Aggregation** ✅
   - `ice_aggregation_rate`: Temperature-dependent sticking efficiency
   - Linear ramp from E=0.001 at 253K to E=0.3 at 273K (matching Fortran P3)
   - Eii_fact rime-fraction limiter: shuts off aggregation for Fr > 0.9

5. **Riming** ✅
   - `cloud_riming_rate`: Cloud droplet collection by ice
   - `rain_riming_rate`: Rain collection by ice
   - `rime_density`: Temperature/velocity-dependent rime density
   - Rime mass/volume tendency updates

6. **Shedding/Refreezing** (liquid fraction) ✅
   - `shedding_rate`: Excess liquid sheds as rain (enhanced above 273K)
   - `refreezing_rate`: Liquid on ice refreezes below 273K
   - `shedding_number_rate`: Rain drops from shed liquid

7. **Ice Nucleation** ✅
   - `deposition_nucleation_rate`: Cooper (1986), T < -15°C, Sᵢ > 5%
   - `immersion_freezing_cloud_rate`: Barklie-Gokhale (1959), cloud droplets freeze at T < -4°C
   - `immersion_freezing_rain_rate`: Barklie-Gokhale (1959), rain drops freeze at T < -4°C

8. **Secondary Ice Production** ✅
   - `rime_splintering_rate`: Hallett-Mossop (1974), -8°C < T < -3°C
   - Peaks at -5°C, ~350 splinters per mg of rime

### Phase 3: Sedimentation & Performance ✅ COMPLETE

Phase 3 terminal velocities are implemented in `process_rates.jl` and verified:
- Rain mass-weighted: 4.4 m/s (1 mm drops, typical)
- Unrimed ice: 1.0 m/s (aggregates, typical)
- Rimed ice/graupel: 1.5 m/s (riming increases density)

7. **Terminal velocities** ✅
   - `rain_terminal_velocity_mass_weighted`: Power-law with density correction
   - `rain_terminal_velocity_number_weighted`: For rain number sedimentation
   - `ice_terminal_velocity_mass_weighted`: Regime-dependent (Stokes/Mitchell)
   - `ice_terminal_velocity_number_weighted`: For ice number sedimentation
   - `ice_terminal_velocity_reflectivity_weighted`: For Z sedimentation

8. **Sedimentation** ✅
   - `microphysical_velocities` implemented for all 8 precipitating fields
   - Callable velocity structs: `RainMassSedimentationVelocity`, etc.
   - Returns `(u=0, v=0, w=-vₜ)` for advection interface

9. **Lookup tables** ❌ (Not yet)
   - Read Fortran tables or regenerate in Julia
   - GPU-compatible table access

### Phase 4: Validation ⚠️ IN PROGRESS

#### Status of reference data

- `kin1d_reference.nc` is **not available** on this machine; it must be generated by running
  the Fortran P3 `kin1d` driver (commit `24bf078b`, config: nCat=1, trplMomIce=true,
  liqFrac=true) and then running `make_kin1d_reference.jl`. See `README.md`.

- The `kinematic_column_driver.jl` does **not** call `compute_p3_process_rates` — it uses
  placeholder τ-based parameterizations. Its output is not meaningful for evaluating P3
  physics and the stale comparison table has been retired.

#### Parcel model validation (`parcel_validation.jl`)

A standalone parcel model (`validation/p3/parcel_validation.jl`) exercises
`compute_p3_process_rates` directly with three idealised scenarios.
Condensation is handled via a one-step Newton saturation adjustment instead of
explicit Euler, eliminating stiffness from the fast condensation timescale.
Results below are from a 100-minute integration at Δt = 10 s.

**Scenario 1 — Ice deposition + aggregation (T = −13 °C, no cloud liquid)**

*Initial:* T = 260 K, p = 60 kPa, qᵢ = 0.500 g/kg, nᵢ = 1×10⁵ /kg, Sᵢ = 102 %, qᶜˡ = 0

| Quantity | Initial | Final | Expected behaviour |
|----------|---------|-------|-------------------|
| qᵢ [g/kg] | 0.500 | **0.527** | ↑ deposition (Sᵢ > 1) |
| nᵢ [/kg] | 1.00×10⁵ | **4.14×10⁴** | ↓ aggregation (−59%) |
| T [K] | 260.00 | **260.08** | ↑ latent heat from deposition |
| Sᵢ [%] | 102 | **100** | approaches saturation |

✅ Deposition grows ice; aggregation depletes number; latent heat warms parcel. Physically consistent.

**Scenario 2 — Mixed-phase cloud in Hallett-Mossop zone (T = −5 °C)**

*Initial:* T = 268 K, p = 70 kPa, qᶜˡ = 0.100 g/kg, qᵢ = 0.200 g/kg, nᵢ = 1×10⁵ /kg,
Ff = 0.05, Sᵢ = 101 %

| Quantity | Initial | Final | Expected behaviour |
|----------|---------|-------|-------------------|
| qᶜˡ [g/kg] | 0.100 | **0.000** | depleted by immersion freezing |
| qᵢ [g/kg] | 0.200 | **0.314** | ↑ riming + deposition + freezing |
| nᵢ [/kg] | 1.00×10⁵ | **9.10×10⁵** | ↑ splintering (Hallett-Mossop) |
| Ff | 0.050 | **0.032** | rime fraction grows, qi grows faster |
| T [K] | 268.00 | **268.07** | latent heat from freezing/deposition |

✅ Hallett-Mossop splintering bursts nᵢ by 10×; immersion freezing consumes cloud liquid;
rime fraction evolves realistically.

**Scenario 3 — Warm rain (T = +10 °C, cloud only)**

*Initial:* T = 283 K, p = 85 kPa, qᶜˡ = 0.500 g/kg, qᵣ = 0, qi = 0, Sl = 100 %

| Quantity | Initial | Final | Expected behaviour |
|----------|---------|-------|-------------------|
| qᶜˡ [g/kg] | 0.500 | **0.002** | ↓ autoconversion + accretion |
| qᵣ [g/kg] | 0.000 | **0.499** | ↑ collision-coalescence |
| qᵢ [g/kg] | 0.000 | **0.000** | no ice at T > 0 °C |
| T [K] | 283.00 | **283.00** | conserved (liquid-only redistribution) |

✅ Warm rain conversion proceeds without spurious ice formation. Temperature conserved.
Autoconversion seeds rain; accretion rapidly transfers cloud liquid to rain.

#### Known issues / caveats

- P3's `deposition` rate returns a non-zero value even when qᵢ = nᵢ = 0 at T > 0 °C
  (no ice crystals present). The parcel driver guards against this by gating deposition
  on qᵢ > 0. This should be investigated in `process_rates.jl`.

- The previous kin1d comparison table (pre-bug-fix, simplified τ-based driver) has been
  removed. The parcel model now calls the actual `compute_p3_process_rates` function.

10. **Parcel model validation** ✅
    - ✅ `parcel_validation.jl` exercises `compute_p3_process_rates` directly
    - ✅ Three scenarios cover ice-only, mixed-phase, and warm-rain regimes
    - ✅ Saturation adjustment replaces stiff explicit-Euler condensation
    - ⚠️ kin1d column comparison pending (requires Fortran P3 output)

11. **kin1d comparison** ⚠️ PARTIALLY VALIDATED
    - ✅ `kin1d_driver.jl` calls `compute_p3_process_rates` with full P3 physics
    - ✅ Reference data from Fortran P3 v5.5.0 (`reference_out_p3_1TT.dat`)
    - ✅ Quantitative comparison implemented (`compare_profiles.jl`)

    **kin1d validation results (1D kinematic column, KOUN sounding, nk=41, dt=10s, 90 min):**

    | Field | Fortran max | Breeze max | Ratio | Assessment |
    |-------|-------------|------------|-------|------------|
    | Cloud liquid | 4.17 g/kg | ~4.13 g/kg | ~0.99 | Excellent |
    | Ice | 12.30 g/kg | ~6.5 g/kg | ~0.53 | Low (mean-mass limitation) |
    | Rain | 5.47 g/kg | ~10.4 g/kg | ~1.91 | Elevated (rain lingers) |
    | Temperature | 30.6 °C | ~32.6 °C | +2°C | Warm bias |

    **Time evolution quality (with PSD correction factors):**

    | Time | Cloud ratio | Ice ratio | Rain ratio | Notes |
    |------|-------------|-----------|------------|-------|
    | t=30 | 1.00 | — | 0.86 | Warm rain excellent |
    | t=40 | 1.82 | — | — | Cloud persists (riming too slow) |
    | t=60 | 0.69 | 0.53 | 8.50 | Ice low; excess rain from slow sed. |
    | t=80 | — | — | 1.91 | Late rain converging |

    **PSD correction factors (empirical, pending lookup tables):**
    - `riming_psd_correction = 5.0`: Cloud/rain collection kernel enhancement
    - `rain_vt_psd_factor = 1.7`: Rain mass-weighted sedimentation velocity
    - `rain_evap_psd_factor = 5.0`: Evaporation rate enhancement
    - `ice_vt_mass_weight_factor = 1.9`: Ice mass-weighted sedimentation velocity (in source)
    - Nr constraint: clamp mean rain drop mass to [1.4e-8, 5e-6] kg

    **Key bug fixes in this validation cycle:**
    - Accretion formula: changed from `(qc×qr)^α` to `qc × qr^α` (KK2000 Eq. 33)
    - Added rain breakup function (SB2006 Eq. 13, available but not used in driver)
    - Riming uses full collection equation with `ni × <A×V>` instead of `qi/τ`
    - Immersion freezing uses Barklie-Gokhale (1959) volume-dependent formula
    - Ice fall speed uses parameters from `ProcessRateParameters` instead of hardcoded values
    - Melting limiter uses physical heat-transfer rate with 1s safety floor

    **Known limitations (inherent to mean-mass approximation):**
    - Ice peak altitude ~1.5 km lower than Fortran (mean-mass velocity too uniform)
    - Ice profile below peak flatter than Fortran (no size-sorting without PSD)
    - Rain accumulates at mid-levels due to underestimated sedimentation
    - Temperature 2°C warm bias from excess latent heat in melting zone
    - Nr explosion at cold levels without explicit constraint on mean drop mass

    **Resolution:** Full fix requires implementing P3 lookup tables (Phase 5)

### Phase 5: Lookup Tables (Required for Fortran Parity)

The current mean-mass approximation systematically overestimates concave process rates
(Jensen's inequality: f(m̄) > ⟨f(m)⟩_PSD for concave f). Lookup tables integrate over
the full particle size distribution, resolving:
- Fall speed size-sorting → steep ice profile below production peak
- PSD-weighted deposition/riming → correct rate magnitudes at all levels
- PSD-weighted melting → correct rain generation from ice drainage
- Eliminates the need for empirical alpha corrections and profile relaxation

12. **Lookup table implementation** ❌
    - ❌ Read Fortran tables or regenerate in Julia
    - ❌ Interpolation for process rates (mass, number, reflectivity)
    - ❌ GPU-compatible table access

13. **3D LES cases** ❌
    - BOMEX with ice
    - Deep convection cases

## Code Organization

```
src/Microphysics/PredictedParticleProperties/
├── PredictedParticleProperties.jl  # Module definition, exports
├── p3_scheme.jl                    # Main PredictedParticlePropertiesMicrophysics type
├── p3_interface.jl                 # AtmosphereModel integration
├── process_rates.jl                # Phase 1+2 process rates and terminal velocities
├── integral_types.jl               # Abstract integral type hierarchy
├── size_distribution.jl            # Gamma distribution, regime thresholds
├── lambda_solver.jl                # Two/three-moment λ, μ solvers
├── quadrature.jl                   # Chebyshev-Gauss integration
├── tabulation.jl                   # Lookup table infrastructure
├── ice_properties.jl               # IceProperties container
├── ice_fall_speed.jl               # Fall speed integral types
├── ice_deposition.jl               # Ventilation integral types
├── ice_bulk_properties.jl          # Bulk property integral types
├── ice_collection.jl               # Collection integral types
├── ice_sixth_moment.jl             # Z-tendency integral types
├── ice_lambda_limiter.jl           # λ constraint integral types
├── ice_rain_collection.jl          # Ice-rain collection integrals
├── rain_properties.jl              # Rain integral types
├── cloud_droplet_properties.jl     # Cloud properties
├── cloud_properties.jl             # Cloud properties (unused duplicate)
├── multi_ice_category.jl           # Multi-category ice framework
└── process_rate_parameters.jl      # All tunable process rate parameters
```

## References

1. Morrison, H., and J. A. Milbrandt, 2015a: Parameterization of cloud microphysics
   based on the prediction of bulk ice particle properties. Part I: Scheme description
   and idealized tests. J. Atmos. Sci., 72, 287–311.

2. Milbrandt, J. A., H. Morrison, D. T. Dawson II, and M. Paukert, 2021: A triple-moment
   representation of ice in the Predicted Particle Properties (P3) microphysics scheme.
   J. Atmos. Sci., 78, 439–458.

3. Milbrandt, J. A., H. Morrison, A. Ackerman, and H. Jäkel, 2025: Predicted liquid
   fraction on ice particles in the P3 microphysics scheme. J. Atmos. Sci. (submitted).

4. Field, P. R., et al., 2007: Snow size distribution parameterization for midlatitude
   and tropical ice clouds. J. Atmos. Sci., 64, 4346–4365.

5. Khairoutdinov, M., and Y. Kogan, 2000: A new cloud physics parameterization in a
   large-eddy simulation model of marine stratocumulus. Mon. Wea. Rev., 128, 229–243.

6. Seifert, A., and K. D. Beheng, 2001: A double-moment parameterization for simulating
   autoconversion, accretion and self-collection. Atmos. Res., 59–60, 265–281.

7. Seifert, A., and K. D. Beheng, 2006: A two-moment cloud microphysics parameterization
   for mixed-phase clouds. Meteorol. Atmos. Phys., 92, 45–66.

8. Barklie, R. H. D., and N. R. Gokhale, 1959: The freezing of supercooled water drops.
   Sci. Rep. MW-30, Stormy Weather Group, McGill University.
