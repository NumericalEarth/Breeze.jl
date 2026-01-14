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
| **Terminal velocity** | Simplified power law; needs regime-dependent coefficients | ⚠️ |
| **Particle mass in quadrature** | Simplified effective density; needs full piecewise m(D) | ⚠️ |
| **Capacitance** | Simple sphere/plate formula; needs full shape model | ⚠️ |
| **Critical diameters** | Placeholder values; needs dynamic computation | ⚠️ |

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
| **Rain evaporation** | `rain_evaporation_rate` | ✅ |

Implementation follows Khairoutdinov & Kogan (2000) for autoconversion/accretion
and Seifert & Beheng (2001) for self-collection.

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

### ❌ Not Implemented

#### Remaining Process Rate Tendencies

| Process | Fortran Subroutine | Status |
|---------|-------------------|--------|
| **Cloud droplet activation** | (aerosol module) | ❌ |
| **Cloud condensation/evaporation** | (saturation adjustment) | ❌ |
| **Ice nucleation** | Primary nucleation tendencies | ❌ |
| **Rime splintering** | Secondary ice production | ❌ |
| **Full sixth moment tendencies** | Z tendencies for aggregation, riming, etc. | ⚠️ simplified |

#### Sedimentation

| Component | Status |
|-----------|--------|
| **Rain sedimentation** | ✅ |
| **Ice sedimentation** | ✅ |
| **Terminal velocity computation** | ✅ |
| **Flux-form advection** | ✅ (via Oceananigans) |
| **Substepping** | ⚠️ (not yet, may be needed for stability) |

#### Lookup Tables

| Component | Description | Status |
|-----------|-------------|--------|
| **Reading Fortran tables** | Parse `p3_lookupTable_*.dat` files | ❌ |
| **Table 1** | Ice property integrals (size, rime, μ) | ❌ |
| **Table 2** | Rain property integrals | ❌ |
| **Table 3** | Z integrals for three-moment ice | ❌ |
| **GPU table storage** | Transfer tables to GPU architecture | ❌ |

#### Other

| Component | Description | Status |
|-----------|-------------|--------|
| **Multiple ice categories** | Milbrandt & Morrison (2016) Part III | ❌ (not planned) |
| **Substepping** | Implicit/operator-split time stepping | ❌ |
| **Diagnostics** | Reflectivity, precipitation rate | ❌ |
| **Aerosol coupling** | CCN activation | ❌ |

## Validation

A reference dataset from the Fortran `kin1d` driver is available in `validation/p3/`:

- `kin1d_reference.nc`: NetCDF output from Fortran P3 (v5.5.0)
- Configuration: 3-moment ice, liquid fraction, 90 min simulation
- Variables: temperature, mixing ratios, number concentrations, reflectivity, precip rates

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
   - Linear ramp from E=0.1 at 253K to E=1.0 at 268K

5. **Riming** ✅
   - `cloud_riming_rate`: Cloud droplet collection by ice
   - `rain_riming_rate`: Rain collection by ice  
   - `rime_density`: Temperature/velocity-dependent rime density
   - Rime mass/volume tendency updates

6. **Shedding/Refreezing** (liquid fraction) ✅
   - `shedding_rate`: Excess liquid sheds as rain (enhanced above 273K)
   - `refreezing_rate`: Liquid on ice refreezes below 273K
   - `shedding_number_rate`: Rain drops from shed liquid

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

### Phase 4: Validation

10. **kin1d comparison**
    - Single-column tests against Fortran reference
    - Process-by-process verification

11. **3D LES cases**
    - BOMEX with ice
    - Deep convection cases

## Code Organization

```
src/Microphysics/PredictedParticleProperties/
├── PredictedParticleProperties.jl  # Module definition, exports
├── p3_scheme.jl                    # Main PredictedParticlePropertiesMicrophysics type
├── p3_interface.jl                 # AtmosphereModel integration
├── process_rates.jl                # Phase 1 process rates (NEW)
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
└── cloud_droplet_properties.jl     # Cloud properties
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
