# P3 Implementation Status Report

**Date:** 2026-03-13 (bug fix batch: C1, H6, H7, M3, M8, M9, M10, M11, M12, M13, M14, L6, L7)
**Branch:** glw/p3
**Reference:** Fortran P3 v5.5.0 / lookup table v6.9-2momI
**Reviewed:** All 29 source files in `src/Microphysics/PredictedParticleProperties/`

---

## Executive Summary

The Breeze P3 implementation is **comprehensive and physically sound**. All major process
categories from Fortran P3 v5.5.0 are implemented: warm rain (KK2000), ice
deposition/sublimation with ventilation, partitioned melting (Milbrandt et al. 2025),
aggregation, cloud and rain riming, shedding, refreezing, nucleation (Cooper, Barklie-Gokhale,
homogeneous), and Hallett-Mossop splintering. The 3-moment ice framework is complete.

**19 of 21 Fortran main-table columns implemented** (missing only aerosol collection).
All 9 sixth-moment (3-moment) integrals implemented. Lookup table validation shows
median <1%, P90 <3% error for unrimed ice.

**Physics review (2026-03-13)** identified **2 CRITICAL**, **7 HIGH**, and
**14 MEDIUM** issues. After this batch fix: **1 CRITICAL** (C1), **2 HIGH** (H6, H7),
**4 MEDIUM** code fixes (M3, M9, M11, M14), and **6 MEDIUM/LOW** documentation fixes
(M8, M10, M12, M13, L6, L7) are resolved. Remaining CRITICAL: lambda solver div-by-zero (C2).

---

## File-by-File Status

### Core Infrastructure

**`PredictedParticleProperties.jl`** (215 lines) -- COMPLETE
- Fortran equivalent: `microphy_p3` module header
- Differences from Fortran:
  - Tables generated at runtime via `tabulate()` vs Fortran reading binary files from `P3_INIT`
  - No `P3_MAIN` monolithic subroutine -- decomposed into `compute_p3_process_rates` + individual tendency functions
  - SCPF (sub-column precipitation fraction) not exported or implemented (`scpf_on=.false.` in Fortran ref too)

**`p3_scheme.jl`** (190 lines) -- COMPLETE
- Fortran equivalent: Module-level variables + `P3_INIT`
- Differences from Fortran:
  - Cloud droplet number is **prescribed** (not prognostic). Fortran passes prognostic Nc
  - `minimum_mass_mixing_ratio = 1e-14` and `minimum_number_mixing_ratio = 1e-16` match Fortran `qsmall` and `nsmall` exactly
  - `precipitation_boundary_condition = nothing` (open boundary). Fortran uses upwind sedimentation with `dz2d` layer thicknesses. Breeze delegates sedimentation to Oceananigans advection

**`p3_interface.jl`** (350 lines) -- COMPLETE
- Fortran equivalent: `P3_MAIN` argument list, main loop structure, tendency application
- Differences from Fortran:
  - 9 prognostic fields matching Fortran v5.5: `ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ`
  - No sedimentation sub-stepping (Fortran sub-steps for CFL stability with fast-falling graupel)
  - Fortran applies tendencies sequentially with intermediate clipping (Lie splitting); Breeze computes all rates simultaneously
  - No prognostic supersaturation (`ssat`); Breeze diagnoses saturation from thermodynamic state
  - `ρqᶠ` and `ρbᶠ` sediment with ice mass-weighted velocity (matches Fortran convention)
  - No cloud fraction (SCPF)

**`process_rate_parameters.jl`** (350 lines) -- COMPLETE
- Fortran equivalent: Hardcoded constants throughout `P3_MAIN` and `P3_INIT`
- Differences from Fortran:
  - All 65+ parameters match Fortran defaults. Key verified values: `ρ₀ = 80000/(287.15*273.15)` (Fortran `rhosur`), `mi0 = 4π/3*900*(1e-6)^3` (ρ=900 not 917), `eci=0.5`, `eri=1.0`, `aimm=0.65`, `bimm=2.0`
  - `riming_psd_correction = 2.0` is Breeze-specific (empirical bridge for mean-mass path; Fortran uses table-integrated collection)
  - `aggregation_timescale = 600s` and `reference_concentration = 1e4` are Breeze-specific tuning parameters not in Fortran
  - `sink_limiting_timescale = 1s` is Breeze-specific (Fortran clips after application, not before)
  - Condensation timescale lives in `CloudDropletProperties` (1s default); Fortran uses prognostic `ssat` equation

**`integral_types.jl`** (180 lines) -- COMPLETE
- Fortran equivalent: Integer indices into `itab`, `itab_3mom` arrays
- Differences from Fortran:
  - Each integral is a separate singleton type enabling compile-time dispatch (vs Fortran integer indexing with runtime branching)
  - All 29 Fortran ice table entries have corresponding types; no missing integrals
  - Aerosol collection types (`nawcol`, `naicol` -- Fortran columns 20-21) not implemented

**`size_distribution.jl`** (100 lines) -- COMPLETE
- Fortran equivalent: Implicit PSD in `P3_MAIN` local variables (N0, mu_i, lam)
- Differences from Fortran:
  - Explicit `IceSizeDistributionState{FT}` struct with 11 fields vs Fortran's implicit local variables
  - `liquid_fraction` field present but defaults to zero (Milbrandt et al. 2025 extension)
  - Default mass-diameter parameters match: α=0.0121, β=1.9
  - Default ice reference density: `60000/(287.15*253.15) ≈ 0.825 kg/m³` matches Fortran `rhosui`

### Lambda Solver & Tabulation

**`lambda_solver.jl`** (1144 lines) -- COMPLETE
- Fortran equivalent: `FIND_LOOKUPTABLE_INDICES_1/2`, `compute_mu_lambda`, regime threshold logic in `create_p3_lookupTable_1.f90`
- Differences from Fortran:
  - **Secant method** (Breeze) vs **bisection** (Fortran) for lambda solver. Secant converges faster but is less robust; clamped bounds prevent divergence. **BUG (C2)** — Two-moment secant solver at line 790 has `Δx = f₁ * (x₁ - x₀) / (f₁ - f₀)` with no guard against `f₁ == f₀`, causing division-by-zero. The three-moment solver at line 860 correctly guards with `abs(denom) < eps(FT) && return x₁`
  - Mu-lambda closure matches exactly: `0.00191*λ^0.8 - 2` is algebraically identical to Fortran's `0.076*(lam/100)^0.8 - 2`; clamp [0,6] for two-moment, [0,20] for P3Closure large regime
  - P3Closure D_mvd threshold = 0.2mm, large regime formula `0.25*(D_mvd_mm - 0.2)*f_rho*Ff` -- all match Fortran
  - Regime thresholds (D_spherical, D_graupel, D_partial_rime) use same formulas but different algebraic form for D_cr; produces identical results
  - `deposited_ice_density` implements MM15a Eq. 16 -- matches Fortran
  - **Lambda bounds are wider**: Breeze `DiameterBounds` D_max=40mm vs Fortran effective ~500μm (Fortran `lammin=1/(10*dcs)` with `dcs=500e-6`). Breeze allows much larger mean diameters
  - Fortran indexes tables by `i_Qnorm`; Breeze indexes by `log10(mean_mass)` directly

**`tabulation.jl`** (1420 lines) -- COMPLETE
- Fortran equivalent: `create_p3_lookupTable_1.f90` (table generation), `find_lookupTable_indices_1` (runtime interpolation)
- Differences from Fortran:
  - **Missing rime_density axis**: Fortran table has outer loop over 5 rime densities (50, 250, 450, 650, 900 kg/m³). Breeze uses fixed `rime_density=400` for all tabulated evaluations. This causes 10-30% error for dense graupel (ρ_r=900)
  - Table grid: 150 mass × 8 Fr × 4 Fl points (Breeze) vs 50 Qnorm × 4 Fr × 4 Fl × 5 ρ_r (Fortran). More mass resolution but missing ρ_r dimension
  - **No ice-rain binned sub-table**: Fortran has `itabcol(i_Qnorm, i_Drscale, i_Fr, i_Fl)` with 30 rain-size bins. Breeze tabulates ice-rain integrals as standard 3D tables without the rain diameter dimension
  - Interpolation: both use trilinear; Breeze adds safe truncation with `max(unsafe_trunc(...), 0)` guard
  - Table stores `lambda_i` and `mu_i` as diagnostic columns (integrand returns zero) -- matches Fortran columns 13-14
  - `Adapt.adapt_structure` defined for all 10 container structs (GPU transfer support)
  - `P3IntegralEvaluator` can use either `P3Closure` or `FixedShapeParameter(0)` for table generation; Fortran always uses μ=0

**`quadrature.jl`** (951 lines) -- COMPLETE
- Fortran equivalent: Inner loops of `create_p3_lookupTable_1.f90` (integrands at each D)
- Differences from Fortran:
  - Reference conditions match: T=253.15K, P=60000Pa, Sutherland viscosity
  - **MH2005 fall speed**: `delta0=5.83, C0=0.6` match Fortran exactly. Stokes fallback at X<1e-5 assumes spherical drag (acceptable for tiny particles)
  - **Rain fall speed uses piecewise Gunn-Kinzer/Beard** (4-regime) vs Fortran's single `V=842*D^0.8`. More physical but produces quantitative differences at small and large D
  - Particle mass: 4-regime branchless `ifelse` cascade matches Fortran `m(D)`. Uses `rho_w=997` for liquid mass in rain fall speed (Fortran uses 1000; 0.3% difference)
  - **Particle area**: regime-4 (partially rimed, D≥D_cr) returns spherical area. Fortran uses aggregate-like blended area for regime 4. May overestimate cross-section of large partially-rimed aggregates
  - Capacitance: `cap=1 → capm=D` convention matches Fortran exactly. Rate equations use `2π×capm` not `4π×C`
  - Ventilation: `a_v=0.65, b_v=0.44` match Fortran. Table stores raw `0.44*sqrt(V*D)` without `Sc^(1/3)/sqrt(nu)` -- applied at runtime
  - `nrwat` (RainCollectionNumber): D≥100μm threshold matches Fortran
  - Aggregation double integral: full O(n²) with factor 1/2 for self-collection matches Fortran upper-triangle convention
  - Reflectivity: `K_refl = 0.1892*(6/(π*917))²` matches Fortran
  - Effective radius normalization uses `ρ_ice=916.7` (matches Fortran `eff` formula, differs from 917 used elsewhere)
  - All 9 sixth-moment integrands implemented; `SixthMomentDeposition` == `SixthMomentSublimation` (same in Fortran; distinction only at runtime)
  - Aerosol collection integrands (`nawcol`, `naicol`) not implemented

### Ice Properties

**`ice_properties.jl`** (120 lines) -- COMPLETE
- Fortran equivalent: No single struct; Fortran stores lookup arrays globally indexed by `iice_t`
- Differences from Fortran:
  - Named, typed sub-containers replace Fortran's integer-indexed table columns
  - `minimum_rime_density=50`, `maximum_rime_density=900` match Fortran clamping
  - `maximum_shape_parameter=10` is vestigial -- not used by any closure (Fortran uses `mu_i_max=6` for two-moment, `mu_i_max=20` for three-moment; both live in their respective closure structs)
  - No aerosol collection sub-containers

**`ice_fall_speed.jl`** (80 lines) -- COMPLETE
- Fortran equivalent: `iice_t(1)=uns`, `iice_t(2)=ums`, `uzs` (3-moment)
- Differences from Fortran:
  - 3 fall speed integrals match Fortran exactly
  - Reference air density `P3_REF_RHO = 60000/(287.15*253.15)` matches Fortran `rhosui`
  - Density correction exponent 0.54 (Heymsfield et al. 2006) applied in quadrature -- matches Fortran
  - **Dead parameters**: stored `fall_speed_coefficient=11.72` and `fall_speed_exponent=0.41` are never used by the quadrature/tabulated path (MH2005 Best-number formulation is used instead)

**`ice_deposition.jl`** (100 lines) -- COMPLETE
- Fortran equivalent: `iice_t(5)=vdep`, `iice_t(10)=vdep1`, `iice_t(15-18)=vdepm1-4`
- Differences from Fortran:
  - All 6 ventilation integrals match Fortran table columns
  - Stored defaults `K_a=0.024`, `D_v=2.2e-5` are superseded at runtime by T,P-dependent `air_transport_properties(T, P)` from `transport_properties.jl`. Defaults only used during tabulation at reference conditions
  - Fortran ventilation constants `a_v=0.65, b_v=0.44` confirmed in quadrature integrands

**`ice_bulk_properties.jl`** (120 lines) -- COMPLETE
- Fortran equivalent: `iice_t(6)=eff`, `iice_t(11)=dmm`, `iice_t(12)=rhomm`, `iice_t(9)=refl`, `iice_t(13-14)=lambda_i/mu_i_save`, `iice_t(19)=qshed`
- Differences from Fortran:
  - All 7 bulk property types match Fortran
  - `SlopeParameter` and `ShapeParameter` are diagnostic (integrand returns zero) -- stored from lambda solver, not integrated
  - `SheddingRate` integrand is simplified (`Fl*m*Np`) vs Fortran's more detailed T-dependent shedding physics. Runtime shedding in `collection_rates.jl` handles the detailed physics separately
  - Diameter bounds: D_max=20mm (Breeze) vs Fortran's tighter bounds

**`ice_sixth_moment.jl`** (100 lines) -- COMPLETE
- Fortran equivalent: 3-moment table entries `m6rime, m6dep, m6dep1, m6mlt1, m6mlt2, m6agg, m6shd, m6sub, m6sub1`
- Differences from Fortran:
  - All 9 sixth-moment integrals implemented with correct formulas
  - `SixthMomentDeposition` and `SixthMomentSublimation` have identical integrands (same in Fortran; distinction matters only at runtime when sign is applied)
  - `SixthMomentAggregation` uses Wisner single-integral approximation (matches Fortran table convention)
  - **Not yet validated** against Fortran 3-moment reference tables

**`ice_lambda_limiter.jl`** (60 lines) -- COMPLETE
- Fortran equivalent: `iice_t(7)=i_qsmall`, `iice_t(8)=i_qlarge`
- Differences from Fortran:
  - Both lambda limiter integrals match Fortran
  - **Different clamping strategy**: Fortran uses tabulated `i_qsmall`/`i_qlarge` to rescale prognostic variables when lambda hits bounds. Breeze clamps lambda directly in `enforce_diameter_bounds` and recomputes N0. Physically equivalent for well-resolved distributions
  - Breeze `D_max=40mm` is more permissive than Fortran's effective upper bound (~500μm for μ=0)

### Collection & Rain

**`ice_collection.jl`** (80 lines) -- COMPLETE
- Fortran equivalent: `eci`, `eri` parameters in `P3_INIT`, table dimensions for `nagg`/`nrwat`
- Differences from Fortran:
  - Runtime riming reads `Eci` from `ProcessRateParameters.cloud_ice_collection_efficiency=0.5` (matches Fortran `eci=0.5`), NOT from this struct's deprecated `ice_cloud_collection_efficiency=0.1`
  - `Eri=1.0` matches Fortran

**`ice_rain_collection.jl`** (60 lines) -- PARTIAL
- Fortran equivalent: Ice-rain collection sub-table `itabcol(i_Qnorm, i_Drscale, i_Fr, i_Fl)`
- Differences from Fortran:
  - **No rain-size binning**: Fortran bins rain into 30 discrete size categories and integrates collection for each. Breeze uses continuous bulk treatment (q_r, N_r). Deliberate architectural difference
  - 3 integral types (mass, number, sixth_moment) vs Fortran's 2D sub-table structure

**`collection_rates.jl`** (510 lines) -- COMPLETE
- Fortran equivalent: Aggregation, cloud riming, rain riming, shedding sections of `P3_MAIN`
- Differences from Fortran:
  - **Aggregation**: T-dependent Eii (cold=0.001, ramp to 0.3 at 273.15K) and rime-fraction limiter Eii_fact match Fortran exactly. Factor 1/2 for self-collection matches upper-triangle convention. Higher qi threshold (1e-8 vs Fortran's 1e-14). ~~BUG (H6)~~ **FIXED**: Added ρ parameter; rate now `ρ * K * n²` matching cloud riming convention
  - **Cloud riming**: `Eci=0.5`, density correction `(rho0_ice/rho)^0.54` match Fortran. Only active T<T0
  - **Above-freezing cloud collection**: `ncshdc = qcshd * 1.923e6` matches Fortran (shed as 1mm drops)
  - **Rain riming**: `Eri=1.0`, no Mizuno gate -- matches Fortran v5.5.0
  - **Rime density**: **Deliberate physics upgrade** -- Cober-List (1993) with Stokes impact parameter replaces Fortran's simpler `max(min(-0.5*T_C + 475, 900), 170)`. Will cause quantitative differences
  - **Shedding and refreezing**: Milbrandt et al. (2025) liquid-fraction extensions NOT in Fortran v5.5.0
  - No D≥100μm threshold in analytical (non-tabulated) riming path; Fortran's `nrwat` table enforces this

**`rain_properties.jl`** (80 lines) -- COMPLETE
- Fortran equivalent: Rain parameters in `P3_INIT`
- Differences from Fortran:
  - `fall_speed_coefficient=842`, `fall_speed_exponent=0.8` match Fortran `ar=842, br=0.8`
  - Docstring error: says "default 4854" but actual is 842
  - Integral types are initially stubs; replaced by `TabulatedFunction1D` when `tabulate(p3, :rain, CPU())` is called

**`rain_process_rates.jl`** (290 lines) -- COMPLETE
- Fortran equivalent: Autoconversion, accretion, self-collection, breakup, evaporation in `P3_MAIN`
- Differences from Fortran:
  - **Autoconversion**: KK2000 `k1*qc^2.47*(Nc/Nc_ref)^(-1.79)` -- matches Fortran exactly
  - **Accretion**: ~~BUG (C1)~~ **FIXED**: Now uses `k2 * (qc * qr)^1.15` matching KK2000 Eq. 5 and Fortran P3
  - **Self-collection**: SB2001 `k_rr=5.78` -- matches Fortran exactly
  - **Breakup**: SB2006 three-piece function with `D_eq=0.9mm, kappa_br=2300` -- matches Fortran. **D_r clamped at 2.5mm** (not in Fortran; prevents extreme rates from numerical transients)
  - **Evaporation (tabulated path)**: Uses `2π*N0*I_evap*(S-1)/Φ` with exponential DSD (μ_r=0); lambda clamped [125, 50000]. Saturation vapor pressure correctly inverted from qvsl. Matches Fortran
  - **Evaporation (mean-mass fallback)**: Uses constant `RAIN_NU=1.5e-5` instead of T,P-dependent nu. Documented as intentional for table consistency, but introduces altitude-dependent bias
  - No μ_r diagnostic (Fortran has empirical μ_r(λ_r) relation). Breeze assumes μ_r=0 throughout, consistent with Fortran for evaporation integral

**`rain_quadrature.jl`** (280 lines) -- COMPLETE
- Fortran equivalent: Rain lookup tables in `P3_INIT`
- Differences from Fortran:
  - **Physics upgrade**: Uses piecewise Gunn-Kinzer/Beard fall speed (4-regime, physically captures rolloff at large D and Stokes at small D) vs Fortran's analytical gamma-function solution for `V=842*D^0.8`
  - 128-point Chebyshev-Gauss quadrature vs Fortran analytical formulas
  - `rho_w=997` in evaluators vs 1000 in `ProcessRateParameters` (0.3% inconsistency)
  - Ventilation: `f1r=0.78, f2r=0.308` (Sc^(1/3) baked into f2r) with `nu=1.5e-5` at reference conditions -- matches Fortran convention
  - Exponential DSD only (μ_r=0) -- matches Fortran for lookup tables

### Process Rates & Physics

**`process_rates.jl`** (1050 lines) -- COMPLETE
- Fortran equivalent: Body of `P3_MAIN` (~lines 2500-4100)
- Differences from Fortran:
  - **Condensation**: Relaxation-to-saturation with psychrometric correction and `τ=1s`. Fortran uses Newton-iteration exact saturation adjustment. Nearly equivalent for small dt; differs for large dt or supersaturation-sensitive processes
  - **Rate application**: All rates computed simultaneously from same state (GPU-friendly). Fortran applies sequentially with intermediate clipping (Lie splitting). Sink limiting approximates Fortran's sequential approach
  - **Sink limiting**: Proportional pre-limiting of cloud/rain/ice/vapor species before tendency application. Fortran clips after application. Uses `sink_limiting_timescale=1s`
  - **Hardcoded latent heats**: `L_s=2.835e6, L_f=3.34e5, L_v=2.5e6` match Fortran but are inconsistent with condensation path (which uses T-dependent `liquid_latent_heat`). Energy budget not exactly closed
  - **Hardcoded gas constants**: `R_v=461.5, R_d=287.0` inline (match Fortran, differ from Breeze thermo module)
  - **Deposition ventilation fallback**: ~~BUG (H7)~~ **FIXED**: Fall speed coefficients now blended with rime fraction, matching `_collection_kernel_per_particle`
  - Transport properties computed once per grid point and shared (same as Fortran)
  - No `clbfact_dep/clbfact_sub` calibration multipliers for deposition/sublimation
  - `::Nothing` fallback methods return `zero(rho)` for each tendency (clean dispatch pattern not in Fortran)

**`ice_nucleation_rates.jl`** (400 lines) -- COMPLETE
- Fortran equivalent: Nucleation (~line 2900), immersion freezing (~line 3050), homogeneous (~line 2760), splintering (~line 3200) in `P3_MAIN`
- Differences from Fortran:
  - **Cooper nucleation**: `c_nuc=5.0/m³, exponent=0.304` -- matches Fortran exactly
  - **Immersion freezing**: `aimm=0.65, bimm=2.0` (Barklie-Gokhale 1959, NOT Bigg 1953) -- matches Fortran. PSD correction factors (Breeze-specific) for both cloud and rain
  - **Immersion T check**: Breeze `269.15K` vs Fortran `269K` (0.15K difference, negligible)
  - **Homogeneous freezing**: Instantaneous at T<233.15K with `τ_hom=1s` -- matches Fortran. **ni-explosion cap** (`N_hom ≤ Q_hom/minimum_cloud_drop_mass`) is Breeze-specific, needed because Nc is prescribed
  - **Hallett-Mossop**: **Gaussian** efficiency profile around T_peak=268.15K (-5C) vs Fortran's **piecewise linear** (zero below -8C and above -3C, ramp to peak at -5C). Same temperature range but different taper near boundaries
  - Contact/condensation-freezing nucleation: NOT in Fortran v5.5.0, NOT in Breeze

**`melting_rates.jl`** (200 lines) -- COMPLETE
- Fortran equivalent: Melting section (~lines 3400-3500) of `P3_MAIN`
- Differences from Fortran:
  - Heat balance equation from MM15a Eq. 44 with `2π×capm` convention -- matches Fortran
  - **Saturation vapor pressure at T0**: Breeze uses constant `e_s0=611 Pa`. Fortran computes `polysvp1(273.15, 0)=611.2 Pa`. 0.03% difference, negligible
  - **Partitioned melting** (Milbrandt et al. 2025): partial (liquid coating) vs complete (shed to rain) based on `maximum_liquid_fraction=0.3`. Fortran v5.5.0 has this when `liqFrac=.true.`. **Docstring error (M14)**: comment at line 152 says "about 14% liquid by mass" but parameter is 0.3 (30%)
  - **Safety limiter**: Breeze uses `max_melt = qi/1s`. Fortran limits melting to available ice per actual dt. Breeze's 1-second timescale is cruder
  - Hardcoded `L_f=3.34e5, L_v=2.5e6, R_v=461.5, e_s0=611` -- match Fortran but should come from shared constants

**`terminal_velocities.jl`** (350 lines) -- COMPLETE
- Fortran equivalent: Fall speed computation in `P3_MAIN`, lookup tables `itabvs/itabvn/itabvz`
- Differences from Fortran:
  - **Rain**: Table dispatch or mean-mass fallback with `ar=842, br=0.8, (ρ₀/ρ)^0.54` -- matches Fortran
  - **Ice**: Table dispatch or analytical fallback with regime-dependent coefficients (unrimed: `a=11.72, b=0.41`; rimed: `a=19.3, b=0.37`; small: Stokes `700*D²`) -- matches Fortran regimes
  - **Analytical mass-weight factor**: `1.787` is correct for `b=0.3966` but applied to ALL regimes including Stokes (`b=2`), where correct factor is `Γ(6)/Γ(4)=30`. ~17x underestimate for D<100μm. With tables active, this is bypassed
  - Number/reflectivity-weighted fallbacks use fixed ratios (0.6 and 1.2 of mass-weighted). Approximate; not used when tables active
  - Ice reference density correctly separated from rain reference density: `rhosui` for ice, `rhosur` for rain
  - Breeze uses hard `ifelse` regime switch vs Fortran's smooth blending through table integration

**`transport_properties.jl`** (60 lines) -- COMPLETE
- Fortran equivalent: `dv, mu, kap, nu` computation (~lines 2250-2280) of `P3_MAIN`
- Differences from Fortran:
  - **Exact match**: `D_v = 8.794e-5*T^1.81/P`, `mu = 1.496e-6*T^1.5/(T+120)`, `K_a = 1414*mu`, `nu = mu*R_d*T/P`
  - `R_d=287.0` hardcoded (comment notes deliberate match to Fortran, not 287.04)
  - Called once per grid point and shared across deposition, melting, and evaporation

**`cloud_droplet_properties.jl`** (50 lines) -- COMPLETE
- Fortran equivalent: Fortran carries prognostic Nc and ssat
- Differences from Fortran:
  - **Prescribed Nc** with default `100e6/m³` (continental). Fortran predicts Nc via external aerosol activation (not built into P3 itself)
  - **No prognostic supersaturation**. Breeze diagnoses saturation from thermodynamic state
  - `condensation_timescale=1s` for relaxation approach (Fortran uses Newton-iteration equilibrium solve)
  - `autoconversion_threshold=25μm` is dead code -- KK2000 has no threshold parameter

**`psd_corrections.jl`** (80 lines) -- COMPLETE
- Fortran equivalent: None -- Breeze-specific
- Differences from Fortran:
  - **Not in Fortran**. Analytical PSD correction `Γ(μ+7)Γ(μ+1)/Γ(μ+4)²` bridges mean-mass approximation to PSD-integrated rates for volume-dependent freezing
  - Exact values: μ=0 gives 20.0; μ=2 gives 5.6
  - Used in `ProcessRateParameters` for cloud and rain freezing PSD corrections

**`multi_ice_category.jl`** (170 lines) -- PARTIAL
- Fortran equivalent: `nCat > 1` with full inter-category collection in `P3_MAIN`
- Differences from Fortran:
  - Framework only: `MultiIceCategory{N, ICE}` container, field name generation, indexing
  - `inter_category_collection` uses heuristic `E*qi1*qi2/τ_agg` (dimensionally suspect placeholder). Fortran computes full PSD-integrated collision kernel between category PSDs
  - Not integrated into `compute_p3_process_rates`
  - No multi-category sedimentation or category-interaction with warm processes

---

## Fortran Lookup Table Column Mapping

| Col | Fortran Name | Julia Type | Container | Status |
|-----|-------------|------------|-----------|--------|
| 1 | uns | `NumberWeightedFallSpeed` | `IceFallSpeed` | DONE |
| 2 | ums | `MassWeightedFallSpeed` | `IceFallSpeed` | DONE |
| 3 | nagg | `AggregationNumber` | `IceCollection` | DONE |
| 4 | nrwat | `RainCollectionNumber` | `IceCollection` | DONE |
| 5 | vdep | `Ventilation` | `IceDeposition` | DONE |
| 6 | eff | `EffectiveRadius` | `IceBulkProperties` | DONE |
| 7 | i_qsmall | `NumberMomentLambdaLimit` | `IceLambdaLimiter` | DONE |
| 8 | i_qlarge | `MassMomentLambdaLimit` | `IceLambdaLimiter` | DONE |
| 9 | refl | `Reflectivity` | `IceBulkProperties` | DONE |
| 10 | vdep1 | `VentilationEnhanced` | `IceDeposition` | DONE |
| 11 | dmm | `MeanDiameter` | `IceBulkProperties` | DONE |
| 12 | rhomm | `MeanDensity` | `IceBulkProperties` | DONE |
| 13 | lambda_i | `SlopeParameter` | `IceBulkProperties` | DONE (diagnostic) |
| 14 | mu_i_save | `ShapeParameter` | `IceBulkProperties` | DONE (diagnostic) |
| 15 | vdepm1 | `SmallIceVentilationConstant` | `IceDeposition` | DONE |
| 16 | vdepm2 | `SmallIceVentilationReynolds` | `IceDeposition` | DONE |
| 17 | vdepm3 | `LargeIceVentilationConstant` | `IceDeposition` | DONE |
| 18 | vdepm4 | `LargeIceVentilationReynolds` | `IceDeposition` | DONE |
| 19 | qshed | `SheddingRate` | `IceBulkProperties` | DONE |
| 20 | nawcol | -- | -- | NOT IMPLEMENTED (aerosol) |
| 21 | naicol | -- | -- | NOT IMPLEMENTED (aerosol) |
| -- | uzs (3-mom) | `ReflectivityWeightedFallSpeed` | `IceFallSpeed` | DONE |
| -- | m6* (3-mom) | 9 `SixthMoment*` types | `IceSixthMoment` | DONE |

---

## Deliberate Differences from Fortran

These are intentional design choices, not bugs:

1. **Prescribed Nc** (not prognostic) -- Fortran carries prognostic cloud droplet number
2. **Relaxation condensation** (not prognostic supersaturation) -- Fortran uses prognostic `ssat`
3. **Cober-List rime density** -- Physics upgrade over Fortran's simple T-dependent formula
4. **Piecewise Gunn-Kinzer rain fall speed** -- More physical than Fortran's single power law
5. **Gaussian Hallett-Mossop** -- Smooth vs Fortran's piecewise linear (same T range)
6. **Sink limiting** -- Proportional pre-limiting vs Fortran's post-application clipping
7. **Concurrent rate computation** -- All rates from same state vs Fortran's sequential Lie splitting
8. **Shedding + refreezing** -- Milbrandt et al. (2025) liquid-fraction extensions
9. **PSD correction factors** -- Breeze-specific analytical bridge for mean-mass approximation
10. **ni-explosion cap** -- Mass-number consistency on homogeneous freezing (needed for prescribed Nc)
11. **Table generation at runtime** -- vs Fortran's binary file I/O

---

## Issue Summary

### CRITICAL Priority (2 issues, 1 fixed)

| # | Issue | File(s) | Impact |
|---|-------|---------|--------|
| ~~C1~~ | ~~**KK2000 accretion exponent placement**~~ | rain_process_rates.jl:64 | **FIXED**: Changed `k₂ * qcl * qr^α` → `k₂ * (qcl * qr)^α` (KK2000 Eq. 5 / Fortran P3). |
| C2 | **Lambda solver div-by-zero** (two-moment) | lambda_solver.jl:790 | `Δx = f₁ * (x₁ - x₀) / (f₁ - f₀)` has no guard for `f₁ == f₀`. The three-moment solver at line 860 correctly checks `abs(denom) < eps(FT)`. Can produce NaN/Inf lambda → cascading NaN in all ice properties. |

### HIGH Priority (7 issues, 3 fixed)

| # | Issue | File(s) | Impact |
|---|-------|---------|--------|
| H1 | **Hardcoded latent heats** inconsistent with Breeze thermodynamics | process_rates.jl, melting_rates.jl | Energy budget not exactly closed across condensation (T-dependent L) and ice processes (constant L) |
| H2 | **Missing rime_density axis** in lookup tables | tabulation.jl | Tables assume rho_r=400 for all rime densities; 10-30% error for dense graupel (rho_r=900). Documented; TabulatedFunction4D deferred to future PR. |
| ~~H3~~ | ~~**Analytical ice fall speed mass-weight factor** wrong for Stokes regime~~ | terminal_velocities.jl | **FIXED**: Regime-dependent factor — Γ(6)/Γ(4)=20 for Stokes, 1.787 for large particles |
| H4 | **VentilationEnhanced table convention** requires Sc/nu correction at runtime | quadrature.jl -> process_rates.jl | Table stores raw 0.44*sqrt(V*D); process_rates.jl must multiply by Sc^(1/3)/sqrt(nu). Cross-file coupling documented but fragile. |
| H5 | **Sedimentation sub-stepping absent** | p3_interface.jl | Production LES with large dt needs sub-stepping for CFL stability with fast-falling graupel |
| ~~H6~~ | ~~**Aggregation missing ρ factor**~~ | collection_rates.jl:82 | **FIXED**: Added ρ parameter to `ice_aggregation_rate`; rate now `ρ * K * n²` matching cloud riming convention. |
| ~~H7~~ | ~~**Deposition ventilation uses unrimed-only fall speed**~~ | process_rates.jl:104 | **FIXED**: Fall speed coefficients now blended with rime fraction, matching `_collection_kernel_per_particle`. |

### MEDIUM Priority (14 issues, 7 fixed/documented)

| # | Issue | File(s) |
|---|-------|---------|
| M1 | Particle area treats partially rimed (regime 4) as spherical instead of aggregate-like | quadrature.jl |
| M2 | No ice-rain sub-table with rain diameter dimension (structural difference from Fortran) | tabulation.jl |
| ~~M3~~ | ~~Duplicated regime threshold code with different eps guards~~ | quadrature.jl + lambda_solver.jl | **FIXED**: Harmonized eps guards; cross-reference comments added. |
| M4 | Sixth moment integrands unvalidated against Fortran | quadrature.jl |
| M5 | Hallett-Mossop Gaussian vs Fortran piecewise linear | ice_nucleation_rates.jl |
| M6 | Fixed e_s0=611 Pa in melting (should derive from thermodynamics) | melting_rates.jl |
| M7 | Sign convention mixing in tendency assembly (maintenance risk) | process_rates.jl |
| ~~M8~~ | ~~Breakup D_r clamped at 2.5mm (not in Fortran)~~ | rain_process_rates.jl | **DOCUMENTED**: Fortran divergence noted in code comment. |
| ~~M9~~ | ~~No D>=100um threshold in analytical (non-tabulated) riming path~~ | process_rates.jl | **FIXED**: Added D >= 100 μm threshold matching Fortran nrwat convention. |
| ~~M10~~ | ~~`inter_category_collection` dimensionally suspect (placeholder)~~ | multi_ice_category.jl | **DOCUMENTED**: Marked as PLACEHOLDER with dimensional analysis warning. |
| ~~M11~~ | ~~**qʷⁱ sinks not sink-limited**~~ | process_rates.jl | **FIXED**: Added shedding+refreezing sink limiting for qʷⁱ alongside existing species limiters. |
| ~~M12~~ | ~~**SixthMomentAggregation uses single integral**~~ | quadrature.jl | **DOCUMENTED**: Wisner approximation warning added; matches Fortran convention. |
| ~~M13~~ | ~~**Rain V(D) inconsistency** between tabulated and analytical paths~~ | rain_quadrature.jl, terminal_velocities.jl | **DOCUMENTED**: Cross-reference comments added in both files. |
| ~~M14~~ | ~~**Melting docstring says 14% but parameter is 30%**~~ | melting_rates.jl:152 | **FIXED**: Comment corrected to 30% matching `maximum_liquid_fraction=0.3`. |

### LOW Priority (8 issues, 7 fixed/documented)

| # | Issue | File(s) |
|---|-------|---------|
| ~~L1~~ | ~~Dead `fall_speed_coefficient/exponent` in IceFallSpeed~~ | ice_fall_speed.jl | **FIXED**: Removed dead fields from struct, constructor, Adapt, and tests |
| ~~L2~~ | ~~Deprecated `ice_cloud_collection_efficiency` defaults to 0.1~~ | ice_collection.jl | **FIXED**: Default changed to 0.5 (matches Fortran `eci`) |
| ~~L3~~ | ~~rho_w=997 in rain_quadrature.jl vs 1000 elsewhere~~ | rain_quadrature.jl, quadrature.jl | **FIXED**: Unified to 1000 kg/m³ |
| ~~L4~~ | ~~Docstring says "default 4854" but actual is 842~~ | rain_properties.jl | **FIXED**: Docstring corrected |
| L5 | Vestigial `maximum_shape_parameter` in IceProperties | ice_properties.jl |
| ~~L6~~ | ~~Dead `autoconversion_threshold` parameter~~ | cloud_droplet_properties.jl | **DOCUMENTED**: Marked DEPRECATED in struct comment and docstring. |
| ~~L7~~ | ~~R_d=287.0 hardcoded~~ | transport_properties.jl, process_rates.jl, rain_process_rates.jl | **DOCUMENTED**: Fortran-parity comments added to all occurrences. |
| ~~L8~~ | ~~rho_ice=916.7 in effective radius vs 917 elsewhere~~ | tabulation.jl | **FIXED**: Unified to 917 kg/m³ |

---

## Parameter Verification Summary

### Collection Efficiencies

| Parameter | Fortran | Breeze | Match |
|-----------|---------|--------|-------|
| eci (cloud-ice) | 0.5 | 0.5 | YES |
| eri (rain-ice) | 1.0 | 1.0 | YES |
| Eii_cold (T<253.15K) | 0.001 | 0.001 | YES |
| Eii_max (T>=273.15K) | 0.3 | 0.3 | YES |
| Eii_fact (Fr<0.6) | 1.0 | 1.0 | YES |
| Eii_fact (0.6<=Fr<=0.9) | ramp to 0 | ramp to 0 | YES |

### Nucleation Parameters

| Parameter | Fortran | Breeze | Match |
|-----------|---------|--------|-------|
| c_nuc (Cooper) | 5.0 /m3 | 5.0 /m3 | YES |
| Cooper exponent | 0.304 | 0.304 | YES |
| aimm (immersion) | 0.65 | 0.65 | YES |
| bimm (immersion) | 2.0 | 2.0 | YES |
| T_hom | 233.15 K | 233.15 K | YES |
| mi0 | 4pi/3*900*(1e-6)^3 | 4pi/3*900*(1e-6)^3 | YES |

### Rain Parameters

| Parameter | Fortran | Breeze | Match |
|-----------|---------|--------|-------|
| ar | 842 | 842 | YES |
| br | 0.8 | 0.8 | YES |
| f1r | 0.78 | 0.78 | YES |
| f2r | 0.308 | 0.308 | YES |
| k_rr (self-collection) | 5.78 | 5.78 | YES |
| D_eq (breakup) | 0.9 mm | 0.9 mm | YES |
| kappa_br | 2300 /m | 2300 /m | YES |

### Reference Densities

| Parameter | Fortran | Breeze | Match |
|-----------|---------|--------|-------|
| rhosur (rain) | 80000/(287.15*273.15) | 80000/(287.15*273.15) | YES |
| rhosui (ice) | 60000/(287.15*253.15) | 60000/(287.15*253.15) | YES |

### Mu-Lambda Closure

| Aspect | Fortran | Breeze | Match |
|--------|---------|--------|-------|
| Field et al. coefficients | 0.076*(lam/100)^0.8 - 2 | 0.00191*lam^0.8 - 2 | YES (algebraically identical) |
| Mu clamp (small) | [0, 6] | [0, 6] | YES |
| Mu clamp (large) | [0, 20] | [0, 20] | YES |
| D_mvd threshold | 0.2 mm | 0.2 mm | YES |

---

## What Fortran Has That Breeze Does Not

1. **Prognostic Nc and ssat** -- Breeze prescribes Nc, diagnoses saturation
2. **SCPF** (sub-column precipitation fraction) -- `scpf_on=.false.` in Fortran ref too
3. **Aerosol collection** (nawcol, naicol) -- columns 20-21, aerosol-aware mode
4. **Type diagnostics** (qi_type: cloud ice/snow/graupel/hail classification)
5. **Precipitation rate diagnostics** by hydrometeor type
6. **Effective radius diagnostics** (diag_reffc, diag_reffi)
7. **Rime density table dimension** (5 rho_r values vs fixed 400)
8. **Ice-rain binned sub-table** (rain size dimension)
9. **Multi-category ice interactions** (nCat > 1, full inter-category collection)
10. **Sequential rate application** with intermediate clipping (Lie splitting)

## What Breeze Has That Fortran Does Not

1. **Mass-number consistency cap** on homogeneous freezing (ni-explosion limiter)
2. **PSD correction factors** for immersion freezing (analytical Gamma approach)
3. **Cober-List (1993) rime density** (physics upgrade)
4. **Piecewise Gunn-Kinzer rain fall speed** (physics upgrade)
5. **Partitioned melting with liquid fraction** (Milbrandt et al. 2025)
6. **Shedding and refreezing** (liquid-fraction extensions)
7. **Table dispatch via Julia's type system** (zero-cost GPU abstraction)
8. **Sink limiting** (proportional pre-limiting of all species)
9. **T,P-dependent transport** in all processes (shared computation)
10. **Runtime table generation** via quadrature (no binary I/O)

---

## Remaining Work (Priority Order)

1. **Fix lambda solver div-by-zero (C2)** -- Add `abs(f₁ - f₀) < eps(FT) && return x₁` guard in `lambda_solver.jl:790`, matching three-moment solver pattern at line 860.
2. **Sedimentation sub-stepping (H5)** -- Required for production LES with large dt
3. **Thread constants** for consistent latent heats (H1)
4. **Add rime_density table dimension** (H2) -- Currently fixed at 400 kg/m³; need TabulatedFunction4D with 5 rime density values (50, 250, 450, 650, 900 kg/m³) to match Fortran
5. **3D LES validation cases** (BOMEX+ice, deep convection)
6. **Validate sixth moment integrals** against Fortran 3-moment tables (M4)
7. **Fix regime-4 particle area** (M1)
8. **Prognostic Nc** (future: removes need for ni-explosion cap)
9. **Multi-ice category integration** into process rates
10. **Radar reflectivity diagnostics**
