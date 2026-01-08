#####
##### P3 Integral Types
#####
##### This file defines abstract and concrete types for the 29 ice integrals
##### plus rain integrals used in the P3 microphysics scheme.
#####
##### Each integral type can be evaluated via quadrature or tabulated for efficiency.
##### The type hierarchy groups integrals by physical concept.
#####

#####
##### Abstract hierarchy
#####

"""
    AbstractP3Integral

Abstract supertype for all P3 scheme integrals over particle size distributions.
"""
abstract type AbstractP3Integral end

"""
    AbstractIceIntegral <: AbstractP3Integral

Abstract supertype for ice particle integrals.
"""
abstract type AbstractIceIntegral <: AbstractP3Integral end

"""
    AbstractRainIntegral <: AbstractP3Integral

Abstract supertype for rain particle integrals.
"""
abstract type AbstractRainIntegral <: AbstractP3Integral end

#####
##### Ice integral categories (29 total)
#####

# Fall speed integrals (3)
abstract type AbstractFallSpeedIntegral <: AbstractIceIntegral end

# Deposition/ventilation integrals (6)
abstract type AbstractDepositionIntegral <: AbstractIceIntegral end

# Bulk property integrals (7)
abstract type AbstractBulkPropertyIntegral <: AbstractIceIntegral end

# Collection integrals (2)
abstract type AbstractCollectionIntegral <: AbstractIceIntegral end

# Sixth moment integrals (9)
abstract type AbstractSixthMomentIntegral <: AbstractIceIntegral end

# Lambda limiter integrals (2)
abstract type AbstractLambdaLimiterIntegral <: AbstractIceIntegral end

#####
##### Fall speed integrals (3)
#####
##### uns, ums, uzs in Fortran
#####

"""
    NumberWeightedFallSpeed <: AbstractFallSpeedIntegral

Number-weighted mean fall speed:

```math
V_n = \\frac{\\int_0^\\infty V(D) N'(D) \\, dD}{\\int_0^\\infty N'(D) \\, dD}
```

Corresponds to `uns` in P3 Fortran code.
"""
struct NumberWeightedFallSpeed <: AbstractFallSpeedIntegral end

"""
    MassWeightedFallSpeed <: AbstractFallSpeedIntegral

Mass-weighted mean fall speed:

```math
V_m = \\frac{\\int_0^\\infty V(D) m(D) N'(D) \\, dD}{\\int_0^\\infty m(D) N'(D) \\, dD}
```

Corresponds to `ums` in P3 Fortran code.
"""
struct MassWeightedFallSpeed <: AbstractFallSpeedIntegral end

"""
    ReflectivityWeightedFallSpeed <: AbstractFallSpeedIntegral

Reflectivity-weighted (6th moment) mean fall speed for 3-moment ice:

```math
V_z = \\frac{\\int_0^\\infty V(D) D^6 N'(D) \\, dD}{\\int_0^\\infty D^6 N'(D) \\, dD}
```

Corresponds to `uzs` in P3 Fortran code.
"""
struct ReflectivityWeightedFallSpeed <: AbstractFallSpeedIntegral end

#####
##### Deposition/ventilation integrals (6)
#####
##### vdep, vdep1, vdepm1, vdepm2, vdepm3, vdepm4 in Fortran
#####

"""
    Ventilation <: AbstractDepositionIntegral

Basic ventilation factor for vapor diffusion.
Corresponds to `vdep` in P3 Fortran code.
"""
struct Ventilation <: AbstractDepositionIntegral end

"""
    VentilationEnhanced <: AbstractDepositionIntegral

Enhanced ventilation factor for particles > 100 μm.
Corresponds to `vdep1` in P3 Fortran code.
"""
struct VentilationEnhanced <: AbstractDepositionIntegral end

"""
    SmallIceVentilationConstant <: AbstractDepositionIntegral

Ventilation for small ice (D ≤ D_crit), constant term.
Melted water from these particles transfers to rain.
Corresponds to `vdepm1` in P3 Fortran code.
"""
struct SmallIceVentilationConstant <: AbstractDepositionIntegral end

"""
    SmallIceVentilationReynolds <: AbstractDepositionIntegral

Ventilation for small ice (D ≤ D_crit), Reynolds-dependent term.
Melted water from these particles transfers to rain.
Corresponds to `vdepm2` in P3 Fortran code.
"""
struct SmallIceVentilationReynolds <: AbstractDepositionIntegral end

"""
    LargeIceVentilationConstant <: AbstractDepositionIntegral

Ventilation for large ice (D > D_crit), constant term.
Melted water from these particles accumulates as liquid on ice.
Corresponds to `vdepm3` in P3 Fortran code.
"""
struct LargeIceVentilationConstant <: AbstractDepositionIntegral end

"""
    LargeIceVentilationReynolds <: AbstractDepositionIntegral

Ventilation for large ice (D > D_crit), Reynolds-dependent term.
Melted water from these particles accumulates as liquid on ice.
Corresponds to `vdepm4` in P3 Fortran code.
"""
struct LargeIceVentilationReynolds <: AbstractDepositionIntegral end

#####
##### Bulk property integrals (7)
#####
##### eff, dmm, rhomm, refl, lambda_i, mu_i_save, qshed in Fortran
#####

"""
    EffectiveRadius <: AbstractBulkPropertyIntegral

Effective radius for radiative calculations.
Corresponds to `eff` in P3 Fortran code.
"""
struct EffectiveRadius <: AbstractBulkPropertyIntegral end

"""
    MeanDiameter <: AbstractBulkPropertyIntegral

Mass-weighted mean diameter.
Corresponds to `dmm` in P3 Fortran code.
"""
struct MeanDiameter <: AbstractBulkPropertyIntegral end

"""
    MeanDensity <: AbstractBulkPropertyIntegral

Mass-weighted mean particle density.
Corresponds to `rhomm` in P3 Fortran code.
"""
struct MeanDensity <: AbstractBulkPropertyIntegral end

"""
    Reflectivity <: AbstractBulkPropertyIntegral

Radar reflectivity factor (6th moment of size distribution).
Corresponds to `refl` in P3 Fortran code.
"""
struct Reflectivity <: AbstractBulkPropertyIntegral end

"""
    SlopeParameter <: AbstractBulkPropertyIntegral

Slope parameter λ of the gamma size distribution.
Corresponds to `lambda_i` in P3 Fortran code.
"""
struct SlopeParameter <: AbstractBulkPropertyIntegral end

"""
    ShapeParameter <: AbstractBulkPropertyIntegral

Shape parameter μ of the gamma size distribution.
Corresponds to `mu_i_save` in P3 Fortran code.
"""
struct ShapeParameter <: AbstractBulkPropertyIntegral end

"""
    SheddingRate <: AbstractBulkPropertyIntegral

Rate of meltwater shedding from ice particles.
Corresponds to `qshed` in P3 Fortran code.
"""
struct SheddingRate <: AbstractBulkPropertyIntegral end

#####
##### Collection integrals (2)
#####
##### nagg, nrwat in Fortran
#####

"""
    AggregationNumber <: AbstractCollectionIntegral

Number tendency from ice-ice aggregation.
Corresponds to `nagg` in P3 Fortran code.
"""
struct AggregationNumber <: AbstractCollectionIntegral end

"""
    RainCollectionNumber <: AbstractCollectionIntegral

Number tendency from rain collection by ice.
Corresponds to `nrwat` in P3 Fortran code.
"""
struct RainCollectionNumber <: AbstractCollectionIntegral end

#####
##### Sixth moment integrals (9)
#####
##### m6rime, m6dep, m6dep1, m6mlt1, m6mlt2, m6agg, m6shd, m6sub, m6sub1 in Fortran
#####

"""
    SixthMomentRime <: AbstractSixthMomentIntegral

Sixth moment tendency from riming.
Corresponds to `m6rime` in P3 Fortran code.
"""
struct SixthMomentRime <: AbstractSixthMomentIntegral end

"""
    SixthMomentDeposition <: AbstractSixthMomentIntegral

Sixth moment tendency from vapor deposition.
Corresponds to `m6dep` in P3 Fortran code.
"""
struct SixthMomentDeposition <: AbstractSixthMomentIntegral end

"""
    SixthMomentDeposition1 <: AbstractSixthMomentIntegral

Sixth moment tendency from vapor deposition (enhanced ventilation).
Corresponds to `m6dep1` in P3 Fortran code.
"""
struct SixthMomentDeposition1 <: AbstractSixthMomentIntegral end

"""
    SixthMomentMelt1 <: AbstractSixthMomentIntegral

Sixth moment tendency from melting (term 1).
Corresponds to `m6mlt1` in P3 Fortran code.
"""
struct SixthMomentMelt1 <: AbstractSixthMomentIntegral end

"""
    SixthMomentMelt2 <: AbstractSixthMomentIntegral

Sixth moment tendency from melting (term 2).
Corresponds to `m6mlt2` in P3 Fortran code.
"""
struct SixthMomentMelt2 <: AbstractSixthMomentIntegral end

"""
    SixthMomentAggregation <: AbstractSixthMomentIntegral

Sixth moment tendency from aggregation.
Corresponds to `m6agg` in P3 Fortran code.
"""
struct SixthMomentAggregation <: AbstractSixthMomentIntegral end

"""
    SixthMomentShedding <: AbstractSixthMomentIntegral

Sixth moment tendency from meltwater shedding.
Corresponds to `m6shd` in P3 Fortran code.
"""
struct SixthMomentShedding <: AbstractSixthMomentIntegral end

"""
    SixthMomentSublimation <: AbstractSixthMomentIntegral

Sixth moment tendency from sublimation.
Corresponds to `m6sub` in P3 Fortran code.
"""
struct SixthMomentSublimation <: AbstractSixthMomentIntegral end

"""
    SixthMomentSublimation1 <: AbstractSixthMomentIntegral

Sixth moment tendency from sublimation (enhanced ventilation).
Corresponds to `m6sub1` in P3 Fortran code.
"""
struct SixthMomentSublimation1 <: AbstractSixthMomentIntegral end

#####
##### Lambda limiter integrals (2)
#####
##### i_qsmall, i_qlarge in Fortran
#####

"""
    SmallQLambdaLimit <: AbstractLambdaLimiterIntegral

Lambda limiter for small ice mass mixing ratios.
Corresponds to `i_qsmall` in P3 Fortran code.
"""
struct SmallQLambdaLimit <: AbstractLambdaLimiterIntegral end

"""
    LargeQLambdaLimit <: AbstractLambdaLimiterIntegral

Lambda limiter for large ice mass mixing ratios.
Corresponds to `i_qlarge` in P3 Fortran code.
"""
struct LargeQLambdaLimit <: AbstractLambdaLimiterIntegral end

#####
##### Rain integrals
#####

"""
    RainShapeParameter <: AbstractRainIntegral

Shape parameter μ_r for rain gamma distribution.
"""
struct RainShapeParameter <: AbstractRainIntegral end

"""
    RainVelocityNumber <: AbstractRainIntegral

Number-weighted rain fall speed.
"""
struct RainVelocityNumber <: AbstractRainIntegral end

"""
    RainVelocityMass <: AbstractRainIntegral

Mass-weighted rain fall speed.
"""
struct RainVelocityMass <: AbstractRainIntegral end

"""
    RainEvaporation <: AbstractRainIntegral

Rain evaporation rate integral.
"""
struct RainEvaporation <: AbstractRainIntegral end

#####
##### Ice-rain collection integrals (3 per rain size bin)
#####

"""
    IceRainMassCollection <: AbstractIceIntegral

Mass collection rate for ice collecting rain.
"""
struct IceRainMassCollection <: AbstractIceIntegral end

"""
    IceRainNumberCollection <: AbstractIceIntegral

Number collection rate for ice collecting rain.
"""
struct IceRainNumberCollection <: AbstractIceIntegral end

"""
    IceRainSixthMomentCollection <: AbstractIceIntegral

Sixth moment collection rate for ice collecting rain (3-moment).
"""
struct IceRainSixthMomentCollection <: AbstractIceIntegral end

#####
##### Tabulated integral wrapper
#####

"""
    TabulatedIntegral{A}

A tabulated (precomputed) version of an integral stored as an array.
Used for efficient lookup during simulation.

# Fields
- `data`: Array containing tabulated integral values indexed by
  normalized ice mass, rime fraction, and liquid fraction.
"""
struct TabulatedIntegral{A}
    data :: A
end

# Allow indexing into tabulated integrals
Base.getindex(t::TabulatedIntegral, args...) = getindex(t.data, args...)
Base.size(t::TabulatedIntegral) = size(t.data)

