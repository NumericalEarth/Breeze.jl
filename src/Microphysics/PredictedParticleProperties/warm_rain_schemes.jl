#####
##### Warm-rain (autoconversion / accretion / self-collection) schemes
#####
##### Mirrors the three options of Fortran P3 v5.5.0 `autoAccr_param`:
#####   1 → Seifert and Beheng (2001)
#####   2 → Khairoutdinov and Kogan (2000) [Breeze default]
#####   3 → Kogan (2013)
#####
##### A scheme is selected by setting the `warm_rain_scheme` field on the
##### top-level [`PredictedParticlePropertiesMicrophysics`](@ref) struct and
##### dispatched on inside the rain-process rate functions.
#####

"""
$(TYPEDEF)

Abstract supertype for warm-rain parameterizations (autoconversion, accretion,
rain self-collection, cloud self-collection) used by P3.

Concrete subtypes:
- [`KhairoutdinovKogan2000`](@ref) (default)
- [`SeifertBeheng2001`](@ref)
- [`Kogan2013`](@ref)
"""
abstract type AbstractWarmRainScheme end

"""
$(TYPEDEF)

[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000) warm-rain
parameterization. Cloud self-collection is zero by Fortran convention here
(only [`SeifertBeheng2001`](@ref) carries an explicit cloud self-collection term).

!!! note "Subgrid fraction factors"
    Breeze applies all warm-rain rates to grid-mean state; Fortran P3 scales
    by in-cloud / in-precipitation fractions (`iSCF`, `iSPF`, `SPF-SPF_clr`).
    Without subgrid cloud/precip fraction prognostics in Breeze, those factors
    are dropped (equivalent to `SCF = SPF = 1`, `SPF_clr = 0`).
"""
struct KhairoutdinovKogan2000 <: AbstractWarmRainScheme end

"""
$(TYPEDEF)

[Kogan (2013)](@cite Kogan2013) warm-rain parameterization. Cloud self-collection is zero
by Fortran convention here (only [`SeifertBeheng2001`](@ref) carries an
explicit cloud self-collection term). Subgrid cloud / precipitation fraction
factors are dropped — see [`KhairoutdinovKogan2000`](@ref) for details.
"""
struct Kogan2013 <: AbstractWarmRainScheme end

"""
$(TYPEDEF)

[Seifert and Beheng (2001)](@cite SeifertBeheng2001) warm-rain parameterization.

`ν = nothing` (default) matches Fortran P3 by deriving the cloud droplet
mass-distribution shape parameter dynamically from `μ_c` via the 16-entry
`dnu` lookup. Pass an explicit `ν` only for controlled sensitivity tests.

Unlike [`KhairoutdinovKogan2000`](@ref) and [`Kogan2013`](@ref), this scheme
contributes an explicit cloud self-collection number sink (Fortran `ncslf`).
Subgrid cloud / precipitation fraction factors are dropped — see
[`KhairoutdinovKogan2000`](@ref) for details.
"""
struct SeifertBeheng2001{V} <: AbstractWarmRainScheme
    ν :: V
end

SeifertBeheng2001() = SeifertBeheng2001{Nothing}(nothing)
SeifertBeheng2001(ν::Real) = SeifertBeheng2001{typeof(float(ν))}(float(ν))
