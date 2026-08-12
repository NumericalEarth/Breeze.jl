#####
##### Warm-rain (autoconversion / accretion / self-collection) schemes
#####
##### Breeze implements option 2 of Fortran P3 v5.5.0 `autoAccr_param`:
#####   2 → Khairoutdinov and Kogan (2000)
#####
##### The scheme is selected by setting the `warm_rain_scheme` field on the
##### top-level [`PredictedParticlePropertiesMicrophysics`](@ref) struct and
##### dispatched on inside the rain-process rate functions.
#####

"""
$(TYPEDEF)

Abstract supertype for warm-rain parameterizations (autoconversion, accretion,
rain self-collection, cloud self-collection) used by P3.

Concrete subtypes:
- [`KhairoutdinovKogan2000`](@ref) (default)
"""
abstract type AbstractWarmRainScheme end

"""
$(TYPEDEF)

[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000) warm-rain
parameterization. Cloud self-collection is zero, by Fortran convention for
this scheme.

!!! note "Subgrid fraction factors"
    Breeze applies all warm-rain rates to grid-mean state; Fortran P3 scales
    by in-cloud / in-precipitation fractions (`iSCF`, `iSPF`, `SPF-SPF_clr`).
    Without subgrid cloud/precip fraction prognostics in Breeze, those factors
    are dropped (equivalent to `SCF = SPF = 1`, `SPF_clr = 0`).
"""
struct KhairoutdinovKogan2000 <: AbstractWarmRainScheme end
