#####
##### Warm-rain (autoconversion / accretion / self-collection) schemes
#####
##### Breeze implements Khairoutdinov and Kogan (2000).
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
parameterization. Cloud self-collection is zero in this scheme.

!!! note "Subgrid fraction factors"
    Breeze applies all warm-rain rates to the grid-mean state. A subgrid
    formulation would instead scale them by in-cloud and in-precipitation
    fractions; without those prognostics the factors are dropped, which is
    equivalent to `SCF = SPF = 1` and `SPF_clr = 0`.
"""
struct KhairoutdinovKogan2000 <: AbstractWarmRainScheme end
