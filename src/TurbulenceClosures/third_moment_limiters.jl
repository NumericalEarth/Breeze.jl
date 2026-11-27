#####
##### Stratification number bound: the stratification number of the third-moment diagnosis
#####

abstract type AbstractStratificationNumberBound <: AbstractSubgridDistributionComponent end
"""
$(TYPEDEF)

The stratification number of the third-moment diagnosis, saturating in stable air toward `pole_fraction` times its
first pole along a rational function.
"""
struct RationalSaturatedStratificationNumber{FT} <: AbstractStratificationNumberBound
    pole_fraction :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `RationalSaturatedStratificationNumber{FT}` with `pole_fraction = 1/2`.
"""
RationalSaturatedStratificationNumber(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(RationalSaturatedStratificationNumber, FT, (; pole_fraction = 0.5), kw)

Adapt.@adapt_structure RationalSaturatedStratificationNumber

@inline function bound_stratification_number(bound::RationalSaturatedStratificationNumber, αᴺ, first_pole)
    αᴺᵐᵃˣ = bound.pole_fraction * first_pole
    return αᴺ / (1 + max(αᴺ, zero(αᴺ)) / αᴺᵐᵃˣ)
end

#####
##### Skewness: how the third moment is kept within its bound
#####

abstract type AbstractSkewnessLimiter <: AbstractSubgridDistributionComponent end

const default_maximum_skewness = 1.2 * sqrt(2)

@inline third_moment_bound(skewness::AbstractSkewnessLimiter, variance) =
    skewness.maximum_skewness * (variance * zero_safe_sqrt(variance))
"""
$(TYPEDEF)

The limiter of the third moment: it saturates toward `±maximum_skewness (w′²)^{3/2}` along an odd quintic polynomial.
"""
struct PolynomialSaturatedSkewness{FT} <: AbstractSkewnessLimiter
    maximum_skewness :: FT
end
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `PolynomialSaturatedSkewness{FT}` with the bound of E3SM `shoc.F90`,
`maximum_skewness = 1.2 √2`.
"""
PolynomialSaturatedSkewness(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(PolynomialSaturatedSkewness, FT, (; maximum_skewness = default_maximum_skewness), kw)

Adapt.@adapt_structure PolynomialSaturatedSkewness

@inline function saturating_quintic(u)
    FT = typeof(u)
    u² = u * u
    direct = u * (15 - u² * (10 - 3 * u²)) / 8
    v = 1 - abs(u)
    complement = v * v * v * (8 + 9 * abs(u) + 3 * u²) / 8
    return ifelse(abs(u) > convert(FT, 1//2), sign(u) * (1 - complement), direct)
end

@inline function limit_third_moment(skewness::PolynomialSaturatedSkewness, variance, w′³)
    FT = typeof(w′³)
    bound = third_moment_bound(skewness, variance)
    reach = (convert(FT, 15) / 8) * bound # the magnitude at which the bound is reached
    bounded = reach > 0
    u = clamp(w′³ / ifelse(bounded, reach, one(FT)), -one(FT), one(FT))
    return ifelse(bounded, bound * saturating_quintic(u), zero(FT))
end
