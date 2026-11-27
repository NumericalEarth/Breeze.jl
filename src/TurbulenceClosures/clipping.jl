"""
$(TYPEDEF)

Impose the bounds of an assumed PDF with smooth functions of smoothing length `width × scale`, so that the bounded
quantities have continuous derivatives in the state and in the coefficients.
"""
struct SmoothClipping{FT} <: AbstractSubgridDistributionComponent
    width :: FT
end

check_smoothing_width(width) =
    width > 0 || throw(ArgumentError("The width of SmoothClipping must be positive, but width = $width."))

SmoothClipping(width::AbstractFloat) = (check_smoothing_width(width); SmoothClipping{typeof(width)}(width))
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SmoothClipping{FT}` with the smoothing `width`.
"""
SmoothClipping(FT::DataType = Oceananigans.defaults.FloatType; width = 1e-2) = SmoothClipping(convert(FT, width))

Adapt.adapt_structure(to, policy::SmoothClipping) = (width = adapt(to, policy.width); SmoothClipping{typeof(width)}(width))

@inline smooth_maximum(a, b, δ) = (a + b + sqrt((a - b) * (a - b) + δ * δ)) / 2

@inline function smooth_clamp(x, x⁻, x⁺, δ)
    d⁻ = x - x⁻
    d⁺ = x - x⁺
    t = (d⁻ + d⁺) / (hypotenuse(d⁻, δ) + hypotenuse(d⁺, δ))
    return (x⁻ * (1 - t) + x⁺ * (1 + t)) / 2
end

@inline clipped_maximum(policy::SmoothClipping, a, b, scale) = smooth_maximum(a, b, policy.width * scale)

@inline clip_between(policy::SmoothClipping, x, lower_bound, upper_bound, scale) =
    smooth_clamp(x, lower_bound, upper_bound, policy.width * scale)

#####
##### The square root of a quantity that is not negative but for round-off
#####

@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))
