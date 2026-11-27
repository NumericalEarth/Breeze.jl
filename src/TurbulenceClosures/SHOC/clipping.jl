#####
##### Smooth bounds: how the bounds of the assumed PDFs are imposed
#####
##### The bounds of an assumed PDF that the state can reach (of the weight of the updraft component, of the
##### correlations, of the within-component variances, the least standard deviation of a scalar) go through the
##### primitives below, which take the `SmoothClipping` of the PDF and a `scale`: a positive magnitude characteristic
##### of the bounded quantity, a parameter or a literal.
#####

"""
$(TYPEDEF)

Impose the bounds of an assumed PDF with smooth functions, so that the bounded quantities have continuous
derivatives of every order in the state and in the coefficients. With `d = a - b` and the smoothing length
`δ = width × scale`, where `scale` is the magnitude each call site gives for its quantity,

    smooth_maximum(a, b) = (a + b + √(d² + δ²)) / 2 ≥ max(a, b),

and the two-sided bound of `x` between `x⁻` and `x⁺` is

    x⁻ (1 - t) / 2 + x⁺ (1 + t) / 2,    t = ((x - x⁻) + (x - x⁺)) / (√((x - x⁻)² + δ²) + √((x - x⁺)² + δ²)),

which is `(x⁻ + x⁺ + √((x - x⁻)² + δ²) - √((x - x⁺)² + δ²)) / 2` without the difference of the square roots. Each
square root is no smaller than the magnitude of the difference under it, so `|t| ≤ 1` in floating point too and the
bounded value is a weighted mean of `x⁻` and `x⁺`: between the bounds by construction. The price is a bias of
`δ / 2` at a bound that decays as `δ² / (4 |d|)` away from it.

$(TYPEDFIELDS)
"""
struct SmoothClipping{FT} <: AbstractSubgridDistributionComponent
    "The smoothing length of a bound over the scale of the bounded quantity; positive"
    width :: FT
end

check_smoothing_width(width) =
    width > 0 || throw(ArgumentError("The width of SmoothClipping must be positive, but width = $width."))

# A checked width; a traced scalar (which cannot be tested) takes the unchecked default constructor
SmoothClipping(width::AbstractFloat) = (check_smoothing_width(width); SmoothClipping{typeof(width)}(width))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SmoothClipping{FT}` with the smoothing `width`.
"""
SmoothClipping(FT::DataType = Oceananigans.defaults.FloatType; width = 1e-2) = SmoothClipping(convert(FT, width))

# `adapt` rebuilds the policy with the unchecked constructor: it is called inside kernels too (where
# `getclosure` rebuilds the closure of a column), and a check there formats a message, which a GPU
# kernel cannot compile.
Adapt.adapt_structure(to, policy::SmoothClipping) = (width = adapt(to, policy.width); SmoothClipping{typeof(width)}(width))

@inline smooth_maximum(a, b, δ) = (a + b + sqrt((a - b) * (a - b) + δ * δ)) / 2

@inline function smooth_clamp(x, x⁻, x⁺, δ)
    d⁻ = x - x⁻
    d⁺ = x - x⁺
    # `hypotenuse` does not overflow where `x` is many orders of magnitude beyond the bounds
    t = (d⁻ + d⁺) / (hypotenuse(d⁻, δ) + hypotenuse(d⁺, δ))
    return (x⁻ * (1 - t) + x⁺ * (1 + t)) / 2
end

# The larger of `a` and `b`
@inline clipped_maximum(policy::SmoothClipping, a, b, scale) = smooth_maximum(a, b, policy.width * scale)

# `x`, between `lower_bound` and `upper_bound`
@inline clip_between(policy::SmoothClipping, x, lower_bound, upper_bound, scale) =
    smooth_clamp(x, lower_bound, upper_bound, policy.width * scale)

#####
##### The square root of a quantity that is not negative but for round-off
#####

# √x whose value is exactly that of `sqrt` for x ≥ 0, with a finite (zero) derivative at x = 0.
@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))
