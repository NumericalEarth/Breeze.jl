#####
##### Clipping policy: how SHOC's bounds are imposed
#####
##### Every floor, ceiling and bound of the column physics that the state can reach goes through
##### one of the primitives below, which dispatch on the clipping policy, so that every clip can be
##### swapped at once: `HardClipping` or `SmoothClipping`. `HardClipping` is the
##### `max` / `min` / `if` of E3SM `shoc.F90`: each primitive forwards to the idiom its call site
##### has in the Fortran, with the arguments in the order of the call site. The idioms are kept
##### apart because they are not the same function: `max(lower_bound, x)` and `max(x, lower_bound)` return
##### different zeros for `(0.0, -0.0)`, and `ifelse(y < lower_bound, lower_bound, y)` keeps a negative zero and
##### passes a NaN where `max` would not.
#####
##### The last argument, `scale`, is a positive magnitude characteristic of the clipped quantity,
##### a parameter or a literal. `HardClipping` ignores it.
#####
##### Not routed through the policy, in any configuration: guards against round-off whose bound
##### the exact value already satisfies (interpolated positive quantities, `aᶜ ≤ 1`, `qˡ ≥ 0`, the
##### variance of `s`). They use `max`, `min` or `hard_floor` directly.
#####

abstract type AbstractClippingPolicy <: AbstractSubgridDistributionComponent end

"""
$(TYPEDEF)

Impose SHOC's bounds with `max`, `min` and `ifelse`, as E3SM `shoc.F90` does. The clipped
quantities are continuous with a kink at each bound.
"""
struct HardClipping <: AbstractClippingPolicy end

Base.summary(::HardClipping) = "HardClipping"
Base.show(io::IO, policy::HardClipping) = print(io, summary(policy), "()")

# The larger and the smaller of `a` and `b`
@inline clipped_maximum(::HardClipping, a, b, scale) = max(a, b)
@inline clipped_minimum(::HardClipping, a, b, scale) = min(a, b)

# `x`, between `lower_bound` and `upper_bound`
@inline clip_between(::HardClipping, x, lower_bound, upper_bound, scale) = max(lower_bound, min(upper_bound, x))

#####
##### Smooth clipping
#####

"""
$(TYPEDEF)

Impose SHOC's bounds with smooth functions, so that the clipped quantities have continuous
derivatives of every order in the state and in the coefficients. With `d = a - b` and the smoothing length
`δ = width × scale`, where `scale` is the magnitude each call site gives for its quantity,

    smooth_maximum(a, b) = (a + b + √(d² + δ²)) / 2,    smooth_minimum(a, b) = (a + b - √(d² + δ²)) / 2

and the two-sided bound between `x⁻` and `x⁺` is `(x⁻ + x⁺ + √((x - x⁻)² + δ²) - √((x - x⁺)² + δ²)) / 2`. These
are one-sided: `smooth_maximum ≥ max` and `smooth_minimum ≤ min` exactly, and the two-sided bound lies
strictly between `x⁻` and `x⁺`, so floors, ceilings and realizability bounds still hold. The
price is a bias of `δ / 2` at the bound that decays as `δ² / (4 |d|)` away from it.

$(TYPEDFIELDS)
"""
struct SmoothClipping{FT} <: AbstractClippingPolicy
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
@inline smooth_minimum(a, b, δ) = (a + b - sqrt((a - b) * (a - b) + δ * δ)) / 2
@inline smooth_clamp(x, x⁻, x⁺, δ) = (x⁻ + x⁺ + sqrt((x - x⁻) * (x - x⁻) + δ * δ) - sqrt((x - x⁺) * (x - x⁺) + δ * δ)) / 2

@inline clipped_maximum(policy::SmoothClipping, a, b, scale) = smooth_maximum(a, b, policy.width * scale)
@inline clipped_minimum(policy::SmoothClipping, a, b, scale) = smooth_minimum(a, b, policy.width * scale)
# The difference of the two square roots loses digits where `x` is far beyond the bounds; the
# hard bound guards that round-off only (the exact value is strictly inside).
@inline clip_between(policy::SmoothClipping, x, lower_bound, upper_bound, scale) =
    max(lower_bound, min(upper_bound, smooth_clamp(x, lower_bound, upper_bound, policy.width * scale)))

#####
##### Guards that are hard in every configuration
#####

# √x whose value is exactly that of `sqrt` for x ≥ 0, with a finite (zero) derivative at x = 0.
@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))

# `if (y < minthresh) y = minthresh` of shoc.F90, which keeps a negative zero where `max` would not.
@inline hard_floor(y, lower_bound) = ifelse(y < lower_bound, lower_bound, y)
