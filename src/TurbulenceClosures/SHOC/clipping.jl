#####
##### Clipping policy: how SHOC's bounds are imposed
#####
##### Every floor, ceiling and bound of the column physics that the state can reach goes through
##### one of the primitives below, which dispatch on the clipping policy, so that every clip can be
##### swapped at once: `HardClipping` or `SmoothClipping`. `HardClipping` is the
##### `max` / `min` / `if` of E3SM `shoc.F90`: each primitive forwards to the idiom its call site
##### has in the Fortran, with the arguments in the order of the call site. The idioms are kept
##### apart because they are not the same function: `max(lo, x)` and `max(x, lo)` return different
##### zeros for `(0.0, -0.0)`, and `ifelse(y < lo, lo, y)` keeps a negative zero and passes a NaN
##### where `max` would not.
#####
##### The last argument, `scale`, is a positive magnitude characteristic of the clipped quantity,
##### a parameter or a literal. `HardClipping` ignores it.
#####
##### Not routed through the policy, in any configuration: the discrete search for the boundary
##### layer top, the cold-surface fallback of the diffusivities, and guards against round-off
##### whose bound the exact value already satisfies (interpolated positive quantities, `C ≤ 1`,
##### `qˡ ≥ 0`, the variance of `s`). They use `max`, `min` or `hard_floor` directly.
#####

abstract type AbstractClippingPolicy <: AbstractSHOCComponent end

"""
$(TYPEDEF)

Impose SHOC's bounds with `max`, `min` and `ifelse`, as E3SM `shoc.F90` does. The clipped
quantities are continuous with a kink at each bound.
"""
struct HardClipping <: AbstractClippingPolicy end

Base.summary(::HardClipping) = "HardClipping"
Base.show(io::IO, policy::HardClipping) = print(io, summary(policy), "()")

# The larger and the smaller of `a` and `b`
@inline clip_max(::HardClipping, a, b, scale) = max(a, b)
@inline clip_min(::HardClipping, a, b, scale) = min(a, b)

# `y`, not below `lo`: the Fortran `if (y < lo) y = lo`
@inline clip_below(::HardClipping, y, lo, scale) = ifelse(y < lo, lo, y)

# `x`, between `lo` and `hi`
@inline clip_between(::HardClipping, x, lo, hi, scale) = max(lo, min(hi, x))

# The positive part of `x`
@inline clip_negative(::HardClipping, x, scale) = ifelse(x >= 0, x, zero(x))

# `λ N²` where the stratification `N²` is stable, zero elsewhere: the Fortran sets `λ = 0` where
# `N² ≤ 0` and multiplies afterwards.
@inline stable_damping(::HardClipping, λ, N², scale) = ifelse(N² <= 0, zero(λ), λ) * N²

#####
##### Smooth clipping
#####

"""
$(TYPEDEF)

Impose SHOC's bounds with smooth functions, so that the clipped quantities are infinitely
differentiable in the state and in the coefficients. With `d = a - b` and the smoothing length
`δ = width × scale`, where `scale` is the magnitude each call site gives for its quantity,

    smooth_max(a, b) = (a + b + √(d² + δ²)) / 2,    smooth_min(a, b) = (a + b - √(d² + δ²)) / 2

and the two-sided bound is `(lo + hi + √((x - lo)² + δ²) - √((x - hi)² + δ²)) / 2`. These are
one-sided: `smooth_max ≥ max` and `smooth_min ≤ min` exactly, and the two-sided bound lies
strictly between `lo` and `hi`, so floors, ceilings and realizability bounds still hold. The
price is a bias of `δ / 2` at the bound that decays as `δ² / (4 |d|)` away from it.

$(TYPEDFIELDS)
"""
struct SmoothClipping{FT} <: AbstractClippingPolicy
    "The smoothing length of a bound over the scale of the bounded quantity; positive"
    width :: FT

    function SmoothClipping{FT}(width) where FT
        check_smoothing_width(width)
        return new{FT}(width)
    end
end

SmoothClipping(width::FT) where FT = SmoothClipping{FT}(width)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SmoothClipping{FT}` with the smoothing `width`.
"""
SmoothClipping(FT::DataType = Oceananigans.defaults.FloatType; width = 1e-2) = SmoothClipping{FT}(convert(FT, width))

check_smoothing_width(width::AbstractFloat) =
    width > 0 || throw(ArgumentError("The width of SmoothClipping must be positive, but width = $width."))

# A traced scalar cannot be tested when the closure is rebuilt inside a compiled function
check_smoothing_width(width) = nothing

Adapt.@adapt_structure SmoothClipping

@inline smooth_max(a, b, δ) = (a + b + sqrt((a - b) * (a - b) + δ * δ)) / 2
@inline smooth_min(a, b, δ) = (a + b - sqrt((a - b) * (a - b) + δ * δ)) / 2
@inline smooth_clamp(x, lo, hi, δ) = (lo + hi + sqrt((x - lo) * (x - lo) + δ * δ) - sqrt((x - hi) * (x - hi) + δ * δ)) / 2

@inline clip_max(policy::SmoothClipping, a, b, scale) = smooth_max(a, b, policy.width * scale)
@inline clip_min(policy::SmoothClipping, a, b, scale) = smooth_min(a, b, policy.width * scale)
@inline clip_below(policy::SmoothClipping, y, lo, scale) = smooth_max(y, lo, policy.width * scale)
# The difference of the two square roots loses digits where `x` is far beyond the bounds; the
# hard bound guards that round-off only (the exact value is strictly inside).
@inline clip_between(policy::SmoothClipping, x, lo, hi, scale) =
    max(lo, min(hi, smooth_clamp(x, lo, hi, policy.width * scale)))
@inline clip_negative(policy::SmoothClipping, x, scale) = smooth_max(x, zero(x), policy.width * scale)
@inline stable_damping(policy::SmoothClipping, λ, N², scale) = λ * clip_negative(policy, N², scale)

#####
##### Guards that are hard in every configuration
#####

# √x whose value is exactly that of `sqrt` for x ≥ 0, with a finite (zero) derivative at x = 0.
@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))

# Fortran `if (y < minthresh) y = minthresh`, which keeps a negative zero where `max` would not.
@inline hard_floor(y, minthresh) = ifelse(y < minthresh, minthresh, y)
