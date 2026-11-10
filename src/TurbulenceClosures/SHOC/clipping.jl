#####
##### Clipping policy: how SHOC's bounds are imposed
#####
##### Every floor, ceiling and bound of the column physics that the state can reach goes through
##### one of the primitives below, which dispatch on the clipping policy. `HardClipping` is the
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

abstract type AbstractClippingPolicy end

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
##### Guards that are hard in every configuration
#####

# √x whose value is exactly that of `sqrt` for x ≥ 0, with a finite (zero) derivative at x = 0.
@inline zero_safe_sqrt(x) = ifelse(x > 0, sqrt(ifelse(x > 0, x, one(x))), zero(x))

# Fortran `if (y < minthresh) y = minthresh`, which keeps a negative zero where `max` would not.
@inline hard_floor(y, minthresh) = ifelse(y < minthresh, minthresh, y)
