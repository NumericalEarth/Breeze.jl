#####
##### Stability functions of the stratification number and the shear number
#####
##### Turbulence of velocity √e and size ℓ has the time scale τ = ℓ / √e, and it feels the stratification
##### and the shear through the two numbers
#####
#####   α_N = τ² N²,   α_S = τ² (S² + S²ᵐⁱⁿ).
#####
##### The Richardson number Ri = α_N / α_S of `RationalStabilityFunction` jumps between its limits with the
##### sign of N² where the shear vanishes. Here the functions keep their three endpoints (unstable limit,
##### neutral, stable limit) and are evaluated at the distance from neutral
#####
#####   x = |α_N| / (ℂᴺ⁻ + Ri⁻ α_S)   (α_N < 0),     x = α_N / (ℂᴺ⁺ (1 - α_N)⁺ + Ri⁺ α_S)   (α_N ≥ 0),
#####
##### through the shape s = x / (1 + x). Where the shear number is large x = |Ri| / Ri^±; where there is
##### no shear x is a continuous function of N² through neutral.
#####
##### The coefficients are stored unconstrained (ϑ): each endpoint ratio C^± / C⁰ that the well-posedness
##### of the functions bounds is a smooth bijection of a real number onto its bounds, so that any value of
##### the stored scalars (as a calibration may produce) gives functions within them.
#####

"""
$(TYPEDEF)

Interpolation of a stability function between its neutral value ``C⁰`` and its limit ``C^±`` that is
linear in the shape ``s = x / (1 + x)``: ``C⁰ + (C^± - C⁰) s``, the form of
[`RationalStabilityFunction`](@ref).
"""
struct LinearInterpolation end

"""
$(TYPEDEF)

Interpolation of a stability function between its neutral value ``C⁰`` and its limit ``C^±`` that is
linear in the logarithm: ``C⁰ (C^± / C⁰)^s`` with the shape ``s = x / (1 + x)``. The logarithmic
derivative with respect to ``s`` is the constant ``\\ln(C^± / C⁰)``, the smallest that connects the two
endpoints, where the linear interpolation has ``C^± / C⁰ - 1`` at neutral.
"""
struct LogarithmicInterpolation end

Base.summary(::LinearInterpolation) = "LinearInterpolation"
Base.summary(::LogarithmicInterpolation) = "LogarithmicInterpolation"

# The coordinate in which an interpolation is linear in the shape: ln(C / C⁰) or C / C⁰
@inline ratio_coordinate(::LogarithmicInterpolation, r) = log(r)
@inline ratio_coordinate(::LinearInterpolation, r) = r
@inline coordinate_ratio(::LogarithmicInterpolation, g) = exp(g)
@inline coordinate_ratio(::LinearInterpolation, g) = g

# C⁰ (C^± / C⁰)^s and C⁰ + (C^± - C⁰) s, from the coordinate g of C^± / C⁰
@inline interpolate_coordinate(::LogarithmicInterpolation, C⁰, g, s) = C⁰ * exp(s * g)
@inline interpolate_coordinate(::LinearInterpolation, C⁰, g, s) = C⁰ * (1 + (g - 1) * s)

# The interval of the coordinate of a ratio within a bound b on the slope at neutral:
# |ln(C / C⁰)| ≤ b, or |C / C⁰ - 1| ≤ b with C / C⁰ > 0
@inline slope_bounded_interval(::LogarithmicInterpolation, b) = (-b, b)
@inline slope_bounded_interval(::LinearInterpolation, b) = (max(zero(b), 1 - b), 1 + b)

# The upper bound of ℂᵘ⁺ / ℂᵘ⁰, beyond which the stable side of the momentum function is ill posed
# (notes: the condition of §4.2 of the derivation)
stable_momentum_ratio_bound(::LogarithmicInterpolation) = 7.4
stable_momentum_ratio_bound(::LinearInterpolation) = 9

"""
The least stable transition scale ``Ri⁺`` of [`StratificationNumberStabilityFunction`](@ref): a
smaller one is a step of the functions in ``N²`` where there is shear, in any form.
"""
const MINIMUM_STABLE_TRANSITION_SCALE = 0.1

"""
The intervals of the stratification numbers ``ℂᴺ⁻`` and ``ℂᴺ⁺`` of
[`StratificationNumberStabilityFunction`](@ref).
"""
const UNSTABLE_STRATIFICATION_NUMBER_BOUNDS = (0.3, 10)
const STABLE_STRATIFICATION_NUMBER_BOUNDS = (0.1, 2)

# The smooth bijection of the real line onto (a, b), and its inverse
@inline bounded(ϑ, a, b) = a + (b - a) * (1 + tanh(ϑ)) / 2
unbounded(g, a, b) = atanh(2 * (g - a) / (b - a) - 1)

# ln(1 + exp(x)) without overflow
@inline softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

"""
$(TYPEDEF)

Stability functions for [`TKEBasedTurbulenceClosure`](@ref) of the stratification number and the
shear number of the turbulence,

```math
α_N = τ² N², \\qquad α_S = τ² (S² + S²ᵐⁱⁿ), \\qquad τ = ℓ / \\sqrt{e},
```

where ``ℓ`` is the stored mixing length, ``\\sqrt{e}`` is floored and reconstructed as in the
diffusivities, and ``S²`` and ``S²ᵐⁱⁿ`` are those of the closure's `shear`. Each of the four
functions ``𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ`` passes from its neutral value ``C⁰`` to its unstable limit ``C⁻`` or its
stable limit ``C⁺`` through the shape ``s = x / (1 + x)`` of the distance from neutral

```math
x = \\frac{|α_N|}{ℂᴺ⁻ + Ri⁻ α_S} \\quad (α_N < 0), \\qquad
x = \\frac{α_N}{ℂᴺ⁺ (1 - α_N)⁺ + Ri⁺ α_S} \\quad (α_N ≥ 0),
```

by the `interpolation`, ``𝕊 = C⁰ (C^± / C⁰)^s`` ([`LogarithmicInterpolation`](@ref), the default) or
``𝕊 = C⁰ + (C^± - C⁰) s`` ([`LinearInterpolation`](@ref)).

Where the shear number is large, ``x = |Ri| / Ri^±`` with ``Ri = N² / (S² + S²ᵐⁱⁿ)``: with the linear
interpolation the functions are those of [`RationalStabilityFunction`](@ref). Where the shear vanishes
the functions are continuous in ``N²`` through neutral, with a transition of width ``ℂᴺ^± e / ℓ²``:
turbulence feels the stratification where ``τ² N² = O(ℂᴺ^±)``. The mixing length is bounded by
``\\sqrt{e} / N``, so ``α_N ≤ 1``, and where ``ℓ = \\sqrt{e} / N`` (``α_N = 1``) ``x = Ri / Ri⁺`` at any
shear; ``ℂᴺ⁺`` acts only where the wall or the stratification envelope bounds ``ℓ``.

# The coefficients and their bounds

The scalars of the type (those [`closure_parameters`](@ref) lists and [`with_parameters`](@ref)
replaces) are unconstrained: every real value of the ``ϑ`` gives functions within the bounds below,
so that a calibration cannot leave the region where they are well posed. With ``s`` the
`maximum_logarithmic_slope` and ``σ(ϑ) = (1 + \\tanh ϑ) / 2``,

| coefficient | from the scalars | bound |
|:--|:--|:--|
| ``ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰, ℂᵘ⁻, Ri⁻`` | themselves | positive |
| ``ℂᴺ⁻`` | ``\\exp[\\ln 0.3 + σ(ϑᴺ⁻) \\ln(10 / 0.3)]`` | ``0.3 < ℂᴺ⁻ < 10`` |
| ``ℂᴺ⁺`` | ``\\exp[\\ln 0.1 + σ(ϑᴺ⁺) \\ln(2 / 0.1)]`` | ``0.1 < ℂᴺ⁺ < 2`` |
| ``ℂᶜ⁻, ℂᵉ⁻, ℂᴰ⁻`` | ``\\ln(C⁻ / C⁰) = b⁻ \\tanh ϑ``, ``b⁻ = s ℂᴺ⁻`` | ``\\lvert\\ln(C⁻ / C⁰)\\rvert < s ℂᴺ⁻`` |
| ``ℂᵉ⁺, ℂᴰ⁺`` | ``\\ln(C⁺ / C⁰) = b⁺ \\tanh ϑ``, ``b⁺ = s (ℂᴺ⁺⁻⁴ + ℂᴺ⁺⁴)^{-1/4}`` | ``\\lvert\\ln(C⁺ / C⁰)\\rvert < b⁺`` |
| ``ℂᶜ⁺`` | ``\\ln(ℂᶜ⁺ / ℂᶜ⁰) = -b⁺ + σ(ϑᶜ⁺) [\\min(b⁺, -\\ln ℂᶜ⁰) + b⁺]`` | the same, and ``ℂᶜ⁺ < 1`` |
| ``ℂᵘ⁺`` | ``ℂᵘ⁺ / ℂᵘ⁰ = 7.4 / (1 + e^{-ϑᵘ⁺})`` | ``ℂᵘ⁺ / ℂᵘ⁰ < 7.4`` |
| ``Ri⁺`` | ``0.1 + e^{ϑᴿⁱ⁺}`` | ``Ri⁺ > 0.1`` |

(The ϑ of each ratio is named after the ratio: `ϑᶜ⁻` for ``ℂᶜ⁻``, and so on.) The bound on the
logarithmic slope keeps ``∂ \\ln 𝕊 / ∂ α_N`` of order one where the shear vanishes, so that a
stratification number that changes by its own size changes the functions by a factor of order one; on
the stable side the slope near ``α_N = 1`` is ``ℂᴺ⁺ |\\ln(C⁺ / C⁰)|``, hence a bound below
``\\min(ℂᴺ⁺, 1 / ℂᴺ⁺)``: its smooth 4-norm ``(ℂᴺ⁺⁻⁴ + ℂᴺ⁺⁴)^{-1/4}`` (``2^{-1/4} ≈ 0.84`` at ``ℂᴺ⁺ = 1``), so that
the coefficients are differentiable in ``ϑᴺ⁺`` everywhere.
``ℂᶜ⁺ < 1`` and ``ℂᵘ⁺ / ℂᵘ⁰ < 7.4`` keep the stable equilibrium of the TKE budget of the functions
unique, and ``Ri⁺`` below about 0.1 is a step in ``N²`` wherever there is shear. With
[`LinearInterpolation`](@ref) the slope bounds are on ``|C / C⁰ - 1|`` instead of
``|\\ln(C / C⁰)|`` (the interval is ``(\\max(0, 1 - b), 1 + b)``, and ``ℂᶜ⁺ / ℂᶜ⁰ < 1 / ℂᶜ⁰``), and
the bound of ``ℂᵘ⁺ / ℂᵘ⁰`` is 9.

[`stability_endpoints`](@ref) returns the constrained coefficients as a
[`RationalStabilityFunction`](@ref), and [`transition_stratification_numbers`](@ref) ``ℂᴺ⁻, ℂᴺ⁺``.
`StratificationNumberStabilityFunction(rational; ℂᴺ⁻, ℂᴺ⁺)` maps a set of constrained coefficients
onto the scalars, clipping it into the bounds.

Fields
======

$(TYPEDFIELDS)
"""
struct StratificationNumberStabilityFunction{FT, I}
    "momentum, unstable limit ``ℂᵘ⁻ > 0``"
    ℂᵘ⁻ :: FT
    "momentum, neutral ``ℂᵘ⁰ > 0``"
    ℂᵘ⁰ :: FT
    "momentum, stable limit: ``ℂᵘ⁺ / ℂᵘ⁰ = 7.4 / (1 + e^{-ϑᵘ⁺})`` (9 with `LinearInterpolation`)"
    ϑᵘ⁺ :: FT
    "tracers, unstable limit: ``\\ln(ℂᶜ⁻ / ℂᶜ⁰) = s ℂᴺ⁻ \\tanh ϑᶜ⁻``"
    ϑᶜ⁻ :: FT
    "tracers, neutral ``ℂᶜ⁰ > 0``"
    ℂᶜ⁰ :: FT
    "tracers, stable limit: ``\\ln(ℂᶜ⁺ / ℂᶜ⁰)`` between ``-b⁺`` and ``\\min(b⁺, -\\ln ℂᶜ⁰)``, so that ``ℂᶜ⁺ < 1``"
    ϑᶜ⁺ :: FT
    "TKE, unstable limit: ``\\ln(ℂᵉ⁻ / ℂᵉ⁰) = s ℂᴺ⁻ \\tanh ϑᵉ⁻``"
    ϑᵉ⁻ :: FT
    "TKE, neutral ``ℂᵉ⁰ > 0``"
    ℂᵉ⁰ :: FT
    "TKE, stable limit: ``\\ln(ℂᵉ⁺ / ℂᵉ⁰) = b⁺ \\tanh ϑᵉ⁺``"
    ϑᵉ⁺ :: FT
    "dissipation, unstable limit: ``\\ln(ℂᴰ⁻ / ℂᴰ⁰) = s ℂᴺ⁻ \\tanh ϑᴰ⁻``"
    ϑᴰ⁻ :: FT
    "dissipation, neutral ``ℂᴰ⁰ > 0``"
    ℂᴰ⁰ :: FT
    "dissipation, stable limit: ``\\ln(ℂᴰ⁺ / ℂᴰ⁰) = b⁺ \\tanh ϑᴰ⁺``"
    ϑᴰ⁺ :: FT
    "unstable Richardson transition scale ``Ri⁻ > 0``"
    Ri⁻ :: FT
    "stable Richardson transition scale: ``Ri⁺ = 0.1 + e^{ϑᴿⁱ⁺}``"
    ϑᴿⁱ⁺ :: FT
    "unstable stratification number of the transition without shear: ``ℂᴺ⁻ ∈ (0.3, 10)``"
    ϑᴺ⁻ :: FT
    "stable stratification number of the transition without shear: ``ℂᴺ⁺ ∈ (0.1, 2)``"
    ϑᴺ⁺ :: FT
    "bound ``s`` of the logarithmic slope of ``𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ`` per unit stratification number (not calibrated)"
    maximum_logarithmic_slope :: FT
    "interpolation between the neutral value and the limits"
    interpolation :: I
end

#####
##### The constrained coefficients
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The stratification numbers ``ℂᴺ⁻`` and ``ℂᴺ⁺`` of the transitions of `sf` where the shear vanishes.
"""
@inline function transition_stratification_numbers(sf::StratificationNumberStabilityFunction)
    FT = typeof(sf.ϑᴺ⁻)
    a⁻, b⁻ = UNSTABLE_STRATIFICATION_NUMBER_BOUNDS
    a⁺, b⁺ = STABLE_STRATIFICATION_NUMBER_BOUNDS
    ℂᴺ⁻ = exp(bounded(sf.ϑᴺ⁻, log(convert(FT, a⁻)), log(convert(FT, b⁻))))
    ℂᴺ⁺ = exp(bounded(sf.ϑᴺ⁺, log(convert(FT, a⁺)), log(convert(FT, b⁺))))
    return ℂᴺ⁻, ℂᴺ⁺
end

@inline stable_transition_scale(sf::StratificationNumberStabilityFunction) =
    convert(typeof(sf.ϑᴿⁱ⁺), MINIMUM_STABLE_TRANSITION_SCALE) + exp(sf.ϑᴿⁱ⁺)

# The bounds b⁻ and b⁺ of the slopes of 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ
@inline function slope_bounds(sf::StratificationNumberStabilityFunction)
    ℂᴺ⁻, ℂᴺ⁺ = transition_stratification_numbers(sf)
    s = sf.maximum_logarithmic_slope
    # A smooth bound below min(ℂᴺ⁺, 1 / ℂᴺ⁺), whose kink at ℂᴺ⁺ = 1 would leave the endpoints
    # without a derivative in ϑᴺ⁺ there
    return s * ℂᴺ⁻, s / sqrt(sqrt(ℂᴺ⁺^4 + 1 / ℂᴺ⁺^4))
end

# The coordinates (ln of the ratio, or the ratio) of the unstable and stable limits of each function
@inline function momentum_coordinates(sf::StratificationNumberStabilityFunction)
    interp = sf.interpolation
    R = convert(typeof(sf.ϑᵘ⁺), stable_momentum_ratio_bound(interp))
    g⁻ = ratio_coordinate(interp, sf.ℂᵘ⁻ / sf.ℂᵘ⁰)
    g⁺ = momentum_stable_coordinate(interp, R, sf.ϑᵘ⁺)
    return g⁻, g⁺
end

@inline momentum_stable_coordinate(::LogarithmicInterpolation, R, ϑ) = log(R) - softplus(-ϑ)
@inline momentum_stable_coordinate(::LinearInterpolation, R, ϑ) = R / (1 + exp(-ϑ))

@inline function slope_bounded_coordinate(interp, ϑ, b)
    a, c = slope_bounded_interval(interp, b)
    return bounded(ϑ, a, c)
end

# The stable tracer limit, also below ℂᶜ⁺ = 1
@inline function tracer_stable_interval(interp, b⁺, ℂᶜ⁰)
    a, c = slope_bounded_interval(interp, b⁺)
    c = min(c, ratio_coordinate(interp, 1 / ℂᶜ⁰))
    return min(a, c), c
end

@inline function tracer_coordinates(sf::StratificationNumberStabilityFunction)
    b⁻, b⁺ = slope_bounds(sf)
    a, c = tracer_stable_interval(sf.interpolation, b⁺, sf.ℂᶜ⁰)
    return slope_bounded_coordinate(sf.interpolation, sf.ϑᶜ⁻, b⁻), bounded(sf.ϑᶜ⁺, a, c)
end

@inline function tke_coordinates(sf::StratificationNumberStabilityFunction)
    b⁻, b⁺ = slope_bounds(sf)
    return slope_bounded_coordinate(sf.interpolation, sf.ϑᵉ⁻, b⁻), slope_bounded_coordinate(sf.interpolation, sf.ϑᵉ⁺, b⁺)
end

@inline function dissipation_coordinates(sf::StratificationNumberStabilityFunction)
    b⁻, b⁺ = slope_bounds(sf)
    return slope_bounded_coordinate(sf.interpolation, sf.ϑᴰ⁻, b⁻), slope_bounded_coordinate(sf.interpolation, sf.ϑᴰ⁺, b⁺)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The constrained endpoints ``C⁻, C⁰, C⁺`` of the four functions of `sf` and its transition scales
``Ri⁻, Ri⁺``, as a [`RationalStabilityFunction`](@ref) (whose own functions are those of `sf` at
large shear with [`LinearInterpolation`](@ref)).
"""
function stability_endpoints(sf::StratificationNumberStabilityFunction)
    interp = sf.interpolation
    endpoint(C⁰, g) = C⁰ * coordinate_ratio(interp, g)
    gᵘ⁻, gᵘ⁺ = momentum_coordinates(sf)
    gᶜ⁻, gᶜ⁺ = tracer_coordinates(sf)
    gᵉ⁻, gᵉ⁺ = tke_coordinates(sf)
    gᴰ⁻, gᴰ⁺ = dissipation_coordinates(sf)
    return RationalStabilityFunction(endpoint(sf.ℂᵘ⁰, gᵘ⁻), sf.ℂᵘ⁰, endpoint(sf.ℂᵘ⁰, gᵘ⁺),
                                     endpoint(sf.ℂᶜ⁰, gᶜ⁻), sf.ℂᶜ⁰, endpoint(sf.ℂᶜ⁰, gᶜ⁺),
                                     endpoint(sf.ℂᵉ⁰, gᵉ⁻), sf.ℂᵉ⁰, endpoint(sf.ℂᵉ⁰, gᵉ⁺),
                                     endpoint(sf.ℂᴰ⁰, gᴰ⁻), sf.ℂᴰ⁰, endpoint(sf.ℂᴰ⁰, gᴰ⁺),
                                     sf.Ri⁻, stable_transition_scale(sf))
end

#####
##### Construction
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Construct a [`StratificationNumberStabilityFunction`](@ref) from its unconstrained scalars (see the
type for the mapping onto the coefficients).

The defaults are the coefficients of the provisional calibration of the TKE closure in the
calibration harness (a [`RationalStabilityFunction`](@ref) with ``ℂᶜ⁻ / ℂᶜ⁰ = 0.41``,
``ℂᴰ⁺ / ℂᴰ⁰ = 27.5``, ``Ri⁺ = 0.05``, ...), with ``ℂᴺ⁻ = 2`` and ``ℂᴺ⁺ = 1``, mapped onto the
scalars of the logarithmic form and clipped into its bounds by
`StratificationNumberStabilityFunction(rational; ℂᴺ⁻ = 2, ℂᴺ⁺ = 1)` with the default
`interior_fraction = 0.9`. The bounds that clip are those of ``ℂᵉ⁻, ℂᵉ⁺, ℂᴰ⁺, ℂᶜ⁺`` and ``Ri⁺``.

```jldoctest
using Breeze

StratificationNumberStabilityFunction()

# output
StratificationNumberStabilityFunction{Float64}
├── ℂᴺ⁻ = 2.0, ℂᴺ⁺ = 1.0, maximum_logarithmic_slope = 1.0, LogarithmicInterpolation
├── Ri → -∞ (ℂᵘ⁻, ℂᶜ⁻, ℂᵉ⁻, ℂᴰ⁻): 0.36113, 0.0817085, 0.0442406, 0.00540707
├── Ri = 0  (ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰): 0.159006, 0.198722, 0.00731292, 0.0291824
├── Ri → ∞  (ℂᵘ⁺, ℂᶜ⁺, ℂᵉ⁺, ℂᴰ⁺): 0.0109563, 0.423567, 0.0155872, 0.062201
└── transition scales: Ri⁻ = 12.6697, Ri⁺ = 0.11
```
"""
function StratificationNumberStabilityFunction(FT::DataType = Oceananigans.defaults.FloatType;
                                               ℂᵘ⁻ = 0.36113018152281917,
                                               ℂᵘ⁰ = 0.15900569445804447,
                                               ϑᵘ⁺ = -4.6671479898684884,
                                               ϑᶜ⁻ = -0.47766722502290965,
                                               ℂᶜ⁰ = 0.19872163735761372,
                                               ϑᶜ⁺ = 1.4722194895832204,
                                               ϑᵉ⁻ = 1.4722194895832204,
                                               ℂᵉ⁰ = 0.00731292096489522,
                                               ϑᵉ⁺ = 1.4722194895832204,
                                               ϑᴰ⁻ = -1.2312052908653401,
                                               ℂᴰ⁰ = 0.029182354585815418,
                                               ϑᴰ⁺ = 1.4722194895832204,
                                               Ri⁻ = 12.669701017862883,
                                               ϑᴿⁱ⁺ = -4.605170185988092,
                                               ϑᴺ⁻ = 0.08222597171681865,
                                               ϑᴺ⁺ = 0.6002726829148098,
                                               maximum_logarithmic_slope = 1,
                                               interpolation = LogarithmicInterpolation())

    positive = (ℂᵘ⁻, ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰, Ri⁻, maximum_logarithmic_slope)
    all(C -> C > 0, positive) ||
        throw(ArgumentError("ℂᵘ⁻, ℂᵘ⁰, ℂᶜ⁰, ℂᵉ⁰, ℂᴰ⁰, Ri⁻ and maximum_logarithmic_slope of " *
                            "StratificationNumberStabilityFunction must be positive, but they are $positive"))
    unconstrained = (ϑᵘ⁺, ϑᶜ⁻, ϑᶜ⁺, ϑᵉ⁻, ϑᵉ⁺, ϑᴰ⁻, ϑᴰ⁺, ϑᴿⁱ⁺, ϑᴺ⁻, ϑᴺ⁺)
    all(isfinite, unconstrained) ||
        throw(ArgumentError("The unconstrained scalars of StratificationNumberStabilityFunction must be finite, " *
                            "but (ϑᵘ⁺, ϑᶜ⁻, ϑᶜ⁺, ϑᵉ⁻, ϑᵉ⁺, ϑᴰ⁻, ϑᴰ⁺, ϑᴿⁱ⁺, ϑᴺ⁻, ϑᴺ⁺) = $unconstrained"))

    scalars = map(x -> convert(FT, x), (ℂᵘ⁻, ℂᵘ⁰, ϑᵘ⁺, ϑᶜ⁻, ℂᶜ⁰, ϑᶜ⁺, ϑᵉ⁻, ℂᵉ⁰, ϑᵉ⁺, ϑᴰ⁻, ℂᴰ⁰, ϑᴰ⁺,
                                         Ri⁻, ϑᴿⁱ⁺, ϑᴺ⁻, ϑᴺ⁺, maximum_logarithmic_slope))
    return StratificationNumberStabilityFunction(scalars..., interpolation)
end

# The unconstrained value of a coordinate g in (a, b), clipped to the inner `f` of the interval
clipped_unbounded(g, a, b, f) = atanh(clamp(2 * (g - a) / (b - a) - 1, -f, f))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Map the endpoints and transition scales of the [`RationalStabilityFunction`](@ref) `rational` onto
the unconstrained scalars of a [`StratificationNumberStabilityFunction`](@ref) with the stratification
numbers `ℂᴺ⁻` and `ℂᴺ⁺`, clipping each bounded coefficient into the inner `interior_fraction` of its
bounds (in the variable ``2 σ - 1 = \\tanh ϑ``; ``ℂᵘ⁺ / ℂᵘ⁰ ≤ 7.4 (1 + f) / 2`` and
``Ri⁺ - 0.1 ≥ (1 - f) 0.1``). The coefficients
within that are kept exactly.

```jldoctest
using Breeze
using Breeze.TurbulenceClosures: stability_endpoints

sf = StratificationNumberStabilityFunction(RationalStabilityFunction(); ℂᴺ⁻ = 2, ℂᴺ⁺ = 1)
stability_endpoints(sf).ℂᵘ⁰ == RationalStabilityFunction().ℂᵘ⁰

# output
true
```
"""
function StratificationNumberStabilityFunction(rational::RationalStabilityFunction{FT}; ℂᴺ⁻ = 2, ℂᴺ⁺ = 1,
                                               interpolation = LogarithmicInterpolation(),
                                               maximum_logarithmic_slope = 1,
                                               interior_fraction = 0.9) where FT
    f = interior_fraction
    0 < f < 1 || throw(ArgumentError("interior_fraction must be in (0, 1), but it is $f"))
    r = rational
    a⁻, c⁻ = log.(UNSTABLE_STRATIFICATION_NUMBER_BOUNDS)
    a⁺, c⁺ = log.(STABLE_STRATIFICATION_NUMBER_BOUNDS)
    ϑᴺ⁻ = clipped_unbounded(log(ℂᴺ⁻), a⁻, c⁻, f)
    ϑᴺ⁺ = clipped_unbounded(log(ℂᴺ⁺), a⁺, c⁺, f)

    # The bounds of the slopes with the stratification numbers as they are clipped
    template = StratificationNumberStabilityFunction(FT; ϑᴺ⁻, ϑᴺ⁺, maximum_logarithmic_slope, interpolation)
    b⁻, b⁺ = slope_bounds(template)
    coordinate(C, C⁰) = ratio_coordinate(interpolation, C / C⁰)
    slope_bounded(C, C⁰, b) = clipped_unbounded(coordinate(C, C⁰), slope_bounded_interval(interpolation, b)..., f)

    R = stable_momentum_ratio_bound(interpolation)
    # ℂᵘ⁺ / ℂᵘ⁰ = R σ with σ = 1 / (1 + e^{-ϑ}): clipped below R only (zero is not a bound that clips)
    σᵘ = min(r.ℂᵘ⁺ / r.ℂᵘ⁰ / R, (1 + f) / 2)
    ϑᵘ⁺ = log(σᵘ / (1 - σᵘ))
    ϑᶜ⁺ = clipped_unbounded(coordinate(r.ℂᶜ⁺, r.ℂᶜ⁰), tracer_stable_interval(interpolation, b⁺, r.ℂᶜ⁰)..., f)
    Ri⁺ᵐⁱⁿ = MINIMUM_STABLE_TRANSITION_SCALE
    ϑᴿⁱ⁺ = log(max(r.Ri⁺ - Ri⁺ᵐⁱⁿ, (1 - f) * Ri⁺ᵐⁱⁿ))

    return StratificationNumberStabilityFunction(FT; r.ℂᵘ⁻, r.ℂᵘ⁰, ϑᵘ⁺,
                                                 ϑᶜ⁻ = slope_bounded(r.ℂᶜ⁻, r.ℂᶜ⁰, b⁻), r.ℂᶜ⁰, ϑᶜ⁺,
                                                 ϑᵉ⁻ = slope_bounded(r.ℂᵉ⁻, r.ℂᵉ⁰, b⁻), r.ℂᵉ⁰,
                                                 ϑᵉ⁺ = slope_bounded(r.ℂᵉ⁺, r.ℂᵉ⁰, b⁺),
                                                 ϑᴰ⁻ = slope_bounded(r.ℂᴰ⁻, r.ℂᴰ⁰, b⁻), r.ℂᴰ⁰,
                                                 ϑᴰ⁺ = slope_bounded(r.ℂᴰ⁺, r.ℂᴰ⁰, b⁺),
                                                 r.Ri⁻, ϑᴿⁱ⁺, ϑᴺ⁻, ϑᴺ⁺, maximum_logarithmic_slope, interpolation)
end

@inline convert_eltype(::Type{FT}, sf::StratificationNumberStabilityFunction) where FT =
    StratificationNumberStabilityFunction((convert(FT, getfield(sf, name)) for name in fieldnames(StratificationNumberStabilityFunction)[1:end-1])...,
                                          sf.interpolation)

Adapt.@adapt_structure StratificationNumberStabilityFunction

Base.summary(sf::StratificationNumberStabilityFunction{FT}) where FT = "StratificationNumberStabilityFunction{$FT}"

function show_stability_function_lines(io::IO, sf::StratificationNumberStabilityFunction, prefix)
    ℂᴺ⁻, ℂᴺ⁺ = transition_stratification_numbers(sf)
    print(io, prefix, "├── ℂᴺ⁻ = ", prettysummary(ℂᴺ⁻), ", ℂᴺ⁺ = ", prettysummary(ℂᴺ⁺),
              ", maximum_logarithmic_slope = ", prettysummary(sf.maximum_logarithmic_slope), ", ",
              summary(sf.interpolation), '\n')
    show_stability_function_lines(io, stability_endpoints(sf), prefix)
    return nothing
end

function Base.show(io::IO, sf::StratificationNumberStabilityFunction)
    print(io, summary(sf), '\n')
    show_stability_function_lines(io, sf, "")
end

#####
##### The functions of the stratification and shear numbers
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The shape ``s = x / (1 + x) ∈ [0, 1]`` of the stability functions `sf` at the stratification number
`αᴺ` and the shear number `αˢ`, with ``x = |α_N| / (ℂᴺ + Ri^± α_S)``, where ``ℂᴺ`` is ``ℂᴺ⁻`` for
``α_N < 0`` and ``ℂᴺ⁺ (1 - α_N)⁺`` for ``α_N ≥ 0``. Zero at `αᴺ = 0` and one where the denominator
vanishes.
"""
@inline function stability_shape(sf::StratificationNumberStabilityFunction, αᴺ, αˢ)
    ℂᴺ⁻, ℂᴺ⁺ = transition_stratification_numbers(sf)
    neutral = αᴺ == 0
    stable = αᴺ ≥ 0
    # Both branches of an `ifelse` are evaluated and differentiated, so the unselected one is formed at |α_N| = 1
    a = ifelse(neutral, one(αᴺ), abs(αᴺ))
    # On the stable side ℂᴺ⁺ (1 - α_N): the stable limit is reached at α_N = 1, where ℓ = √e / N
    ℂᴺ = ifelse(stable, ℂᴺ⁺ * max(0, 1 - a), ℂᴺ⁻)
    Riᵇ = ifelse(stable, stable_transition_scale(sf), sf.Ri⁻)
    x⁻¹ = (ℂᴺ + Riᵇ * αˢ) / a
    return ifelse(neutral, zero(αᴺ), 1 / (1 + x⁻¹))
end

@inline function interpolate_stability(sf::StratificationNumberStabilityFunction, αᴺ, αˢ, C⁰, (g⁻, g⁺))
    s = stability_shape(sf, αᴺ, αˢ)
    return interpolate_coordinate(sf.interpolation, C⁰, ifelse(αᴺ ≥ 0, g⁺, g⁻), s)
end

@inline momentum_stability_function(sf::StratificationNumberStabilityFunction, αᴺ, αˢ) =
    interpolate_stability(sf, αᴺ, αˢ, sf.ℂᵘ⁰, momentum_coordinates(sf))

@inline tracer_stability_function(sf::StratificationNumberStabilityFunction, αᴺ, αˢ) =
    interpolate_stability(sf, αᴺ, αˢ, sf.ℂᶜ⁰, tracer_coordinates(sf))

@inline tke_stability_function(sf::StratificationNumberStabilityFunction, αᴺ, αˢ) =
    interpolate_stability(sf, αᴺ, αˢ, sf.ℂᵉ⁰, tke_coordinates(sf))

@inline dissipation_stability_function(sf::StratificationNumberStabilityFunction, αᴺ, αˢ) =
    interpolate_stability(sf, αᴺ, αˢ, sf.ℂᴰ⁰, dissipation_coordinates(sf))

#####
##### The functions of the local gradients, for the closures that form them at a point
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The stability functions ``(𝕊ᵘ, 𝕊ᶜ, 𝕊ᵉ, 𝕊ᴰ)`` of `sf` from the squared buoyancy frequency `N²`, the
squared shear `S²` and its floor `S²ᵐⁱⁿ`, and the squared time scale `τ²` ``= ℓ² / e`` of the
turbulence (which the functions of the Richardson number do not use).
"""
@inline local_stability_functions(sf::ConstantStabilityFunctions, N², S², S²ᵐⁱⁿ, τ²) = (sf.ℂᵘ, sf.ℂᶜ, sf.ℂᵉ, sf.ℂᴰ)

@inline function local_stability_functions(sf::RationalStabilityFunction, N², S², S²ᵐⁱⁿ, τ²)
    Ri = richardson_number(N², S², S²ᵐⁱⁿ)
    return (momentum_stability_function(sf, Ri), tracer_stability_function(sf, Ri),
            tke_stability_function(sf, Ri), dissipation_stability_function(sf, Ri))
end

@inline function local_stability_functions(sf::StratificationNumberStabilityFunction, N², S², S²ᵐⁱⁿ, τ²)
    αᴺ = τ² * N²
    αˢ = τ² * (S² + S²ᵐⁱⁿ)
    return (momentum_stability_function(sf, αᴺ, αˢ), tracer_stability_function(sf, αᴺ, αˢ),
            tke_stability_function(sf, αᴺ, αˢ), dissipation_stability_function(sf, αᴺ, αˢ))
end

#####
##### Evaluation on the grid
#####

const StratificationNumberClosure = TKEBasedTurbulenceClosure{<:Any, <:Any, <:StratificationNumberStabilityFunction}

# The stratification and shear numbers at the face, with τ² = ℓ² / e from the stored mixing length and
# √e floored and reconstructed as in the diffusivity ℓ √e. Both vanish where ℓ = 0 (the surface).
@inline function stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    w★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
    τ² = (@inbounds ℓ[i, j, k] / w★)^2
    S² = effective_shear_squaredᶜᶜᶠ(i, j, k, grid, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    return τ² * @inbounds(N²[i, j, k]), τ² * S²
end

# At the center, where the dissipation lives: N² and S² reconstructed from the two faces, as for the
# Richardson number, and the time scale of the reconstructed mixing length bounded by the penetration
# time 1 / N of the center, so that α_N ≤ 1 there as at the faces.
@inline function stratification_and_shear_numbersᶜᶜᶜ(i, j, k, grid, closure, velocities, N², e, ℓ)
    S² = ℑbzᵃᵃᶜ(i, j, k, grid, effective_shear_squaredᶜᶜᶠ, closure, velocities, e, ℓ) + closure.shear.minimum_shear_squared
    N²ᵢ = ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, N²)
    τ²ʷ = (ℑbzᵃᵃᶜ(i, j, k, grid, face_valueᶜᶜᶠ, ℓ) / turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e))^2
    stratified = N²ᵢ > 0
    τ² = ifelse(stratified, min(τ²ʷ, 1 / ifelse(stratified, N²ᵢ, one(N²ᵢ))), τ²ʷ)
    return τ² * N²ᵢ, τ² * S²
end

@inline function momentum_stability_functionᶜᶜᶠ(i, j, k, grid, closure::StratificationNumberClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return momentum_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function tracer_stability_functionᶜᶜᶠ(i, j, k, grid, closure::StratificationNumberClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return tracer_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function tke_stability_functionᶜᶜᶠ(i, j, k, grid, closure::StratificationNumberClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return tke_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure::StratificationNumberClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return dissipation_stability_function(closure.stability_functions, αᴺ, αˢ)
end

@inline function dissipation_stability_functionᶜᶜᶜ(i, j, k, grid, closure::StratificationNumberClosure, velocities, N², e, ℓ)
    αᴺ, αˢ = stratification_and_shear_numbersᶜᶜᶜ(i, j, k, grid, closure, velocities, N², e, ℓ)
    return dissipation_stability_function(closure.stability_functions, αᴺ, αˢ)
end
