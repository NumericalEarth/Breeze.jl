#####
##### Slot: third moment of vertical velocity (compute_diag_third_shoc_moment and its helpers)
#####
##### A third-moment component implements
#####
#####     third_moment(tm, g, τ, N², w′², θˡⁱ, compact_inverse_spacing, wide_inverse_spacing,
#####                  Δθˡⁱ′², Δw′θˡⁱ′, Δw′², Δe, w′θˡⁱ′) -> w′³
#####
##### at an interior interface: the diagnosis of the third moment from the second moments. The
##### method that takes a stratification number bound after the component,
#####
#####     third_moment(tm, stratification_number_bound, g, ...) -> w′³
#####
##### applies the bound to the stratification number αᴺ = τ² N² inside the diagnosis; the method
##### without it leaves αᴺ unbounded.
#####
##### Slot: stratification number bound (no counterpart in shoc.F90)
#####
##### A stratification number bound implements
#####
#####     bound_stratification_number(bound, αᴺ, first_pole) -> αᴺ
#####
##### where `first_pole = 1 / (a₁ + a₃)` is the smallest positive stratification number at which
##### the diagnosis has a pole.
#####
##### Slot: skewness (clipping_diag_third_shoc_moments)
#####
##### A skewness component keeps the diagnosed third moment realizable: it implements
#####
#####     limit_third_moment(skewness, w′², w′³) -> w′³
#####
##### which returns a third moment no larger in magnitude than `maximum_skewness (w′²)^{3/2}`, that
##### is, a bounded skewness `w′³ / (w′²)^{3/2}`. The two slots are independent: any diagnosis goes
##### with any limiter.
#####

"""
$(TYPEDEF)

The third moment of vertical velocity diagnosed algebraically from the second moments and their
vertical gradients: the stationary third-moment equations of Canuto et al. (2001, J. Atmos. Sci.
58, 1169–1172) solved for `w′³`, as E3SM `shoc.F90` computes it
(`compute_diag_third_shoc_moment`). With the return-to-isotropy time scale `τ`, the stratification
`N²` and the stratification number `αᴺ = τ² N²` (the notation of Oceananigans'
`TKEDissipationVerticalDiffusivity`),

    w′³ = (A₁ - ℂʷ₉ X₁ - ℂʷ₁₀ f₅) / (c - ℂʷ₉ X₀ + A₀),    c = ℂʷ₂

where `f₀` to `f₅` are products of `τ`, the buoyancy parameter `g / θˡⁱ`, the second moments and
their gradients, and

    X₀ = a₂ αᴺ (1 - a₃ αᴺ) / (1 - (a₁ + a₃) αᴺ),          Y₀ = 2 a₂ αᴺ X₀ / (1 - a₃ αᴺ)
    X₁ = (a₀ f₀ + a₁ f₁ + a₂ (1 - a₃ αᴺ) f₂) / (1 - (a₁ + a₃) αᴺ)
    Y₁ = 2 a₂ (αᴺ X₁ + (a₀ / a₁) f₀ + f₁) / (1 - a₃ αᴺ)
    ω₀ = a₄ / (1 - a₅ αᴺ),    ω₁ = ω₀ / (2 c),    ω₂ = ω₁ f₃ + ℂʷ₁₁ ω₀ f₄
    A₀ = ω₀ X₀ + ω₁ Y₀,      A₁ = ω₀ X₁ + ω₁ Y₁ + ω₂

The numbers `a₀` to `a₅` follow from `c` and the numerators below, so that changing `c` alone
changes them as it does in `shoc.F90`:

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂʷ₂`  | `c_diag_3rd_mom` | 7 | the coefficient `c` |
| `ℂʷ₃`  | literal `0.52` | 0.52 | `a₀ = ℂʷ₃ / (c² (c - 2))` |
| `ℂʷ₄`  | literal `0.87` | 0.87 | `a₁ = ℂʷ₄ / c²` |
| `ℂʷ₅`  | literal `0.5`  | 0.5  | `a₂ = ℂʷ₅ / c` |
| `ℂʷ₆`  | literal `0.6`  | 0.6  | `a₃ = ℂʷ₆ / (c (c - 2))` |
| `ℂʷ₇`  | literal `2.4`  | 2.4  | `a₄ = ℂʷ₇ / (3 c + 5)` |
| `ℂʷ₈`  | literal `0.6`  | 0.6  | `a₅ = ℂʷ₈ / (c (3 + 5 c))` |
| `ℂʷ₉`  | literal `1.2`  | 1.2  | weight of `X₀` and `X₁` |
| `ℂʷ₁₀` | literal `1.5`  | 1.5  | weight of `f₅` |
| `ℂʷ₁₁` | literal `5/4`  | 1.25 | weight of `f₄` in `ω₂` |

The expression has poles in stable air, where `αᴺ` reaches `1 / (a₁ + a₃)`, `1 / a₃` and `1 / a₅`
(28.7, 58.3 and 443 with the defaults). At the first of these `X₀` and `X₁` diverge together and
their ratio stays finite, so `w′³` itself has a finite limit there; the first singularity of `w′³`
in stable air is where its denominator `c - ℂʷ₉ X₀ + A₀` vanishes, at `αᴺ ≈ 24.3`, 0.85 of the
first pole with the defaults. The integers `3`, `5` and `2` of `a₀` to `a₅` and of
`ω₁`, and the factors `1/2` and `2` of `f₁` and `f₂`, belong to the derivation and are not fields.

The stratification number enters the expression through the closure's
`stratification_number_bound`: [`UnboundedStratificationNumber`](@ref), as in `shoc.F90`, or
[`RationalSaturatedStratificationNumber`](@ref), which keeps it below the first pole.

$(TYPEDFIELDS)
"""
struct AlgebraicThirdMoment{FT, L} <: AbstractSubgridDistributionComponent
    "Coefficient `c` of the diagnostic third moment of vertical velocity (`c_diag_3rd_mom`)"
    ℂʷ₂ :: FT
    "Numerator of `a₀` (literal `0.52`)"
    ℂʷ₃ :: FT
    "Numerator of `a₁` (literal `0.87`)"
    ℂʷ₄ :: FT
    "Numerator of `a₂` (literal `0.5`)"
    ℂʷ₅ :: FT
    "Numerator of `a₃` (literal `0.6`)"
    ℂʷ₆ :: FT
    "Numerator of `a₄` (literal `2.4`)"
    ℂʷ₇ :: FT
    "Numerator of `a₅` (literal `0.6`)"
    ℂʷ₈ :: FT
    "Weight of `X₀` and `X₁` in the third moment (literal `1.2`)"
    ℂʷ₉ :: FT
    "Weight of `f₅` in the third moment (literal `1.5`)"
    ℂʷ₁₀ :: FT
    "Weight of `f₄` in `ω₂` (literal `5/4`)"
    ℂʷ₁₁ :: FT
    "Where the third moment is diagnosed: `Face()` (the default, as `shoc.F90`) or `Center()`"
    location :: L
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `AlgebraicThirdMoment{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument. `location` is where the third moment is diagnosed: at the faces (`Face()`, the default), with the
stencils of `shoc.F90`, and interpolated to the centers where the assumed PDF reads it, or at the centers
(`Center()`).
"""
function AlgebraicThirdMoment(FT::DataType = Oceananigans.defaults.FloatType; location = Face(), kw...)
    defaults = (ℂʷ₂ = 7, ℂʷ₃ = 0.52, ℂʷ₄ = 0.87, ℂʷ₅ = 0.5, ℂʷ₆ = 0.6, ℂʷ₇ = 2.4, ℂʷ₈ = 0.6, ℂʷ₉ = 1.2, ℂʷ₁₀ = 1.5, ℂʷ₁₁ = 1.25)
    names = keys(defaults)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("$(setdiff(keys(kw), names)) are not fields of AlgebraicThirdMoment; its fields are $(names) and location."))
    validate_third_moment_location(location)
    values = map(value -> convert(FT, value), Tuple(merge(defaults, NamedTuple(kw))))
    return AlgebraicThirdMoment(values..., location)
end

validate_third_moment_location(::Union{Face, Center}) = nothing
validate_third_moment_location(location) =
    throw(ArgumentError("The location of a third moment is Face() or Center(), not $(location)."))

Adapt.@adapt_structure AlgebraicThirdMoment

# The third moment at an interior interface from the interface values of the return-to-isotropy time scale `τ`, the
# buoyancy frequency `N²`, the vertical velocity variance `w′²` and `θˡⁱ`, the inverse distances over which the
# differences are taken (`compact_inverse_spacing` between the two adjacent nodes, `wide_inverse_spacing` across the
# two cells), and the differences of the moments across the interface; `w′θˡⁱ′` is the flux at the interface.
# `g` is the gravitational acceleration of the model.
@inline third_moment(tm::AlgebraicThirdMoment, g::Number, arguments...) =
    third_moment(tm, UnboundedStratificationNumber(), g, arguments...)

@inline function third_moment(tm::AlgebraicThirdMoment, stratification_number_bound, g, τ, N², w′², θˡⁱ,
                              compact_inverse_spacing, wide_inverse_spacing, Δθˡⁱ′², Δw′θˡⁱ′, Δw′², Δe, w′θˡⁱ′)
    FT = typeof(w′θˡⁱ′)
    c = tm.ℂʷ₂

    a₀ = (tm.ℂʷ₃ * (1 / (c * c))) / (c - 2)
    a₁ = tm.ℂʷ₄ / (c * c)
    a₂ = tm.ℂʷ₅ / c
    a₃ = tm.ℂʷ₆ / (c * (c - 2))
    first_pole = 1 / (a₁ + a₃)

    # fterms_input_for_diag_third_shoc_moment
    τ² = τ * τ
    αᴺ = bound_stratification_number(stratification_number_bound, τ² * N², first_pole)
    β = g / θˡⁱ

    f₀ = wide_inverse_spacing * (β * β * β) * ((τ * τ) * (τ * τ)) * w′θˡⁱ′ * Δθˡⁱ′²
    f₁ = wide_inverse_spacing * (β * β) * (τ * τ * τ) * (w′θˡⁱ′ * Δw′θˡⁱ′ + convert(FT, 0.5) * w′² * Δθˡⁱ′²)
    f₂ = compact_inverse_spacing * β * τ² * w′θˡⁱ′ * Δw′² + 2 * wide_inverse_spacing * β * τ² * w′² * Δw′θˡⁱ′
    f₃ = wide_inverse_spacing * β * τ² * w′² * Δw′θˡⁱ′ + compact_inverse_spacing * β * τ² * (w′θˡⁱ′ * Δe)
    f₄ = compact_inverse_spacing * τ * w′² * (Δw′² + Δe)
    f₅ = compact_inverse_spacing * τ * w′² * Δw′²

    # omega_terms_diag_third_shoc_moment
    a₄ = tm.ℂʷ₇ / (3 * c + 5)
    a₅ = tm.ℂʷ₈ / (c * (3 + 5 * c))
    ω₀ = a₄ / (1 - a₅ * αᴺ)
    ω₁ = ω₀ / (2 * c)
    ω₂ = ω₁ * f₃ + tm.ℂʷ₁₁ * ω₀ * f₄

    # x_y_terms_diag_third_shoc_moment
    X₀ = (a₂ * αᴺ * (1 - a₃ * αᴺ)) / (1 - (a₁ + a₃) * αᴺ)
    Y₀ = (2 * a₂ * αᴺ * X₀) / (1 - a₃ * αᴺ)
    X₁ = (a₀ * f₀ + a₁ * f₁ + a₂ * (1 - a₃ * αᴺ) * f₂) / (1 - (a₁ + a₃) * αᴺ)
    Y₁ = (2 * a₂ * (αᴺ * X₁ + (a₀ / a₁) * f₀ + f₁)) / (1 - a₃ * αᴺ)

    # aa_terms_diag_third_shoc_moment
    A₀ = ω₀ * X₀ + ω₁ * Y₀
    A₁ = ω₀ * X₁ + ω₁ * Y₁ + ω₂

    # w3_diag_third_shoc_moment
    return (A₁ - tm.ℂʷ₉ * X₁ - tm.ℂʷ₁₀ * f₅) / (c - tm.ℂʷ₉ * X₀ + A₀)
end

#####
##### Stratification number bound: the stratification number of the third-moment diagnosis
#####

"""
Supertype of the components that bound the stratification number `αᴺ = τ² N²` of the diagnosis
of the third moment of vertical velocity ([`AlgebraicThirdMoment`](@ref)). The bound applies
there only: the time scale `τ` and the frequency `N²` used elsewhere in the closure are unchanged.
"""
abstract type AbstractStratificationNumberBound <: AbstractSubgridDistributionComponent end

"""
$(TYPEDEF)

The stratification number enters the diagnosis of the third moment as it is, as in E3SM
`shoc.F90`. The diagnosis is then singular in stable air: its denominator vanishes at
`αᴺ ≈ 24.3` and the terms `X₀`, `X₁` have a pole at `αᴺ = 1 / (a₁ + a₃) = 28.7` (with the
coefficients of `shoc.F90`).
"""
struct UnboundedStratificationNumber <: AbstractStratificationNumberBound end

Adapt.@adapt_structure UnboundedStratificationNumber

@inline bound_stratification_number(::UnboundedStratificationNumber, αᴺ, first_pole) = αᴺ

"""
$(TYPEDEF)

In stable stratification the stratification number saturates toward `αᴺᵐᵃˣ`,

    αᴺ ← αᴺ / (1 + αᴺ / αᴺᵐᵃˣ)    for αᴺ > 0,    αᴺ ← αᴺ    for αᴺ ≤ 0,

    αᴺᵐᵃˣ = pole_fraction / (a₁ + a₃)

where `1 / (a₁ + a₃)` is the first pole of the diagnosis of [`AlgebraicThirdMoment`](@ref), the
smallest positive `αᴺ` at which `1 - (a₁ + a₃) αᴺ` vanishes. The bounded value is below `αᴺᵐᵃˣ`
for every `αᴺ`, so with `pole_fraction < 1` the stratification number stays a fixed fraction
below the first pole (and below the poles at `1 / a₃` and `1 / a₅`, which are further out).
Because `αᴺᵐᵃˣ` follows from the coefficients of the diagnosis, it moves with them when they are
calibrated. The denominator of the diagnosis vanishes a little before the first pole (at 0.85 of
it with the coefficients of `shoc.F90`), so `pole_fraction` is to stay below that: with the
default `1/2` the denominator is at least `5.3`, against `c = 7` in neutral stratification.

The map is increasing, has slope one and value zero at `αᴺ = 0`, so value and first derivative
are continuous everywhere; the second derivative jumps at zero from `0` to `-2 / αᴺᵐᵃˣ`. Weak
stratification is changed little: by 7 % at `αᴺ = αᴺᵐᵃˣ / 14`, and unstable stratification not
at all. One division, no transcendental function. With `pole_fraction = 1/2` (the default)
`αᴺᵐᵃˣ = 14.3` with the coefficients of `shoc.F90`.

$(TYPEDFIELDS)
"""
struct RationalSaturatedStratificationNumber{FT} <: AbstractStratificationNumberBound
    "Upper bound of the stratification number as a fraction of the first pole of the diagnosis, `γ`"
    pole_fraction :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `RationalSaturatedStratificationNumber{FT}` with `pole_fraction = 1/2`.
"""
RationalSaturatedStratificationNumber(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(RationalSaturatedStratificationNumber, FT, (; pole_fraction = 0.5), kw)

Adapt.@adapt_structure RationalSaturatedStratificationNumber

# One expression for both signs: the denominator is one where αᴺ ≤ 0
@inline function bound_stratification_number(bound::RationalSaturatedStratificationNumber, αᴺ, first_pole)
    αᴺᵐᵃˣ = bound.pole_fraction * first_pole
    return αᴺ / (1 + max(αᴺ, zero(αᴺ)) / αᴺᵐᵃˣ)
end

#####
##### Skewness: how the third moment is kept within its bound
#####

"""
Supertype of the components that limit the third moment of vertical velocity. Each holds
`maximum_skewness`, the largest magnitude of the skewness `w′³ / (w′²)^{3/2}`: the bound of the
third moment is

    w′³ᵐᵃˣ = maximum_skewness (w′²)^{3/2}

The default, `1.2 √2 ≈ 1.697`, is the bound of E3SM `shoc.F90`, which writes it
`w3clip √(2 (w′²)³)` with `w3clip = 1.2`.
"""
abstract type AbstractSkewnessLimiter <: AbstractSubgridDistributionComponent end

# The largest skewness of E3SM `shoc.F90`: `w3clip √2` with `w3clip = 1.2`
const default_maximum_skewness = 1.2 * sqrt(2)

# The largest magnitude of the third moment; `variance` is w′² interpolated to the interface
@inline third_moment_bound(skewness::AbstractSkewnessLimiter, variance) =
    skewness.maximum_skewness * (variance * sqrt(variance))

"""
$(TYPEDEF)

A third moment whose magnitude exceeds the bound `maximum_skewness (w′²)^{3/2}` is reset to the
constant `reset_third_moment`, whatever its sign; a third moment within the bound is unchanged.
This is `clipping_diag_third_shoc_moments` of E3SM `shoc.F90`. The third moment jumps where it
crosses the bound: from `±w′³ᵐᵃˣ` to `+reset_third_moment`.

$(TYPEDFIELDS)
"""
struct ResetAboveThresholdSkewness{FT} <: AbstractSkewnessLimiter
    "Largest magnitude of the skewness `w′³ / (w′²)^{3/2}` (`w3clip √2` of `shoc.F90`)"
    maximum_skewness :: FT
    "The value a third moment beyond the bound is reset to, m³ s⁻³ (`w3clipdef` of `shoc.F90`); dimensional"
    reset_third_moment :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ResetAboveThresholdSkewness{FT}` with the values of E3SM `shoc.F90`:
`maximum_skewness = 1.2 √2` and `reset_third_moment = 0.02` m³ s⁻³. Every field is a keyword
argument.
"""
ResetAboveThresholdSkewness(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ResetAboveThresholdSkewness, FT,
                            (maximum_skewness = default_maximum_skewness, reset_third_moment = 0.02), kw)

Adapt.@adapt_structure ResetAboveThresholdSkewness

@inline function limit_third_moment(skewness::ResetAboveThresholdSkewness, variance, w′³)
    bound = third_moment_bound(skewness, variance)
    return ifelse(abs(w′³) > bound, skewness.reset_third_moment, w′³)
end

"""
$(TYPEDEF)

The third moment saturates toward `±w′³ᵐᵃˣ = ±maximum_skewness (w′²)^{3/2}` along an odd quintic
polynomial. With `u = (8 / 15) w′³ / w′³ᵐᵃˣ` limited to `[-1, 1]`,

    w′³ ← (15 / 8) w′³ᵐᵃˣ u (1 - (2/3) u² + (1/5) u⁴)

which

- is odd, so the sign of the third moment is kept, and increasing: its slope is `(1 - u²)²`;
- has slope one at zero: a third moment at a tenth of the bound is changed by 0.2 %, at half
  the bound by 4.6 %, at the bound by 17 %;
- reaches the bound exactly at `|w′³| = (15 / 8) w′³ᵐᵃˣ` with zero slope and zero curvature, and is
  the bound beyond, so that value, first and second derivative are continuous everywhere;
- costs one division and no transcendental function.

The numbers `15/8`, `2/3` and `1/5` are fixed by those conditions, not fitted. The slope with
respect to the diagnosed third moment is nonzero up to 1.875 times the bound, which covers the
threshold at which [`ResetAboveThresholdSkewness`](@ref) jumps. The bound is taken as zero where
the vertical velocity variance is not positive, and its square root is guarded there.

$(TYPEDFIELDS)
"""
struct PolynomialSaturatedSkewness{FT} <: AbstractSkewnessLimiter
    "Largest magnitude of the skewness `w′³ / (w′²)^{3/2}` (`w3clip √2` of `shoc.F90`)"
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

# The bound with a guarded square root: zero, with zero slope, where the variance is not positive
@inline function third_moment_bound(skewness::PolynomialSaturatedSkewness, variance)
    variance = max(variance, zero(variance))
    return skewness.maximum_skewness * (variance * zero_safe_sqrt(variance))
end

@inline function limit_third_moment(skewness::PolynomialSaturatedSkewness, variance, w′³)
    FT = typeof(w′³)
    bound = third_moment_bound(skewness, variance)
    reach = (convert(FT, 15) / 8) * bound # the magnitude at which the bound is reached
    bounded = reach > 0
    u = max(-one(FT), min(one(FT), w′³ / ifelse(bounded, reach, one(FT))))
    u² = u * u
    saturated = reach * u * (1 - u² * (convert(FT, 2) / 3 - u² / 5))
    # The polynomial is the bound at |u| = 1 but for round-off, which the hard bound removes
    saturated = max(-bound, min(bound, saturated))
    return ifelse(bounded, saturated, zero(FT))
end

"""
$(TYPEDEF)

The third moment saturates toward `±w′³ᵐᵃˣ = ±maximum_skewness (w′²)^{3/2}` along a hyperbolic tangent,

    w′³ ← w′³ᵐᵃˣ tanh(w′³ / w′³ᵐᵃˣ)

so that the skewness `w′³ / (w′²)^{3/2}` lies strictly inside `±maximum_skewness` for every input. The map

- is odd and increasing, with slope `sech²(w′³ / w′³ᵐᵃˣ)`: one at zero, positive everywhere (unlike
  [`PolynomialSaturatedSkewness`](@ref), whose slope is zero beyond 15/8 of the bound), at most one;
- has bounded sensitivity to the variance too: `∂w′³/∂w′³ᵐᵃˣ = tanh(y) - y sech²(y) ∈ [0, 1)`, `y = w′³ / w′³ᵐᵃˣ`;
- changes a third moment at a tenth of the bound by 0.3 %, at half the bound by 7.6 %, at the bound by 24 %;
- is zero, with zero slope in the variance, where the variance is not positive.

The default `maximum_skewness = 2.5` is above the skewness of the large-eddy simulations the third-moment
models were compared with (|Sk| ≲ 2 in cumulus), so the map is a guard and not a fitted part of the model.

$(TYPEDFIELDS)
"""
struct TanhSaturatedSkewness{FT} <: AbstractSkewnessLimiter
    "Supremum of the magnitude of the skewness `w′³ / (w′²)^{3/2}`"
    maximum_skewness :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `TanhSaturatedSkewness{FT}` with `maximum_skewness = 2.5`.
"""
TanhSaturatedSkewness(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(TanhSaturatedSkewness, FT, (; maximum_skewness = 2.5), kw)

Adapt.@adapt_structure TanhSaturatedSkewness

@inline function limit_third_moment(skewness::TanhSaturatedSkewness, variance, w′³)
    FT = typeof(w′³)
    variance = max(variance, zero(variance))
    bound = skewness.maximum_skewness * (variance * zero_safe_sqrt(variance))
    bounded = bound > 0
    return ifelse(bounded, bound * tanh(w′³ / ifelse(bounded, bound, one(FT))), zero(FT))
end

"""
$(TYPEDEF)

A third moment beyond the bound `b = maximum_skewness (w′²)^{3/2}` is set to the bound of its
sign: `w′³ ← max(-b, min(b, w′³))`. Continuous, with a kink at the bound; [`PolynomialSaturatedSkewness`](@ref)
and [`TanhSaturatedSkewness`](@ref) bound it smoothly.

$(TYPEDFIELDS)
"""
struct ClampedSkewness{FT} <: AbstractSkewnessLimiter
    "Largest magnitude of the skewness `w′³ / (w′²)^{3/2}` (`w3clip √2` of `shoc.F90`)"
    maximum_skewness :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ClampedSkewness{FT}` with the bound of E3SM `shoc.F90`, `maximum_skewness = 1.2 √2`.
"""
ClampedSkewness(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ClampedSkewness, FT, (; maximum_skewness = default_maximum_skewness), kw)

Adapt.@adapt_structure ClampedSkewness

# The bound is positive: the interface variance it is made of is floored
@inline function limit_third_moment(skewness::ClampedSkewness, variance, w′³)
    bound = third_moment_bound(skewness, variance)
    return max(-bound, min(bound, w′³))
end
