#####
##### Slot: third moment of vertical velocity (compute_diag_third_shoc_moment and its helpers)
#####
##### A third-moment component implements
#####
#####     third_moment(tm, constants, τ_zi, N²_zi, w_sec_zi, thetal_zi, thedz, thedz2,
#####                  Δthl_sec, Δwthl_sec, Δw_sec, Δtke, wthl) -> w3
#####
##### at an interior interface: the diagnosis of the third moment from the second moments. The
##### method that takes a stratification number bound after the component,
#####
#####     third_moment(tm, stratification_number_bound, constants, ...) -> w3
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
#####     limit_third_moment(skewness, cp, w_sec_zi, w3) -> w3
#####
##### which returns a third moment no larger in magnitude than `maximum_skewness (w'w')^{3/2}`, that
##### is, a bounded skewness `w'³ / (w'w')^{3/2}`. The two slots are independent: any diagnosis goes
##### with any limiter.
#####

"""
$(TYPEDEF)

The third moment of vertical velocity diagnosed algebraically from the second moments and their
vertical gradients: the stationary third-moment equations of Canuto et al. (2001, J. Atmos. Sci.
58, 1169–1172) solved for `w'³`, as E3SM `shoc.F90` computes it
(`compute_diag_third_shoc_moment`). With the return-to-isotropy time scale `τ`, the stratification
`N²` and the stratification number `αᴺ = τ² N²` (the notation of Oceananigans'
`TKEDissipationVerticalDiffusivity`),

    w'³ = (A₁ - ℂʷ₉ X₁ - ℂʷ₁₀ f₅) / (c - ℂʷ₉ X₀ + A₀),    c = ℂʷ₂

where `f₀` to `f₅` are products of `τ`, the buoyancy parameter `g / θˡ`, the second moments and
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
their ratio stays finite, so `w'³` itself has a finite limit there; the first singularity of `w'³`
in stable air is where its denominator `c - ℂʷ₉ X₀ + A₀` vanishes, at `αᴺ ≈ 24.3`, 0.85 of the
first pole with the defaults. The integers `3`, `5` and `2` of `a₀` to `a₅` and of
`ω₁`, and the factors `1/2` and `2` of `f₁` and `f₂`, belong to the derivation and are not fields.

The stratification number enters the expression through the closure's
`stratification_number_bound`: [`UnboundedStratificationNumber`](@ref), as in `shoc.F90`, or
[`RationalSaturatedStratificationNumber`](@ref), which keeps it below the first pole.

$(TYPEDFIELDS)
"""
struct AlgebraicThirdMoment{FT} <: AbstractSHOCComponent
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
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `AlgebraicThirdMoment{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
AlgebraicThirdMoment(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(AlgebraicThirdMoment, FT,
                            (ℂʷ₂ = 7, ℂʷ₃ = 0.52, ℂʷ₄ = 0.87, ℂʷ₅ = 0.5, ℂʷ₆ = 0.6, ℂʷ₇ = 2.4, ℂʷ₈ = 0.6, ℂʷ₉ = 1.2,
                             ℂʷ₁₀ = 1.5, ℂʷ₁₁ = 1.25), kw)

Adapt.@adapt_structure AlgebraicThirdMoment

# The third moment at an interior interface from the interface values of isotropy, Brunt-Vaisala
# frequency, w_sec and thetal, the inverse thicknesses `thedz` and `thedz2`, and the differences
# of the moments across the interface; `wthl` is wthl_sec at the interface.
@inline third_moment(tm::AlgebraicThirdMoment, constants::SHOCConstants, arguments...) =
    third_moment(tm, UnboundedStratificationNumber(), constants, arguments...)

@inline function third_moment(tm::AlgebraicThirdMoment, stratification_number_bound, constants, isotropy_zi, brunt_zi, w_sec_zi, thetal_zi, thedz, thedz2,
                              thl_sec_diff, wthl_sec_diff, wsec_diff, tke_diff, wthl)
    FT = typeof(wthl)
    c = tm.ℂʷ₂

    a0 = (tm.ℂʷ₃ * (1 / (c * c))) / (c - 2)
    a1 = tm.ℂʷ₄ / (c * c)
    a2 = tm.ℂʷ₅ / c
    a3 = tm.ℂʷ₆ / (c * (c - 2))
    first_pole = 1 / (a1 + a3)

    # fterms_input_for_diag_third_shoc_moment
    iso = isotropy_zi
    isosqrd = iso * iso
    αᴺ = bound_stratification_number(stratification_number_bound, isosqrd * brunt_zi, first_pole)
    bet2 = constants.ggr / thetal_zi

    f0 = thedz2 * (bet2 * bet2 * bet2) * ((iso * iso) * (iso * iso)) * wthl * thl_sec_diff
    f1 = thedz2 * (bet2 * bet2) * (iso * iso * iso) * (wthl * wthl_sec_diff + convert(FT, 0.5) * w_sec_zi * thl_sec_diff)
    f2 = thedz * bet2 * isosqrd * wthl * wsec_diff + 2 * thedz2 * bet2 * isosqrd * w_sec_zi * wthl_sec_diff
    f3 = thedz2 * bet2 * isosqrd * w_sec_zi * wthl_sec_diff + thedz * bet2 * isosqrd * (wthl * tke_diff)
    f4 = thedz * iso * w_sec_zi * (wsec_diff + tke_diff)
    f5 = thedz * iso * w_sec_zi * wsec_diff

    # omega_terms_diag_third_shoc_moment
    a4 = tm.ℂʷ₇ / (3 * c + 5)
    a5 = tm.ℂʷ₈ / (c * (3 + 5 * c))
    omega0 = a4 / (1 - a5 * αᴺ)
    omega1 = omega0 / (2 * c)
    omega2 = omega1 * f3 + tm.ℂʷ₁₁ * omega0 * f4

    # x_y_terms_diag_third_shoc_moment
    x0 = (a2 * αᴺ * (1 - a3 * αᴺ)) / (1 - (a1 + a3) * αᴺ)
    y0 = (2 * a2 * αᴺ * x0) / (1 - a3 * αᴺ)
    x1 = (a0 * f0 + a1 * f1 + a2 * (1 - a3 * αᴺ) * f2) / (1 - (a1 + a3) * αᴺ)
    y1 = (2 * a2 * (αᴺ * x1 + (a0 / a1) * f0 + f1)) / (1 - a3 * αᴺ)

    # aa_terms_diag_third_shoc_moment
    aa0 = omega0 * x0 + omega1 * y0
    aa1 = omega0 * x1 + omega1 * y1 + omega2

    # w3_diag_third_shoc_moment
    return (aa1 - tm.ℂʷ₉ * x1 - tm.ℂʷ₁₀ * f5) / (c - tm.ℂʷ₉ * x0 + aa0)
end

#####
##### Stratification number bound: the stratification number of the third-moment diagnosis
#####

"""
Supertype of the components that bound the stratification number `αᴺ = τ² N²` of the diagnosis
of the third moment of vertical velocity ([`AlgebraicThirdMoment`](@ref)). The bound applies
there only: the time scale `τ` and the frequency `N²` used elsewhere in the closure are unchanged.
"""
abstract type AbstractStratificationNumberBound <: AbstractSHOCComponent end

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
`maximum_skewness`, the largest magnitude of the skewness `w'³ / (w'w')^{3/2}`: the bound of the
third moment is

    w'³ᵐᵃˣ = maximum_skewness (w'w')^{3/2}

The default, `1.2 √2 ≈ 1.697`, is the bound of E3SM `shoc.F90`, which writes it
`w3clip √(2 w'w'³)` with `w3clip = 1.2`.
"""
abstract type AbstractSkewnessLimiter <: AbstractSHOCComponent end

# The largest skewness of E3SM `shoc.F90`: `w3clip √2` with `w3clip = 1.2`
const default_maximum_skewness = 1.2 * sqrt(2)

# The largest magnitude of the third moment; `variance` is w_sec interpolated to the interface
@inline third_moment_bound(skewness::AbstractSkewnessLimiter, variance) =
    skewness.maximum_skewness * (variance * sqrt(variance))

"""
$(TYPEDEF)

A third moment whose magnitude exceeds the bound `maximum_skewness (w'w')^{3/2}` is reset to the
constant `reset_third_moment`, whatever its sign; a third moment within the bound is unchanged.
This is `clipping_diag_third_shoc_moments` of E3SM `shoc.F90`. The third moment jumps where it
crosses the bound: from `±w'³ᵐᵃˣ` to `+reset_third_moment`.

$(TYPEDFIELDS)
"""
struct ResetAboveThresholdSkewness{FT} <: AbstractSkewnessLimiter
    "Largest magnitude of the skewness `w'³ / (w'w')^{3/2}` (`w3clip √2` of `shoc.F90`)"
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

@inline function limit_third_moment(skewness::ResetAboveThresholdSkewness, cp, variance, w3)
    bound = third_moment_bound(skewness, variance)
    return ifelse(abs(w3) > bound, skewness.reset_third_moment, w3)
end

"""
$(TYPEDEF)

The third moment saturates toward `±w'³ᵐᵃˣ = ±maximum_skewness (w'w')^{3/2}` along an odd quintic
polynomial. With `u = (8 / 15) w'³ / w'³ᵐᵃˣ` limited to `[-1, 1]`,

    w'³ ← (15 / 8) w'³ᵐᵃˣ u (1 - (2/3) u² + (1/5) u⁴)

which

- is odd, so the sign of the third moment is kept, and increasing: its slope is `(1 - u²)²`;
- has slope one at zero: a third moment at a tenth of the bound is changed by 0.2 %, at half
  the bound by 4.6 %, at the bound by 17 %;
- reaches the bound exactly at `|w'³| = (15 / 8) w'³ᵐᵃˣ` with zero slope and zero curvature, and is
  the bound beyond, so that value, first and second derivative are continuous everywhere;
- costs one division and no transcendental function.

The numbers `15/8`, `2/3` and `1/5` are fixed by those conditions, not fitted. The slope with
respect to the diagnosed third moment is nonzero up to 1.875 times the bound, which covers the
threshold at which [`ResetAboveThresholdSkewness`](@ref) jumps. The bound is taken as zero where
the vertical velocity variance is not positive, and its square root is guarded there.

$(TYPEDFIELDS)
"""
struct PolynomialSaturatedSkewness{FT} <: AbstractSkewnessLimiter
    "Largest magnitude of the skewness `w'³ / (w'w')^{3/2}` (`w3clip √2` of `shoc.F90`)"
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

@inline function limit_third_moment(skewness::PolynomialSaturatedSkewness, cp, variance, w3)
    FT = typeof(w3)
    bound = third_moment_bound(skewness, variance)
    reach = (convert(FT, 15) / 8) * bound # the magnitude at which the bound is reached
    bounded = reach > 0
    u = max(-one(FT), min(one(FT), w3 / ifelse(bounded, reach, one(FT))))
    u² = u * u
    saturated = reach * u * (1 - u² * (convert(FT, 2) / 3 - u² / 5))
    # The polynomial is the bound at |u| = 1 but for round-off, which the hard bound removes
    saturated = max(-bound, min(bound, saturated))
    return ifelse(bounded, saturated, zero(FT))
end

"""
$(TYPEDEF)

A third moment beyond the bound `b = maximum_skewness (w'w')^{3/2}` is set to the bound of its
sign: `w'³ ← clip_between(w'³, -b, b)` through the clipping policy. Continuous, with a kink at the
bound under [`HardClipping`](@ref); under [`SmoothClipping`](@ref) the kink is rounded over its
width times `b`.

$(TYPEDFIELDS)
"""
struct ClampedSkewness{FT} <: AbstractSkewnessLimiter
    "Largest magnitude of the skewness `w'³ / (w'w')^{3/2}` (`w3clip √2` of `shoc.F90`)"
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
@inline function limit_third_moment(skewness::ClampedSkewness, cp, variance, w3)
    bound = third_moment_bound(skewness, variance)
    return clip_between(cp, w3, -bound, bound, bound)
end
