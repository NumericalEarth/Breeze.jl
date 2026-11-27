#####
##### Slot: the assumed double-Gaussian PDF (shoc_assumed_pdf and its helpers)
#####
##### An assumed-PDF component implements
#####
#####     assumed_pdf(pdf, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
#####         -> (aᶜ, qˡ, qˡ′², w′qˡ′, w′θᵛ′)
#####
##### for a level at height `zᶜ` between the interfaces at `zᶠ⁻` and `zᶠ⁺`: from the pressure,
##### the means and the vertical velocity variance of the level and the interface moments
##### `(w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)` below and above it, the subgrid cloud
##### fraction, liquid water, liquid water variance, liquid water flux and buoyancy flux. The
##### `clipping` of the PDF (`SmoothClipping`) imposes its bounds: of the weight of the updraft component, of the
##### correlations and of the within-component variances.
#####

"""
$(TYPEDEF)

The Analytic Double Gaussian 1 of Larson et al. (2002), the assumed PDF of E3SM `shoc.F90`
(`shoc_assumed_pdf`): two Gaussian components in vertical velocity, θˡⁱ and qᵗ whose weights, means
and widths follow from the first three moments of vertical velocity and the second moments of
the scalars, each condensing where it is saturated.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵍ₁` | `w2_1`, `w2_2` (literal `0.4`) | 0.4 | within-component fraction of the vertical velocity variance |
| `ℂᵍ₂` | literal `1.2`    | 1.2    | skewness of total water relative to that of vertical velocity |
| `ℂᵍ₃` | literal `0.2`    | 0.2    | component separation at which the skewness of total water starts to rise |
| `ℂᵍ₄` | literal `0.4`    | 0.4    | component separation at which total water has taken the skewness `ℂᵍ₂ Sk` |

`shoc.F90` ramps the skewness of total water linearly in the separation `d` of its normalized component means, from
zero at `ℂᵍ₃` to `ℂᵍ₂ Sk` at `ℂᵍ₄`. Here the share of `ℂᵍ₂ Sk` that total water takes is the rational function

    xⁿ / (1 + xⁿ),    x = 2 d / (ℂᵍ₃ + ℂᵍ₄),    n = 2 (ℂᵍ₃ + ℂᵍ₄) / (ℂᵍ₄ - ℂᵍ₃),

in `(0, 1)` for every separation, one half at the middle of the ramp with the slope `1 / (ℂᵍ₄ - ℂᵍ₃)` of the ramp there
(`n = 6` with the defaults), and vanishing as `dⁿ` where the components coincide, so that the skewness a mixture of
coinciding components cannot have vanishes with their separation. `ℂᵍ₃ < ℂᵍ₄` is required.

Where `shoc.F90` switches (calm air, `w′² ≤ vertical_velocity_tolerance²`, and a uniform scalar, variance below its
tolerance squared, collapse the components onto the mean), the components here collapse continuously: the skewness
`w′³ / (w′²)^{3/2}` and the correlations `w′φ′ / (σw σφ)` are formed with a vertical velocity variance no smaller than
`vertical_velocity_tolerance²` and standard deviations of the scalars no smaller than their tolerances (smooth
maxima, through `clipping`), and the component means are `φ̄ + φ̂ᵢ √φ′²` and `w̄ + ŵᵢ √w′²` with the standard
deviations themselves, which vanish with the variances. Above the tolerances the PDF is that of `shoc.F90`. The
fields after the coefficients are tolerances and bounds, not empirical coefficients, and the `clipping` that imposes
the bounds ([`SmoothClipping`](@ref)).

$(TYPEDFIELDS)
"""
struct DoubleGaussianPDF{FT, CP} <: AbstractSubgridDistributionComponent
    "Within-component fraction of the vertical velocity variance (literal `0.4`)"
    ℂᵍ₁ :: FT
    "Skewness of total water relative to that of vertical velocity (literal `1.2`)"
    ℂᵍ₂ :: FT
    "Component separation below which total water is unskewed (literal `0.2`)"
    ℂᵍ₃ :: FT
    "Component separation above which total water is fully skewed (literal `0.4`)"
    ℂᵍ₄ :: FT
    "Least standard deviation of θˡⁱ in its correlations, K (`thl_tol`)"
    potential_temperature_tolerance :: FT
    "Least standard deviation of qᵗ in its correlations, kg kg⁻¹ (`rt_tol`)"
    total_water_tolerance :: FT
    "Least standard deviation of vertical velocity in the skewness and the correlations, m s⁻¹ (`w_tol`)"
    vertical_velocity_tolerance :: FT
    "Lowest weight of the updraft component (literal `0.01`)"
    minimum_updraft_fraction :: FT
    "Highest weight of the updraft component (literal `0.99`)"
    maximum_updraft_fraction :: FT
    "Highest variance of a scalar within a component over its variance (literal `100`)"
    maximum_component_variance_ratio :: FT
    "How the bounds are imposed: [`SmoothClipping`](@ref)"
    clipping :: CP
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `DoubleGaussianPDF` with the values of E3SM `shoc.F90`, its bounds imposed by `clipping`. Every field is a
keyword argument.
"""
DoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; clipping = SmoothClipping(FT), kw...) =
    component_from_keywords(DoubleGaussianPDF, FT,
                            (ℂᵍ₁ = 0.4, ℂᵍ₂ = 1.2, ℂᵍ₃ = 0.2, ℂᵍ₄ = 0.4,
                             potential_temperature_tolerance = 1e-2, total_water_tolerance = 1e-4,
                             vertical_velocity_tolerance = 2e-2,
                             minimum_updraft_fraction = 0.01, maximum_updraft_fraction = 0.99,
                             maximum_component_variance_ratio = 100), kw,
                            TurbulenceClosures.convert_eltype(FT, clipping))

Adapt.@adapt_structure DoubleGaussianPDF

"""
$(TYPEDEF)

The double-Gaussian PDF of [`DoubleGaussianPDF`](@ref) (the updraft component of weight `a↑` and the
downdraft component of weight `a↓ = 1 - a↑`, from the first three moments of vertical velocity)
with the within-component variances of θˡⁱ and qᵗ shared equally between the components, as for
vertical velocity:

    σᵢ² = (1 - r² / (1 - ℂᵍ)) φ′²,    r = w′φ′ / √(w′² φ′²),

so that the components' means carry the flux and the rest of the variance is within them.

The within-component fraction of `w′²` is a smooth function of the skewness `Sk` of w,

    ℂᵍ(Sk) = (1 + y / √(1 + y²)) / 2,    y = y₀ + ℂᵍˢ Sk + ℂᵍˢˢ Sk²,

an algebraic sigmoid in (0, 1) with `ℂᵍ(0) = ℂᵍ₀` (`y₀ = (2ℂᵍ₀ - 1) / √(1 - (2ℂᵍ₀ - 1)²)`; with zero slopes
`ℂᵍ = ℂᵍ₀` to rounding). It sets the weight of the updraft component, `a↑ = (1 - Sk / √(4 (1 - ℂᵍ)³ + Sk²)) / 2`, and with it the kurtosis of w: for this
equal-spread mixture

    Kurt = 3c² + 6c (1 - c) + (1 - c)² (1 - 3a↑ + 3a↑²) / (a↑ (1 - a↑)),    c = ℂᵍ(Sk),

so choosing `ℂᵍ(Sk)` is choosing a kurtosis–skewness relation. The realizability of the scalar
components requires `r² < 1 - ℂᵍ(Sk)`. The
scalar skewness follows (it is not imposed). SHOC's rule instead matches an imposed scalar
skewness (zero for θˡⁱ, a ramp of `ℂᵍ₂ Sk_w` for qᵗ), which for `|Sk_w| ≳ 1.25` asks for a negative
variance in the narrow component, which is bounded at zero: a component of no width, whose cloud is all
or nothing. The equal share needs `r² < 1 - ℂᵍ`, which the correlation the components carry satisfies by
construction: `r = r₀ / (1 + (r₀² / (1 - ℂᵍ))² + (r₀ σ / (φ̄ min|ŵᵢ|))⁴)^{1/4}` from the correlation `r₀` of the
moments, which also keeps the component means of total water positive. The
within-component θˡⁱ–qᵗ correlation is that of the covariance left after the components' means.

$(TYPEDFIELDS)
"""
struct EqualWidthDoubleGaussianPDF{FT, CP} <: AbstractSubgridDistributionComponent
    "Within-component fraction of the vertical velocity variance at zero skewness"
    ℂᵍ₀ :: FT
    "Slope of the within-component fraction in the skewness of w (in the sigmoid's argument)"
    ℂᵍˢ :: FT
    "Curvature of the within-component fraction in the skewness of w (in the sigmoid's argument)"
    ℂᵍˢˢ :: FT
    "Least standard deviation of θˡⁱ in its correlations, K"
    potential_temperature_tolerance :: FT
    "Least standard deviation of qᵗ in its correlations, kg kg⁻¹"
    total_water_tolerance :: FT
    "Least standard deviation of vertical velocity in the skewness and the correlations, m s⁻¹"
    vertical_velocity_tolerance :: FT
    "Lowest weight of the updraft component"
    minimum_updraft_fraction :: FT
    "Highest weight of the updraft component"
    maximum_updraft_fraction :: FT
    "How the bounds are imposed: [`SmoothClipping`](@ref)"
    clipping :: CP
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return an `EqualWidthDoubleGaussianPDF` with the coefficients and tolerances of [`DoubleGaussianPDF`](@ref), its
bounds imposed by `clipping`. Every field is a keyword argument.
"""
function EqualWidthDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; ℂᵍ₀ = 0.4, clipping = SmoothClipping(FT), kw...)
    0 < ℂᵍ₀ < 1 || throw(ArgumentError("ℂᵍ₀ must be in (0, 1), got $ℂᵍ₀"))
    return component_from_keywords(EqualWidthDoubleGaussianPDF, FT,
                                   (ℂᵍ₀ = ℂᵍ₀, ℂᵍˢ = 0, ℂᵍˢˢ = 0,
                                    potential_temperature_tolerance = 1e-2, total_water_tolerance = 1e-4,
                                    vertical_velocity_tolerance = 2e-2,
                                    minimum_updraft_fraction = 0.01, maximum_updraft_fraction = 0.99), kw,
                                   TurbulenceClosures.convert_eltype(FT, clipping))
end

Adapt.@adapt_structure EqualWidthDoubleGaussianPDF

const ComponentDoubleGaussianPDF = Union{DoubleGaussianPDF, EqualWidthDoubleGaussianPDF}

# The within-component fraction of the vertical velocity variance: SHOC's constant, or ℂᵍ(Sk)
@inline within_component_share(pdf::DoubleGaussianPDF, Sk) = pdf.ℂᵍ₁

@inline function within_component_share(pdf::EqualWidthDoubleGaussianPDF, Sk)
    # y₀ is derived here from ℂᵍ₀, not stored, so that rebinding ℂᵍ₀ (with_parameters) moves it; and there is no
    # branch on zero slopes, which would select the constant ℂᵍ₀ and zero the derivatives in ℂᵍˢ and ℂᵍˢˢ
    x = 2 * pdf.ℂᵍ₀ - 1
    y₀ = x / sqrt(1 - x * x)
    y = y₀ + pdf.ℂᵍˢ * Sk + pdf.ℂᵍˢˢ * Sk * Sk
    return (1 + y / sqrt(1 + y * y)) / 2
end

# The largest magnitude of the skewness of vertical velocity that the double Gaussian represents. Its two components
# have the same width in w, so its third moment is that of their means: with the weight a↑ of the updraft component
# and the within-component share c of w′²,
#
#     Sk = (1 - 2a↑) (1 - c)^{3/2} / √(a↑ (1 - a↑)),
#
# which grows without bound as a↑ → 0 or 1 and is largest, for the weights the PDF allows, at the least weight of
# either component. Beyond it the PDF holds a↑ at its bound and has less skewness than it was given.
@inline representable_skewness(pdf::ComponentDoubleGaussianPDF) =
    representable_skewness(pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction,
                           within_component_share(pdf, zero(pdf.vertical_velocity_tolerance)))

@inline function representable_skewness(aᵐⁱⁿ, aᵐᵃˣ, c)
    a = max(aᵐⁱⁿ, 1 - aᵐᵃˣ)
    return (1 - 2a) * (1 - c) * sqrt((1 - c) / (a * (1 - a)))
end

# shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters share one form: the component
# means and variances of a scalar from its variance, its flux correlation with w, and the
# skewness of the scalar (identically zero for θˡⁱ, where `dothetal_skew = false`). `x̂₁` and `x̂₂` are the
# component means relative to the mean, in units of the standard deviation.
@inline function component_variance(clipping, first_component, x̂₁, x̂₂, a, skewness, variance, maximum_ratio)
    unexplained = 1 - a * (x̂₁ * x̂₁) - (1 - a) * (x̂₂ * x̂₂)
    third = skewness - a * (x̂₁ * x̂₁ * x̂₁) - (1 - a) * (x̂₂ * x̂₂ * x̂₂)
    numerator = ifelse(first_component, 3 * x̂₂ * unexplained - third, -3 * x̂₁ * unexplained + third)
    denominator = ifelse(first_component, 3 * a * (x̂₂ - x̂₁), 3 * (1 - a) * (x̂₂ - x̂₁))
    # Where the component means coincide the ratio is 0 / 0, with the limit one: the variance is all within the components
    degenerate = denominator == 0
    ratio = ifelse(degenerate, one(numerator), numerator / ifelse(degenerate, one(denominator), denominator))
    return clip_between(clipping, ratio, 0, maximum_ratio, 1) * variance
end

# The share of the skewness ℂᵍ₂ Sk that total water takes at the separation `d` of its normalized component means:
# xⁿ / (1 + xⁿ) with x = 2 d / (ℂᵍ₃ + ℂᵍ₄) and n = 2 (ℂᵍ₃ + ℂᵍ₄) / (ℂᵍ₄ - ℂᵍ₃). Zero, with zero slope, at d = 0.
@inline function skewness_share(pdf, d)
    FT = typeof(d)
    n = 2 * (pdf.ℂᵍ₃ + pdf.ℂᵍ₄) / (pdf.ℂᵍ₄ - pdf.ℂᵍ₃)
    separated = d > 0
    x = ifelse(separated, 2 * d / (pdf.ℂᵍ₃ + pdf.ℂᵍ₄), one(FT))
    xⁿ = x^n
    return ifelse(separated, xⁿ / (1 + xⁿ), zero(FT))
end

# The means `x₁`, `x₂`, variances `x′²₁`, `x′²₂` and spreads of a scalar in the two components, from its `flux`, its
# `variance` and its mean `x̄`. `σw₊` and `σ₊` are the standard deviations of vertical velocity and of the scalar that
# form the correlation, each no smaller than its tolerance, and `ŵ₁ > 0 > ŵ₂` the component velocities in units of the
# standard deviation of vertical velocity. The component means are apart by the standard deviation `√variance`
# itself, so that the components collapse onto the mean as the variance vanishes.
@inline function scalar_component_parameters(clipping, flux, σw₊, σ₊, variance, x̄, ŵ₁, ŵ₂, Sk, a, skewed, pdf::DoubleGaussianPDF)
    FT = typeof(variance)
    r = clip_between(clipping, flux / (σw₊ * σ₊), -1, 1, 1)

    # the normalized component means
    x̂₁ = -r / ŵ₂
    x̂₂ = -r / ŵ₁

    skewness = ifelse(skewed, pdf.ℂᵍ₂ * Sk * skewness_share(pdf, abs(x̂₂ - x̂₁)), zero(FT))

    x′²₁ = component_variance(clipping, true, x̂₁, x̂₂, a, skewness, variance, pdf.maximum_component_variance_ratio)
    x′²₂ = component_variance(clipping, false, x̂₁, x̂₂, a, skewness, variance, pdf.maximum_component_variance_ratio)

    σ = zero_safe_sqrt(variance)
    x₁ = x̂₁ * σ + x̄
    x₂ = x̂₂ * σ + x̄

    return x₁, x₂, x′²₁, x′²₂, zero_safe_sqrt(x′²₁), zero_safe_sqrt(x′²₂)
end

# The correlation of w with a scalar that the components carry, bounded smoothly so that the
# components are realizable by construction: r² < 1 - ℂᵍ(Sk), so that the within-component
# variance (1 - r² / (1 - ℂᵍ)) x′² is positive and no component has zero width, and
# |r| σ / |ŵᵢ| < x̄ for both components, so that the component means x̄ + (deviation) of a non-negative scalar
# (total water) stay positive. With r₁ = √(1 - ℂᵍ) and rᵢ = x̄ |ŵᵢ| / σ,
#
#     r = r₀ / (1 + (r₀ / r₁)⁴ + (r₀ σ / x̄)⁴ (ŵ₁⁻⁴ + ŵ₂⁻⁴))^(1/4),
#
# smooth in r₀, odd, and below every bound; r ≈ r₀ (1 - (r₀/r₁)⁴/4 - ...) where r₀ is well inside
# them. The flux the components carry is reduced by the same factor. A mean that is not positive leaves no room for
# a deviation: the correlation is zero there, the limit of the expression as x̄ → 0.
@inline function realizable_correlation(r₀, one_minus_share, x̄, σ, ŵ₁, ŵ₂)
    FT = typeof(r₀)
    t₁ = r₀ * r₀ / one_minus_share
    positive = x̄ > 0
    t₂ = r₀ * σ / ifelse(positive, x̄, one(FT))
    t₂² = t₂ * t₂
    ŵ₁⁻² = 1 / (ŵ₁ * ŵ₁)
    ŵ₂⁻² = 1 / (ŵ₂ * ŵ₂)
    r = r₀ / sqrt(sqrt(1 + t₁ * t₁ + t₂² * t₂² * (ŵ₁⁻² * ŵ₁⁻² + ŵ₂⁻² * ŵ₂⁻²)))
    return ifelse(positive, r, zero(FT))
end

# The equal share of the within-component variance (EqualWidthDoubleGaussianPDF)
@inline function scalar_component_parameters(clipping, flux, σw₊, σ₊, variance, x̄, ŵ₁, ŵ₂, Sk, a, skewed,
                                             pdf::EqualWidthDoubleGaussianPDF)
    r₀ = clip_between(clipping, flux / (σw₊ * σ₊), -1, 1, 1)
    one_minus_share = 1 - within_component_share(pdf, Sk)
    r = realizable_correlation(r₀, one_minus_share, x̄, σ₊, ŵ₁, ŵ₂)
    x̂₁ = -r / ŵ₂
    x̂₂ = -r / ŵ₁
    # r² < 1 - ℂᵍ by construction (`realizable_correlation`): the `max` removes round-off only
    x′²ᵢ = max(0, 1 - r * r / one_minus_share) * variance
    σ = zero_safe_sqrt(variance)
    x₁ = x̂₁ * σ + x̄
    x₂ = x̂₂ * σ + x̄
    σᵢ = zero_safe_sqrt(x′²ᵢ)
    return x₁, x₂, x′²ᵢ, x′²ᵢ, σᵢ, σᵢ
end

# shoc_assumed_pdf_compute_s for a component with total water `qᵗ`, variances `θˡⁱ′²`, `qᵗ′²`, spreads `σθ`, `σq` and
# correlation `r` of θˡⁱ and qᵗ, with `qᵛ⁺`, `β`, `cᵖᵐℒ⁻¹ = cᵖᵐ / ℒ` and `∂Tˡ∂θ = ∂Tˡ/∂θˡⁱ` of `component_saturation`: the saturation
# excess `s`, its spread `σₛ`, the condensate `qᶜ` and the saturated fraction `aᶜ` of the component
@inline function component_condensate(qᵗ, qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ, θˡⁱ′², qᵗ′², σθ, σq, r)
    FT = typeof(qᵗ)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qᵗ - qᵛ⁺ * ((1 + β * qᵗ) / (1 + β * qᵛ⁺))
    denominator = 1 + β * qᵛ⁺
    c_θ = ((1 + β * qᵗ) / (denominator * denominator)) * cᵖᵐℒ⁻¹ * β * qᵛ⁺ * ∂Tˡ∂θ
    c_q = 1 / (1 + β * qᵛ⁺)
    # A quadratic form with |r| ≤ 1: not negative but for round-off, which `zero_safe_sqrt` takes to zero
    σₛ = zero_safe_sqrt((c_θ * c_θ) * θˡⁱ′² + (c_q * c_q) * qᵗ′² - 2 * c_θ * σθ * c_q * σq * r)

    # A division by the spread is defined where the spread is: below, the limit of the Gaussian of vanishing width
    resolved = σₛ > sqrt(floatmin(FT)) * 100
    σₛ₊ = ifelse(resolved, σₛ, one(FT))

    # A PDF of finite width
    wide_fraction = convert(FT, 0.5) * (1 + erf(s / (sqrt2 * σₛ₊)))
    ŝ = s / σₛ₊
    wide_condensate = ifelse(wide_fraction != 0, s * wide_fraction + (σₛ₊ / sqrt2pi) * exp(convert(FT, -0.5) * (ŝ * ŝ)), zero(FT))

    # A PDF of no width, the limit of the one above as its width vanishes: all or nothing
    narrow_fraction = ifelse(s > 0, one(FT), zero(FT))
    narrow_condensate = ifelse(s > 0, s, zero(FT))

    aᶜ = ifelse(resolved, wide_fraction, narrow_fraction)
    qᶜ = ifelse(resolved, wide_condensate, narrow_condensate)

    # The mean of the positive part of a Gaussian is positive. Far below saturation its two terms cancel to round-off
    # (or underflow), which may leave a condensate that is not positive: there is none, and no saturated fraction
    empty = qᶜ <= 0
    aᶜ = ifelse(empty, zero(FT), aᶜ)
    qᶜ = ifelse(empty, zero(FT), qᶜ)

    return s, σₛ, qᶜ, aᶜ
end

# The liquid of a component is no more than its total water: with x = qᶜ / qᵗ,
#
#     qˡ = qᶜ / (1 + x⁴)^(1/4),
#
# below both, which changes a condensate well below the total water by the fraction x⁴ / 4. None where the component
# has no condensate or no water (the limit of the expression there).
@inline function bounded_condensate(qᶜ, qᵗ)
    FT = typeof(qᶜ)
    wet = (qᶜ > 0) & (qᵗ > 0)
    x = ifelse(wet, qᶜ / ifelse(wet, qᵗ, one(FT)), zero(FT))
    x² = x * x
    return ifelse(wet, qᶜ / sqrt(sqrt(1 + x² * x²)), zero(FT))
end

# The saturated fraction, the liquid and the variance of the liquid of a mixture of two components of weights `a` and
# `1 - a`. The fraction is a weighted mean of fractions and the liquid of liquids, each within its bounds; the
# variance, E[qˡ²] - E[qˡ]² with E[s qˡ] = sᵢ qˡᵢ + aᶜᵢ σₛᵢ² in a Gaussian component, is a difference that is not
# negative. The `min` and the `max` remove the round-off of the weighted mean and of the difference only.
@inline function mixture_cloud(a, aᶜ₁, aᶜ₂, qˡ₁, qˡ₂, s₁, s₂, σₛ₁, σₛ₂)
    aᶜ = min(1, a * aᶜ₁ + (1 - a) * aᶜ₂)
    qˡ = a * qˡ₁ + (1 - a) * qˡ₂
    qˡ′² = max(0, a * (s₁ * qˡ₁ + aᶜ₁ * (σₛ₁ * σₛ₁)) + (1 - a) * (s₂ * qˡ₂ + aᶜ₂ * (σₛ₂ * σₛ₂)) - qˡ * qˡ)
    return aᶜ, qˡ, qˡ′²
end

# The two components of the double Gaussian at a level, which `assumed_pdf` and `component_vapor_excess` share: the
# interface moments (w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′² of the interfaces below, `moments_below`, and above,
# `moments_above`) interpolated to the level, the weight `a` of the updraft component, the normalized vertical
# velocities `ŵ₁ > 0 > ŵ₂` of the components (before `shoc_assumed_pdf_tilde_to_real`), the means, variances and spreads
# of θˡⁱ (`potential_temperature`) and qᵗ (`total_water`) of each component, and their within-component correlation `r`.
@inline function double_gaussian_components(pdf, zᶠ⁺, zᶠ⁻, zᶜ, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    clipping = pdf.clipping
    FT = typeof(θˡⁱ)
    potential_temperature_tolerance = pdf.potential_temperature_tolerance
    total_water_tolerance = pdf.total_water_tolerance
    calm_variance = pdf.vertical_velocity_tolerance * pdf.vertical_velocity_tolerance

    interpolate(n) = linearly_interpolate(zᶠ⁺, zᶠ⁻, moments_above[n], moments_below[n], zᶜ)
    w′³ = interpolate(1)
    θˡⁱ′² = interpolate(2)
    w′θˡⁱ′ = interpolate(3)
    θˡⁱ′qᵗ′ = interpolate(4)
    w′qᵗ′ = interpolate(5)
    qᵗ′² = interpolate(6)

    σw = zero_safe_sqrt(w′²)
    # The vertical velocity variance and the standard deviations that form the skewness and the correlations: no
    # smaller than their tolerances
    w′²₊ = clipped_maximum(clipping, calm_variance, w′², calm_variance)
    σw₊ = sqrt(w′²₊)
    σθ₊ = clipped_maximum(clipping, potential_temperature_tolerance, zero_safe_sqrt(θˡⁱ′²), potential_temperature_tolerance)
    σq₊ = clipped_maximum(clipping, total_water_tolerance, zero_safe_sqrt(qᵗ′²), total_water_tolerance)

    # shoc_assumed_pdf_vv_parameters
    Sk = w′³ / (w′²₊ * σw₊)
    share = within_component_share(pdf, Sk)
    one_minus_share = 1 - share
    a = clip_between(clipping, convert(FT, 0.5) * (1 - Sk * sqrt(1 / (4 * (one_minus_share * one_minus_share * one_minus_share) + Sk * Sk))),
                     pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction, 1)
    sqrt_one_minus_share = sqrt(one_minus_share)
    ŵ₁ = sqrt((1 - a) / a) * sqrt_one_minus_share
    ŵ₂ = -1 * sqrt(a / (1 - a)) * sqrt_one_minus_share

    # shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters
    potential_temperature = scalar_component_parameters(clipping, w′θˡⁱ′, σw₊, σθ₊, θˡⁱ′², θˡⁱ, ŵ₁, ŵ₂, Sk, a, false, pdf)
    total_water = scalar_component_parameters(clipping, w′qᵗ′, σw₊, σq₊, qᵗ′², qᵗ, ŵ₁, ŵ₂, Sk, a, true, pdf)

    # shoc_assumed_pdf_inplume_correlations
    θˡⁱ₁, θˡⁱ₂, _, _, σθ₁, σθ₂ = potential_temperature
    qᵗ₁, qᵗ₂, _, _, σq₁, σq₂ = total_water
    spread = a * σq₁ * σθ₁ + (1 - a) * σq₂ * σθ₂
    # Components without width have no correlation within them, and none enters their saturation excess
    uncorrelated = spread == 0
    r = clip_between(clipping, (θˡⁱ′qᵗ′ - a * (qᵗ₁ - qᵗ) * (θˡⁱ₁ - θˡⁱ)
                                - (1 - a) * (qᵗ₂ - qᵗ) * (θˡⁱ₂ - θˡⁱ)) /
                               ifelse(uncorrelated, one(FT), spread), -1, 1, 1)
    r = ifelse(uncorrelated, zero(FT), r)

    return (; a, ŵ₁, ŵ₂, σw, w′θˡⁱ′, w′qᵗ′, potential_temperature, total_water, r)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The assumed PDF of level `k` (Analytic Double Gaussian 1 of Larson et al. 2002): return the
subgrid cloud fraction, liquid water, liquid water variance, liquid water flux and buoyancy flux.
The components condense with the `thermodynamics` of the model (`distribution_thermodynamics`): its thermodynamic
constants and the phase equilibrium of the condensate of its microphysics.
"""
@inline function assumed_pdf(pdf::ComponentDoubleGaussianPDF, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′²,
                             moments_below, moments_above)
    c = double_gaussian_components(pdf, zᶠ⁺, zᶠ⁻, zᶜ, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    a, w′θˡⁱ′, w′qᵗ′, r = c.a, c.w′θˡⁱ′, c.w′qᵗ′, c.r
    θˡⁱ₁, θˡⁱ₂, θˡⁱ′²₁, θˡⁱ′²₂, σθ₁, σθ₂ = c.potential_temperature
    qᵗ₁, qᵗ₂, qᵗ′²₁, qᵗ′²₂, σq₁, σq₂ = c.total_water

    # shoc_assumed_pdf_tilde_to_real
    w₁ = c.ŵ₁ * c.σw + w
    w₂ = c.ŵ₂ * c.σw + w

    # shoc_assumed_pdf_compute_temperature
    Tˡ₁ = liquid_water_temperature(thermodynamics, θˡⁱ₁, qᵗ₁, p)
    Tˡ₂ = liquid_water_temperature(thermodynamics, θˡⁱ₂, qᵗ₂, p)

    # shoc_assumed_pdf_compute_qs
    qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁ = component_saturation(thermodynamics, Tˡ₁, qᵗ₁, p)
    qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂ = component_saturation(thermodynamics, Tˡ₂, qᵗ₂, p)

    # shoc_assumed_pdf_compute_s
    s₁, σₛ₁, qᶜ₁, aᶜ₁ = component_condensate(qᵗ₁, qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r)
    s₂, σₛ₂, qᶜ₂, aᶜ₂ = component_condensate(qᵗ₂, qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r)

    qˡ₁ = bounded_condensate(qᶜ₁, qᵗ₁)
    qˡ₂ = bounded_condensate(qᶜ₂, qᵗ₂)

    aᶜ, qˡ, qˡ′² = mixture_cloud(a, aᶜ₁, aᶜ₂, qˡ₁, qˡ₂, s₁, s₂, σₛ₁, σₛ₂)

    w′qˡ′ = a * ((w₁ - w) * qˡ₁) + (1 - a) * ((w₂ - w) * qˡ₂)

    Aᵗ, Aᶜ = buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
    w′θᵛ′ = w′θˡⁱ′ + Aᵗ * w′qᵗ′ + Aᶜ * w′qˡ′

    return aᶜ, qˡ, qˡ′², w′qˡ′, w′θᵛ′
end

#####
##### The components of the double Gaussian in the vapor excess over liquid, for the microphysics
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The double Gaussian of `pdf` at a level (the moments are those [`assumed_pdf`](@ref) reads) as the
distribution of the vapor excess over liquid ``δqᵛ⁺ˡ = qᵗ - qᵛ⁺ˡ`` of each component's air without
condensate, as the microphysics defines it: the weight `a` of the updraft component (1) and, per
component, the mean ``μᵢ = qᵗᵢ - qᵛ⁺ˡ(Tᵢ, ρᵢ)`` at its liquid-water temperature ``Tᵢ`` and the
density ``ρᵢ = p / (Rᵐ Tᵢ)``, and the within-component spread
``σᵢ² = qᵗ′²ᵢ + (∂qᵛ⁺ˡ/∂θ)² θ′²ᵢ - 2 (∂qᵛ⁺ˡ/∂θ) r σθᵢ σqᵢ``, with the model's saturation over liquid
(the formula the microphysics uses); and the same moments of the vapor excess over ice, ``μⁱᵢ = qᵗᵢ - qᵛ⁺ⁱ(Tᵢ, ρᵢ)``
and ``σⁱᵢ`` with ``∂qᵛ⁺ⁱ/∂θ``, for the ice outside the distributed water. Returns
`(a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂, r₁, r₂)`, with ``rᵢ`` the within-component correlation of the two excesses.
"""
@inline function component_vapor_excess(pdf::ComponentDoubleGaussianPDF, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′²,
                                        moments_below, moments_above)
    c = double_gaussian_components(pdf, zᶠ⁺, zᶠ⁻, zᶜ, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    a, r = c.a, c.r
    θˡⁱ₁, θˡⁱ₂, θˡⁱ′²₁, θˡⁱ′²₂, σθ₁, σθ₂ = c.potential_temperature
    qᵗ₁, qᵗ₂, qᵗ′²₁, qᵗ′²₂, σq₁, σq₂ = c.total_water

    liquid = (PlanarLiquidSurface(), WarmPhaseEquilibrium())
    ice = (PlanarIceSurface(), IceSurfaceEquilibrium())
    μ₁, σ₁, ∂₁ = component_vapor_excess_moments(thermodynamics, p, θˡⁱ₁, qᵗ₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r, liquid)
    μ₂, σ₂, ∂₂ = component_vapor_excess_moments(thermodynamics, p, θˡⁱ₂, qᵗ₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r, liquid)
    μⁱ₁, σⁱ₁, ∂ⁱ₁ = component_vapor_excess_moments(thermodynamics, p, θˡⁱ₁, qᵗ₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r, ice)
    μⁱ₂, σⁱ₂, ∂ⁱ₂ = component_vapor_excess_moments(thermodynamics, p, θˡⁱ₂, qᵗ₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r, ice)
    r₁ = component_excess_correlation(qᵗ′²₁, θˡⁱ′²₁, σθ₁, σq₁, r, ∂₁, ∂ⁱ₁, σ₁, σⁱ₁)
    r₂ = component_excess_correlation(qᵗ′²₂, θˡⁱ′²₂, σθ₂, σq₂, r, ∂₂, ∂ⁱ₂, σ₂, σⁱ₂)
    return a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂, r₁, r₂
end

# The within-component correlation of the excesses over liquid and over ice, sˡ′ = qᵗ′ - aˡ θ′ and sⁱ′ = qᵗ′ - aⁱ θ′
# (aˡ, aⁱ = ∂qᵛ⁺/∂θ over each surface): cov = qᵗ′² - (aˡ + aⁱ) r σθ σq + aˡ aⁱ θ′². One where either has no spread
# (a point mass; the correlation does not enter).
@inline function component_excess_correlation(qᵗ′², θˡⁱ′², σθ, σq, r, aˡ, aⁱ, σ, σⁱ)
    covariance = qᵗ′² - (aˡ + aⁱ) * r * σθ * σq + aˡ * aⁱ * θˡⁱ′²
    spread = σ * σⁱ
    resolved = spread > 0
    return ifelse(resolved, covariance / ifelse(resolved, spread, one(spread)), one(spread))
end

# The moments of the vapor excess of a component over the surface `(surface, equilibrium)` (liquid, or ice), at the
# liquid-water temperature of the component, at which `assumed_pdf` condenses it, with the model's saturation over
# that surface (the formula the microphysics uses)
@inline function component_vapor_excess_moments(thermodynamics, p, θˡⁱ, qᵗ, θˡⁱ′², qᵗ′², σθ, σq, r, (surface, equilibrium))
    constants = thermodynamics.constants
    ∂Tˡ∂θ = liquid_water_temperature_slope(thermodynamics, qᵗ, p)
    T = ∂Tˡ∂θ * θˡⁱ
    Rᵐ = mixture_gas_constant(component_moisture(thermodynamics, qᵗ), constants)
    qᵛ⁺ = saturation_specific_humidity(T, p / (Rᵐ * T), constants, surface)
    _, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    ∂qᵛ⁺∂θ = qᵛ⁺ * log_slope * ∂Tˡ∂θ
    # A quadratic form with |r| ≤ 1: not negative but for round-off, which `zero_safe_sqrt` takes to zero
    σ = zero_safe_sqrt(qᵗ′² + ∂qᵛ⁺∂θ * ∂qᵛ⁺∂θ * θˡⁱ′² - 2 * ∂qᵛ⁺∂θ * r * σθ * σq)
    return qᵗ - qᵛ⁺, σ, ∂qᵛ⁺∂θ
end

# A PDF without a component form here supplies no spread: the microphysics' grid-mean rate
@inline function component_vapor_excess(pdf, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    FT = typeof(θˡⁱ)
    return convert(FT, 0.5), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), one(FT), one(FT)
end
