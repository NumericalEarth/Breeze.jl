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
##### `clipping` policy of the PDF (`HardClipping` or `SmoothClipping`) imposes its bounds: of the
##### weight of the updraft component, of the correlations and of the within-component variances.
#####

"""
$(TYPEDEF)

The Analytic Double Gaussian 1 of Larson et al. (2002) as E3SM `shoc.F90` computes it
(`shoc_assumed_pdf`): two Gaussian components in vertical velocity, θˡⁱ and qᵗ whose weights, means
and widths follow from the first three moments of vertical velocity and the second moments of
the scalars, each condensing where it is saturated. The Fortran switches are kept: where the
vertical velocity variance is below `vertical_velocity_tolerance²` or a scalar variance below its tolerance
squared, the components collapse onto the mean.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵍ₁` | `w2_1`, `w2_2` (literal `0.4`) | 0.4 | within-component fraction of the vertical velocity variance |
| `ℂᵍ₂` | literal `1.2`    | 1.2    | skewness of total water relative to that of vertical velocity |
| `ℂᵍ₃` | literal `0.2`    | 0.2    | component separation below which total water is unskewed |
| `ℂᵍ₄` | literal `0.4`    | 0.4    | component separation above which total water takes the full skewness `ℂᵍ₂ Sk` |

Between `ℂᵍ₃` and `ℂᵍ₄` the skewness of total water ramps linearly; the Fortran writes the width
of the ramp as the literal `0.2`, which is taken here as `ℂᵍ₄ - ℂᵍ₃` so that the ramp stays
continuous when the two are changed. `ℂᵍ₃ < ℂᵍ₄` is required. The fields after the coefficients
are tolerances and bounds, not empirical coefficients, and the `clipping` policy that imposes the bounds
(`HardClipping()`, the `max`, `min` and `if` of `shoc.F90`, by default).

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
    "Standard deviation of θˡⁱ below which it is uniform, K (`thl_tol`)"
    potential_temperature_tolerance :: FT
    "Standard deviation of qᵗ below which it is uniform, kg kg⁻¹ (`rt_tol`)"
    total_water_tolerance :: FT
    "Standard deviation of vertical velocity below which the air is calm, m s⁻¹ (`w_tol`)"
    vertical_velocity_tolerance :: FT
    "Lowest liquid water temperature of a component, K"
    minimum_liquid_water_temperature :: FT
    "Lowest weight of the updraft component (literal `0.01`)"
    minimum_updraft_fraction :: FT
    "Highest weight of the updraft component (literal `0.99`)"
    maximum_updraft_fraction :: FT
    "Highest variance of a scalar within a component over its variance (literal `100`)"
    maximum_component_variance_ratio :: FT
    "How the bounds are imposed: [`HardClipping`](@ref) or [`SmoothClipping`](@ref)"
    clipping :: CP
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `DoubleGaussianPDF` with the values of E3SM `shoc.F90`, its bounds imposed by `clipping` (as in
`shoc.F90` by default). Every field is a keyword argument.
"""
DoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; clipping = HardClipping(), kw...) =
    component_from_keywords(DoubleGaussianPDF, FT,
                            (ℂᵍ₁ = 0.4, ℂᵍ₂ = 1.2, ℂᵍ₃ = 0.2, ℂᵍ₄ = 0.4,
                             potential_temperature_tolerance = 1e-2, total_water_tolerance = 1e-4,
                             vertical_velocity_tolerance = 2e-2, minimum_liquid_water_temperature = 100,
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
variance in the narrow component, clipped to zero: a component of no width, whose cloud is all
or nothing. The equal share needs `r² ≤ 1 - ℂᵍ`; beyond it the within-component variance is zero
(the flux-correlation bound `ℂʳ` of [`DoubleGaussian`](@ref) keeps `r` below it). The
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
    "Standard deviation of θˡⁱ below which it is uniform, K"
    potential_temperature_tolerance :: FT
    "Standard deviation of qᵗ below which it is uniform, kg kg⁻¹"
    total_water_tolerance :: FT
    "Standard deviation of vertical velocity below which the air is calm, m s⁻¹"
    vertical_velocity_tolerance :: FT
    "Lowest liquid water temperature of a component, K"
    minimum_liquid_water_temperature :: FT
    "Lowest weight of the updraft component"
    minimum_updraft_fraction :: FT
    "Highest weight of the updraft component"
    maximum_updraft_fraction :: FT
    "How the bounds are imposed: [`HardClipping`](@ref) or [`SmoothClipping`](@ref)"
    clipping :: CP
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return an `EqualWidthDoubleGaussianPDF` with the coefficients and tolerances of [`DoubleGaussianPDF`](@ref), its
bounds imposed smoothly by default. Every field is a keyword argument.
"""
function EqualWidthDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; ℂᵍ₀ = 0.4, clipping = SmoothClipping(FT), kw...)
    0 < ℂᵍ₀ < 1 || throw(ArgumentError("ℂᵍ₀ must be in (0, 1), got $ℂᵍ₀"))
    return component_from_keywords(EqualWidthDoubleGaussianPDF, FT,
                                   (ℂᵍ₀ = ℂᵍ₀, ℂᵍˢ = 0, ℂᵍˢˢ = 0,
                                    potential_temperature_tolerance = 1e-2, total_water_tolerance = 1e-4,
                                    vertical_velocity_tolerance = 2e-2, minimum_liquid_water_temperature = 100,
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
    # Where the component means coincide the Fortran forms 0/0 and MAX(0, NaN) drops it; that is zero.
    degenerate = denominator == 0
    ratio = ifelse(degenerate, zero(numerator), numerator / ifelse(degenerate, one(denominator), denominator))
    return clip_between(clipping, ratio, 0, maximum_ratio, 1) * variance
end

# The means `x₁`, `x₂`, variances `x′²₁`, `x′²₂` and spreads of a scalar in the two components, from its `flux`, its
# standard deviation `σ` (bounded below by its `tolerance`), its `variance` and its mean `x̄`; `σw` is the standard
# deviation of vertical velocity and `ŵ₁`, `ŵ₂` the component velocities in units of it.
@inline function scalar_component_parameters(clipping, flux, σw, σ, variance, tolerance, x̄, ŵ₁, ŵ₂, Sk, a, skewed, pdf::DoubleGaussianPDF)
    FT = typeof(variance)
    r = clip_between(clipping, flux / (σw * σ), -1, 1, 1)

    uniform = (variance <= tolerance * tolerance) | (abs(ŵ₂ - ŵ₁) <= 0)

    # the normalized component means
    x̂₁ = (-1 * r) / ifelse(uniform, one(FT), ŵ₂)
    x̂₂ = (-1 * r) / ifelse(uniform, one(FT), ŵ₁)

    separation = abs(x̂₂ - x̂₁)
    ramped_skewness = ifelse(separation > pdf.ℂᵍ₄, pdf.ℂᵍ₂ * Sk,
                             ifelse(separation <= pdf.ℂᵍ₃, zero(FT),
                                    ((pdf.ℂᵍ₂ * Sk) / (pdf.ℂᵍ₄ - pdf.ℂᵍ₃)) * (separation - pdf.ℂᵍ₃)))
    skewness = ifelse(skewed, ramped_skewness, zero(FT))

    x′²₁ = component_variance(clipping, true, x̂₁, x̂₂, a, skewness, variance, pdf.maximum_component_variance_ratio)
    x′²₂ = component_variance(clipping, false, x̂₁, x̂₂, a, skewness, variance, pdf.maximum_component_variance_ratio)

    x₁ = x̂₁ * σ + x̄
    x₂ = x̂₂ * σ + x̄

    x₁ = ifelse(uniform, x̄, x₁)
    x₂ = ifelse(uniform, x̄, x₂)
    x′²₁ = ifelse(uniform, zero(FT), x′²₁)
    x′²₂ = ifelse(uniform, zero(FT), x′²₂)

    return x₁, x₂, x′²₁, x′²₂, zero_safe_sqrt(x′²₁), zero_safe_sqrt(x′²₂)
end

# The correlation of w with a scalar that the components carry, bounded smoothly so that the
# components are realizable by construction: r² < 1 - ℂᵍ(Sk), so that the within-component
# variance (1 - r² / (1 - ℂᵍ)) x′² is positive and no component has zero width, and
# |r| σ / min|ŵᵢ| < x̄, so that the component means x̄ + (deviation) of a non-negative scalar
# (total water) stay positive. With r₁ = √(1 - ℂᵍ) and r₂ = x̄ min|ŵᵢ| / σ,
#
#     r = r₀ / (1 + (r₀ / r₁)⁴ + (r₀ / r₂)⁴)^(1/4),
#
# smooth in r₀, odd, and below both bounds; r ≈ r₀ (1 - (r₀/r₁)⁴/4 - ...) where r₀ is well inside
# them. The flux the components carry is reduced by the same factor.
@inline function realizable_correlation(r₀, one_minus_share, x̄, σ, ŵᵐⁱⁿ)
    FT = typeof(r₀)
    t₁ = r₀ * r₀ / one_minus_share
    t₂ = r₀ * σ / max(max(x̄, zero(FT)) * ŵᵐⁱⁿ, floatmin(FT))
    t₂² = t₂ * t₂
    return r₀ / sqrt(sqrt(1 + t₁ * t₁ + t₂² * t₂²))
end

# The equal share of the within-component variance (EqualWidthDoubleGaussianPDF)
@inline function scalar_component_parameters(clipping, flux, σw, σ, variance, tolerance, x̄, ŵ₁, ŵ₂, Sk, a, skewed,
                                             pdf::EqualWidthDoubleGaussianPDF)
    FT = typeof(variance)
    r₀ = clip_between(clipping, flux / (σw * σ), -1, 1, 1)
    uniform = (variance <= tolerance * tolerance) | (abs(ŵ₂ - ŵ₁) <= 0)
    ŵ₁ˢᵃᶠᵉ = ifelse(uniform, one(FT), ŵ₁)
    ŵ₂ˢᵃᶠᵉ = ifelse(uniform, -one(FT), ŵ₂)
    r = realizable_correlation(r₀, 1 - within_component_share(pdf, Sk), x̄, σ, min(abs(ŵ₁ˢᵃᶠᵉ), abs(ŵ₂ˢᵃᶠᵉ)))
    x̂₁ = (-1 * r) / ŵ₂ˢᵃᶠᵉ
    x̂₂ = (-1 * r) / ŵ₁ˢᵃᶠᵉ
    x′²ᵢ = max(0, 1 - r * r / (1 - within_component_share(pdf, Sk))) * variance
    x₁ = ifelse(uniform, x̄, x̂₁ * σ + x̄)
    x₂ = ifelse(uniform, x̄, x̂₂ * σ + x̄)
    x′²ᵢ = ifelse(uniform, zero(FT), x′²ᵢ)
    σᵢ = zero_safe_sqrt(x′²ᵢ)
    return x₁, x₂, x′²ᵢ, x′²ᵢ, σᵢ, σᵢ
end

# shoc_assumed_pdf_compute_s for a component with total water `qᵗ`, variances `θˡⁱ′²`, `qᵗ′²`, spreads `σθ`, `σq` and
# correlation `r` of θˡⁱ and qᵗ, with `qᵛ⁺`, `β`, `ĉᵖℒ⁻¹ = ĉᵖ / ℒ` and `Π` of `component_saturation`: the saturation
# excess `s`, its spread `σₛ`, the condensate `qᶜ` and the saturated fraction `aᶜ` of the component
@inline function component_condensate(qᵗ, qᵛ⁺, β, ĉᵖℒ⁻¹, Π, θˡⁱ′², qᵗ′², σθ, σq, r)
    FT = typeof(qᵗ)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qᵗ - qᵛ⁺ * ((1 + β * qᵗ) / (1 + β * qᵛ⁺))
    denominator = 1 + β * qᵛ⁺
    c_θ = ((1 + β * qᵗ) / (denominator * denominator)) * ĉᵖℒ⁻¹ * β * qᵛ⁺ * Π
    c_q = 1 / (1 + β * qᵛ⁺)
    σₛ² = max(0, (c_θ * c_θ) * θˡⁱ′² + (c_q * c_q) * qᵗ′² - 2 * c_θ * σθ * c_q * σq * r)
    σₛ = zero_safe_sqrt(σₛ²)

    resolved = σₛ > sqrt(floatmin(FT)) * 100
    σₛˢᵃᶠᵉ = ifelse(resolved, σₛ, one(FT))

    # A PDF of finite width
    wide_fraction = convert(FT, 0.5) * (1 + erf(s / (sqrt2 * σₛˢᵃᶠᵉ)))
    ŝ = s / σₛˢᵃᶠᵉ
    wide_condensate = ifelse(wide_fraction != 0, s * wide_fraction + (σₛˢᵃᶠᵉ / sqrt2pi) * exp(convert(FT, -0.5) * (ŝ * ŝ)), zero(FT))

    # A PDF of no width: all or nothing
    narrow_fraction = ifelse(s > 0, one(FT), zero(FT))
    narrow_condensate = ifelse(s > 0, s, zero(FT))

    aᶜ = ifelse(resolved, wide_fraction, narrow_fraction)
    qᶜ = ifelse(resolved, wide_condensate, narrow_condensate)

    # Prevent empty clouds, or the rare occurrence of negative cloud liquid
    empty = qᶜ <= 0
    aᶜ = ifelse(empty, zero(FT), aᶜ)
    qᶜ = ifelse(empty, zero(FT), qᶜ)

    return s, σₛ, qᶜ, aᶜ
end

# The two components of the double Gaussian at a level, which `assumed_pdf` and `component_vapor_excess` share: the
# interface moments (w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′² of the interfaces below, `moments_below`, and above,
# `moments_above`) interpolated to the level, the weight `a` of the updraft component, the normalized vertical
# velocities `ŵ₁`, `ŵ₂` of the components (before `shoc_assumed_pdf_tilde_to_real`), the means, variances and spreads
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
    σθ = clipped_maximum(clipping, potential_temperature_tolerance, zero_safe_sqrt(θˡⁱ′²), potential_temperature_tolerance)
    σq = clipped_maximum(clipping, total_water_tolerance, zero_safe_sqrt(qᵗ′²), total_water_tolerance)

    # shoc_assumed_pdf_vv_parameters
    calm = w′² <= calm_variance
    w′²ˢᵃᶠᵉ = ifelse(calm, one(FT), w′²)
    Sk = w′³ / sqrt(w′²ˢᵃᶠᵉ * w′²ˢᵃᶠᵉ * w′²ˢᵃᶠᵉ)
    share = within_component_share(pdf, Sk)
    one_minus_share = 1 - share
    a = clip_between(clipping, convert(FT, 0.5) * (1 - Sk * sqrt(1 / (4 * (one_minus_share * one_minus_share * one_minus_share) + Sk * Sk))),
                     pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction, 1)
    sqrt_one_minus_share = sqrt(1 - share)
    ŵ₁ = sqrt((1 - a) / a) * sqrt_one_minus_share
    ŵ₂ = -1 * sqrt(a / (1 - a)) * sqrt_one_minus_share

    Sk = ifelse(calm, zero(FT), Sk)
    a = ifelse(calm, convert(FT, 0.5), a)
    ŵ₁ = ifelse(calm, w, ŵ₁)
    ŵ₂ = ifelse(calm, w, ŵ₂)

    # shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters
    potential_temperature = scalar_component_parameters(clipping, w′θˡⁱ′, σw, σθ, θˡⁱ′², potential_temperature_tolerance, θˡⁱ, ŵ₁, ŵ₂, Sk, a, false, pdf)
    total_water = scalar_component_parameters(clipping, w′qᵗ′, σw, σq, qᵗ′², total_water_tolerance, qᵗ, ŵ₁, ŵ₂, Sk, a, true, pdf)

    # shoc_assumed_pdf_inplume_correlations
    θˡⁱ₁, θˡⁱ₂, _, _, σθ₁, σθ₂ = potential_temperature
    qᵗ₁, qᵗ₂, _, _, σq₁, σq₂ = total_water
    spread = a * σq₁ * σθ₁ + (1 - a) * σq₂ * σθ₂
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
    clipping = pdf.clipping
    total_water_tolerance = pdf.total_water_tolerance
    c = double_gaussian_components(pdf, zᶠ⁺, zᶠ⁻, zᶜ, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    a, w′θˡⁱ′, w′qᵗ′, r = c.a, c.w′θˡⁱ′, c.w′qᵗ′, c.r
    θˡⁱ₁, θˡⁱ₂, θˡⁱ′²₁, θˡⁱ′²₂, σθ₁, σθ₂ = c.potential_temperature
    qᵗ₁, qᵗ₂, qᵗ′²₁, qᵗ′²₂, σq₁, σq₂ = c.total_water

    # shoc_assumed_pdf_tilde_to_real
    w₁ = c.ŵ₁ * c.σw + w
    w₂ = c.ŵ₂ * c.σw + w

    # shoc_assumed_pdf_compute_temperature
    Tˡ₁ = component_liquid_water_temperature(pdf, thermodynamics, θˡⁱ₁, qᵗ₁, p)
    Tˡ₂ = component_liquid_water_temperature(pdf, thermodynamics, θˡⁱ₂, qᵗ₂, p)

    # shoc_assumed_pdf_compute_qs
    qᵛ⁺₁, β₁, ĉᵖℒ⁻¹₁, Π₁ = component_saturation(thermodynamics, Tˡ₁, qᵗ₁, p)
    qᵛ⁺₂, β₂, ĉᵖℒ⁻¹₂, Π₂ = component_saturation(thermodynamics, Tˡ₂, qᵗ₂, p)

    # shoc_assumed_pdf_compute_s
    s₁, σₛ₁, qᶜ₁, aᶜ₁ = component_condensate(qᵗ₁, qᵛ⁺₁, β₁, ĉᵖℒ⁻¹₁, Π₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r)
    s₂, σₛ₂, qᶜ₂, aᶜ₂ = component_condensate(qᵗ₂, qᵛ⁺₂, β₂, ĉᵖℒ⁻¹₂, Π₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r)

    qˡ₁ = clipped_minimum(clipping, qᶜ₁, qᵗ₁, total_water_tolerance)
    qˡ₂ = clipped_minimum(clipping, qᶜ₂, qᵗ₂, total_water_tolerance)

    aᶜ = min(1, a * aᶜ₁ + (1 - a) * aᶜ₂)
    qˡ = max(0, a * qˡ₁ + (1 - a) * qˡ₂)

    qˡ′² = a * (s₁ * qˡ₁ + aᶜ₁ * (σₛ₁ * σₛ₁)) + (1 - a) * (s₂ * qˡ₂ + aᶜ₂ * (σₛ₂ * σₛ₂)) - (qˡ * qˡ)
    qˡ′² = max(0, qˡ′²)

    w′qˡ′ = a * ((w₁ - w) * qˡ₁) + (1 - a) * ((w₂ - w) * qˡ₂)

    A_q, A_c = buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
    w′θᵛ′ = w′θˡⁱ′ + A_q * w′qᵗ′ + A_c * w′qˡ′

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
component, the mean ``μᵢ = qᵗᵢ - qᵛ⁺ˡ(Tᵢ, ρᵢ)`` at its liquid-water temperature ``Tᵢ = Π θᵢ`` and the
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
    μ₁, σ₁, ∂₁ = component_vapor_excess_moments(pdf, thermodynamics, p, θˡⁱ₁, qᵗ₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r, liquid)
    μ₂, σ₂, ∂₂ = component_vapor_excess_moments(pdf, thermodynamics, p, θˡⁱ₂, qᵗ₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r, liquid)
    μⁱ₁, σⁱ₁, ∂ⁱ₁ = component_vapor_excess_moments(pdf, thermodynamics, p, θˡⁱ₁, qᵗ₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r, ice)
    μⁱ₂, σⁱ₂, ∂ⁱ₂ = component_vapor_excess_moments(pdf, thermodynamics, p, θˡⁱ₂, qᵗ₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r, ice)
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

# The liquid-water temperature of a component of the double Gaussian, the one definition every function of the PDF uses:
# that of the model's thermodynamics, raised to `minimum_liquid_water_temperature` where it is below
# (`shoc_assumed_pdf_compute_temperature`)
@inline function component_liquid_water_temperature(pdf, thermodynamics, θˡⁱ, qᵗ, p)
    Tˡ = liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p)
    return ifelse(Tˡ <= pdf.minimum_liquid_water_temperature, pdf.minimum_liquid_water_temperature, Tˡ)
end

# The moments of the vapor excess of a component over the surface `(surface, equilibrium)` (liquid, or ice), at the
# temperature of the component that `assumed_pdf` condenses at, with the model's saturation over that surface (the
# formula the microphysics uses)
@inline function component_vapor_excess_moments(pdf, thermodynamics, p, θˡⁱ, qᵗ, θˡⁱ′², qᵗ′², σθ, σq, r, (surface, equilibrium))
    constants = thermodynamics.constants
    Π = component_exner_function(thermodynamics, qᵗ, p)
    T = component_liquid_water_temperature(pdf, thermodynamics, θˡⁱ, qᵗ, p)
    Rᵐ = mixture_gas_constant(component_moisture(thermodynamics, qᵗ), constants)
    qᵛ⁺ = saturation_specific_humidity(T, p / (Rᵐ * T), constants, surface)
    _, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    ∂qᵛ⁺∂θ = qᵛ⁺ * log_slope * Π
    σ = zero_safe_sqrt(max(0, qᵗ′² + ∂qᵛ⁺∂θ * ∂qᵛ⁺∂θ * θˡⁱ′² - 2 * ∂qᵛ⁺∂θ * r * σθ * σq))
    return qᵗ - qᵛ⁺, σ, ∂qᵛ⁺∂θ
end

# A PDF without a component form here supplies no spread: the microphysics' grid-mean rate
@inline function component_vapor_excess(pdf, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    FT = typeof(θˡⁱ)
    return convert(FT, 0.5), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), one(FT), one(FT)
end
