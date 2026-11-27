#####
##### The assumed double-Gaussian PDF without switches
#####

"""
$(TYPEDEF)

The Analytic Double Gaussian 1 of [`DoubleGaussianPDF`](@ref) with the tolerances of E3SM
`shoc.F90` as regularizations added to the variances, so that the cloud fraction,
the liquid water and the buoyancy flux are smooth functions of the moments and of the
coefficients where a variance vanishes too:

| `shoc.F90` | `SmoothDoubleGaussianPDF` |
|:--------------------|:--------------------------|
| calm, `w′² ≤ vertical_velocity_tolerance²`: no skewness, equal components at the mean | skewness `w′³ / (w′² + vertical_velocity_tolerance²)^{3/2}`, which vanishes with `w′²`; the components collapse onto the mean as `√w′²` |
| `√θˡⁱ′²` and `√qᵗ′²` floored at their tolerance | `√(variance + tolerance²)` |
| uniform, variance ≤ tolerance²: components at the mean, no width | component separation scaled by `variance / √(variance + tolerance²)`; the widths vanish with the variance |
| within-component variance from a ratio that is `0 / 0` where the component means coincide, set to zero there | the same ratio with the common factor cancelled, which tends to one |
| within-component correlation zero where the components have no width | covariance over the product of the widths plus a tenth of the product of the tolerances |
| saturation deficit of no spread: all or nothing | spread `√(variance + minimum_saturation_spread²)`, the error function everywhere |
| no liquid where its mean is not positive | the mean of the positive part of a Gaussian, which is positive |

The bounds of the updraft fraction, the correlations and the variance ratio go through the `clipping`
of the PDF ([`SmoothClipping`](@ref)), as in `DoubleGaussianPDF`, and the skewness of total water rises with the
separation of its components as it does there. Where the variances are well above their tolerances the two agree
closely; where a variance is near its tolerance `DoubleGaussianPDF` is the PDF of `shoc.F90` down to the tolerance
(a smooth maximum), and this PDF differs from it by the fraction `tolerance² / variance`.

$(TYPEDFIELDS)
"""
struct SmoothDoubleGaussianPDF{FT, CP} <: AbstractSubgridDistributionComponent
    "Within-component fraction of the vertical velocity variance (literal `0.4`)"
    ℂᵍ₁ :: FT
    "Skewness of total water relative to that of vertical velocity (literal `1.2`)"
    ℂᵍ₂ :: FT
    "Component separation below which total water is unskewed (literal `0.2`)"
    ℂᵍ₃ :: FT
    "Component separation above which total water is fully skewed (literal `0.4`)"
    ℂᵍ₄ :: FT
    "Standard deviation of θˡⁱ that regularizes its vanishing, K"
    potential_temperature_tolerance :: FT
    "Standard deviation of qᵗ that regularizes its vanishing, kg kg⁻¹"
    total_water_tolerance :: FT
    "Standard deviation of vertical velocity that regularizes its vanishing, m s⁻¹"
    vertical_velocity_tolerance :: FT
    "Lowest weight of the updraft component (literal `0.01`)"
    minimum_updraft_fraction :: FT
    "Highest weight of the updraft component (literal `0.99`)"
    maximum_updraft_fraction :: FT
    "Highest variance of a scalar within a component over its variance (literal `100`)"
    maximum_component_variance_ratio :: FT
    "Smallest spread of the saturation deficit within a component, kg kg⁻¹"
    minimum_saturation_spread :: FT
    "Floor of the width of a component over the tolerance of its scalar, which keeps the derivative of the width finite"
    component_width_floor :: FT
    "Regularization of the within-component correlation of θˡⁱ and qᵗ: the fraction of `potential_temperature_tolerance × total_water_tolerance` added to the product of the widths"
    correlation_regularization :: FT
    "How the bounds are imposed: [`SmoothClipping`](@ref)"
    clipping :: CP
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `SmoothDoubleGaussianPDF` with the coefficients and tolerances of E3SM `shoc.F90`, its bounds
imposed by `clipping`. Every field is a keyword argument.
"""
SmoothDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; clipping = SmoothClipping(FT), kw...) =
    component_from_keywords(SmoothDoubleGaussianPDF, FT,
                            (ℂᵍ₁ = 0.4, ℂᵍ₂ = 1.2, ℂᵍ₃ = 0.2, ℂᵍ₄ = 0.4,
                             potential_temperature_tolerance = 1e-2, total_water_tolerance = 1e-4,
                             vertical_velocity_tolerance = 2e-2,
                             minimum_updraft_fraction = 0.01, maximum_updraft_fraction = 0.99,
                             maximum_component_variance_ratio = 100,
                             minimum_saturation_spread = 1e-6, component_width_floor = 1e-3, correlation_regularization = 0.1), kw,
                            TurbulenceClosures.convert_eltype(FT, clipping))

Adapt.@adapt_structure SmoothDoubleGaussianPDF

# The two components have the same width in w (the share ℂᵍ₁ of w′²), as in `DoubleGaussianPDF`
@inline representable_skewness(pdf::SmoothDoubleGaussianPDF) =
    representable_skewness(pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction, pdf.ℂᵍ₁)

# The component means of a scalar relative to its mean, in units of its standard deviation, and the
# within-component variances over the variance, from the correlation `r` of the scalar with w and the
# component velocities `ŵ₁ > 0 > ŵ₂` (in units of the standard deviation of w, so that
# `ŵ₁ ŵ₂ = -(1 - ℂᵍ₁)` and `a ŵ₁ + (1 - a) ŵ₂ = 0`). The ratios are those of
# `component_variance` with the factor `r` that numerator and denominator share cancelled.
@inline function smooth_scalar_components(pdf, clipping, r, ŵ₁, ŵ₂, Sk, a, skewed)
    FT = typeof(r)
    α = ŵ₁
    β = -ŵ₂
    s² = α * β                     # 1 - ℂᵍ₁

    x̂₁ = r / β
    x̂₂ = -r / α
    separation = x̂₂ - x̂₁

    # The skewness of the scalar, which rises from zero as the components separate (`skewness_share`), over the
    # separation: it vanishes with the separation, where the division is not carried out.
    skewness = ifelse(skewed, pdf.ℂᵍ₂ * Sk * skewness_share(pdf, abs(separation)), zero(FT))
    apart = separation != 0
    skewness_over_separation = ifelse(apart, skewness / ifelse(apart, separation, one(FT)), zero(FT))

    unexplained = 1 - r * r / s²                          # 1 - a x̂₁² - (1 - a) x̂₂²
    asymmetry = r * r * (a * α) * (β * β - α * α) / (s² * s² * s²)

    ratio₁ = s² * (3 / α * unexplained + asymmetry) / (3 * a * (α + β)) - skewness_over_separation / (3 * a)
    ratio₂ = s² * (3 / β * unexplained - asymmetry) / (3 * (1 - a) * (α + β)) + skewness_over_separation / (3 * (1 - a))

    maximum_ratio = pdf.maximum_component_variance_ratio
    return x̂₁, x̂₂, clip_between(clipping, ratio₁, 0, maximum_ratio, 1), clip_between(clipping, ratio₂, 0, maximum_ratio, 1)
end

# shoc_assumed_pdf_compute_s with a spread that does not vanish (the arguments of `component_condensate`)
@inline function smooth_component_condensate(pdf, qᵗ, qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ, θˡⁱ′², qᵗ′², σθ, σq, r)
    FT = typeof(qᵗ)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qᵗ - qᵛ⁺ * ((1 + β * qᵗ) / (1 + β * qᵛ⁺))
    denominator = 1 + β * qᵛ⁺
    c_θ = ((1 + β * qᵗ) / (denominator * denominator)) * cᵖᵐℒ⁻¹ * β * qᵛ⁺ * ∂Tˡ∂θ
    c_q = 1 / (1 + β * qᵛ⁺)
    # A quadratic form with |r| ≤ 1: not negative but for round-off, which the `max` removes
    variance = max(0, (c_θ * c_θ) * θˡⁱ′² + (c_q * c_q) * qᵗ′² - 2 * c_θ * σθ * c_q * σq * r)
    σₛ = sqrt(variance + pdf.minimum_saturation_spread * pdf.minimum_saturation_spread)

    ŝ = s / σₛ
    aᶜ = convert(FT, 0.5) * (1 + erf(ŝ / sqrt2))
    # The mean of the positive part of the saturation excess, which is not negative but for round-off
    qᶜ = max(0, s * aᶜ + (σₛ / sqrt2pi) * exp(convert(FT, -0.5) * (ŝ * ŝ)))

    return s, σₛ, qᶜ, aᶜ
end

@inline function assumed_pdf(pdf::SmoothDoubleGaussianPDF, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′²,
                             moments_below, moments_above)
    clipping = pdf.clipping
    FT = typeof(θˡⁱ)
    potential_temperature_tolerance = pdf.potential_temperature_tolerance
    total_water_tolerance = pdf.total_water_tolerance
    calm_variance = pdf.vertical_velocity_tolerance * pdf.vertical_velocity_tolerance

    # Interpolate the interface moments to the level
    interpolate(n) = linearly_interpolate(zᶠ⁺, zᶠ⁻, moments_above[n], moments_below[n], zᶜ)
    w′³ = interpolate(1)
    θˡⁱ′² = interpolate(2)
    w′θˡⁱ′ = interpolate(3)
    θˡⁱ′qᵗ′ = interpolate(4)
    w′qᵗ′ = interpolate(5)
    qᵗ′² = interpolate(6)

    # The standard deviations that form the correlations do not vanish with the variances
    σw = zero_safe_sqrt(w′²)
    σθ = sqrt(θˡⁱ′² + potential_temperature_tolerance * potential_temperature_tolerance)
    σq = sqrt(qᵗ′² + total_water_tolerance * total_water_tolerance)

    # Vertical velocity: skewness, updraft fraction, component velocities over the standard deviation
    w′²₊ = w′² + calm_variance
    σw₊ = sqrt(w′²₊)
    Sk = w′³ / (w′²₊ * σw₊)
    share = pdf.ℂᵍ₁
    one_minus_share = 1 - share
    a = clip_between(clipping, convert(FT, 0.5) * (1 - Sk * sqrt(1 / (4 * (one_minus_share * one_minus_share * one_minus_share) + Sk * Sk))),
                     pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction, 1)
    sqrt_one_minus_share = sqrt(one_minus_share)
    ŵ₁ = sqrt((1 - a) / a) * sqrt_one_minus_share
    ŵ₂ = -1 * sqrt(a / (1 - a)) * sqrt_one_minus_share

    # The scalars: correlation with w, component means and within-component variances
    rθ = clip_between(clipping, w′θˡⁱ′ / (σw₊ * σθ), -1, 1, 1)
    rq = clip_between(clipping, w′qᵗ′ / (σw₊ * σq), -1, 1, 1)

    θ̂₁, θ̂₂, potential_temperature_ratio₁, potential_temperature_ratio₂ = smooth_scalar_components(pdf, clipping, rθ, ŵ₁, ŵ₂, Sk, a, false)
    q̂₁, q̂₂, total_water_ratio₁, total_water_ratio₂ = smooth_scalar_components(pdf, clipping, rq, ŵ₁, ŵ₂, Sk, a, true)

    # The separation of the components is √variance where the variance is resolved, and vanishes
    # linearly with it below the tolerance
    potential_temperature_separation = θˡⁱ′² / σθ
    total_water_separation = qᵗ′² / σq
    θˡⁱ₁ = θ̂₁ * potential_temperature_separation + θˡⁱ
    θˡⁱ₂ = θ̂₂ * potential_temperature_separation + θˡⁱ
    qᵗ₁ = q̂₁ * total_water_separation + qᵗ
    qᵗ₂ = q̂₂ * total_water_separation + qᵗ

    θˡⁱ′²₁ = potential_temperature_ratio₁ * θˡⁱ′²
    θˡⁱ′²₂ = potential_temperature_ratio₂ * θˡⁱ′²
    qᵗ′²₁ = total_water_ratio₁ * qᵗ′²
    qᵗ′²₂ = total_water_ratio₂ * qᵗ′²

    # Widths with a floor far below the tolerances, so that their derivative stays finite
    potential_temperature_width_floor = pdf.component_width_floor * potential_temperature_tolerance
    total_water_width_floor = pdf.component_width_floor * total_water_tolerance
    σθ₁ = sqrt(θˡⁱ′²₁ + potential_temperature_width_floor * potential_temperature_width_floor)
    σθ₂ = sqrt(θˡⁱ′²₂ + potential_temperature_width_floor * potential_temperature_width_floor)
    σq₁ = sqrt(qᵗ′²₁ + total_water_width_floor * total_water_width_floor)
    σq₂ = sqrt(qᵗ′²₂ + total_water_width_floor * total_water_width_floor)

    # The component velocities
    w₁ = ŵ₁ * σw + w
    w₂ = ŵ₂ * σw + w

    # The correlation of θˡⁱ and qᵗ within the components
    spread = a * σq₁ * σθ₁ + (1 - a) * σq₂ * σθ₂
    r = clip_between(clipping, (θˡⁱ′qᵗ′ - a * (qᵗ₁ - qᵗ) * (θˡⁱ₁ - θˡⁱ)
                                - (1 - a) * (qᵗ₂ - qᵗ) * (θˡⁱ₂ - θˡⁱ)) /
                               (spread + pdf.correlation_regularization * potential_temperature_tolerance * total_water_tolerance), -1, 1, 1)

    # Liquid water temperature and saturation of the components
    Tˡ₁ = liquid_water_temperature(thermodynamics, θˡⁱ₁, qᵗ₁, p)
    Tˡ₂ = liquid_water_temperature(thermodynamics, θˡⁱ₂, qᵗ₂, p)

    qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁ = component_saturation(thermodynamics, Tˡ₁, qᵗ₁, p)
    qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂ = component_saturation(thermodynamics, Tˡ₂, qᵗ₂, p)

    s₁, σₛ₁, qᶜ₁, aᶜ₁ = smooth_component_condensate(pdf, qᵗ₁, qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁, θˡⁱ′²₁, qᵗ′²₁, σθ₁, σq₁, r)
    s₂, σₛ₂, qᶜ₂, aᶜ₂ = smooth_component_condensate(pdf, qᵗ₂, qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂, θˡⁱ′²₂, qᵗ′²₂, σθ₂, σq₂, r)

    qˡ₁ = bounded_condensate(qᶜ₁, qᵗ₁)
    qˡ₂ = bounded_condensate(qᶜ₂, qᵗ₂)

    aᶜ, qˡ, qˡ′² = mixture_cloud(a, aᶜ₁, aᶜ₂, qˡ₁, qˡ₂, s₁, s₂, σₛ₁, σₛ₂)

    w′qˡ′ = a * ((w₁ - w) * qˡ₁) + (1 - a) * ((w₂ - w) * qˡ₂)

    Aᵗ, Aᶜ = buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
    w′θᵛ′ = w′θˡⁱ′ + Aᵗ * w′qᵗ′ + Aᶜ * w′qˡ′

    return aᶜ, qˡ, qˡ′², w′qˡ′, w′θᵛ′
end
