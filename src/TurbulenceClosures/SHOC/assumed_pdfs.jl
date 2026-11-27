"""
$(TYPEDEF)

The Analytic Double Gaussian 1 of Larson et al. (2002), the assumed PDF of E3SM `shoc.F90`
(`shoc_assumed_pdf`): two Gaussian components in vertical velocity, θˡⁱ and qᵗ whose weights, means
and widths follow from the first three moments of vertical velocity and the second moments of
the scalars, each condensing where it is saturated.
"""
struct DoubleGaussianPDF{FT, CP} <: AbstractSubgridDistributionComponent
    ℂᵍ₁ :: FT
    ℂᵍ₂ :: FT
    ℂᵍ₃ :: FT
    ℂᵍ₄ :: FT
    potential_temperature_tolerance :: FT
    total_water_tolerance :: FT
    vertical_velocity_tolerance :: FT
    minimum_updraft_fraction :: FT
    maximum_updraft_fraction :: FT
    maximum_component_variance_ratio :: FT
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
                            convert_eltype(FT, clipping))

Adapt.@adapt_structure DoubleGaussianPDF
"""
$(TYPEDEF)

The double-Gaussian PDF of [`DoubleGaussianPDF`](@ref) (the updraft component of weight `a↑` and the
downdraft component of weight `a↓ = 1 - a↑`, from the first three moments of vertical velocity)
with the within-component variances of θˡⁱ and qᵗ shared equally between the components, as for
vertical velocity:
"""
struct EqualWidthDoubleGaussianPDF{FT, CP} <: AbstractSubgridDistributionComponent
    ℂᵍ₀ :: FT
    ℂᵍˢ :: FT
    ℂᵍˢˢ :: FT
    potential_temperature_tolerance :: FT
    total_water_tolerance :: FT
    vertical_velocity_tolerance :: FT
    minimum_updraft_fraction :: FT
    maximum_updraft_fraction :: FT
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
                                   convert_eltype(FT, clipping))
end

Adapt.@adapt_structure EqualWidthDoubleGaussianPDF

const ComponentDoubleGaussianPDF = Union{DoubleGaussianPDF, EqualWidthDoubleGaussianPDF}

@inline within_component_share(pdf::DoubleGaussianPDF, Sk) = pdf.ℂᵍ₁

@inline function within_component_share(pdf::EqualWidthDoubleGaussianPDF, Sk)
    x = 2 * pdf.ℂᵍ₀ - 1
    y₀ = x / sqrt(1 - x * x)
    y = y₀ + pdf.ℂᵍˢ * Sk + pdf.ℂᵍˢˢ * Sk * Sk
    return (1 + y / sqrt(1 + y * y)) / 2
end

@inline representable_skewness(pdf::ComponentDoubleGaussianPDF) =
    representable_skewness(pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction,
                           within_component_share(pdf, zero(pdf.vertical_velocity_tolerance)))

@inline function representable_skewness(aᵐⁱⁿ, aᵐᵃˣ, c)
    a = max(aᵐⁱⁿ, 1 - aᵐᵃˣ)
    return (1 - 2a) * (1 - c) * sqrt((1 - c) / (a * (1 - a)))
end

@inline function component_variance(clipping, first_component, x̂₁, x̂₂, a, skewness, variance, maximum_ratio)
    unexplained = 1 - a * (x̂₁ * x̂₁) - (1 - a) * (x̂₂ * x̂₂)
    third = skewness - a * (x̂₁ * x̂₁ * x̂₁) - (1 - a) * (x̂₂ * x̂₂ * x̂₂)
    numerator = ifelse(first_component, 3 * x̂₂ * unexplained - third, -3 * x̂₁ * unexplained + third)
    denominator = ifelse(first_component, 3 * a * (x̂₂ - x̂₁), 3 * (1 - a) * (x̂₂ - x̂₁))
    degenerate = denominator == 0
    ratio = ifelse(degenerate, one(numerator), numerator / ifelse(degenerate, one(denominator), denominator))
    return clip_between(clipping, ratio, 0, maximum_ratio, 1) * variance
end

@inline function skewness_share(pdf, d)
    FT = typeof(d)
    n = 2 * (pdf.ℂᵍ₃ + pdf.ℂᵍ₄) / (pdf.ℂᵍ₄ - pdf.ℂᵍ₃)
    separated = d > 0
    x = ifelse(separated, 2 * d / (pdf.ℂᵍ₃ + pdf.ℂᵍ₄), one(FT))
    xⁿ = x^n
    return ifelse(separated, xⁿ / (1 + xⁿ), zero(FT))
end

@inline function scalar_component_parameters(clipping, flux, σw₊, σ₊, variance, x̄, ŵ₁, ŵ₂, Sk, a, skewed, pdf::DoubleGaussianPDF)
    FT = typeof(variance)
    r = clip_between(clipping, flux / (σw₊ * σ₊), -1, 1, 1)

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

@inline function scalar_component_parameters(clipping, flux, σw₊, σ₊, variance, x̄, ŵ₁, ŵ₂, Sk, a, skewed,
                                             pdf::EqualWidthDoubleGaussianPDF)
    r₀ = clip_between(clipping, flux / (σw₊ * σ₊), -1, 1, 1)
    one_minus_share = 1 - within_component_share(pdf, Sk)
    r = realizable_correlation(r₀, one_minus_share, x̄, σ₊, ŵ₁, ŵ₂)
    x̂₁ = -r / ŵ₂
    x̂₂ = -r / ŵ₁
    x′²ᵢ = max(0, 1 - r * r / one_minus_share) * variance
    σ = zero_safe_sqrt(variance)
    x₁ = x̂₁ * σ + x̄
    x₂ = x̂₂ * σ + x̄
    σᵢ = zero_safe_sqrt(x′²ᵢ)
    return x₁, x₂, x′²ᵢ, x′²ᵢ, σᵢ, σᵢ
end

@inline function component_condensate(qᵗ, qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ, θˡⁱ′², qᵗ′², σθ, σq, r)
    FT = typeof(qᵗ)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qᵗ - qᵛ⁺ * ((1 + β * qᵗ) / (1 + β * qᵛ⁺))
    denominator = 1 + β * qᵛ⁺
    c_θ = ((1 + β * qᵗ) / (denominator * denominator)) * cᵖᵐℒ⁻¹ * β * qᵛ⁺ * ∂Tˡ∂θ
    c_q = 1 / (1 + β * qᵛ⁺)
    σₛ = zero_safe_sqrt((c_θ * c_θ) * θˡⁱ′² + (c_q * c_q) * qᵗ′² - 2 * c_θ * σθ * c_q * σq * r)

    resolved = σₛ > sqrt(floatmin(FT)) * 100
    σₛ₊ = ifelse(resolved, σₛ, one(FT))

    wide_fraction = convert(FT, 0.5) * (1 + erf(s / (sqrt2 * σₛ₊)))
    ŝ = s / σₛ₊
    wide_condensate = ifelse(wide_fraction != 0, s * wide_fraction + (σₛ₊ / sqrt2pi) * exp(convert(FT, -0.5) * (ŝ * ŝ)), zero(FT))

    narrow_fraction = ifelse(s > 0, one(FT), zero(FT))
    narrow_condensate = ifelse(s > 0, s, zero(FT))

    aᶜ = ifelse(resolved, wide_fraction, narrow_fraction)
    qᶜ = ifelse(resolved, wide_condensate, narrow_condensate)

    empty = qᶜ <= 0
    aᶜ = ifelse(empty, zero(FT), aᶜ)
    qᶜ = ifelse(empty, zero(FT), qᶜ)

    return s, σₛ, qᶜ, aᶜ
end

@inline function bounded_condensate(qᶜ, qᵗ)
    FT = typeof(qᶜ)
    wet = (qᶜ > 0) & (qᵗ > 0)
    x = ifelse(wet, qᶜ / ifelse(wet, qᵗ, one(FT)), zero(FT))
    x² = x * x
    return ifelse(wet, qᶜ / sqrt(sqrt(1 + x² * x²)), zero(FT))
end

@inline function mixture_cloud(a, aᶜ₁, aᶜ₂, qˡ₁, qˡ₂, s₁, s₂, σₛ₁, σₛ₂)
    aᶜ = min(1, a * aᶜ₁ + (1 - a) * aᶜ₂)
    qˡ = a * qˡ₁ + (1 - a) * qˡ₂
    qˡ′² = max(0, a * (s₁ * qˡ₁ + aᶜ₁ * (σₛ₁ * σₛ₁)) + (1 - a) * (s₂ * qˡ₂ + aᶜ₂ * (σₛ₂ * σₛ₂)) - qˡ * qˡ)
    return aᶜ, qˡ, qˡ′²
end

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
    w′²₊ = clipped_maximum(clipping, calm_variance, w′², calm_variance)
    σw₊ = sqrt(w′²₊)
    σθ₊ = clipped_maximum(clipping, potential_temperature_tolerance, zero_safe_sqrt(θˡⁱ′²), potential_temperature_tolerance)
    σq₊ = clipped_maximum(clipping, total_water_tolerance, zero_safe_sqrt(qᵗ′²), total_water_tolerance)

    Sk = w′³ / (w′²₊ * σw₊)
    share = within_component_share(pdf, Sk)
    one_minus_share = 1 - share
    a = clip_between(clipping, convert(FT, 0.5) * (1 - Sk * sqrt(1 / (4 * (one_minus_share * one_minus_share * one_minus_share) + Sk * Sk))),
                     pdf.minimum_updraft_fraction, pdf.maximum_updraft_fraction, 1)
    sqrt_one_minus_share = sqrt(one_minus_share)
    ŵ₁ = sqrt((1 - a) / a) * sqrt_one_minus_share
    ŵ₂ = -1 * sqrt(a / (1 - a)) * sqrt_one_minus_share

    potential_temperature = scalar_component_parameters(clipping, w′θˡⁱ′, σw₊, σθ₊, θˡⁱ′², θˡⁱ, ŵ₁, ŵ₂, Sk, a, false, pdf)
    total_water = scalar_component_parameters(clipping, w′qᵗ′, σw₊, σq₊, qᵗ′², qᵗ, ŵ₁, ŵ₂, Sk, a, true, pdf)

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

@inline function assumed_pdf(pdf::ComponentDoubleGaussianPDF, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′²,
                             moments_below, moments_above)
    c = double_gaussian_components(pdf, zᶠ⁺, zᶠ⁻, zᶜ, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    a, w′θˡⁱ′, w′qᵗ′, r = c.a, c.w′θˡⁱ′, c.w′qᵗ′, c.r
    θˡⁱ₁, θˡⁱ₂, θˡⁱ′²₁, θˡⁱ′²₂, σθ₁, σθ₂ = c.potential_temperature
    qᵗ₁, qᵗ₂, qᵗ′²₁, qᵗ′²₂, σq₁, σq₂ = c.total_water

    w₁ = c.ŵ₁ * c.σw + w
    w₂ = c.ŵ₂ * c.σw + w

    Tˡ₁ = liquid_water_temperature(thermodynamics, θˡⁱ₁, qᵗ₁, p)
    Tˡ₂ = liquid_water_temperature(thermodynamics, θˡⁱ₂, qᵗ₂, p)

    qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁ = component_saturation(thermodynamics, Tˡ₁, qᵗ₁, p)
    qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂ = component_saturation(thermodynamics, Tˡ₂, qᵗ₂, p)

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

@inline function component_excess_correlation(qᵗ′², θˡⁱ′², σθ, σq, r, aˡ, aⁱ, σ, σⁱ)
    covariance = qᵗ′² - (aˡ + aⁱ) * r * σθ * σq + aˡ * aⁱ * θˡⁱ′²
    spread = σ * σⁱ
    resolved = spread > 0
    return ifelse(resolved, covariance / ifelse(resolved, spread, one(spread)), one(spread))
end

@inline function component_vapor_excess_moments(thermodynamics, p, θˡⁱ, qᵗ, θˡⁱ′², qᵗ′², σθ, σq, r, (surface, equilibrium))
    constants = thermodynamics.constants
    ∂Tˡ∂θ = liquid_water_temperature_slope(thermodynamics, qᵗ, p)
    T = ∂Tˡ∂θ * θˡⁱ
    Rᵐ = mixture_gas_constant(component_moisture(thermodynamics, qᵗ), constants)
    qᵛ⁺ = saturation_specific_humidity(T, p / (Rᵐ * T), constants, surface)
    _, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    ∂qᵛ⁺∂θ = qᵛ⁺ * log_slope * ∂Tˡ∂θ
    σ = zero_safe_sqrt(qᵗ′² + ∂qᵛ⁺∂θ * ∂qᵛ⁺∂θ * θˡⁱ′² - 2 * ∂qᵛ⁺∂θ * r * σθ * σq)
    return qᵗ - qᵛ⁺, σ, ∂qᵛ⁺∂θ
end

@inline function component_vapor_excess(pdf, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′², moments_below, moments_above)
    FT = typeof(θˡⁱ)
    return convert(FT, 0.5), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), one(FT), one(FT)
end
