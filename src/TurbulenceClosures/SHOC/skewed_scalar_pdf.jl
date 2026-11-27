#####
##### Slot: the assumed PDF. A double Gaussian of the saturation excess whose skewness is the one the third moment of
##### vertical velocity and the flux of the excess imply
#####

"""
$(TYPEDEF)

An assumed PDF of the subgrid distribution in which the saturation excess ``s`` of the distributed water is a double
Gaussian skewed by the third moment of vertical velocity. Cumulus cloud is the skewed tail of ``s``: its updrafts are
few, strong and moist. The skewness of ``s`` is the one that the skewness ``Sk_w = w′³ / (w′²)^{3/2}`` of vertical
velocity and the correlation ``r = w′s′ / \\sqrt{w′² s′²}`` imply in a two-component mixture whose components share
their weight and their within-component share ``ℂᵍ₀`` of ``w′²`` and carry the flux by the difference of their
means:

    Sk_s = ℂᵍᵏ Sk_w r̂³ / (1 - ℂᵍ₀)³,    r̂ = r / (1 + r⁴)^{1/4},

with the coefficient ``ℂᵍᵏ`` (one in that mixture) and the correlation bounded below one by a rational function. The
distribution of ``s`` is then the two-component mixture of this skewness with its own within-component share ``ℂᵍᶜ``
of ``s′²``,

    a = (1 - Sk_s / \\sqrt{4 (1 - ℂᵍᶜ)³ + Sk_s²}) / 2,    s₁ - s̄ = \\sqrt{(1 - a) / a} \\sqrt{1 - ℂᵍᶜ} σₛ,    s₂ - s̄ = -\\sqrt{a / (1 - a)} \\sqrt{1 - ℂᵍᶜ} σₛ,

the weight ``a`` of the moist component, in ``(0, 1)`` for every finite skewness, without bounds. In θˡⁱ and qᵗ the
components lie along the regression of the two scalars on ``s``, ``(θ, q)ᵢ = (θ̄, q̄) + Σ c (sᵢ - s̄) / σₛ²`` with
``c = (-c_θ, c_q)`` the coefficients of the saturation excess at the mean and ``Σ`` the covariance of the scalars,
and each has the covariance ``Σ - (1 - ℂᵍᶜ) Σ c cᵀ Σ / σₛ²``, positive semidefinite for ``ℂᵍᶜ ∈ (0, 1)``: the
mixture has the means, variances and covariance it is given. Each component condenses with the model's
thermodynamics and the phase equilibrium of the condensate of its microphysics, linearized about its own state.
The flux of liquid is that of the regression of ``w`` on ``s``, ``w′qˡ′ = (w′s′ / σₛ²) cov(s, qˡ)``.

$(TYPEDFIELDS)
"""
struct SkewedScalarDoubleGaussianPDF{FT} <: AbstractSubgridDistributionComponent
    "Within-component fraction of the vertical velocity variance of the mixture that implies the skewness of the saturation excess"
    ℂᵍ₀ :: FT
    "Skewness of the saturation excess relative to the one that mixture implies"
    ℂᵍᵏ :: FT
    "Within-component fraction of the variance of the saturation excess"
    ℂᵍᶜ :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SkewedScalarDoubleGaussianPDF`](@ref). Every field is a keyword argument. The defaults are the
within-component share of `w′²` of [`EqualWidthDoubleGaussianPDF`](@ref), `ℂᵍ₀ = 0.4`, the skewness of that mixture
(`ℂᵍᵏ = 1`) and half of the variance of the saturation excess within the components (`ℂᵍᶜ = 0.5`).
"""
function SkewedScalarDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; kw...)
    pdf = component_from_keywords(SkewedScalarDoubleGaussianPDF, FT,
                                  (ℂᵍ₀ = 0.4, ℂᵍᵏ = 1, ℂᵍᶜ = 0.5), kw)
    0 < pdf.ℂᵍ₀ < 1 || throw(ArgumentError("ℂᵍ₀ must be in (0, 1), got $(pdf.ℂᵍ₀)"))
    0 < pdf.ℂᵍᶜ < 1 || throw(ArgumentError("ℂᵍᶜ must be in (0, 1), got $(pdf.ℂᵍᶜ)"))
    return pdf
end

Adapt.@adapt_structure SkewedScalarDoubleGaussianPDF

# The within-component share of w′² of the mixture that implies the skewness (the transport of the scalar variances
# reads it), and the largest skewness of w its weights represent at the bounds of the other double Gaussians, the
# bound of the third-moment limiter that `DoubleGaussian` takes by default
@inline within_component_share(pdf::SkewedScalarDoubleGaussianPDF, Sk) = pdf.ℂᵍ₀
@inline representable_skewness(pdf::SkewedScalarDoubleGaussianPDF) = representable_skewness(convert(typeof(pdf.ℂᵍ₀), 0.01), convert(typeof(pdf.ℂᵍ₀), 0.99), pdf.ℂᵍ₀)

# The weight of the first component of a two-component mixture of equal within-component share c and skewness Sk, and
# the standardized deviations of the two component means
@inline function skewed_mixture(Sk, c)
    b = 1 - c
    a = (1 - Sk / sqrt(4 * b * b * b + Sk * Sk)) / 2
    return a, sqrt(b * (1 - a) / a), -sqrt(b * a / (1 - a))
end

# The components of the mixture at a level: the weight a of the moist component, its (θ, q) and the other's, the
# within-component variances and correlation of θˡⁱ and qᵗ (subscript ᵢ: those of either component), and w′s′ / σₛ²
# (the regression of w on s)
@inline function skewed_scalar_components(pdf, thermodynamics, p, θ̄, q̄, w′², w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
    FT = typeof(θ̄)
    tiny = sqrt(floatmin(FT))
    # the saturation excess s = c_q qᵗ′ - c_θ θˡⁱ′ linearized at the mean (`component_condensate`)
    Tˡ = liquid_water_temperature(thermodynamics, θ̄, q̄, p)
    qᵛ⁺, β, cᵖᵐℒ⁻¹, ∂Tˡ∂θ = component_saturation(thermodynamics, Tˡ, q̄, p)
    denominator = 1 + β * qᵛ⁺
    c_θ = ((1 + β * q̄) / (denominator * denominator)) * cᵖᵐℒ⁻¹ * β * qᵛ⁺ * ∂Tˡ∂θ
    c_q = 1 / denominator
    σₛ² = c_θ * c_θ * θˡⁱ′² + c_q * c_q * qᵗ′² - 2 * c_θ * c_q * θˡⁱ′qᵗ′
    resolved = (σₛ² > tiny * tiny) & (w′² > tiny)
    σₛ²ˢᵃᶠᵉ = ifelse(resolved, σₛ², one(FT))
    w′²ˢᵃᶠᵉ = ifelse(resolved, w′², one(FT))
    w′s′ = c_q * w′qᵗ′ - c_θ * w′θˡⁱ′

    # the skewness of s implied by the skewness of w and the correlation of w with s
    Sk_w = w′³ / (w′²ˢᵃᶠᵉ * sqrt(w′²ˢᵃᶠᵉ))
    r = w′s′ / sqrt(w′²ˢᵃᶠᵉ * σₛ²ˢᵃᶠᵉ)
    r̂ = r / sqrt(sqrt(1 + r * r * r * r))
    b = 1 - pdf.ℂᵍ₀
    Sk_s = ifelse(resolved, pdf.ℂᵍᵏ * Sk_w * r̂ * r̂ * r̂ / (b * b * b), zero(FT))

    # the mixture in s, and its components along the regression of (θ, q) on s
    a, ŝ₁, ŝ₂ = skewed_mixture(Sk_s, pdf.ℂᵍᶜ)
    σₛ = sqrt(σₛ²ˢᵃᶠᵉ)
    g_θ = ifelse(resolved, (c_q * θˡⁱ′qᵗ′ - c_θ * θˡⁱ′²) / σₛ²ˢᵃᶠᵉ, zero(FT))
    g_q = ifelse(resolved, (c_q * qᵗ′² - c_θ * θˡⁱ′qᵗ′) / σₛ²ˢᵃᶠᵉ, zero(FT))
    δ₁ = ŝ₁ * σₛ
    δ₂ = ŝ₂ * σₛ
    between = (1 - pdf.ℂᵍᶜ) * ifelse(resolved, σₛ², zero(FT))
    θˡⁱ′²ᵢ = θˡⁱ′² - between * g_θ * g_θ
    qᵗ′²ᵢ = qᵗ′² - between * g_q * g_q
    θˡⁱ′qᵗ′ᵢ = θˡⁱ′qᵗ′ - between * g_θ * g_q
    σθᵢ = zero_safe_sqrt(θˡⁱ′²ᵢ)
    σqᵢ = zero_safe_sqrt(qᵗ′²ᵢ)
    spread = σθᵢ * σqᵢ
    rᵢ = ifelse(spread > 0, θˡⁱ′qᵗ′ᵢ / ifelse(spread > 0, spread, one(FT)), zero(FT))
    regression = ifelse(resolved, w′s′ / σₛ²ˢᵃᶠᵉ, zero(FT))
    return (; a, θ₁ = θ̄ + g_θ * δ₁, q₁ = q̄ + g_q * δ₁, θ₂ = θ̄ + g_θ * δ₂, q₂ = q̄ + g_q * δ₂,
              θˡⁱ′²ᵢ, qᵗ′²ᵢ, θˡⁱ′qᵗ′ᵢ, σθᵢ, σqᵢ, rᵢ, regression, c_θ, c_q)
end

# The moments at the level: linearly interpolated from the interfaces below and above (non-negative variances stay so)
@inline function skewed_scalar_level_moments(zᶠ⁺, zᶠ⁻, zᶜ, moments_below, moments_above)
    interpolate(n) = linearly_interpolate(zᶠ⁺, zᶠ⁻, moments_above[n], moments_below[n], zᶜ)
    return interpolate(1), interpolate(2), interpolate(3), interpolate(4), interpolate(5), interpolate(6)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The assumed PDF of level `k` for [`SkewedScalarDoubleGaussianPDF`](@ref): the subgrid cloud fraction, liquid water,
liquid water variance, liquid water flux and buoyancy flux.
"""
@inline function assumed_pdf(pdf::SkewedScalarDoubleGaussianPDF, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′²,
                             moments_below, moments_above)
    w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′² = skewed_scalar_level_moments(zᶠ⁺, zᶠ⁻, zᶜ, moments_below, moments_above)
    m = skewed_scalar_components(pdf, thermodynamics, p, θˡⁱ, qᵗ, w′², w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
    a = m.a

    Tˡ₁ = liquid_water_temperature(thermodynamics, m.θ₁, m.q₁, p)
    Tˡ₂ = liquid_water_temperature(thermodynamics, m.θ₂, m.q₂, p)
    qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁ = component_saturation(thermodynamics, Tˡ₁, m.q₁, p)
    qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂ = component_saturation(thermodynamics, Tˡ₂, m.q₂, p)
    s₁, σ₁, qˡ₁, aᶜ₁ = component_condensate(m.q₁, qᵛ⁺₁, β₁, cᵖᵐℒ⁻¹₁, ∂Tˡ∂θ₁, m.θˡⁱ′²ᵢ, m.qᵗ′²ᵢ, m.σθᵢ, m.σqᵢ, m.rᵢ)
    s₂, σ₂, qˡ₂, aᶜ₂ = component_condensate(m.q₂, qᵛ⁺₂, β₂, cᵖᵐℒ⁻¹₂, ∂Tˡ∂θ₂, m.θˡⁱ′²ᵢ, m.qᵗ′²ᵢ, m.σθᵢ, m.σqᵢ, m.rᵢ)

    aᶜ = a * aᶜ₁ + (1 - a) * aᶜ₂
    qˡ = a * qˡ₁ + (1 - a) * qˡ₂
    # E[s qˡ] of each Gaussian component is μ qˡ + σ² aᶜ; the variance of the liquid (to round-off non-negative)
    second₁ = s₁ * qˡ₁ + aᶜ₁ * σ₁ * σ₁
    second₂ = s₂ * qˡ₂ + aᶜ₂ * σ₂ * σ₂
    qˡ′² = max(0, a * second₁ + (1 - a) * second₂ - qˡ * qˡ)
    s̄ = a * s₁ + (1 - a) * s₂
    # cov(s, qˡ), and the flux of liquid that the regression of w on s carries
    covariance = a * (second₁ - s̄ * qˡ₁) + (1 - a) * (second₂ - s̄ * qˡ₂)
    w′qˡ′ = m.regression * covariance

    Aᵗ, Aᶜ = buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
    w′θᵛ′ = w′θˡⁱ′ + Aᵗ * w′qᵗ′ + Aᶜ * w′qˡ′
    return aᶜ, qˡ, qˡ′², w′qˡ′, w′θᵛ′
end

# The components in the vapor excess over liquid and over ice, for the microphysics and the saturated share of the fluxes
@inline function component_vapor_excess(pdf::SkewedScalarDoubleGaussianPDF, thermodynamics, zᶠ⁺, zᶠ⁻, zᶜ, p, θˡⁱ, w, qᵗ, w′²,
                                        moments_below, moments_above)
    w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′² = skewed_scalar_level_moments(zᶠ⁺, zᶠ⁻, zᶜ, moments_below, moments_above)
    m = skewed_scalar_components(pdf, thermodynamics, p, θˡⁱ, qᵗ, w′², w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
    liquid = (PlanarLiquidSurface(), WarmPhaseEquilibrium())
    ice = (PlanarIceSurface(), IceSurfaceEquilibrium())
    moments(θ, q, surface) = component_vapor_excess_moments(thermodynamics, p, θ, q, m.θˡⁱ′²ᵢ, m.qᵗ′²ᵢ, m.σθᵢ, m.σqᵢ, m.rᵢ, surface)
    μ₁, σ₁, ∂₁ = moments(m.θ₁, m.q₁, liquid)
    μ₂, σ₂, ∂₂ = moments(m.θ₂, m.q₂, liquid)
    μⁱ₁, σⁱ₁, ∂ⁱ₁ = moments(m.θ₁, m.q₁, ice)
    μⁱ₂, σⁱ₂, ∂ⁱ₂ = moments(m.θ₂, m.q₂, ice)
    r₁ = component_excess_correlation(m.qᵗ′²ᵢ, m.θˡⁱ′²ᵢ, m.σθᵢ, m.σqᵢ, m.rᵢ, ∂₁, ∂ⁱ₁, σ₁, σⁱ₁)
    r₂ = component_excess_correlation(m.qᵗ′²ᵢ, m.θˡⁱ′²ᵢ, m.σθᵢ, m.σqᵢ, m.rᵢ, ∂₂, ∂ⁱ₂, σ₂, σⁱ₂)
    return m.a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂, r₁, r₂
end
