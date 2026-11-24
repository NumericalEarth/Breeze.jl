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
    "Lowest liquid water temperature of a component, K"
    Tl_min :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SkewedScalarDoubleGaussianPDF`](@ref). Every field is a keyword argument. The defaults are the
within-component share of `w′²` of [`EqualWidthDoubleGaussianPDF`](@ref), `ℂᵍ₀ = 0.4`, the skewness of that mixture
(`ℂᵍᵏ = 1`) and half of the variance of the saturation excess within the components (`ℂᵍᶜ = 0.5`).
"""
function SkewedScalarDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; kw...)
    pdf = component_from_keywords(SkewedScalarDoubleGaussianPDF, FT, (ℂᵍ₀ = 0.4, ℂᵍᵏ = 1, ℂᵍᶜ = 0.5, Tl_min = 100), kw)
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
# within-component variances and correlation of θˡⁱ and qᵗ, and w′s′ / σₛ² (the regression of w on s)
@inline function skewed_scalar_components(pdf, thermodynamics, pval, θ̄, q̄, wsec, w3, θv, wθ, θq, wq, qv)
    FT = typeof(θ̄)
    tiny = sqrt(floatmin(FT))
    # the saturation excess s = c_q qᵗ′ - c_θ θˡⁱ′ linearized at the mean (`plume_condensate`)
    Tl = component_liquid_water_temperature(pdf, thermodynamics, θ̄, q̄, pval)
    qs, β, cp_over_L, Π = plume_saturation(thermodynamics, Tl, q̄, pval)
    opbq = 1 + β * qs
    c_θ = ((1 + β * q̄) / (opbq * opbq)) * cp_over_L * β * qs * Π
    c_q = 1 / opbq
    σs² = c_θ * c_θ * θv + c_q * c_q * qv - 2 * c_θ * c_q * θq
    resolved = (σs² > tiny * tiny) & (wsec > tiny)
    σs²ˢ = ifelse(resolved, σs², one(FT))
    wsecˢ = ifelse(resolved, wsec, one(FT))
    ws = c_q * wq - c_θ * wθ

    # the skewness of s implied by the skewness of w and the correlation of w with s
    Sk_w = w3 / (wsecˢ * sqrt(wsecˢ))
    r = ws / sqrt(wsecˢ * σs²ˢ)
    r̂ = r / sqrt(sqrt(1 + r * r * r * r))
    b = 1 - pdf.ℂᵍ₀
    Sk_s = ifelse(resolved, pdf.ℂᵍᵏ * Sk_w * r̂ * r̂ * r̂ / (b * b * b), zero(FT))

    # the mixture in s, and its components along the regression of (θ, q) on s
    a, ŝ₁, ŝ₂ = skewed_mixture(Sk_s, pdf.ℂᵍᶜ)
    σs = sqrt(σs²ˢ)
    g_θ = ifelse(resolved, (c_q * θq - c_θ * θv) / σs²ˢ, zero(FT))
    g_q = ifelse(resolved, (c_q * qv - c_θ * θq) / σs²ˢ, zero(FT))
    δ₁ = ŝ₁ * σs
    δ₂ = ŝ₂ * σs
    between = (1 - pdf.ℂᵍᶜ) * ifelse(resolved, σs², zero(FT))
    θv_c = θv - between * g_θ * g_θ
    qv_c = qv - between * g_q * g_q
    θq_c = θq - between * g_θ * g_q
    σθ_c = zero_safe_sqrt(θv_c)
    σq_c = zero_safe_sqrt(qv_c)
    spread = σθ_c * σq_c
    r_c = ifelse(spread > 0, θq_c / ifelse(spread > 0, spread, one(FT)), zero(FT))
    regression = ifelse(resolved, ws / σs²ˢ, zero(FT))
    return (; a, θ₁ = θ̄ + g_θ * δ₁, q₁ = q̄ + g_q * δ₁, θ₂ = θ̄ + g_θ * δ₂, q₂ = q̄ + g_q * δ₂,
              θv_c, qv_c, θq_c, σθ_c, σq_c, r_c, regression, c_θ, c_q)
end

# The moments at the level: linearly interpolated from the interfaces below and above (non-negative variances stay so)
@inline function skewed_scalar_level_moments(zi_u, zi_l, zt_k, moments_l, moments_u)
    interpolate(n) = linear_interp(zi_u, zi_l, moments_u[n], moments_l[n], zt_k)
    return interpolate(1), interpolate(2), interpolate(3), interpolate(4), interpolate(5), interpolate(6)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The assumed PDF of level `k` for [`SkewedScalarDoubleGaussianPDF`](@ref): the subgrid cloud fraction, liquid water,
liquid water variance, liquid water flux and buoyancy flux.
"""
@inline function assumed_pdf(pdf::SkewedScalarDoubleGaussianPDF, thermodynamics, zi_u, zi_l, zt_k, pval, thl_first, w_first,
                             qw_first, wsec, moments_l, moments_u)
    w3, θv, wθ, θq, wq, qv = skewed_scalar_level_moments(zi_u, zi_l, zt_k, moments_l, moments_u)
    m = skewed_scalar_components(pdf, thermodynamics, pval, thl_first, qw_first, wsec, w3, θv, wθ, θq, wq, qv)
    a = m.a

    Tl₁ = component_liquid_water_temperature(pdf, thermodynamics, m.θ₁, m.q₁, pval)
    Tl₂ = component_liquid_water_temperature(pdf, thermodynamics, m.θ₂, m.q₂, pval)
    qs₁, β₁, cpL₁, Π₁ = plume_saturation(thermodynamics, Tl₁, m.q₁, pval)
    qs₂, β₂, cpL₂, Π₂ = plume_saturation(thermodynamics, Tl₂, m.q₂, pval)
    s₁, σ₁, ql₁, C₁ = plume_condensate(m.q₁, qs₁, β₁, cpL₁, Π₁, m.θv_c, m.qv_c, m.σθ_c, m.σq_c, m.r_c)
    s₂, σ₂, ql₂, C₂ = plume_condensate(m.q₂, qs₂, β₂, cpL₂, Π₂, m.θv_c, m.qv_c, m.σθ_c, m.σq_c, m.r_c)

    cloud_fraction = a * C₁ + (1 - a) * C₂
    ql = a * ql₁ + (1 - a) * ql₂
    # E[s qˡ] of each Gaussian component is μ qˡ + σ² C; the variance of the liquid (to round-off non-negative)
    second₁ = s₁ * ql₁ + C₁ * σ₁ * σ₁
    second₂ = s₂ * ql₂ + C₂ * σ₂ * σ₂
    ql2 = max(0, a * second₁ + (1 - a) * second₂ - ql * ql)
    s̄ = a * s₁ + (1 - a) * s₂
    # cov(s, qˡ), and the flux of liquid that the regression of w on s carries
    covariance = a * (second₁ - s̄ * ql₁) + (1 - a) * (second₂ - s̄ * ql₂)
    wql = m.regression * covariance

    A_q, A_c = buoyancy_flux_coefficients(thermodynamics, pval, thl_first, qw_first, ql)
    wthv = wθ + A_q * wq + A_c * wql
    return cloud_fraction, ql, ql2, wql, wthv
end

# The components in the vapor excess over liquid and over ice, for the microphysics and the saturated share of the fluxes
@inline function component_vapor_excess(pdf::SkewedScalarDoubleGaussianPDF, thermodynamics, zi_u, zi_l, zt_k, pval,
                                        thl_first, w_first, qw_first, wsec, moments_l, moments_u)
    w3, θv, wθ, θq, wq, qv = skewed_scalar_level_moments(zi_u, zi_l, zt_k, moments_l, moments_u)
    m = skewed_scalar_components(pdf, thermodynamics, pval, thl_first, qw_first, wsec, w3, θv, wθ, θq, wq, qv)
    liquid = (PlanarLiquidSurface(), WarmPhaseEquilibrium())
    ice = (PlanarIceSurface(), IceSurfaceEquilibrium())
    moments(θ, q, surface) = component_vapor_excess_moments(pdf, thermodynamics, pval, θ, q, m.θv_c, m.qv_c, m.σθ_c, m.σq_c, m.r_c, surface)
    μ₁, σ₁, ∂₁ = moments(m.θ₁, m.q₁, liquid)
    μ₂, σ₂, ∂₂ = moments(m.θ₂, m.q₂, liquid)
    μⁱ₁, σⁱ₁, ∂ⁱ₁ = moments(m.θ₁, m.q₁, ice)
    μⁱ₂, σⁱ₂, ∂ⁱ₂ = moments(m.θ₂, m.q₂, ice)
    ρ₁ = component_excess_correlation(m.qv_c, m.θv_c, m.σθ_c, m.σq_c, m.r_c, ∂₁, ∂ⁱ₁, σ₁, σⁱ₁)
    ρ₂ = component_excess_correlation(m.qv_c, m.θv_c, m.σθ_c, m.σq_c, m.r_c, ∂₂, ∂ⁱ₂, σ₂, σⁱ₂)
    return m.a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂, ρ₁, ρ₂
end
