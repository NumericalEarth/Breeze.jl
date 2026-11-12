#####
##### The assumed double-Gaussian PDF without switches
#####

"""
$(TYPEDEF)

The Analytic Double Gaussian 1 of [`DoubleGaussianPDF`](@ref) with every switch of E3SM
`shoc.F90` replaced by a regularization that has the switch's limit, so that the cloud fraction,
the liquid water and the buoyancy flux are smooth functions of the moments and of the
coefficients:

| `DoubleGaussianPDF` | `SmoothDoubleGaussianPDF` |
|:--------------------|:--------------------------|
| calm, `w'w' ≤ w_tol²`: no skewness, equal plumes at the mean | skewness `w'³ / (w'w' + w_tol²)^{3/2}`, which vanishes with `w'w'`; the plumes collapse onto the mean as `√w'w'` |
| `√θˡ'²` and `√qᵗ'²` floored at their tolerance | `√(variance + tolerance²)` |
| uniform, variance ≤ tolerance²: plumes at the mean, no width | plume separation scaled by `variance / √(variance + tolerance²)`; the widths vanish with the variance |
| in-plume variance from a ratio that is `0 / 0` where the plume means coincide, set to zero there | the same ratio with the common factor cancelled, which tends to one |
| skewness of total water ramping linearly between the plume separations `ℂᵍ₃` and `ℂᵍ₄` | a cubic smooth step over the same interval |
| in-plume correlation zero where the plumes have no width | covariance over the product of the widths plus a tenth of the product of the tolerances |
| saturation deficit of no spread: all or nothing | spread `√(variance + minimum_saturation_spread²)`, the error function everywhere |
| no liquid where its mean is not positive | the mean of the positive part of a Gaussian, which is positive |

The bounds of the plume fraction, the correlations and the variance ratio go through the clipping
policy, as in `DoubleGaussianPDF`. Where the variances are well above their tolerances and the
plumes well apart the two agree closely; in calm or nearly uniform air this PDF keeps the Gaussian
spread that `shoc.F90` drops.

$(TYPEDFIELDS)
"""
struct SmoothDoubleGaussianPDF{FT} <: AbstractSHOCComponent
    "Within-plume fraction of the vertical velocity variance (literal `0.4`)"
    ℂᵍ₁ :: FT
    "Skewness of total water relative to that of vertical velocity (literal `1.2`)"
    ℂᵍ₂ :: FT
    "Plume separation below which total water is unskewed (literal `0.2`)"
    ℂᵍ₃ :: FT
    "Plume separation above which total water is fully skewed (literal `0.4`)"
    ℂᵍ₄ :: FT
    "Standard deviation of θˡ that regularizes its vanishing, K"
    thl_tol :: FT
    "Standard deviation of qᵗ that regularizes its vanishing, kg kg⁻¹"
    rt_tol :: FT
    "Standard deviation of vertical velocity that regularizes its vanishing, m s⁻¹"
    w_tol :: FT
    "Lowest liquid water temperature of a plume, K"
    Tl_min :: FT
    "Lowest fraction of the first plume (literal `0.01`)"
    plume_fraction_min :: FT
    "Highest fraction of the first plume (literal `0.99`)"
    plume_fraction_max :: FT
    "Highest variance of a scalar within a plume over its variance (literal `100`)"
    plume_variance_ratio_max :: FT
    "Smallest spread of the saturation deficit within a plume, kg kg⁻¹"
    minimum_saturation_spread :: FT
    "Floor of the width of a plume over the tolerance of its scalar, which keeps the derivative of the width finite"
    plume_width_floor :: FT
    "Regularization of the in-plume correlation of θˡ and qᵗ: the fraction of `thl_tol × rt_tol` added to the product of the widths"
    correlation_regularization :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `SmoothDoubleGaussianPDF{FT}` with the coefficients and tolerances of E3SM `shoc.F90`.
Every field is a keyword argument.
"""
SmoothDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(SmoothDoubleGaussianPDF, FT,
                            (ℂᵍ₁ = 0.4, ℂᵍ₂ = 1.2, ℂᵍ₃ = 0.2, ℂᵍ₄ = 0.4, thl_tol = 1e-2, rt_tol = 1e-4, w_tol = 2e-2, Tl_min = 100,
                             plume_fraction_min = 0.01, plume_fraction_max = 0.99, plume_variance_ratio_max = 100,
                             minimum_saturation_spread = 1e-6, plume_width_floor = 1e-3, correlation_regularization = 0.1), kw)

Adapt.@adapt_structure SmoothDoubleGaussianPDF

# 0 below 0, 1 above 1, and 3x² - 2x³ between: continuous with its first derivative
@inline function smooth_step(x)
    x = max(0, min(1, x))
    return x * x * (3 - 2 * x)
end

# The plume means of a scalar relative to its mean, in units of its standard deviation, and the
# in-plume variances over the variance, from the correlation `corr` of the scalar with w and the
# plume velocities `w1_1 > 0 > w1_2` (in units of the standard deviation of w, so that
# `w1_1 w1_2 = -(1 - ℂᵍ₁)` and `a w1_1 + (1 - a) w1_2 = 0`). The ratios are those of
# `plume_variance` with the factor `corr` that numerator and denominator share cancelled.
@inline function smooth_scalar_plumes(pdf, cp, corr, w1_1, w1_2, Skew_w, a, skewed)
    FT = typeof(corr)
    α = w1_1
    β = -w1_2
    s² = α * β                     # 1 - ℂᵍ₁

    x1_1 = corr / β
    x1_2 = -corr / α
    separation = x1_2 - x1_1

    # The skewness of the scalar, which rises smoothly from zero as the plumes separate; where
    # it is zero the division by the separation is not carried out.
    tsign = abs(separation)
    skew = ifelse(skewed, pdf.ℂᵍ₂ * Skew_w * smooth_step((tsign - pdf.ℂᵍ₃) / (pdf.ℂᵍ₄ - pdf.ℂᵍ₃)), zero(FT))
    close = tsign < pdf.ℂᵍ₃ / 2
    skew_over_separation = ifelse(close, zero(FT), skew / ifelse(close, one(FT), separation))

    unexplained = 1 - corr * corr / s²                          # 1 - a x1_1² - (1 - a) x1_2²
    asymmetry = corr * corr * (a * α) * (β * β - α * α) / (s² * s² * s²)

    ratio_1 = s² * (3 / α * unexplained + asymmetry) / (3 * a * (α + β)) - skew_over_separation / (3 * a)
    ratio_2 = s² * (3 / β * unexplained - asymmetry) / (3 * (1 - a) * (α + β)) + skew_over_separation / (3 * (1 - a))

    ratio_max = pdf.plume_variance_ratio_max
    return x1_1, x1_2, clip_between(cp, ratio_1, 0, ratio_max, 1), clip_between(cp, ratio_2, 0, ratio_max, 1)
end

# shoc_assumed_pdf_compute_s with a spread that does not vanish
@inline function smooth_plume_condensate(pdf, qw1, qs1, beta, cp_over_L, Π, thl2, qw2, sqrtthl2, sqrtqw2, r_qwthl)
    FT = typeof(qw1)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qw1 - qs1 * ((1 + beta * qw1) / (1 + beta * qs1))
    opbq = 1 + beta * qs1
    cthl = ((1 + beta * qw1) / (opbq * opbq)) * cp_over_L * beta * qs1 * Π
    cqt = 1 / (1 + beta * qs1)
    # Not negative but for round-off
    variance = max(0, (cthl * cthl) * thl2 + (cqt * cqt) * qw2 - 2 * cthl * sqrtthl2 * cqt * sqrtqw2 * r_qwthl)
    std_s = sqrt(variance + pdf.minimum_saturation_spread * pdf.minimum_saturation_spread)

    sos = s / std_s
    C = convert(FT, 0.5) * (1 + erf(sos / sqrt2))
    # The mean of the positive part of the saturation excess, which is not negative but for round-off
    qn = max(0, s * C + (std_s / sqrt2pi) * exp(convert(FT, -0.5) * (sos * sos)))

    return s, std_s, qn, C
end

@inline function assumed_pdf(pdf::SmoothDoubleGaussianPDF, sat, thermodynamics, cp, c, zi_u, zi_l, zt_k, pval, thl_first,
                             w_first, qw_first, wsec, moments_l, moments_u)
    FT = typeof(thl_first)
    largeneg = convert(FT, -99999999.99)
    thl_tol = pdf.thl_tol
    rt_tol = pdf.rt_tol
    w_tol_sqd = pdf.w_tol * pdf.w_tol

    # Interpolate the interface moments to the level
    interpolate(n, minthresh) = hard_floor(linear_interp(zi_u, zi_l, moments_u[n], moments_l[n], zt_k), minthresh)
    w3var = interpolate(1, largeneg)
    thlsec = interpolate(2, zero(FT))
    wthlsec = interpolate(3, largeneg)
    qwthlsec = interpolate(4, largeneg)
    wqwsec = interpolate(5, largeneg)
    qwsec = interpolate(6, zero(FT))

    # The vertical velocity variance is floored with the turbulent kinetic energy; the scalar
    # variances may vanish
    sqrtw2 = zero_safe_sqrt(wsec)
    sqrtthl = sqrt(thlsec + thl_tol * thl_tol)
    sqrtqt = sqrt(qwsec + rt_tol * rt_tol)

    # Vertical velocity: skewness, plume fraction, plume velocities over the standard deviation
    wsec_regular = wsec + w_tol_sqd
    Skew_w = w3var / sqrt(wsec_regular * wsec_regular * wsec_regular)
    w2_1 = pdf.ℂᵍ₁
    omw = 1 - w2_1
    a = clip_between(cp, convert(FT, 0.5) * (1 - Skew_w * sqrt(1 / (4 * (omw * omw * omw) + Skew_w * Skew_w))),
                     pdf.plume_fraction_min, pdf.plume_fraction_max, 1)
    sqrtw2t = sqrt(omw)
    w1_1 = sqrt((1 - a) / a) * sqrtw2t
    w1_2 = -1 * sqrt(a / (1 - a)) * sqrtw2t

    # The scalars: correlation with w, plume means and in-plume variances
    corr_thl = clip_between(cp, wthlsec / (sqrtw2 * sqrtthl), -1, 1, 1)
    corr_qw = clip_between(cp, wqwsec / (sqrtw2 * sqrtqt), -1, 1, 1)

    thl1_1, thl1_2, thl_ratio_1, thl_ratio_2 = smooth_scalar_plumes(pdf, cp, corr_thl, w1_1, w1_2, Skew_w, a, false)
    qw1_1, qw1_2, qw_ratio_1, qw_ratio_2 = smooth_scalar_plumes(pdf, cp, corr_qw, w1_1, w1_2, Skew_w, a, true)

    # The separation of the plumes is √variance where the variance is resolved, and vanishes
    # linearly with it below the tolerance
    thl_separation = thlsec / sqrtthl
    qw_separation = qwsec / sqrtqt
    thl1_1 = thl1_1 * thl_separation + thl_first
    thl1_2 = thl1_2 * thl_separation + thl_first
    qw1_1 = qw1_1 * qw_separation + qw_first
    qw1_2 = qw1_2 * qw_separation + qw_first

    thl2_1 = thl_ratio_1 * thlsec
    thl2_2 = thl_ratio_2 * thlsec
    qw2_1 = qw_ratio_1 * qwsec
    qw2_2 = qw_ratio_2 * qwsec

    # Widths with a floor far below the tolerances, so that their derivative stays finite
    thl_width_floor = pdf.plume_width_floor * thl_tol
    qw_width_floor = pdf.plume_width_floor * rt_tol
    sqrtthl2_1 = sqrt(thl2_1 + thl_width_floor * thl_width_floor)
    sqrtthl2_2 = sqrt(thl2_2 + thl_width_floor * thl_width_floor)
    sqrtqw2_1 = sqrt(qw2_1 + qw_width_floor * qw_width_floor)
    sqrtqw2_2 = sqrt(qw2_2 + qw_width_floor * qw_width_floor)

    # The plume velocities
    w1_1 = w1_1 * sqrtw2 + w_first
    w1_2 = w1_2 * sqrtw2 + w_first

    # The correlation of θˡ and qᵗ within the plumes
    testvar = a * sqrtqw2_1 * sqrtthl2_1 + (1 - a) * sqrtqw2_2 * sqrtthl2_2
    r_qwthl_1 = clip_between(cp, (qwthlsec - a * (qw1_1 - qw_first) * (thl1_1 - thl_first)
                                  - (1 - a) * (qw1_2 - qw_first) * (thl1_2 - thl_first)) /
                                 (testvar + pdf.correlation_regularization * thl_tol * rt_tol), -1, 1, 1)

    # Liquid water temperature and saturation of the plumes
    Tl1_1 = max(pdf.Tl_min, plume_liquid_water_temperature(sat, thermodynamics, c, thl1_1, qw1_1, pval))
    Tl1_2 = max(pdf.Tl_min, plume_liquid_water_temperature(sat, thermodynamics, c, thl1_2, qw1_2, pval))

    qs1, beta1, cp_over_L1, Π1 = plume_saturation(sat, thermodynamics, c, Tl1_1, qw1_1, pval)
    qs2, beta2, cp_over_L2, Π2 = plume_saturation(sat, thermodynamics, c, Tl1_2, qw1_2, pval)

    s1, std_s1, qn1, C1 = smooth_plume_condensate(pdf, qw1_1, qs1, beta1, cp_over_L1, Π1, thl2_1, qw2_1, sqrtthl2_1, sqrtqw2_1, r_qwthl_1)
    s2, std_s2, qn2, C2 = smooth_plume_condensate(pdf, qw1_2, qs2, beta2, cp_over_L2, Π2, thl2_2, qw2_2, sqrtthl2_2, sqrtqw2_2, r_qwthl_1)

    ql1 = clip_min(cp, qn1, qw1_1, rt_tol)
    ql2 = clip_min(cp, qn2, qw1_2, rt_tol)

    shoc_cldfrac = min(1, a * C1 + (1 - a) * C2)
    shoc_ql = max(0, a * ql1 + (1 - a) * ql2)

    shoc_ql2 = a * (s1 * ql1 + C1 * (std_s1 * std_s1)) + (1 - a) * (s2 * ql2 + C2 * (std_s2 * std_s2)) - (shoc_ql * shoc_ql)
    shoc_ql2 = max(0, shoc_ql2)

    wqls = a * ((w1_1 - w_first) * ql1) + (1 - a) * ((w1_2 - w_first) * ql2)

    A_q, A_c = buoyancy_flux_coefficients(sat, thermodynamics, c, pval, thl_first, qw_first, shoc_ql)
    wthv_sec = wthlsec + A_q * wqwsec + A_c * wqls

    return shoc_cldfrac, shoc_ql, shoc_ql2, wqls, wthv_sec
end
