#####
##### Slot: the assumed double-Gaussian PDF (shoc_assumed_pdf and its helpers)
#####
##### An assumed-PDF component implements
#####
#####     assumed_pdf(pdf, saturation, thermodynamics, cp, c, zi_u, zi_l, zt_k, pval, thl, w, qw, w_sec, moments_l, moments_u)
#####         -> (cldfrac, ql, ql2, wqls, wthv_sec)
#####
##### for a level at height `zt_k` between the interfaces at `zi_l` and `zi_u`: from the pressure,
##### the means and the vertical velocity variance of the level and the interface moments
##### `(w3, thl_sec, wthl_sec, qwthl_sec, wqw_sec, qw_sec)` below and above it, the subgrid cloud
##### fraction, liquid water, liquid water variance, liquid water flux and buoyancy flux.
#####

"""
$(TYPEDEF)

The Analytic Double Gaussian 1 of Larson et al. (2002) as E3SM `shoc.F90` computes it
(`shoc_assumed_pdf`): two Gaussian plumes in vertical velocity, θˡ and qᵗ whose weights, means
and widths follow from the first three moments of vertical velocity and the second moments of
the scalars, each condensing where it is saturated. The Fortran switches are kept: where the
vertical velocity variance is below `w_tol²` or a scalar variance below its tolerance squared,
the plumes collapse onto the mean.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵍ₁` | `w2_1`, `w2_2` (literal `0.4`) | 0.4 | within-plume fraction of the vertical velocity variance |
| `ℂᵍ₂` | literal `1.2`    | 1.2    | skewness of total water relative to that of vertical velocity |
| `ℂᵍ₃` | literal `0.2`    | 0.2    | plume separation below which total water is unskewed |
| `ℂᵍ₄` | literal `0.4`    | 0.4    | plume separation above which total water takes the full skewness `ℂᵍ₂ Skew_w` |

Between `ℂᵍ₃` and `ℂᵍ₄` the skewness of total water ramps linearly; the Fortran writes the width
of the ramp as the literal `0.2`, which is taken here as `ℂᵍ₄ - ℂᵍ₃` so that the ramp stays
continuous when the two are changed. `ℂᵍ₃ < ℂᵍ₄` is required. The fields after the coefficients
are tolerances and bounds, not empirical coefficients.

$(TYPEDFIELDS)
"""
struct DoubleGaussianPDF{FT} <: AbstractSHOCComponent
    "Within-plume fraction of the vertical velocity variance (literal `0.4`)"
    ℂᵍ₁ :: FT
    "Skewness of total water relative to that of vertical velocity (literal `1.2`)"
    ℂᵍ₂ :: FT
    "Plume separation below which total water is unskewed (literal `0.2`)"
    ℂᵍ₃ :: FT
    "Plume separation above which total water is fully skewed (literal `0.4`)"
    ℂᵍ₄ :: FT
    "Standard deviation of θˡ below which it is uniform, K"
    thl_tol :: FT
    "Standard deviation of qᵗ below which it is uniform, kg kg⁻¹"
    rt_tol :: FT
    "Standard deviation of vertical velocity below which the air is calm, m s⁻¹"
    w_tol :: FT
    "Lowest liquid water temperature of a plume, K"
    Tl_min :: FT
    "Lowest fraction of the first plume (literal `0.01`)"
    plume_fraction_min :: FT
    "Highest fraction of the first plume (literal `0.99`)"
    plume_fraction_max :: FT
    "Highest variance of a scalar within a plume over its variance (literal `100`)"
    plume_variance_ratio_max :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `DoubleGaussianPDF{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
DoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(DoubleGaussianPDF, FT,
                            (ℂᵍ₁ = 0.4, ℂᵍ₂ = 1.2, ℂᵍ₃ = 0.2, ℂᵍ₄ = 0.4, thl_tol = 1e-2, rt_tol = 1e-4, w_tol = 2e-2, Tl_min = 100,
                             plume_fraction_min = 0.01, plume_fraction_max = 0.99, plume_variance_ratio_max = 100), kw)

Adapt.@adapt_structure DoubleGaussianPDF

"""
$(TYPEDEF)

The double-Gaussian PDF of [`DoubleGaussianPDF`](@ref) (the updraft component of weight `a↑` and the
downdraft component of weight `a↓ = 1 - a↑`, from the first three moments of vertical velocity)
with the within-component variances of θˡⁱ and qᵗ shared equally between the components, as for
vertical velocity:

    σᵢ² = (1 - r² / (1 - ℂᵍ₁)) φ′²,    r = w′φ′ / √(w′² φ′²),

so that the components' means carry the flux and the rest of the variance is within them. The
scalar skewness follows (it is not imposed). SHOC's rule instead matches an imposed scalar
skewness (zero for θˡ, a ramp of `ℂᵍ₂ Sk_w` for qᵗ), which for `|Sk_w| ≳ 1.25` asks for a negative
variance in the narrow component, clipped to zero: a component of no width, whose cloud is all
or nothing. The equal share needs `r² ≤ 1 - ℂᵍ₁`; beyond it the within-component variance is zero
(the flux-correlation bound `ℂʳ` of [`SubgridDistribution`](@ref) keeps `r` below it). The
within-component θˡⁱ–qᵗ correlation is that of the covariance left after the components' means.

$(TYPEDFIELDS)
"""
struct EqualShareDoubleGaussianPDF{FT} <: AbstractSHOCComponent
    "Within-component fraction of the vertical velocity variance"
    ℂᵍ₁ :: FT
    "Standard deviation of θˡ below which it is uniform, K"
    thl_tol :: FT
    "Standard deviation of qᵗ below which it is uniform, kg kg⁻¹"
    rt_tol :: FT
    "Standard deviation of vertical velocity below which the air is calm, m s⁻¹"
    w_tol :: FT
    "Lowest liquid water temperature of a component, K"
    Tl_min :: FT
    "Lowest weight of the updraft component"
    plume_fraction_min :: FT
    "Highest weight of the updraft component"
    plume_fraction_max :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `EqualShareDoubleGaussianPDF{FT}` with the coefficients and tolerances of
[`DoubleGaussianPDF`](@ref). Every field is a keyword argument.
"""
EqualShareDoubleGaussianPDF(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(EqualShareDoubleGaussianPDF, FT,
                            (ℂᵍ₁ = 0.4, thl_tol = 1e-2, rt_tol = 1e-4, w_tol = 2e-2, Tl_min = 100,
                             plume_fraction_min = 0.01, plume_fraction_max = 0.99), kw)

Adapt.@adapt_structure EqualShareDoubleGaussianPDF

const ComponentDoubleGaussianPDF = Union{DoubleGaussianPDF, EqualShareDoubleGaussianPDF}

# Saturation vapor pressure over liquid of Murphy and Koop (2005) (wv_sat_scream.F90, MurphyKoop_svp)
@inline function murphy_koop_liquid_svp(t::FT) where FT
    logt = log(t)
    tmp = convert(FT, 54.842763) - (convert(FT, 6763.22) / t) - (convert(FT, 4.210) * logt) + (convert(FT, 0.000367) * t) +
          (tanh(convert(FT, 0.0415) * (t - convert(FT, 218.8))) * (convert(FT, 53.878) - (convert(FT, 1331.22) / t) -
          (convert(FT, 9.44523) * logt) + convert(FT, 0.014025) * t))
    return exp(tmp)
end

# shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters share one form: the plume
# means and variances of a scalar from its variance `sec`, its flux correlation with w, and the
# skewness `skew` of the scalar (identically zero for thetal, where `dothetal_skew = false`).
@inline function plume_variance(cp, first_plume, x1_1, x1_2, a, skew, sec, ratio_max)
    one_minus = 1 - a * (x1_1 * x1_1) - (1 - a) * (x1_2 * x1_2)
    third = skew - a * (x1_1 * x1_1 * x1_1) - (1 - a) * (x1_2 * x1_2 * x1_2)
    numerator = ifelse(first_plume, 3 * x1_2 * one_minus - third, -3 * x1_1 * one_minus + third)
    denominator = ifelse(first_plume, 3 * a * (x1_2 - x1_1), 3 * (1 - a) * (x1_2 - x1_1))
    # Where the plume means coincide the Fortran forms 0/0 and MAX(0, NaN) drops it; that is zero.
    degenerate = denominator == 0
    ratio = ifelse(degenerate, zero(numerator), numerator / ifelse(degenerate, one(denominator), denominator))
    return clip_between(cp, ratio, 0, ratio_max, 1) * sec
end

@inline function scalar_plume_parameters(cp, flux, sqrtw2, sqrt_sec, sec, tol, x_first, w1_1, w1_2, Skew_w, a, skewed, p::DoubleGaussianPDF)
    FT = typeof(sec)
    corrtest = clip_between(cp, flux / (sqrtw2 * sqrt_sec), -1, 1, 1)

    uniform = (sec <= tol * tol) | (abs(w1_2 - w1_1) <= 0)

    # tilde plume means
    x1_1 = (-1 * corrtest) / ifelse(uniform, one(FT), w1_2)
    x1_2 = (-1 * corrtest) / ifelse(uniform, one(FT), w1_1)

    tsign = abs(x1_2 - x1_1)
    skew_qw = ifelse(tsign > p.ℂᵍ₄, p.ℂᵍ₂ * Skew_w,
                     ifelse(tsign <= p.ℂᵍ₃, zero(FT),
                            ((p.ℂᵍ₂ * Skew_w) / (p.ℂᵍ₄ - p.ℂᵍ₃)) * (tsign - p.ℂᵍ₃)))
    skew = ifelse(skewed, skew_qw, zero(FT))

    x2_1 = plume_variance(cp, true, x1_1, x1_2, a, skew, sec, p.plume_variance_ratio_max)
    x2_2 = plume_variance(cp, false, x1_1, x1_2, a, skew, sec, p.plume_variance_ratio_max)

    x1_1 = x1_1 * sqrt_sec + x_first
    x1_2 = x1_2 * sqrt_sec + x_first

    x1_1 = ifelse(uniform, x_first, x1_1)
    x1_2 = ifelse(uniform, x_first, x1_2)
    x2_1 = ifelse(uniform, zero(FT), x2_1)
    x2_2 = ifelse(uniform, zero(FT), x2_2)

    return x1_1, x1_2, x2_1, x2_2, zero_safe_sqrt(x2_1), zero_safe_sqrt(x2_2)
end

# The equal share of the within-component variance (EqualShareDoubleGaussianPDF)
@inline function scalar_plume_parameters(cp, flux, sqrtw2, sqrt_sec, sec, tol, x_first, w1_1, w1_2, Skew_w, a, skewed,
                                         p::EqualShareDoubleGaussianPDF)
    FT = typeof(sec)
    r = clip_between(cp, flux / (sqrtw2 * sqrt_sec), -1, 1, 1)
    uniform = (sec <= tol * tol) | (abs(w1_2 - w1_1) <= 0)
    x1_1 = (-1 * r) / ifelse(uniform, one(FT), w1_2)
    x1_2 = (-1 * r) / ifelse(uniform, one(FT), w1_1)
    x2 = max(0, 1 - r * r / (1 - p.ℂᵍ₁)) * sec
    x1_1 = ifelse(uniform, x_first, x1_1 * sqrt_sec + x_first)
    x1_2 = ifelse(uniform, x_first, x1_2 * sqrt_sec + x_first)
    x2 = ifelse(uniform, zero(FT), x2)
    σ = zero_safe_sqrt(x2)
    return x1_1, x1_2, x2, x2, σ, σ
end

# shoc_assumed_pdf_compute_qs for one plume, with the fixed constants of shoc.F90
@inline plume_saturation(Tl, pval, p) =
    plume_saturation(FixedConstantsLiquidSaturation(), nothing, p, Tl, zero(Tl), pval)[1:2]

# shoc_assumed_pdf_compute_s, with `cp_over_L` and `Π` of the saturation component
@inline function plume_condensate(qw1, qs1, beta, cp_over_L, Π, thl2, qw2, sqrtthl2, sqrtqw2, r_qwthl)
    FT = typeof(qw1)
    sqrt2 = sqrt(convert(FT, 2))
    sqrt2pi = sqrt(2 * convert(FT, 3.14159265358979323))

    s = qw1 - qs1 * ((1 + beta * qw1) / (1 + beta * qs1))
    opbq = 1 + beta * qs1
    cthl = ((1 + beta * qw1) / (opbq * opbq)) * cp_over_L * beta * qs1 * Π
    cqt = 1 / (1 + beta * qs1)
    tmp_val = max(0, (cthl * cthl) * thl2 + (cqt * cqt) * qw2 - 2 * cthl * sqrtthl2 * cqt * sqrtqw2 * r_qwthl)
    std_s = zero_safe_sqrt(tmp_val)

    resolved = std_s > sqrt(floatmin(FT)) * 100
    std_safe = ifelse(resolved, std_s, one(FT))

    # A PDF of finite width
    C_wide = convert(FT, 0.5) * (1 + erf(s / (sqrt2 * std_safe)))
    sos = s / std_safe
    qn_wide = ifelse(C_wide != 0, s * C_wide + (std_safe / sqrt2pi) * exp(convert(FT, -0.5) * (sos * sos)), zero(FT))

    # A PDF of no width: all or nothing
    C_narrow = ifelse(s > 0, one(FT), zero(FT))
    qn_narrow = ifelse(s > 0, s, zero(FT))

    C = ifelse(resolved, C_wide, C_narrow)
    qn = ifelse(resolved, qn_wide, qn_narrow)

    # Prevent empty clouds, or the rare occurrence of negative cloud liquid
    empty = qn <= 0
    C = ifelse(empty, zero(FT), C)
    qn = ifelse(empty, zero(FT), qn)

    return s, std_s, qn, C
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The assumed PDF of level `k` (Analytic Double Gaussian 1 of Larson et al. 2002): return the
subgrid cloud fraction, liquid water, liquid water variance, liquid water flux and buoyancy flux.
The plumes saturate by the saturation component `sat` (see [`FixedConstantsLiquidSaturation`](@ref)
and [`ModelThermodynamicsSaturation`](@ref)), with the `thermodynamics` of the model.
"""
@inline function assumed_pdf(pdf::ComponentDoubleGaussianPDF, sat, thermodynamics, cp, c, zi_u, zi_l, zt_k, pval, thl_first, w_first,
                             qw_first, wsec, moments_l, moments_u)
    p = c # the physical constants, under the name they have in the expressions below
    FT = typeof(thl_first)
    largeneg = convert(FT, -99999999.99)
    thl_tol = pdf.thl_tol
    rt_tol = pdf.rt_tol
    w_tol_sqd = pdf.w_tol * pdf.w_tol
    Tl_min = pdf.Tl_min

    # Interpolate the interface moments (w3, thl_sec, wthl_sec, qwthl_sec, wqw_sec, qw_sec of the
    # interfaces below, `moments_l`, and above, `moments_u`, the level) to the level
    interpolate(n, minthresh) = hard_floor(linear_interp(zi_u, zi_l, moments_u[n], moments_l[n], zt_k), minthresh)
    w3var = interpolate(1, largeneg)
    thlsec = interpolate(2, zero(FT))
    wthlsec = interpolate(3, largeneg)
    qwthlsec = interpolate(4, largeneg)
    wqwsec = interpolate(5, largeneg)
    qwsec = interpolate(6, zero(FT))

    sqrtw2 = zero_safe_sqrt(wsec)
    sqrtthl = clip_max(cp, thl_tol, zero_safe_sqrt(thlsec), thl_tol)
    sqrtqt = clip_max(cp, rt_tol, zero_safe_sqrt(qwsec), rt_tol)

    # shoc_assumed_pdf_vv_parameters
    calm = wsec <= w_tol_sqd
    wsec_safe = ifelse(calm, one(FT), wsec)
    Skew_w = w3var / sqrt(wsec_safe * wsec_safe * wsec_safe)
    w2_1 = pdf.ℂᵍ₁
    omw = 1 - w2_1
    a = clip_between(cp, convert(FT, 0.5) * (1 - Skew_w * sqrt(1 / (4 * (omw * omw * omw) + Skew_w * Skew_w))),
                     pdf.plume_fraction_min, pdf.plume_fraction_max, 1)
    sqrtw2t = sqrt(1 - w2_1)
    w1_1 = sqrt((1 - a) / a) * sqrtw2t
    w1_2 = -1 * sqrt(a / (1 - a)) * sqrtw2t

    Skew_w = ifelse(calm, zero(FT), Skew_w)
    a = ifelse(calm, convert(FT, 0.5), a)
    w1_1 = ifelse(calm, w_first, w1_1)
    w1_2 = ifelse(calm, w_first, w1_2)

    # shoc_assumed_pdf_thl_parameters and shoc_assumed_pdf_qw_parameters
    thl1_1, thl1_2, thl2_1, thl2_2, sqrtthl2_1, sqrtthl2_2 =
        scalar_plume_parameters(cp, wthlsec, sqrtw2, sqrtthl, thlsec, thl_tol, thl_first, w1_1, w1_2, Skew_w, a, false, pdf)

    qw1_1, qw1_2, qw2_1, qw2_2, sqrtqw2_1, sqrtqw2_2 =
        scalar_plume_parameters(cp, wqwsec, sqrtw2, sqrtqt, qwsec, rt_tol, qw_first, w1_1, w1_2, Skew_w, a, true, pdf)

    # shoc_assumed_pdf_tilde_to_real
    w1_1 = w1_1 * sqrtw2 + w_first
    w1_2 = w1_2 * sqrtw2 + w_first

    # shoc_assumed_pdf_inplume_correlations
    testvar = a * sqrtqw2_1 * sqrtthl2_1 + (1 - a) * sqrtqw2_2 * sqrtthl2_2
    uncorrelated = testvar == 0
    r_qwthl_1 = clip_between(cp, (qwthlsec - a * (qw1_1 - qw_first) * (thl1_1 - thl_first)
                                  - (1 - a) * (qw1_2 - qw_first) * (thl1_2 - thl_first)) /
                                 ifelse(uncorrelated, one(FT), testvar), -1, 1, 1)
    r_qwthl_1 = ifelse(uncorrelated, zero(FT), r_qwthl_1)

    # shoc_assumed_pdf_compute_temperature
    Tl1_1 = plume_liquid_water_temperature(sat, thermodynamics, p, thl1_1, qw1_1, pval)
    Tl1_2 = plume_liquid_water_temperature(sat, thermodynamics, p, thl1_2, qw1_2, pval)
    Tl1_1 = ifelse(Tl1_1 <= Tl_min, Tl_min, Tl1_1)
    Tl1_2 = ifelse(Tl1_2 <= Tl_min, Tl_min, Tl1_2)

    # shoc_assumed_pdf_compute_qs
    qs1, beta1, cp_over_L1, Π1 = plume_saturation(sat, thermodynamics, p, Tl1_1, qw1_1, pval)
    qs2, beta2, cp_over_L2, Π2 = plume_saturation(sat, thermodynamics, p, Tl1_2, qw1_2, pval)

    # shoc_assumed_pdf_compute_s
    s1, std_s1, qn1, C1 = plume_condensate(qw1_1, qs1, beta1, cp_over_L1, Π1, thl2_1, qw2_1, sqrtthl2_1, sqrtqw2_1, r_qwthl_1)
    s2, std_s2, qn2, C2 = plume_condensate(qw1_2, qs2, beta2, cp_over_L2, Π2, thl2_2, qw2_2, sqrtthl2_2, sqrtqw2_2, r_qwthl_1)

    ql1 = clip_min(cp, qn1, qw1_1, rt_tol)
    ql2 = clip_min(cp, qn2, qw1_2, rt_tol)

    shoc_cldfrac = min(1, a * C1 + (1 - a) * C2)
    shoc_ql = max(0, a * ql1 + (1 - a) * ql2)

    shoc_ql2 = a * (s1 * ql1 + C1 * (std_s1 * std_s1)) + (1 - a) * (s2 * ql2 + C2 * (std_s2 * std_s2)) - (shoc_ql * shoc_ql)
    shoc_ql2 = max(0, shoc_ql2)

    wqls = a * ((w1_1 - w_first) * ql1) + (1 - a) * ((w1_2 - w_first) * ql2)

    A_q, A_c = buoyancy_flux_coefficients(sat, thermodynamics, p, pval, thl_first, qw_first, shoc_ql)
    wthv_sec = wthlsec + A_q * wqwsec + A_c * wqls

    return shoc_cldfrac, shoc_ql, shoc_ql2, wqls, wthv_sec
end

# The saturation of shoc.F90, when none is given
@inline assumed_pdf(pdf, cp, c, zi_u, zi_l, zt_k, pval, thl_first, w_first, qw_first, wsec, moments_l, moments_u) =
    assumed_pdf(pdf, FixedConstantsLiquidSaturation(), nothing, cp, c, zi_u, zi_l, zt_k, pval, thl_first, w_first, qw_first,
                wsec, moments_l, moments_u)
