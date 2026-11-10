#####
##### Slot: second moments
#####
##### A second-moment component implements
#####
#####     surface_second_moments(sm, cp, c, wthl_sfc, wqw_sfc, uw_sfc, vw_sfc) -> (thl_sec, qw_sec, qwthl_sec, wtke_sec)
#####     vertical_velocity_variance(sm, e) -> w_sec
#####     interface_second_moments(sm, τ_zi, Kᶜ_zi, Kᵘ_zi, grid_dz, Δthetal, Δqw, Δtke, Δu, Δv)
#####         -> (thl_sec, qw_sec, qwthl_sec, wthl_sec, wqw_sec, wtke_sec, uw_sec, vw_sec)
#####
##### the moments at the surface, on a level, and at an interior interface from the time scale and
##### the diffusivities interpolated to the interface, the inverse distance `grid_dz` between the
##### adjacent levels and the differences across the interface.
#####

"""
$(TYPEDEF)

The second moments of E3SM `shoc.F90` (`diag_second_shoc_moments`), diagnosed down the gradient:
the fluxes are `-K ∂z` of the mean, the variances and the covariance of θˡ and qᵗ are
`ℂᵛ τ Kᶜ ∂z a ∂z b`, and the vertical velocity variance is `ℂʷ₁ (2/3) e`.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵛ₁` | `thl2tune`   | 1   | scales the variance of liquid water potential temperature |
| `ℂᵛ₂` | `qw2tune`    | 1   | scales the variance of total water |
| `ℂᵛ₃` | `qwthl2tune` | 1   | scales the covariance of total water and liquid water potential temperature |
| `ℂʷ₁` | `w2tune`     | 1   | scales the vertical velocity variance, `w'w' = ℂʷ₁ (2/3) e` |
| `ℂᶠ`  | `a_const`    | 1.8 | surface variances and covariance, `θˡ'² = 0.4 ℂᶠ (w'θˡ' / u_f)²` at the lowest interface |

$(TYPEDFIELDS)
"""
struct DowngradientSecondMoments{FT} <: AbstractSHOCComponent
    "Temperature variance factor (`thl2tune`)"
    ℂᵛ₁ :: FT
    "Moisture variance factor (`qw2tune`)"
    ℂᵛ₂ :: FT
    "Temperature-moisture covariance factor (`qwthl2tune`)"
    ℂᵛ₃ :: FT
    "Vertical velocity variance factor (`w2tune`)"
    ℂʷ₁ :: FT
    "Surface variance coefficient (`a_const`)"
    ℂᶠ :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `DowngradientSecondMoments{FT}` with the values of E3SM `shoc.F90`. Every field is a
keyword argument.
"""
DowngradientSecondMoments(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(DowngradientSecondMoments, FT, (ℂᵛ₁ = 1, ℂᵛ₂ = 1, ℂᵛ₃ = 1, ℂʷ₁ = 1, ℂᶠ = 1.8), kw)

Adapt.@adapt_structure DowngradientSecondMoments

# diag_second_moments_srf, diag_second_moments_lbycond
@inline function surface_second_moments(sm::DowngradientSecondMoments, cp, c, wthl_sfc, wqw_sfc, uw_sfc, vw_sfc)
    FT = typeof(wthl_sfc)
    g = c.ggr

    ustar2 = zero_safe_sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc)
    heated = wthl_sfc > 0
    wstar_cubed = ifelse(heated, 1 / c.basetemp * g * wthl_sfc * 1, one(FT))
    wstar = ifelse(heated, wstar_cubed^(convert(FT, 1) / 3), zero(FT))

    a_const = sm.ℂᶠ
    uf = zero_safe_sqrt(ustar2 + convert(FT, 0.3) * wstar * wstar)
    uf = clip_max(cp, convert(FT, 0.01), uf, convert(FT, 0.01))
    maxval = clip_max(cp, zero_safe_sqrt(ustar2), convert(FT, 0.01), convert(FT, 0.01))

    thl_sec = convert(FT, 0.4) * a_const * ((wthl_sfc / uf) * (wthl_sfc / uf))
    qw_sec = convert(FT, 0.4) * a_const * ((wqw_sfc / uf) * (wqw_sfc / uf))
    qwthl_sec = convert(FT, 0.2) * a_const * (wthl_sfc / uf) * (wqw_sfc / uf)
    wtke_sec = maxval * maxval * maxval

    return thl_sec, qw_sec, qwthl_sec, wtke_sec
end

@inline vertical_velocity_variance(sm::DowngradientSecondMoments, tke_k) =
    (sm.ℂʷ₁ * (convert(typeof(tke_k), 2) / 3)) * tke_k

# calc_shoc_varorcovar and calc_shoc_vertflux at an interior interface
@inline function interface_second_moments(sm::DowngradientSecondMoments, isotropy_zi, tkh_zi, tk_zi, grid_dz,
                                          Δthetal, Δqw, Δtke, Δu, Δv)
    grid_dz2 = grid_dz * grid_dz
    isotropic_diffusivity = isotropy_zi * tkh_zi

    thl_sec = sm.ℂᵛ₁ * isotropic_diffusivity * grid_dz2 * Δthetal * Δthetal
    qw_sec = sm.ℂᵛ₂ * isotropic_diffusivity * grid_dz2 * Δqw * Δqw
    qwthl_sec = sm.ℂᵛ₃ * isotropic_diffusivity * grid_dz2 * Δthetal * Δqw

    wthl_sec = -1 * tkh_zi * grid_dz * Δthetal
    wqw_sec = -1 * tkh_zi * grid_dz * Δqw
    wtke_sec = -1 * tkh_zi * grid_dz * Δtke
    uw_sec = -1 * tk_zi * grid_dz * Δu
    vw_sec = -1 * tk_zi * grid_dz * Δv

    return thl_sec, qw_sec, qwthl_sec, wthl_sec, wqw_sec, wtke_sec, uw_sec, vw_sec
end
