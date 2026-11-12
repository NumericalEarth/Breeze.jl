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
| `ℂᶠ`  | `a_const`    | 1.8 | surface variances and covariance, `θˡ'² = ℂᶠ₂ ℂᶠ (w'θˡ' / u_f)²` at the lowest interface |
| `ℂᶠ₂` | literal `0.4` | 0.4 | surface variances of θˡ and qᵗ, see `ℂᶠ` |
| `ℂᶠ₃` | literal `0.2` | 0.2 | surface covariance of θˡ and qᵗ, `ℂᶠ₃ ℂᶠ (w'θˡ' / u_f) (w'qᵗ' / u_f)` |
| `ℂᶠ₄` | literal `0.3` | 0.3 | weight of the convective velocity in the surface velocity scale, `u_f² = u★² + ℂᶠ₄ w★²` |
| `ℂᶠ₅` | `z_const`     | 1   | depth of the convective velocity scale `w★³ = (g / θ₀) w'θˡ' ℂᶠ₅`, **in meters** |

`ℂᶠ₅` is dimensional. `ℂᶠ` multiplies `ℂᶠ₂` and `ℂᶠ₃` and nothing else, and `ℂᶠ₅` enters only
with `ℂᶠ₄`, as `ℂᶠ₄ ℂᶠ₅^{2/3}`. The fields after the coefficients are safeguards.

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
    "Surface variance factor of θˡ and qᵗ (literal `0.4`)"
    ℂᶠ₂ :: FT
    "Surface covariance factor of θˡ and qᵗ (literal `0.2`)"
    ℂᶠ₃ :: FT
    "Weight of the squared convective velocity in the surface velocity scale (literal `0.3`)"
    ℂᶠ₄ :: FT
    "Depth of the convective velocity scale at the surface, m (`z_const`); dimensional"
    ℂᶠ₅ :: FT
    "Minimum surface velocity scale `u_f`, m s⁻¹ (`ufmin`)"
    ufmin :: FT
    "Minimum friction velocity in the surface flux of turbulent kinetic energy `w'e'`, m s⁻¹ (literal `0.01`)"
    wtke_ustar_min :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `DowngradientSecondMoments{FT}` with the values of E3SM `shoc.F90`. Every field is a
keyword argument.
"""
DowngradientSecondMoments(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(DowngradientSecondMoments, FT, (ℂᵛ₁ = 1, ℂᵛ₂ = 1, ℂᵛ₃ = 1, ℂʷ₁ = 1, ℂᶠ = 1.8, ℂᶠ₂ = 0.4, ℂᶠ₃ = 0.2, ℂᶠ₄ = 0.3, ℂᶠ₅ = 1,
                             ufmin = 0.01, wtke_ustar_min = 0.01), kw)

Adapt.@adapt_structure DowngradientSecondMoments

# diag_second_moments_srf, diag_second_moments_lbycond
@inline function surface_second_moments(sm::DowngradientSecondMoments, cp, c, wthl_sfc, wqw_sfc, uw_sfc, vw_sfc)
    FT = typeof(wthl_sfc)
    g = c.ggr

    ustar2 = zero_safe_sqrt(uw_sfc * uw_sfc + vw_sfc * vw_sfc)
    heated = wthl_sfc > 0
    wstar_cubed = ifelse(heated, 1 / c.basetemp * g * wthl_sfc * sm.ℂᶠ₅, one(FT))
    wstar = ifelse(heated, wstar_cubed^(convert(FT, 1) / 3), zero(FT))

    a_const = sm.ℂᶠ
    uf = zero_safe_sqrt(ustar2 + sm.ℂᶠ₄ * wstar * wstar)
    uf = clip_max(cp, sm.ufmin, uf, sm.ufmin)
    maxval = clip_max(cp, zero_safe_sqrt(ustar2), sm.wtke_ustar_min, sm.wtke_ustar_min)

    thl_sec = sm.ℂᶠ₂ * a_const * ((wthl_sfc / uf) * (wthl_sfc / uf))
    qw_sec = sm.ℂᶠ₂ * a_const * ((wqw_sfc / uf) * (wqw_sfc / uf))
    qwthl_sec = sm.ℂᶠ₃ * a_const * (wthl_sfc / uf) * (wqw_sfc / uf)
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
