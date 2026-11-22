#####
##### Second moments
#####
##### A second-moment component implements
#####
#####     vertical_velocity_variance(sm, e) -> w′²
#####     interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, Δθˡⁱ, Δqᵗ, Δe, Δu, Δv)
#####         -> (θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′, w′e′, w′u′, w′v′)
#####
##### the vertical velocity variance at a center, and the moments at an interior face from the time scale and
##### the diffusivities there, the inverse distance between the adjacent centers and the differences across
##### the face.
#####

"""
$(TYPEDEF)

The second moments of E3SM `shoc.F90` (`diag_second_shoc_moments`), diagnosed down the gradient: the fluxes
are `-K ∂z` of the mean, the variances and the covariance of θˡⁱ and qᵗ are `ℂᵛ τ Kᶜ ∂z a ∂z b`, and the
vertical velocity variance is `ℂʷ₁ (2/3) e`. The surface values of `shoc.F90` (`diag_second_moments_srf`) are
not used: the surface face takes the moments of the face above.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵛ₁` | `thl2tune`   | 1   | scales the variance of the liquid-ice potential temperature |
| `ℂᵛ₂` | `qw2tune`    | 1   | scales the variance of total water |
| `ℂᵛ₃` | `qwthl2tune` | 1   | scales the covariance of total water and liquid-ice potential temperature |
| `ℂʷ₁` | `w2tune`     | 1   | scales the vertical velocity variance, `w′² = ℂʷ₁ (2/3) e` |

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
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `DowngradientSecondMoments{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword argument.
"""
DowngradientSecondMoments(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(DowngradientSecondMoments, FT, (ℂᵛ₁ = 1, ℂᵛ₂ = 1, ℂᵛ₃ = 1, ℂʷ₁ = 1), kw)

Adapt.@adapt_structure DowngradientSecondMoments

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
