#####
##### Slot: stability and turbulence time scale
#####
##### A time scale component implements
#####
#####     time_scale_column(ts, cp, c, brunt_int)                -> column
#####     dissipation_coefficient(ts, column, N², shear_term, tke_update) -> Sᴰ, with ε = Sᴰ e^{3/2} / ℓ
#####     time_scale_and_diffusivities(ts, cp, c, column, e, ε, ℓ, N², shear_term, tke_update, runaway) -> (τ, Kᶜ, Kᵘ)
#####
##### `brunt_int` is the stability integrated over the lower troposphere, `shear_term` the shear
##### production term of the TKE update (the squared shear times its `ℂᴾ`), `e` the turbulent
##### kinetic energy after the TKE update and `ε` the dissipation the update used. `τ` is the
##### return-to-isotropy time scale that the second and third moments use.
#####

"""
$(TYPEDEF)

The return-to-isotropy time scale and eddy diffusivities of E3SM `shoc.F90` (`isotropic_ts`,
`eddy_diffusivities`): `τ₀ = 2 e / ε` damped in stable air by the stability of the whole column,

    τ = min(maxiso, τ₀ / (1 + λ N²⁺ τ₀²)),    Kᶜ = ℂᶜ τ e,    Kᵘ = ℂᵘ τ e

where `λ = ℂᵝ₁ + (∫N² dz / g - ℂᵝ₄) ℂᵝ₃`, kept between `ℂᵝ₁` and `ℂᵝ₂`, is one number per column,
and the dissipation is `ε = Sᴰ e^{3/2} / ℓ` with the constant `Sᴰ = ℂᴰ₁³ / ℂᴰ₂⁴`. Over a surface
colder than 182 K the diffusivities fall back to `ℂᶜₛ ℓ² |∂z u|` and `ℂᵘₛ ℓ² |∂z u|` up to 200 m
above the boundary layer top.

| coefficient | `shoc.F90` | default | meaning |
|:------------|:-----------|:--------|:--------|
| `ℂᵘ`  | `Ckm`            | 0.1    | eddy diffusivity of momentum, `Kᵘ = ℂᵘ τ e` |
| `ℂᶜ`  | `Ckh`            | 0.1    | eddy diffusivity of heat and scalars, `Kᶜ = ℂᶜ τ e` |
| `ℂᵘₛ` | `Ckm_s`          | 0.1    | momentum diffusivity of the cold-surface fallback |
| `ℂᶜₛ` | `Ckh_s`          | 0.1    | heat diffusivity of the cold-surface fallback |
| `ℂᵝ₁` | `lambda_low`     | 0.001  | lowest value of the stability damping `λ` |
| `ℂᵝ₂` | `lambda_high`    | 0.08   | highest value of `λ` |
| `ℂᵝ₃` | `lambda_slope`   | 2.65   | slope of `λ` in the integrated column stability |
| `ℂᵝ₄` | `lambda_thresh`  | 0.02   | integrated column stability at which `λ` starts to rise |
| `ℂᴰ₁` | `Ck` (`adv_sgs_tke`) | 0.1  | dissipation: `Ce = ℂᴰ₁³ / ℂᴰ₂⁴`, `ε = (0.19 + 0.51) / 0.7 Ce e^{3/2} / ℓ` |
| `ℂᴰ₂` | `Cs` (`adv_sgs_tke`) | 0.15 | dissipation, see `ℂᴰ₁` |

$(TYPEDFIELDS)
"""
struct ColumnStabilityDamping{FT} <: AbstractSHOCComponent
    "Eddy diffusivity coefficient for momentum (`Ckm`)"
    ℂᵘ :: FT
    "Eddy diffusivity coefficient for heat and scalars (`Ckh`)"
    ℂᶜ :: FT
    "Cold-surface diffusivity coefficient for momentum (`Ckm_s`)"
    ℂᵘₛ :: FT
    "Cold-surface diffusivity coefficient for heat (`Ckh_s`)"
    ℂᶜₛ :: FT
    "Lowest stability damping of the return-to-isotropy time scale (`lambda_low`)"
    ℂᵝ₁ :: FT
    "Highest stability damping (`lambda_high`)"
    ℂᵝ₂ :: FT
    "Slope of the stability damping (`lambda_slope`)"
    ℂᵝ₃ :: FT
    "Column stability at which the stability damping rises (`lambda_thresh`)"
    ℂᵝ₄ :: FT
    "Dissipation coefficient (`Ck` of `adv_sgs_tke`)"
    ℂᴰ₁ :: FT
    "Dissipation coefficient (`Cs` of `adv_sgs_tke`)"
    ℂᴰ₂ :: FT
    "Maximum return-to-isotropy time scale, s"
    maxiso :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ColumnStabilityDamping{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
ColumnStabilityDamping(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ColumnStabilityDamping, FT,
                            (ℂᵘ = 0.1, ℂᶜ = 0.1, ℂᵘₛ = 0.1, ℂᶜₛ = 0.1, ℂᵝ₁ = 0.001, ℂᵝ₂ = 0.08, ℂᵝ₃ = 2.65, ℂᵝ₄ = 0.02,
                             ℂᴰ₁ = 0.1, ℂᴰ₂ = 0.15, maxiso = 20000), kw)

Adapt.@adapt_structure ColumnStabilityDamping

# The stability damping λ of the column and the dissipation coefficient
@inline function time_scale_column(ts::ColumnStabilityDamping, cp, c, brunt_int)
    FT = typeof(brunt_int)

    Cs = ts.ℂᴰ₂
    Ck = ts.ℂᴰ₁
    Ce = (Ck * Ck * Ck) / ((Cs * Cs) * (Cs * Cs))
    Ce1 = Ce / convert(FT, 0.7) * convert(FT, 0.19)
    Ce2 = Ce / convert(FT, 0.7) * convert(FT, 0.51)
    Cee = Ce1 + Ce2

    lambda_column = ts.ℂᵝ₁ + ((brunt_int / c.ggr) - ts.ℂᵝ₄) * ts.ℂᵝ₃
    lambda_column = clip_between(cp, lambda_column, ts.ℂᵝ₁, ts.ℂᵝ₂, ts.ℂᵝ₁)

    return (; lambda_column, Cee)
end

@inline dissipation_coefficient(::ColumnStabilityDamping, column, brunt_k, shear_term, tke_update) = column.Cee

@inline function time_scale_and_diffusivities(ts::ColumnStabilityDamping, cp, c, column, tke_k, a_diss, mix,
                                              brunt_k, shear_term, tke_update, runaway)
    FT = typeof(tke_k)
    brunt_scale = convert(FT, 1e-6) # scale of the squared buoyancy frequency for the clipping policy [s⁻²]
    iso_scale_floor = convert(FT, 100) # scale of the time scale for the clipping policy [s]

    # isotropic_ts
    iso_scale = (2 * tke_k) / a_diss
    iso = clip_min(cp, ts.maxiso, iso_scale / (1 + stable_damping(cp, column.lambda_column, brunt_k, brunt_scale) * (iso_scale * iso_scale)), iso_scale_floor)

    # eddy_diffusivities
    sqrt_sterm = zero_safe_sqrt(shear_term)
    tkh_k = ifelse(runaway, ts.ℂᶜₛ * (mix * mix) * sqrt_sterm, ts.ℂᶜ * iso * tke_k)
    tk_k = ifelse(runaway, ts.ℂᵘₛ * (mix * mix) * sqrt_sterm, ts.ℂᵘ * iso * tke_k)

    return iso, tkh_k, tk_k
end
