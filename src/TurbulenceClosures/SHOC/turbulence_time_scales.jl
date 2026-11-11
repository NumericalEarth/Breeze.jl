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

#####
##### Stability functions of the local Richardson number
#####

"""
$(TYPEDEF)

Eddy diffusivities and dissipation from stability functions of the local gradient Richardson
number, as in [`TKEBasedTurbulenceClosure`](@ref) and with its functions (by default
[`RationalStabilityFunction`](@ref), reused as it is):

    Kᶜ = Sᶜ(Ri) ℓ √e,    Kᵘ = Sᵘ(Ri) ℓ √e,    ε = Sᴰ(Ri) e^{3/2} / ℓ,    Ri = N² / (S² + S²ᵐⁱⁿ)

in place of the damping by the stability of the whole column of
[`ColumnStabilityDamping`](@ref), which is one number per column and jumps to zero where the
air is unstable. There is no cold-surface fallback.

The second and third moments need a return-to-isotropy time scale, which is taken as `ℂᵗ`
eddy turnover times, limited by the buoyancy period in stable air:

    τ = τ₀ / √(1 + τ₀² N²⁺),    τ₀ = ℂᵗ ℓ / √e

With `ℂᵗ = 1`, `τ₀` is that of `ColumnStabilityDamping` with its published dissipation
coefficient for the same length (`2 e / ε` is `1.01 ℓ / √e` there). It is not `2 e / ε` of this
component: the dissipation coefficient of the rational stability functions is some thirty
times smaller, and the time scale it implies puts `τ² N²` far beyond the poles of the
diagnostic third moment.

The default coefficients of `RationalStabilityFunction` were calibrated for the TKE closure
with `GradientLimitedMixingLength` and a shear production `Kᵘ S²`, so they are a starting point
for a calibration of SHOC rather than a tuned set, and they go with a shear production
coefficient `ℂᴾ = 1` of the TKE update.

$(TYPEDFIELDS)
"""
struct RichardsonStabilityFunctions{SF, FT} <: AbstractSHOCComponent
    "The stability functions of the Richardson number"
    functions :: SF
    "Return-to-isotropy time scale of the moments in eddy turnover times `ℓ / √e`"
    ℂᵗ :: FT
    "Shear floor `S²ᵐⁱⁿ` in `Ri = N² / (S² + S²ᵐⁱⁿ)`, s⁻²"
    minimum_shear_squared :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `RichardsonStabilityFunctions` with the stability `functions`, by default the
`RationalStabilityFunction` with its published coefficients.
"""
RichardsonStabilityFunctions(FT::DataType = Oceananigans.defaults.FloatType;
                             functions = RationalStabilityFunction(), ℂᵗ = 1, minimum_shear_squared = 1e-10) =
    RichardsonStabilityFunctions(convert_eltype(FT, functions), convert(FT, ℂᵗ), convert(FT, minimum_shear_squared))

Adapt.@adapt_structure RichardsonStabilityFunctions

Base.summary(ts::RichardsonStabilityFunctions) = string("RichardsonStabilityFunctions with ", summary(ts.functions))

# The Richardson number of a level: the shear term of the TKE update is its ℂᴾ times S²
@inline level_richardson_number(ts::RichardsonStabilityFunctions, brunt_k, shear_term, tke_update) =
    richardson_number(brunt_k, shear_term / tke_update.ℂᴾ, ts.minimum_shear_squared)

@inline time_scale_column(::RichardsonStabilityFunctions, cp, c, brunt_int) = nothing

@inline dissipation_coefficient(ts::RichardsonStabilityFunctions, column, brunt_k, shear_term, tke_update) =
    dissipation_stability_function(ts.functions, level_richardson_number(ts, brunt_k, shear_term, tke_update))

@inline function time_scale_and_diffusivities(ts::RichardsonStabilityFunctions, cp, c, column, tke_k, a_diss, mix,
                                              brunt_k, shear_term, tke_update, runaway)
    FT = typeof(tke_k)
    brunt_scale = convert(FT, 1e-6) # scale of the squared buoyancy frequency for the clipping policy [s⁻²]

    Ri = level_richardson_number(ts, brunt_k, shear_term, tke_update)
    turbulent_velocity = sqrt(clip_max(cp, c.mintke, tke_k, c.mintke))

    tkh_k = tracer_stability_function(ts.functions, Ri) * mix * turbulent_velocity
    tk_k = momentum_stability_function(ts.functions, Ri) * mix * turbulent_velocity

    iso_scale = ts.ℂᵗ * mix / turbulent_velocity
    iso = iso_scale / sqrt(1 + iso_scale * iso_scale * clip_negative(cp, brunt_k, brunt_scale))

    return iso, tkh_k, tk_k
end
