#####
##### Slot: stability and turbulence time scale
#####
##### A time scale component implements
#####
#####     time_scale_column(ts, cp, c, brunt_int)                -> column
#####     dissipation_coefficient(ts, column, N², shear_term, tke_update) -> Sᴰ, with ε = Sᴰ e^{3/2} / ℓ
#####     cold_surface_fallback(ts, tabs_sfc, z, pblh)           -> runaway
#####     time_scale_and_diffusivities(ts, cp, c, column, e, ε, ℓ, N², shear_term, tke_update, runaway) -> (τ, Kᶜ, Kᵘ)
#####
##### `brunt_int` is the stability integrated over the lower troposphere, `shear_term` the shear
##### production term of the TKE update (the squared shear times its `ℂᴾ`), `e` the turbulent
##### kinetic energy after the TKE update and `ε` the dissipation the update used. `τ` is the
##### return-to-isotropy time scale that the second and third moments use. `runaway` says whether
##### the level takes the fallback diffusivities of a very cold surface (of temperature `tabs_sfc`,
##### under a boundary layer of depth `pblh`); a component without that fallback returns `false`.
#####

@inline cold_surface_fallback(time_scale, tabs_sfc, zt_k, pblh) = false

"""
$(TYPEDEF)

The return-to-isotropy time scale and eddy diffusivities of E3SM `shoc.F90` (`isotropic_ts`,
`eddy_diffusivities`): `τ₀ = 2 e / ε` damped in stable air by the stability of the whole column,

    τ = min(maxiso, τ₀ / (1 + λ N²⁺ τ₀²)),    Kᶜ = ℂᶜ τ e,    Kᵘ = ℂᵘ τ e

where `λ = ℂᵝ₁ + (∫N² dz / g - ℂᵝ₄) ℂᵝ₃`, kept between `ℂᵝ₁` and `ℂᵝ₂`, is one number per column,
and the dissipation is `ε = Sᴰ e^{3/2} / ℓ` with the constant
`Sᴰ = (ℂᴰ₃ + ℂᴰ₄) / ℂᴰ₅ × ℂᴰ₁³ / ℂᴰ₂⁴`. Over a surface colder than `temp_crit` (182 K) the
diffusivities fall back to `ℂᶜₛ ℓ² |∂z u|` and `ℂᵘₛ ℓ² |∂z u|` up to `pbl_trans` (200 m) above
the boundary layer top.

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
| `ℂᴰ₁` | `Ck` (`adv_sgs_tke`) | 0.1  | dissipation: `Ce = ℂᴰ₁³ / ℂᴰ₂⁴`, `ε = (ℂᴰ₃ + ℂᴰ₄) / ℂᴰ₅ Ce e^{3/2} / ℓ` |
| `ℂᴰ₂` | `Cs` (`adv_sgs_tke`) | 0.15 | dissipation, see `ℂᴰ₁` |
| `ℂᴰ₃` | literal `0.19` (`Ce1 = Ce / 0.7 × 0.19`) | 0.19 | dissipation, see `ℂᴰ₁` |
| `ℂᴰ₄` | literal `0.51` (`Ce2 = Ce / 0.7 × 0.51`) | 0.51 | dissipation, see `ℂᴰ₁` |
| `ℂᴰ₅` | literal `0.7`  | 0.7  | dissipation, see `ℂᴰ₁` |

The five `ℂᴰ` enter the closure only through the one product `Sᴰ`, so a calibration can
determine no more than one of them. The fields after them are bounds and switches, all
dimensional; none is an empirical coefficient.

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
    "Dissipation coefficient (literal `0.19` of `Ce1`)"
    ℂᴰ₃ :: FT
    "Dissipation coefficient (literal `0.51` of `Ce2`)"
    ℂᴰ₄ :: FT
    "Dissipation coefficient (literal `0.7` that divides `Ce` in `Ce1` and `Ce2`)"
    ℂᴰ₅ :: FT
    "Maximum return-to-isotropy time scale, s"
    maxiso :: FT
    "Surface temperature below which the diffusivities take the cold-surface fallback, K (`temp_crit`)"
    temp_crit :: FT
    "Depth above the boundary layer top over which the cold-surface fallback applies, m (`pbl_trans`)"
    pbl_trans :: FT
    "Magnitude of the time scale that scales the smoothing of its upper bound, s; `HardClipping` ignores it"
    isotropy_scale :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `ColumnStabilityDamping{FT}` with the values of E3SM `shoc.F90`. Every field is a keyword
argument.
"""
ColumnStabilityDamping(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    component_from_keywords(ColumnStabilityDamping, FT,
                            (ℂᵘ = 0.1, ℂᶜ = 0.1, ℂᵘₛ = 0.1, ℂᶜₛ = 0.1, ℂᵝ₁ = 0.001, ℂᵝ₂ = 0.08, ℂᵝ₃ = 2.65, ℂᵝ₄ = 0.02,
                             ℂᴰ₁ = 0.1, ℂᴰ₂ = 0.15, ℂᴰ₃ = 0.19, ℂᴰ₄ = 0.51, ℂᴰ₅ = 0.7,
                             maxiso = 20000, temp_crit = 182, pbl_trans = 200, isotropy_scale = 100), kw)

Adapt.@adapt_structure ColumnStabilityDamping

# The stability damping λ of the column and the dissipation coefficient
@inline function time_scale_column(ts::ColumnStabilityDamping, cp, c, brunt_int)
    Cs = ts.ℂᴰ₂
    Ck = ts.ℂᴰ₁
    Ce = (Ck * Ck * Ck) / ((Cs * Cs) * (Cs * Cs))
    Ce1 = Ce / ts.ℂᴰ₅ * ts.ℂᴰ₃
    Ce2 = Ce / ts.ℂᴰ₅ * ts.ℂᴰ₄
    Cee = Ce1 + Ce2

    lambda_column = ts.ℂᵝ₁ + ((brunt_int / c.ggr) - ts.ℂᵝ₄) * ts.ℂᵝ₃
    lambda_column = clip_between(cp, lambda_column, ts.ℂᵝ₁, ts.ℂᵝ₂, ts.ℂᵝ₁)

    return (; lambda_column, Cee)
end

@inline dissipation_coefficient(::ColumnStabilityDamping, column, brunt_k, shear_term, tke_update) = column.Cee

# The fallback applies over a surface colder than `temp_crit`, up to `pbl_trans` above the boundary layer top
@inline cold_surface_fallback(ts::ColumnStabilityDamping, tabs_sfc, zt_k, pblh) =
    (tabs_sfc < ts.temp_crit) & (zt_k < pblh + ts.pbl_trans)

@inline function time_scale_and_diffusivities(ts::ColumnStabilityDamping, cp, c, column, tke_k, a_diss, mix,
                                              brunt_k, shear_term, tke_update, runaway)
    brunt_scale = c.brunt_scale # scale of the squared buoyancy frequency for the clipping policy [s⁻²]
    iso_scale_floor = ts.isotropy_scale # scale of the time scale for the clipping policy [s]

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

Only the formulation is shared with the TKE closure. This component holds its own stability
functions, so their coefficients (for `RationalStabilityFunction`: `ℂᵘ⁻, ℂᵘ⁰, ℂᵘ⁺, ℂᶜ⁻, ℂᶜ⁰, ℂᶜ⁺,
ℂᴰ⁻, ℂᴰ⁰, ℂᴰ⁺, Ri⁻, Ri⁺`, and `ℂᵉ⁻, ℂᵉ⁰, ℂᵉ⁺`, which SHOC does not use) are scalars of the SHOC
closure, listed by [`closure_parameters`](@ref) and replaced by [`with_parameters`](@ref)
independently of any `TKEBasedTurbulenceClosure`. Their default values were calibrated for the
TKE closure with `GradientLimitedMixingLength` and a shear production `Kᵘ S²`: they are where a
calibration of SHOC starts, not a tuned set, and they go with a shear production coefficient
`ℂᴾ = 1` of the TKE update, which is a coefficient of SHOC's TKE update as well.

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
`RationalStabilityFunction` with its published coefficients. The component holds its own copy
of the coefficients of `functions`.
"""
RichardsonStabilityFunctions(FT::DataType = Oceananigans.defaults.FloatType;
                             functions = RationalStabilityFunction(), ℂᵗ = 1, minimum_shear_squared = 1e-10) =
    RichardsonStabilityFunctions(convert_eltype(FT, functions), convert(FT, ℂᵗ), convert(FT, minimum_shear_squared))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `RichardsonStabilityFunctions` with the stability `functions` given first, as in
`RichardsonStabilityFunctions(RationalStabilityFunction())`.
"""
RichardsonStabilityFunctions(functions, FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    RichardsonStabilityFunctions(FT; functions, kw...)

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
    brunt_scale = c.brunt_scale # scale of the squared buoyancy frequency for the clipping policy [s⁻²]

    Ri = level_richardson_number(ts, brunt_k, shear_term, tke_update)
    turbulent_velocity = sqrt(clip_max(cp, c.mintke, tke_k, c.mintke))

    tkh_k = tracer_stability_function(ts.functions, Ri) * mix * turbulent_velocity
    tk_k = momentum_stability_function(ts.functions, Ri) * mix * turbulent_velocity

    iso_scale = ts.ℂᵗ * mix / turbulent_velocity
    iso = iso_scale / sqrt(1 + iso_scale * iso_scale * clip_negative(cp, brunt_k, brunt_scale))

    return iso, tkh_k, tk_k
end

#####
##### A limit of the return-to-isotropy time scale in stable stratification
#####

# The time scale of the moments at a level or an interface, from that of the component and the
# squared buoyancy frequency there. Every component but `StratificationLimitedTimeScale` returns it unchanged.
@inline limit_time_scale(time_scale, τ, N²) = τ

"""
$(TYPEDEF)

Another time scale component (`time_scale`, by default [`ColumnStabilityDamping`](@ref)) with
its return-to-isotropy time scale `τ₀` limited in stable stratification by the buoyancy
frequency, so that the stratification number `αᴺ = τ² N²` of the moments stays below `(ℂᴺ)²`.
The rate `1 / τ` is the `p`-norm of the rate `1 / τ₀` of the component and the buoyancy rate
`N⁺ / ℂᴺ`,

    1 / τ = ((1 / τ₀)ᵖ + (N⁺ / ℂᴺ)ᵖ)^(1/p),    N⁺ = √(max(N², 0) + N²ₘᵢₙ) - Nₘᵢₙ

that is `τ = τ₀ / (1 + xᵖ)^(1/p)` with `x = τ₀ N⁺ / ℂᴺ`: a smooth minimum of `τ₀` and the
buoyancy time scale `ℂᴺ / N⁺`. Then `τ N⁺ = ℂᴺ x / (1 + xᵖ)^(1/p) < ℂᴺ` for every `τ₀` and every
`p ≥ 1`, `τ → τ₀` where `x ≪ 1`, and `τ → ℂᴺ / N⁺` where the turbulence outlives the buoyancy
period. The integer exponent `p` sets how sharp the minimum is: `p = 1` is the harmonic sum
`1 / τ = 1 / τ₀ + N⁺ / ℂᴺ`, which shortens `τ` by the factor `1 / (1 + x)` (a half at `x = 1`, a
tenth already at `x = 0.1`); `p = 2` shortens it by `1 / √(1 + x²)` (`0.5 %` at `x = 0.1`, `29 %`
at `x = 1`), and `p = 4` by `1 / (1 + x⁴)^(1/4)` (`16 %` at `x = 1`). The exponent is a type
parameter: `p = 1, 2, 4` cost a division, a square root, two square roots; other integers a
power. It is not a trainable coefficient. With `p ≥ 2` the factor has zero slope in `x` at
`x = 0`, so the limit is smooth at `N² = 0` for every `p`.

It is the limit that the minimum dissipation of Oceananigans' `TKEDissipationVerticalDiffusivity`
sets on its time scale `e / ε` (`ε ≥ 𝕊u₀³ e^{3/2} / ℓ` with the length `ℓ = Cᴺ √(e / N²)`, so
`e / ε ≤ Cᴺ / (𝕊u₀³ N)`), here in the smooth form of a norm of rates.

By default the limit applies to the time scale of the moments only: the second moments
(`τ Kᶜ` in the variances of [`DowngradientSecondMoments`](@ref)) and the diagnosis of the third
moment ([`AlgebraicThirdMoment`](@ref), whose denominator vanishes at `αᴺ ≈ 24.3` with the
coefficients of `shoc.F90`). The moments live at the interfaces, so the limit is applied there, once, to `τ₀` and
`N²` interpolated to the interface; the bound then holds exactly where the moments use it (a
limit at the levels would not survive the interpolation of `τ` and `N²` separately, and applied
at both it would compound). The diagnostic `isotropy` of the levels is `τ₀`.

With `limit_diffusivities = true` the eddy diffusivities of the levels are limited as well, as
the time scale of a k-ε closure limits its diffusivities: they are multiplied by the factor
`τ / τ₀ = 1 / (1 + xᵖ)^(1/p)` of the level (`τ₀` and `N²` of the level), which for
[`ColumnStabilityDamping`](@ref) is `Kᶜ = ℂᶜ τ e` and `Kᵘ = ℂᵘ τ e` with the limited `τ`. The
diffusivities of the cold-surface fallback are not limited. For
[`RichardsonStabilityFunctions`](@ref), whose diffusivities are not proportional to its `τ₀`,
the same factor multiplies them. The dissipation of the turbulent kinetic energy is unchanged.

The frequency `Nₘᵢₙ` regularizes the square root, whose slope is otherwise infinite at
`N² = 0`: the slope of `N⁺` in `N²` is `1 / (2 Nₘᵢₙ)` at `N² = 0⁺` and zero below. It makes
`N⁺` smaller than `N` by less than `Nₘᵢₙ` (so `τ N < ℂᴺ N / (N - Nₘᵢₙ)`), which is negligible where the limit acts
(`N ≈ ℂᴺ / τ₀ ≫ Nₘᵢₙ`).

| coefficient | default | meaning |
|:------------|:--------|:--------|
| `ℂᴺ` | 3.5 | largest `τ N`; `αᴺ < (ℂᴺ)² = 12.25`, half the first zero 24.3 of the denominator of the third moment |

Where `N² ≤ 0`, and everywhere with `ℂᴺ = Inf`, the time scale (and the diffusivities) are
exactly those of `time_scale`; the original configuration is `time_scale` itself, the default of
[`SimplifiedHigherOrderClosure`](@ref).

$(TYPEDFIELDS)
"""
struct StratificationLimitedTimeScale{P, D, TS, FT} <: AbstractSHOCComponent
    "The time scale component whose return-to-isotropy time scale is limited"
    time_scale :: TS
    "Largest product of the time scale and the buoyancy frequency"
    ℂᴺ :: FT
    "Frequency `Nₘᵢₙ` that regularizes the square root of `N²` at `N² = 0`, s⁻¹"
    minimum_buoyancy_frequency :: FT

    function StratificationLimitedTimeScale{P, D}(time_scale::TS, ℂᴺ::FT, minimum_buoyancy_frequency::FT) where {P, D, TS, FT}
        (P isa Int && P ≥ 1) || throw(ArgumentError("The exponent of StratificationLimitedTimeScale must be an integer p ≥ 1, got $P."))
        D isa Bool || throw(ArgumentError("limit_diffusivities must be true or false, got $D."))
        return new{P, D, TS, FT}(time_scale, ℂᴺ, minimum_buoyancy_frequency)
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `StratificationLimitedTimeScale` of the component `time_scale`, by default
`ColumnStabilityDamping` with the values of `shoc.F90`, with scalars of type `FT`, the integer
`exponent` `p ≥ 1` of the norm of the rates, and whether the eddy diffusivities are limited too
(`limit_diffusivities`).
"""
StratificationLimitedTimeScale(FT::DataType = Oceananigans.defaults.FloatType;
                               time_scale = ColumnStabilityDamping(FT), ℂᴺ = 3.5, exponent = 1,
                               limit_diffusivities = false, minimum_buoyancy_frequency = 1e-4) =
    StratificationLimitedTimeScale{exponent, limit_diffusivities}(convert_eltype(FT, time_scale), convert(FT, ℂᴺ),
                                                                  convert(FT, minimum_buoyancy_frequency))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `StratificationLimitedTimeScale` of `time_scale` given first, as in
`StratificationLimitedTimeScale(ColumnStabilityDamping(); ℂᴺ = 3.5, exponent = 2)`.
"""
StratificationLimitedTimeScale(time_scale::AbstractSHOCComponent, FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    StratificationLimitedTimeScale(FT; time_scale, kw...)

const SLTS = StratificationLimitedTimeScale

# The exponent and the switch are type parameters, which the generic rebuilds of a component would drop
@inline map_scalars(f, ts::SLTS{P, D}) where {P, D} =
    SLTS{P, D}(map_scalars(f, ts.time_scale), f(ts.ℂᴺ), f(ts.minimum_buoyancy_frequency))

@inline function rebuild_component(ts::SLTS{P, D}, scalars, n) where {P, D}
    time_scale, n = rebuild_component(ts.time_scale, scalars, n)
    ℂᴺ, n = rebuild_component(ts.ℂᴺ, scalars, n)
    minimum_buoyancy_frequency, n = rebuild_component(ts.minimum_buoyancy_frequency, scalars, n)
    return SLTS{P, D}(time_scale, ℂᴺ, minimum_buoyancy_frequency), n
end

Adapt.adapt_structure(to, ts::SLTS{P, D}) where {P, D} =
    SLTS{P, D}(Adapt.adapt(to, ts.time_scale), Adapt.adapt(to, ts.ℂᴺ), Adapt.adapt(to, ts.minimum_buoyancy_frequency))

function Base.summary(ts::SLTS{P, D}) where {P, D}
    options = string(P == 1 ? "" : " (exponent $P)", D ? " with limited diffusivities" : "")
    return string("StratificationLimitedTimeScale", options, " of ", summary(ts.time_scale))
end

# (1 + xᵖ)^(1/p), the factor τ₀ / τ of the limit, for x = τ₀ N⁺ / ℂᴺ ≥ 0
@inline rate_norm_factor(::Val{1}, x) = 1 + x
@inline rate_norm_factor(::Val{2}, x) = sqrt(1 + x * x)
@inline function rate_norm_factor(::Val{4}, x)
    x² = x * x
    return sqrt(sqrt(1 + x² * x²))
end
@inline rate_norm_factor(::Val{P}, x) where P = (1 + x^P)^inv(convert(typeof(x), P))

# N⁺ = √(N²⁺ + Nₘᵢₙ²) - Nₘᵢₙ, exactly zero where N² ≤ 0
@inline function regularized_buoyancy_frequency(ts::SLTS, N²)
    Nₘᵢₙ = ts.minimum_buoyancy_frequency
    N²⁺ = max(N², zero(N²))
    return N²⁺ / (sqrt(N²⁺ + Nₘᵢₙ * Nₘᵢₙ) + Nₘᵢₙ)
end

@inline function limit_time_scale(ts::SLTS{P}, τ, N²) where P
    N⁺ = regularized_buoyancy_frequency(ts, N²)
    return τ / rate_norm_factor(Val(P), τ * N⁺ / ts.ℂᴺ)
end

@inline time_scale_column(ts::SLTS, cp, c, brunt_int) = time_scale_column(ts.time_scale, cp, c, brunt_int)

@inline dissipation_coefficient(ts::SLTS, column, brunt_k, shear_term, tke_update) =
    dissipation_coefficient(ts.time_scale, column, brunt_k, shear_term, tke_update)

@inline cold_surface_fallback(ts::SLTS, tabs_sfc, zt_k, pblh) =
    cold_surface_fallback(ts.time_scale, tabs_sfc, zt_k, pblh)

# The levels carry the time scale of the wrapped component; the limit of the moments acts at the
# interfaces (limit_time_scale). With limit_diffusivities the diffusivities of the level take the
# factor τ / τ₀ of the level, except those of the cold-surface fallback.
@inline function time_scale_and_diffusivities(ts::SLTS{P, D}, cp, c, column, tke_k, a_diss, mix,
                                              brunt_k, shear_term, tke_update, runaway) where {P, D}
    iso, tkh_k, tk_k = time_scale_and_diffusivities(ts.time_scale, cp, c, column, tke_k, a_diss, mix,
                                                    brunt_k, shear_term, tke_update, runaway)
    D || return iso, tkh_k, tk_k
    N⁺ = regularized_buoyancy_frequency(ts, brunt_k)
    factor = rate_norm_factor(Val(P), iso * N⁺ / ts.ℂᴺ)
    return iso, ifelse(runaway, tkh_k, tkh_k / factor), ifelse(runaway, tk_k, tk_k / factor)
end
