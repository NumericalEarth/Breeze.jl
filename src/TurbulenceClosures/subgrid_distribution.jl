#####
##### The subgrid distribution of `TKEBasedTurbulenceClosure` (`subgrid_distribution = DoubleGaussian(...)`):
##### SHOC's assumed double-Gaussian distribution on the closure's own mixing length, turbulent kinetic
##### energy and diffusivities, and what the closure does with it:
#####
#####   * the distribution: the second moments θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′ down the gradient,
#####     `ℂᵛ τ Kᶜ ∂z a ∂z b` and `-Kᶜ ∂z a`, with the closure's `Kᶜ` and a moment time scale `τ` built
#####     from its `ℓ`, `e` and `N²`; `w′² = ℂʷ₁ (2/3) e`; the third moment w′³ and its limiter; and the
#####     assumed double-Gaussian distribution, condensing on the condensate phase of the microphysics, which gives the cloud fraction aᶜ,
#####     the cloud liquid, the condensate flux w′qᶜ′ and the buoyancy flux w′θᵛ′ = w′θˡⁱ′ + Aᵗ w′qᵗ′ + Aᶜ w′qᶜ′.
#####   * the energetics (`AppliedFluxEnergetics`, internal): the TKE equation takes its shear and
#####     buoyancy production from the energy the fluxes the closure applies take from the mean state
#####     (applied_flux_energetics.jl), and the dissipated energy heats the air.
#####   * the coupling with the static stability: with `static_stability =
#####     SaturationWeightedStaticStability(CondensateFluxFraction())` the saturated fraction is
#####     aᴺ = w′qᶜ′ / w′s′ of the distribution, solved with N² by the closure's `solver`.
#####   * the statistics of the vapor excess for the microphysics (`TurbulenceClosureStatistics`); the
#####     microphysics condenses its own cloud liquid.
#####
##### The components of part 1 are those of `SimplifiedHigherOrderClosure`: the formulation is
##### shared, the coefficients are this closure's own. The time stepping is the closure's: the
##### moments are diagnosed once per stage, after the diffusivities, and there is no split solve.
#####
##### The fields of part 1 (the second moments at faces, w′², w′³ and the cloud at centers) are stored.
#####

using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ

using .SimplifiedHigherOrderClosures: interface_second_moments, vertical_velocity_variance,
                                      third_moment, limit_third_moment, assumed_pdf,
                                      rate_norm_factor, linearly_interpolate, component_vapor_excess,
                                      representable_skewness, within_component_share, ice_saturation_thermodynamics,
                                      saturation_surface, condensate_latent_heat_and_log_slope,
                                      condensate_liquid_fraction as distributed_condensate_liquid_fraction


#####
##### The time scale of the moments
#####

"""
$(TYPEDEF)

The time scale `τ` of the second moments (`θˡⁱ′² = ℂᵛ₁ τ Kᶜ (∂z θˡⁱ)²`, ...) and of the third-moment
diagnosis, at the faces, from the closure's mixing length `ℓ`, turbulent velocity `√e` and
static stability `N²` there: `ℂᵗ` eddy turnover times, limited by the buoyancy period in stable air and
then by a `p`-norm of its rate and the buoyancy rate `N⁺ / ℂᴺ`,

    τ₀ = τᵉ / √(1 + τᵉ² N²₊),    τᵉ = ℂᵗ ℓ / √e,
    τ  = τ₀ / (1 + (τ₀ N⁺ / ℂᴺ)ᵖ)^(1/p),    N⁺ = √(N²₊ + Nₘᵢₙ²) - Nₘᵢₙ

so that `τ N⁺ < ℂᴺ`. `ℂᴺ = Inf` removes the stratification limit. The exponent `p` is a type
parameter.

$(TYPEDFIELDS)
"""
struct StratificationLimitedMomentTimeScale{P, FT}
    "Eddy turnovers in the time scale, `τᵉ = ℂᵗ ℓ / √e`"
    ℂᵗ :: FT
    "Largest `τ N`"
    ℂᴺ :: FT
    "Frequency that regularizes `N⁺` at `N² = 0`, s⁻¹"
    minimum_buoyancy_frequency :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`StratificationLimitedMomentTimeScale`](@ref): `ℂᵗ = 1` eddy turnover, the limit `ℂᴺ = 4.5` on `τ N` and
the exponent `p = 4` of its norm by default.
"""
StratificationLimitedMomentTimeScale(FT::DataType = Oceananigans.defaults.FloatType;
                                     ℂᵗ = 1, ℂᴺ = 4.5, exponent = 4, minimum_buoyancy_frequency = 1e-4) =
    StratificationLimitedMomentTimeScale{exponent, FT}(convert(FT, ℂᵗ), convert(FT, ℂᴺ), convert(FT, minimum_buoyancy_frequency))

function Adapt.adapt_structure(to, ts::StratificationLimitedMomentTimeScale{P}) where P
    ℂᵗ = adapt(to, ts.ℂᵗ)
    return StratificationLimitedMomentTimeScale{P, typeof(ℂᵗ)}(ℂᵗ, adapt(to, ts.ℂᴺ), adapt(to, ts.minimum_buoyancy_frequency))
end

convert_eltype(::Type{FT}, ts::StratificationLimitedMomentTimeScale{P}) where {P, FT} =
    StratificationLimitedMomentTimeScale{P, FT}(convert(FT, ts.ℂᵗ), convert(FT, ts.ℂᴺ), convert(FT, ts.minimum_buoyancy_frequency))

Base.summary(ts::StratificationLimitedMomentTimeScale{P}) where P =
    string("StratificationLimitedMomentTimeScale(ℂᵗ = ", prettysummary(ts.ℂᵗ), ", ℂᴺ = ", prettysummary(ts.ℂᴺ), ", p = ", P, ")")

@inline function moment_time_scale(ts::StratificationLimitedMomentTimeScale{P}, ℓ, u, N²) where P
    N²₊ = max(N², zero(N²))
    τᵉ = ts.ℂᵗ * ℓ / u
    τ₀ = τᵉ / sqrt(1 + τᵉ * τᵉ * N²₊)
    Nₘᵢₙ = ts.minimum_buoyancy_frequency
    N⁺ = N²₊ / (sqrt(N²₊ + Nₘᵢₙ * Nₘᵢₙ) + Nₘᵢₙ)
    return τ₀ / rate_norm_factor(Val(P), τ₀ * N⁺ / ts.ℂᴺ)
end

#####
##### Time scales of the second moments from the scalar-variance budget
#####

"""
$(TYPEDEF)

The time scale `τ` of the second moments from the variance budget of a scalar `φ` (θˡⁱ or qᵗ) in
local equilibrium, production equal to dissipation, with a scalar-variance dissipation time
tied to the dissipation time of the closure's TKE,

    2 Kᶜ (∂z φ)² = φ′² / τᵠ,    τᵠ = e / (ℂᵠ ε) = ℓ / (ℂᵠ 𝕊ᴰ(Ri) √e),    ε = 𝕊ᴰ(Ri) e^{3/2} / ℓ,

so that `φ′² = 2 τᵠ Kᶜ (∂z φ)²`. `ℓ`, `Ri` and the dissipation stability function `𝕊ᴰ` are those
of the closure (`Kᶜ = 𝕊ᶜ(Ri) ℓ √e`). The second moments are `ℂᵛ τ Kᶜ ∂z a ∂z b`, so this type
returns `τ = 2 τᵠ` and the budget holds with `ℂᵛ = 1`. There is no stratification limit of its
own: stability enters through `ℓ`, `𝕊ᶜ` and `𝕊ᴰ`. The squared correlation of `w` with `φ` is

    r² = (w′φ′)² / (w′² φ′²) = Kᶜ / (ℂᵛ τ w′²) = 3 ℂᵠ 𝕊ᶜ(Ri) 𝕊ᴰ(Ri) / (4 ℂᵛ ℂʷ₁),

with `w′² = ℂʷ₁ (2/3) e`, independent of the gradient and of `ℓ`. The components of the
double-Gaussian PDF have width only for `r² < 1 - ℂᵍ₁`, so for every `Ri`

    ℂᵠ < 4 ℂᵛ ℂʷ₁ (1 - ℂᵍ₁) / (3 max(𝕊ᶜ 𝕊ᴰ)).

For `RationalStabilityFunction`, `𝕊ᶜ` and `𝕊ᴰ` share the shape `s = 1 / (1 + Riᵇ / |Ri|)` on each
branch, so `𝕊ᶜ 𝕊ᴰ` is a quadratic in `s ∈ [0, 1)`; where both move the same way on a branch its
maximum is at an end point, `max(ℂᶜ⁻ ℂᴰ⁻, ℂᶜ⁰ ℂᴰ⁰, ℂᶜ⁺ ℂᴰ⁺)`. With `ℂᵛ = ℂʷ₁ = 1` and `ℂᵍ₁ = 0.4`:

| stability functions | `𝕊ᶜ 𝕊ᴰ` at Ri → -∞ / 0 / +∞ | bound on `ℂᵠ` | `ρ²` at Ri → -∞ / 0 / +∞ with `ℂᵠ = 2` |
|---|---|---|---|
| `RationalStabilityFunction()` | 5.39 / 0.0107 / 0.0140 | 0.148 | 8.1 / 0.016 / 0.021 |
| published SHOC-derived set (`ℂᶜ⁺ = 0.533`, `ℂᴰ⁺ = 0.802`) | 0.00044 / 0.0058 / 0.427 | 1.87 | 0.00066 / 0.0087 / 0.64 |

The default `ℂᵠ = 2` is the measured ratio of the scalar to the mechanical time scale,
`τᵠ ε / e ≈ 0.5` (Béguier et al. 1978; 0.5–0.7), not a fit. In a neutral surface layer
(`e = c u★²`, `ε = u★³ / (κ z)`, `Kᶜ = κ u★ z / Pr`) the closure has `𝕊ᶜ⁰ 𝕊ᴰ⁰ = 1 / (Pr c²)` for any
`ℓ`, so `ρ² = 3 ℂᵠ / (4 ℂʷ₁ Pr c²)`: about 0.05–0.15 for `c = 3`–`6` and `Pr ≈ 0.7`–`1`, the
observed range. A neutral `ρ²` far below that means `𝕊ᶜ⁰ 𝕊ᴰ⁰` (or `c = (𝕊ᵘ⁰ 𝕊ᴰ⁰)^{-1/2}`) of the
stability functions is off, and is not compensated with `ℂᵠ`.

The third-moment diagnosis keeps its own time scale, `third_moment_time_scale` (default:
SHOC's [`StratificationLimitedMomentTimeScale`](@ref)).

$(TYPEDFIELDS)
"""
struct DissipationTimeScale{FT, T3}
    "Ratio of the TKE dissipation time `e / ε` to the scalar-variance dissipation time `τᵠ`"
    ℂᵠ :: FT
    "Time scale of the third-moment diagnosis"
    third_moment_time_scale :: T3
end


"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`DissipationTimeScale`](@ref); the third moment keeps `StratificationLimitedMomentTimeScale(FT)`.
"""
DissipationTimeScale(FT::DataType = Oceananigans.defaults.FloatType; ℂᵠ = 2,
                     third_moment_time_scale = StratificationLimitedMomentTimeScale(FT)) =
    DissipationTimeScale(convert(FT, ℂᵠ), third_moment_time_scale)

for TS in (:DissipationTimeScale,)
    @eval begin
        Adapt.adapt_structure(to, ts::$TS) = $TS(adapt(to, ts.ℂᵠ), adapt(to, ts.third_moment_time_scale))
        convert_eltype(::Type{FT}, ts::$TS) where FT = $TS(convert(FT, ts.ℂᵠ), convert_eltype(FT, ts.third_moment_time_scale))
        Base.summary(ts::$TS) = string($(string(TS)), "(ℂᵠ = ", prettysummary(ts.ℂᵠ), ", third moment: ",
                                       summary(ts.third_moment_time_scale), ")")
    end
end

#####
##### The return-to-isotropy time scale of the stability functions
#####

"""
$(TYPEDEF)

The time scale `τ` of the second moments (`θˡⁱ′² = ℂᵛ₁ τ Kᶜ (∂z θˡⁱ)²`, ...) and of the third-moment diagnosis
taken from the closure's stability functions: the return-to-isotropy time scale with which they form the
diffusivities, `Kᶜ = ℂᶜ τ e` (SHOC's `isotropy`, [`ColumnStabilityDamping`](@ref)), so that the moments and the
fluxes share one time scale. It needs stability functions that have one.
"""
struct ReturnToIsotropyTimeScale end

convert_eltype(::Type{FT}, ts::ReturnToIsotropyTimeScale) where FT = ts
Base.summary(::ReturnToIsotropyTimeScale) = "ReturnToIsotropyTimeScale()"
Base.show(io::IO, ts::ReturnToIsotropyTimeScale) = print(io, summary(ts))

validate_moment_time_scale(stability_functions, time_scale) = nothing

# The time scale of the second moments at face k, from the closure's ℓ, √e (u), N² and Ri there.
# `e` and the closure fields (with the stored mixing length) are those the closure's shear reads,
# so that `𝕊ᴰ` sees the same stratification and shear numbers as the diffusivities.
@inline moment_time_scaleᶜᶜᶠ(i, j, k, grid, ts, closure, velocities, closure_fields, e, ℓ, u) =
    moment_time_scale(ts, ℓ, u, @inbounds closure_fields.N²[i, j, k])

@inline function moment_time_scaleᶜᶜᶠ(i, j, k, grid, ts::DissipationTimeScale, closure, velocities, closure_fields, e, ℓ, u)
    𝕊ᴰ = dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ)
    return 2 * ℓ / (ts.ℂᵠ * 𝕊ᴰ * u)
end

@inline moment_time_scaleᶜᶜᶠ(i, j, k, grid, ::ReturnToIsotropyTimeScale, closure, velocities, closure_fields, e, ℓ, u) =
    return_to_isotropy_time_scaleᶜᶜᶠ(i, j, k, grid, closure.stability_functions, closure, closure_fields, ℓ, u)

# The time scale of the third-moment diagnosis: the stored second-moment time scale, or its own
@inline third_moment_time_scale(::StratificationLimitedMomentTimeScale, τ, ℓ, u, N²) = τ
@inline third_moment_time_scale(::ReturnToIsotropyTimeScale, τ, ℓ, u, N²) = τ
@inline third_moment_time_scale(ts::DissipationTimeScale, τ, ℓ, u, N²) =
    moment_time_scale(ts.third_moment_time_scale, ℓ, u, N²)

#####
##### Part 1: the subgrid distribution
#####

"""
$(TYPEDEF)

The subgrid distribution of θˡⁱ, qᵗ and w of [`TKEBasedTurbulenceClosure`](@ref) (`subgrid_distribution`): the
second moments, the third moment and its limiter and the assumed PDF, evaluated on the mixing length, turbulent
kinetic energy, diffusivities and static stability of the closure. The PDF condenses with the thermodynamics of
the model, on the phase of the condensate of the water it distributes (`condensate_phase_equilibrium`): liquid, or
liquid and ice for mixed-phase saturation adjustment. With P3 the distributed water is the vapor and the cloud
liquid, which condenses over liquid; P3's ice enters through its relaxation rate and the saturation excess over ice.

$(TYPEDFIELDS)
"""
struct DoubleGaussian{TS, SM, TM, SB, SK, PDF, VT, VS, FT}
    "Time scale of the moments ([`StratificationLimitedMomentTimeScale`](@ref), [`DissipationTimeScale`](@ref), [`EddyTurnoverTimeScale`](@ref) or [`ReturnToIsotropyTimeScale`](@ref))"
    time_scale :: TS
    "Second moments ([`DowngradientSecondMoments`](@ref)); `ℂᶠ`… of the surface are not used"
    second_moments :: SM
    "Third moment of vertical velocity ([`AlgebraicThirdMoment`](@ref) or [`TKEBudgetThirdMoment`](@ref)), or `nothing` for none (unskewed)"
    third_moment :: TM
    "Bound of the stratification number of the third-moment diagnosis"
    stratification_number_bound :: SB
    "Limiter of the third moment"
    skewness :: SK
    "Assumed PDF ([`EqualWidthDoubleGaussianPDF`](@ref), [`DoubleGaussianPDF`](@ref) or [`SmoothDoubleGaussianPDF`](@ref)), with the clipping policy of its bounds"
    assumed_pdf :: PDF
    "Vertical transport of θˡⁱ′², qᵗ′² and θˡⁱ′qᵗ′ ([`ScalarVarianceTransport`](@ref)), or `nothing` for the local diagnosis"
    variance_transport :: VT
    "Stability functions of the local scalar variances and covariance ([`VarianceStabilityFunctions`](@ref)), or `nothing` for `ℂᵛ τ Kᶜ ∂z a ∂z b`"
    variance_stability_functions :: VS
    "Mixing time of air between the saturated and subsaturated parts of the distribution, in eddy turnover times ℓ / √e"
    ℂᵐ :: FT
end

Adapt.@adapt_structure DoubleGaussian

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`DoubleGaussian`](@ref). The defaults are the moment time scale of the scalar-variance budget
([`DissipationTimeScale`](@ref) with `ℂᵠ = 1.3`), SHOC's down-gradient second moments
([`DowngradientSecondMoments`](@ref)), the third moment of the TKE budget ([`TKEBudgetThirdMoment`](@ref), in place
of SHOC's [`AlgebraicThirdMoment`](@ref)) with the bound [`RationalSaturatedStratificationNumber`](@ref) of the
stratification number, the double Gaussian with equal within-component widths
([`EqualWidthDoubleGaussianPDF`](@ref), whose bounds are smooth) and the limiter
[`PolynomialSaturatedSkewness`](@ref) with the largest skewness that `assumed_pdf` represents,

    (1 - 2a) (1 - ℂᵍ)^{3/2} / √(a (1 - a)),

with `a` the least weight the PDF allows a component and `ℂᵍ` its within-component share of `w′²` at
zero skewness (4.58 for `a = 0.01`, `ℂᵍ = 0.4`), in place of SHOC's 1.2 √2. The two components of
the PDF have the same width in `w`, so the third moment is carried by their weights alone, and a
bound on the skewness is a bound on the weight of the updraft component: with 1.2 √2 it is no less
than 0.06, with 4.58 no less than 0.01. Large-eddy simulations of cumulus have a skewness of 2 to
5 in the cloud layer, where the saturated updrafts cover 0.5 to 5 %.

`time_scale` is [`DissipationTimeScale`](@ref) (the default), [`EddyTurnoverTimeScale`](@ref),
[`StratificationLimitedMomentTimeScale`](@ref) or [`ReturnToIsotropyTimeScale`](@ref) (SHOC's). The first two derive `τ` from
the scalar-variance budget with the same `ℓ`, `√e` and `Ri` as `Kᶜ`, so that the squared
correlation of `w` with θˡⁱ and qᵗ is a function of `Ri` alone, bounded by a bound on `ℂᵠ`.

`variance_stability_functions` is `nothing` (the default), for the local diagnosis
``ℂᵛ τ Kᶜ ∂_z a ∂_z b``, whose stability function is ``(2ℂᵛ/ℂᵠ) 𝕊ᶜ/𝕊ᴰ`` with
[`DissipationTimeScale`](@ref), or [`VarianceStabilityFunctions`](@ref), which give θˡⁱ′², qᵗ′² and
θˡⁱ′qᵗ′ stability functions of their own, ``𝕊ᵛ₁ ℓ² (∂_z θˡⁱ)²``, …; the fluxes are unchanged.

`ℂᵐ` sets the rate ``ωᵐ = √e / (ℂᵐ ℓ)`` at which air is exchanged between the saturated and the
subsaturated parts of the distribution, which the microphysics reads to place cloud liquid between
them (`AllProcesses` of P3's `SubgridCondensation`). The default, 1, is one eddy turnover time
``ℓ / √e``: the time an eddy of the mixing length takes to carry air across the boundary of the
saturated part.

"""
function DoubleGaussian(FT::DataType = Oceananigans.defaults.FloatType;
                             time_scale = DissipationTimeScale(FT; ℂᵠ = 1.3),
                             second_moments = DowngradientSecondMoments(FT),
                             third_moment = TKEBudgetThirdMoment(FT),
                             stratification_number_bound = RationalSaturatedStratificationNumber(FT),
                             assumed_pdf = EqualWidthDoubleGaussianPDF(FT),
                             skewness = PolynomialSaturatedSkewness(FT; maximum_skewness = representable_skewness(assumed_pdf)),
                             variance_transport = nothing,
                             variance_stability_functions = nothing,
                             ℂᵐ = 1)
    variance_transport = convert_component(FT, variance_transport)
    variance_stability_functions = convert_component(FT, variance_stability_functions)
    return DoubleGaussian(time_scale, second_moments, third_moment, stratification_number_bound,
                               skewness, assumed_pdf, variance_transport,
                               variance_stability_functions, convert(FT, ℂᵐ))
end

convert_eltype(::Type{FT}, d::DoubleGaussian) where FT =
    DoubleGaussian(convert_eltype(FT, d.time_scale), map(c -> convert_component(FT, c),
                        (d.second_moments, d.third_moment, d.stratification_number_bound, d.skewness,
                         d.assumed_pdf, d.variance_transport,
                         d.variance_stability_functions, d.ℂᵐ))...)

convert_component(FT, ::Nothing) = nothing
convert_component(FT, c) = convert_eltype(FT, c)
convert_component(FT, c::Number) = convert(FT, c)

Base.summary(d::DoubleGaussian) = string("DoubleGaussian(", nameof(typeof(d.assumed_pdf)), ", third moment: ",
                                              d.third_moment === nothing ? "none" : nameof(typeof(d.third_moment)), ")")

#####
##### The energetics of the applied fluxes (applied_flux_energetics.jl) with the solver of the coupling
#####

struct AppliedFluxEnergetics end

#####
##### The saturated fraction of the static stability from the distribution: the condensate flux fraction
#####

"""
$(TYPEDEF)

The `fraction` of [`SaturationWeightedStaticStability`](@ref) given by the closure's own
[`DoubleGaussian`](@ref) distribution: the saturated fraction is the ratio of the subgrid condensate
flux to the flux of the saturation excess `s = c_q qᵗ - c_θ θˡⁱ` (linearized at the level),

    aᴺ = w′qᶜ′ / w′s′,    w′s′ = -Kᶜ ∂z s̄,

so that the static stability `N² = (1 - aᴺ) N²ᶜ + aᴺ N²⁺`, with `N²ᶜ = (g / θ)(∂z θˡⁱ + Aᵗ ∂z qᵗ)` the
stratification of the conserved variables and `N²⁺ = N²ᶜ + (g / θ) Aᶜ ∂z s̄` the saturated one, is
`N²ᶜ - Bˡ / Kᶜ`, `Bˡ = (g / θ) Aᶜ w′qᶜ′`: minus the buoyancy flux of the distribution divided by
`Kᶜ` (`Bˡ / Kᶜ` is left out where `Kᶜ` vanishes). It is evaluated in that form. For the
double-Gaussian distribution `aᴺ` lies in `[0, 1]` up to the interpolation of the fluxes to the faces.

`aᴺ` depends on the distribution, which depends on `Kᶜ`, `ℓ` and the moment time scale, which
depend on `N²`: a fixed point, solved by the closure's `solver` ([`SinglePass`](@ref),
[`DampedFixedPoint`](@ref), [`BracketedFractionSolve`](@ref), ...) after the diffusivities. Before
the distribution of the update exists, the static stability is `initial_stability`
(`DryStaticStability()` by default).

The buoyancy production of the TKE always carries the latent part of the subgrid condensation. `stratification`
chooses the static stability that the mixing length and the stability functions read once the distribution exists:
[`LatentStratification`](@ref) (the default, the form above), [`MeanLiquidStratification`](@ref) (that of the
grid-mean state with the distribution's mean liquid as its condensate, the Brunt–Väisälä frequency of `shoc.F90`) or
[`InitialStratification`](@ref) (`initial_stability` itself).

$(TYPEDFIELDS)
"""
struct CondensateFluxFraction{S, T}
    "Static stability of the first evaluation of an update, before the distribution exists"
    initial_stability :: S
    "Static stability once the distribution exists: `LatentStratification()`, `MeanLiquidStratification()` or `InitialStratification()`"
    stratification :: T
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`CondensateFluxFraction`](@ref).
"""
CondensateFluxFraction(initial_stability = DryStaticStability(); stratification = LatentStratification()) =
    CondensateFluxFraction(initial_stability, stratification)

Base.show(io::IO, f::CondensateFluxFraction) =
    print(io, "CondensateFluxFraction(", summary(f.initial_stability), stratification_summary(f.stratification), ")")

"""
$(TYPEDEF)

The static stability of [`CondensateFluxFraction`](@ref) once the distribution exists that carries the latent part of
the subgrid condensation, `N² = N²ᶜ + (b_qˡ - b_qᵛ) [η aʷ ∂z s̄ + (1 - η) ∂z qᶜ]`: the share `aᴺ = η aʷ` of the flux of the
saturation excess that condenses feels the saturated stratification. The static stability and the buoyancy production
of the TKE are then one.
"""
struct LatentStratification end

"""
$(TYPEDEF)

The static stability of [`CondensateFluxFraction`](@ref) once the distribution exists of the grid-mean state with the
distribution's mean liquid `q̄ˡ` as its condensate, the stratification of a displacement without phase change,

    N² = N²ᵇ + (b_qˡ - b_qᵛ) ∂z (q̄ˡ - qᶜ),

`N²ᵇ` that of the state with the condensate `qᶜ` of the microphysics and `b_χ` the derivatives of the buoyancy of the
dynamics: the Brunt–Väisälä frequency of the virtual potential temperature of `shoc.F90`, whose liquid is that of its
assumed PDF. The latent part of the subgrid condensation enters the buoyancy production of the TKE only.
"""
struct MeanLiquidStratification end

Base.summary(::LatentStratification) = "LatentStratification()"
Base.summary(::MeanLiquidStratification) = "MeanLiquidStratification()"

# The default stratification, `LatentStratification()`, is not shown
stratification_summary(::LatentStratification) = ""
stratification_summary(stratification) = string(", ", summary(stratification))

# The static stability at an interior face once the distribution exists: N²ᵉ = N²ᵇ - ΔN²ᴸ the latent one, N²ᵇ that of the
# state with the condensate of the microphysics, bˡᵛ = b_qˡ - b_qᵛ at the face, Δq̄ˡ - Δqᶜ the differences across the face
# of the distribution's mean liquid and of the microphysics' condensate, N²⁰ the static stability before the pass
@inline condensate_flux_stratification(::LatentStratification, N²ᵉ, N²ᵇ, bˡᵛ, Δqˡ, Δqᶜ, Δz, N²⁰) = N²ᵉ
@inline condensate_flux_stratification(::MeanLiquidStratification, N²ᵉ, N²ᵇ, bˡᵛ, Δqˡ, Δqᶜ, Δz, N²⁰) = N²ᵇ + bˡᵛ * (Δqˡ - Δqᶜ) / Δz

const CondensateFluxFractionStability = SaturationWeightedStaticStability{<:CondensateFluxFraction}

# Before the distribution of the update exists: the initial stability
@inline static_stabilityᶜᶜᶠ(i, j, k, grid, ss::CondensateFluxFractionStability, buoyancy, tracers, closure_fields) =
    static_stabilityᶜᶜᶠ(i, j, k, grid, ss.fraction.initial_stability, buoyancy, tracers, closure_fields)

#####
##### Validation of the keywords of the closure
#####

function validate_subgrid_coupling(static_stability, distribution)
    if static_stability isa CondensateFluxFractionStability && isnothing(distribution)
        throw(ArgumentError("SaturationWeightedStaticStability(CondensateFluxFraction(...)) needs a subgrid_distribution, " *
                            "whose condensate flux gives the saturated fraction."))
    end
    return nothing
end

#####
##### Fields
#####

"""
$(TYPEDEF)

The fields of the [`DoubleGaussian`](@ref) subgrid distribution in the closure fields of
[`TKEBasedTurbulenceClosure`](@ref): the second moments at faces (`τ`, `θˡⁱ′²`, `qᵗ′²`, `θˡⁱ′qᵗ′`,
`w′θˡⁱ′`, `w′qᵗ′`) and the cloud at centers (`w′²`, `w′³`, cloud fraction `aᶜ`, liquid `qˡ`, its
variance `qˡ′²`, the fluxes `w′qˡ′` and `w′θᵛ′`, the part `Bˡ = (g / θ) Aᶜ w′qˡ′` of its buoyancy
flux carried by the condensate flux, and the shear and buoyancy production `P` and `B` of the TKE
equation from the applied fluxes).
"""
struct SubgridCloudFields{BF, M, C, P}
    energetics :: BF
    moments :: M
    cloud :: C
    presolve :: P # every closure scalar just before its implicit solve (the stage energy budget), or `nothing`
end


Adapt.adapt_structure(to, f::SubgridCloudFields) =
    SubgridCloudFields(f.energetics, adapt(to, f.moments), adapt(to, f.cloud),
                       adapt(to, f.presolve))

# Without a distribution: the plain closure, or, with a saturated fraction from the microphysics
# (or any fraction other than the distribution's), the energetics of the applied fluxes with the
# latent conversion of the condensate flux that the fraction implies
build_subgrid_cloud_fields(grid, ::Nothing, static_stability, tracer_names, heating_budget_flux = nothing) = nothing

function build_subgrid_cloud_fields(grid, distribution, static_stability, tracer_names, heating_budget_flux = nothing)
    energetics = AppliedFluxEnergetics()
    moments = (τ = ZFaceField(grid), third_moment_location_fields(grid, third_moment_location(distribution.third_moment))...,
               θˡⁱ′² = ZFaceField(grid), qᵗ′² = ZFaceField(grid), θˡⁱ′qᵗ′ = ZFaceField(grid),
               w′θˡⁱ′ = ZFaceField(grid), w′qᵗ′ = ZFaceField(grid),
               unit_variance = ZFaceField(grid), qⁱ′² = ZFaceField(grid), qⁱ′sⁱ′ = ZFaceField(grid), qᵈ′² = ZFaceField(grid),
               scalar_variance_transport_fields(grid, distribution.variance_transport)...)
    cloud = (w′² = CenterField(grid), w′³ = CenterField(grid), aᶜ = CenterField(grid), qˡ = CenterField(grid), qˡ′² = CenterField(grid),
             w′qˡ′ = CenterField(grid), w′θᵛ′ = CenterField(grid), Bˡ = CenterField(grid),
             vapor_excess_component_fields(grid)...,
             qⁱ′sⁱ′ = CenterField(grid), qᵈ′² = CenterField(grid),
             applied_flux_conversion_fields(grid, energetics)...,
             third_moment_fields(grid, distribution.third_moment)...,
             heating_budget_flux_fields(grid, heating_budget_flux)...)
    return SubgridCloudFields(energetics, moments, cloud,
                              build_presolve_fields(grid, energetics, tracer_names))
end

# The production of the TKE equation from the applied fluxes and what the stage hooks need
# (applied_flux_energetics.jl): the stage conversions (with their surface parts), the buoyancy sink
# rate, the stored derivatives b_χ of the buoyancy and ĉᵖ Π, momentum before its solve, and the
# conversions of the solved state and the dissipation of the last stage (diagnostics)
function applied_flux_conversion_fields(grid, ::AppliedFluxEnergetics)
    centers = NamedTuple(name => CenterField(grid) for name in (:B, :Bᴸ, :Bᴸⁱ, :Bⁱ, :P, :Pⁱ, :Bˢ, :Pˢ, :ωᴮ, :b_θ, :b_qᵛ, :b_qˡ, :b_qⁱ, :b_qᶜ, :ĉᵖΠ, :Bᵖ, :Pᵖ, :ε, :given, :stage_b_θ, :stage_b_qᵛ, :stage_b_qˡ, :stage_b_qⁱ, :stage_surface, :shortfall, :after_solves_ρθ, :stage_combination))
    faces = (aʷ = ZFaceField(grid), aᴺ = ZFaceField(grid), ΔN²ᴸ = ZFaceField(grid), η = ZFaceField(grid), c_q = ZFaceField(grid), c_θ = ZFaceField(grid),
             ηⁱ = ZFaceField(grid), ηʷⁱ = ZFaceField(grid), c_qⁱ = ZFaceField(grid), c_θⁱ = ZFaceField(grid))
    return merge(centers, faces, (ρu⁻ = XFaceField(grid), ρv⁻ = YFaceField(grid),
                                  stage_time_step = Oceananigans.Fields.Field{Nothing, Nothing, Nothing}(grid)))
end

# Every closure scalar just before its solve (diagnostic: the stage's energy budget)
build_presolve_fields(grid, ::AppliedFluxEnergetics, tracer_names) = NamedTuple(name => CenterField(grid) for name in tracer_names)

#####
##### Specific values from densities, for the kernels
#####

# `f / ρ` at (i, j, k); with `ρ = nothing`, `f` itself
struct SpecificValue{F, R}
    field :: F
    density :: R
end

Adapt.adapt_structure(to, s::SpecificValue) = SpecificValue(adapt(to, s.field), adapt(to, s.density))
@inline Base.getindex(s::SpecificValue, i, j, k) = @inbounds s.field[i, j, k] / s.density[i, j, k]
@inline Base.getindex(s::SpecificValue{<:Any, Nothing}, i, j, k) = @inbounds s.field[i, j, k]

# The velocity of a momentum density at its faces, `ρu / ℑx ρ` at the x-faces and `ρv / ℑy ρ` at the y-faces: the
# density of the centers interpolated to the face of the momentum, as the model diagnoses its velocities
struct XFaceVelocity{F, R, G}
    momentum :: F
    density :: R
    grid :: G
end

struct YFaceVelocity{F, R, G}
    momentum :: F
    density :: R
    grid :: G
end

Adapt.adapt_structure(to, u::XFaceVelocity) = XFaceVelocity(adapt(to, u.momentum), adapt(to, u.density), adapt(to, u.grid))
Adapt.adapt_structure(to, v::YFaceVelocity) = YFaceVelocity(adapt(to, v.momentum), adapt(to, v.density), adapt(to, v.grid))
@inline Base.getindex(u::XFaceVelocity, i, j, k) = @inbounds u.momentum[i, j, k] / ℑxᶠᵃᵃ(i, j, k, u.grid, u.density)
@inline Base.getindex(v::YFaceVelocity, i, j, k) = @inbounds v.momentum[i, j, k] / ℑyᵃᶠᵃ(i, j, k, v.grid, v.density)

# The total water of the distribution, the vapor and the cloud liquid, (ρqᵛ + ρqᶜˡ) / ρ
struct SpecificTotalWater{V, L, R}
    vapor :: V
    cloud_liquid :: L
    density :: R
end

Adapt.adapt_structure(to, q::SpecificTotalWater) =
    SpecificTotalWater(adapt(to, q.vapor), adapt(to, q.cloud_liquid), adapt(to, q.density))

@inline Base.getindex(q::SpecificTotalWater, i, j, k) =
    @inbounds (q.vapor[i, j, k] + q.cloud_liquid[i, j, k]) / q.density[i, j, k]
@inline Base.getindex(q::SpecificTotalWater{<:Any, Nothing}, i, j, k) =
    @inbounds q.vapor[i, j, k] / q.density[i, j, k]

@inline cloud_liquid_density(microphysical_fields) =
    hasproperty(microphysical_fields, :ρqᶜˡ) ? microphysical_fields.ρqᶜˡ : nothing

function distribution_potential_temperature_density(model)
    AtmosphereModels.thermodynamic_density_name(model.formulation) === :ρθ ||
        throw(ArgumentError("The subgrid distribution of TKEBasedTurbulenceClosure requires the liquid-ice " *
                            "potential temperature formulation, but the model formulation is $(summary(model.formulation))."))
    return Oceananigans.prognostic_fields(model.formulation).ρθ
end

# The liquid-ice potential temperature of the distributed water, θʷ. The model's θˡⁱ subtracts the latent heat of all
# its condensate, ĉᵖ Π θˡⁱ = cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ. Where the distributed water is the vapor and the cloud liquid
# (a microphysics with prognostic cloud liquid: P3, Kessler, non-equilibrium cloud formation), the condensate outside
# it (rain, ice, the liquid on ice) is held, not evaporated, in the all-vapor state of the distribution:
#
#     ĉᵖ Π θʷ = cᵖᵐ T - ℒˡ₀ qᶜˡ,    θʷ = θˡⁱ + (ℒˡ₀ (qˡ - qᶜˡ) + ℒⁱ₀ qⁱ) / (ĉᵖ Π),
#
# with ĉᵖ, Π those of the model at the cell, so that (θʷ, qᵛ + qᶜˡ) describe the same water and Π θʷ is the
# temperature of the cell with its cloud liquid evaporated. Where all the water is distributed (saturation
# adjustment, or no condensate), θʷ = θˡⁱ.
struct DistributedWaterPotentialTemperature{Θ, R, V, L, P, G, M, F, C, X, S}
    ρθ :: Θ
    density :: R
    moisture_density :: V
    cloud_liquid :: L
    pressure :: P
    grid :: G
    microphysics :: M
    microphysical_fields :: F
    constants :: C
    exner_exponent :: X
    standard_pressure :: S
end

Adapt.adapt_structure(to, θ::DistributedWaterPotentialTemperature) =
    DistributedWaterPotentialTemperature(adapt(to, θ.ρθ), adapt(to, θ.density), adapt(to, θ.moisture_density),
                                         adapt(to, θ.cloud_liquid), adapt(to, θ.pressure), adapt(to, θ.grid),
                                         adapt(to, θ.microphysics), adapt(to, θ.microphysical_fields),
                                         θ.constants, θ.exner_exponent, θ.standard_pressure)

@inline function Base.getindex(θ::DistributedWaterPotentialTemperature, i, j, k)
    constants = θ.constants
    @inbounds begin
        ρ = θ.density[i, j, k]
        θˡⁱ = θ.ρθ[i, j, k] / ρ
        qᶜˡ = θ.cloud_liquid[i, j, k] / ρ
        p = θ.pressure[i, j, k]
        qᵛᵉ = θ.moisture_density[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, θ.grid, θ.microphysics, ρ, qᵛᵉ, θ.microphysical_fields)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    Π = (p / θ.standard_pressure)^exner_exponent(θ.exner_exponent, q, constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    return θˡⁱ + (ℒˡ₀ * (q.liquid - qᶜˡ) + ℒⁱ₀ * q.ice) / (ĉᵖ * Π)
end

# The condensate a distribution does not distribute, at a cell: (liquid, ice), the rain, the liquid on ice and the ice
# of a microphysics with prognostic cloud liquid. The assumed PDF holds it as condensate in the all-vapor state of
# its water (`component_moisture`).
struct HeldCondensate{R, V, L, G, M, F}
    density :: R
    moisture_density :: V
    cloud_liquid :: L
    grid :: G
    microphysics :: M
    microphysical_fields :: F
end

Adapt.adapt_structure(to, h::HeldCondensate) =
    HeldCondensate(adapt(to, h.density), adapt(to, h.moisture_density), adapt(to, h.cloud_liquid), adapt(to, h.grid),
                   adapt(to, h.microphysics), adapt(to, h.microphysical_fields))

@inline function Base.getindex(h::HeldCondensate, i, j, k)
    @inbounds begin
        ρ = h.density[i, j, k]
        qᶜˡ = h.cloud_liquid[i, j, k] / ρ
        qᵛᵉ = h.moisture_density[i, j, k] / ρ
    end
    q = grid_moisture_fractions(i, j, k, h.grid, h.microphysics, ρ, qᵛᵉ, h.microphysical_fields)
    return (; liquid = q.liquid - qᶜˡ, ice = q.ice)
end

# The held condensate of the applied flux state `s` (`applied_flux_state`) on `grid`
@inline held_condensate_field(s, grid, ::Nothing) = nothing
@inline held_condensate_field(s, grid, ρqᶜˡ) = HeldCondensate(s.ρ, s.ρqᵛᵉ, ρqᶜˡ, grid, s.microphysics, s.microphysical_fields)

# The thermodynamics of the distribution at the cell (i, j, k): with the condensate it holds there
@inline level_thermodynamics(thermodynamics::NamedTuple{names}, i, j, k) where names =
    level_thermodynamics(Val(:held_condensate ∈ names), thermodynamics, i, j, k)
@inline level_thermodynamics(::Val{false}, thermodynamics, i, j, k) = thermodynamics
@inline level_thermodynamics(::Val{true}, thermodynamics, i, j, k) =
    level_thermodynamics(thermodynamics.held_condensate, thermodynamics, i, j, k)
@inline level_thermodynamics(::Nothing, thermodynamics, i, j, k) = thermodynamics
@inline level_thermodynamics(held, thermodynamics, i, j, k) = merge(thermodynamics, (; held = held[i, j, k]))

# θʷ of the applied flux state `s` on `grid`
@inline distributed_water_potential_temperature(s, grid, ::Nothing) = SpecificValue(s.ρθ, s.ρ)

@inline distributed_water_potential_temperature(s, grid, ρqᶜˡ) =
    DistributedWaterPotentialTemperature(s.ρθ, s.ρ, s.ρqᵛᵉ, ρqᶜˡ, s.p, grid, s.microphysics, s.microphysical_fields, s.constants,
                                         s.exner_exponent, s.standard_pressure)

# The distributed water of the applied flux state `s` (`applied_flux_state`) on `grid`: its liquid-ice potential
# temperature θʷ, its specific total water qʷ and the thermodynamics of the distribution, with the condensate it holds.
# They wrap the fields of `s`, so a kernel forms them from `s` where it reads them: the microphysics and its fields,
# which each of them carries, then reach the kernel once, in `s`, rather than three times (the arguments of a GPU kernel
# share about 32 KiB of parameter memory).
@inline function distributed_water(s, grid)
    ρqᶜˡ = cloud_liquid_density(s.microphysical_fields)
    θ = distributed_water_potential_temperature(s, grid, ρqᶜˡ)
    qᵗ = SpecificTotalWater(s.ρqᵛᵉ, ρqᶜˡ, s.ρ)
    thermodynamics = (constants = s.constants, standard_pressure = s.standard_pressure, exner_exponent = s.exner_exponent,
                      phase_equilibrium = s.phase_equilibrium, held_condensate = held_condensate_field(s, grid, ρqᶜˡ))
    return (; θ, qᵗ, thermodynamics)
end

#####
##### The column computation of the distribution
#####

@inline subgrid_third_moment(::Nothing, d, g, args...) = zero(eltype(args[1]))
@inline subgrid_third_moment(tm, d, g, args...) = third_moment(tm, d.stratification_number_bound, g, args...)

# The third moment at the face or the center k before its limiter: from the local arguments of the algebraic diagnosis,
# or from fields stored at that location (`TKEBudgetThirdMoment`, tke_budget_third_moment.jl)
@inline third_moment_diagnosis(tm, i, j, k, d, cl, g, algebraic_arguments) =
    subgrid_third_moment(tm, d, g, algebraic_arguments...)

# w′³ at the interior face k, with the numerics of `diag_third_shoc_moments` of `shoc.F90`: the time scale, N² and
# the mixing length are those of the face; w′² and θˡⁱ are interpolated to it from the centers k - 1 and k; the
# gradients of center quantities (w′², e) are compact across the face, over the distance of the two centers, and
# those of face quantities (θˡⁱ′², w′θˡⁱ′) are taken across the faces k - 1 and k + 1, over the thicknesses of the
# two cells between them. It is returned before its limiter: the skewness is a ratio formed at the centers.
# `g` is the gravitational acceleration.
@inline function face_third_moment(i, j, k, grid, d, closure, closure_fields, m, cl, g, θ, e)
    FT = grid_float_type(grid)
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    @inbounds begin
        zᵘ = znode(i, j, k, grid, ccc...)
        zˡ = znode(i, j, k - 1, grid, ccc...)
        zᶠ = znode(i, j, k, grid, Center(), Center(), Face())
        ℓ = closure_fields.ℓᵐ[i, j, k]
        N² = closure_fields.N²[i, j, k]
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        τ³ = third_moment_time_scale(d.time_scale, m.τ[i, j, k], ℓ, u, N²)
        τ³ = ifelse(isfinite(τ³), τ³, zero(FT))
        w′² = linearly_interpolate(zᵘ, zˡ, cl.w′²[i, j, k], cl.w′²[i, j, k-1], zᶠ)
        θᶠ = linearly_interpolate(zᵘ, zˡ, θ[i, j, k], θ[i, j, k-1], zᶠ)
        algebraic_arguments = (τ³, N², w′², θᶠ,
                               1 / (zᵘ - zˡ), 1 / (Δzᶜᶜᶜ(i, j, k, grid) + Δzᶜᶜᶜ(i, j, k-1, grid)),
                               m.θˡⁱ′²[i, j, k+1] - m.θˡⁱ′²[i, j, k-1],
                               m.w′θˡⁱ′[i, j, k+1] - m.w′θˡⁱ′[i, j, k-1],
                               cl.w′²[i, j, k] - cl.w′²[i, j, k-1],
                               max(e[i, j, k], eᵐⁱⁿ) - max(e[i, j, k-1], eᵐⁱⁿ),
                               m.w′θˡⁱ′[i, j, k])
        w′³ = third_moment_diagnosis(d.third_moment, i, j, k, d, cl, g, algebraic_arguments)
    end
    return w′³
end

# Where the third moment is diagnosed (`Face()` or `Center()`): the `location` of the third-moment component, the faces
# without one
third_moment_location(tm) = tm.location
third_moment_location(::Nothing) = Face()

# The face field of w′³ of a third moment diagnosed at the faces; the centers hold w′³ in `cloud.w′³` either way
third_moment_location_fields(grid, ::Face) = (w′³ = ZFaceField(grid),)
third_moment_location_fields(grid, ::Center) = NamedTuple()

# `Face()`: w′³ at the faces (zero at the bottom and the top), then interpolated linearly to the centers, as `shoc.F90`
# interpolates its interface w′³ to the levels, and limited there, where the PDF forms the skewness w′³ / (w′²)^{3/2}
# with the center w′² (`shoc.F90` limits at the interfaces; the bound of a ratio formed at the centers holds there). `Center()`: w′³ at the centers,
# which needs the gradients of the center quantities w′² and e across two cells.
@inline third_moment_profile!(m, cl, i, j, grid, d, closure, closure_fields, g, θ, e) =
    third_moment_profile!(third_moment_location(d.third_moment), m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)

@inline function third_moment_profile!(::Face, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())
    @inbounds begin
        m.w′³[i, j, 1] = zero(FT)
        m.w′³[i, j, Nz+1] = zero(FT)
        for k in 2:Nz
            m.w′³[i, j, k] = face_third_moment(i, j, k, grid, d, closure, closure_fields, m, cl, g, θ, e)
        end
        for k in 1:Nz
            zᶜ = znode(i, j, k, grid, ccc...)
            zˡ = znode(i, j, k, grid, ccf...)
            zᵘ = znode(i, j, k+1, grid, ccf...)
            w′³ = linearly_interpolate(zᵘ, zˡ, m.w′³[i, j, k+1], m.w′³[i, j, k], zᶜ)
            cl.w′³[i, j, k] = limit_third_moment(d.skewness, cl.w′²[i, j, k], w′³)
        end
    end
    return nothing
end

# The third moment diagnosed at the center k (`Center()`), where the PDF forms the skewness w′³ / (w′²)^{3/2}. The second moments of the faces
# k and k+1 are differenced across the cell (compact), w′² and e across the neighbouring centers, and the
# coefficients are the averages of the two faces. `lower` and `upper` are (τ, ℓ, N², θˡⁱ′², w′θˡⁱ′) at the
# faces k and k+1; `w′θˡⁱ′ᶜ` is the flux interpolated to the center; `g` is the gravitational acceleration.
@inline function center_third_moment(i, j, k, grid, d, closure, cl, g, θ, e, w′θˡⁱ′ᶜ, lower, upper)
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    τˡ, ℓˡ, N²ˡ, θˡⁱ′²ˡ, w′θˡⁱ′ˡ = lower
    τᵘ, ℓᵘ, N²ᵘ, θˡⁱ′²ᵘ, w′θˡⁱ′ᵘ = upper
    @inbounds begin
        kᵘ = min(k + 1, Nz)
        kˡ = max(k - 1, 1)
        ℓ = (ℓˡ + ℓᵘ) / 2
        N² = (N²ˡ + N²ᵘ) / 2
        u = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
        τ³ = third_moment_time_scale(d.time_scale, (τˡ + τᵘ) / 2, ℓ, u, N²)
        τ³ = ifelse(isfinite(τ³), τ³, zero(FT))
        w′² = cl.w′²[i, j, k]
        Δzʷ = znode(i, j, kᵘ, grid, ccc...) - znode(i, j, kˡ, grid, ccc...)
        compact_inverse_spacing = ifelse(kᵘ > kˡ, 1 / Δzʷ, zero(FT))
        algebraic_arguments = (τ³, N², w′², θ[i, j, k],
                               compact_inverse_spacing, 1 / Δzᶜᶜᶜ(i, j, k, grid),
                               θˡⁱ′²ᵘ - θˡⁱ′²ˡ,
                               w′θˡⁱ′ᵘ - w′θˡⁱ′ˡ,
                               cl.w′²[i, j, kᵘ] - cl.w′²[i, j, kˡ],
                               max(e[i, j, kᵘ], eᵐⁱⁿ) - max(e[i, j, kˡ], eᵐⁱⁿ),
                               w′θˡⁱ′ᶜ)
        w′³ = third_moment_diagnosis(d.third_moment, i, j, k, d, cl, g, algebraic_arguments)
    end
    return limit_third_moment(d.skewness, w′², w′³)
end

@inline function third_moment_profile!(::Center, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())
    @inbounds for k in 1:size(grid, 3)
        zᶜ = znode(i, j, k, grid, ccc...)
        zˡ = znode(i, j, k, grid, ccf...)
        zᵘ = znode(i, j, k+1, grid, ccf...)
        w′θˡⁱ′ᶜ = linearly_interpolate(zᵘ, zˡ, m.w′θˡⁱ′[i, j, k+1], m.w′θˡⁱ′[i, j, k], zᶜ)
        lower = (m.τ[i, j, k], closure_fields.ℓᵐ[i, j, k], closure_fields.N²[i, j, k], m.θˡⁱ′²[i, j, k], m.w′θˡⁱ′[i, j, k])
        upper = (m.τ[i, j, k+1], closure_fields.ℓᵐ[i, j, k+1], closure_fields.N²[i, j, k+1], m.θˡⁱ′²[i, j, k+1], m.w′θˡⁱ′[i, j, k+1])
        cl.w′³[i, j, k] = center_third_moment(i, j, k, grid, d, closure, cl, g, θ, e, w′θˡⁱ′ᶜ, lower, upper)
    end
    return nothing
end

@inline function subgrid_distribution_column!(m, cl, i, j, grid, d, closure, closure_fields, θ, qᵗ, e, velocities, p, thermodynamics)
    w = velocities.w
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())

    # The gravitational acceleration of the model, for the third moment
    g = thermodynamics.constants.gravitational_acceleration

    @inbounds begin
        # The second moments at the interior faces, down the gradient
        for k in 2:Nz
            Δz = Δzᶜᶜᶠ(i, j, k, grid)
            Kᶜ = closure_fields.Kᶜ[i, j, k]
            Kᵘ = closure_fields.Kᵘ[i, j, k]
            ℓ = closure_fields.ℓᵐ[i, j, k]
            u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
            τ = moment_time_scaleᶜᶜᶠ(i, j, k, grid, d.time_scale, closure, velocities, closure_fields, e, ℓ, u)
            τ = ifelse(isfinite(τ), τ, zero(FT))

            Δθ = θ[i, j, k] - θ[i, j, k-1]
            Δq = qᵗ[i, j, k] - qᵗ[i, j, k-1]
            θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′, _, _, _ =
                interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, Δθ, Δq, zero(FT), zero(FT), zero(FT))

            # With variance stability functions, the local variances and covariance of their own
            θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′ = local_scalar_second_moments(d.variance_stability_functions, i, j, k, grid, closure, velocities,
                                                             closure_fields, e, Δθ, Δq, Δz, θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′)

            # The fluxes of θˡⁱ and qᵗ that the turbulence carries include the budget fluxes (zero without them)
            w′θˡⁱ′ += applied_heating_flux(cl, i, j, k)


            # The variance of a moisture-like scalar of unit gradient, with the same machinery: its dissipation time
            # is that of the budgets of the condensate outside the distribution (`compute_outside_condensate_moments!`)
            _, unit_variance, _ = interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, zero(FT), Δz, zero(FT), zero(FT), zero(FT))
            _, unit_variance, _ = local_scalar_second_moments(d.variance_stability_functions, i, j, k, grid, closure, velocities,
                                                              closure_fields, e, zero(FT), Δz, Δz, zero(FT), unit_variance, zero(FT))
            m.unit_variance[i, j, k] = unit_variance

            m.τ[i, j, k] = τ
            m.θˡⁱ′²[i, j, k] = θˡⁱ′²
            m.qᵗ′²[i, j, k] = qᵗ′²
            m.θˡⁱ′qᵗ′[i, j, k] = θˡⁱ′qᵗ′
            m.w′θˡⁱ′[i, j, k] = w′θˡⁱ′
            m.w′qᵗ′[i, j, k] = w′qᵗ′
        end

        # The surface face takes the moments of the face above (the closure leaves the surface
        # flux of TKE to the boundary condition, and its buoyancy production at the first level is
        # that of the face above, too); the top face has none
        k′ = min(2, Nz)
        m.τ[i, j, 1] = m.τ[i, j, k′]
        m.θˡⁱ′²[i, j, 1] = m.θˡⁱ′²[i, j, k′]
        m.qᵗ′²[i, j, 1] = m.qᵗ′²[i, j, k′]
        m.θˡⁱ′qᵗ′[i, j, 1] = m.θˡⁱ′qᵗ′[i, j, k′]
        m.w′θˡⁱ′[i, j, 1] = m.w′θˡⁱ′[i, j, k′]
        m.w′qᵗ′[i, j, 1] = m.w′qᵗ′[i, j, k′]
        m.τ[i, j, Nz+1] = 0
        m.θˡⁱ′²[i, j, Nz+1] = 0
        m.qᵗ′²[i, j, Nz+1] = 0
        m.θˡⁱ′qᵗ′[i, j, Nz+1] = 0
        m.w′θˡⁱ′[i, j, Nz+1] = 0
        m.w′qᵗ′[i, j, Nz+1] = 0

        # The vertical velocity variance at the centers
        for k in 1:Nz
            cl.w′²[i, j, k] = vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ))
        end

        # With a transport of the scalar second moments, their steady budget (scalar_variance_transport.jl)
        transport_scalar_variances!(d.variance_transport, m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)

        m.θˡⁱ′²[i, j, 1] = m.θˡⁱ′²[i, j, k′]
        m.qᵗ′²[i, j, 1] = m.qᵗ′²[i, j, k′]
        m.θˡⁱ′qᵗ′[i, j, 1] = m.θˡⁱ′qᵗ′[i, j, k′]

        # The third moment at the faces and at the centers
        third_moment_profile!(m, cl, i, j, grid, d, closure, closure_fields, g, θ, e)

        # The assumed PDF at the centers, from moments co-located there: the second moments and w′³ are
        # interpolated from the faces to the center, so that the skewness w′³ / (w′²)^{3/2} the PDF forms is a ratio
        # of quantities at one location
        # (`@inbounds` does not reach into the body of a closure: a bounds check there is an early exit of the column
        # loop, which Reactant raises to a loop of unknown trip count that Enzyme cannot differentiate)
        interface_moments(k) = @inbounds (m.θˡⁱ′²[i, j, k], m.w′θˡⁱ′[i, j, k], m.θˡⁱ′qᵗ′[i, j, k], m.w′qᵗ′[i, j, k], m.qᵗ′²[i, j, k])
        for k in 1:Nz
            zᶜ = znode(i, j, k, grid, ccc...)
            zˡ = znode(i, j, k, grid, ccf...)
            zᵘ = znode(i, j, k+1, grid, ccf...)
            wᶜ = (w[i, j, k] + w[i, j, k+1]) / 2
            θᵏ = θ[i, j, k]
            w′²ᶜ = cl.w′²[i, j, k]
            mˡ = interface_moments(k)
            mᵘ = interface_moments(k+1)
            mᶜ = map((yᵘ, yˡ) -> linearly_interpolate(zᵘ, zˡ, yᵘ, yˡ, zᶜ), mᵘ, mˡ)
            mᶜ = (cl.w′³[i, j, k], mᶜ...)
            thermodynamicsᵏ = level_thermodynamics(thermodynamics, i, j, k)
            C, qˡ, qˡ′², w′qˡ′, w′θᵛ′ = assumed_pdf(d.assumed_pdf, thermodynamicsᵏ, zᵘ, zˡ, zᶜ,
                                                   p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], w′²ᶜ, mᶜ, mᶜ)
            store_vapor_excess_components!(cl, i, j, k,
                                           component_vapor_excess(d.assumed_pdf, thermodynamicsᵏ, zᵘ, zˡ, zᶜ,
                                                                  p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], w′²ᶜ, mᶜ, mᶜ))
            cl.aᶜ[i, j, k] = C
            # The exchange rate of air between the saturated and subsaturated parts, √e / (ℂᵐ ℓ)
            ℓᶜ = (closure_fields.ℓᵐ[i, j, k] + closure_fields.ℓᵐ[i, j, k+1]) / 2
            uᶜ = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
            cl.ωᵐ[i, j, k] = ifelse(ℓᶜ > 0, uᶜ / (d.ℂᵐ * ifelse(ℓᶜ > 0, ℓᶜ, one(FT))), zero(FT))
            cl.qˡ[i, j, k] = qˡ
            cl.qˡ′²[i, j, k] = qˡ′²
            cl.w′qˡ′[i, j, k] = w′qˡ′
            cl.w′θᵛ′[i, j, k] = w′θᵛ′
            # The stability uses the exact derivatives b_χ of the model's buoyancy at the level
            # (g / θᵥ, the moisture dependence of Π and the loading included):
            # w′b′ = b_θ w′θʷ′ + b_qᵛ w′qᵗ′ + (b_qᶜ - b_qᵛ) w′qˡ′, b_qᶜ that of the distributed condensate. The buoyancy production of the TKE
            # equation is not this distribution's: it is the conversion of the fluxes the closure
            # applies (`compute_applied_flux_conversions!`).
            cl.Bˡ[i, j, k] = (cl.b_qᶜ[i, j, k] - cl.b_qᵛ[i, j, k]) * w′qˡ′
        end
    end

    return nothing
end

# The kernels of the subgrid cloud read its fields from `closure_fields.subgrid_cloud` and the distributed water from the
# applied flux state `s` (`distributed_water`), so that each reaches the kernel once
@kernel function _compute_subgrid_distribution!(closure_fields, grid, closure, e, velocities, s)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    d = closure_ij.subgrid_distribution
    fields = closure_fields.subgrid_cloud
    water = distributed_water(s, grid)
    subgrid_distribution_column!(fields.moments, fields.cloud, i, j, grid, d, closure_ij, closure_fields, water.θ, water.qᵗ, e,
                                 velocities, s.p, water.thermodynamics)
end

function compute_subgrid_distribution!(fields::SubgridCloudFields, closure_fields, closure, model)
    grid = model.grid
    e = SpecificValue(model.tracers[TKE_NAME], nothing) # within `update_state!` the tracer holds its specific value
    launch!(grid.architecture, grid, :xy, _compute_subgrid_distribution!, closure_fields, grid, closure, e, model.velocities,
            applied_flux_state(model))
    compute_outside_condensate_moments!(fields, closure_fields, model)
    return nothing
end

#####
##### The second moments of the condensate outside the distribution and of the departure of the cloud liquid
#####
#
# Diagnostic steady budgets at the interior faces, with the dissipation time τᵠ = (unit-gradient variance) / (2Kᶜ) of a
# moisture-like scalar from the closure's own scalar-moment machinery (ℂᵛ₂ τ / 2 down the gradient, 𝕊ᵛ₂ ℓ² / (2Kᶜ)
# with variance stability functions), the down-gradient fluxes w′φ′ = -Kᶜ ∂z φ̄, and the phase relaxation rates of
# the microphysics at the face:
#
#   * the ice qⁱ outside the distributed water (P3's ice) and the excess of the distributed water over ice sⁱ:
#
#       ∂t qⁱ′sⁱ′ = 2Kᶜ ∂z q̄ⁱ ∂z s̄ⁱ + ωⁱ sⁱ′² - (1/τᵠ + ωⁱ) qⁱ′sⁱ′,
#       ∂t qⁱ′² = 2Kᶜ (∂z q̄ⁱ)² + 2ωⁱ qⁱ′sⁱ′ - qⁱ′² / τᵠ,
#
#     production down both gradients; the deposition ωⁱ sⁱ′ of each fluctuation of the excess onto the ice, which
#     correlates the ice with the excess; and the excess the ice consumes, which decorrelates them at ωⁱ. sⁱ′² =
#     vⁱ_q² qᵗ′² - 2vⁱ_q vⁱ_θ θ′qᵗ′ + vⁱ_θ² θ′² and ∂z s̄ⁱ = vⁱ_q ∂z qʷ - vⁱ_θ ∂z θʷ come from the distribution's moments,
#     with the coefficients of the vapor excess over ice at fixed condensate linearized about the state of each cell
#     (`vapor_excess_coefficients`, the numerators of the coefficients of the latent correction). The covariance is bounded by the realizability |qⁱ′sⁱ′| < (qⁱ′² sⁱ′²)^{1/2} in the rational form of
#     `realizable_correlation`. Sedimentation with a uniform fall speed transports the variance without producing
#     it locally; the production by fluctuations of the fall speed (with the size of the particles) is omitted.
#   * the departure d = qᶜˡ - qᶜ of the cloud liquid from the distribution's equilibrium condensate qᶜ (its mean
#     liquid), which relaxes back at the droplets' rate ωˡ:
#
#       ∂t d′² = 2Kᶜ (∂z d̄)² - 2ωˡ d′² - d′² / τᵠ.
#
# Steady: qⁱ′sⁱ′ = τᵠ (2Kᶜ ∂z q̄ⁱ ∂z s̄ⁱ + ωⁱ sⁱ′²) / (1 + τᵠ ωⁱ), qⁱ′² = τᵠ (2Kᶜ (∂z q̄ⁱ)² + 2ωⁱ qⁱ′sⁱ′),
# d′² = 2Kᶜ τᵠ (∂z d̄)² / (1 + 2τᵠ ωˡ). They are statistics: the water and the energy are unchanged.
compute_outside_condensate_moments!(fields, closure_fields, model) = nothing

function compute_outside_condensate_moments!(fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields, model)
    grid = model.grid
    launch!(grid.architecture, grid, :xy, _compute_outside_condensate_moments!, fields.moments, fields.cloud, grid,
            closure_fields.Kᶜ, applied_flux_state(model))
    return nothing
end

# The ice outside the distributed water at a cell (zero where nothing is held)
@inline outside_ice(::Nothing, i, j, k) = 0
@inline outside_ice(held, i, j, k) = held[i, j, k].ice

@kernel function _compute_outside_condensate_moments!(m, cl, grid, Kᶜ, s)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    θ, qʷ, thermodynamics = distributed_water(s, grid)
    ice = ice_saturation_thermodynamics(thermodynamics)
    held = thermodynamics.held_condensate
    @inbounds begin
        for k in 2:Nz
            Δz = Δzᶜᶜᶠ(i, j, k, grid)
            K = Kᶜ[i, j, k]
            resolved = K > 0
            τᵠ = ifelse(resolved, m.unit_variance[i, j, k] / (2 * ifelse(resolved, K, one(FT))), zero(FT))
            # The vapor excess over ice sⁱ = vⁱ_q qʷ - vⁱ_θ θʷ, linearized about the state of each cell
            vⁱ_q⁺, vⁱ_θ⁺ = vapor_excess_coefficients(ice, i, j, k, grid, s, θ[i, j, k])
            vⁱ_q⁻, vⁱ_θ⁻ = vapor_excess_coefficients(ice, i, j, k-1, grid, s, θ[i, j, k-1])
            vⁱ_q = (vⁱ_q⁺ + vⁱ_q⁻) / 2
            vⁱ_θ = (vⁱ_θ⁺ + vⁱ_θ⁻) / 2
            sⁱ′² = max(0, vⁱ_q * vⁱ_q * m.qᵗ′²[i, j, k] - 2 * vⁱ_q * vⁱ_θ * m.θˡⁱ′qᵗ′[i, j, k] + vⁱ_θ * vⁱ_θ * m.θˡⁱ′²[i, j, k])
            ∂z_sⁱ = (vⁱ_q * (qʷ[i, j, k] - qʷ[i, j, k-1]) - vⁱ_θ * (θ[i, j, k] - θ[i, j, k-1])) / Δz
            ∂z_qⁱ = (convert(FT, outside_ice(held, i, j, k)) - convert(FT, outside_ice(held, i, j, k-1))) / Δz
            ω⁺ = cell_phase_relaxation_rates(i, j, k, grid, s)
            ω⁻ = cell_phase_relaxation_rates(i, j, k-1, grid, s)
            ωⁱ = (ω⁺.ωᵈⁱ + ω⁻.ωᵈⁱ) / 2
            ωˡ = (ω⁺.ωˡ + ω⁻.ωˡ) / 2
            covariance = τᵠ * (2 * K * ∂z_qⁱ * ∂z_sⁱ + ωⁱ * sⁱ′²) / (1 + τᵠ * ωⁱ)
            variance = τᵠ * (2 * K * ∂z_qⁱ * ∂z_qⁱ + 2 * ωⁱ * covariance)
            # The covariance bounded below (qⁱ′² sⁱ′²)^{1/2}: C₀ / (1 + (C₀ / bound)⁴)^{1/4}, scaled by the larger
            bound = sqrt(max(0, variance) * sⁱ′²)
            scale = max(bound, abs(covariance))
            resolved = scale > 0
            scaleₛ = ifelse(resolved, scale, one(FT))
            m.qⁱ′²[i, j, k] = max(0, variance)
            m.qⁱ′sⁱ′[i, j, k] = ifelse(resolved, covariance * bound / (scaleₛ * sqrt(sqrt((bound / scaleₛ)^4 + (covariance / scaleₛ)^4))), zero(FT))
            # The departure of the cloud liquid from the equilibrium condensate of the distribution
            d⁺ = distributed_condensateᶜᶜᶜ(i, j, k, grid, s) - cl.qˡ[i, j, k]
            d⁻ = distributed_condensateᶜᶜᶜ(i, j, k-1, grid, s) - cl.qˡ[i, j, k-1]
            ∂z_d = (d⁺ - d⁻) / Δz
            m.qᵈ′²[i, j, k] = 2 * K * τᵠ * ∂z_d * ∂z_d / (1 + 2 * τᵠ * ωˡ)
        end
        m.qⁱ′²[i, j, 1] = m.qⁱ′²[i, j, min(2, Nz)]
        m.qⁱ′sⁱ′[i, j, 1] = m.qⁱ′sⁱ′[i, j, min(2, Nz)]
        m.qᵈ′²[i, j, 1] = m.qᵈ′²[i, j, min(2, Nz)]
        m.qⁱ′²[i, j, Nz+1] = 0
        m.qⁱ′sⁱ′[i, j, Nz+1] = 0
        m.qᵈ′²[i, j, Nz+1] = 0
        for k in 1:Nz
            cl.qⁱ′sⁱ′[i, j, k] = (m.qⁱ′sⁱ′[i, j, k] + m.qⁱ′sⁱ′[i, j, k+1]) / 2
            cl.qᵈ′²[i, j, k] = (m.qᵈ′²[i, j, k] + m.qᵈ′²[i, j, k+1]) / 2
        end
    end
end

#####
##### Coupling to the closure
#####

# The buoyancy derivatives of the prognostic state, unless the third moment computed them already
compute_subgrid_cloud_buoyancy_derivatives!(fields, distribution, model) = compute_buoyancy_derivatives!(fields, model)

# The budget fluxes of the prognostic state, of the diabatic heating (heating_budget_flux.jl) and of the total water
# (total_water_budget_flux.jl)
function compute_budget_fluxes!(fields, closure_fields, closure, model, tracers)
    compute_heating_budget_flux!(fields, heating_budget_flux(closure), closure_fields, closure, model, tracers)
    return nothing
end

# The budget fluxes, likewise unless the third moment computed them already
compute_subgrid_cloud_budget_fluxes!(fields, distribution, closure_fields, closure, model, tracers) =
    compute_budget_fluxes!(fields, closure_fields, closure, model, tracers)

# Within `update_state!`, after the diffusivities and before the implicit coefficient
function compute_subgrid_cloud!(fields::SubgridCloudFields, closure_fields, closure, model, tracers, parameters)
    compute_subgrid_cloud_buoyancy_derivatives!(fields, subgrid_distribution(closure), model)
    compute_subgrid_cloud_budget_fluxes!(fields, subgrid_distribution(closure), closure_fields, closure, model, tracers)
    compute_subgrid_distribution!(fields, closure_fields, closure, model)
    solve_static_stability!(closure.static_stability, fields.energetics, fields, closure_fields, closure, model, tracers, parameters)
    return nothing
end

# The coupling of the static stability with the distribution is solved only when the static
# stability slot reads the distribution, with the closure's solver
solve_static_stability!(static_stability, energetics, args...) = nothing
solve_static_stability!(::CondensateFluxFractionStability, energetics, args...) = applied_flux_stability!(energetics, args...)


# The stratification of the mixed variables N²ᶜ = Σ b̄_χ δz χ at the interior face k and the latent-corrected
# N²ᵇ = N²ᶜ - Bˡ / Kᶜ, with Bˡ / Kᶜ in the form in which Kᶜ cancels: Bˡ = (b_qˡ - b_qᵛ) w′qˡ′ and
# w′qˡ′ = aʷ w′s′, w′s′ = -Kᶜ ∂z s̄, so -Bˡ / Kᶜ = (b_qˡ - b_qᵛ) aʷ ∂z s̄, with (b_qˡ - b_qᵛ) aʷ formed at the
# centers from the saturated share of the components there (`center_condensate_flux_fraction`) and averaged to the
# face. Also returns ∂z s̄ and the face b̄ᶜ - b̄ᵛ.
@inline function condensate_flux_stabilityᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
    km = max(k - 1, 1)
    g = saturation_excess_gradientᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
    @inbounds begin
        b_qᵛᶠ = (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, km]) / 2
        bˡᵛaʷ⁺ = (cl.b_qᶜ[i, j, k] - cl.b_qᵛ[i, j, k]) * center_condensate_flux_fraction(i, j, k, cl)
        bˡᵛaʷ⁻ = (cl.b_qᶜ[i, j, km] - cl.b_qᵛ[i, j, km]) * center_condensate_flux_fraction(i, j, km, cl)
    end
    N²ᶜ = g.b_θᶠ * g.∂zθ + b_qᵛᶠ * g.∂zq
    N²ᵇ = N²ᶜ + (bˡᵛaʷ⁺ + bˡᵛaʷ⁻) / 2 * g.∂zs
    return N²ᶜ, N²ᵇ, g.∂zs, g.bᶜᵛ
end

# Across the face k (between the centers k - 1 and k): the gradients of θʷ and qʷ, that of the saturation excess
# linearized about the state of each cell, ∂z s̄ = c̄_q ∂z qʷ - c̄_θ ∂z θʷ (`saturated_condensate_coefficients`), the face
# b̄_θ (`b_θᶠ`) and the buoyancy of condensate in place of vapor, b̄ᶜ - b̄ᵛ, so that N²⁺ - N²ᶜ = (b̄ᶜ - b̄ᵛ) ∂z s̄: the one
# linearization of the bracket, of the target of the solve and of the latent correction
@inline function saturation_excess_gradientᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
    km = max(k - 1, 1)
    @inbounds begin
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        Δθ = θ[i, j, k] - θ[i, j, km]
        Δq = qᵗ[i, j, k] - qᵗ[i, j, km]
        b_θᶠ = (cl.b_θ[i, j, k] + cl.b_θ[i, j, km]) / 2
        bᶜᵛ = (cl.b_qᶜ[i, j, k] + cl.b_qᶜ[i, j, km] - cl.b_qᵛ[i, j, k] - cl.b_qᵛ[i, j, km]) / 2
        c_q⁺, c_θ⁺ = saturated_condensate_coefficients(thermodynamics, i, j, k, grid, s, θ[i, j, k])
        c_q⁻, c_θ⁻ = saturated_condensate_coefficients(thermodynamics, i, j, km, grid, s, θ[i, j, km])
    end
    ∂zθ = Δθ / Δz
    ∂zq = Δq / Δz
    ∂zs = (c_q⁺ + c_q⁻) / 2 * ∂zq - (c_θ⁺ + c_θ⁻) / 2 * ∂zθ
    return (; Δz, Δθ, Δq, ∂zθ, ∂zq, b_θᶠ, ∂zs, bᶜᵛ)
end

applied_flux_stability!(::AppliedFluxEnergetics, fields, closure_fields, closure, model, tracers, parameters) =
    applied_flux_stability_pass!(fields, closure_fields, closure, model, tracers, parameters)

# The buoyancy production of the TKE equation: the conversion of the applied fluxes
@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:AppliedFluxEnergetics}, closure_fields) =
    @inbounds fields.cloud.B[i, j, k]

#####
##### The coefficients of the closure, as one flat view (as for `SimplifiedHigherOrderClosure`)
#####

SimplifiedHigherOrderClosures.closure_components(closure::TKEBasedTurbulenceClosure) =
    (closure.mixing_length, closure.stability_functions, closure.static_stability, closure.shear, closure.subgrid_distribution,
     closure.heating_budget_flux)

SimplifiedHigherOrderClosures.rebuild_component(::Nothing, scalars, n) = nothing, n

# The exponent of the stratification limit is a type parameter
SimplifiedHigherOrderClosures.component_constructor(::Type{<:StratificationLimitedMomentTimeScale{P}}) where P =
    (ℂᵗ, ℂᴺ, N) -> StratificationLimitedMomentTimeScale{P, typeof(ℂᵗ)}(ℂᵗ, ℂᴺ, N)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a copy of the [`TKEBasedTurbulenceClosure`](@ref) `closure` in which the scalars named by the
keyword arguments are replaced, whichever component (mixing length, stability functions, static
stability, shear, subgrid distribution, heating budget flux) holds them. [`closure_parameters`](@ref) lists the
coefficients. The non-integer scalars are promoted with the replacements, so that traced scalars
give a closure whose components all have the promoted scalar type.
"""
function SimplifiedHigherOrderClosures.with_parameters(closure::TKEBasedTurbulenceClosure{TD}; kw...) where TD
    scalars = SimplifiedHigherOrderClosures.closure_scalars(closure)
    names = keys(scalars)
    all(name -> name ∈ names, keys(kw)) ||
        throw(ArgumentError("with_parameters: $(setdiff(keys(kw), names)) are not scalars of the components of " *
                            "$(summary(closure)); they are $(names)."))
    replaced = map(name -> haskey(kw, name) ? kw[name] : scalars[name], names)
    is_count(x) = x isa Integer
    T = promote_type(map(typeof, filter(!is_count, replaced))...)
    promoted = map(x -> is_count(x) ? x : convert(T, x), replaced)

    n = 1
    components = ()
    for component in SimplifiedHigherOrderClosures.closure_components(closure)
        component, n = SimplifiedHigherOrderClosures.rebuild_component(component, promoted, n)
        components = (components..., component)
    end
    mixing_length, stability_functions, static_stability, shear, subgrid_distribution, heating_budget_flux = components

    return TKEBasedTurbulenceClosure{TD}(mixing_length, stability_functions, static_stability,
                                         closure.minimum_tke, shear,
                                         closure.negative_tke_damping_time_scale, subgrid_distribution,
                                         heating_budget_flux)
end

#####
##### The statistics of the vapor excess for the microphysics (`TurbulenceClosureStatistics`)
#####

# The double Gaussian at the centers as the distribution of the vapor excess over liquid: the
# weight `a` of the updraft component, and each component's mean `μ₁`, `μ₂` and spread `σ₁`, `σ₂`; and over ice,
# `μⁱ₁`, `μⁱ₂`, `σⁱ₁`, `σⁱ₂`.
# The saturated fraction `f⁺` is the distribution's own cloud fraction `aᶜ`.
vapor_excess_component_fields(grid) = (a₁ = CenterField(grid), μ₁ = CenterField(grid), μ₂ = CenterField(grid),
                                       σ₁ = CenterField(grid), σ₂ = CenterField(grid), ωᵐ = CenterField(grid),
                                       μⁱ₁ = CenterField(grid), μⁱ₂ = CenterField(grid),
                                       σⁱ₁ = CenterField(grid), σⁱ₂ = CenterField(grid),
                                       𝕏ⁱ₁ = CenterField(grid), 𝕏ⁱ₂ = CenterField(grid))

@inline function store_vapor_excess_components!(components, i, j, k, (a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂, r₁, r₂))
    @inbounds begin
        components.a₁[i, j, k] = a
        components.μ₁[i, j, k] = μ₁
        components.μ₂[i, j, k] = μ₂
        components.σ₁[i, j, k] = σ₁
        components.σ₂[i, j, k] = σ₂
        components.μⁱ₁[i, j, k] = μⁱ₁
        components.μⁱ₂[i, j, k] = μⁱ₂
        components.σⁱ₁[i, j, k] = σⁱ₁
        components.σⁱ₂[i, j, k] = σⁱ₂
        components.𝕏ⁱ₁[i, j, k] = r₁
        components.𝕏ⁱ₂[i, j, k] = r₂
    end
    return nothing
end

@inline function stored_vapor_excess_components(i, j, k, components)
    @inbounds begin
        a = components.a₁[i, j, k]
        return (; a = (a, 1 - a), μ = (components.μ₁[i, j, k], components.μ₂[i, j, k]),
                  σ = (components.σ₁[i, j, k], components.σ₂[i, j, k]), f⁺ = components.aᶜ[i, j, k],
                  ωᵐ = components.ωᵐ[i, j, k],
                  μⁱ = (components.μⁱ₁[i, j, k], components.μⁱ₂[i, j, k]),
                  σⁱ = (components.σⁱ₁[i, j, k], components.σⁱ₂[i, j, k]),
                  𝕏ⁱ = (components.𝕏ⁱ₁[i, j, k], components.𝕏ⁱ₂[i, j, k]),
                  ice_covariance = components.qⁱ′sⁱ′[i, j, k], departure_variance = components.qᵈ′²[i, j, k])
    end
end

@inline AtmosphereModels.supersaturation_statistics(::TurbulenceClosureStatistics, i, j, k, grid, ::FlavorOfTKEClosure,
                                                    closure_fields::TKEClosureFields, model_fields) =
    closure_vapor_excess_statistics(i, j, k, grid, closure_fields.subgrid_cloud)

# Without a subgrid distribution: no spread, the grid-mean rate
@inline function closure_vapor_excess_statistics(i, j, k, grid, ::Nothing)
    FT = grid_float_type(grid)
    return (; μ = zero(FT), σ = zero(FT), skewness = zero(FT), f⁺ = zero(FT))
end

@inline closure_vapor_excess_statistics(i, j, k, grid, fields::SubgridCloudFields{<:Any, Nothing}) =
    closure_vapor_excess_statistics(i, j, k, grid, nothing)

@inline closure_vapor_excess_statistics(i, j, k, grid, fields::SubgridCloudFields) =
    stored_vapor_excess_components(i, j, k, fields.cloud)
