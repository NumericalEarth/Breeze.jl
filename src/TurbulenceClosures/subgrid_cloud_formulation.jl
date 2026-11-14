#####
##### The subgrid cloud formulation of `TKEBasedTurbulenceClosure`: SHOC's subgrid cloud scheme on
##### the closure's own mixing length, turbulent kinetic energy and diffusivities
#####
##### Three parts, each switched on or off independently (`nothing` is off):
#####
#####   1. `distribution` (`SubgridDistribution`): the second moments θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′
#####      down the gradient, `ℂᵛ τ Kᶜ ∂z a ∂z b` and `-Kᶜ ∂z a`, with the closure's `Kᶜ` and a
#####      moment time scale `τ` built from its `ℓ`, `e` and `N²`; `w′² = ℂʷ₁ (2/3) e`; the third
#####      moment w′³ of SHOC's algebraic diagnosis and its limiter; and the assumed double-Gaussian
#####      distribution of SHOC with its saturation (`pdf_saturation`), which gives the cloud
#####      fraction, the cloud liquid, the liquid flux w′qˡ′ and the buoyancy flux
#####      w′θᵥ′ = w′θˡ′ + A_q w′qᵗ′ + A_c w′qˡ′.
#####   2. `buoyancy_flux` (`CloudBuoyancyFlux`): the TKE equation takes its shear and buoyancy
#####      production from the energy the fluxes the closure applies take from the mean state
#####      (applied_flux_energetics.jl), and the dissipated energy heats the air. The buoyancy
#####      production is not the distribution's w′θᵥ′. The positive part is an explicit source and the
#####      negative part joins the implicit linear coefficient, as the buoyancy sink does without
#####      the formulation. Its `stability_solver` and passes solve the coupling of the
#####      distribution with the static stability when the closure's `static_stability` is
#####      `SubgridCloudStaticStability(LiquidFluxFraction(...))`, whose weight is
#####      ℵ = w′qˡ′ / w′s′ (`stability = true` is a shortcut for that static stability).
#####   3. `cloud` (`ClosureOwnedCloud`): with a microphysics whose subgrid choice is `ClosureCloud`,
#####      the closure supplies the cloud fraction and cloud liquid (`ClosureCloudStatistics`) and
#####      sets the cloud liquid of the model once per time step, after the last stage, to the
#####      liquid of the distribution, keeping the total water and θˡⁱ. Without it the
#####      microphysics keeps its own condensation.
#####
##### The components of part 1 are those of `SimplifiedHigherOrderClosure`: the formulation is
##### shared, the coefficients are this closure's own. The time stepping is the closure's: the
##### moments are diagnosed once per stage, after the diffusivities, and there is no split solve.
#####
##### The fields of part 1 (the moments at faces, w′² and w′³, the cloud at centers) are stored,
##### so that a microphysics correlation may later read the spread of the saturation deficit or
##### the skewness through `subgrid_statisticsᶜᶜᶜ` without changing this structure.
#####

using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ

using .SimplifiedHigherOrderClosures: interface_second_moments, vertical_velocity_variance,
                                      third_moment, limit_third_moment, assumed_pdf, buoyancy_flux_coefficients,
                                      plume_liquid_water_temperature, plume_saturation,
                                      clip_below, shoc_saturation_thermodynamics,
                                      rate_norm_factor

using ..AtmosphereModels: ClosureCloud, ClosureCloudStatistics, subgrid_correlation

#####
##### The time scale of the moments
#####

"""
$(TYPEDEF)

The time scale `τ` of the second moments (`θˡ′² = ℂᵛ₁ τ Kᶜ (∂z θˡ)²`, ...) and of the third-moment
diagnosis, at the faces, from the closure's mixing length `ℓ`, turbulent velocity `√e` and
static stability `N²` there: the return-to-isotropy time scale of SHOC's
`RichardsonStabilityFunctions` limited by stratification as SHOC's
`StratificationLimitedTimeScale`,

    τ₀ = τᵉ / √(1 + τᵉ² N²⁺),    τᵉ = ℂᵗ ℓ / √e,
    τ  = τ₀ / (1 + (τ₀ N⁺ / ℂᴺ)ᵖ)^(1/p),    N⁺ = √(N²⁺ + Nₘᵢₙ²) - Nₘᵢₙ

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

Return a [`StratificationLimitedMomentTimeScale`](@ref) with the values of the fixed smooth SHOC
configuration: `ℂᵗ = 1`, `ℂᴺ = 4.5`, `exponent = 4`.
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
    N²⁺ = max(N², zero(N²))
    τᵉ = ts.ℂᵗ * ℓ / u
    τ₀ = τᵉ / sqrt(1 + τᵉ * τᵉ * N²⁺)
    Nₘᵢₙ = ts.minimum_buoyancy_frequency
    N⁺ = N²⁺ / (sqrt(N²⁺ + Nₘᵢₙ * Nₘᵢₙ) + Nₘᵢₙ)
    return τ₀ / rate_norm_factor(Val(P), τ₀ * N⁺ / ts.ℂᴺ)
end

#####
##### Time scales of the second moments from the scalar-variance budget
#####

"""
$(TYPEDEF)

The time scale `τ` of the second moments from the variance budget of a scalar `φ` (θˡ or qᵗ) in
local equilibrium, production equal to dissipation, with a scalar-variance dissipation time
tied to the dissipation time of the closure's TKE,

    2 Kᶜ (∂z φ)² = φ′² / τᵠ,    τᵠ = e / (ℂᵠ ε) = ℓ / (ℂᵠ Sᴰ(Ri) √e),    ε = Sᴰ(Ri) e^{3/2} / ℓ,

so that `φ′² = 2 τᵠ Kᶜ (∂z φ)²`. `ℓ`, `Ri` and the dissipation stability function `Sᴰ` are those
of the closure (`Kᶜ = Sᶜ(Ri) ℓ √e`). The second moments are `ℂᵛ τ Kᶜ ∂z a ∂z b`, so this type
returns `τ = 2 τᵠ` and the budget holds with `ℂᵛ = 1`. There is no stratification limit of its
own: stability enters through `ℓ`, `Sᶜ` and `Sᴰ`. The squared correlation of `w` with `φ` is

    ρ² = (w′φ′)² / (w′² φ′²) = Kᶜ / (ℂᵛ τ w′²) = 3 ℂᵠ Sᶜ(Ri) Sᴰ(Ri) / (4 ℂᵛ ℂʷ₁),

with `w′² = ℂʷ₁ (2/3) e`, independent of the gradient and of `ℓ`. The plumes of the
double-Gaussian PDF have width only for `ρ² < 1 - ℂᵍ₁`, so for every `Ri`

    ℂᵠ < 4 ℂᵛ ℂʷ₁ (1 - ℂᵍ₁) / (3 max(Sᶜ Sᴰ)).

For `RationalStabilityFunction`, `Sᶜ` and `Sᴰ` share the shape `s = 1 / (1 + Riᵇ / |Ri|)` on each
branch, so `Sᶜ Sᴰ` is a quadratic in `s ∈ [0, 1)`; where both move the same way on a branch its
maximum is at an end point, `max(ℂᶜ⁻ ℂᴰ⁻, ℂᶜ⁰ ℂᴰ⁰, ℂᶜ⁺ ℂᴰ⁺)`. With `ℂᵛ = ℂʷ₁ = 1` and `ℂᵍ₁ = 0.4`:

| stability functions | `Sᶜ Sᴰ` at Ri → -∞ / 0 / +∞ | bound on `ℂᵠ` | `ρ²` at Ri → -∞ / 0 / +∞ with `ℂᵠ = 2` |
|---|---|---|---|
| `RationalStabilityFunction()` | 5.39 / 0.0107 / 0.0140 | 0.148 | 8.1 / 0.016 / 0.021 |
| published SHOC-derived set (`ℂᶜ⁺ = 0.533`, `ℂᴰ⁺ = 0.802`) | 0.00044 / 0.0058 / 0.427 | 1.87 | 0.00066 / 0.0087 / 0.64 |

The default `ℂᵠ = 2` is the measured ratio of the scalar to the mechanical time scale,
`τᵠ ε / e ≈ 0.5` (Béguier et al. 1978; 0.5–0.7), not a fit. In a neutral surface layer
(`e = c u★²`, `ε = u★³ / (κ z)`, `Kᶜ = κ u★ z / Pr`) the closure has `Sᶜ⁰ Sᴰ⁰ = 1 / (Pr c²)` for any
`ℓ`, so `ρ² = 3 ℂᵠ / (4 ℂʷ₁ Pr c²)`: about 0.05–0.15 for `c = 3`–`6` and `Pr ≈ 0.7`–`1`, the
observed range. A neutral `ρ²` far below that means `Sᶜ⁰ Sᴰ⁰` (or `c = (Sᵘ⁰ Sᴰ⁰)^{-1/2}`) of the
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
$(TYPEDEF)

As [`DissipationTimeScale`](@ref) without the stability function of the dissipation:
`τᵠ = ℓ / (ℂᵠ √e)`, `τ = 2 τᵠ`, so that `ρ² = 3 ℂᵠ Sᶜ(Ri) / (4 ℂᵛ ℂʷ₁)` and
`ℂᵠ < 4 ℂᵛ ℂʷ₁ (1 - ℂᵍ₁) / (3 max Sᶜ)`.

$(TYPEDFIELDS)
"""
struct EddyTurnoverTimeScale{FT, T3}
    "Eddy turnovers `ℓ / √e` per scalar-variance dissipation time, `τᵠ = ℓ / (ℂᵠ √e)`"
    ℂᵠ :: FT
    "Time scale of the third-moment diagnosis"
    third_moment_time_scale :: T3
end

const VarianceBudgetTimeScale = Union{DissipationTimeScale, EddyTurnoverTimeScale}

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`DissipationTimeScale`](@ref); the third moment keeps `StratificationLimitedMomentTimeScale(FT)`.
"""
DissipationTimeScale(FT::DataType = Oceananigans.defaults.FloatType; ℂᵠ = 2,
                     third_moment_time_scale = StratificationLimitedMomentTimeScale(FT)) =
    DissipationTimeScale(convert(FT, ℂᵠ), third_moment_time_scale)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return an [`EddyTurnoverTimeScale`](@ref); the third moment keeps `StratificationLimitedMomentTimeScale(FT)`.
"""
EddyTurnoverTimeScale(FT::DataType = Oceananigans.defaults.FloatType; ℂᵠ = 1,
                      third_moment_time_scale = StratificationLimitedMomentTimeScale(FT)) =
    EddyTurnoverTimeScale(convert(FT, ℂᵠ), third_moment_time_scale)

for TS in (:DissipationTimeScale, :EddyTurnoverTimeScale)
    @eval begin
        Adapt.adapt_structure(to, ts::$TS) = $TS(adapt(to, ts.ℂᵠ), adapt(to, ts.third_moment_time_scale))
        convert_eltype(::Type{FT}, ts::$TS) where FT = $TS(convert(FT, ts.ℂᵠ), convert_eltype(FT, ts.third_moment_time_scale))
        Base.summary(ts::$TS) = string($(string(TS)), "(ℂᵠ = ", prettysummary(ts.ℂᵠ), ", third moment: ",
                                       summary(ts.third_moment_time_scale), ")")
    end
end

# The time scale of the second moments at face k, from the closure's ℓ, √e (u), N² and Ri there.
# `e` and `ℓᵐ` (the specific TKE and the stored mixing length) are those the closure's shear reads,
# so that `Sᴰ` sees the same Richardson number as the diffusivities (`TurbulenceAugmentedShear`).
@inline moment_time_scaleᶜᶜᶠ(i, j, k, grid, ts, closure, velocities, N², e, ℓᵐ, ℓ, u) =
    moment_time_scale(ts, ℓ, u, @inbounds N²[i, j, k])

@inline moment_time_scaleᶜᶜᶠ(i, j, k, grid, ts::EddyTurnoverTimeScale, closure, velocities, N², e, ℓᵐ, ℓ, u) =
    2 * ℓ / (ts.ℂᵠ * u)

@inline function moment_time_scaleᶜᶜᶠ(i, j, k, grid, ts::DissipationTimeScale, closure, velocities, N², e, ℓᵐ, ℓ, u)
    Sᴰ = dissipation_stability_functionᶜᶜᶠ(i, j, k, grid, closure, velocities, N², e, ℓᵐ)
    return 2 * ℓ / (ts.ℂᵠ * Sᴰ * u)
end

# The time scale of the third-moment diagnosis: the stored second-moment time scale, or its own
@inline third_moment_time_scale(::StratificationLimitedMomentTimeScale, τ, ℓ, u, N²) = τ
@inline third_moment_time_scale(ts::VarianceBudgetTimeScale, τ, ℓ, u, N²) =
    moment_time_scale(ts.third_moment_time_scale, ℓ, u, N²)

#####
##### Part 1: the subgrid distribution
#####

"""
$(TYPEDEF)

The subgrid distribution of θˡ, qᵗ and w of a [`SubgridCloudFormulation`](@ref): the
components of `SimplifiedHigherOrderClosure` for the second moments, the third moment and its
limiter, the assumed PDF and its saturation, evaluated on the mixing length, turbulent kinetic
energy, diffusivities and static stability of [`TKEBasedTurbulenceClosure`](@ref).

$(TYPEDFIELDS)
"""
struct SubgridDistribution{TS, SM, TM, SB, SK, PDF, PS, CP, C, R}
    "Time scale of the moments ([`StratificationLimitedMomentTimeScale`](@ref), [`DissipationTimeScale`](@ref) or [`EddyTurnoverTimeScale`](@ref))"
    time_scale :: TS
    "Second moments ([`DowngradientSecondMoments`](@ref)); `ℂᶠ`… of the surface are not used"
    second_moments :: SM
    "Third moment of vertical velocity ([`AlgebraicThirdMoment`](@ref)), or `nothing` for none (unskewed)"
    third_moment :: TM
    "Bound of the stratification number of the third-moment diagnosis"
    stratification_number_bound :: SB
    "Limiter of the third moment"
    skewness :: SK
    "Assumed PDF ([`DoubleGaussianPDF`](@ref) or [`SmoothDoubleGaussianPDF`](@ref))"
    assumed_pdf :: PDF
    "Saturation of the assumed PDF: [`ModelThermodynamicsSaturation`](@ref), the thermodynamics of the model"
    saturation :: PS
    "Clipping policy of the SHOC components"
    clipping :: CP
    "SHOC's constants the components read (`ggr`, `basetemp`, `mintke`, …)"
    constants :: C
    "Maximum flux correlation of w with θˡ and with qᵗ at the faces, |w′a′| / √(w′² a′²) ≲ ℂʳ, or `nothing` for no bound"
    ℂʳ :: R
end

Adapt.@adapt_structure SubgridDistribution

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SubgridDistribution`](@ref). The defaults are the components of the fixed smooth SHOC
configuration with published coefficients, except the largest skewness of the third-moment
limiter, 0.85 in place of SHOC's 1.2 √2 (hand-tuned against excess cumulus cover on the
master-comparison cases).

`ℂʳ` is the maximum flux correlation: where the down-gradient variance ``a′²`` of ``a = θˡ`` or
``qᵗ`` is too small for the down-gradient flux, ``(w′a′)² > (ℂʳ)² w′² a′²``, the variance is raised
smoothly to ``(w′a′)² / ((ℂʳ)² w′²)`` (see `bound_flux_correlation`), and ``θˡ′qᵗ′`` is scaled with
the variances. It must satisfy ``0 < (ℂʳ)² < 1 - ℂᵍ₁`` (≈ 0.6 with ``ℂᵍ₁ = 0.4`` of
[`DoubleGaussianPDF`](@ref)); above that the plumes of the double-Gaussian PDF have no width. SHOC's
own diffusivity, ``0.1 τ e``, corresponds to ``(ℂʳ)² = 0.15``. The default, `nothing`, imposes no bound.

`time_scale` is [`StratificationLimitedMomentTimeScale`](@ref) (the default, SHOC's),
[`DissipationTimeScale`](@ref) or [`EddyTurnoverTimeScale`](@ref). The latter two derive `τ` from
the scalar-variance budget with the same `ℓ`, `√e` and `Ri` as `Kᶜ`, so that the squared
correlation of `w` with θˡ and qᵗ is a function of `Ri` alone, bounded by a bound on `ℂᵠ`.
"""
function SubgridDistribution(FT::DataType = Oceananigans.defaults.FloatType;
                             time_scale = StratificationLimitedMomentTimeScale(FT),
                             second_moments = DowngradientSecondMoments(FT),
                             third_moment = AlgebraicThirdMoment(FT),
                             stratification_number_bound = RationalSaturatedStratificationNumber(FT),
                             skewness = PolynomialSaturatedSkewness(FT; maximum_skewness = 0.85),
                             assumed_pdf = DoubleGaussianPDF(FT),
                             saturation = ModelThermodynamicsSaturation(),
                             clipping = HardClipping(),
                             constants = SHOCConstants(FT),
                             ℂʳ = nothing)
    validate_distribution_saturation(saturation)
    ℂʳ = isnothing(ℂʳ) ? nothing : convert(FT, ℂʳ)
    return SubgridDistribution(time_scale, second_moments, third_moment, stratification_number_bound,
                               skewness, assumed_pdf, saturation, clipping, constants, ℂʳ)
end

# The distribution of this closure saturates with the thermodynamics of the model that holds it.
# The fixed constants of shoc.F90 (liquid-only Murphy–Koop saturation, fixed `lcond`, `cp`, `rgas`,
# `basetemp`) belong to `SimplifiedHigherOrderClosure`, which reproduces the Fortran.
validate_distribution_saturation(saturation) = nothing
validate_distribution_saturation(::FixedConstantsLiquidSaturation) =
    throw(ArgumentError("The SubgridDistribution of TKEBasedTurbulenceClosure saturates with the model's " *
                        "thermodynamics (ModelThermodynamicsSaturation); FixedConstantsLiquidSaturation is " *
                        "SimplifiedHigherOrderClosure's."))

convert_eltype(::Type{FT}, d::SubgridDistribution) where FT =
    SubgridDistribution(convert_eltype(FT, d.time_scale), map(c -> convert_component(FT, c),
                        (d.second_moments, d.third_moment, d.stratification_number_bound, d.skewness,
                         d.assumed_pdf, d.saturation, d.clipping, d.constants, d.ℂʳ))...)

convert_component(FT, ::Nothing) = nothing
convert_component(FT, c) = convert_eltype(FT, c)
convert_component(FT, c::Number) = convert(FT, c)

Base.summary(d::SubgridDistribution) = string("SubgridDistribution(", nameof(typeof(d.assumed_pdf)), ", ",
                                              nameof(typeof(d.saturation)), ", third moment: ",
                                              d.third_moment === nothing ? "none" : nameof(typeof(d.third_moment)), ")")

#####
##### Part 2: the cloud buoyancy flux
#####

"""
$(TYPEDEF)

The energy conversions of the TKE equation from the fluxes the closure applies to the mean state.
Every scalar the closure mixes (θˡⁱ, vapor, cloud liquid, rain, ice, ...) is diffused with `Kᶜ`
and momentum with `Kᵘ`; the buoyancy production is the exact discrete conversion of those fluxes,

    Bᶠ = -Kᶜ Σₛ βₛᶠ δz φₛ,    βₛ = ∂b/∂φₛ,

with `b` the buoyancy of the dynamics and its exact derivatives with respect to θˡⁱ and the vapor,
liquid and ice mass fractions (all liquid and all ice, so that the latent heat inside θˡⁱ and the
loading of every condensate species are counted), and the shear production is `Kᵘ S²` at the faces.
Both are moved from the faces to the centers with half-cell, density-weighted volumes, so that
`Σ ρ Δz B` is the change of the dynamic enthalpy and `Σ ρ Δz P` the loss of mean kinetic energy
that the fluxes cause. The subgrid distribution does not enter them. The positive part of `B` enters
the stage tendency; the negative part enters the implicit linear coefficient, with `e` floored only
at a tiny value so that the implicit sink removes exactly the energy the mean state gains wherever
`e > 0`. With `dissipative_heating = true` (the default) the dissipation `ε` that the TKE equation
removes heats the air, `ρ ĉᵖ Π ∂t θˡⁱ += ρ ε`. The surface stress work and the surface buoyancy
flux are converted within the bottom half-cell (no external TKE surface flux is needed). Under the
vertically implicit time discretization the stage tendency carries the conversions of the stage
state; after the implicit solves of each SSP Runge–Kutta stage the closure corrects the TKE to the
conversions of the solved state (including the dissipation of backward Euler itself) and heats the
air by what the implicit sinks removed from the solved `e`, so the energy budget of the stage closes
to round-off. Where `e ≈ 0` cannot pay what the solved fluxes converted (the diffusivities are
floored at `minimum_tke`), the shortfall is taken from the internal energy instead of driving `e`
negative. The acoustic Runge–Kutta time stepper does not call these hooks.

`stability = true` is a shortcut: [`TKEBasedTurbulenceClosure`](@ref) then replaces its
`static_stability` by `SubgridCloudStaticStability(LiquidFluxFraction(static_stability))`. The
static stability slot stays the only owner of `N²`; its cloud model [`LiquidFluxFraction`](@ref)
gives, at the interior faces,

    N²ᵇ = (g / θ) (∂z θˡ + A_q ∂z qᵗ) - Bˡ / Kᶜ,    Bˡ = (g / θ) A_c w′qˡ′,

the stratification of the conserved variables less the part carried by the subgrid liquid flux,
so that `-Kᶜ N²ᵇ` is the buoyancy flux of the distribution (not the buoyancy production of the
TKE equation, which is the conversion of the applied fluxes); `Bˡ / Kᶜ` is left out where `Kᶜ` vanishes. In clear air it is the stratification of
θᵥ. This is `N²ᵇ = (1 - ℵ) N²ᶜ + ℵ N²ˢ` with the effective saturated fraction `ℵ = w′qˡ′ / w′s′`
(see [`LiquidFluxFraction`](@ref)). The mixing length and the diffusivities are then formed again
from it. The passes, the relaxation and the solver below are those of the closure update that
solves this coupling; they are held here, not by the static stability.

The distribution is first computed with the static stability of the closure (`N²`; with
`DryStaticStability` the dry one) and the mixing length and diffusivities formed from it, while
`N²ᵇ` reads the liquid flux of that distribution, so the two are coupled. With
`stability_iterations = n > 0` the coupling is iterated `n` more times: each iteration computes
the distribution again with the stored `N²`, mixing length and `Kᶜ` (those of `N²ᵇ` from the pass
before, which also set the moment time scale), then `N²ᵇ` from its liquid flux, then the mixing
length and the diffusivities from `N²ᵇ`. The number is a type parameter, so the passes are
unrolled when the closure is compiled.

In the extra passes the new stability is blended into the stored one with
`stability_relaxation = ω` (`0 < ω ≤ 1`): `N² ← (1 - ω) N² + ω N²ᵇ` at the interior faces, before
the mixing length and the diffusivities are formed from it. The first pass (from the closure's
static stability to `N²ᵇ`) is never relaxed. `ω = 1` is the plain fixed-point iteration; `ω < 1`
damps the pass-to-pass oscillation between the stable and unstable branches.

`stability_solver` chooses how the extra passes update `N²` at each interior face, treating the
coupling as the scalar root `r(N²) = N²ᵇ(N²) - N² = 0` of the face (the coupling to the
neighbouring faces through the mixing length and the interpolated moments is lagged):

- [`FixedPointIteration`](@ref) (default): `N² ← (1 - ω) N² + ω N²ᵇ`, as above.
- [`SecantIteration`](@ref): a secant step from the last two passes, falling back to the relaxed
  step where the secant denominator is tiny, and kept inside the bracket below.
- [`BisectionIteration`](@ref): halves the bracket each pass by the sign of `r`; with
  `final_interpolation = true` the last update is the secant step of the last two passes, kept in
  the bracket, so that the result varies continuously with the parameters.

The bracket of the secant and the bisection is `[min(N²ᶜ, N²ˢ), max(N²ᶜ, N²ˢ)]` with
`N²ᶜ = (g / θ)(∂z θˡ + A_q ∂z qᵗ)`, the stratification of the conserved variables, and
`N²ˢ = N²ᶜ + (g / θ) A_c ∂z s̄`, the saturated one (`s = c_q qᵗ - c_θ θˡ`, SHOC's linearized
saturation excess at the level): where `w′qˡ′ ≈ C w′s′` and `w′s′ = -Kᶜ ∂z s̄`,
`N²ᵇ ≈ (1 - C) N²ᶜ + C N²ˢ` lies in it for any `0 ≤ C ≤ 1`. The first pass (from the closure's
static stability) is the same for all three; the secant and the bisection then start from it and
from the closure's static stability.

After the last pass the stored moments and cloud fields are those of the last distribution; `N²` is the relaxed blend of the stored `N²` and the `N²ᵇ` of that
distribution's liquid flux (exactly that `N²ᵇ` when `ω = 1`; for the secant and the bisection,
the update of the solver from that `N²ᵇ`), and `ℓᵐ`, `Kᶜ`, `Kᵘ`, `Kᵉ` are
formed from that `N²`. The last distribution was itself computed with the diffusivities of the
pass before (one pass behind, as with `n = 0`, where they are the dry ones).

$(TYPEDFIELDS)
"""
struct CloudBuoyancyFlux{N, W, S}
    "Whether the closure's static stability is made `SubgridCloudStaticStability(LiquidFluxFraction(...))` (a constructor shortcut)"
    stability :: Bool
    "The weight `ω` of the new `N²ᵇ` in the extra passes, `0 < ω ≤ 1`"
    stability_relaxation :: W
    "The update of `N²` in the extra passes"
    stability_solver :: S
    "Whether the energy the TKE equation dissipates heats the air (`ρθ`)"
    dissipative_heating :: Bool
    CloudBuoyancyFlux{N}(stability::Bool, ω::W = 1, solver::S = FixedPointIteration(), heating::Bool = true) where {N, W, S} =
        new{N, W, S}(stability, ω, solver, heating)
end

"""
    FixedPointIteration()

The extra passes of [`CloudBuoyancyFlux`](@ref) set `N² ← (1 - ω) N² + ω N²ᵇ`.
"""
struct FixedPointIteration end

"""
    SecantIteration()

The extra passes of [`CloudBuoyancyFlux`](@ref) take a secant step on `r(N²) = N²ᵇ - N²` per face,
from the last two passes, kept in the bracket of `N²ᶜ` and `N²ˢ`; where the secant denominator is
tiny they take the relaxed step `N² + ω r`.
"""
struct SecantIteration end

"""
    BisectionIteration(; final_interpolation = true)

The extra passes of [`CloudBuoyancyFlux`](@ref) halve the bracket of `N²ᶜ` and `N²ˢ` per face by
the sign of `r(N²) = N²ᵇ - N²` (positive below the root). With `final_interpolation` the last pass
instead takes the secant step of the last two passes, kept in the bracket.
"""
struct BisectionIteration
    final_interpolation :: Bool
end

BisectionIteration(; final_interpolation = true) = BisectionIteration(final_interpolation)

"""
    LaggedStability()

The first distribution of [`CloudBuoyancyFlux`](@ref) is formed with the `N²` of the previous
evaluation of the closure fields (the previous stage, or the previous step) at the interior faces,
instead of the closure's static stability; the closure's static stability on the first
evaluation. The extra passes are then those of [`FixedPointIteration`](@ref). With
`stability_iterations = 0` this lags `N²ᵇ` by one evaluation, as SHOC lags `wthv_sec`.
"""
struct LaggedStability end

Base.summary(::LaggedStability) = "LaggedStability"
Base.summary(::FixedPointIteration) = "FixedPointIteration"
Base.summary(::SecantIteration) = "SecantIteration"
Base.summary(b::BisectionIteration) = string("BisectionIteration(final_interpolation = ", b.final_interpolation, ")")

CloudBuoyancyFlux(stability::Bool) = CloudBuoyancyFlux{0}(stability)

function CloudBuoyancyFlux(; stability = true, stability_iterations = 0, stability_relaxation = 1,
                           stability_solver = FixedPointIteration(), dissipative_heating = true)
    n = Int(stability_iterations)
    n >= 0 || throw(ArgumentError("stability_iterations must be ≥ 0, got $n."))
    (n == 0 || stability) || throw(ArgumentError("stability_iterations > 0 needs stability = true."))
    0 < stability_relaxation <= 1 || throw(ArgumentError("stability_relaxation must be in (0, 1], got $stability_relaxation."))
    return CloudBuoyancyFlux{n}(stability, stability_relaxation, stability_solver, dissipative_heating)
end

convert_eltype(::Type{FT}, b::CloudBuoyancyFlux{N}) where {FT, N} =
    CloudBuoyancyFlux{N}(b.stability, convert(FT, b.stability_relaxation), b.stability_solver, b.dissipative_heating)

"""
    stability_iterations(buoyancy_flux::CloudBuoyancyFlux)

The number of extra passes of the distribution and the cloud buoyancy stability (the type parameter).
"""
stability_iterations(::CloudBuoyancyFlux{N}) where N = N

Base.summary(b::CloudBuoyancyFlux) = string("CloudBuoyancyFlux(stability = ", b.stability,
                                            ", stability_iterations = ", stability_iterations(b),
                                            ", stability_relaxation = ", b.stability_relaxation,
                                            ", stability_solver = ", summary(b.stability_solver),
                                            ", dissipative_heating = ", b.dissipative_heating, ")")

#####
##### The cloud model of the static stability slot: the liquid flux fraction of the distribution
#####

"""
$(TYPEDEF)

The cloud model of [`SubgridCloudStaticStability`](@ref) given by the closure's own
[`SubgridDistribution`](@ref): the effective saturated fraction is the ratio of the subgrid liquid
flux to the flux of the saturation excess `s = c_q qᵗ - c_θ θˡ` (SHOC's linearization at the level),

    ℵ = w′qˡ′ / w′s′,    w′s′ = -Kᶜ ∂z s̄,

so that the static stability `N² = (1 - ℵ) N²ᶜ + ℵ N²ˢ`, with `N²ᶜ = (g / θ)(∂z θˡ + A_q ∂z qᵗ)` the
stratification of the conserved variables and `N²ˢ = N²ᶜ + (g / θ) A_c ∂z s̄` the saturated one, is
`N²ᶜ - Bˡ / Kᶜ`, `Bˡ = (g / θ) A_c w′qˡ′`: minus the buoyancy flux of the distribution divided by
`Kᶜ`. It is evaluated in that form (see [`CloudBuoyancyFlux`](@ref)). For the double-Gaussian
distribution `ℵ` lies in `[0, 1]` up to the interpolation of the fluxes to the faces.

`ℵ` depends on the distribution, which depends on `Kᶜ`, `ℓ` and the moment time scale, which
depend on `N²`: a fixed point. The closure update solves it after the diffusivities, with the
passes and the solver of the formulation's [`CloudBuoyancyFlux`](@ref). Before the distribution of
the update exists, the slot evaluates `initial_stability` (for example `DryStaticStability()`).

`CloudBuoyancyFlux(stability = true)` in the closure's [`SubgridCloudFormulation`](@ref) is a
shortcut for `static_stability = SubgridCloudStaticStability(LiquidFluxFraction(static_stability))`.

$(TYPEDFIELDS)
"""
struct LiquidFluxFraction{S}
    "Static stability of the first evaluation of an update, before the distribution exists"
    initial_stability :: S
end

LiquidFluxFraction() = LiquidFluxFraction(DryStaticStability())

Base.show(io::IO, f::LiquidFluxFraction) = print(io, "LiquidFluxFraction(", summary(f.initial_stability), ")")

const LiquidFluxFractionStability = SubgridCloudStaticStability{<:LiquidFluxFraction}

# Before the distribution of the update exists: the initial stability
@inline static_stabilityᶜᶜᶠ(i, j, k, grid, ss::LiquidFluxFractionStability, buoyancy, tracers, closure_fields) =
    static_stabilityᶜᶜᶠ(i, j, k, grid, ss.cloud_model.initial_stability, buoyancy, tracers, closure_fields)

#####
##### Part 3: cloud owned by the closure
#####

"""
$(TYPEDEF)

The closure owns the cloud of the model: with a microphysics whose subgrid choice is
`ClosureCloud`, the closure supplies `ClosureCloudStatistics` (the cloud fraction and liquid of
the subgrid distribution) and, once per time step after the last stage, sets the cloud liquid of
the model to the liquid of the distribution diagnosed from the state at the end of the step,
moving the difference to or from the vapor; the total water and the liquid-ice potential
temperature are unchanged. With any other microphysics it does nothing.
"""
struct ClosureOwnedCloud end
Base.summary(::ClosureOwnedCloud) = "ClosureOwnedCloud"

#####
##### The formulation
#####

"""
$(TYPEDEF)

SHOC's subgrid cloud scheme as a property of [`TKEBasedTurbulenceClosure`](@ref), in three
independently switchable parts (`nothing` switches a part off):

1. `distribution`: a [`SubgridDistribution`](@ref);
2. `buoyancy_flux`: a [`CloudBuoyancyFlux`](@ref), which needs the distribution;
3. `cloud`: [`ClosureOwnedCloud`](@ref), which needs the distribution and a microphysics with
   `subgrid = ClosureCloud(...)`.

$(TYPEDFIELDS)
"""
struct SubgridCloudFormulation{D, B, O}
    "The subgrid distribution, or `nothing`"
    distribution :: D
    "The cloud buoyancy flux in the TKE equation, or `nothing`"
    buoyancy_flux :: B
    "Cloud ownership, or `nothing`"
    cloud :: O
end

Adapt.@adapt_structure SubgridCloudFormulation

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SubgridCloudFormulation`](@ref), all three parts on by default.

The configuration that scored best on the 15 master-comparison cases (hand-tuned, not calibrated)
is this default with the closure keywords `static_stability = DryStaticStability()` and
`minimum_tke = 4e-4`:

```julia
TKEBasedTurbulenceClosure(; static_stability = DryStaticStability(), minimum_tke = 4e-4,
                          subgrid_cloud_formulation = SubgridCloudFormulation())
```
"""
function SubgridCloudFormulation(FT::DataType = Oceananigans.defaults.FloatType;
                                 distribution = SubgridDistribution(FT),
                                 buoyancy_flux = CloudBuoyancyFlux(),
                                 cloud = ClosureOwnedCloud())
    if isnothing(distribution) && !(isnothing(buoyancy_flux) && isnothing(cloud))
        throw(ArgumentError("The cloud buoyancy flux and cloud ownership read the subgrid distribution, " *
                            "but `distribution = nothing`."))
    end
    return SubgridCloudFormulation(distribution, buoyancy_flux, cloud)
end

convert_eltype(::Type{FT}, f::SubgridCloudFormulation) where FT =
    SubgridCloudFormulation(convert_component(FT, f.distribution), convert_component(FT, f.buoyancy_flux), f.cloud)

Base.summary(f::SubgridCloudFormulation) =
    string("SubgridCloudFormulation(", summary(f.distribution), ", ", summary(f.buoyancy_flux), ", ", summary(f.cloud), ")")

# The static stability of a closure with a subgrid cloud formulation: `CloudBuoyancyFlux(stability = true)`
# makes it the liquid flux fraction of the distribution
liquid_flux_fraction_stability(static_stability, formulation) = static_stability

function liquid_flux_fraction_stability(static_stability, formulation::SubgridCloudFormulation)
    bf = formulation.buoyancy_flux
    shortcut = bf isa CloudBuoyancyFlux && bf.stability && !(static_stability isa LiquidFluxFractionStability)
    return shortcut ? SubgridCloudStaticStability(LiquidFluxFraction(static_stability)) : static_stability
end

validate_static_stability(static_stability, formulation) = nothing

function validate_static_stability(::LiquidFluxFractionStability, formulation)
    formulation isa SubgridCloudFormulation && !isnothing(formulation.distribution) ||
        throw(ArgumentError("SubgridCloudStaticStability(LiquidFluxFraction(...)) needs a SubgridCloudFormulation " *
                            "with a distribution, whose liquid flux gives the weight."))
    return nothing
end

#####
##### Fields
#####

"""
$(TYPEDEF)

The fields of a [`SubgridCloudFormulation`](@ref) in the closure fields of
[`TKEBasedTurbulenceClosure`](@ref): the moments at faces (`τ`, `θˡ′²`, `qᵗ′²`, `θˡ′qᵗ′`,
`w′θˡ′`, `w′qᵗ′`, `w′³`) and the cloud at centers (`w′²`, cloud fraction `C`, liquid `qˡ`, its
variance `qˡ′²`, the fluxes `w′qˡ′` and `w′θᵥ′`, the part `Bˡ = (g / θ) A_c w′qˡ′` of its buoyancy
flux carried by the liquid flux, and, with the cloud buoyancy flux, the shear and buoyancy
production `P` and `B` of the TKE equation from the applied fluxes).
"""
struct SubgridCloudFields{BF, CO, M, C, S, P}
    buoyancy_flux :: BF
    cloud_ownership :: CO
    moments :: M
    cloud :: C
    solver_state :: S
    presolve :: P # every closure scalar just before its implicit solve (the stage energy budget), or `nothing`
end

SubgridCloudFields(bf, co, m, c) = SubgridCloudFields(bf, co, m, c, nothing, nothing)
SubgridCloudFields(bf, co, m, c, s) = SubgridCloudFields(bf, co, m, c, s, nothing)

Adapt.adapt_structure(to, f::SubgridCloudFields) =
    SubgridCloudFields(f.buoyancy_flux, f.cloud_ownership, adapt(to, f.moments), adapt(to, f.cloud), adapt(to, f.solver_state),
                       adapt(to, f.presolve))

build_subgrid_cloud_fields(grid, ::SubgridCloudFormulation{Nothing}, tracer_names) = nothing

function build_subgrid_cloud_fields(grid, f::SubgridCloudFormulation, tracer_names)
    moments = (τ = ZFaceField(grid), θˡ′² = ZFaceField(grid), qᵗ′² = ZFaceField(grid), θˡ′qᵗ′ = ZFaceField(grid),
               w′θˡ′ = ZFaceField(grid), w′qᵗ′ = ZFaceField(grid), w′³ = ZFaceField(grid))
    cloud = (w′² = CenterField(grid), C = CenterField(grid), qˡ = CenterField(grid), qˡ′² = CenterField(grid),
             w′qˡ′ = CenterField(grid), w′θᵥ′ = CenterField(grid), Bˡ = CenterField(grid),
             applied_flux_conversion_fields(grid, f.buoyancy_flux, tracer_names)...,
             gθ = CenterField(grid), gA = CenterField(grid))
    return SubgridCloudFields(f.buoyancy_flux, f.cloud, moments, cloud, build_solver_state(grid, f.buoyancy_flux),
                              build_presolve_fields(grid, f.buoyancy_flux, tracer_names))
end

# The production of the TKE equation from the applied fluxes and what the stage hooks need
# (applied_flux_energetics.jl): the stage conversions (with their surface parts), the buoyancy sink
# rate, the stored derivatives of the buoyancy and ĉᵖ Π, momentum before its solve, and the
# conversions of the solved state and the dissipation of the last stage (diagnostics)
applied_flux_conversion_fields(grid, buoyancy_flux, tracer_names) = (; B = CenterField(grid))

function applied_flux_conversion_fields(grid, ::CloudBuoyancyFlux, tracer_names)
    centers = NamedTuple(name => CenterField(grid) for name in (:B, :P, :Bˢ, :Pˢ, :ωᴮ, :βᶿ, :βᵛ, :βˡ, :βⁱ, :ĉᵖΠ, :Bᵖ, :Pᵖ, :ε, :given, :stage_βᶿ, :stage_βᵛ, :stage_βˡ, :stage_βⁱ, :stage_surface, :shortfall, :shortfall_ρθ))
    return merge(centers, (ρu⁻ = XFaceField(grid), ρv⁻ = YFaceField(grid)))
end

# Every closure scalar just before its solve (diagnostic: the stage's energy budget)
build_presolve_fields(grid, buoyancy_flux, tracer_names) = nothing
build_presolve_fields(grid, ::CloudBuoyancyFlux, tracer_names) = NamedTuple(name => CenterField(grid) for name in tracer_names)

# The last pass (N², r) and the bracket of the secant and the bisection, at the faces
build_solver_state(grid, buoyancy_flux) = nothing
build_solver_state(grid, ::CloudBuoyancyFlux{N, W, <:FixedPointIteration}) where {N, W} = nothing

function build_solver_state(grid, ::CloudBuoyancyFlux{N, W, <:LaggedStability}) where {N, W}
    N²₋ = ZFaceField(grid)
    fill!(parent(N²₋), NaN) # not yet evaluated
    return (; N²₋)
end
build_solver_state(grid, ::CloudBuoyancyFlux{N, W, <:Union{SecantIteration, BisectionIteration}}) where {N, W} =
    (x₋ = ZFaceField(grid), r₋ = ZFaceField(grid), lower = ZFaceField(grid), upper = ZFaceField(grid))

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

cloud_liquid_density(microphysical_fields) =
    hasproperty(microphysical_fields, :ρqᶜˡ) ? microphysical_fields.ρqᶜˡ : nothing

function distribution_potential_temperature_density(model)
    AtmosphereModels.thermodynamic_density_name(model.formulation) === :ρθ ||
        throw(ArgumentError("The subgrid distribution of TKEBasedTurbulenceClosure requires the liquid-ice " *
                            "potential temperature formulation, but the model formulation is $(summary(model.formulation))."))
    return Oceananigans.prognostic_fields(model.formulation).ρθ
end

# The state the distribution reads: θˡⁱ, qᵗ, e, w, p and the saturation thermodynamics. `e_density`
# is the density of the specific TKE `ρe` holds: `nothing` within `update_state!`, where the tracer
# holds its specific value, and `ρ` after the stages, where it holds its density.
function distribution_state(model, e_density)
    ρ = AtmosphereModels.total_density(model.dynamics)
    θ = SpecificValue(distribution_potential_temperature_density(model), ρ)
    qᵗ = SpecificTotalWater(model.moisture_density, cloud_liquid_density(model.microphysical_fields), ρ)
    e = SpecificValue(model.tracers[TKE_NAME], e_density)
    w = model.velocities.w
    p = AtmosphereModels.dynamics_pressure(model.dynamics)
    thermodynamics = shoc_saturation_thermodynamics(model)
    return (; θ, qᵗ, e, w, p, thermodynamics)
end

#####
##### The column computation of the distribution
#####

@inline subgrid_third_moment(::Nothing, d, c, args...) = zero(eltype(args[1]))
@inline subgrid_third_moment(tm, d, c, args...) = third_moment(tm, d.stratification_number_bound, c, args...)

# Raise θˡ′² and qᵗ′² smoothly to (w′a′)² / ((ℂʳ)² w′²) where the down-gradient variances are too
# small for the down-gradient fluxes (|correlation with w| > ℂʳ), and scale θˡ′qᵗ′ with them so that
# the θˡ-qᵗ correlation is unchanged. With `Kᶜ` and `τ` of this closure, Kᶜ / (τ w′²) (the squared
# correlation) is often above 1 - ℂᵍ₁ = 0.6, where the plumes of the double-Gaussian PDF have no
# width and its cloud fraction is all or nothing in each plume (0, a, 1 - a or 1); the bound needs
# 0 < (ℂʳ)² < 1 - ℂᵍ₁. SHOC's own diffusivity, 0.1 τ e, gives (ℂʳ)² = 0.15.
#
# The floor is a smooth maximum with continuous derivative and no branch on the state,
#
#     smooth_max(a, b) = (a + b + √((a - b)² + (δ b)²)) / 2,   δ = 0.1,
#
# so smooth_max(a, 0) = a exactly, smooth_max ≥ max(a, b), and the excess, δ b / 2 at a = b,
# decays as (δ b)² / (4 |a - b|) away from it. `floatmin` under the root keeps the derivative
# finite where a = b = 0.
@inline bound_flux_correlation(::Nothing, w′², θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′) = (θˡ′², qᵗ′², θˡ′qᵗ′)

@inline function variance_floor_smooth_max(a, b, FT)
    δb = b / 10 # δ = 0.1
    return (a + b + sqrt((a - b) * (a - b) + δb * δb + floatmin(FT))) / 2
end

@inline function bound_flux_correlation(ℂʳ, w′², θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′)
    FT = typeof(θˡ′²)
    d = ℂʳ * ℂʳ * w′²
    resolved = d > floatmin(FT)
    dₛ = ifelse(resolved, d, one(FT))
    θ² = variance_floor_smooth_max(θˡ′², ifelse(resolved, w′θˡ′ * w′θˡ′ / dₛ, zero(FT)), FT)
    q² = variance_floor_smooth_max(qᵗ′², ifelse(resolved, w′qᵗ′ * w′qᵗ′ / dₛ, zero(FT)), FT)
    fθ = ifelse(θˡ′² > 0, θ² / ifelse(θˡ′² > 0, θˡ′², one(FT)), one(FT))
    fq = ifelse(qᵗ′² > 0, q² / ifelse(qᵗ′² > 0, qᵗ′², one(FT)), one(FT))
    return θ², q², θˡ′qᵗ′ * sqrt(fθ * fq)
end

@inline function subgrid_distribution_column!(m, cl, i, j, grid, d, closure, closure_fields, θ, qᵗ, e, velocities, p, thermodynamics)
    w = velocities.w
    Nz = size(grid, 3)
    FT = eltype(grid)
    c = d.constants
    cp = d.clipping
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())

    # The gravitational acceleration of the model, in place of SHOC's `ggr`, for the third moment
    # and the stability of the liquid flux fraction
    g = thermodynamics.constants.gravitational_acceleration
    c³ = (; ggr = g)

    @inbounds begin
        # The second moments at the interior faces, down the gradient
        for k in 2:Nz
            Δz = Δzᶜᶜᶠ(i, j, k, grid)
            Kᶜ = closure_fields.Kᶜ[i, j, k]
            Kᵘ = closure_fields.Kᵘ[i, j, k]
            ℓ = closure_fields.ℓᵐ[i, j, k]
            u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
            τ = moment_time_scaleᶜᶜᶠ(i, j, k, grid, d.time_scale, closure, velocities, closure_fields.N², e, closure_fields.ℓᵐ, ℓ, u)
            τ = ifelse(isfinite(τ), τ, zero(FT))

            θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′, _, _, _ =
                interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / Δz, θ[i, j, k] - θ[i, j, k-1], qᵗ[i, j, k] - qᵗ[i, j, k-1],
                                         zero(FT), zero(FT), zero(FT))

            # The variances no smaller than the fluxes allow at the largest correlation with w
            w′²ᶠ = (vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ)) +
                    vertical_velocity_variance(sm, max(e[i, j, k-1], eᵐⁱⁿ))) / 2
            θˡ′², qᵗ′², θˡ′qᵗ′ = bound_flux_correlation(d.ℂʳ, w′²ᶠ, θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′)

            m.τ[i, j, k] = τ
            m.θˡ′²[i, j, k] = θˡ′²
            m.qᵗ′²[i, j, k] = qᵗ′²
            m.θˡ′qᵗ′[i, j, k] = θˡ′qᵗ′
            m.w′θˡ′[i, j, k] = w′θˡ′
            m.w′qᵗ′[i, j, k] = w′qᵗ′
        end

        # The surface face takes the moments of the face above (the closure leaves the surface
        # flux of TKE to the boundary condition, and its buoyancy production at the first level is
        # that of the face above, too); the top face has none
        k′ = min(2, Nz)
        m.τ[i, j, 1] = m.τ[i, j, k′]
        m.θˡ′²[i, j, 1] = m.θˡ′²[i, j, k′]
        m.qᵗ′²[i, j, 1] = m.qᵗ′²[i, j, k′]
        m.θˡ′qᵗ′[i, j, 1] = m.θˡ′qᵗ′[i, j, k′]
        m.w′θˡ′[i, j, 1] = m.w′θˡ′[i, j, k′]
        m.w′qᵗ′[i, j, 1] = m.w′qᵗ′[i, j, k′]
        m.τ[i, j, Nz+1] = 0
        m.θˡ′²[i, j, Nz+1] = 0
        m.qᵗ′²[i, j, Nz+1] = 0
        m.θˡ′qᵗ′[i, j, Nz+1] = 0
        m.w′θˡ′[i, j, Nz+1] = 0
        m.w′qᵗ′[i, j, Nz+1] = 0
        m.w′³[i, j, 1] = 0
        m.w′³[i, j, Nz+1] = 0

        # The vertical velocity variance at the centers
        for k in 1:Nz
            cl.w′²[i, j, k] = vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ))
        end

        # The third moment at the interior faces
        w′²ᵐⁱⁿ = (convert(FT, 2) / 3) * c.mintke
        for k in 2:Nz
            w′²ᶠ = clip_below(cp, (cl.w′²[i, j, k] + cl.w′²[i, j, k-1]) / 2, w′²ᵐⁱⁿ, c.mintke)
            θᶠ = (θ[i, j, k] + θ[i, j, k-1]) / 2
            thedz = 1 / Δzᶜᶜᶠ(i, j, k, grid)
            thedz2 = 1 / (Δzᶜᶜᶜ(i, j, k-1, grid) + Δzᶜᶜᶜ(i, j, k, grid))
            τ³ = third_moment_time_scale(d.time_scale, m.τ[i, j, k], closure_fields.ℓᵐ[i, j, k],
                                         ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e), closure_fields.N²[i, j, k])
            τ³ = ifelse(isfinite(τ³), τ³, zero(FT))
            w′³ = subgrid_third_moment(d.third_moment, d, c³, τ³, closure_fields.N²[i, j, k], w′²ᶠ, θᶠ, thedz, thedz2,
                                       m.θˡ′²[i, j, k+1] - m.θˡ′²[i, j, k-1],
                                       m.w′θˡ′[i, j, k+1] - m.w′θˡ′[i, j, k-1],
                                       cl.w′²[i, j, k] - cl.w′²[i, j, k-1],
                                       max(e[i, j, k], eᵐⁱⁿ) - max(e[i, j, k-1], eᵐⁱⁿ),
                                       m.w′θˡ′[i, j, k])
            m.w′³[i, j, k] = limit_third_moment(d.skewness, cp, w′²ᶠ, w′³)
        end

        # The assumed PDF at the centers
        interface_moments(k) = (m.w′³[i, j, k], m.θˡ′²[i, j, k], m.w′θˡ′[i, j, k],
                                m.θˡ′qᵗ′[i, j, k], m.w′qᵗ′[i, j, k], m.qᵗ′²[i, j, k])
        for k in 1:Nz
            zᶜ = znode(i, j, k, grid, ccc...)
            zˡ = znode(i, j, k, grid, ccf...)
            zᵘ = znode(i, j, k+1, grid, ccf...)
            wᶜ = (w[i, j, k] + w[i, j, k+1]) / 2
            θᵏ = θ[i, j, k]
            C, qˡ, qˡ′², w′qˡ′, w′θᵥ′ = assumed_pdf(d.assumed_pdf, d.saturation, thermodynamics, cp, c, zᵘ, zˡ, zᶜ,
                                                   p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], cl.w′²[i, j, k],
                                                   interface_moments(k), interface_moments(k+1))
            cl.C[i, j, k] = C
            cl.qˡ[i, j, k] = qˡ
            cl.qˡ′²[i, j, k] = qˡ′²
            cl.w′qˡ′[i, j, k] = w′qˡ′
            cl.w′θᵥ′[i, j, k] = w′θᵥ′
            A_q, A_c = buoyancy_flux_coefficients(d.saturation, thermodynamics, c, p[i, j, k], θᵏ, qᵗ[i, j, k], qˡ)
            # The stability uses the buoyancy parameter of the level, g / θˡⁱ. The buoyancy
            # production of the TKE equation is not this distribution's: it is the conversion of
            # the fluxes the closure applies (`compute_applied_flux_conversions!`).
            cl.gθ[i, j, k] = g / θᵏ
            cl.gA[i, j, k] = g / θᵏ * A_q
            cl.Bˡ[i, j, k] = g / θᵏ * A_c * w′qˡ′
        end
    end

    return nothing
end

@kernel function _compute_subgrid_distribution!(moments, cloud, grid, closure, closure_fields, θ, qᵗ, e, velocities, p, thermodynamics)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    d = closure_ij.subgrid_cloud_formulation.distribution
    subgrid_distribution_column!(moments, cloud, i, j, grid, d, closure_ij, closure_fields, θ, qᵗ, e, velocities, p, thermodynamics)
end

function compute_subgrid_distribution!(fields::SubgridCloudFields, closure_fields, closure, model, e_density)
    grid = model.grid
    s = distribution_state(model, e_density)
    launch!(grid.architecture, grid, :xy, _compute_subgrid_distribution!,
            fields.moments, fields.cloud, grid, closure, closure_fields, s.θ, s.qᵗ, s.e, model.velocities, s.p, s.thermodynamics)
    return nothing
end

#####
##### Coupling to the closure
#####

# With the lagged stability: the stability of the previous evaluation, the mixing length and the
# diffusivities from it, then the distribution and the passes; the final N² is kept for the next one
@kernel function _set_lagged_stability!(N², N²₋)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        previous = N²₋[i, j, k]
        N²[i, j, k] = ifelse((k > 1) & !isnan(previous), previous, N²[i, j, k])
    end
end

function compute_subgrid_cloud!(fields::SubgridCloudFields{<:CloudBuoyancyFlux{N, W, <:LaggedStability}},
                                closure_fields, closure, model, tracers, parameters) where {N, W}
    grid = model.grid
    arch = grid.architecture
    N²₋ = fields.solver_state.N²₋
    launch!(arch, grid, parameters, _set_lagged_stability!, closure_fields.N², N²₋)
    compute_mixing_length!(closure_fields.ℓᵐ, closure_fields.ℓˢ, grid, closure, tracers[TKE_NAME], closure_fields.N²)
    launch!(arch, grid, parameters, _compute_tke_closure_fields!, closure_fields, grid, closure, model.velocities, tracers)
    compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
    solve_static_stability!(closure.static_stability, fields.buoyancy_flux, fields, closure_fields, closure, model, tracers, parameters)
    parent(N²₋) .= parent(closure_fields.N²)
    return nothing
end

# Within `update_state!`, after the diffusivities and before the implicit coefficient
function compute_subgrid_cloud!(fields::SubgridCloudFields, closure_fields, closure, model, tracers, parameters)
    compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
    solve_static_stability!(closure.static_stability, fields.buoyancy_flux, fields, closure_fields, closure, model, tracers, parameters)
    return nothing
end

# The coupling of the static stability with the distribution is solved only when the static
# stability slot reads the distribution, with the solver of the cloud buoyancy flux
solve_static_stability!(static_stability, buoyancy_flux, args...) = nothing
solve_static_stability!(::LiquidFluxFractionStability, buoyancy_flux, args...) = cloud_buoyancy_stability!(buoyancy_flux, args...)

# Without a cloud buoyancy flux there is no solver: one evaluation of the slot from the distribution
cloud_buoyancy_stability!(::Nothing, fields, closure_fields, closure, model, tracers, parameters) =
    cloud_buoyancy_stability_pass!(fields, closure_fields, closure, model, tracers, parameters)

# N²ᵇ = (g / θ)(∂z θˡ + A_q ∂z qᵗ) - Bˡ / Kᶜ at the interior faces, then the mixing length and the
# diffusivities again from it
@inline relax_stability(N², N²ᵇ, ::Nothing) = N²ᵇ
@inline relax_stability(N², N²ᵇ, ω) = (1 - ω) * N² + ω * N²ᵇ

@kernel function _cloud_buoyancy_stability!(N², grid, θ, qᵗ, gθ, gA, Bˡ, Kᶜ, ω)
    i, j, k = @index(Global, NTuple)
    FT = eltype(grid)
    @inbounds begin
        interior_face = k > 1
        km = max(k - 1, 1)
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        ∂zθ = (θ[i, j, k] - θ[i, j, km]) / Δz
        ∂zq = (qᵗ[i, j, k] - qᵗ[i, j, km]) / Δz
        gθᶠ = (gθ[i, j, k] + gθ[i, j, km]) / 2
        gAᶠ = (gA[i, j, k] + gA[i, j, km]) / 2
        Bᶠ = (Bˡ[i, j, k] + Bˡ[i, j, km]) / 2
        Kᶜᵢ = Kᶜ[i, j, k]
        resolved = Kᶜᵢ > sqrt(floatmin(FT))
        N²ᵇ = gθᶠ * ∂zθ + gAᶠ * ∂zq - ifelse(resolved, Bᶠ / ifelse(resolved, Kᶜᵢ, one(FT)), zero(FT))
        N²[i, j, k] = ifelse(interior_face, relax_stability(N²[i, j, k], N²ᵇ, ω), N²[i, j, k])
    end
end

function cloud_buoyancy_stability!(buoyancy_flux::CloudBuoyancyFlux{N, W, <:Union{SecantIteration, BisectionIteration}},
                                   fields, closure_fields, closure, model, tracers, parameters) where {N, W}
    solver = buoyancy_flux.stability_solver
    ω = buoyancy_flux.stability_relaxation
    solve_stability_pass!(solver, Val(0), Val(N), ω, fields, closure_fields, closure, model, tracers, parameters)
    for p in 1:N
        compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
        solve_stability_pass!(solver, Val(p), Val(N), ω, fields, closure_fields, closure, model, tracers, parameters)
    end
    return nothing
end

function solve_stability_pass!(solver, ::Val{P}, ::Val{N}, ω, fields, closure_fields, closure, model, tracers, parameters) where {P, N}
    grid = model.grid
    arch = grid.architecture
    s = distribution_state(model, nothing)
    cl = fields.cloud
    d = closure.subgrid_cloud_formulation.distribution
    launch!(arch, grid, parameters, _solve_cloud_buoyancy_stability!, closure_fields.N², fields.solver_state, solver,
            P == 0, P == N, ω, grid, s.θ, s.qᵗ, s.p, s.thermodynamics, d.saturation, d.constants,
            cl.gθ, cl.gA, cl.Bˡ, closure_fields.Kᶜ)
    compute_mixing_length!(closure_fields.ℓᵐ, closure_fields.ℓˢ, grid, closure, tracers[TKE_NAME], closure_fields.N²)
    launch!(arch, grid, parameters, _compute_tke_closure_fields!, closure_fields, grid, closure, model.velocities, tracers)
    return nothing
end

# SHOC's linearization of the saturation excess at a level, s′ = c_q qᵗ′ - c_θ θˡ′, and A_c
@inline function saturation_excess_coefficients(sat, thermodynamics, c, θ, qᵗ, p)
    Tˡ = plume_liquid_water_temperature(sat, thermodynamics, c, θ, qᵗ, p)
    qˢ, β, cp_over_L, Π = plume_saturation(sat, thermodynamics, c, Tˡ, qᵗ, p)
    opbq = 1 + β * qˢ
    c_θ = ((1 + β * qᵗ) / (opbq * opbq)) * cp_over_L * β * qˢ * Π
    c_q = 1 / opbq
    _, A_c = buoyancy_flux_coefficients(sat, thermodynamics, c, p, θ, qᵗ, zero(θ))
    return c_q, c_θ, A_c
end

@inline secant_root(x, r, x₋, r₋, fallback) =
    (Δr = r - r₋; ok = abs(Δr) > sqrt(floatmin(typeof(x))) * (abs(r) + abs(r₋)) + floatmin(typeof(x));
     ifelse(ok, x - r * (x - x₋) / ifelse(ok, Δr, one(Δr)), fallback))

# The first pass: r at the closure's stability; the next point is N²ᵇ (secant) or the bracket's middle (bisection)
@inline function first_stability_update(::SecantIteration, x, r, G, L, U)
    return clamp(G, L, U), L, U
end

@inline function first_stability_update(::BisectionIteration, x, r, G, L, U)
    inside = (L < x) & (x < U)
    L′ = ifelse(inside & (r > 0), x, L)
    U′ = ifelse(inside & (r <= 0), x, U)
    return (L′ + U′) / 2, L′, U′
end

@inline function stability_update(::SecantIteration, last, ω, x, r, x₋, r₋, L, U)
    return clamp(secant_root(x, r, x₋, r₋, x + ω * r), L, U), L, U
end

@inline function stability_update(b::BisectionIteration, last, ω, x, r, x₋, r₋, L, U)
    L′ = ifelse(r > 0, max(L, x), L)
    U′ = ifelse(r > 0, U, min(U, x))
    middle = (L′ + U′) / 2
    interpolate = last & b.final_interpolation
    x′ = ifelse(interpolate, clamp(secant_root(x, r, x₋, r₋, middle), L′, U′), middle)
    return x′, L′, U′
end

@kernel function _solve_cloud_buoyancy_stability!(N², state, solver, first, last, ω, grid, θ, qᵗ, p, thermodynamics, sat, c,
                                                  gθ, gA, Bˡ, Kᶜ)
    i, j, k = @index(Global, NTuple)
    FT = eltype(grid)
    @inbounds begin
        interior_face = k > 1
        km = max(k - 1, 1)
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        ∂zθ = (θ[i, j, k] - θ[i, j, km]) / Δz
        ∂zq = (qᵗ[i, j, k] - qᵗ[i, j, km]) / Δz
        gθᶠ = (gθ[i, j, k] + gθ[i, j, km]) / 2
        gAᶠ = (gA[i, j, k] + gA[i, j, km]) / 2
        Bᶠ = (Bˡ[i, j, k] + Bˡ[i, j, km]) / 2
        Kᶜᵢ = Kᶜ[i, j, k]
        resolved = Kᶜᵢ > sqrt(floatmin(FT))
        N²ᶜ = gθᶠ * ∂zθ + gAᶠ * ∂zq
        G = N²ᶜ - ifelse(resolved, Bᶠ / ifelse(resolved, Kᶜᵢ, one(FT)), zero(FT))

        c_q⁺, c_θ⁺, A⁺ = saturation_excess_coefficients(sat, thermodynamics, c, θ[i, j, k], qᵗ[i, j, k], p[i, j, k])
        c_q⁻, c_θ⁻, A⁻ = saturation_excess_coefficients(sat, thermodynamics, c, θ[i, j, km], qᵗ[i, j, km], p[i, j, km])
        ∂zs = (c_q⁺ + c_q⁻) / 2 * ∂zq - (c_θ⁺ + c_θ⁻) / 2 * ∂zθ
        N²ˢ = N²ᶜ + gθᶠ * (A⁺ + A⁻) / 2 * ∂zs
        L = min(N²ᶜ, N²ˢ)
        U = max(N²ᶜ, N²ˢ)

        x = N²[i, j, k]
        r = G - x
        x₋ = state.x₋[i, j, k]
        r₋ = state.r₋[i, j, k]
        L₋ = ifelse(first, L, state.lower[i, j, k])
        U₋ = ifelse(first, U, state.upper[i, j, k])

        x₀, L₀, U₀ = first_stability_update(solver, x, r, G, L, U)
        xₙ, Lₙ, Uₙ = stability_update(solver, last, ω, x, r, x₋, r₋, L₋, U₋)
        x′ = ifelse(first, x₀, xₙ)

        state.x₋[i, j, k] = x
        state.r₋[i, j, k] = r
        state.lower[i, j, k] = ifelse(first, L₀, Lₙ)
        state.upper[i, j, k] = ifelse(first, U₀, Uₙ)
        N²[i, j, k] = ifelse(interior_face, x′, N²[i, j, k])
    end
end

function cloud_buoyancy_stability!(buoyancy_flux::CloudBuoyancyFlux, fields, closure_fields, closure, model, tracers, parameters)
    cloud_buoyancy_stability_pass!(fields, closure_fields, closure, model, tracers, parameters)
    # The extra passes of the coupling; the count is a type parameter, so the loop is unrolled when traced
    for _ in 1:stability_iterations(buoyancy_flux)
        compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
        cloud_buoyancy_stability_pass!(fields, closure_fields, closure, model, tracers, parameters,
                                       buoyancy_flux.stability_relaxation)
    end
    return nothing
end

# N²ᵇ from the stored distribution (blended into the stored N² with weight ω, unless ω = nothing), then
# the mixing length and the diffusivities from it
function cloud_buoyancy_stability_pass!(fields, closure_fields, closure, model, tracers, parameters, ω = nothing)
    grid = model.grid
    arch = grid.architecture
    s = distribution_state(model, nothing)
    cl = fields.cloud
    launch!(arch, grid, parameters, _cloud_buoyancy_stability!, closure_fields.N², grid, s.θ, s.qᵗ, cl.gθ, cl.gA, cl.Bˡ, closure_fields.Kᶜ, ω)
    compute_mixing_length!(closure_fields.ℓᵐ, closure_fields.ℓˢ, grid, closure, tracers[TKE_NAME], closure_fields.N²)
    launch!(arch, grid, parameters, _compute_tke_closure_fields!, closure_fields, grid, closure, model.velocities, tracers)
    return nothing
end

# The buoyancy production of the TKE equation: the conversion of the applied fluxes with the cloud buoyancy flux
@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, fields::SubgridCloudFields{<:CloudBuoyancyFlux}, closure_fields) =
    @inbounds fields.cloud.B[i, j, k]

@inline tke_buoyancy_productionᶜᶜᶜ(i, j, k, grid, ::SubgridCloudFields{Nothing}, closure_fields) =
    ℑbzᵃᵃᶜ(i, j, k, grid, buoyancy_productionᶜᶜᶠ, closure_fields.Kᶜ, closure_fields.N²)

#####
##### Part 3: the cloud of the model
#####

@inline AtmosphereModels.subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::ClosureCloud, ::FlavorOfTKEClosure, fields::TKEClosureFields) =
    closure_cloud_statistics(i, j, k, grid, fields.subgrid_cloud)

@inline closure_cloud_statistics(i, j, k, grid, fields) = ClosureCloudStatistics(zero(grid), zero(grid))

@inline closure_cloud_statistics(i, j, k, grid, fields::SubgridCloudFields{<:Any, ClosureOwnedCloud}) =
    @inbounds ClosureCloudStatistics(fields.cloud.C[i, j, k], fields.cloud.qˡ[i, j, k])

# The liquid of the distribution becomes the cloud liquid of the model, bounded by the water there is
@kernel function _set_closure_cloud_liquid!(ρqᵛ, ρqᶜˡ, ρ, qˡ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵢ = ρ[i, j, k]
        ρqᵗ = ρqᵛ[i, j, k] + ρqᶜˡ[i, j, k]
        qᵗ = ρqᵗ / ρᵢ
        qᶜˡ = min(max(0, qˡ[i, j, k]), max(0, qᵗ))
        ρqᶜˡ[i, j, k] = ρᵢ * qᶜˡ
        ρqᵛ[i, j, k] = ρqᵗ - ρᵢ * qᶜˡ
    end
end

function Oceananigans.TimeSteppers.step_closure_prognostics!(closure_fields::TKEClosureFields, closure::FlavorOfTKEClosure, model, Δt)
    step_subgrid_cloud!(closure_fields.subgrid_cloud, closure_fields, closure, model)
    return nothing
end

step_subgrid_cloud!(fields, closure_fields, closure, model) = nothing

function step_subgrid_cloud!(fields::SubgridCloudFields{<:Any, ClosureOwnedCloud}, closure_fields, closure, model)
    subgrid_correlation(model.microphysics) isa ClosureCloud || return nothing
    ρqᶜˡ = cloud_liquid_density(model.microphysical_fields)
    isnothing(ρqᶜˡ) && return nothing

    # The distribution of the state at the end of the step, with the diffusivities of its last stage
    ρ = AtmosphereModels.total_density(model.dynamics)
    compute_subgrid_distribution!(fields, closure_fields, closure, model, ρ)

    grid = model.grid
    launch!(grid.architecture, grid, :xyz, _set_closure_cloud_liquid!, model.moisture_density, ρqᶜˡ, ρ, fields.cloud.qˡ)
    return nothing
end
