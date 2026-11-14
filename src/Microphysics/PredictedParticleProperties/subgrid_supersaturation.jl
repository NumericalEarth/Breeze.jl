#####
##### Condensation of cloud liquid: grid mean, or averaged over the subgrid supersaturation
#####
#
# P3 condenses and evaporates cloud liquid at a rate proportional to the vapor excess over
# liquid, δqᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ, with the coefficient ωˡ of `phase_relaxation_rate`. The `condensation`
# option of P3 decides which vapor excess:
#
#   * `GridMeanCondensation()`: the grid-mean δqᵛ⁺ˡ;
#   * `SubgridCondensation(statistics)`: δqᵛ⁺ˡ + Γ, the vapor excess averaged over the subgrid
#     distribution of δqᵛ⁺ˡ, truncated at zero where no condensate is present (see
#     `subgrid_vapor_excess_correlation`). `statistics` supplies the distribution through
#     `AtmosphereModels.supersaturation_statistics`.
#
# Γ enters the vapor exchange of cloud liquid only: its diffusional growth and decay, the seed
# mass with which prescribed-number droplets start that growth, and the saturation-adjustment
# caps on those two rates. Rain, ice, aerosol activation and every collection process keep
# their grid-mean form. Evaporation stays limited by the cloud liquid present (the clamp
# -qᶜˡ / τ in `coupled_saturation_adjustment_rates`).
#
# An object of any other type in the `condensation` slot is an external model of Γ (the former
# `subgrid` slot; Squall's `SupersaturationCorrelation`). It extends the three functions
# `subgrid_supersaturation_fields`, `subgrid_supersaturation_statistics` and
# `subgrid_supersaturation`. That path is deprecated: such a model should become a source of
# `supersaturation_statistics` used through `SubgridCondensation`.

"""
$(TYPEDEF)

Cloud liquid condenses and evaporates at the rate set by the grid-mean vapor excess over
liquid, ``ωˡ δqᵛ⁺ˡ`` (the default `condensation` of [`PredictedParticlePropertiesMicrophysics`](@ref)).

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: GridMeanCondensation

julia> GridMeanCondensation()
GridMeanCondensation()
```
"""
struct GridMeanCondensation end

"""
$(TYPEDEF)

Cloud liquid condenses and evaporates at the rate averaged over the subgrid distribution of the
vapor excess over liquid, ``ωˡ (μ + Γ)``, where ``μ = δqᵛ⁺ˡ`` is the grid-mean vapor excess and

```math
Γ = σ G(μ / σ) - μ ≥ 0, \\qquad G(Q) = φ(Q) + Q Φ(Q),
```

with ``φ`` and ``Φ`` the standard normal density and distribution function, so that
``μ + Γ = ⟨\\max(0, δqᵛ⁺ˡ)⟩``. The spread ``σ`` comes from
`supersaturation_statistics(statistics, ...)`. For a mixture of Gaussian plumes with weights
``aᵢ``, offsets ``μᵢ - Σⱼ aⱼ μⱼ`` and spreads ``σᵢ``, ``Γ = Σᵢ aᵢ [σᵢ G(μ̃ᵢ / σᵢ) - μ̃ᵢ]`` with
``μ̃ᵢ = μ + μᵢ - Σⱼ aⱼ μⱼ``. As ``σ → 0``, ``Γ → 0`` and the grid-mean rate is recovered.

$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation

julia> using Breeze.AtmosphereModels: PrescribedStatistics

julia> SubgridCondensation(PrescribedStatistics(1e-4))
SubgridCondensation{PrescribedStatistics{Float64}}(PrescribedStatistics{Float64}(0.0001))
```
"""
struct SubgridCondensation{S}
    "Source of the subgrid statistics of the vapor excess, passed to `supersaturation_statistics`"
    statistics :: S
end

Adapt.adapt_structure(to, condensation::SubgridCondensation) =
    SubgridCondensation(Adapt.adapt(to, condensation.statistics))

#####
##### The cell-averaged excess over zero of a Gaussian vapor excess
#####

# Γ(μ, σ) = σ G(μ/σ) - μ = σ [φ(Q) - Q Φ(-Q)], Q = μ/σ, written without the cancellation of
# σ G(Q) - μ for large Q. The spread is floored at a tiny value so that σ = 0 gives Γ = 0 and the
# rate is smooth in σ; `max(0, ⋅)` removes round-off below zero.
@inline function gaussian_excess_correlation(μ, σ)
    FT = typeof(μ)
    σ₊ = max(σ, floatmin(FT) / eps(FT))
    Q = μ / σ₊
    φ = exp(-Q^2 / 2) / sqrt(2 * FT(π))
    Φ⁻ = erfc(Q / sqrt(FT(2))) / 2
    return max(0, σ₊ * (φ - Q * Φ⁻))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid vapor excess correlation ``Γ ≥ 0`` [kg kg⁻¹] for the statistics `statistics`
(see `AtmosphereModels.supersaturation_statistics`) about the grid-mean vapor excess over liquid
`μ`, such that ``μ + Γ`` is the cell average of ``\\max(0, δqᵛ⁺ˡ)``. See [`SubgridCondensation`](@ref).

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: subgrid_vapor_excess_correlation

julia> subgrid_vapor_excess_correlation((; μ = 0.0, σ = 1.0, skewness = 0.0, f⁺ = 0.5), 0.0) ≈ 1 / sqrt(2π)
true
```
"""
@inline subgrid_vapor_excess_correlation(statistics::NamedTuple{names}, μ) where names =
    subgrid_vapor_excess_correlation(Val(:a ∈ names), statistics, μ)

# One Gaussian: only its spread is read.
@inline subgrid_vapor_excess_correlation(::Val{false}, statistics, μ) =
    gaussian_excess_correlation(μ, convert(typeof(μ), statistics.σ))

# Gaussian plumes, shifted so that their weighted mean is the grid-mean vapor excess μ.
@inline function subgrid_vapor_excess_correlation(::Val{true}, statistics, μ)
    FT = typeof(μ)
    a = map(x -> convert(FT, x), statistics.a)
    μᵖ = map(x -> convert(FT, x), statistics.μ)
    σᵖ = map(x -> convert(FT, x), statistics.σ)
    plume_mean = sum(map(*, a, μᵖ))
    terms = map((aᵢ, μᵢ, σᵢ) -> aᵢ * gaussian_excess_correlation(μ + μᵢ - plume_mean, σᵢ), a, μᵖ, σᵖ)
    return sum(terms)
end

#####
##### The three hook functions read by the P3 kernels
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return what the `condensation` option of P3 reads on the grid, as an object passed to the P3
kernels and then to [`subgrid_supersaturation_statistics`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation_statistics).
`nothing` for [`GridMeanCondensation`](@ref); the closure, its fields and
`supersaturation_statistics_model_fields(model)` for [`SubgridCondensation`](@ref).
"""
subgrid_supersaturation_fields(::GridMeanCondensation, model) = nothing
subgrid_supersaturation_fields(::AbstractSubgridCorrelation, model) = nothing

subgrid_supersaturation_fields(::SubgridCondensation, model) =
    (model.closure, model.closure_fields, AM.supersaturation_statistics_model_fields(model))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid statistics of cell `(i, j, k)` that the `condensation` option of P3 forms from
`fields`, the result of [`subgrid_supersaturation_fields`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation_fields).
`nothing` for [`GridMeanCondensation`](@ref).
"""
@inline subgrid_supersaturation_statistics(i, j, k, grid, ::GridMeanCondensation, fields) = nothing
@inline subgrid_supersaturation_statistics(i, j, k, grid, ::AbstractSubgridCorrelation, fields) = nothing

@inline subgrid_supersaturation_statistics(i, j, k, grid, condensation::SubgridCondensation, fields) =
    AM.supersaturation_statistics(condensation.statistics, i, j, k, grid, fields...)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid supersaturation correlation Γ [kg kg⁻¹] of a cell: the amount added to the
grid-mean vapor excess `qᵛ - qᵛ⁺ˡ` in the condensation and evaporation of cloud liquid. The
arguments are the vapor, saturation and cloud liquid mass fractions, the temperature `T`,
pressure `P` and density `ρ` that P3 evaluates its rates with, and the thermodynamic
`constants`. For [`GridMeanCondensation`](@ref) the result is `nothing` and P3 uses the grid-mean
vapor excess unchanged; for [`SubgridCondensation`](@ref) it is
[`subgrid_vapor_excess_correlation`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_vapor_excess_correlation).

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: subgrid_supersaturation, cloud_vapor_excess, GridMeanCondensation

julia> Γ = subgrid_supersaturation(GridMeanCondensation(), nothing, 1e-2, 1.1e-2, 0.0, 285.0, 9e4, 1.1, nothing)

julia> cloud_vapor_excess(Γ, -1e-3)
-0.001

julia> cloud_vapor_excess(0.5, -1.0)
-0.5
```
"""
@inline subgrid_supersaturation(::GridMeanCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) = nothing
@inline subgrid_supersaturation(::AbstractSubgridCorrelation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) = nothing

@inline subgrid_supersaturation(::SubgridCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) =
    subgrid_vapor_excess_correlation(statistics, qᵛ - qᵛ⁺ˡ)

# The vapor excess, and the vapor, that cloud liquid sees
@inline cloud_vapor_excess(::Nothing, vapor_excess) = vapor_excess
@inline cloud_vapor_excess(Γ, vapor_excess) = vapor_excess + Γ

# The seed mass of prescribed-number droplets starts cloud condensation, so it sees Γ; aerosol
# activation, which predicts the droplet number, keeps the grid-mean vapor.
@inline seed_vapor(aerosol, Γ, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, ::Nothing, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, Γ, qᵛ) = qᵛ + Γ

#####
##### The `condensation` keyword of P3
#####

validate_p3_condensation(::NoSubgridCorrelation) = GridMeanCondensation()
validate_p3_condensation(condensation::GridMeanCondensation) = condensation
validate_p3_condensation(condensation::SubgridCondensation) = condensation

# An external model of Γ (deprecated path)
function validate_p3_condensation(external_correlation)
    Base.depwarn("An external subgrid supersaturation model in the `condensation` slot of P3 " *
                 "(extending `subgrid_supersaturation`) is deprecated; provide its statistics through " *
                 "`AtmosphereModels.supersaturation_statistics` and use `SubgridCondensation(statistics)`.",
                 :PredictedParticlePropertiesMicrophysics)
    return external_correlation
end

#####
##### Γ in the P3 rates
#####

# The grid-mean and external paths: the hook above.
@inline cloud_condensation_correlation(condensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants, p3, state) =
    subgrid_supersaturation(condensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants)

# Subgrid condensation. Locally cloud liquid changes at ωˡ δqᵛ⁺ˡ, but evaporates no faster than
# the cloud liquid available allows, qᶜˡ / τ (τ the sink-limiting time scale). The local rate is
# therefore ωˡ max(δqᵛ⁺ˡ, -e) with e = qᶜˡ / (τ ωˡ), and its cell average is
#
#   ωˡ (μ + Γ),   Γ = ⟨max(0, δqᵛ⁺ˡ + e)⟩ - (μ + e),
#
# the Gaussian truncation of `subgrid_vapor_excess_correlation` about μ + e. Without cloud liquid
# (e = 0) only supersaturated air condenses; as σ → 0, μ + Γ → max(μ, -e), the grid-mean rate
# with its evaporation limit.
@inline function cloud_condensation_correlation(::SubgridCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ,
                                                T, P, ρ, constants, p3, state)
    FT = typeof(qᵛ)
    ωˡ = p3_cloud_phase_relaxation_rate(p3, qᶜˡ, qᵛ⁺ˡ, T, ρ, constants, state)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    e = max(0, qᶜˡ) / (τ * max(ωˡ, floatmin(FT)))
    μ = qᵛ - qᵛ⁺ˡ
    return subgrid_vapor_excess_correlation(statistics, μ + e)
end

# ωˡ = εˡ / ξˡ with the cloud coefficient and psychrometric correction of
# `p3_vapor_relaxation_coefficients`.
@inline function p3_cloud_phase_relaxation_rate(p3, qᶜˡ, qᵛ⁺ˡ, T, ρ, constants, state)
    FT = typeof(T)
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    ξˡ = 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
    εˡ = cloud_vapor_relaxation_coefficient(p3, qᶜˡ, ρ, state.Dᵛ, state.μᶜˡ, state.λᶜˡ, state.nᶜˡ)
    return εˡ / ξˡ
end
