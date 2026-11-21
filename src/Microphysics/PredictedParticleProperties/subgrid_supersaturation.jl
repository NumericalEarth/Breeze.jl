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
# caps on those two rates. With `processes = AllProcesses()` the distribution also enters
# activation, ice nucleation, autoconversion and accretion (`subgrid_microphysics.jl`); otherwise
# rain, ice, aerosol activation and every collection process keep their grid-mean form. Evaporation stays limited by the cloud liquid present (the clamp
# -qᶜˡ / τ in `coupled_saturation_adjustment_rates`).

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

The distribution is that of the saturation excess of total water. Where cloud liquid is present
the truncation is taken about the grid-mean saturation excess ``s̄ = μ + ξˡ qᶜˡ`` instead of
``μ`` (``ξˡ`` the psychrometric correction), so that the rate is ``ωˡ (⟨s⁺⟩ - ξˡ qᶜˡ)`` and
vanishes when the cloud liquid is the distribution's equilibrium condensate.

$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation

julia> using Breeze.AtmosphereModels: PrescribedStatistics

julia> SubgridCondensation(PrescribedStatistics(1e-4))
SubgridCondensation{PrescribedStatistics{Float64}, CloudCondensationOnly}(PrescribedStatistics{Float64}(0.0001), CloudCondensationOnly())
```
"""
struct SubgridCondensation{S, P}
    "Source of the subgrid statistics of the vapor excess, passed to `supersaturation_statistics`"
    statistics :: S
    "Which processes read the distribution: `CloudCondensationOnly()` or `AllProcesses()`"
    processes :: P
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return [`SubgridCondensation`](@ref) with the subgrid statistics `statistics`. With
`processes = CloudCondensationOnly()` the distribution enters cloud condensation only; with
[`AllProcesses`](@ref) it also enters activation, ice nucleation, autoconversion and accretion.
"""
SubgridCondensation(statistics; processes = CloudCondensationOnly()) =
    SubgridCondensation(statistics, validate_subgrid_processes(processes))

Adapt.adapt_structure(to, condensation::SubgridCondensation) =
    SubgridCondensation(Adapt.adapt(to, condensation.statistics), condensation.processes)

#####
##### The cell-averaged excess over zero of a Gaussian vapor excess
#####

# Γ(μ, σ) = σ G(μ/σ) - μ = σ [φ(Q) - Q Φ(-Q)], Q = μ/σ, written without the cancellation of
# σ G(Q) - μ for large Q. The spread is floored at a tiny value so that σ = 0 gives Γ = 0 and the
# rate is smooth in σ; `max(0, ⋅)` removes round-off below zero.
#
# Beyond |Q| = 40 the Gaussian tail is below round-off in Float64 (and Float32), and Γ is its limit,
# max(0, -μ): 0 for Q > 40 and -μ for Q < -40. Q is clamped inside the formula, so that an
# overflowing Q (σ → 0, or a huge evaporation shift) never forms ∞ ⋅ 0.
@inline function gaussian_excess_correlation(μ, σ)
    FT = typeof(μ)
    σ₊ = max(σ, floatmin(FT) / eps(FT))
    Q = μ / σ₊
    Qᵐᵃˣ = convert(FT, 40)
    Qᶜ = clamp(Q, -Qᵐᵃˣ, Qᵐᵃˣ)
    φ = exp(-Qᶜ^2 / 2) / sqrt(2 * FT(π))
    Φ⁻ = erfc(Qᶜ / sqrt(FT(2))) / 2
    tail = ifelse(Q > Qᵐᵃˣ, zero(FT), -μ)
    return max(0, ifelse(abs(Q) > Qᵐᵃˣ, tail, σ₊ * (φ - Qᶜ * Φ⁻)))
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
##### What the P3 kernels read of the `condensation` option
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return what the `condensation` option of P3 reads on the grid, as an object passed to the P3
kernels and then to [`subgrid_supersaturation_statistics`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation_statistics).
`nothing` for [`GridMeanCondensation`](@ref); the closure, its fields and
`supersaturation_statistics_model_fields(model)` for [`SubgridCondensation`](@ref).
"""
subgrid_supersaturation_fields(::GridMeanCondensation, model) = nothing

subgrid_supersaturation_fields(::SubgridCondensation, model) =
    (model.closure, model.closure_fields, AM.supersaturation_statistics_model_fields(model))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid statistics of cell `(i, j, k)` that the `condensation` option of P3 forms from
`fields`, the result of [`subgrid_supersaturation_fields`](@ref Breeze.Microphysics.PredictedParticleProperties.subgrid_supersaturation_fields).
`nothing` for [`GridMeanCondensation`](@ref).
"""
@inline subgrid_supersaturation_statistics(i, j, k, grid, ::GridMeanCondensation, fields) = nothing

@inline subgrid_supersaturation_statistics(i, j, k, grid, condensation::SubgridCondensation, fields) =
    AM.supersaturation_statistics(condensation.statistics, i, j, k, grid, fields...)

# The vapor excess, and the vapor, that cloud liquid sees
@inline cloud_vapor_excess(::Nothing, vapor_excess) = vapor_excess
@inline cloud_vapor_excess(Γ, vapor_excess) = vapor_excess + Γ

# The seed mass of prescribed-number droplets starts cloud condensation, so it sees Γ; aerosol
# activation, which predicts the droplet number, keeps the grid-mean vapor.
@inline seed_vapor(aerosol, Γ, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, ::Nothing, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, Γ, qᵛ) = qᵛ + Γ
@inline seed_vapor(aerosol, Γ, qᵛ, ::Nothing) = seed_vapor(aerosol, Γ, qᵛ)
# With `AllProcesses` activation reads the distribution instead (`activation_vapor`)
@inline seed_vapor(aerosol, Γ, qᵛ, distribution) = qᵛ

#####
##### The `condensation` keyword of P3
#####

validate_p3_condensation(condensation::GridMeanCondensation) = condensation
validate_p3_condensation(condensation::SubgridCondensation) = condensation

validate_p3_condensation(condensation) =
    throw(ArgumentError("The `condensation` of P3 must be `GridMeanCondensation()` or " *
                        "`SubgridCondensation(statistics)`; got $(summary(condensation))."))

#####
##### Γ in the P3 rates
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid supersaturation correlation Γ [kg kg⁻¹] of a cell: the amount added to the
grid-mean vapor excess `qᵛ - qᵛ⁺ˡ` in the condensation and evaporation of cloud liquid. The
arguments are the `condensation` option of P3 and the `statistics` it formed for the cell, the
vapor, saturation and cloud liquid mass fractions, the temperature `T`, pressure `P` and density
`ρ` that P3 evaluates its rates with, and the thermodynamic `constants`. For
[`GridMeanCondensation`](@ref) the result is `nothing` and P3 uses the grid-mean vapor excess
unchanged.

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: cloud_condensation_correlation, cloud_vapor_excess, GridMeanCondensation

julia> Γ = cloud_condensation_correlation(GridMeanCondensation(), nothing, 1e-2, 1.1e-2, 0.0, 285.0, 9e4, 1.1, nothing)

julia> cloud_vapor_excess(Γ, -1e-3)
-0.001

julia> cloud_vapor_excess(0.5, -1.0)
-0.5
```
"""
@inline cloud_condensation_correlation(::GridMeanCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) = nothing

# Subgrid condensation. The statistics are those of the saturation excess of total water,
# s = qᵗ - qᵛ⁺ˡ(Tˡ), which in P3's linearization is s = δqᵛ⁺ˡ + ξˡ qᶜˡ: the vapor excess plus the
# vapor the cloud liquid would return, ξˡ = 1 + ℒˡ ∂qᵛ⁺ˡ/∂T / cᵖᵈ accounting for the cooling by
# its evaporation. Cloud liquid is taken to lie where s > 0, shaped as the equilibrium condensate
# s⁺ / ξˡ: locally qᶜˡ = λ s⁺ / ξˡ with λ = ξˡ q̄ᶜˡ / ⟨s⁺⟩, so that the local vapor excess is
# s - λ s⁺ where s > 0 and s, with no liquid to evaporate, elsewhere. Locally cloud liquid changes
# at ωˡ δqᵛ⁺ˡ, and the cell average is
#
#   ωˡ (1 - λ) ⟨s⁺⟩ = ωˡ (⟨s⁺⟩ - ξˡ q̄ᶜˡ) = ωˡ (μ + Γ),   Γ = ⟨s⁺⟩ - s̄,   s̄ = μ + ξˡ q̄ᶜˡ,
#
# the Gaussian truncation of `subgrid_vapor_excess_correlation` about the grid-mean saturation
# excess s̄ (not about μ: the positive part of the distribution is already condensed). The rate
# vanishes when the cloud liquid is the distribution's equilibrium condensate ⟨s⁺⟩ / ξˡ. Without
# cloud liquid s̄ = μ and only supersaturated air condenses. As σ → 0, μ + Γ → max(μ, -ξˡ q̄ᶜˡ):
# the grid-mean rate, with evaporation no faster than ωˡ ξˡ q̄ᶜˡ (all the liquid relaxing at
# the droplets' rate εˡ = ωˡ ξˡ).
@inline function cloud_condensation_correlation(condensation::SubgridCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ,
                                                T, P, ρ, constants)
    ξˡ = p3_liquid_psychrometric_correction(qᵛ⁺ˡ, T, constants)
    μ = qᵛ - qᵛ⁺ˡ
    s̄ = μ + ξˡ * max(0, qᶜˡ)
    cell = (; vapor_excess = μ, saturation_excess = s̄, ξˡ, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ)
    return vapor_excess_correlation(condensation.statistics, statistics, cell)
end

"""
    vapor_excess_correlation(source, statistics, cell)

Return the subgrid vapor excess correlation ``Γ`` [kg kg⁻¹] of [`SubgridCondensation`](@ref):
cloud liquid condenses at ``ωˡ (δqᵛ⁺ˡ + Γ)`` and relaxes implicitly toward
``\\max(0, s̄ + Γ) / ξˡ``. `source` is the `statistics` of `SubgridCondensation(statistics)`,
`statistics` what `supersaturation_statistics(source, ...)` returned for the cell, and `cell` the
named tuple `(; vapor_excess, saturation_excess, ξˡ, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ)`: the grid-mean vapor
excess over liquid ``μ = qᵛ - qᵛ⁺ˡ``, the saturation excess of total water ``s̄ = μ + ξˡ qᶜˡ``,
the psychrometric correction, the vapor, saturation and cloud liquid mass fractions, and the
temperature, pressure and density P3 evaluates its rates with.

The default is the truncation of a Gaussian (or a mixture of Gaussian plumes) about ``s̄``,
`subgrid_vapor_excess_correlation(statistics, s̄)`. A source of statistics with a different
distribution extends this function for its type to supply its own ``Γ``; the method must be
pure and type-stable in the element type of `cell`.

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: vapor_excess_correlation

julia> using Breeze.AtmosphereModels: PrescribedStatistics

julia> cell = (; vapor_excess = 0.0, saturation_excess = 0.0, ξˡ = 1.0, qᵛ = 1e-2, qᵛ⁺ˡ = 1e-2,
               qᶜˡ = 0.0, T = 285.0, P = 9e4, ρ = 1.1);

julia> vapor_excess_correlation(PrescribedStatistics(1.0), (; μ = 0.0, σ = 1.0, skewness = 0.0, f⁺ = 0.5), cell) ≈ 1 / sqrt(2π)
true
```
"""
@inline vapor_excess_correlation(source, statistics, cell) =
    subgrid_vapor_excess_correlation(statistics, cell.saturation_excess)

# ξˡ = 1 + ℒˡ ∂qᵛ⁺ˡ/∂T / cᵖᵈ, with P3's dry-air heat capacity convention
@inline function p3_liquid_psychrometric_correction(qᵛ⁺ˡ, T, constants)
    FT = typeof(T)
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    return 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
end
