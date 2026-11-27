#####
##### Subgrid supersaturation statistics and the phase relaxation rate
#####
#
# Two functions connect the microphysics and a turbulence closure without either reading the
# other's internals:
#
#   * `supersaturation_statistics(statistics, i, j, k, grid, closure, closure_fields, model_fields)`:
#     the subgrid distribution of the vapor excess over liquid, δqᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ, in one cell.
#     A microphysics that condenses over the subgrid distribution (P3 with
#     `SubgridCondensation(statistics)`) calls it. The `statistics` object selects the source:
#     `PrescribedStatistics` and `PrescribedMixtureStatistics` here; a turbulence closure may add
#     its own source type.
#
#   * `phase_relaxation_rate(i, j, k, grid, microphysics, microphysical_fields, 𝒰, constants)`:
#     ωᶜ, the rate [s⁻¹] at which the condensate converts vapor excess into condensate,
#     ∂ₜqᶜ = ωᶜ δqᵛ⁺. A turbulence closure calls it, for instance to weight its moist stability.

"""
$(TYPEDEF)

Abstract supertype for the sources of subgrid supersaturation statistics. Each subtype extends
[`supersaturation_statistics`](@ref).
"""
abstract type AbstractSupersaturationStatistics end

"""
    supersaturation_statistics(statistics, i, j, k, grid, closure, closure_fields, model_fields)

Return the subgrid distribution of the vapor excess over liquid, ``δqᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ``
[kg kg⁻¹], in cell `(i, j, k)`, from the source `statistics`. `closure` and `closure_fields`
are the model's turbulence closure and its fields; `model_fields` is the named tuple of
[`supersaturation_statistics_model_fields`](@ref).

# Contract

The result is an isbits `NamedTuple` in one of two forms:

1. A single Gaussian, `(; μ, σ, skewness, f⁺)`:
   - `μ`: mean of ``δqᵛ⁺ˡ`` as the source computes it [kg kg⁻¹];
   - `σ ≥ 0`: standard deviation of ``δqᵛ⁺ˡ`` [kg kg⁻¹];
   - `skewness`: skewness of ``δqᵛ⁺ˡ`` (dimensionless; informational);
   - `f⁺ ∈ [0, 1]`: supersaturated fraction of the cell.
2. A mixture of Gaussian components, `(; a, μ, σ)` or `(; a, μ, σ, f⁺)`, `a`, `μ` and `σ` each a
   tuple with one entry per component:
   - `a`: component weights, ``Σ aᵢ = 1``;
   - `μ`: component means ``μᵢ`` of ``δqᵛ⁺ˡ`` [kg kg⁻¹];
   - `σ`: within-component standard deviations ``σᵢ ≥ 0`` [kg kg⁻¹];
   - `f⁺` (optional): the saturated fraction, when the source holds its own (for instance the
     cloud fraction of a closure's distribution). Without it [`saturated_fraction`](@ref)
     forms ``Σᵢ aᵢ Φ(μᵢ / σᵢ)``.

[`saturated_fraction`](@ref) returns ``f⁺`` from either form; a turbulence closure reads it as
the saturated fraction of its static stability (`MicrophysicsSaturatedFraction`).

The microphysics uses its own grid-mean vapor excess ``μ̄`` as the mean of the
distribution: it reads only `σ` from the first form, and only the offsets ``μᵢ - Σⱼ aⱼ μⱼ``
and the `σᵢ` from the second. The source's means need therefore not agree exactly with the
microphysics' saturation. All entries must be finite; `σ` may be zero.
"""
function supersaturation_statistics end

"""
$(TYPEDEF)

A prescribed, spatially uniform standard deviation `σ` [kg kg⁻¹] of the vapor excess over liquid
(see [`supersaturation_statistics`](@ref)). Its mean is unknown to it, so it returns `μ = 0`,
`skewness = 0` and `f⁺ = 1/2` (a distribution centered on saturation). The microphysics reads only
`σ`; [`saturated_fraction`](@ref) returns that `f⁺`.

```jldoctest
julia> using Breeze.AtmosphereModels: PrescribedStatistics, supersaturation_statistics

julia> supersaturation_statistics(PrescribedStatistics(1e-4), 1, 1, 1, nothing, nothing, nothing, nothing)
(μ = 0.0, σ = 0.0001, skewness = 0.0, f⁺ = 0.5)
```
"""
struct PrescribedStatistics{FT} <: AbstractSupersaturationStatistics
    "Standard deviation of the vapor excess over liquid, kg kg⁻¹"
    σ :: FT
end

@inline function supersaturation_statistics(statistics::PrescribedStatistics, i, j, k, grid,
                                            closure, closure_fields, model_fields)
    σ = statistics.σ
    return (; μ = zero(σ), σ, skewness = zero(σ), f⁺ = one(σ) / 2)
end

"""
$(TYPEDEF)

A prescribed mixture of Gaussian components of the vapor excess over liquid: weights `a`, means `μ` and
within-component standard deviations `σ`, each a tuple with one entry per component (see
[`supersaturation_statistics`](@ref)).

```jldoctest
julia> using Breeze.AtmosphereModels: PrescribedMixtureStatistics, supersaturation_statistics

julia> s = PrescribedMixtureStatistics((0.3, 0.7), (6e-4, -2e-4), (1e-4, 2e-4));

julia> supersaturation_statistics(s, 1, 1, 1, nothing, nothing, nothing, nothing).a
(0.3, 0.7)
```
"""
struct PrescribedMixtureStatistics{A, M, S} <: AbstractSupersaturationStatistics
    "Component weights"
    a :: A
    "Component means of the vapor excess over liquid, kg kg⁻¹"
    μ :: M
    "Within-component standard deviations of the vapor excess over liquid, kg kg⁻¹"
    σ :: S
end

@inline supersaturation_statistics(statistics::PrescribedMixtureStatistics, i, j, k, grid,
                                   closure, closure_fields, model_fields) =
    (; a = statistics.a, μ = statistics.μ, σ = statistics.σ)

# The standard score (a - b) / σ of a Gaussian of spread σ, in a form that is regular at σ = 0:
#
#   z = (a - b) / √(σ² + δ²),   δ = ϵ (|a| + |b|) + floatmin,
#
# with ϵ the machine epsilon, so that δ is the round-off of the difference a - b itself (and the smallest normal number
# where both vanish). Where the spread is resolved, σ ≫ δ, it is (a - b) / σ to round-off. A spread below δ is the
# point mass: z = ±1/ϵ for any difference that is resolved from zero, whose distribution function is the step, and
# z = 0 where a = b. The derivatives are finite at σ = 0 (∂z/∂σ = 0 there). An infinite difference has an infinite
# score. The one treatment of a vanishing spread for the saturated fraction and for the partial moments of the
# microphysics.
@inline function standard_score(a, b, σ)
    FT = typeof(σ)
    x = a - b
    δ = eps(FT) * (abs(a) + abs(b)) + floatmin(FT)
    return ifelse(isfinite(x), x / hypot(σ, δ), x)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the saturated fraction ``f⁺ ∈ [0, 1]``, the probability that the vapor excess over liquid
is positive, from `statistics`, a result of [`supersaturation_statistics`](@ref): its `f⁺` when
it has one, otherwise, for a mixture of Gaussian components, ``Σᵢ aᵢ Φ(μᵢ / σᵢ)`` with ``Φ`` the
standard normal distribution function (a step where ``σᵢ = 0``).

```jldoctest
julia> using Breeze.AtmosphereModels: saturated_fraction

julia> saturated_fraction((; μ = 0.0, σ = 1e-4, skewness = 0.0, f⁺ = 0.3))
0.3

julia> saturated_fraction((; a = (0.5, 0.5), μ = (1e-3, -1e-3), σ = (0.0, 0.0)))
0.5
```
"""
@inline saturated_fraction(statistics::NamedTuple{names}) where names =
    saturated_fraction(Val(:f⁺ ∈ names), statistics)

@inline saturated_fraction(::Val{true}, statistics) = statistics.f⁺

@inline function saturated_fraction(::Val{false}, statistics)
    terms = map(statistics.a, statistics.μ, statistics.σ) do aᵢ, μᵢ, σᵢ
        FT = typeof(μᵢ / one(σᵢ))
        z = standard_score(convert(FT, μᵢ), zero(FT), convert(FT, σᵢ))
        aᵢ * erfc(-z / sqrt(FT(2))) / 2
    end
    return sum(terms)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The model fields passed to [`supersaturation_statistics`](@ref) as `model_fields`: the
velocities, the temperature, the microphysical fields and the moisture density.
"""
supersaturation_statistics_model_fields(model) =
    (; velocities = model.velocities,
       temperature = model.temperature,
       microphysical_fields = model.microphysical_fields,
       moisture_density = model.moisture_density)

#####
##### Phase relaxation rate
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return ``ωᶜ`` [s⁻¹], the rate at which the condensate of cell `(i, j, k)` converts vapor
excess into condensate, so that the condensation rate is ``ωᶜ δqᵛ⁺`` (with the latent-heat
feedback on the saturation already included). `𝒰` is the thermodynamic state of the cell and
`constants` the `ThermodynamicConstants`.

* No microphysics, or one without phase change (the default): ``ωᶜ = 0``.
* Saturation adjustment: ``ωᶜ = ∞``; see also [`instantaneous_phase_relaxation`](@ref).
* `PredictedParticlePropertiesMicrophysics`: ``ωᶜ = ωˡ + ωᵈⁱ + ωʷⁱ`` from the cloud droplets, the dry ice and
  the liquid coating of the ice; see its documentation.
"""
@inline phase_relaxation_rate(i, j, k, grid, microphysics, microphysical_fields, 𝒰, constants) =
    zero(eltype(grid))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `(; ωˡ, ωᵈⁱ, ωʷⁱ)` [s⁻¹]: the phase relaxation rate ``ωˡ`` of the condensate in the water that a subgrid
distribution distributes (the equilibrium condensate of `condensate_phase_equilibrium`: cloud liquid, or the
condensate of a saturation adjustment), and those of the condensate outside it, which fills the cell: ``ωᵈⁱ`` of
the ice, on which vapor deposits at saturation over ice (the dry ice of P3 and the ice of non-equilibrium cloud
formation), and ``ωʷⁱ`` of the liquid on ice, on which vapor condenses at saturation over liquid (P3's liquid coating
of ice). Their sum is [`phase_relaxation_rate`](@ref). The default is `ωˡ = phase_relaxation_rate(...)` and
`ωᵈⁱ = ωʷⁱ = 0`, a microphysics with nothing outside the distributed water.
"""
@inline function phase_relaxation_rates(i, j, k, grid, microphysics, microphysical_fields, 𝒰, constants)
    ωˡ = phase_relaxation_rate(i, j, k, grid, microphysics, microphysical_fields, 𝒰, constants)
    return (; ωˡ, ωᵈⁱ = zero(ωˡ), ωʷⁱ = zero(ωˡ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

`true` when `microphysics` removes vapor excess instantly (saturation adjustment), so that
[`phase_relaxation_rate`](@ref) is infinite. A caller that uses ``η(t ωᶜ)`` with
``η → 1`` as ``ωᶜ → ∞`` may set ``η = 1`` directly on this trait.
"""
instantaneous_phase_relaxation(microphysics) = false
