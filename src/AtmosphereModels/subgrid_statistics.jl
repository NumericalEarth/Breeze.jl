#####
##### Subgrid statistics: the closure → microphysics interface
#####
#
# Every microphysical rate R(x) can be evaluated as the mean ⟨R⟩ = ∫ R(x) P(x | x̄, Σ) dx over a
# subgrid distribution whose mean x̄ is the resolved state and whose covariance Σ is supplied by
# the turbulence closure. The microphysics never reads closure fields: it receives an isbits
# `AbstractSubgridStatistics` built per cell by `subgrid_statisticsᶜᶜᶜ`. `DiracStatistics` (Σ = 0)
# selects today's grid-mean code path, unchanged.
#
# The statistics are those of the conserved pair (qᵗ, θˡⁱ): total (cloud) water and liquid-ice
# potential temperature. Microphysics schemes derive the saturation-deficit spread σₛ from them
# with their own thermodynamics.

"""
$(TYPEDEF)

Abstract supertype for the per-cell subgrid statistics that a turbulence closure supplies to the
microphysics. See [`DiracStatistics`](@ref) and [`GaussianStatistics`](@ref).
"""
abstract type AbstractSubgridStatistics{FT} end

"""
$(TYPEDEF)

Subgrid statistics with every variance zero: the microphysics evaluates its rates at the
grid-mean state, exactly as without a subgrid model.

```jldoctest
julia> using Breeze.AtmosphereModels: DiracStatistics

julia> DiracStatistics{Float64}()
DiracStatistics{Float64}()
```
"""
struct DiracStatistics{FT} <: AbstractSubgridStatistics{FT} end

"""
$(TYPEDEF)

Gaussian subgrid statistics of the conserved pair (qᵗ, θˡⁱ) in one cell.

# Fields
$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.AtmosphereModels: GaussianStatistics

julia> GaussianStatistics(1e-8, 0.01, -1e-5)
GaussianStatistics{Float64}(1.0e-8, 0.01, -1.0e-5)
```
"""
struct GaussianStatistics{FT} <: AbstractSubgridStatistics{FT}
    "total-water variance ⟨qᵗ'²⟩ [kg² kg⁻²]"
    σ²ᵗ :: FT
    "liquid-ice potential temperature variance ⟨θˡⁱ'²⟩ [K²]"
    σ²ᶿ :: FT
    "covariance ⟨qᵗ' θˡⁱ'⟩ [K kg kg⁻¹]"
    cᵗᶿ :: FT
end

GaussianStatistics{FT}() where FT = GaussianStatistics{FT}(zero(FT), zero(FT), zero(FT))

# Statistics contributed by different closures in a tuple add (independent contributions).
@inline merge_statistics(a::GaussianStatistics, b::GaussianStatistics) =
    GaussianStatistics(a.σ²ᵗ + b.σ²ᵗ, a.σ²ᶿ + b.σ²ᶿ, a.cᵗᶿ + b.cᵗᶿ)

"""
$(TYPEDEF)

Abstract supertype for the microphysics-side choice of subgrid distribution.
"""
abstract type AbstractSubgridCorrelation end

"""
$(TYPEDEF)

No subgrid distribution: [`subgrid_statisticsᶜᶜᶜ`](@ref) returns [`DiracStatistics`](@ref) and the
microphysics runs its grid-mean code unchanged.
"""
struct NoSubgridCorrelation <: AbstractSubgridCorrelation end

"""
$(TYPEDEF)

A Gaussian subgrid distribution of the conserved variables (Sommeria and Deardorff 1977;
Mellor 1977). Closures that do not prognose second moments supply them with the down-gradient
form ⟨φ'ψ'⟩ = (`variance_coefficient` ℓ² + `grid_variance_coefficient` Δz²) ∂zφ ∂zψ: the
mixing-length variance of Bechtold et al. (1995), with ℓ the closure's mixing length, plus the
variance of the resolved gradient within the cell (Δz²/12 for a linear profile), which matters
where ℓ collapses, as at an inversion, and vanishes as Δz → 0. The microphysics floors the saturation-deficit spread smoothly,
σₛ = √(σₛ² + `minimum_saturation_deficit_spread`²), which keeps the distribution non-degenerate
in the absence of turbulence.

# Fields
$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.AtmosphereModels: GaussianSubgridCorrelation

julia> GaussianSubgridCorrelation(Float64)
GaussianSubgridCorrelation{Float64}(1.0, 0.08333333333333333, 0.0001)
```
"""
struct GaussianSubgridCorrelation{FT} <: AbstractSubgridCorrelation
    "coefficient of the mixing-length (co)variances ℓ² ∂zφ ∂zψ"
    variance_coefficient :: FT
    "coefficient of the within-cell gradient (co)variances Δz² ∂zφ ∂zψ"
    grid_variance_coefficient :: FT
    "background saturation-deficit spread σₛ [kg kg⁻¹]"
    minimum_saturation_deficit_spread :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`GaussianSubgridCorrelation`](@ref) with float type `FT`.
"""
GaussianSubgridCorrelation(FT::DataType = Oceananigans.defaults.FloatType;
                           variance_coefficient = 1,
                           grid_variance_coefficient = 1/12,
                           minimum_saturation_deficit_spread = 1e-4) =
    GaussianSubgridCorrelation(convert(FT, variance_coefficient), convert(FT, grid_variance_coefficient),
                               convert(FT, minimum_saturation_deficit_spread))

"""
    subgrid_correlation(microphysics)

Return the subgrid distribution choice of `microphysics`; [`NoSubgridCorrelation`](@ref) by default.
"""
subgrid_correlation(microphysics) = NoSubgridCorrelation()

"""
    subgrid_statisticsᶜᶜᶜ(i, j, k, grid, correlation, closure, closure_fields, state)

Return the [`AbstractSubgridStatistics`](@ref) of cell `(i, j, k)` for the subgrid distribution
`correlation`, from the turbulence `closure` and its `closure_fields`. `state` is a `NamedTuple`
holding the specific conserved variables `θ` (liquid-ice potential temperature) and `qᵗ` (total
cloud water), each indexable at `[i, j, k]`. Closures that know their second moments add a method;
a closure without one contributes zero variance.
"""
@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::NoSubgridCorrelation, closure, closure_fields, state) =
    DiracStatistics{eltype(grid)}()

@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::AbstractSubgridCorrelation, closure, closure_fields, state) =
    GaussianStatistics{eltype(grid)}()

@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::NoSubgridCorrelation, ::Tuple, ::Tuple, state) =
    DiracStatistics{eltype(grid)}()

@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::AbstractSubgridCorrelation, ::Tuple{}, ::Tuple{}, state) =
    GaussianStatistics{eltype(grid)}()

@inline subgrid_statisticsᶜᶜᶜ(i, j, k, grid, correlation::AbstractSubgridCorrelation,
                              closures::Tuple, closure_fields::Tuple, state) =
    merge_statistics(subgrid_statisticsᶜᶜᶜ(i, j, k, grid, correlation, first(closures), first(closure_fields), state),
                     subgrid_statisticsᶜᶜᶜ(i, j, k, grid, correlation, Base.tail(closures), Base.tail(closure_fields), state))
