#####
##### Subgrid statistics: the closure → microphysics interface
#####
#
# A microphysical rate R(x) is evaluated as the mean ⟨R⟩ = ∫ R(x) P(x | x̄, Σ) dx over a subgrid
# distribution whose mean x̄ is the resolved state and whose covariance Σ is supplied by the
# turbulence closure. The microphysics never reads closure fields: it receives an isbits
# `AbstractSubgridStatistics` built per cell by `subgrid_statisticsᶜᶜᶜ`, which closures extend.
# `DiracStatistics` (Σ = 0) selects the grid-mean code path, unchanged.
#
# The statistics are those of the conserved pair (qᵗ, θˡⁱ): non-precipitating total water and
# liquid-ice potential temperature. Microphysics schemes derive the saturation-deficit spread σₛ
# from them with their own thermodynamics.

"""
$(TYPEDEF)

Abstract supertype for the per-cell subgrid statistics that a turbulence closure supplies to the
microphysics. See [`DiracStatistics`](@ref), [`GaussianStatistics`](@ref) and
[`DoubleGaussianStatistics`](@ref).
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

"""
$(TYPEDEF)

Two-component (updraft and environment) Gaussian mixture of the conserved pair (qᵗ, θˡⁱ), as
supplied by an eddy-diffusivity mass-flux closure (Lewellen and Yoo 1993; Larson et al. 2002).
The updraft has weight `aᵘ` and mean offset (`δqᵗ`, `δθ`) from the cell mean; the environment has
weight `1 - aᵘ` and the offset that keeps the mixture mean equal to the cell mean,
`-aᵘ / (1 - aᵘ)` times the updraft offset.

# Fields
$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.AtmosphereModels: DoubleGaussianStatistics, GaussianStatistics

julia> 𝒮 = DoubleGaussianStatistics(0.1, 1e-3, -0.5, GaussianStatistics{Float64}(), GaussianStatistics{Float64}());

julia> 𝒮.aᵘ
0.1
```
"""
struct DoubleGaussianStatistics{FT} <: AbstractSubgridStatistics{FT}
    "updraft area fraction"
    aᵘ :: FT
    "updraft total-water offset from the cell mean, q̄ᵗᵘ - q̄ᵗ [kg kg⁻¹]"
    δqᵗ :: FT
    "updraft liquid-ice potential temperature offset from the cell mean, θ̄ˡⁱᵘ - θ̄ˡⁱ [K]"
    δθ :: FT
    "within-updraft statistics"
    updraft :: GaussianStatistics{FT}
    "within-environment statistics"
    environment :: GaussianStatistics{FT}
end

# Closures in a tuple compose. Gaussian contributions add as independent variances; a Gaussian
# contribution joins the environment of a mixture (the mass-flux member defines the components,
# the eddy-diffusivity member fills the environment variance).
@inline merge_statistics(a::DiracStatistics, b::DiracStatistics) = a
@inline merge_statistics(a::GaussianStatistics, b::GaussianStatistics) =
    GaussianStatistics(a.σ²ᵗ + b.σ²ᵗ, a.σ²ᶿ + b.σ²ᶿ, a.cᵗᶿ + b.cᵗᶿ)

@inline merge_statistics(a::GaussianStatistics, b::DoubleGaussianStatistics) =
    DoubleGaussianStatistics(b.aᵘ, b.δqᵗ, b.δθ, b.updraft, merge_statistics(a, b.environment))

@inline merge_statistics(a::DoubleGaussianStatistics, b::GaussianStatistics) = merge_statistics(b, a)

"""
$(TYPEDEF)

Abstract supertype for the microphysics-side choice of subgrid distribution.
"""
abstract type AbstractSubgridCorrelation end

"""
$(TYPEDEF)

No subgrid distribution: [`subgrid_statisticsᶜᶜᶜ`](@ref) returns [`DiracStatistics`](@ref) and the
microphysics runs its grid-mean code unchanged.

```jldoctest
julia> using Breeze.AtmosphereModels: NoSubgridCorrelation

julia> NoSubgridCorrelation()
NoSubgridCorrelation()
```
"""
struct NoSubgridCorrelation <: AbstractSubgridCorrelation end

"""
$(TYPEDEF)

A Gaussian subgrid distribution of the saturation deficit (Sommeria and Deardorff 1977;
Mellor 1977). Closures that do not prognose second moments supply them with the down-gradient
form ⟨φ'ψ'⟩ = (`variance_coefficient` ℓ² + `grid_variance_coefficient` Δz²) ∂zφ ∂zψ: the
mixing-length variance of Bechtold et al. (1995), with ℓ the closure's mixing length, plus the
variance of the resolved gradient within the cell (Δz²/12 for a linear profile). The microphysics
floors the saturation-deficit spread in quadrature, σₛ² + `minimum_saturation_deficit_spread`².

Cloud liquid relaxes toward the distribution's equilibrium condensate over the relaxation
timescale τ of the condensate-formation model `condensation` (for example an
`ExponentialRateCondensateFormation`, whose exact exponential increment keeps long steps
stable), blended with the cloud fraction C to τ [C + (1 - C) Γ]. With `condensation = nothing`
the microphysics scheme picks its own timescale.
Autoconversion is averaged over an in-cloud gamma distribution of shape ν, which multiplies the
in-cloud rate by `autoconversion_enhancement` E = Γ(ν + p) / (Γ(ν) νᵖ)
(Morrison and Gettelman 2008); the cloud fraction used to form in-cloud means is floored at
`minimum_cloud_fraction`.

# Fields
$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.AtmosphereModels: GaussianSubgridCorrelation

julia> c = GaussianSubgridCorrelation(Float64);

julia> c.grid_variance_coefficient, c.minimum_saturation_deficit_spread
(0.08333333333333333, 0.0001)

julia> round(c.autoconversion_enhancement, digits=4)
3.0247
```
"""
struct GaussianSubgridCorrelation{FT, CF} <: AbstractSubgridCorrelation
    "coefficient c of the mixing-length (co)variances ℓ² ∂zφ ∂zψ"
    variance_coefficient :: FT
    "coefficient c_Δ of the within-cell gradient (co)variances Δz² ∂zφ ∂zψ"
    grid_variance_coefficient :: FT
    "background saturation-deficit spread σ₀ [kg kg⁻¹]"
    minimum_saturation_deficit_spread :: FT
    "condensate-formation model setting the relaxation timescale τ, or `nothing`"
    condensation :: CF
    "in-cloud autoconversion enhancement factor E"
    autoconversion_enhancement :: FT
    "floor on the cloud fraction used to form in-cloud means"
    minimum_cloud_fraction :: FT
end

"""
$(TYPEDSIGNATURES)

Return a [`GaussianSubgridCorrelation`](@ref) with float type `FT`. The autoconversion enhancement
is computed from the in-cloud gamma shape `cloud_liquid_shape` ν and the autoconversion exponent
`autoconversion_exponent` p (2.47 for Khairoutdinov and Kogan 2000).
"""
function GaussianSubgridCorrelation(FT::DataType = Oceananigans.defaults.FloatType;
                                    variance_coefficient = 1,
                                    grid_variance_coefficient = 1/12,
                                    minimum_saturation_deficit_spread = 1e-4,
                                    condensation = nothing,
                                    cloud_liquid_shape = 1,
                                    autoconversion_exponent = 2.47,
                                    minimum_cloud_fraction = 0.01)
    ν = cloud_liquid_shape
    p = autoconversion_exponent
    E = gamma(ν + p) / (gamma(ν) * ν^p)
    return GaussianSubgridCorrelation(convert(FT, variance_coefficient),
                                      convert(FT, grid_variance_coefficient),
                                      convert(FT, minimum_saturation_deficit_spread),
                                      condensation,
                                      convert(FT, E),
                                      convert(FT, minimum_cloud_fraction))
end

"""
$(TYPEDSIGNATURES)

Return the subgrid distribution choice of `microphysics`; [`NoSubgridCorrelation`](@ref) by default.
"""
subgrid_correlation(microphysics) = NoSubgridCorrelation()

"""
$(TYPEDSIGNATURES)

Return the [`AbstractSubgridStatistics`](@ref) of cell `(i, j, k)` for the subgrid distribution
`correlation`, from the turbulence `closure` and its `closure_fields`. `state` is a `NamedTuple`
holding the specific conserved variables `θ` (liquid-ice potential temperature) and `qᵗ`
(non-precipitating total water), each indexable at `[i, j, k]`. Closures that know their second
moments add a method, after `compute_closure_fields!` has filled `closure_fields`; a closure
without one contributes zero variance. Closures in a tuple compose through `merge_statistics`.
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
