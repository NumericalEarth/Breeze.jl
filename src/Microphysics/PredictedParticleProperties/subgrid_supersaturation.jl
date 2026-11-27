struct GridMeanCondensation end

struct SubgridCondensation{S, P}
    statistics :: S
    processes :: P
end

SubgridCondensation(statistics; processes = CloudCondensationOnly()) =
    SubgridCondensation(statistics, validate_subgrid_processes(processes))

Adapt.adapt_structure(to, condensation::SubgridCondensation) =
    SubgridCondensation(Adapt.adapt(to, condensation.statistics), condensation.processes)

#####
##### The cell-averaged excess over zero of a Gaussian vapor excess
#####

@inline function gaussian_excess_correlation(μ, σ)
    FT = typeof(μ)
    Q = standard_score(μ, zero(FT), σ)
    σ̃ = hypotenuse(σ, eps(FT) * abs(μ) + floatmin(FT))
    φ = exp(-Q^2 / 2) / sqrt(2 * FT(π))
    Φ⁻ = erfc(Q / sqrt(FT(2))) / 2
    return max(0, σ̃ * (φ - Q * Φ⁻))
end

@inline subgrid_vapor_excess_correlation(statistics::NamedTuple{names}, μ) where names =
    subgrid_vapor_excess_correlation(Val(:a ∈ names), statistics, μ)

@inline subgrid_vapor_excess_correlation(::Val{false}, statistics, μ) =
    gaussian_excess_correlation(μ, convert(typeof(μ), statistics.σ))

@inline function subgrid_vapor_excess_correlation(::Val{true}, statistics, μ)
    FT = typeof(μ)
    weights = map(x -> convert(FT, x), statistics.a)
    means = map(x -> convert(FT, x), statistics.μ)
    spreads = map(x -> convert(FT, x), statistics.σ)
    mixture_mean = sum(map(*, weights, means))
    terms = map((aᵢ, μᵢ, σᵢ) -> aᵢ * gaussian_excess_correlation(μ + μᵢ - mixture_mean, σᵢ), weights, means, spreads)
    return sum(terms)
end

#####
##### What the P3 kernels read of the `condensation` option
#####

subgrid_supersaturation_fields(::GridMeanCondensation, model) = nothing

subgrid_supersaturation_fields(::SubgridCondensation, model) =
    (model.closure, model.closure_fields, AM.supersaturation_statistics_model_fields(model))

@inline subgrid_supersaturation_statistics(i, j, k, grid, ::GridMeanCondensation, fields) = nothing

@inline subgrid_supersaturation_statistics(i, j, k, grid, condensation::SubgridCondensation, fields) =
    AM.supersaturation_statistics(condensation.statistics, i, j, k, grid, fields...)

@inline cloud_vapor_excess(::Nothing, vapor_excess) = vapor_excess
@inline cloud_vapor_excess(Γ, vapor_excess) = vapor_excess + Γ

@inline seed_vapor(aerosol, Γ, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, ::Nothing, qᵛ) = qᵛ
@inline seed_vapor(::Nothing, Γ, qᵛ) = qᵛ + Γ
@inline seed_vapor(aerosol, Γ, qᵛ, ::Nothing) = seed_vapor(aerosol, Γ, qᵛ)
@inline seed_vapor(aerosol, Γ, qᵛ, distribution) = qᵛ

@inline p3_activation_vapor(p3, Γ, qᵛ, qᵛ⁺ˡ, distribution) =
    activation_vapor(distribution, seed_vapor(p3.aerosol, Γ, qᵛ, distribution), qᵛ⁺ˡ,
                     activation_relative_threshold(p3.aerosol, p3))

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

@inline cloud_condensation_correlation(::GridMeanCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, q, T, p, ρ, constants) = nothing

@inline function cloud_condensation_correlation(condensation::SubgridCondensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ,
                                                q, T, p, ρ, constants)
    ξˡ = p3_liquid_psychrometric_correction(qᵛ⁺ˡ, q, T, constants)
    μ = qᵛ - qᵛ⁺ˡ
    s̄ = μ + ξˡ * max(0, qᶜˡ)
    cell = (; vapor_excess = μ, saturation_excess = s̄, ξˡ, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, p, ρ)
    return vapor_excess_correlation(condensation.statistics, statistics, cell)
end

@inline vapor_excess_correlation(source, statistics, cell) =
    subgrid_vapor_excess_correlation(statistics, cell.saturation_excess)

@inline function p3_liquid_psychrometric_correction(qᵛ⁺ˡ, q, T, constants)
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ∂qᵛ⁺ˡ∂T = qᵛ⁺ˡ * saturation_vapor_pressure_log_slope(T, constants, PlanarLiquidSurface())
    return 1 + ℒˡ * ∂qᵛ⁺ˡ∂T / cᵖᵐ
end
