#####
##### Gaussian subgrid distribution of the saturation deficit: closed forms
#####
#
# With the saturation deficit s = a [qᵗ - qᵛ⁺(Tˡ)] distributed as N(s̄, σₛ²) within a cell
# (Sommeria and Deardorff 1977; Mellor 1977), the cloud fraction and the mean liquid are
#
#   Q = s̄ / σₛ,    C = Φ(Q),    q̄ˡ = σₛ G(Q),    G(Q) = Q Φ(Q) + φ(Q),
#
# with Φ, φ the standard normal distribution and density. G is smooth, G' = Φ, and σₛ G(s̄/σₛ)
# tends to max(0, s̄) as σₛ → 0: the all-or-nothing grid-mean condensate is the Dirac limit.
# The fluctuation is linear in the conserved pair, s' = a qᵗ' - b θˡⁱ', with a = 1/Γ and
# b = a Π ∂qᵛ⁺/∂T, so σₛ² = a² σ²ᵗ - 2ab cᵗᶿ + b² σ²ᶿ.

# Branch-free complementary error function (Numerical Recipes erfcc, relative error < 1.2e-7):
# arithmetic and `exp` only, so it runs in GPU kernels and traces under Reactant and Enzyme.
@inline function complementary_error_function(x)
    z = abs(x)
    t = 1 / (1 + z / 2)
    p = -z^2 - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    r = t * exp(p)
    return ifelse(x ≥ 0, r, 2 - r)
end

"""
$(TYPEDSIGNATURES)

Standard normal cumulative distribution Φ(x).

```jldoctest
julia> using Breeze.Microphysics: normal_cumulative_distribution

julia> round(normal_cumulative_distribution(1.0), digits=6)
0.841345
```
"""
@inline normal_cumulative_distribution(x) = complementary_error_function(-x / sqrt(oftype(x, 2))) / 2

"""
$(TYPEDSIGNATURES)

Standard normal density φ(x).

```jldoctest
julia> using Breeze.Microphysics: normal_density

julia> round(normal_density(0.0), digits=6)
0.398942
```
"""
@inline normal_density(x) = exp(-x^2 / 2) / sqrt(2 * oftype(x, π))

"""
$(TYPEDSIGNATURES)

Gaussian cloud fraction C = Φ(s̄ / σₛ) for mean saturation deficit `s̄` and spread `σₛ`.

```jldoctest
julia> using Breeze.Microphysics: gaussian_cloud_fraction

julia> round(gaussian_cloud_fraction(0.0, 1e-3), digits=6)
0.5
```
"""
@inline gaussian_cloud_fraction(s̄, σₛ) = normal_cumulative_distribution(s̄ / σₛ)

"""
$(TYPEDSIGNATURES)

Mean condensate σₛ G(s̄ / σₛ), with G(Q) = Q Φ(Q) + φ(Q), of a Gaussian saturation deficit with
mean `s̄` and spread `σₛ` (Mellor 1977).

```jldoctest
julia> using Breeze.Microphysics: gaussian_equilibrium_condensate

julia> round(gaussian_equilibrium_condensate(-1e-3, 1e-3) * 1e3, digits=4) # g/kg at Q = -1
0.0833
```
"""
@inline function gaussian_equilibrium_condensate(s̄, σₛ)
    Q = s̄ / σₛ
    return σₛ * (Q * normal_cumulative_distribution(Q) + normal_density(Q))
end

"""
$(TYPEDSIGNATURES)

Spread σₛ of the saturation deficit s' = a qᵗ' - b θˡⁱ' from the conserved-variable statistics
`𝒮`, with the floor `σ₀` added in quadrature.
"""
@inline function saturation_deficit_spread(𝒮::GaussianStatistics, a, b, σ₀)
    σ²ₛ = max(0, a^2 * 𝒮.σ²ᵗ - 2a * b * 𝒮.cᵗᶿ + b^2 * 𝒮.σ²ᶿ)
    return sqrt(σ²ₛ + σ₀^2)
end

"""
$(TYPEDSIGNATURES)

Cloud fraction `C` and equilibrium mean condensate `q⁺` of the subgrid distribution `𝒮` of the
saturation deficit, with cell mean `s̄`, linearisation coefficients `a`, `b` and spread floor
`σ₀`. A [`DoubleGaussianStatistics`](@ref Breeze.AtmosphereModels.DoubleGaussianStatistics)
mixture sums its components: C = Σ aₙ Φ(Qₙ), q⁺ = Σ aₙ σₙ G(Qₙ).

```jldoctest
julia> using Breeze.Microphysics: subgrid_cloud

julia> using Breeze.AtmosphereModels: GaussianStatistics

julia> cloud = subgrid_cloud(GaussianStatistics{Float64}(), 0.0, 0.4, 1e-3, 1e-4);

julia> round(cloud.C, digits=6)
0.5
```
"""
@inline function subgrid_cloud(𝒮::GaussianStatistics, s̄, a, b, σ₀)
    σₛ = saturation_deficit_spread(𝒮, a, b, σ₀)
    return (; C = gaussian_cloud_fraction(s̄, σₛ), q⁺ = gaussian_equilibrium_condensate(s̄, σₛ))
end

@inline function subgrid_cloud(𝒮::DoubleGaussianStatistics, s̄, a, b, σ₀)
    aᵘ = 𝒮.aᵘ
    δsᵘ = a * 𝒮.δqᵗ - b * 𝒮.δθ
    δsᵉ = -aᵘ * δsᵘ / (1 - aᵘ)
    updraft = subgrid_cloud(𝒮.updraft, s̄ + δsᵘ, a, b, σ₀)
    environment = subgrid_cloud(𝒮.environment, s̄ + δsᵉ, a, b, σ₀)
    C = aᵘ * updraft.C + (1 - aᵘ) * environment.C
    q⁺ = aᵘ * updraft.q⁺ + (1 - aᵘ) * environment.q⁺
    return (; C, q⁺)
end

#####
##### Relaxation timescales for the PDF condensation (reusing `relaxation_timescale`)
#####

# Effective timescale of a relaxation over `scale` × the formation's timescale. An
# `ExponentialRateCondensateFormation` returns the exact-increment timescale for its Δt,
# Δt / (1 - exp(-Δt / (scale τ))), so the update never overshoots its target.
@inline scaled_relaxation_timescale(::Nothing, τ, scale) = τ * scale
@inline scaled_relaxation_timescale(formation, τ, scale) = relaxation_timescale(formation) * scale
@inline scaled_relaxation_timescale(formation::ExponentialRateCondensateFormation, τ, scale) =
    relaxation_timescale(ExponentialRateCondensateFormation(formation.rate / scale, formation.Δt))

# Exact-increment rate factor (1 - exp(-λ Δt)) / Δt for a first-order sink of rate λ over the
# formation's window (the model time step for an exponential formation, else `τ`).
@inline relaxation_window(::Nothing, τ) = τ
@inline relaxation_window(formation, τ) = τ
@inline relaxation_window(formation::ExponentialRateCondensateFormation, τ) = formation.Δt
@inline exponential_rate_factor(λ, Δt) = -expm1(-λ * Δt) / Δt
