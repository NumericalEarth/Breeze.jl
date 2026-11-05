using Breeze.Thermodynamics: exner_function

#####
##### Gaussian subgrid distribution of the saturation deficit: closed forms
#####
#
# With the saturation deficit s = qᵗ − qᵛ⁺(Tˡ) distributed as N(s̄, σₛ²) within a cell
# (Sommeria and Deardorff 1977; Mellor 1977), the cloud fraction and the mean liquid are
#
#   Q = s̄ / σₛ,    C = Φ(Q),    q̄ˡ = σₛ G(Q),    G(Q) = Q Φ(Q) + φ(Q),
#
# with Φ, φ the standard normal distribution and density. G is smooth, G' = Φ, and σₛ G(s̄/σₛ)
# tends to max(0, s̄) as σₛ → 0: the all-or-nothing grid-mean condensate is the Dirac limit.

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

Spread σₛ of the saturation deficit s = (qᵗ − qᵛ⁺(Tˡ)) / Γ, linearised about the grid-mean
state, from the conserved-variable statistics `𝒮`: s' = (qᵗ' − β θˡⁱ') / Γ with
β = Π ∂qᵛ⁺/∂T (Mellor 1977). The floor `σ₀` is added in quadrature.
"""
@inline function saturation_deficit_spread(𝒮::GaussianStatistics, β, Γ, σ₀)
    σ²ₛ = max(0, 𝒮.σ²ᵗ - 2β * 𝒮.cᵗᶿ + β^2 * 𝒮.σ²ᶿ) / Γ^2
    return sqrt(σ²ₛ + σ₀^2)
end

"""
$(TYPEDSIGNATURES)

Condensation rate of cloud liquid (kg/kg/s) over a subgrid distribution of the saturation
deficit. With [`DiracStatistics`](@ref) this is the grid-mean [`condensation_rate`](@ref).
With [`GaussianStatistics`](@ref) the cloud liquid relaxes toward the distribution's mean
condensate (Tompkins 2002),

    ⟨Sᶜᵒⁿᵈ⟩ = (σₛ G(s̄ / σₛ) − qᶜˡ) / τ,    s̄ = (qᵛ − qᵛ⁺) / Γ + qᶜˡ,

where s̄ already carries the latent-heat factor Γ, so that for σₛ → 0 in a saturated cell this
is the grid-mean rate (qᵛ − qᵛ⁺) / (Γ τ). (Where the grid-mean scheme evaporates liquid at its
limiter −qᶜˡ / (Γ τ), the Gaussian relaxes at −qᶜˡ / τ toward a vanishing equilibrium.) The Exner function Π comes from the
thermodynamic state `𝒰`, and `correlation` supplies the background spread.
"""
@inline condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants, 𝒰, ::DiracStatistics, correlation) =
    condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)

@inline function condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants, 𝒰, 𝒮::GaussianStatistics, correlation)
    Π = exner_function(𝒰, constants)
    ℒˡ = liquid_latent_heat(T, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ * (ℒˡ / (Rᵛ * T^2) - 1 / T)
    Γˡ = thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    σₛ = saturation_deficit_spread(𝒮, Π * dqᵛ⁺_dT, Γˡ, correlation.minimum_saturation_deficit_spread)
    s̄ = (qᵛ - qᵛ⁺) / Γˡ + qᶜˡ
    qˡ⁺ = gaussian_equilibrium_condensate(s̄, σₛ)
    return (qˡ⁺ - qᶜˡ) / τᶜˡ
end
