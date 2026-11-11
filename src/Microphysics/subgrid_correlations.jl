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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Standard normal cumulative distribution Φ(x).

```jldoctest
julia> using Breeze.Microphysics: normal_cumulative_distribution

julia> round(normal_cumulative_distribution(1.0), digits=6)
0.841345
```
"""
@inline normal_cumulative_distribution(x) = complementary_error_function(-x / sqrt(oftype(x, 2))) / 2

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Standard normal density φ(x).

```jldoctest
julia> using Breeze.Microphysics: normal_density

julia> round(normal_density(0.0), digits=6)
0.398942
```
"""
@inline normal_density(x) = exp(-x^2 / 2) / sqrt(2 * oftype(x, π))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Gaussian cloud fraction C = Φ(s̄ / σₛ) for mean saturation deficit `s̄` and spread `σₛ`.

```jldoctest
julia> using Breeze.Microphysics: gaussian_cloud_fraction

julia> round(gaussian_cloud_fraction(0.0, 1e-3), digits=6)
0.5
```
"""
@inline gaussian_cloud_fraction(s̄, σₛ) = normal_cumulative_distribution(s̄ / σₛ)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Spread σₛ of the saturation deficit s' = a qᵗ' - b θˡⁱ' from the conserved-variable statistics
`𝒮`, with the floor `σ₀` added in quadrature and σₛ² bounded below by a tiny positive value so
that σₛ > 0 even when both vanish.
"""
@inline saturation_deficit_spread(𝒮::GaussianStatistics, a, b, σ₀) =
    sqrt(max(saturation_deficit_variance(𝒮, a, b, σ₀), minimum_saturation_deficit_variance))

# σₛ² + σ₀², before the lower bound
@inline saturation_deficit_variance(𝒮::GaussianStatistics, a, b, σ₀) =
    max(0, a^2 * 𝒮.σ²ᵗ - 2a * b * 𝒮.cᵗᶿ + b^2 * 𝒮.σ²ᶿ) + σ₀^2

# Lower bound on σₛ² that keeps σₛ > 0 when both the variance and the floor σ₀ vanish, so that
# s̄ / σₛ, and the derivative of the square root, stay finite (no NaN in C or q̄ˡ, nor under AD).
# A literal (Float32 `floatmin`, σₛ ≈ 1e-19 kg/kg) rather than `floatmin(FT)`, which a traced number
# type does not define. It never binds for any physical floor (σ₀ = 1e-4 gives σₛ² ≥ 1e-8), and at
# this spread C and q̄ˡ take their all-or-nothing limits, C = 1 or 0 and q̄ˡ = max(0, s̄).
const minimum_saturation_deficit_variance = floatmin(Float32)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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
@inline subgrid_cloud(𝒮::GaussianStatistics, s̄, a, b, σ₀) =
    gaussian_cloud(s̄, saturation_deficit_variance(𝒮, a, b, σ₀))

# Cloud fraction and mean condensate of a Gaussian saturation deficit with mean s̄ and variance σ²
@inline function gaussian_cloud(s̄, σ²)
    σₛ = sqrt(max(σ², minimum_saturation_deficit_variance))
    # A vanishing spread is the Dirac limit: all-or-nothing cloud (C = 0 at s̄ = 0, where Φ(0) = 1/2)
    dirac = σ² < minimum_saturation_deficit_variance
    C = ifelse(dirac, ifelse(s̄ > 0, one(s̄), zero(s̄)), gaussian_cloud_fraction(s̄, σₛ))
    return (; C, q⁺ = gaussian_equilibrium_condensate(s̄, σₛ))
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
##### Skewed two-Gaussian (binormal) distribution of the saturation deficit
#####
#
# Cumulus layers hold their liquid in a positively skewed tail that a Gaussian of the same variance
# misses. The binormal is the mixture of two Gaussians of equal width that has the Gaussian's mean
# s̄ and variance σₛ² and a prescribed non-dimensional skewness Sk = ⟨s'³⟩ / σₛ³. With each
# component holding the fraction γ of the variance, a wet component of weight aʷ at s̄ + δsʷ and
# a dry one of weight 1 - aʷ at s̄ + δsᵈ,
#
#   aʷ = [1 - Sk / √(4 (1 - γ)³ + Sk²)] / 2,
#   δsʷ = σₛ √((1 - γ) (1 - aʷ) / aʷ),    δsᵈ = -σₛ √((1 - γ) aʷ / (1 - aʷ)),    σ² = γ σₛ²,
#
# so that aʷ δsʷ + (1 - aʷ) δsᵈ = 0, the offsets carry the variance (1 - γ) σₛ² and the third
# moment aʷ δsʷ³ + (1 - aʷ) δsᵈ³ = Sk σₛ³. Everything is smooth in Sk and σₛ. The weight is kept
# within [1e-4, 1 - 1e-4], which binds only for |Sk| ≳ 100 (1 - γ)^(3/2).

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Components of the skewed two-Gaussian (binormal) distribution with spread `σₛ`, non-dimensional
skewness `Sk` and component variance fraction `γ`: the weight `aʷ` of the wet component, the
offsets `δsʷ` and `δsᵈ` of the wet and dry component means from the mean, and the component
variance `σ²`. The mixture has zero mean offset, variance σₛ² and third moment Sk σₛ³.

```jldoctest
julia> using Breeze.Microphysics: binormal_components

julia> c = binormal_components(1.0, 2.0, 0.4);

julia> round(c.aʷ, digits=4), round(c.δsʷ, digits=4), round(c.δsᵈ, digits=4), round(c.σ², digits=4)
(0.0466, 3.5045, -0.1712, 0.4)

julia> round(c.aʷ * c.δsʷ^3 + (1 - c.aʷ) * c.δsᵈ^3, digits=12) # the skewness
2.0
```
"""
@inline function binormal_components(σₛ, Sk, γ)
    aʷ = (1 - Sk / sqrt(4 * (1 - γ)^3 + Sk^2)) / 2
    ϵ = oftype(aʷ, 1e-4)
    aʷ = min(max(aʷ, ϵ), 1 - ϵ)
    δsʷ = σₛ * sqrt((1 - γ) * (1 - aʷ) / aʷ)
    δsᵈ = -σₛ * sqrt((1 - γ) * aʷ / (1 - aʷ))
    return (; aʷ, δsʷ, δsᵈ, σ² = γ * σₛ^2)
end

# Cloud fraction and mean condensate of the binormal with mean s̄, variance σ², skewness Sk
@inline function binormal_cloud(s̄, σ², Sk, γ)
    σₛ = sqrt(max(σ², minimum_saturation_deficit_variance))
    components = binormal_components(σₛ, Sk, γ)
    aʷ = components.aʷ
    wet = gaussian_cloud(s̄ + components.δsʷ, γ * σ²)
    dry = gaussian_cloud(s̄ + components.δsᵈ, γ * σ²)
    C = aʷ * wet.C + (1 - aʷ) * dry.C
    q⁺ = aʷ * wet.q⁺ + (1 - aʷ) * dry.q⁺
    return (; C, q⁺)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Cloud fraction `C` and equilibrium mean condensate `q⁺` of the subgrid distribution that
`correlation` assigns to the statistics `𝒮`. A
[`GaussianSubgridCorrelation`](@ref Breeze.AtmosphereModels.GaussianSubgridCorrelation) with a
non-zero `saturation_deficit_skewness` turns Gaussian statistics into the skewed two-Gaussian
distribution of [`binormal_components`](@ref Breeze.Microphysics.binormal_components); zero skewness returns the Gaussian values exactly.
Any other pair is the distribution of `𝒮` itself.

```jldoctest
julia> using Breeze.Microphysics: subgrid_cloud

julia> using Breeze.AtmosphereModels: GaussianStatistics, GaussianSubgridCorrelation

julia> 𝒮 = GaussianStatistics(1e-7, 0.0, 0.0);

julia> gaussian = subgrid_cloud(GaussianSubgridCorrelation(Float64), 𝒮, -4e-4, 1.0, 0.0, 0.0);

julia> skewed = subgrid_cloud(GaussianSubgridCorrelation(Float64; saturation_deficit_skewness=2), 𝒮, -4e-4, 1.0, 0.0, 0.0);

julia> gaussian == subgrid_cloud(𝒮, -4e-4, 1.0, 0.0, 0.0)
true

julia> round(skewed.q⁺ / gaussian.q⁺, digits=1) # the skewed tail holds more liquid
2.2
```
"""
@inline subgrid_cloud(correlation, 𝒮, s̄, a, b, σ₀) = subgrid_cloud(𝒮, s̄, a, b, σ₀)

@inline function subgrid_cloud(correlation::GaussianSubgridCorrelation, 𝒮::GaussianStatistics, s̄, a, b, σ₀)
    σ² = saturation_deficit_variance(𝒮, a, b, σ₀)
    # The parameters take the number type of the state, so that a traced parameter mixes with
    # the literals below like any other value.
    Sk = correlation.saturation_deficit_skewness * one(σ²)
    γ = correlation.component_variance_fraction * one(σ²)
    gaussian = gaussian_cloud(s̄, σ²)
    skewed = binormal_cloud(s̄, σ², Sk, γ)
    # Selected by value, so that Sk can be a traced or differentiated parameter; both branches
    # are finite for every Sk.
    symmetric = Sk == 0
    return (; C = ifelse(symmetric, gaussian.C, skewed.C), q⁺ = ifelse(symmetric, gaussian.q⁺, skewed.q⁺))
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

#####
##### Droplet phase-relaxation timescale (opt-in; replaces the constant condensation timescale)
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Phase-relaxation rate λ = α 4π Dᵛ N r̄ of a droplet population ([Squires (1952)](@cite Squires1952);
[Politovich and Cooper (1988)](@cite PolitovichCooper1988)), with vapor diffusivity `Dᵛ`, droplet number `N` [m⁻³], the mean-volume radius
r̄ = (3 ρ qˡ / (4π ρʷ N))^(1/3) of the liquid `qˡ` in air of density `ρ`, liquid density `ρʷ`,
and the non-dimensional coefficient `α`. The radius is floored at the activation radius `r_act`,
inside the cube root so that the derivative stays finite at qˡ = 0, which keeps λ > 0 and
τ = 1 / λ finite as the liquid vanishes. The latent-heat correction is not included; the subgrid
condensation applies it through its Γ blend.

```jldoctest
julia> using Breeze.Microphysics: phase_relaxation_rate

julia> λ = phase_relaxation_rate(1.0, 2.5e-5, 2e8, 1e-4, 1.0, 1e3, 1e-6);

julia> round(1 / λ, digits=2) # τ [s] for 0.1 g/kg of liquid over 200 cm⁻³ droplets
3.23
```
"""
@inline function phase_relaxation_rate(α, Dᵛ, N, qˡ, ρ, ρʷ, r_act)
    FT = typeof(r_act)
    r̄ = cbrt(max(r_act^3, 3 * ρ * max(0, qˡ) / (4 * FT(π) * ρʷ * N)))
    return α * 4 * FT(π) * Dᵛ * N * r̄
end

# The formation model with its constant rate replaced by the per-cell phase-relaxation rate λ:
# the exponential increment keeps its time step, and with no formation model the relaxation is
# the plain first-order one of rate λ. `nothing` (the default) keeps the formation model as it is.
@inline phase_relaxation_formation(::Nothing, formation, args...) = formation
@inline phase_relaxation_formation(α, formation, args...) =
    with_relaxation_rate(formation, phase_relaxation_rate(α, args...))

@inline with_relaxation_rate(::Nothing, λ) = ConstantRateCondensateFormation(λ)
@inline with_relaxation_rate(::ConstantRateCondensateFormation, λ) = ConstantRateCondensateFormation(λ)
@inline with_relaxation_rate(formation::ExponentialRateCondensateFormation, λ) =
    ExponentialRateCondensateFormation(λ, formation.Δt)
