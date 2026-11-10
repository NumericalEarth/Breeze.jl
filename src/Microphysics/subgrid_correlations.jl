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
    σ² = saturation_deficit_variance(𝒮, a, b, σ₀)
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
$(TYPEDSIGNATURES)

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
