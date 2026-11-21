#####
##### The subgrid distribution of the saturation excess in every P3 process
#####
#
# With `SubgridCondensation(statistics; processes = AllProcesses())` the one distribution that
# sets the condensation of cloud liquid (`subgrid_supersaturation.jl`) also sets every P3 process
# that is nonlinear in the local state. The distribution is that of the saturation excess of total
# water s, a mixture of Gaussians with weights aᵢ, means mᵢ = s̄ + μᵢ - Σⱼ aⱼ μⱼ (about P3's own
# grid mean s̄ = δqᵛ⁺ˡ + ξˡ q̄ᶜˡ) and spreads σᵢ. Each cell-mean rate is the expectation of the
# local rate over it, with these placements of the hydrometeors:
#
#   * cloud liquid lies where s > 0, as the equilibrium condensate scaled to the cloud liquid
#     present: qᶜˡ(s) = q̄ᶜˡ s⁺ / ⟨s⁺⟩ (λ s⁺ / ξˡ with λ = ξˡ q̄ᶜˡ / ⟨s⁺⟩, as for Γ). The local vapor
#     excess over liquid is then δ(s) = s - ξˡ qᶜˡ(s) = s for s ≤ 0 and (1 - λ) s for s > 0, whose
#     mean is the grid-mean δqᵛ⁺ˡ;
#   * the droplet number concentration Nᶜˡ is uniform where liquid is present;
#   * rain, ice, its liquid coating and the ice number are spread over the whole cell,
#     independently of s (random overlap with the cloud).
#
# The vapor exchange of rain, the deposition and sublimation of ice and the condensation and
# evaporation of its liquid coating are linear in δ, so with these placements their cell means
# are their grid-mean rates, exactly: they are unchanged. What changes is
#
#   * activation, in the part of the cell where δ exceeds the activation threshold: the tail
#     fraction P, times the rate at the tail-mean supersaturation;
#   * deposition nucleation, in the part where the excess over ice δ + qᵛ⁺ˡ - qᵛ⁺ⁱ reaches its
#     threshold: the fraction of the cell that nucleates times the grid-mean (ungated) rate;
#   * autoconversion and accretion, which are powers of qᶜˡ(s): partial moments of the mixture,
#     by Gauss-Legendre quadrature.
#
# Each rate is still one number per process. Every source is paired with its sink as on the grid
# mean, so mass and energy are conserved as before. With all spreads zero the grid-mean rates are
# returned unchanged.

"""
$(TYPEDEF)

The subgrid distribution of [`SubgridCondensation`](@ref) enters only the condensation and
evaporation of cloud liquid (the default `processes`).
"""
struct CloudCondensationOnly end

"""
$(TYPEDEF)

The subgrid distribution of [`SubgridCondensation`](@ref) enters every P3 process that is
nonlinear in the local state: besides cloud condensation, droplet activation, deposition nucleation
of ice, autoconversion and accretion. Cloud liquid is placed where the saturation excess ``s > 0``,
as ``qᶜˡ(s) = q̄ᶜˡ s⁺ / ⟨s⁺⟩``, with a uniform droplet number concentration; rain and ice fill the
cell independently of ``s``. The vapor exchange of rain and ice, being linear in the local vapor
excess, keeps its grid-mean rate under this placement.

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, AllProcesses

julia> using Breeze.AtmosphereModels: PrescribedStatistics

julia> SubgridCondensation(PrescribedStatistics(1e-4); processes = AllProcesses()).processes
AllProcesses()
```
"""
struct AllProcesses end

validate_subgrid_processes(processes::CloudCondensationOnly) = processes
validate_subgrid_processes(processes::AllProcesses) = processes
validate_subgrid_processes(processes) =
    throw(ArgumentError("The `processes` of `SubgridCondensation` must be `CloudCondensationOnly()` " *
                        "or `AllProcesses()`; got $(summary(processes))."))

#####
##### The distribution P3 forms for a cell
#####

# The mixture of a cell, as tuples of weights, means about P3's grid mean and spreads
@inline function shifted_mixture(statistics::NamedTuple{names}, s̄) where names
    return shifted_mixture(Val(:a ∈ names), statistics, s̄)
end

@inline shifted_mixture(::Val{false}, statistics, s̄) =
    ((one(s̄),), (s̄,), (convert(typeof(s̄), statistics.σ),))

@inline function shifted_mixture(::Val{true}, statistics, s̄)
    FT = typeof(s̄)
    a = map(x -> convert(FT, x), statistics.a)
    μ = map(x -> convert(FT, x), statistics.μ)
    σ = map(x -> convert(FT, x), statistics.σ)
    plume_mean = sum(map(*, a, μ))
    m = map(μᵢ -> s̄ + μᵢ - plume_mean, μ)
    return a, m, σ
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid distribution that the P3 processes other than cloud condensation read, or
`nothing` when they use their grid-mean forms: for [`GridMeanCondensation`](@ref) and for
[`SubgridCondensation`](@ref) with `processes = CloudCondensationOnly()`. With
[`AllProcesses`](@ref) the result holds the mixture of the saturation excess about P3's grid mean
``s̄ = qᵛ - qᵛ⁺ˡ + ξˡ qᶜˡ`` (weights `a`, means `m`, spreads `σ`), the psychrometric correction
`ξˡ`, the cloud liquid `qᶜˡ`, the liquid scale ``λ = ξˡ qᶜˡ / ⟨s⁺⟩`` with ``⟨s⁺⟩ = s̄ + Γ``, the
saturation `qᵛ⁺ˡ`, and `degenerate`, true when every spread is zero.
"""
@inline subgrid_process_distribution(::GridMeanCondensation, args...) = nothing

@inline subgrid_process_distribution(condensation::SubgridCondensation, args...) =
    subgrid_process_distribution(condensation.processes, args...)

@inline subgrid_process_distribution(::CloudCondensationOnly, args...) = nothing

@inline function subgrid_process_distribution(::AllProcesses, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, constants, Γ)
    FT = typeof(qᵛ)
    ξˡ = p3_liquid_psychrometric_correction(qᵛ⁺ˡ, T, constants)
    q̄ᶜˡ = max(0, qᶜˡ)
    s̄ = qᵛ - qᵛ⁺ˡ + ξˡ * q̄ᶜˡ
    a, m, σ = shifted_mixture(statistics, s̄)
    mean_positive_excess = max(0, s̄ + Γ)
    has_liquid = (q̄ᶜˡ > 0) & (mean_positive_excess > 0)
    λ = ifelse(has_liquid, ξˡ * q̄ᶜˡ / ifelse(has_liquid, mean_positive_excess, one(FT)), zero(FT))
    degenerate = sum(σ) == 0
    return (; a, m, σ, ξˡ, qᶜˡ = q̄ᶜˡ, λ, qᵛ⁺ˡ, degenerate)
end

#####
##### Gaussian integrals over intervals of s
#####

# Upper tail 1 - Φ(z), with the argument clamped where the tail is below round-off
@inline function gaussian_upper_tail(z)
    FT = typeof(z)
    zᶜ = clamp(z, -convert(FT, 40), convert(FT, 40))
    return erfc(zᶜ / sqrt(FT(2))) / 2
end

@inline function gaussian_density(z)
    FT = typeof(z)
    zᶜ = clamp(z, -convert(FT, 40), convert(FT, 40))
    return exp(-zᶜ^2 / 2) / sqrt(2 * FT(π))
end

# The probability and first moment of one Gaussian over l ≤ s < u, with `bounded` false for
# u = ∞. A zero spread is the point mass at the mean.
@inline function gaussian_interval_moments(m, σ, l, u, bounded)
    FT = typeof(m)
    σ₊ = max(σ, floatmin(FT) / eps(FT))
    zˡ = (l - m) / σ₊
    zᵘ = (u - m) / σ₊
    tailᵘ = ifelse(bounded, gaussian_upper_tail(zᵘ), zero(FT))
    φᵘ = ifelse(bounded, gaussian_density(zᵘ), zero(FT))
    P = max(0, gaussian_upper_tail(zˡ) - tailᵘ)
    M = m * P + σ₊ * (gaussian_density(zˡ) - φᵘ)
    # Empty interval
    nonempty = !bounded | (u > l)
    return ifelse(nonempty, P, zero(FT)), ifelse(nonempty, M, zero(FT))
end

# The same over the mixture
@inline function mixture_interval_moments(distribution, l, u, bounded)
    moments = map((mᵢ, σᵢ) -> gaussian_interval_moments(mᵢ, σᵢ, l, u, bounded), distribution.m, distribution.σ)
    P = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[1], distribution.a, moments))
    M = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[2], distribution.a, moments))
    return P, M
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the fraction `P` of the cell where the local vapor excess over liquid ``δ(s)`` exceeds
`threshold` [kg kg⁻¹], and the mean of ``δ`` over that part times `P`. ``δ(s) = s`` where
``s ≤ 0`` and ``(1 - λ) s`` where ``s > 0``, with the liquid scale ``λ`` of `distribution` (see
[`subgrid_process_distribution`](@ref)); the set ``δ > c`` is the union of a clear interval
``c < s ≤ 0`` and a cloudy interval of ``s > 0``, whose bounds follow from the sign of ``1 - λ``.
"""
@inline function vapor_excess_exceedance(distribution, threshold)
    FT = typeof(threshold)
    c = threshold
    r = 1 - distribution.λ
    # Clear part: c < s ≤ 0
    P⁻, M⁻ = mixture_interval_moments(distribution, min(c, zero(FT)), zero(FT), true)
    # Cloudy part, s > 0 and r s > c:
    #   r > 0:          s > max(c, 0) / r
    #   r ≤ 0, c < 0:   0 < s < c / r (unbounded when r = 0)
    #   r ≤ 0, c ≥ 0:   empty
    r₊ = ifelse(r > 0, r, one(FT))
    r₋ = ifelse(r < 0, r, -one(FT))
    lower = ifelse(r > 0, max(c, zero(FT)) / r₊, zero(FT))
    upper = ifelse(r > 0, zero(FT), ifelse(c < 0, c / r₋, zero(FT)))
    bounded = (r < 0) | ((r == 0) & (c >= 0))
    P⁺, M⁺ = mixture_interval_moments(distribution, lower, upper, bounded)
    return P⁻ + P⁺, M⁻ + r * M⁺
end

#####
##### Partial moments of the placed cloud liquid, by Gauss-Legendre quadrature
#####

# The positive nodes and weights of the 24-point Gauss-Legendre rule on [-1, 1]
const gauss_legendre_nodes = (0.06405689286260585, 0.1911188674736165, 0.31504267969616384,
                              0.4337935076260454, 0.5454214713888397, 0.6480936519369753,
                              0.7401241915785544, 0.820001985973903, 0.8864155270044011,
                              0.9382745520027327, 0.9747285559713095, 0.9951872199970213)

const gauss_legendre_weights = (0.12793819534675185, 0.12583745634682753, 0.12167047292780374,
                                0.11550566805372615, 0.10744427011596513, 0.09761865210411456,
                                0.08619016153195298, 0.07334648141108072, 0.059298584915437,
                                0.04427743881741972, 0.028531388628933545, 0.012341229799987141)

# ∫ₗ^∞ s^β N(s; m, σ²) ds for l ≥ 0, over [max(l, m - 8σ), m + 8σ], beyond which the Gaussian is
# below 10⁻¹⁵ of its peak. A zero spread is the point mass at the mean.
@inline function gaussian_partial_moment(β, m, σ, l)
    FT = typeof(m)
    R = convert(FT, 8)
    σ₊ = max(σ, floatmin(FT) / eps(FT))
    lo = max(l, m - R * σ₊)
    hi = max(lo, m + R * σ₊)
    h = (hi - lo) / 2
    c = (hi + lo) / 2
    terms = map(gauss_legendre_nodes, gauss_legendre_weights) do x, w
        s⁺ = c + h * convert(FT, x)
        s⁻ = c - h * convert(FT, x)
        z⁺ = (s⁺ - m) / σ₊
        z⁻ = (s⁻ - m) / σ₊
        convert(FT, w) * (positive_power(s⁺, β) * exp(-z⁺^2 / 2) +
                          positive_power(s⁻, β) * exp(-z⁻^2 / 2))
    end
    quadrature = sum(terms) * h / (sqrt(2 * FT(π)) * σ₊)
    point_mass = ifelse(m >= l, positive_power(m, β), zero(FT))
    return ifelse(σ > 0, quadrature, point_mass)
end

# s^β for s > 0 and 0 otherwise, with a base that keeps the derivative in β finite at s = 0
@inline function positive_power(s, β)
    positive = s > 0
    return ifelse(positive, ifelse(positive, s, one(s))^β, zero(s))
end

@inline mixture_partial_moment(distribution, β, l) =
    sum(map((aᵢ, mᵢ, σᵢ) -> aᵢ * gaussian_partial_moment(β, mᵢ, σᵢ, l),
            distribution.a, distribution.m, distribution.σ))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the cell mean of ``qᶜˡ(s)^β`` over the part of the cell where ``qᶜˡ(s) ≥`` `threshold`,
for the placed cloud liquid ``qᶜˡ(s) = q̄ᶜˡ s⁺ / ⟨s⁺⟩`` of `distribution` (see
[`subgrid_process_distribution`](@ref)):

```math
⟨qᶜˡ(s)^β ; qᶜˡ ≥ qᵗ⟩ = (q̄ᶜˡ / ⟨s⁺⟩)^β Σᵢ aᵢ ∫_{sᵗ}^∞ s^β N(s; mᵢ, σᵢ²) ds,  sᵗ = qᵗ ⟨s⁺⟩ / q̄ᶜˡ.
```

``⟨s⁺⟩`` is the same quadrature at ``β = 1``, so that the placed liquid sums to ``q̄ᶜˡ``.
Without cloud liquid, or without a saturated part, the result is the grid-mean power
``q̄ᶜˡ^β`` gated at the threshold.
"""
@inline function placed_cloud_liquid_moment(distribution, β, threshold)
    FT = typeof(threshold)
    q̄ = distribution.qᶜˡ
    s⁺ = mixture_partial_moment(distribution, one(FT), zero(FT))
    resolved = (q̄ > 0) & (s⁺ > 0) & !distribution.degenerate
    s⁺ₛ = ifelse(resolved, s⁺, one(FT))
    q̄ₛ = ifelse(resolved, q̄, one(FT))
    sᵗ = max(0, threshold) * s⁺ₛ / q̄ₛ
    subgrid = (q̄ₛ / s⁺ₛ)^β * mixture_partial_moment(distribution, β, sᵗ)
    grid_mean = ifelse(q̄ >= threshold, positive_power(q̄, β), zero(FT))
    return ifelse(resolved, subgrid, grid_mean)
end

#####
##### The rates that read the distribution
#####

# Autoconversion: KK2000 at the local cloud liquid, gated locally at its threshold
@inline subgrid_autoconversion_rate(::Nothing, p3, qᶜˡ, Nᶜˡ, ρ, qʳ) =
    rain_autoconversion_rate(p3, qᶜˡ, Nᶜˡ, ρ, qʳ)

@inline subgrid_autoconversion_rate(distribution, p3, qᶜˡ, Nᶜˡ, ρ, qʳ) =
    subgrid_autoconversion_rate(p3.warm_rain_scheme, distribution, p3, qᶜˡ, Nᶜˡ, ρ, qʳ)

@inline function subgrid_autoconversion_rate(::KhairoutdinovKogan2000, distribution, p3, qᶜˡ, Nᶜˡ, ρ, qʳ)
    parameters = p3.process_rates
    ℂᵃᵘᵗᵒ₁ = parameters.autoconversion_coefficient
    ℂᵃᵘᵗᵒ₂ = parameters.autoconversion_exponent_cloud
    ℂᵃᵘᵗᵒ₃ = parameters.autoconversion_exponent_droplet
    ℂᵃᵘᵗᵒ₄ = parameters.autoconversion_threshold
    scaled_cloud_number = Nᶜˡ / parameters.autoconversion_reference_concentration
    subgrid = ℂᵃᵘᵗᵒ₁ * placed_cloud_liquid_moment(distribution, ℂᵃᵘᵗᵒ₂, ℂᵃᵘᵗᵒ₄) * scaled_cloud_number^ℂᵃᵘᵗᵒ₃
    return ifelse(distribution.degenerate, rain_autoconversion_rate(p3, qᶜˡ, Nᶜˡ, ρ, qʳ), subgrid)
end

# Accretion: KK2000 at the local cloud liquid and the cell's rain, active where both are present
@inline subgrid_accretion_rate(::Nothing, p3, qᶜˡ, qʳ, ρ) = rain_accretion_rate(p3, qᶜˡ, qʳ, ρ)

@inline subgrid_accretion_rate(distribution, p3, qᶜˡ, qʳ, ρ) =
    subgrid_accretion_rate(p3.warm_rain_scheme, distribution, p3, qᶜˡ, qʳ, ρ)

@inline function subgrid_accretion_rate(::KhairoutdinovKogan2000, distribution, p3, qᶜˡ, qʳ, ρ)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates
    ℂᵃᶜᶜʳ₁ = parameters.accretion_coefficient
    ℂᵃᶜᶜʳ₂ = parameters.accretion_exponent
    qʳ₊ = max(0, qʳ)
    rain_present = qʳ₊ >= p3.minimum_mass_mixing_ratio
    moment = placed_cloud_liquid_moment(distribution, ℂᵃᶜᶜʳ₂, p3.minimum_mass_mixing_ratio)
    subgrid = ifelse(rain_present, ℂᵃᶜᶜʳ₁ * moment * qʳ₊^ℂᵃᶜᶜʳ₂, zero(FT))
    return ifelse(distribution.degenerate, rain_accretion_rate(p3, qᶜˡ, qʳ, ρ), subgrid)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `(; qᵛ, fraction)`: the vapor that droplet activation sees and the fraction of the cell
that activates. On the grid mean (`distribution = nothing`) they are `qᵛ` and 1. With a subgrid
distribution, activation acts in the part of the cell where the local vapor excess over liquid
exceeds the activation threshold ``ℂᶠᵒʳᵐ₃ qᵛ⁺ˡ``: `fraction` is that part and `qᵛ` is
``qᵛ⁺ˡ`` plus the mean vapor excess over it (see [`vapor_excess_exceedance`](@ref)).
"""
# The activation threshold on the relative supersaturation over liquid
@inline activation_relative_threshold(::Nothing, p3) = p3.process_rates.activation_supersaturation_threshold
@inline activation_relative_threshold(aerosol, p3) = aerosol.activation_supersaturation_threshold

@inline activation_vapor(::Nothing, qᵛ, qᵛ⁺ˡ, relative_threshold) = (; qᵛ, fraction = one(qᵛ))

@inline function activation_vapor(distribution, qᵛ, qᵛ⁺ˡ, relative_threshold)
    FT = typeof(qᵛ)
    P, M = vapor_excess_exceedance(distribution, relative_threshold * qᵛ⁺ˡ)
    active = (P > 0) & !distribution.degenerate
    tail_excess = M / ifelse(active, P, one(FT))
    return (; qᵛ = ifelse(active, qᵛ⁺ˡ + tail_excess, qᵛ),
              fraction = ifelse(distribution.degenerate, one(FT), P))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the fraction of the cell where deposition nucleation of ice acts: where the local vapor
excess over ice reaches `ℂⁿᵘᶜˡ₂` times `qᵛ⁺ⁱ`. On the grid mean (`distribution = nothing`) it is 1
or 0. With a subgrid distribution the local excess over ice is ``δ(s) + qᵛ⁺ˡ - qᵛ⁺ⁱ``, the local
vapor excess over liquid shifted by the difference of the saturations of the cell, so the fraction
is that where ``δ(s) ≥ ℂⁿᵘᶜˡ₂ qᵛ⁺ⁱ - qᵛ⁺ˡ + qᵛ⁺ⁱ``.
"""
@inline function nucleating_fraction(::Nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, floor)
    FT = typeof(qᵛ)
    Sⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, floor)
    return ifelse(Sⁱ >= ℂⁿᵘᶜˡ₂, one(FT), zero(FT))
end

@inline function nucleating_fraction(distribution, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, floor)
    threshold = ℂⁿᵘᶜˡ₂ * max(qᵛ⁺ⁱ, floor) - qᵛ⁺ˡ + qᵛ⁺ⁱ
    P, _ = vapor_excess_exceedance(distribution, threshold)
    return ifelse(distribution.degenerate, nucleating_fraction(nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, floor), P)
end
