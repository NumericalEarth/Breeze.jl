#####
##### The subgrid distribution of the saturation excess in every P3 process
#####
#
# With `SubgridCondensation(statistics; processes = AllProcesses())` the one distribution that
# sets the condensation of cloud liquid also sets every P3 process that is nonlinear in the local
# state, through one placement of the cloud liquid (`placed_cloud_liquid`). The distribution is
# that of the saturation excess of total water s, a mixture of Gaussians with weights aᵢ, means
# mᵢ = s̄ + μᵢ - Σⱼ aⱼ μⱼ about P3's own grid mean s̄ = δqᵛ⁺ˡ + ξˡ q̄ᶜˡ, and spreads σᵢ. Its
# saturated part s > 0 covers f⁺ = ⟨1; s > 0⟩ of the cell and has ⟨s⁺⟩ = ⟨s; s > 0⟩; the
# subsaturated part has ⟨s⁻⟩ = s̄ - ⟨s⁺⟩ ≤ 0.
#
# Placement of the cloud liquid. A fraction χ of it lies in the saturated part, shaped as the
# equilibrium condensate, and 1 - χ is spread uniformly over the subsaturated part, which covers
# f⁻ = ⟨1; s ≤ 0⟩ = 1 - f⁺ of the cell. The placement is carried by ψ = q⊖ / q̄ᶜˡ, the liquid of the
# subsaturated part relative to the cell mean, so that 1 - χ = f⁻ ψ:
#
#   qᶜˡ(s) = χ q̄ᶜˡ s / ⟨s⁺⟩ for s > 0,   qᶜˡ(s) = q⊖ = ψ q̄ᶜˡ for s ≤ 0.
#
# f⁻ and ⟨s⁻⟩ are evaluated from the lower tails of the mixture, not as 1 - f⁺ and s̄ - ⟨s⁺⟩, and
# nothing is divided by f⁻: where the subsaturated part is small, both are small differences of
# numbers near 1 and s̄.
#
# Droplet number is placed with the liquid in the same proportions, uniformly in each part:
# Nᶜˡ = χ N̄ᶜˡ / f⁺ in the saturated part and ψ N̄ᶜˡ in the subsaturated part.
# Rain and the liquid coating of ice fill the cell, independently of s. The ice fills the cell in two parts,
# q̄ⁱ (1 + γⁱ f⁻) in the saturated part and q̄ⁱ (1 - γⁱ f⁺) in the subsaturated part, its number with its mass; the
# contrast γⁱ follows from the covariance of the ice with the excess over ice where the statistics carry one
# (`ice_placement_contrast`), and is zero (uniform ice) otherwise.
#
# The local vapor excess over liquid is δ(s) = s - ξˡ qᶜˡ(s): (1 - λ) s in the saturated part,
# λ = χ ξˡ q̄ᶜˡ / ⟨s⁺⟩, and s - ξˡ q⊖ in the subsaturated part. Its cell mean is the grid-mean
# δqᵛ⁺ˡ for any χ. So the vapor exchange of rain and of the liquid coating of ice, linear in δ, keeps its
# grid-mean rate exactly. The vapor exchange of the dry ice does not: the ice shares the local excess with the
# cloud liquid of each part, so it is evaluated in each part (below).
#
# Condensation of cloud liquid, at the local droplet population. Droplets of gamma size
# distribution relax the vapor at ε = 2π ρ Dᵛ N (μᶜˡ + 1)/λᶜˡ ∝ N^{2/3} qᶜˡ^{1/3} at a fixed shape,
# so locally ε(s) = ε̄ (Nᶜˡ(s) / N̄ᶜˡ)^{2/3} (qᶜˡ(s) / q̄ᶜˡ)^{1/3} with ε̄ the cell's, and cloud liquid
# changes at (ε(s) / ξˡ) δ(s). With ωˡ = ε̄ / ξˡ the cell's phase relaxation rate, the cell mean is
#
#   R = ωˡ [ χ (f⁺)^{-2/3} (1 - λ) ⟨s^{4/3}; s > 0⟩ / ⟨s⁺⟩^{1/3} + ψ (⟨s⁻⟩ - ξˡ f⁻ ψ q̄ᶜˡ) ],
#
# (the subsaturated part scales both N and qᶜˡ by ψ), written for the rest of P3 as
# R = ωˡ (δqᵛ⁺ˡ + Γ) with Γ = R / ωˡ - δqᵛ⁺ˡ. ⟨s^{4/3}; s > 0⟩ is a Gauss-Legendre quadrature.
#
# The placement ψ (`MixingEvaporationBalance`). Turbulence exchanges air between the two parts at
# the closure's rate ωᵐ = √e / (ℂᵐ ℓ), carrying liquid from the saturated part to the
# subsaturated part at ωᵐ [f⁻ χ - f⁺ (1 - χ)] q̄ᶜˡ = ωᵐ f⁻ (1 - ψ) q̄ᶜˡ, the exchange that would make
# the liquid uniform. In the subsaturated part the liquid evaporates at its local droplets' rate,
# ωˡ ψ |⟨s⁻⟩| to leading order in ξˡ q⊖ / |s|. The steady state of the liquid of the subsaturated
# part, exchange = evaporation, is
#
#   ψᵇ = f⁻ ωᵐ q̄ᶜˡ / (ωˡ |⟨s⁻⟩| + f⁻ ωᵐ q̄ᶜˡ),
#
# which is 0 ≤ ψᵇ ≤ 1: 0 without mixing or without liquid, 1 (uniform liquid, χ = f⁺) when mixing
# is fast against evaporation. Liquid beyond the equilibrium condensate ⟨s⁺⟩ / ξˡ of the saturated
# part cannot be at equilibrium anywhere in the cell; it is spread uniformly over the cell, as the
# uniform part ε = (q̄ᶜˡ - ⟨s⁺⟩ / ξˡ)₊ of the liquid, the share ψᵉ = ε / q̄ᶜˡ of it. The balance places the rest of
# the liquid, so that the subsaturated part holds, relative to the cell mean,
#
#   ψ = ψᵉ + (1 - ψᵉ) ψᵇ = 1 - (1 - ψᵉ) (1 - ψᵇ),   ψᵉ = (1 - ⟨s⁺⟩ / (ξˡ q̄ᶜˡ))₊,
#
# which lies in [0, 1] and is at least ψᵇ and ψᵉ by construction. Without mixing (ψᵇ = 0) the saturated part
# holds its equilibrium condensate and its share f⁺ ε of the excess. The placement is continuous as either part
# vanishes: χ → 1 with f⁻ → 0, χ → 0 with f⁺ → 0, and q⊖ ≤ q̄ᶜˡ throughout. (Placing all of ε in the subsaturated
# part instead makes q⊖ = ε / f⁻ and its evaporation diverge as the subsaturated part vanishes.)
# `SaturatedPartLiquid` fixes ψ = 0 (χ = 1).
#
# The other processes at the same placement:
#
#   * activation, in the part of the cell where δ exceeds the activation threshold: that fraction
#     times the rate at its mean vapor excess;
#   * deposition nucleation, in the part where the excess over ice δ + qᵛ⁺ˡ - qᵛ⁺ⁱ reaches its
#     threshold: that fraction times the grid-mean (ungated) rate;
#   * KK2000 autoconversion and accretion, powers of the local qᶜˡ and Nᶜˡ: partial moments of the
#     mixture in the saturated part (Gauss-Legendre quadrature) plus the uniform subsaturated part;
#   * the vapor exchange of the dry ice, in each part at its local excesses over liquid and over ice and against the
#     cloud liquid of that part: integrated over s in the saturated part and evaluated at the moments of the
#     subsaturated part (`subgrid_ice_growth`);
#   * the cap on the vapor sinks of ice (deposition and nucleation): the cell mean of the positive part of the
#     local excess over ice (`available_ice_excess`), not the positive part of the grid mean.
#
# Each rate is still one number per process, and every source is paired with its sink as on the
# grid mean, so mass and energy are conserved as they are there. With all spreads zero the grid-mean rates
# are returned unchanged.

"""
$(TYPEDEF)

The subgrid distribution of [`SubgridCondensation`](@ref) enters only the condensation and
evaporation of cloud liquid (the default `processes`).
"""
struct CloudCondensationOnly end

"""
$(TYPEDEF)

All of the cloud liquid lies in the saturated part of the distribution (``χ = 1``), shaped as the
equilibrium condensate. See [`AllProcesses`](@ref).
"""
struct SaturatedPartLiquid end

"""
$(TYPEDEF)

The cloud liquid of the subsaturated part of the distribution is the steady state between its
exchange with the saturated part at the closure's mixing rate ``ωᵐ`` and its evaporation at the
droplets' rate; liquid beyond the equilibrium condensate of the saturated part is spread uniformly
over the cell. See [`AllProcesses`](@ref).
"""
struct MixingEvaporationBalance end

"""
$(TYPEDEF)

The subgrid distribution of [`SubgridCondensation`](@ref) enters every P3 process that is
nonlinear in the local state: the condensation of cloud liquid at its local droplet population,
droplet activation, deposition nucleation of ice, autoconversion and accretion. All read one
placement of the cloud liquid: a fraction ``χ`` in the saturated part ``s > 0``, as
``χ q̄ᶜˡ s / ⟨s⁺⟩``, and ``1 - χ`` spread over the subsaturated part, with the droplet number
placed in the same proportions. Rain and the liquid coating of ice fill the cell independently of ``s``;
their vapor exchange, linear in the local vapor excess, keeps its grid-mean rate. The dry ice is placed in
the two parts with its covariance with the excess over ice (uniformly without one); it exchanges vapor in
each part at the local excesses and against the cloud liquid of that part, and its vapor sinks are capped
by the positive part of the local excess over ice.

`liquid_placement` sets ``χ``: [`MixingEvaporationBalance`](@ref) (the default), the steady state of
mixing and evaporation, which reads the mixing rate `ωᵐ` of the statistics (0 when they have
none), or [`SaturatedPartLiquid`](@ref), ``χ = 1``.

$(TYPEDFIELDS)

```jldoctest
julia> using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, AllProcesses, MixingEvaporationBalance

julia> using Breeze.AtmosphereModels: PrescribedStatistics

julia> SubgridCondensation(PrescribedStatistics(1e-4); processes = AllProcesses()).processes
AllProcesses{MixingEvaporationBalance}(MixingEvaporationBalance())
```
"""
struct AllProcesses{P}
    "How the cloud liquid is placed between the saturated and subsaturated parts"
    liquid_placement :: P
end

AllProcesses(; liquid_placement = MixingEvaporationBalance()) = AllProcesses(liquid_placement)

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
    mixture_mean = sum(map(*, a, μ))
    m = map(μᵢ -> s̄ + μᵢ - mixture_mean, μ)
    return a, m, σ
end

# The exchange rate of air between the saturated and subsaturated parts, if the statistics have one
@inline mixing_rate(statistics::NamedTuple{names}, FT) where names = mixing_rate(Val(:ωᵐ ∈ names), statistics, FT)
@inline mixing_rate(::Val{true}, statistics, FT) = convert(FT, statistics.ωᵐ)
@inline mixing_rate(::Val{false}, statistics, FT) = zero(FT)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the subgrid distribution and placement of cloud liquid that the P3 processes read, or
`nothing` when they use their grid-mean forms: for [`GridMeanCondensation`](@ref) and for
[`SubgridCondensation`](@ref) with `processes = CloudCondensationOnly()`. With
[`AllProcesses`](@ref) the arguments are the statistics of the cell, its vapor, saturation and cloud
liquid, its moisture mass fractions `q`, the temperature, the thermodynamic constants, and the scheme,
density, transport properties and cloud droplet distribution from which the cloud's phase relaxation
rate `ωˡ` is formed; the result is [`placed_cloud_liquid`](@ref) of the mixture about P3's grid mean
``s̄ = qᵛ - qᵛ⁺ˡ + ξˡ qᶜˡ``, with the excess over ice, the placement of the ice and the departure of the
cloud liquid from its placement.
"""
@inline subgrid_process_distribution(::GridMeanCondensation, args...) = nothing

@inline subgrid_process_distribution(condensation::SubgridCondensation, args...) =
    subgrid_process_distribution(condensation.processes, args...)

@inline subgrid_process_distribution(::CloudCondensationOnly, args...) = nothing

@inline function subgrid_process_distribution(processes::AllProcesses, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, q, T, constants,
                                              p3, ρ, transport, cloud, qᵛ⁺ⁱ = qᵛ⁺ˡ, qⁱ = zero(qᵛ))
    FT = typeof(qᵛ)
    ξˡ = p3_liquid_psychrometric_correction(qᵛ⁺ˡ, q, T, constants)
    ωˡ = p3_cloud_phase_relaxation_rate(p3, ξˡ, qᶜˡ, ρ, transport, cloud)
    q̄ᶜˡ = max(0, qᶜˡ)
    s̄ = qᵛ - qᵛ⁺ˡ + ξˡ * q̄ᶜˡ
    a, m, σ = shifted_mixture(statistics, s̄)
    ωᵐ = mixing_rate(statistics, FT)
    d = placed_cloud_liquid(processes.liquid_placement, a, m, σ, s̄, ξˡ, q̄ᶜˡ, qᵛ⁺ˡ, ωˡ, ωᵐ)
    # The saturation excess of the distributed water over ice, sⁱ = qᵛ - qᵛ⁺ⁱ + ξⁱˡ qᶜˡ, about P3's own grid mean
    ξⁱˡ = p3_ice_liquid_psychrometric_correction(qᵛ⁺ⁱ, q, T, constants)
    s̄ⁱ = qᵛ - qᵛ⁺ⁱ + ξⁱˡ * q̄ᶜˡ
    mⁱ, βⁱ, σʳ = ice_excess_mixture(statistics, d, s̄ⁱ)
    dⁱ = merge(d, (; mⁱ, βⁱ, σʳ, ξⁱˡ))
    # The placement of the ice between the two parts and the departure of the cloud liquid from its placement
    γⁱ = ice_placement_contrast(statistics, dⁱ, max(0, qⁱ))
    ν = liquid_departure_scale(statistics, dⁱ)
    return merge(dⁱ, (; γⁱ, ν))
end

#####
##### The ice and the departure of the cloud liquid, from their covariance and variance
#####

# The ice fills the cell in two parts, with the mean preserved and both parts positive:
#
#   qⁱ⁺ = q̄ⁱ (1 + γ f⁻) in the saturated part,   qⁱ⁻ = q̄ⁱ (1 - γ f⁺) in the subsaturated part,
#
# whose covariance with the excess over ice is q̄ⁱ γ (f⁻ S⁺ - f⁺ S⁻), S± the P-weighted excess over ice of the
# distributed water in each part. γ is set by the covariance qⁱ′sⁱ′ of the statistics (their steady budget) and kept
# in |γ| max(f⁺, f⁻) < 1 by the rational bound γ₀ / (1 + (γ₀ max(f⁺, f⁻))⁴)^{1/4}. Zero without the covariance.
@inline ice_placement_contrast(statistics::NamedTuple{names}, d, qⁱ) where names =
    ice_placement_contrast(Val(:ice_covariance ∈ names), statistics, d, qⁱ)
@inline ice_placement_contrast(::Val{false}, statistics, d, qⁱ) = zero(qⁱ)

@inline function ice_placement_contrast(::Val{true}, statistics, d, qⁱ)
    FT = typeof(qⁱ)
    C = convert(FT, statistics.ice_covariance)
    S⁺, S⁻ = ice_excess_part_sums(d)
    contrast = d.f⁻ * S⁺ - d.f⁺ * S⁻
    # γ = γ₀ / (1 + (γ₀ f)⁴)^{1/4} with γ₀ = C / (q̄ⁱ contrast), written as C / ((q̄ⁱ contrast)⁴ + (C f)⁴)^{1/4}
    γ = bounded_ratio(C, ifelse(contrast > 0, qⁱ * contrast, zero(FT)), max(d.f⁺, d.f⁻))
    return ifelse(contrast > 0, γ, zero(FT))
end

# x / (y⁴ + (x b)⁴)^{1/4} = x₀ / (1 + (x₀ b)⁴)^{1/4}, x₀ = x / y, bounded by 1/b, formed without x₀ (so finite where y is
# tiny) and scaled by the larger of |y| and |x b|; zero where both are
@inline function bounded_ratio(x, y, b)
    FT = typeof(x)
    u = abs(y)
    v = abs(x * b)
    scale = max(u, v)
    resolved = scale > 0
    scaleₛ = ifelse(resolved, scale, one(FT))
    û = u / scaleₛ
    v̂ = v / scaleₛ
    return ifelse(resolved, x / (scaleₛ * sqrt(sqrt(û^4 + v̂^4))), zero(FT))
end

# The P-weighted excess over ice of the distributed water in the saturated and subsaturated parts (linear in sⁱ, so the
# residual of the regression on s averages out)
@inline function ice_excess_part_sums(d)
    FT = typeof(d.qᶜˡ)
    sums = map(d.a, d.m, d.σ, d.mⁱ, d.βⁱ) do aᵢ, mᵢ, σᵢ, mⁱᵢ, βⁱᵢ
        bᵢ = mⁱᵢ - βⁱᵢ * mᵢ
        P⁺, M⁺ = gaussian_interval_moments(mᵢ, σᵢ, zero(FT), zero(FT), false)
        P⁻, M⁻ = gaussian_lower_moments(mᵢ, σᵢ)
        (aᵢ * (bᵢ * P⁺ + βⁱᵢ * M⁺), aᵢ * (bᵢ * P⁻ + βⁱᵢ * M⁻))
    end
    return sum(map(first, sums)), sum(map(last, sums))
end

# The relative ice of the two parts, qⁱ±/q̄ⁱ (one without a placement of the ice)
@inline ice_part_scales(d::NamedTuple{names}) where names = ice_part_scales(Val(:γⁱ ∈ names), d)
@inline ice_part_scales(::Val{false}, d) = (one(d.qᶜˡ), one(d.qᶜˡ))
@inline ice_part_scales(::Val{true}, d) = (1 + d.γⁱ * d.f⁻, 1 - d.γⁱ * d.f⁺)

# The departure of the cloud liquid from its placement is multiplicative, qᶜˡ = q(s) (1 + ν ζ), ζ ~ N(0, 1)
# independent of s: it keeps the mean and adds ν² ⟨q²⟩ = d′² of variance, the departure variance of the statistics
# (its steady budget). ν is kept below 1/ζ_max of the outermost node of the Gauss-Hermite rule that integrates over ζ,
# so the liquid stays positive there: ν = ν₀ / (1 + (ν₀ ζ_max)⁴)^{1/4}, ν₀² = d′² / ⟨q²⟩. Zero without the variance.
@inline liquid_departure_scale(statistics::NamedTuple{names}, d) where names =
    liquid_departure_scale(Val(:departure_variance ∈ names), statistics, d)
@inline liquid_departure_scale(::Val{false}, statistics, d) = zero(d.qᶜˡ)

@inline function liquid_departure_scale(::Val{true}, statistics, d)
    FT = typeof(d.qᶜˡ)
    second_moment = placed_cloud_liquid_moment(d, convert(FT, 2), zero(FT))
    return bounded_ratio(sqrt(max(0, convert(FT, statistics.departure_variance))), sqrt(max(0, second_moment)),
                         convert(FT, last(gauss_hermite_nodes)))
end

# The probabilists' 8-point Gauss-Hermite rule, ∫ f(ζ) N(ζ; 0, 1) dζ ≈ Σⱼ wⱼ f(ζⱼ): the positive nodes and their weights
const gauss_hermite_nodes = (0.5390798113513751, 1.6365190424351082, 2.8024858612875416, 4.144547186125894)
const gauss_hermite_weights = (0.37301225767907614, 0.11723990766175905, 0.009635220120788268, 0.00011261453837536777)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The cell mean of ``qᶜˡ^β (Nᶜˡ/N̄ᶜˡ)^γ`` above `threshold` with the departure of the cloud liquid from its placement,
``qᶜˡ = q(s) (1 + ν ζ)``: ``Σⱼ wⱼ (1 + ν ζⱼ)^β ⟨q^β (N/N̄)^γ⟩ⱼ`` over the 8-point Gauss-Hermite rule in ``ζ``, with
``⟨⋅⟩ⱼ`` the placed moment of [`placed_cloud_liquid_moment`](@ref) over the part where ``q (1 + ν ζⱼ)`` reaches
`threshold`. Without a departure (``ν = 0``) it is that placed moment.
"""
@inline departed_liquid_moment(d::NamedTuple{names}, β, threshold, γ = zero(β)) where names =
    departed_liquid_moment(Val(:ν ∈ names), d, β, threshold, γ)

@inline departed_liquid_moment(::Val{false}, d, β, threshold, γ) = placed_cloud_liquid_moment(d, β, threshold, γ)

# f^β ⟨q^β; f q ≥ threshold⟩ at the node's factor f ≥ 0 (zero at f = 0, where the bound on ν puts the outermost node
# when the cell has no cloud liquid)
@inline function scaled_placed_moment(d, β, threshold, γ, f)
    present = f > 0
    fₛ = ifelse(present, f, one(f))
    return ifelse(present, fₛ^β * placed_cloud_liquid_moment(d, β, threshold / fₛ, γ), zero(f))
end

@inline function departed_liquid_moment(::Val{true}, d, β, threshold, γ)
    FT = typeof(threshold)
    placed = placed_cloud_liquid_moment(d, β, threshold, γ)
    terms = map(gauss_hermite_nodes, gauss_hermite_weights) do ζ, w
        f⁺ = 1 + d.ν * convert(FT, ζ)
        f⁻ = 1 - d.ν * convert(FT, ζ)
        convert(FT, w) * (scaled_placed_moment(d, β, threshold, γ, f⁺) + scaled_placed_moment(d, β, threshold, γ, f⁻))
    end
    return ifelse(d.ν > 0, sum(terms), placed)
end

# ξⁱˡ = 1 + ℒˡ ∂qᵛ⁺ⁱ/∂T / cᵖᵐ: the saturation excess over ice that the evaporation of cloud liquid returns, per unit of
# liquid (the liquid cools the air by ℒˡ / cᵖᵐ and lowers qᵛ⁺ⁱ), with ∂ ln qᵛ⁺ⁱ/∂T that of the model's saturation
# vapor pressure and the mixture heat capacity cᵖᵐ of the cell's moisture mass fractions `q` (see
# `p3_liquid_psychrometric_correction`)
@inline function p3_ice_liquid_psychrometric_correction(qᵛ⁺ⁱ, q, T, constants)
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ∂qᵛ⁺ⁱ∂T = qᵛ⁺ⁱ * saturation_vapor_pressure_log_slope(T, constants, PlanarIceSurface())
    return 1 + ℒˡ * ∂qᵛ⁺ⁱ∂T / cᵖᵐ
end

# The saturation excess over ice of the components of the mixture, regressed on the excess over liquid s within each
# component: sⁱ = mⁱᵢ + βⁱᵢ (s - mᵢ) + rᵢ, βⁱᵢ = 𝕏ⁱᵢ σⁱᵢ / σᵢ, with the residual rᵢ ~ N(0, (σʳᵢ)²), σʳᵢ = σⁱᵢ √(1 - 𝕏ⁱᵢ²),
# independent of s, and 𝕏ⁱᵢ the within-component correlation of the two excesses that the statistics carry (both are
# linear in the same fluctuations of θ and qᵗ and differ in ∂qᵛ⁺/∂T). The means are placed about P3's grid mean s̄ⁱ
# with the offsets of the statistics. Statistics without the excess over ice (prescribed ones) give it the offsets
# and spreads of the excess over liquid (βⁱᵢ = 1, no residual); without a correlation, 𝕏ⁱᵢ = 1.
@inline ice_excess_mixture(statistics::NamedTuple{names}, d, s̄ⁱ) where names =
    ice_excess_mixture(Val(:μⁱ ∈ names), statistics, d, s̄ⁱ)

@inline function ice_excess_mixture(::Val{false}, statistics, d, s̄ⁱ)
    mⁱ = map(mᵢ -> s̄ⁱ + (mᵢ - d.mean_excess), d.m)
    return mⁱ, map(mᵢ -> one(mᵢ), d.m), map(mᵢ -> zero(mᵢ), d.m)
end

@inline excess_correlation(statistics::NamedTuple{names}, FT) where names = excess_correlation(Val(:𝕏ⁱ ∈ names), statistics, FT)
@inline excess_correlation(::Val{true}, statistics, FT) = map(x -> convert(FT, x), statistics.𝕏ⁱ)
@inline excess_correlation(::Val{false}, statistics, FT) = map(x -> one(FT), statistics.μⁱ)

@inline function ice_excess_mixture(::Val{true}, statistics, d, s̄ⁱ)
    FT = typeof(s̄ⁱ)
    μⁱ = map(x -> convert(FT, x), statistics.μⁱ)
    σⁱ = map(x -> convert(FT, x), statistics.σⁱ)
    𝕏ⁱ = excess_correlation(statistics, FT)
    mixture_mean = sum(map(*, d.a, μⁱ))
    mⁱ = map(μⁱᵢ -> s̄ⁱ + μⁱᵢ - mixture_mean, μⁱ)
    # Without a spread of s the slope multiplies s - mᵢ = 0 and its value does not enter
    βⁱ = map((𝕏ⁱᵢ, σⁱᵢ, σᵢ) -> safe_ratio(𝕏ⁱᵢ * σⁱᵢ, σᵢ, one(FT)), 𝕏ⁱ, σⁱ, d.σ)
    # With no spread of s the residual is all of σⁱ. (Round-off guard: a correlation satisfies 𝕏ⁱ² ≤ 1 exactly, which
    # the quotient of rounded moments that forms it can exceed by a few ϵ.)
    σʳ = map((𝕏ⁱᵢ, σⁱᵢ, σᵢ) -> ifelse(σᵢ > 0, σⁱᵢ * sqrt(max(0, 1 - 𝕏ⁱᵢ * 𝕏ⁱᵢ)), σⁱᵢ), 𝕏ⁱ, σⁱ, d.σ)
    return mⁱ, βⁱ, σʳ
end

#####
##### Gaussian integrals over intervals of s
#####

# The standard scores are those of `AtmosphereModels.standard_score`, regular at a zero spread, where they give the
# point mass at the mean; no integral here divides by a spread.

# Upper tail 1 - Φ(z). The argument is held at ±40, beyond which the tail is 0 or 1 and the density 0 in every float
# type, so that their derivatives are finite there (z e^{-z²/2} is ∞ × 0 for an infinite z): no value changes.
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
    zˡ = standard_score(l, m, σ)
    zᵘ = standard_score(u, m, σ)
    tailᵘ = ifelse(bounded, gaussian_upper_tail(zᵘ), zero(FT))
    φᵘ = ifelse(bounded, gaussian_density(zᵘ), zero(FT))
    # (Round-off guard: the score increases with the bound and the tail decreases with the score, so the exact
    # difference is not negative.)
    P = max(0, gaussian_upper_tail(zˡ) - tailᵘ)
    M = m * P + σ * (gaussian_density(zˡ) - φᵘ)
    nonempty = !bounded | (u > l)
    return ifelse(nonempty, P, zero(FT)), ifelse(nonempty, M, zero(FT))
end

# The same over the mixture `(a, m, σ)`
@inline function mixture_interval_moments(a, m, σ, l, u, bounded)
    moments = map((mᵢ, σᵢ) -> gaussian_interval_moments(mᵢ, σᵢ, l, u, bounded), m, σ)
    P = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[1], a, moments))
    M = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[2], a, moments))
    return P, M
end

@inline mixture_interval_moments(d::NamedTuple, l, u, bounded) = mixture_interval_moments(d.a, d.m, d.σ, l, u, bounded)

# The probability and first moment of one Gaussian over s ≤ 0, from its lower tail: accurate where
# they are small, unlike 1 - P(s > 0) and s̄ - ⟨s; s > 0⟩. A zero spread is the point mass at the mean.
@inline function gaussian_lower_moments(m, σ)
    z = standard_score(m, zero(m), σ)
    P = gaussian_upper_tail(z)
    # (Round-off guard: the exact mean of s over s ≤ 0 is not positive; the two terms cancel where z is large.)
    M = min(0, m * P - σ * gaussian_density(z))
    return P, M
end

@inline function mixture_lower_moments(a, m, σ)
    moments = map(gaussian_lower_moments, m, σ)
    P = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[1], a, moments))
    M = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[2], a, moments))
    return P, M
end

#####
##### Partial moments by Gauss-Legendre quadrature
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

# ∫ₗ^∞ s^β N(s; m, σ²) ds for l ≥ 0, as the probability of s ≥ l times the mean of s^β over that part,
#
#   [1 - Φ(zˡ)] Σⱼ wⱼ φ(zⱼ) (m + σ zⱼ)^β / Σⱼ wⱼ φ(zⱼ),   zˡ = (l - m) / σ,
#
# with the Gauss-Legendre nodes zⱼ of the standard variable over the part of [zˡ, ∞) in the window [-8, 8], beyond
# which the Gaussian is below 10⁻¹⁵ of its peak (`gaussian_window_mean`). The probability is exact and the mean lies
# between the least and the greatest s^β of the nodes, so a zero spread gives the point mass at the mean, m^β where
# m > l, with no division by the spread. Where l lies above the window the probability is zero and the nodes stay at
# its upper edge: at a node far above it s^β overflows (a threshold of the local liquid reaches 10¹⁵⁷ times its mean
# where q̄ᶜˡ is 10⁻¹⁶⁹).
@inline function gaussian_partial_moment(β, m, σ, l)
    FT = typeof(m)
    R = convert(FT, 8)
    zˡ = standard_score(l, m, σ)
    mean, = gaussian_window_mean(s -> (positive_power(s, β),), clamp(zˡ, -R, R), R, m, σ)
    return gaussian_upper_tail(zˡ) * mean
end

# s^β for s > 0 and 0 otherwise, with a base that keeps the derivative in β finite at s = 0
@inline function positive_power(s, β)
    positive = s > 0
    return ifelse(positive, ifelse(positive, s, one(s))^β, zero(s))
end

# Σᵢ aᵢ ∫₀^∞ f(s, i) N(s; mᵢ, σᵢ²) ds: the probability of s > 0 times the mean of f over that part, as in
# `gaussian_partial_moment`; a zero spread is the point mass at the mean. `f(s, i)` returns a number, finite for
# every s.
@inline function positive_part_quadrature(f, d)
    FT = typeof(d.qᶜˡ)
    R = convert(FT, 8)
    components = map(d.a, d.m, d.σ, ntuple(identity, length(d.a))) do aᵢ, mᵢ, σᵢ, i
        z⁰ = standard_score(zero(FT), mᵢ, σᵢ)
        mean, = gaussian_window_mean(s -> (f(s, i),), clamp(z⁰, -R, R), R, mᵢ, σᵢ)
        aᵢ * gaussian_upper_tail(z⁰) * mean
    end
    return sum(components)
end

@inline mixture_partial_moment(a, m, σ, β, l) =
    sum(map((aᵢ, mᵢ, σᵢ) -> aᵢ * gaussian_partial_moment(β, mᵢ, σᵢ, l), a, m, σ))

@inline mixture_partial_moment(d::NamedTuple, β, l) = mixture_partial_moment(d.a, d.m, d.σ, β, l)

# a / b where b > 0, and `fallback` otherwise
@inline safe_ratio(a, b, fallback) = ifelse(b > 0, a / ifelse(b > 0, b, one(b)), fallback)

#####
##### The placement of the cloud liquid
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the placement of the cloud liquid in the mixture `(a, m, σ)` of the saturation excess about
the grid mean `s̄`, read by every P3 process of [`AllProcesses`](@ref): the mixture, the
psychrometric correction `ξˡ`, the cloud liquid `qᶜˡ`, the saturated and subsaturated fractions
`f⁺` and `f⁻` (from the upper and the lower tails), ``⟨s⁺⟩`` (`positive_excess`), ``⟨s⁻⟩``
(`negative_excess`) and `s̄` (`mean_excess`), the fraction `χ = 1 - f⁻ ψ` of the liquid in the
saturated part, the scale ``λ = χ ξˡ qᶜˡ / ⟨s⁺⟩`` of the local vapor excess there, the liquid
``q⊖ = ψ qᶜˡ`` (`subsaturated_liquid`) of the subsaturated part and its share ``ψ``
(`subsaturated_share`) of the cell mean, the saturation `qᵛ⁺ˡ`, and `degenerate`, true when every
spread is zero (the grid mean). `ωˡ` is the cloud's phase relaxation rate and `ωᵐ` the exchange
rate of air between the two parts.
"""
@inline function placed_cloud_liquid(placement, a, m, σ, s̄, ξˡ, q̄ᶜˡ, qᵛ⁺ˡ, ωˡ, ωᵐ)
    FT = typeof(s̄)
    f⁺, s⁺ = mixture_interval_moments(a, m, σ, zero(FT), zero(FT), false)
    f⁻, s⁻ = mixture_lower_moments(a, m, σ)
    ψ = subsaturated_liquid_share(placement, f⁻, s⁺, s⁻, ξˡ, q̄ᶜˡ, ωˡ, ωᵐ)
    χ = 1 - f⁻ * ψ
    λ = safe_ratio(χ * ξˡ * q̄ᶜˡ, s⁺, zero(FT))
    degenerate = sum(σ) == 0
    return (; a, m, σ, ξˡ, qᶜˡ = q̄ᶜˡ, f⁺, f⁻, positive_excess = s⁺, negative_excess = s⁻, mean_excess = s̄,
              χ, λ, subsaturated_share = ψ, subsaturated_liquid = ψ * q̄ᶜˡ, qᵛ⁺ˡ, degenerate)
end

@inline subsaturated_liquid_share(::SaturatedPartLiquid, f⁻, s⁺, s⁻, ξˡ, q̄ᶜˡ, ωˡ, ωᵐ) = zero(f⁻)

@inline function subsaturated_liquid_share(::MixingEvaporationBalance, f⁻, s⁺, s⁻, ξˡ, q̄ᶜˡ, ωˡ, ωᵐ)
    FT = typeof(f⁻)
    evaporation = ωˡ * (-s⁻)
    exchange = f⁻ * ωᵐ * q̄ᶜˡ
    ψᵇ = safe_ratio(exchange, evaporation + exchange, zero(FT))
    # The share of the liquid beyond the saturated part's equilibrium condensate ⟨s⁺⟩ / ξˡ, spread over the cell: the
    # positive part of the excess (the regime of a cell with more liquid than that condensate, not a numerical
    # bound), at most 1 since ⟨s⁺⟩ ≥ 0
    ψᵉ = max(0, 1 - safe_ratio(s⁺, ξˡ * q̄ᶜˡ, one(FT)))
    # The excess is uniform and the balance places the rest: in [0, 1] and at least ψᵇ and ψᵉ by construction
    return ψᵉ + (1 - ψᵉ) * ψᵇ
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return ``Γ`` such that the cell-mean condensation rate of cloud liquid at its local droplet
population and placement `d` (see [`placed_cloud_liquid`](@ref)) is ``ωˡ (δqᵛ⁺ˡ + Γ)``, with the
grid-mean vapor excess ``δqᵛ⁺ˡ = s̄ - ξˡ q̄ᶜˡ``:

```math
δqᵛ⁺ˡ + Γ = χ (f⁺)^{-2/3} (1 - λ) ⟨s^{4/3}; s > 0⟩ / ⟨s⁺⟩^{1/3} + ψ (⟨s⁻⟩ - ξˡ f⁻ ψ q̄ᶜˡ),
```

`Γ` is returned unchanged where `d` is `nothing` or degenerate.
"""
@inline placed_condensation_correlation(::Nothing, Γ) = Γ

@inline function placed_condensation_correlation(d, Γ)
    FT = typeof(d.qᶜˡ)
    s⁺ = d.positive_excess
    saturated_density = safe_ratio(mixture_partial_moment(d, convert(FT, 4//3), zero(FT)),
                                   positive_power(d.f⁺, convert(FT, 2//3)) * positive_power(s⁺, convert(FT, 1//3)),
                                   zero(FT))
    saturated = d.χ * (1 - d.λ) * saturated_density
    ψ = d.subsaturated_share
    subsaturated = ψ * (d.negative_excess - d.ξˡ * d.f⁻ * ψ * d.qᶜˡ)
    vapor_excess = d.mean_excess - d.ξˡ * d.qᶜˡ
    return ifelse(d.degenerate, Γ, saturated + subsaturated - vapor_excess)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the cell mean of ``qᶜˡ(s)^β (Nᶜˡ(s) / N̄ᶜˡ)^γ`` over the part of the cell where
``qᶜˡ(s) ≥`` `threshold`, at the placement `d` (see [`placed_cloud_liquid`](@ref)):

```math
(χ / f⁺)^γ Σᵢ aᵢ ∫_{q ≥ threshold} q^β N(q; g mᵢ, (g σᵢ)²) dq + f⁻ ψ^γ q⊖^β [q⊖ ≥ threshold],
```

with ``g = χ q̄ᶜˡ / ⟨s⁺⟩``: the local liquid of the saturated part, ``q = g s``, is Gaussian in each component
with mean ``g mᵢ`` and spread ``g σᵢ``. Without cloud liquid, or where `d` is degenerate, the result is
the grid-mean ``q̄ᶜˡ^β`` gated at the threshold.
"""
@inline function placed_cloud_liquid_moment(d, β, threshold, γ = zero(β))
    FT = typeof(threshold)
    q̄ = d.qᶜˡ
    s⁺ = d.positive_excess
    saturated_liquid = d.χ * q̄
    saturated_present = (saturated_liquid > 0) & (s⁺ > 0) & (d.f⁺ > 0)
    s⁺ₛ = ifelse(saturated_present, s⁺, one(FT))
    saturated_liquidₛ = ifelse(saturated_present, saturated_liquid, one(FT))
    f⁺ₛ = ifelse(saturated_present, d.f⁺, one(FT))
    q̄ₛ = ifelse(q̄ > 0, q̄, one(FT))
    # The moment of the local liquid q = g s itself, so that no threshold of s is formed (it overflows where the
    # liquid is tiny)
    g = saturated_liquidₛ / s⁺ₛ
    liquid_moment = mixture_partial_moment(d.a, map(mᵢ -> g * mᵢ, d.m), map(σᵢ -> g * σᵢ, d.σ), β, max(0, threshold))
    saturated = (saturated_liquidₛ / (f⁺ₛ * q̄ₛ))^γ * liquid_moment
    saturated = ifelse(saturated_present, saturated, zero(FT))

    subsaturated_present = (d.subsaturated_liquid > 0) & (d.f⁻ > 0)
    subsaturated_liquidₛ = ifelse(subsaturated_present, d.subsaturated_liquid, one(FT))
    ψₛ = ifelse(subsaturated_present, d.subsaturated_share, one(FT))
    subsaturated = d.f⁻ * ψₛ^γ * subsaturated_liquidₛ^β
    subsaturated = ifelse(subsaturated_present & (d.subsaturated_liquid >= threshold), subsaturated, zero(FT))

    resolved = (q̄ > 0) & !d.degenerate
    grid_mean = ifelse(q̄ >= threshold, positive_power(q̄, β), zero(FT))
    return ifelse(resolved, saturated + subsaturated, grid_mean)
end

#####
##### The rates that read the placement
#####

# Autoconversion: KK2000 at the local cloud liquid and droplet number, gated locally at its threshold
@inline subgrid_autoconversion_rate(::Nothing, p3, qᶜˡ, Nᶜˡ, ρ, qʳ) =
    rain_autoconversion_rate(p3, qᶜˡ, Nᶜˡ, ρ, qʳ)

@inline subgrid_autoconversion_rate(d, p3, qᶜˡ, Nᶜˡ, ρ, qʳ) =
    subgrid_autoconversion_rate(p3.warm_rain_scheme, d, p3, qᶜˡ, Nᶜˡ, ρ, qʳ)

@inline function subgrid_autoconversion_rate(::KhairoutdinovKogan2000, d, p3, qᶜˡ, Nᶜˡ, ρ, qʳ)
    parameters = p3.process_rates
    ℂᵃᵘᵗᵒ₁ = parameters.autoconversion_coefficient
    ℂᵃᵘᵗᵒ₂ = parameters.autoconversion_exponent_cloud
    ℂᵃᵘᵗᵒ₃ = parameters.autoconversion_exponent_droplet
    ℂᵃᵘᵗᵒ₄ = parameters.autoconversion_threshold
    scaled_cloud_number = Nᶜˡ / parameters.autoconversion_reference_concentration
    moment = departed_liquid_moment(d, ℂᵃᵘᵗᵒ₂, ℂᵃᵘᵗᵒ₄, ℂᵃᵘᵗᵒ₃)
    subgrid = ℂᵃᵘᵗᵒ₁ * moment * scaled_cloud_number^ℂᵃᵘᵗᵒ₃
    return ifelse(d.degenerate, rain_autoconversion_rate(p3, qᶜˡ, Nᶜˡ, ρ, qʳ), subgrid)
end

# Accretion: KK2000 at the local cloud liquid and the cell's rain, active where both are present
@inline subgrid_accretion_rate(::Nothing, p3, qᶜˡ, qʳ, ρ) = rain_accretion_rate(p3, qᶜˡ, qʳ, ρ)

@inline subgrid_accretion_rate(d, p3, qᶜˡ, qʳ, ρ) =
    subgrid_accretion_rate(p3.warm_rain_scheme, d, p3, qᶜˡ, qʳ, ρ)

@inline function subgrid_accretion_rate(::KhairoutdinovKogan2000, d, p3, qᶜˡ, qʳ, ρ)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates
    ℂᵃᶜᶜʳ₁ = parameters.accretion_coefficient
    ℂᵃᶜᶜʳ₂ = parameters.accretion_exponent
    qʳ₊ = max(0, qʳ)
    rain_present = qʳ₊ >= p3.minimum_mass_mixing_ratio
    moment = departed_liquid_moment(d, ℂᵃᶜᶜʳ₂, p3.minimum_mass_mixing_ratio)
    subgrid = ifelse(rain_present, ℂᵃᶜᶜʳ₁ * moment * qʳ₊^ℂᵃᶜᶜʳ₂, zero(FT))
    return ifelse(d.degenerate, rain_accretion_rate(p3, qᶜˡ, qʳ, ρ), subgrid)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the fraction `P` of the cell where the local vapor excess over liquid ``δ(s)`` exceeds
`threshold` [kg kg⁻¹], and the mean of ``δ`` over that part times `P`, at the placement `d` (see
[`placed_cloud_liquid`](@ref)). ``δ(s) = s - ξˡ q⊖`` where ``s ≤ 0`` and ``(1 - λ) s`` where
``s > 0``; the set ``δ > c`` is the union of an interval of ``s ≤ 0`` and an interval of
``s > 0`` whose bounds follow from the sign of ``1 - λ``.
"""
@inline function vapor_excess_exceedance(d, threshold)
    FT = typeof(threshold)
    c = threshold
    shift = d.ξˡ * d.subsaturated_liquid
    r = 1 - d.λ
    # Subsaturated part: c + ξˡ q⊖ < s ≤ 0, where δ = s - ξˡ q⊖
    P⁻, M⁻ = mixture_interval_moments(d, min(c + shift, zero(FT)), zero(FT), true)
    # Saturated part, s > 0 and r s > c:
    #   r > 0:          s > max(c, 0) / r
    #   r ≤ 0, c < 0:   0 < s < c / r (unbounded when r = 0)
    #   r ≤ 0, c ≥ 0:   empty
    r₊ = ifelse(r > 0, r, one(FT))
    r₋ = ifelse(r < 0, r, -one(FT))
    lower = ifelse(r > 0, max(c, zero(FT)) / r₊, zero(FT))
    upper = ifelse(r > 0, zero(FT), ifelse(c < 0, c / r₋, zero(FT)))
    bounded = (r < 0) | ((r == 0) & (c >= 0))
    P⁺, M⁺ = mixture_interval_moments(d, lower, upper, bounded)
    return P⁻ + P⁺, M⁻ - shift * P⁻ + r * M⁺
end

# The activation threshold on the relative supersaturation over liquid
@inline activation_relative_threshold(::Nothing, p3) = p3.process_rates.activation_supersaturation_threshold
@inline activation_relative_threshold(aerosol, p3) = aerosol.activation_supersaturation_threshold

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `(; qᵛ, fraction)`: the vapor that droplet activation sees and the fraction of the cell
that activates. On the grid mean (`d = nothing`) they are `qᵛ` and 1. With a placement, activation
acts in the part of the cell where the local vapor excess over liquid exceeds the activation
threshold ``ℂᶠᵒʳᵐ₃ qᵛ⁺ˡ``: `fraction` is that part and `qᵛ` is ``qᵛ⁺ˡ`` plus the mean vapor excess
over it (see [`vapor_excess_exceedance`](@ref)).
"""
@inline activation_vapor(::Nothing, qᵛ, qᵛ⁺ˡ, relative_threshold) = (; qᵛ, fraction = one(qᵛ))

@inline function activation_vapor(d, qᵛ, qᵛ⁺ˡ, relative_threshold)
    FT = typeof(qᵛ)
    P, M = vapor_excess_exceedance(d, relative_threshold * qᵛ⁺ˡ)
    active = (P > 0) & !d.degenerate
    tail_excess = M / ifelse(active, P, one(FT))
    return (; qᵛ = ifelse(active, qᵛ⁺ˡ + tail_excess, qᵛ),
              fraction = ifelse(d.degenerate, one(FT), P))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the fraction of the cell where deposition nucleation of ice acts: where the local vapor
excess over ice reaches `ℂⁿᵘᶜˡ₂` times `qᵛ⁺ⁱ`. On the grid mean (`d = nothing`) it is 1 or 0, with the
supersaturation formed over `qᵛ⁺ⁱ` no smaller than `minimum_saturation`, as the grid-mean scheme forms it. With
a placement the local excess over ice is ``δ(s) + qᵛ⁺ˡ - qᵛ⁺ⁱ``, the local vapor excess over liquid
shifted by the difference of the saturations of the cell, so the fraction is that where
``δ(s) ≥ ℂⁿᵘᶜˡ₂ qᵛ⁺ⁱ - qᵛ⁺ˡ + qᵛ⁺ⁱ``.
"""
@inline function nucleating_fraction(::Nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)
    FT = typeof(qᵛ)
    𝒮ⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, minimum_saturation)
    return ifelse(𝒮ⁱ >= ℂⁿᵘᶜˡ₂, one(FT), zero(FT))
end

@inline nucleating_fraction(d::NamedTuple{names}, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation) where names =
    nucleating_fraction(Val(:mⁱ ∈ names), d, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)

# A placement without the excess over ice: the local excess over liquid shifted by the difference of the saturations.
# The threshold is ℂⁿᵘᶜˡ₂ qᵛ⁺ⁱ on the excess itself: the supersaturation is not formed, so the saturation is not a divisor.
@inline function nucleating_fraction(::Val{false}, d, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)
    threshold = ℂⁿᵘᶜˡ₂ * qᵛ⁺ⁱ - qᵛ⁺ˡ + qᵛ⁺ⁱ
    P, _ = vapor_excess_exceedance(d, threshold)
    return ifelse(d.degenerate, nucleating_fraction(nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation), P)
end

# The fraction where the local vapor excess over ice, δⁱ(s) = sⁱ(s) - ξⁱˡ qᶜˡ(s), reaches the threshold ℂⁿᵘᶜˡ₂ qᵛ⁺ⁱ
# (the supersaturation is not formed, so the saturation is not a divisor)
@inline function nucleating_fraction(::Val{true}, d, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)
    P, _ = ice_excess_exceedance(d, ℂⁿᵘᶜˡ₂ * qᵛ⁺ⁱ)
    return ifelse(d.degenerate, nucleating_fraction(nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation), P)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the fraction `P` of the cell where the local vapor excess over ice ``δⁱ`` reaches `threshold` [kg kg⁻¹], and the
mean of ``δⁱ`` over that part times `P`, at the placement `d` with the excess over ice of
[`subgrid_process_distribution`](@ref). Within component ``i`` the excess of the
distributed water over ice is ``sⁱ = bᵢ + βⁱᵢ s``, ``bᵢ = mⁱᵢ - βⁱᵢ mᵢ + rᵢ``, and the local vapor excess over ice is
``δⁱ = sⁱ - ξⁱˡ qᶜˡ(s)``: ``bᵢ + βⁱᵢ s - ξⁱˡ q⊖`` where ``s ≤ 0`` and ``bᵢ + (βⁱᵢ - ξⁱˡ λ / ξˡ) s`` where ``s > 0``, so at a
given residual the set ``δⁱ ≥`` `threshold` is an interval of ``s`` in each part. At given ``s`` the local excess is
Gaussian in the residual, independent of ``s``, with the exceedance ``Φ(z)`` and partial mean ``μ Φ(z) + σʳ φ(z)``,
smooth in ``s`` within each part, so in each part their mean over ``s`` is formed with the 24-point Gauss-Legendre rule
and weighted by the probability of the part (the exact form needs the bivariate normal distribution function);
without a residual, over the exact intervals.
"""
@inline function ice_excess_exceedance(d, threshold)
    FT = typeof(threshold)
    c = threshold
    qᶜˡ⁻ = d.subsaturated_liquid  # the cloud liquid of the subsaturated part, q⊖
    λⁱ = d.ξⁱˡ * d.λ / d.ξˡ
    R = convert(FT, 8)
    terms = map(d.a, d.m, d.σ, d.mⁱ, d.βⁱ, d.σʳ) do aᵢ, mᵢ, σᵢ, mⁱᵢ, βⁱᵢ, σʳᵢ
        bᵢ = mⁱᵢ - βⁱᵢ * mᵢ
        # At given s the local excess over ice is Gaussian in the residual: its exceedance and partial mean
        function conditional(s)
            μ = bᵢ + ifelse(s > 0, (βⁱᵢ - λⁱ) * s, βⁱᵢ * s - d.ξⁱˡ * qᶜˡ⁻)
            z = standard_score(μ, c, σʳᵢ)
            Φ = gaussian_upper_tail(-z)
            φ = gaussian_density(z)
            return (Φ, μ * Φ + σʳᵢ * φ)
        end
        # The subsaturated part s ≤ 0 and the saturated part s > 0, each its probability times the mean over it;
        # with a zero spread of s both means are `conditional(mᵢ)`
        z⁰ = standard_score(zero(FT), mᵢ, σᵢ)
        z⁰ᶜ = clamp(z⁰, -R, R)
        lower = gaussian_window_mean(conditional, -R, z⁰ᶜ, mᵢ, σᵢ)
        upper = gaussian_window_mean(conditional, z⁰ᶜ, R, mᵢ, σᵢ)
        P⁻ = gaussian_upper_tail(-z⁰)
        P⁺ = gaussian_upper_tail(z⁰)
        quadrature = (P⁻ * lower[1] + P⁺ * upper[1], P⁻ * lower[2] + P⁺ * upper[2])
        # Without a residual the conditional exceedance is a step, integrated exactly over intervals of s
        exact = ice_excess_exceedance_without_residual(d, c, mᵢ, σᵢ, bᵢ, βⁱᵢ, qᶜˡ⁻, λⁱ)
        resolved = σʳᵢ > 0
        (aᵢ * ifelse(resolved, quadrature[1], exact[1]), aᵢ * ifelse(resolved, quadrature[2], exact[2]))
    end
    return sum(map(first, terms)), sum(map(last, terms))
end

# The mean of f(m + σ z) over the standard normal variable z between zˡ and zᵘ,
#
#   Σⱼ wⱼ φ(zⱼ) f(m + σ zⱼ) / Σⱼ wⱼ φ(zⱼ),
#
# by the 24-point Gauss-Legendre rule over [zˡ, zᵘ], for f returning a tuple: the one rule of the partial moments, of
# the quadrature over the saturated part and of the exceedance of the excess over ice. The callers keep [zˡ, zᵘ]
# within |z| ≤ 8, where the Gaussian is e⁻³² ≈ 10⁻¹⁴ of its peak and the mass beyond is 10⁻¹⁵: an integration domain
# at the precision of the rule, not a bound on the integrand. The weights wⱼ φ(zⱼ) are positive there, so the mean is
# bounded by the values of f at the nodes; over an empty interval it is f at that point, and with σ = 0 it is f(m).
@inline function gaussian_window_mean(f, zˡ, zᵘ, m, σ)
    FT = typeof(m)
    h = (zᵘ - zˡ) / 2
    c = (zᵘ + zˡ) / 2
    terms = map(gauss_legendre_nodes, gauss_legendre_weights) do x, w
        z⁺ = c + h * convert(FT, x)
        z⁻ = c - h * convert(FT, x)
        n⁺ = convert(FT, w) * exp(-z⁺^2 / 2)
        n⁻ = convert(FT, w) * exp(-z⁻^2 / 2)
        (n⁺ + n⁻, map((y⁺, y⁻) -> y⁺ * n⁺ + y⁻ * n⁻, f(m + σ * z⁺), f(m + σ * z⁻)))
    end
    weight = sum(map(first, terms))
    sums = reduce((x, y) -> x .+ y, map(last, terms))
    return map(x -> x / weight, sums)
end

# One component without a residual: the set δⁱ ≥ c is an interval of s in each part
@inline function ice_excess_exceedance_without_residual(d, c, mᵢ, σᵢ, bᵢ, βⁱᵢ, qᶜˡ⁻, λⁱ)
    FT = typeof(c)
    # Subsaturated part: βⁱᵢ s ≥ Aᵢ with s ≤ 0
    Aᵢ = c + d.ξⁱˡ * qᶜˡ⁻ - bᵢ
    βⁱₛ = ifelse(βⁱᵢ > 0, βⁱᵢ, one(FT))
    P_sloped, M_sloped = gaussian_interval_moments(mᵢ, σᵢ, min(Aᵢ / βⁱₛ, zero(FT)), zero(FT), true)
    P_level, M_level = gaussian_lower_moments(mᵢ, σᵢ)
    reached = Aᵢ <= 0
    P⁻ = ifelse(βⁱᵢ > 0, P_sloped, ifelse(reached, P_level, zero(FT)))
    M⁻ = ifelse(βⁱᵢ > 0, M_sloped, ifelse(reached, M_level, zero(FT)))
    # Saturated part: rᵢ s ≥ c - bᵢ with s > 0
    rᵢ = βⁱᵢ - λⁱ
    P⁺, M⁺ = positive_linear_exceedance(mᵢ, σᵢ, rᵢ, c - bᵢ)
    return (P⁻ + P⁺, (bᵢ - d.ξⁱˡ * qᶜˡ⁻) * P⁻ + βⁱᵢ * M⁻ + bᵢ * P⁺ + rᵢ * M⁺)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the excess over ice that the vapor sinks of ice can draw on when the grid-mean vapor excess over ice
`grid_excess` of the placement `d` becomes `excess` (by the liquid rates and the forcing, uniformly over the cell):
the cell mean of the positive part of the local excess shifted by `excess - grid_excess`, and the positive part of
`excess` on the grid mean (`d = nothing`).
"""
@inline available_ice_excess(::Nothing, grid_excess, excess) = max(0, excess)

@inline available_ice_excess(d::NamedTuple{names}, grid_excess, excess) where names =
    available_ice_excess(Val(:mⁱ ∈ names), d, grid_excess, excess)

@inline available_ice_excess(::Val{false}, d, grid_excess, excess) = max(0, excess)

@inline function available_ice_excess(::Val{true}, d, grid_excess, excess)
    shift = excess - grid_excess
    P, M = ice_excess_exceedance(d, -shift)
    # (Round-off guard: M + shift P is the cell mean of the local excess over the part where it is not negative.)
    return ifelse(d.degenerate, max(0, excess), max(0, M + shift * P))
end

# The probability and first moment of one Gaussian over the part s > 0 where r s > c
@inline function positive_linear_exceedance(m, σ, r, c)
    FT = typeof(m)
    r₊ = ifelse(r > 0, r, one(FT))
    r₋ = ifelse(r < 0, r, -one(FT))
    lower = ifelse(r > 0, max(c, zero(FT)) / r₊, zero(FT))
    upper = ifelse(r > 0, zero(FT), ifelse(c < 0, c / r₋, zero(FT)))
    bounded = (r < 0) | ((r == 0) & (c >= 0))
    return gaussian_interval_moments(m, σ, lower, upper, bounded)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the parts of the cell over which the vapor exchange of ice is evaluated, at the placement `d` with the excess
over ice: for the saturated part (`+`, where the cloud liquid is placed as the equilibrium condensate) and the
subsaturated part (`-`), the fraction ``P`` of the cell, ``P δ`` and ``P δⁱ`` with ``δ`` and ``δⁱ`` the mean local vapor
excesses over liquid and over ice of the part, and ``P r`` with ``r`` the mean phase relaxation rate of the cloud
liquid of the part relative to that of the cell (``ε ∝ Nᶜˡ^{2/3} qᶜˡ^{1/3}`` at a fixed droplet shape:
``r⁺ = (χ / f⁺)^{2/3} (χ / ⟨s⁺⟩)^{1/3} ⟨s^{1/3}; s > 0⟩ / f⁺`` and ``r⁻ = ψ``). The ``P δ`` and ``P δⁱ`` of the two parts
sum to the grid-mean excesses ``qᵛ - qᵛ⁺ˡ`` and ``qᵛ - qᵛ⁺ⁱ``.
"""
@inline function ice_exchange_parts(d)
    FT = typeof(d.qᶜˡ)
    q̄ = d.qᶜˡ
    ψ = d.subsaturated_share
    qᶜˡ⁻ = d.subsaturated_liquid  # the cloud liquid of the subsaturated part, q⊖
    sⁱ⁺, sⁱ⁻ = ice_excess_part_sums(d)
    s⁺ = d.positive_excess
    saturated_rate = safe_ratio(positive_power(d.χ, convert(FT, 2//3)) *
                                positive_power(safe_ratio(d.χ, s⁺, zero(FT)), convert(FT, 1//3)) *
                                mixture_partial_moment(d, convert(FT, 1//3), zero(FT)),
                                positive_power(d.f⁺, convert(FT, 2//3)), zero(FT))
    saturated = (; P = d.f⁺, Pδ = (1 - d.λ) * s⁺, Pδⁱ = sⁱ⁺ - d.ξⁱˡ * d.χ * q̄, Pr = saturated_rate)
    subsaturated = (; P = d.f⁻, Pδ = d.negative_excess - d.ξˡ * d.f⁻ * qᶜˡ⁻, Pδⁱ = sⁱ⁻ - d.ξⁱˡ * d.f⁻ * qᶜˡ⁻, Pr = d.f⁻ * ψ)
    return saturated, subsaturated
end
