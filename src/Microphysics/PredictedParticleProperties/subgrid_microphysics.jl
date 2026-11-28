#####
##### The subgrid distribution of the saturation excess in every P3 process
#####

"""
$(TYPEDEF)

The subgrid distribution of [`SubgridCondensation`](@ref) enters only the condensation and evaporation of cloud liquid.
"""
struct CloudCondensationOnly end

struct MixingEvaporationBalance end

"""
$(TYPEDEF)

The subgrid distribution of [`SubgridCondensation`](@ref) enters every P3 process that is nonlinear in the local
state: condensation, droplet activation, ice nucleation and vapor growth, autoconversion and accretion. The cloud
liquid is placed between the saturated and subsaturated parts of the distribution by the balance of mixing and
evaporation.
"""
struct AllProcesses{P}
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

@inline mixing_rate(statistics::NamedTuple{names}, FT) where names = mixing_rate(Val(:ωᵐ ∈ names), statistics, FT)
@inline mixing_rate(::Val{true}, statistics, FT) = convert(FT, statistics.ωᵐ)
@inline mixing_rate(::Val{false}, statistics, FT) = zero(FT)

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
    ξⁱˡ = p3_ice_liquid_psychrometric_correction(qᵛ⁺ⁱ, q, T, constants)
    s̄ⁱ = qᵛ - qᵛ⁺ⁱ + ξⁱˡ * q̄ᶜˡ
    mⁱ, βⁱ, σʳ = ice_excess_mixture(statistics, d, s̄ⁱ)
    dⁱ = merge(d, (; mⁱ, βⁱ, σʳ, ξⁱˡ))
    γⁱ = ice_placement_contrast(statistics, dⁱ, max(0, qⁱ))
    ν = liquid_departure_scale(statistics, dⁱ)
    return merge(dⁱ, (; γⁱ, ν))
end

#####
##### The ice and the departure of the cloud liquid, from their covariance and variance
#####

@inline ice_placement_contrast(statistics::NamedTuple{names}, d, qⁱ) where names =
    ice_placement_contrast(Val(:ice_covariance ∈ names), statistics, d, qⁱ)
@inline ice_placement_contrast(::Val{false}, statistics, d, qⁱ) = zero(qⁱ)

@inline function ice_placement_contrast(::Val{true}, statistics, d, qⁱ)
    FT = typeof(qⁱ)
    C = convert(FT, statistics.ice_covariance)
    S⁺, S⁻ = ice_excess_part_sums(d)
    contrast = d.f⁻ * S⁺ - d.f⁺ * S⁻
    γ = bounded_ratio(C, ifelse(contrast > 0, qⁱ * contrast, zero(FT)), max(d.f⁺, d.f⁻))
    return ifelse(contrast > 0, γ, zero(FT))
end

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

@inline ice_part_scales(d::NamedTuple{names}) where names = ice_part_scales(Val(:γⁱ ∈ names), d)
@inline ice_part_scales(::Val{false}, d) = (one(d.qᶜˡ), one(d.qᶜˡ))
@inline ice_part_scales(::Val{true}, d) = (1 + d.γⁱ * d.f⁻, 1 - d.γⁱ * d.f⁺)

@inline liquid_departure_scale(statistics::NamedTuple{names}, d) where names =
    liquid_departure_scale(Val(:departure_variance ∈ names), statistics, d)
@inline liquid_departure_scale(::Val{false}, statistics, d) = zero(d.qᶜˡ)

@inline function liquid_departure_scale(::Val{true}, statistics, d)
    FT = typeof(d.qᶜˡ)
    second_moment = placed_cloud_liquid_moment(d, convert(FT, 2), zero(FT))
    return bounded_ratio(sqrt(max(0, convert(FT, statistics.departure_variance))), sqrt(max(0, second_moment)),
                         convert(FT, last(gauss_hermite_nodes)))
end

const gauss_hermite_nodes = (0.5390798113513751, 1.6365190424351082, 2.8024858612875416, 4.144547186125894)
const gauss_hermite_weights = (0.37301225767907614, 0.11723990766175905, 0.009635220120788268, 0.00011261453837536777)

@inline departed_liquid_moment(d::NamedTuple{names}, β, threshold, γ = zero(β)) where names =
    departed_liquid_moment(Val(:ν ∈ names), d, β, threshold, γ)

@inline departed_liquid_moment(::Val{false}, d, β, threshold, γ) = placed_cloud_liquid_moment(d, β, threshold, γ)

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

@inline function p3_ice_liquid_psychrometric_correction(qᵛ⁺ⁱ, q, T, constants)
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ∂qᵛ⁺ⁱ∂T = qᵛ⁺ⁱ * saturation_vapor_pressure_log_slope(T, constants, PlanarIceSurface())
    return 1 + ℒˡ * ∂qᵛ⁺ⁱ∂T / cᵖᵐ
end

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
    βⁱ = map((𝕏ⁱᵢ, σⁱᵢ, σᵢ) -> safe_ratio(𝕏ⁱᵢ * σⁱᵢ, σᵢ, one(FT)), 𝕏ⁱ, σⁱ, d.σ)
    σʳ = map((𝕏ⁱᵢ, σⁱᵢ, σᵢ) -> ifelse(σᵢ > 0, σⁱᵢ * sqrt(max(0, 1 - 𝕏ⁱᵢ * 𝕏ⁱᵢ)), σⁱᵢ), 𝕏ⁱ, σⁱ, d.σ)
    return mⁱ, βⁱ, σʳ
end

#####
##### Gaussian integrals over intervals of s
#####

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

@inline function gaussian_interval_moments(m, σ, l, u, bounded)
    FT = typeof(m)
    zˡ = standard_score(l, m, σ)
    zᵘ = standard_score(u, m, σ)
    tailᵘ = ifelse(bounded, gaussian_upper_tail(zᵘ), zero(FT))
    φᵘ = ifelse(bounded, gaussian_density(zᵘ), zero(FT))
    P = max(0, gaussian_upper_tail(zˡ) - tailᵘ)
    M = m * P + σ * (gaussian_density(zˡ) - φᵘ)
    nonempty = !bounded | (u > l)
    return ifelse(nonempty, P, zero(FT)), ifelse(nonempty, M, zero(FT))
end

@inline function mixture_interval_moments(a, m, σ, l, u, bounded)
    moments = map((mᵢ, σᵢ) -> gaussian_interval_moments(mᵢ, σᵢ, l, u, bounded), m, σ)
    P = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[1], a, moments))
    M = sum(map((aᵢ, Mᵢ) -> aᵢ * Mᵢ[2], a, moments))
    return P, M
end

@inline mixture_interval_moments(d::NamedTuple, l, u, bounded) = mixture_interval_moments(d.a, d.m, d.σ, l, u, bounded)

@inline function gaussian_lower_moments(m, σ)
    z = standard_score(m, zero(m), σ)
    P = gaussian_upper_tail(z)
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

const gauss_legendre_nodes = (0.06405689286260585, 0.1911188674736165, 0.31504267969616384,
                              0.4337935076260454, 0.5454214713888397, 0.6480936519369753,
                              0.7401241915785544, 0.820001985973903, 0.8864155270044011,
                              0.9382745520027327, 0.9747285559713095, 0.9951872199970213)

const gauss_legendre_weights = (0.12793819534675185, 0.12583745634682753, 0.12167047292780374,
                                0.11550566805372615, 0.10744427011596513, 0.09761865210411456,
                                0.08619016153195298, 0.07334648141108072, 0.059298584915437,
                                0.04427743881741972, 0.028531388628933545, 0.012341229799987141)

@inline function gaussian_partial_moment(β, m, σ, l)
    FT = typeof(m)
    R = convert(FT, 8)
    zˡ = standard_score(l, m, σ)
    mean, = gaussian_window_mean(s -> (positive_power(s, β),), clamp(zˡ, -R, R), R, m, σ)
    return gaussian_upper_tail(zˡ) * mean
end

@inline function positive_power(s, β)
    positive = s > 0
    return ifelse(positive, ifelse(positive, s, one(s))^β, zero(s))
end

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

@inline safe_ratio(a, b, fallback) = ifelse(b > 0, a / ifelse(b > 0, b, one(b)), fallback)

#####
##### The placement of the cloud liquid
#####

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

@inline function subsaturated_liquid_share(::MixingEvaporationBalance, f⁻, s⁺, s⁻, ξˡ, q̄ᶜˡ, ωˡ, ωᵐ)
    FT = typeof(f⁻)
    evaporation = ωˡ * (-s⁻)
    exchange = f⁻ * ωᵐ * q̄ᶜˡ
    ψᵇ = safe_ratio(exchange, evaporation + exchange, zero(FT))
    ψᵉ = max(0, 1 - safe_ratio(s⁺, ξˡ * q̄ᶜˡ, one(FT)))
    return ψᵉ + (1 - ψᵉ) * ψᵇ
end

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

@inline function vapor_excess_exceedance(d, threshold)
    FT = typeof(threshold)
    c = threshold
    shift = d.ξˡ * d.subsaturated_liquid
    r = 1 - d.λ
    P⁻, M⁻ = mixture_interval_moments(d, min(c + shift, zero(FT)), zero(FT), true)
    r₊ = ifelse(r > 0, r, one(FT))
    r₋ = ifelse(r < 0, r, -one(FT))
    lower = ifelse(r > 0, max(c, zero(FT)) / r₊, zero(FT))
    upper = ifelse(r > 0, zero(FT), ifelse(c < 0, c / r₋, zero(FT)))
    bounded = (r < 0) | ((r == 0) & (c >= 0))
    P⁺, M⁺ = mixture_interval_moments(d, lower, upper, bounded)
    return P⁻ + P⁺, M⁻ - shift * P⁻ + r * M⁺
end

@inline activation_relative_threshold(::Nothing, p3) = p3.process_rates.activation_supersaturation_threshold
@inline activation_relative_threshold(aerosol, p3) = aerosol.activation_supersaturation_threshold

@inline activation_vapor(::Nothing, qᵛ, qᵛ⁺ˡ, relative_threshold) = (; qᵛ, fraction = one(qᵛ))

@inline function activation_vapor(d, qᵛ, qᵛ⁺ˡ, relative_threshold)
    FT = typeof(qᵛ)
    P, M = vapor_excess_exceedance(d, relative_threshold * qᵛ⁺ˡ)
    active = (P > 0) & !d.degenerate
    tail_excess = M / ifelse(active, P, one(FT))
    return (; qᵛ = ifelse(active, qᵛ⁺ˡ + tail_excess, qᵛ),
              fraction = ifelse(d.degenerate, one(FT), P))
end

@inline function nucleating_fraction(::Nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)
    FT = typeof(qᵛ)
    𝒮ⁱ = (qᵛ - qᵛ⁺ⁱ) / max(qᵛ⁺ⁱ, minimum_saturation)
    return ifelse(𝒮ⁱ >= ℂⁿᵘᶜˡ₂, one(FT), zero(FT))
end

@inline nucleating_fraction(d::NamedTuple{names}, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation) where names =
    nucleating_fraction(Val(:mⁱ ∈ names), d, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)

@inline function nucleating_fraction(::Val{false}, d, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)
    threshold = ℂⁿᵘᶜˡ₂ * qᵛ⁺ⁱ - qᵛ⁺ˡ + qᵛ⁺ⁱ
    P, _ = vapor_excess_exceedance(d, threshold)
    return ifelse(d.degenerate, nucleating_fraction(nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation), P)
end

@inline function nucleating_fraction(::Val{true}, d, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation)
    P, _ = ice_excess_exceedance(d, ℂⁿᵘᶜˡ₂ * qᵛ⁺ⁱ)
    return ifelse(d.degenerate, nucleating_fraction(nothing, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, ℂⁿᵘᶜˡ₂, minimum_saturation), P)
end

@inline function ice_excess_exceedance(d, threshold)
    FT = typeof(threshold)
    c = threshold
    qᶜˡ⁻ = d.subsaturated_liquid  # the cloud liquid of the subsaturated part, q⊖
    λⁱ = d.ξⁱˡ * d.λ / d.ξˡ
    R = convert(FT, 8)
    terms = map(d.a, d.m, d.σ, d.mⁱ, d.βⁱ, d.σʳ) do aᵢ, mᵢ, σᵢ, mⁱᵢ, βⁱᵢ, σʳᵢ
        bᵢ = mⁱᵢ - βⁱᵢ * mᵢ
        function conditional(s)
            μ = bᵢ + ifelse(s > 0, (βⁱᵢ - λⁱ) * s, βⁱᵢ * s - d.ξⁱˡ * qᶜˡ⁻)
            z = standard_score(μ, c, σʳᵢ)
            Φ = gaussian_upper_tail(-z)
            φ = gaussian_density(z)
            return (Φ, μ * Φ + σʳᵢ * φ)
        end
        z⁰ = standard_score(zero(FT), mᵢ, σᵢ)
        z⁰ᶜ = clamp(z⁰, -R, R)
        lower = gaussian_window_mean(conditional, -R, z⁰ᶜ, mᵢ, σᵢ)
        upper = gaussian_window_mean(conditional, z⁰ᶜ, R, mᵢ, σᵢ)
        P⁻ = gaussian_upper_tail(-z⁰)
        P⁺ = gaussian_upper_tail(z⁰)
        quadrature = (P⁻ * lower[1] + P⁺ * upper[1], P⁻ * lower[2] + P⁺ * upper[2])
        exact = ice_excess_exceedance_without_residual(d, c, mᵢ, σᵢ, bᵢ, βⁱᵢ, qᶜˡ⁻, λⁱ)
        resolved = σʳᵢ > 0
        (aᵢ * ifelse(resolved, quadrature[1], exact[1]), aᵢ * ifelse(resolved, quadrature[2], exact[2]))
    end
    return sum(map(first, terms)), sum(map(last, terms))
end

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

@inline function ice_excess_exceedance_without_residual(d, c, mᵢ, σᵢ, bᵢ, βⁱᵢ, qᶜˡ⁻, λⁱ)
    FT = typeof(c)
    Aᵢ = c + d.ξⁱˡ * qᶜˡ⁻ - bᵢ
    βⁱₛ = ifelse(βⁱᵢ > 0, βⁱᵢ, one(FT))
    P_sloped, M_sloped = gaussian_interval_moments(mᵢ, σᵢ, min(Aᵢ / βⁱₛ, zero(FT)), zero(FT), true)
    P_level, M_level = gaussian_lower_moments(mᵢ, σᵢ)
    reached = Aᵢ <= 0
    P⁻ = ifelse(βⁱᵢ > 0, P_sloped, ifelse(reached, P_level, zero(FT)))
    M⁻ = ifelse(βⁱᵢ > 0, M_sloped, ifelse(reached, M_level, zero(FT)))
    rᵢ = βⁱᵢ - λⁱ
    P⁺, M⁺ = positive_linear_exceedance(mᵢ, σᵢ, rᵢ, c - bᵢ)
    return (P⁻ + P⁺, (bᵢ - d.ξⁱˡ * qᶜˡ⁻) * P⁻ + βⁱᵢ * M⁻ + bᵢ * P⁺ + rᵢ * M⁺)
end

@inline function positive_linear_exceedance(m, σ, r, c)
    FT = typeof(m)
    r₊ = ifelse(r > 0, r, one(FT))
    r₋ = ifelse(r < 0, r, -one(FT))
    lower = ifelse(r > 0, max(c, zero(FT)) / r₊, zero(FT))
    upper = ifelse(r > 0, zero(FT), ifelse(c < 0, c / r₋, zero(FT)))
    bounded = (r < 0) | ((r == 0) & (c >= 0))
    return gaussian_interval_moments(m, σ, lower, upper, bounded)
end

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
