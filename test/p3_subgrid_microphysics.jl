include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid distribution of the saturation excess in every P3 process (`AllProcesses`)
#####

using Breeze
using Breeze.AtmosphereModels: PrescribedStatistics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, AerosolActivation, AerosolMode,
                                                       SubgridCondensation, CloudCondensationOnly, AllProcesses,
                                                       SaturatedPartLiquid, MixingEvaporationBalance,
                                                       P3MicrophysicalState, LiquidRelaxationStage,
                                                       compute_p3_process_rates, subgrid_process_distribution,
                                                       placed_cloud_liquid, placed_cloud_liquid_moment,
                                                       placed_condensation_correlation, vapor_excess_exceedance
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density
using Enzyme: Enzyme
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Test

const PPP = PredictedParticleProperties

gaussian(σ) = (; μ = zero(σ), σ, skewness = zero(σ), f⁺ = one(σ) / 2)
mixture(a, μ₁, μ₂, σ₁, σ₂; ωᵐ = 0.0) = (; a = (a, 1 - a), μ = (μ₁, μ₂), σ = (σ₁, σ₂), f⁺ = a, ωᵐ)

function cell_state(qᵛ, qᶜˡ, qʳ, constants; T = 285.0)
    q = MoistureMassFractions(qᵛ, qᶜˡ + qʳ, 0.0)
    return with_temperature(LiquidIcePotentialTemperatureState(0.0, q, 1e5, 9e4), T, constants)
end

cell_saturation(qᵛ, constants; T = 285.0) =
    saturation_specific_humidity(T, density(cell_state(qᵛ, 0.0, 0.0, constants; T), constants), constants, PlanarLiquidSurface())

function cell_rates(p3, qᵛ, qᶜˡ, constants; T = 285.0, qʳ = 0.0, statistics = nothing, stage = nothing, nᶜˡ = 2e8)
    𝒰 = cell_state(qᵛ, qᶜˡ, qʳ, constants; T)
    ℳ = P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, ifelse(qʳ > 0, 1e4, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return compute_p3_process_rates(p3, 1.0, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, statistics, stage)
end

subgrid_p3(processes) = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(0.0); processes))
only_condensation = subgrid_p3(CloudCondensationOnly())
all_processes = subgrid_p3(AllProcesses())
saturated_part = subgrid_p3(AllProcesses(; liquid_placement = SaturatedPartLiquid()))

# The placement of a cell with grid-mean saturation excess s̄ and cloud liquid qᶜˡ
function cell_placement(statistics, s̄, qᶜˡ; placement = MixingEvaporationBalance(), ωˡ = 0.05, ωᵐ = 0.0,
                        ξˡ = 1.15, qᵛ⁺ˡ = 1e-2)
    a, m, σ = PPP.shifted_mixture(statistics, s̄)
    return placed_cloud_liquid(placement, a, m, σ, s̄, ξˡ, qᶜˡ, qᵛ⁺ˡ, ωˡ, ωᵐ)
end

# Points and weights of a midpoint rule over the mixture of `d`
function riemann(d; n = 400_000)
    lo = minimum(d.m .- 10 .* d.σ)
    hi = maximum(d.m .+ 10 .* d.σ)
    Δ = (hi - lo) / n
    s = lo .+ Δ .* ((1:n) .- 0.5)
    w = zeros(n)
    for (a, m, σ) in zip(d.a, d.m, d.σ)
        w .+= a .* Δ .* exp.(-((s .- m) ./ σ) .^ 2 ./ 2) ./ (sqrt(2π) * σ)
    end
    return s, w
end

# The placed liquid and droplet number (relative to the cell's) at the points s
function placed_liquid(d, s)
    q = ifelse.(s .> 0, d.χ * d.qᶜˡ .* s ./ d.positive_excess, d.subsaturated_liquid)
    N = ifelse.(s .> 0, d.χ / d.f⁺, d.subsaturated_share)
    return q, N
end

const cases = ((gaussian(5e-4), -2e-4, 5e-5), (gaussian(5e-4), 3e-4, 3e-4),
               (mixture(0.1, 9e-4, -1e-4, 3e-4, 2e-4), 1e-4, 2e-4),
               (mixture(0.05, 2e-3, -1e-4, 0.0, 3e-4), 2e-3, 1.5e-3))

@testset "P3 subgrid microphysics: construction" begin
    @test SubgridCondensation(PrescribedStatistics(1e-4)).processes === CloudCondensationOnly()
    @test AllProcesses().liquid_placement === MixingEvaporationBalance()
    @test_throws ArgumentError SubgridCondensation(PrescribedStatistics(1e-4); processes = :all)
    @test subgrid_process_distribution(only_condensation.condensation, gaussian(1e-3), 1e-2, 1e-2, 1e-4) === nothing

    # ℂᵐ is a coefficient of the closure's distribution, and sets the mixing rate it stores
    closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian())
    @test closure_parameters(closure).ℂᵐ == 1
    @test with_parameters(closure; ℂᵐ = 3).subgrid_distribution.ℂᵐ == 3
end

@testset "P3 subgrid microphysics: zero spread recovers the grid-mean rates bitwise" begin
    constants = ThermodynamicConstants(Float64)
    for T in (285.0, 250.0)
        qᵛ⁺ = cell_saturation(9.5e-3, constants; T)
        for (excess, qᶜˡ, qʳ) in ((-5e-4, 1e-4, 0.0), (2e-4, 1e-3, 1e-4), (-5e-4, 0.0, 1e-5), (2e-4, 0.0, 0.0), (1e-5, 2e-3, 2e-4))
            qᵛ = qᵛ⁺ + excess * qᵛ⁺ / 1e-2
            for stage in (nothing, LiquidRelaxationStage(30.0, qᶜˡ, 2e8, true)), p3 in (all_processes, saturated_part)
                reference = cell_rates(only_condensation, qᵛ, qᶜˡ, constants; T, qʳ, statistics = gaussian(0.0), stage)
                @test cell_rates(p3, qᵛ, qᶜˡ, constants; T, qʳ, statistics = gaussian(0.0), stage) === reference
            end
        end
    end
end

@testset "P3 subgrid microphysics: the fraction χ of the liquid in the saturated part" begin
    for (statistics, s̄, qᶜˡ) in cases
        d₁ = cell_placement(statistics, s̄, qᶜˡ; placement = SaturatedPartLiquid())
        @test d₁.χ == 1
        # Without mixing all of the liquid is in the saturated part, up to its equilibrium condensate;
        # the liquid beyond it is spread uniformly over the cell
        d = cell_placement(statistics, s̄, qᶜˡ)
        @test d.f⁺ + d.f⁻ ≈ 1
        @test d.χ ≈ 1 - d.f⁻ * max(0, 1 - d.positive_excess / (d.ξˡ * qᶜˡ))
        # Mixing moves it toward the uniform placement, χ = f⁺, and never past it
        χs = [cell_placement(statistics, s̄, qᶜˡ; ωᵐ).χ for ωᵐ in (1e-4, 1e-3, 1e-2, 1e-1, 1e3)]
        @test issorted(χs; rev = true)
        @test all(d.f⁺ - 1e-12 .≤ χs .≤ 1)
        @test χs[end] ≈ d.f⁺ rtol = 1e-3
        # Without liquid all of it (none) is in the saturated part
        @test cell_placement(statistics, s̄, 0.0; ωᵐ = 1e-2).χ == 1
    end

    # The balance of exchange and evaporation of the subsaturated liquid: ωᵐ (χ - f⁺) q̄ = ωˡ ψ |⟨s⁻⟩|
    d = cell_placement(gaussian(5e-4), 1e-4, 1e-4; ωᵐ = 5e-3)
    @test d.χ < 1
    @test 5e-3 * (d.χ - d.f⁺) * 1e-4 ≈ 0.05 * d.subsaturated_share * (-d.negative_excess) rtol = 1e-12
    @test 1 - d.χ ≈ d.f⁻ * d.subsaturated_share
end

# A vanishing subsaturated part (f⁻ → 0, the narrow components of perfectly correlated scalar moments
# in a saturated cell): the placement is continuous into f⁻ = 0, the subsaturated liquid stays
# below the cell mean, and the rates stay bounded, with liquid beyond the saturated part's
# equilibrium condensate (ξˡ q̄ᶜˡ > ⟨s⁺⟩) and within it.
@testset "P3 subgrid microphysics: the placement as the subsaturated part vanishes" begin
    σ = 1.5e-5
    for qᶜˡ in (1.4e-4, 0.9e-4), ωᵐ in (0.0, 1e-2), placement in (MixingEvaporationBalance(), SaturatedPartLiquid())
        # mean excess from 3σ to 40σ: f⁻ from 1e-3 to below floatmin, then exactly 0
        placements = [cell_placement(gaussian(σ), n * σ, qᶜˡ; ωᵐ, placement, ωˡ = 0.1) for n in (3, 5, 8, 10, 20, 30, 40)]
        for d in placements
            @test 0 ≤ d.f⁻ ≤ 1 && d.f⁺ + d.f⁻ ≈ 1
            @test d.negative_excess ≤ 0
            @test 0 ≤ d.subsaturated_share ≤ 1
            @test 0 ≤ d.subsaturated_liquid ≤ qᶜˡ
            @test d.f⁺ - 1e-12 ≤ d.χ ≤ 1
            Γ = placed_condensation_correlation(d, 0.0)
            @test isfinite(Γ)
            # the condensation rate at the placement differs from the grid mean by at most the
            # vapor excess of the saturated part and the liquid there
            @test abs(Γ) ≤ 10 * (d.positive_excess + d.ξˡ * qᶜˡ)
            @test all(isfinite, (placed_cloud_liquid_moment(d, 2.47, 1e-6, -1.79), placed_cloud_liquid_moment(d, 1.15, 1e-8),
                                 vapor_excess_exceedance(d, 1e-6)...))
        end
        @test placements[end].f⁻ == 0
        @test placements[end].χ == 1
        # Continuity into f⁻ = 0: where the subsaturated part is small, the placement and its rates are
        # those of the same cell without a subsaturated part, all of the liquid in the saturated part
        for d in placements
            d.f⁻ < 1e-12 || continue
            d₀ = merge(d, (; f⁻ = 0.0, negative_excess = 0.0, χ = 1.0, λ = d.ξˡ * qᶜˡ / d.positive_excess,
                             subsaturated_share = 0.0, subsaturated_liquid = 0.0))
            @test d.χ ≈ 1 atol = 1e-12
            @test placed_condensation_correlation(d, 0.0) ≈ placed_condensation_correlation(d₀, 0.0) rtol = 1e-9
            @test placed_cloud_liquid_moment(d, 2.47, 1e-6, -1.79) ≈ placed_cloud_liquid_moment(d₀, 2.47, 1e-6, -1.79) rtol = 1e-9
        end
    end
end

@testset "P3 subgrid microphysics: partial moments of the placed cloud liquid" begin
    for (statistics, s̄, qᶜˡ) in cases, ωᵐ in (0.0, 1e-2)
        d = cell_placement(statistics, s̄, qᶜˡ; ωᵐ)
        # The placed liquid sums to the cloud liquid of the cell
        @test placed_cloud_liquid_moment(d, 1.0, 0.0) ≈ qᶜˡ rtol = 1e-7
        @test placed_cloud_liquid_moment(d, 2.47, 1e-4) ≤ placed_cloud_liquid_moment(d, 2.47, 0.0)
    end

    # Against a fine Riemann sum of q(s)^β N(s)^γ, with the liquid and number of both parts
    statistics = mixture(0.2, 6e-4, -1.5e-4, 2e-4, 3e-4)
    for ωᵐ in (0.0, 1e-2)
        d = cell_placement(statistics, 1e-4, 2e-4; ωᵐ)
        s, w = riemann(d)
        q, N = placed_liquid(d, s)
        # (the 24-point Gauss-Legendre rule of the saturated part integrates s^β, whose higher
        # derivatives are singular at s = 0, to about 1e-6)
        for (β, γ) in ((2.47, -1.79), (1.15, 0.0))
            @test placed_cloud_liquid_moment(d, β, 0.0, γ) ≈ sum(w .* q .^ β .* N .^ γ) rtol = 3e-6
        end
    end

    # A narrow distribution about a saturated mean approaches the grid-mean power
    d = cell_placement(gaussian(1e-6), 1e-3, 1e-3)
    @test placed_cloud_liquid_moment(d, 2.47, 0.0) / 1e-3^2.47 ≈ 1 rtol = 1e-4
end

# The placement P3 formed in the lowest cell of a CASS column after the first stage, with cloud liquid of 10⁻¹⁶⁹:
# the saturated part's local liquid reaches the autoconversion threshold 10⁻⁸ only at s ~ 10¹⁵⁷, far above the
# window of its quadrature, where s^β overflows while the Gaussian underflows. The moment above the threshold is 0.
@testset "P3 subgrid microphysics: partial moments at near-zero cloud liquid" begin
    d = (; a = (0.9653098671723126, 0.0346901328276874), m = (-0.0016338398707576808, -0.001662869913752498),
           σ = (0.0007143286336670348, 0.0007155183369953402), ξˡ = 3.3017202704273685, qᶜˡ = 9.904855745303452e-170,
           f⁺ = 0.011055280719942148, f⁻ = 0.9889447192800578, positive_excess = 2.705510077543723e-6,
           negative_excess = -0.0016375524368827082, mean_excess = -0.0016348469268051644, χ = 0.011055280719942218,
           λ = 1.3363156351050678e-165, subsaturated_share = 1.0, subsaturated_liquid = 9.904855745303452e-170,
           qᵛ⁺ˡ = 0.015235428469920211, degenerate = false, mⁱ = (-0.0016334394009483652, -0.001674013643306868),
           k = (1.3974413392531293, 1.3978863069081005), σʳ = (0.0, 0.0), ξⁱˡ = 3.654082217150073, γⁱ = 0.0,
           ν = 0.24128088186510618)
    β, γ, threshold = 2.47, -1.79, 1e-8
    @test PPP.gaussian_partial_moment(β, d.m[1], d.σ[1], 2.1863818530467257e157) == 0
    @test placed_cloud_liquid_moment(d, β, threshold, γ) == 0
    @test PPP.departed_liquid_moment(d, β, threshold, γ) == 0
    @test PPP.subgrid_autoconversion_rate(d, all_processes, d.qᶜˡ, 1.2813550029476943e-5, 1.1488756787969368, 0.0) == 0
    # Without the threshold the moment is that of the placed liquid, ~ (10⁻¹⁶⁹)^β, which underflows
    @test placed_cloud_liquid_moment(d, β, 0.0, γ) == 0
    @test placed_cloud_liquid_moment(d, 1.0, 0.0) ≈ d.qᶜˡ rtol = 1e-7
end

@testset "P3 subgrid microphysics: condensation at the local droplet population" begin
    # Against a fine Riemann sum of the local rate (ε(s) / ε̄) δ(s), ε ∝ N^{2/3} q^{1/3}
    for (statistics, s̄, qᶜˡ) in cases[1:3], ωᵐ in (0.0, 1e-2)
        d = cell_placement(statistics, s̄, qᶜˡ; ωᵐ)
        s, w = riemann(d)
        q, N = placed_liquid(d, s)
        rate = sum(w .* N .^ (2 / 3) .* (q ./ qᶜˡ) .^ (1 / 3) .* (s .- d.ξˡ .* q))
        μ = s̄ - d.ξˡ * qᶜˡ
        @test μ + placed_condensation_correlation(d, 0.0) ≈ rate rtol = 1e-5
    end
    # The correlation is returned unchanged on the grid mean
    @test placed_condensation_correlation(nothing, 1.5) == 1.5
    @test placed_condensation_correlation(cell_placement(gaussian(0.0), 1e-4, 1e-4), 1.5) == 1.5
end

@testset "P3 subgrid microphysics: the activating and nucleating fractions" begin
    # A subsaturated cell with no saturated tail activates nowhere
    P, _ = vapor_excess_exceedance(cell_placement(gaussian(1e-5), -1e-3, 0.0), 0.0)
    @test P < 1e-30
    # A saturated cell without liquid: all of the cell exceeds a threshold far below its mean
    P, M = vapor_excess_exceedance(cell_placement(gaussian(1e-5), 2e-3, 0.0), 0.0)
    @test P ≈ 1
    @test M / P ≈ 2e-3 rtol = 1e-6

    # Against a fine Riemann sum, with the local vapor excess δ(s) = s - ξˡ q(s)
    for ωᵐ in (0.0, 1e-2)
        d = cell_placement(mixture(0.1, 8e-4, -1e-4, 3e-4, 2e-4), 1e-4, 1e-4; ωᵐ)
        fractions = [vapor_excess_exceedance(d, c)[1] for c in (-1e-3, -1e-4, 0.0, 1e-5, 1e-4)]
        @test issorted(fractions; rev = true)
        s, w = riemann(d)
        q, _ = placed_liquid(d, s)
        δ = s .- d.ξˡ .* q
        for c in (-1e-4, 0.0, 2e-5)
            P, M = vapor_excess_exceedance(d, c)
            @test P ≈ sum(w[δ .> c]) rtol = 1e-4
            @test M ≈ sum((w .* δ)[δ .> c]) rtol = 1e-4
        end
    end
end

@testset "P3 subgrid microphysics: rates" begin
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, constants)

    # The vapor exchange of rain is unchanged by the placement
    qᵛ, qᶜˡ, qʳ = qᵛ⁺ - 2e-4, 6e-4, 1e-4
    statistics = mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4)
    only = cell_rates(only_condensation, qᵛ, qᶜˡ, constants; qʳ, statistics)
    for p3 in (all_processes, saturated_part)
        rates = cell_rates(p3, qᵛ, qᶜˡ, constants; qʳ, statistics)
        @test isfinite(rates.autoconversion) && isfinite(rates.accretion) && isfinite(rates.condensation)
        @test rates.autoconversion != only.autoconversion
    end
    # Where mixing is fast against the droplets' evaporation, it moves liquid into the subsaturated
    # part, which changes the condensation (both of its parts, of either sign)
    mixed = cell_rates(all_processes, qᵛ, 1e-4, constants; qʳ, statistics = merge(statistics, (; ωᵐ = 0.5)))
    unmixed = cell_rates(all_processes, qᵛ, 1e-4, constants; qʳ, statistics)
    @test mixed.condensation != unmixed.condensation

    # Activation with aerosol: none in a subsaturated cell without a saturated tail, some with one
    aerosol = AerosolActivation(AerosolMode())
    p3 = P3Microphysics(Float64; aerosol, condensation = SubgridCondensation(PrescribedStatistics(0.0); processes = AllProcesses()))
    @test cell_rates(p3, qᵛ⁺ - 5e-4, 0.0, constants; statistics = gaussian(1e-6), nᶜˡ = 0.0).ccn_activation_number == 0
    @test cell_rates(p3, qᵛ⁺ - 5e-4, 0.0, constants; statistics = gaussian(1e-3), nᶜˡ = 0.0).ccn_activation_number > 0

    # Ice nucleation at 250 K, below the nucleation threshold on the grid mean: only the tail nucleates.
    # (The cell is supersaturated over ice, so the grid-mean cap on the vapor sinks of ice allows it.)
    T = 250.0
    qᵛ⁺ˡ = cell_saturation(1e-3, constants; T)
    qᵛ⁺ⁱ = PPP.p3_ice_saturation_specific_humidity(T, density(cell_state(1e-3, 0.0, 0.0, constants; T), constants),
                                                   constants, all_processes.process_rates.freezing_temperature, qᵛ⁺ˡ)
    qᵛ = 1.02 * qᵛ⁺ⁱ
    @test cell_rates(only_condensation, qᵛ, 0.0, constants; T, statistics = gaussian(qᵛ⁺ⁱ / 10)).nucleation_number == 0
    @test cell_rates(all_processes, qᵛ, 0.0, constants; T, statistics = gaussian(1e-9)).nucleation_number == 0
    @test cell_rates(all_processes, qᵛ, 0.0, constants; T, statistics = gaussian(qᵛ⁺ⁱ / 10)).nucleation_number > 0
end

@testset "P3 subgrid microphysics: water is conserved and θ is untouched" begin
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, constants)
    for statistics in (mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4), mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4; ωᵐ = 1e-2))
        for (excess, qᶜˡ, qʳ) in ((-2e-4, 6e-4, 1e-4), (2e-4, 0.0, 1e-5), (-5e-4, 1e-4, 0.0))
            qᵛ = qᵛ⁺ + excess
            for stage in (nothing, LiquidRelaxationStage(30.0, qᶜˡ, 2e8, true))
                rates = cell_rates(all_processes, qᵛ, qᶜˡ, constants; qʳ, statistics, stage)
                ℳ = P3MicrophysicalState(qᶜˡ, 2e8, qʳ, ifelse(qʳ > 0, 1e4, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                properties = PPP.p3_process_properties(all_processes, 1.0, ℳ)
                result = PPP.p3_tendencies(all_processes, 1.0, ℳ, properties, rates)
                water = result.tendency_ρqᵛ + result.tendency_ρqᶜˡ + result.tendency_ρqʳ + result.tendency_ρqⁱ + result.tendency_ρqʷⁱ
                scale = abs(result.tendency_ρqᵛ) + abs(result.tendency_ρqʳ) + eps()
                @test abs(water) ≤ 10 * eps() * scale
                @test PPP.p3_tendency_component(result, Val(:ρθ)) == 0
            end
        end
    end
end

reverse_gradient(f, x) = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.set_strong_zero(Enzyme.Reverse)), Enzyme.Const(f),
                                         Enzyme.Active, Enzyme.Active(x))[1][1]

@testset "P3 subgrid microphysics: Enzyme against finite differences in ℂᵐ" begin
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, constants)
    qᵛ, qᶜˡ, qʳ = qᵛ⁺ - 2e-4, 1e-4, 1e-4
    # The mixing rate the closure stores, √e / (ℂᵐ ℓ), for e = 1 m² s⁻² and ℓ = 2 m: fast enough
    # against the evaporation of the droplets that χ is set by the balance, not by its bound
    statistics(ℂᵐ) = mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4; ωᵐ = 1 / (ℂᵐ * 2))
    for name in (:condensation, :autoconversion, :accretion)
        f = ℂᵐ -> getproperty(cell_rates(all_processes, qᵛ, qᶜˡ, constants; qʳ, statistics = statistics(ℂᵐ)), name)
        ad = reverse_gradient(f, 1.0)
        fd = (f(1 + 1e-6) - f(1 - 1e-6)) / 2e-6
        @test isfinite(ad)
        @test ad != 0
        @test ad ≈ fd rtol = 1e-4
    end
end

@testset "P3 subgrid microphysics: a column with the closure's distribution and mixing rate" begin
    grid = RectilinearGrid(CPU(); size=16, z=(0, 1600), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation)
    for ℂᵐ in (1, 4)
        closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(; ℂᵐ))
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics, closure)
        set!(model; θ = z -> ifelse(z < 800, 290.0, 290.0 + 8 * (1 - exp(-(z - 800) / 100))),
                    qᵗ = z -> ifelse(z < 800, 10e-3, 4e-3), u = z -> 5 * z / 1600, ρe = z -> 1.2 * 0.3 * exp(-z / 800))
        update_state!(model)
        cloud = model.closure_fields.subgrid_cloud.cloud
        ωᵐ = Array(interior(cloud.ωᵐ, 1, 1, :))
        @test all(ωᵐ .≥ 0) && any(ωᵐ .> 0)
        for _ in 1:3
            time_step!(model, 10)
        end
        @test all(isfinite, Array(interior(model.microphysical_fields.ρqᶜˡ)))
    end
end

# Perfectly correlated scalar moments: with VarianceStabilityFunctions the local covariance of θˡⁱ and qᵗ is
# 𝕏ᵛ √(θˡⁱ′² qᵗ′²), so ℂᵛ₃ = ±1 makes the joint distribution of the components degenerate (the saturation excess of
# a component can have no spread). The column must stay finite there and near there, and the distribution P3 reads
# must stay realizable.
@testset "P3 subgrid microphysics: a column at perfectly and nearly perfectly correlated θˡⁱ and qᵗ" begin
    grid = RectilinearGrid(CPU(); size=24, z=(0, 1800), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=289)
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation)
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.2 * 1.5e-2)),
                             ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.2 * 1.5e-5)))
    for 𝕏ᵛ in (1, -1, 0.999, -0.999)
        variance_stability_functions = VarianceStabilityFunctions(; ℂᵛ₃⁻ = 𝕏ᵛ, ℂᵛ₃ᵐ = 𝕏ᵛ, ℂᵛ₃⁺ = 𝕏ᵛ)
        closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(; variance_stability_functions,
                                                                                       variance_transport = ScalarVarianceTransport()),
                                              static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
        model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), microphysics, closure, boundary_conditions)
        # A stratocumulus-topped mixed layer, saturated below its inversion
        set!(model; θ = z -> z < 840 ? 289.0 : 297.5 + cbrt(z - 840), qᵗ = z -> z < 840 ? 9.5e-3 : 1.5e-3, u = 7,
                    ρe = z -> 1.2 * 0.5 * exp(-z / 900))
        for _ in 1:20
            time_step!(model, 60)
        end
        fields = Oceananigans.prognostic_fields(model)
        @test all(name -> all(isfinite, Array(interior(fields[name]))), keys(fields))
        cloud = model.closure_fields.subgrid_cloud.cloud
        @test any(Array(interior(model.microphysical_fields.qᶜˡ)) .> 0)
        @test all(σ -> all(x -> isfinite(x) && x ≥ 0, Array(interior(σ))), (cloud.σ₁, cloud.σ₂))
        moments = model.closure_fields.subgrid_cloud.moments
        θ², q², θq = map(f -> Array(interior(f)), (moments.θˡⁱ′², moments.qᵗ′², moments.θˡⁱ′qᵗ′))
        @test all(θq .^ 2 .≤ θ² .* q² .* (1 + 1e-10) .+ floatmin())
    end
end
