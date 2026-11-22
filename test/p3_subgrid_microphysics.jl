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
    N = ifelse.(s .> 0, d.χ / d.f⁺, (1 - d.χ) / (1 - d.f⁺))
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
        # Without mixing all of the liquid is in the saturated part, up to its equilibrium condensate
        d = cell_placement(statistics, s̄, qᶜˡ)
        @test d.χ ≈ min(1, d.positive_excess / (d.ξˡ * qᶜˡ))
        # Mixing moves it toward the uniform placement, χ = f⁺, and never past it
        χs = [cell_placement(statistics, s̄, qᶜˡ; ωᵐ).χ for ωᵐ in (1e-4, 1e-3, 1e-2, 1e-1, 1e3)]
        @test issorted(χs; rev = true)
        @test all(d.f⁺ - 1e-12 .≤ χs .≤ 1)
        @test χs[end] ≈ min(d.f⁺, d.positive_excess / (d.ξˡ * qᶜˡ)) rtol = 1e-3
        # Without liquid all of it (none) is in the saturated part
        @test cell_placement(statistics, s̄, 0.0; ωᵐ = 1e-2).χ == 1
    end

    # The balance of exchange and evaporation of the subsaturated liquid: ωᵐ (χ - f⁺) q̄ = ωˡ (1 - χ) |⟨s⁻⟩| / (1 - f⁺)
    d = cell_placement(gaussian(5e-4), 1e-4, 1e-4; ωᵐ = 5e-3)
    @test d.χ < 1
    @test 5e-3 * (d.χ - d.f⁺) * 1e-4 ≈ 0.05 * (1 - d.χ) * (-d.negative_excess) / (1 - d.f⁺) rtol = 1e-12
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
        for (β, γ) in ((2.47, -1.79), (1.15, 0.0))
            @test placed_cloud_liquid_moment(d, β, 0.0, γ) ≈ sum(w .* q .^ β .* N .^ γ) rtol = 1e-6
        end
    end

    # A narrow distribution about a saturated mean approaches the grid-mean power
    d = cell_placement(gaussian(1e-6), 1e-3, 1e-3)
    @test placed_cloud_liquid_moment(d, 2.47, 0.0) / 1e-3^2.47 ≈ 1 rtol = 1e-4
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
