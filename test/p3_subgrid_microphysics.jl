include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid distribution of the saturation excess in every P3 process (`AllProcesses`)
#####

using Breeze
using Breeze.AtmosphereModels: PrescribedStatistics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, AerosolActivation, AerosolMode,
                                                       SubgridCondensation, CloudCondensationOnly, AllProcesses,
                                                       P3MicrophysicalState, LiquidRelaxationStage,
                                                       compute_p3_process_rates, subgrid_process_distribution,
                                                       placed_cloud_liquid_moment, vapor_excess_exceedance,
                                                       cloud_condensation_correlation
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density, temperature
using Test

const PPP = PredictedParticleProperties

gaussian(σ) = (; μ = zero(σ), σ, skewness = zero(σ), f⁺ = one(σ) / 2)
mixture(a, μ₁, μ₂, σ₁, σ₂) = (; a = (a, 1 - a), μ = (μ₁, μ₂), σ = (σ₁, σ₂), f⁺ = a)

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

only_condensation = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(0.0)))
all_processes = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(0.0); processes = AllProcesses()))

# The distribution P3 forms for a cell with grid-mean saturation excess s̄ and cloud liquid qᶜˡ
function cell_distribution(statistics, s̄, qᶜˡ, constants; T = 285.0, qᵛ⁺ˡ = 1e-2)
    ξˡ = PPP.p3_liquid_psychrometric_correction(qᵛ⁺ˡ, T, constants)
    qᵛ = qᵛ⁺ˡ + s̄ - ξˡ * qᶜˡ
    Γ = cloud_condensation_correlation(all_processes.condensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, 9e4, 1.0, constants)
    return subgrid_process_distribution(all_processes.condensation, statistics, qᵛ, qᵛ⁺ˡ, qᶜˡ, T, constants, Γ)
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

@testset "P3 subgrid microphysics: construction" begin
    @test SubgridCondensation(PrescribedStatistics(1e-4)).processes === CloudCondensationOnly()
    @test all_processes.condensation.processes === AllProcesses()
    @test_throws ArgumentError SubgridCondensation(PrescribedStatistics(1e-4); processes = :all)
    @test subgrid_process_distribution(only_condensation.condensation, gaussian(1e-3), 1e-2, 1e-2, 1e-4, 285.0,
                                       ThermodynamicConstants(), 0.0) === nothing
end

@testset "P3 subgrid microphysics: zero spread recovers the grid-mean rates bitwise" begin
    constants = ThermodynamicConstants(Float64)
    for T in (285.0, 250.0)
        qᵛ⁺ = cell_saturation(9.5e-3, constants; T)
        for (excess, qᶜˡ, qʳ) in ((-5e-4, 1e-4, 0.0), (2e-4, 1e-3, 1e-4), (-5e-4, 0.0, 1e-5), (2e-4, 0.0, 0.0), (1e-5, 2e-3, 2e-4))
            qᵛ = qᵛ⁺ + excess * qᵛ⁺ / 1e-2
            for stage in (nothing, LiquidRelaxationStage(30.0, qᶜˡ, 2e8, true))
                reference = cell_rates(only_condensation, qᵛ, qᶜˡ, constants; T, qʳ, statistics = gaussian(0.0), stage)
                @test cell_rates(all_processes, qᵛ, qᶜˡ, constants; T, qʳ, statistics = gaussian(0.0), stage) === reference
            end
        end
    end
end

@testset "P3 subgrid microphysics: partial moments of the placed cloud liquid" begin
    constants = ThermodynamicConstants(Float64)
    β = 2.47
    for statistics in (gaussian(5e-4), mixture(0.1, 9e-4, -1e-4, 3e-4, 2e-4), mixture(0.05, 2e-3, -1e-4, 0.0, 3e-4))
        for (s̄, qᶜˡ) in ((-2e-4, 5e-5), (3e-4, 3e-4), (2e-3, 1.5e-3))
            d = cell_distribution(statistics, s̄, qᶜˡ, constants)
            # The placed liquid sums to the cloud liquid of the cell
            @test placed_cloud_liquid_moment(d, 1.0, 0.0) ≈ qᶜˡ rtol = 1e-12
            # A convex power of the placed liquid exceeds that of the mean
            @test placed_cloud_liquid_moment(d, β, 0.0) > qᶜˡ^β
            # and decreases with the threshold below which it is not counted
            @test placed_cloud_liquid_moment(d, β, 1e-4) ≤ placed_cloud_liquid_moment(d, β, 0.0)
        end
    end

    # Against a fine Riemann sum: E[qᶜˡ(s)^β] with qᶜˡ(s) = q̄ s⁺ / ⟨s⁺⟩
    statistics = mixture(0.2, 6e-4, -1.5e-4, 2e-4, 3e-4)
    d = cell_distribution(statistics, 1e-4, 2e-4, constants)
    s, w = riemann(d)
    s⁺ = sum(w .* max.(s, 0))
    @test placed_cloud_liquid_moment(d, β, 0.0) ≈ sum(w .* (2e-4 .* max.(s, 0) ./ s⁺) .^ β) rtol = 1e-6

    # A narrow distribution about a saturated mean approaches the grid-mean power
    for σ in (1e-5, 1e-6)
        d = cell_distribution(gaussian(σ), 1e-3, 1e-3, constants)
        @test placed_cloud_liquid_moment(d, β, 0.0) / 1e-3^β - 1 < 10 * (σ / 1e-3)^2
    end
end

@testset "P3 subgrid microphysics: the activating and nucleating fractions" begin
    constants = ThermodynamicConstants(Float64)
    # A subsaturated cell with no saturated tail activates nowhere
    d = cell_distribution(gaussian(1e-5), -1e-3, 0.0, constants)
    P, M = vapor_excess_exceedance(d, 0.0)
    @test P < 1e-30
    # A saturated cell without liquid: all of the cell exceeds a threshold far below its mean
    d = cell_distribution(gaussian(1e-5), 2e-3, 0.0, constants)
    P, M = vapor_excess_exceedance(d, 0.0)
    @test P ≈ 1
    @test M / P ≈ 2e-3 rtol = 1e-6
    # The fraction falls as the threshold rises
    d = cell_distribution(mixture(0.1, 8e-4, -1e-4, 3e-4, 2e-4), 1e-4, 1e-4, constants)
    fractions = [vapor_excess_exceedance(d, c)[1] for c in (-1e-3, -1e-4, 0.0, 1e-5, 1e-4)]
    @test issorted(fractions; rev = true)
    @test all(0 .≤ fractions .≤ 1)

    # Against a fine Riemann sum, with the local vapor excess δ(s) = s - λ s⁺
    s, w = riemann(d)
    δ = s .- d.λ .* max.(s, 0)
    for c in (-1e-4, 0.0, 2e-5)
        P, M = vapor_excess_exceedance(d, c)
        @test P ≈ sum(w[δ .> c]) rtol = 1e-4
        @test M ≈ sum((w .* δ)[δ .> c]) rtol = 1e-4
    end
end

@testset "P3 subgrid microphysics: rates" begin
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, constants)

    # Partly cloudy cell: autoconversion and accretion are larger than for uniform liquid
    qᵛ, qᶜˡ, qʳ = qᵛ⁺ - 2e-4, 6e-4, 1e-4
    statistics = mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4)
    only = cell_rates(only_condensation, qᵛ, qᶜˡ, constants; qʳ, statistics)
    all = cell_rates(all_processes, qᵛ, qᶜˡ, constants; qʳ, statistics)
    @test all.autoconversion > only.autoconversion
    @test all.accretion > only.accretion
    # The vapor exchange of cloud liquid and rain is unchanged
    @test all.condensation === only.condensation
    @test all.rain_condensation - all.rain_evaporation ≈ only.rain_condensation - only.rain_evaporation

    # Activation with aerosol: none in a subsaturated cell without a saturated tail, some with one
    aerosol = AerosolActivation(AerosolMode())
    p3 = P3Microphysics(Float64; aerosol, condensation = SubgridCondensation(PrescribedStatistics(0.0); processes = AllProcesses()))
    dry = cell_rates(p3, qᵛ⁺ - 5e-4, 0.0, constants; statistics = gaussian(1e-6), nᶜˡ = 0.0)
    @test dry.ccn_activation_number == 0
    moist = cell_rates(p3, qᵛ⁺ - 5e-4, 0.0, constants; statistics = gaussian(1e-3), nᶜˡ = 0.0)
    @test moist.ccn_activation_number > 0

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

@testset "P3 subgrid microphysics: water is conserved and θ is untouched [$FT]" for FT in (Float64,)
    constants = ThermodynamicConstants(FT)
    qᵛ⁺ = cell_saturation(9.5e-3, constants)
    statistics = mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4)
    for (excess, qᶜˡ, qʳ) in ((-2e-4, 6e-4, 1e-4), (2e-4, 0.0, 1e-5), (-5e-4, 1e-4, 0.0))
        qᵛ = qᵛ⁺ + excess
        for stage in (nothing, LiquidRelaxationStage(30.0, qᶜˡ, 2e8, true))
            rates = cell_rates(all_processes, qᵛ, qᶜˡ, constants; qʳ, statistics, stage)
            ℳ = P3MicrophysicalState(qᶜˡ, 2e8, qʳ, ifelse(qʳ > 0, 1e4, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            properties = PPP.p3_process_properties(all_processes, 1.0, ℳ)
            result = PPP.p3_tendencies(all_processes, 1.0, ℳ, properties, rates)
            water = result.tendency_ρqᵛ + result.tendency_ρqᶜˡ + result.tendency_ρqʳ + result.tendency_ρqⁱ + result.tendency_ρqʷⁱ
            scale = abs(result.tendency_ρqᵛ) + abs(result.tendency_ρqʳ) + eps(FT)
            @test abs(water) ≤ 10 * eps(FT) * scale
            @test PPP.p3_tendency_component(result, Val(:ρθ)) == 0
        end
    end
end
