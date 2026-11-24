include(joinpath(@__DIR__, "setup.jl"))

#####
##### The ice of P3 with the subgrid distribution: the saturation excess over ice for its nucleation and vapor growth,
##### and the ice term of the latent response of the turbulence closure
#####

using Breeze
using Breeze.AtmosphereModels: PrescribedStatistics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, SubgridCondensation, AllProcesses,
                                                       CloudCondensationOnly, P3MicrophysicalState,
                                                       compute_p3_process_rates, placed_cloud_liquid,
                                                       MixingEvaporationBalance
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density
using Breeze.TurbulenceClosures: TurbulenceClosureStatistics
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Grids: znodes
using Test

const PPP = PredictedParticleProperties

ice_mixture(a, μ₁, μ₂, σ₁, σ₂, μⁱ₁, μⁱ₂, σⁱ₁, σⁱ₂; ωᵐ = 0.0) =
    (; a = (a, 1 - a), μ = (μ₁, μ₂), σ = (σ₁, σ₂), f⁺ = a, ωᵐ, μⁱ = (μⁱ₁, μⁱ₂), σⁱ = (σⁱ₁, σⁱ₂))

function cold_cell(qᵛ, qᶜˡ, qⁱ, constants; T = 250.0)
    q = MoistureMassFractions(qᵛ, qᶜˡ, qⁱ)
    return with_temperature(LiquidIcePotentialTemperatureState(0.0, q, 1e5, 8e4), T, constants)
end

function cold_saturations(constants, p3; T = 250.0)
    ρ = density(cold_cell(1e-3, 0.0, 0.0, constants; T), constants)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = PPP.p3_ice_saturation_specific_humidity(T, ρ, constants, p3.process_rates.freezing_temperature, qᵛ⁺ˡ)
    return qᵛ⁺ˡ, qᵛ⁺ⁱ
end

function cold_rates(p3, qᵛ, qᶜˡ, qⁱ, nⁱ, constants; T = 250.0, statistics = nothing)
    𝒰 = cold_cell(qᵛ, qᶜˡ, qⁱ, constants; T)
    ℳ = P3MicrophysicalState(qᶜˡ, 1e8, 0.0, 0.0, qⁱ, nⁱ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return compute_p3_process_rates(p3, 1.0, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, statistics, nothing)
end

subgrid_p3(processes) = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(0.0); processes))
all_processes = subgrid_p3(AllProcesses())
only_condensation = subgrid_p3(CloudCondensationOnly())

# A placement with the excess over ice, as `subgrid_process_distribution` forms it
function ice_placement(statistics, s̄, qᶜˡ, s̄ⁱ; ξˡ = 1.05, ξⁱˡ = 1.06, ωˡ = 0.05)
    a, m, σ = PPP.shifted_mixture(statistics, s̄)
    d = placed_cloud_liquid(MixingEvaporationBalance(), a, m, σ, s̄, ξˡ, qᶜˡ, 1e-3, ωˡ, statistics.ωᵐ)
    mⁱ, k = PPP.ice_excess_mixture(statistics, d, s̄ⁱ)
    return merge(d, (; mⁱ, k, ξⁱˡ))
end

# The points s, their weights, and the excess over ice sⁱ(s) of each component at them
function riemann_ice(d; n = 400_000)
    lo = minimum(d.m .- 10 .* d.σ)
    hi = maximum(d.m .+ 10 .* d.σ)
    Δ = (hi - lo) / n
    s = lo .+ Δ .* ((1:n) .- 0.5)
    q = ifelse.(s .> 0, d.χ * d.qᶜˡ .* s ./ d.positive_excess, d.subsaturated_liquid)
    components = map(zip(d.a, d.m, d.σ, d.mⁱ, d.k)) do (a, m, σ, mⁱ, k)
        w = a .* Δ .* exp.(-((s .- m) ./ σ) .^ 2 ./ 2) ./ (sqrt(2π) * σ)
        (w, mⁱ .+ k .* (s .- m))
    end
    return s, q, components
end

@testset "P3 subgrid ice: the excess over ice of the placement" begin
    statistics = ice_mixture(0.2, 4e-4, -1e-4, 1.2e-4, 1e-4, 4.5e-4, -1.1e-4, 1.1e-4, 0.9e-4; ωᵐ = 1e-2)
    s̄, qᶜˡ, s̄ⁱ = 1e-5, 6e-5, 1.4e-4
    d = ice_placement(statistics, s̄, qᶜˡ, s̄ⁱ)
    s, q, components = riemann_ice(d)

    # The parts of the vapor exchange of ice: the excesses of the two parts sum to the grid means, and agree with a
    # Riemann sum of the local excesses δ(s) = s - ξˡ q(s) and δⁱ(s) = sⁱ(s) - ξⁱˡ q(s)
    saturated, subsaturated = PPP.ice_exchange_parts(d)
    @test saturated.Pδ + subsaturated.Pδ ≈ s̄ - d.ξˡ * qᶜˡ
    @test saturated.Pδⁱ + subsaturated.Pδⁱ ≈ s̄ⁱ - d.ξⁱˡ * qᶜˡ
    Pδⁱ⁺ = sum(sum((w .* (sⁱ .- d.ξⁱˡ .* q))[s .> 0]) for (w, sⁱ) in components)
    Pδⁱ⁻ = sum(sum((w .* (sⁱ .- d.ξⁱˡ .* q))[s .<= 0]) for (w, sⁱ) in components)
    # (against the midpoint rule, to its accuracy relative to the excesses of the parts)
    @test saturated.Pδⁱ ≈ Pδⁱ⁺ atol = 1e-4 * s̄ⁱ
    @test subsaturated.Pδⁱ ≈ Pδⁱ⁻ atol = 1e-4 * s̄ⁱ

    # The part where the local excess over ice reaches a threshold
    for c in (-1e-4, 0.0, 5e-5, 1.5e-4)
        P = sum(sum(w[(sⁱ .- d.ξⁱˡ .* q) .>= c]) for (w, sⁱ) in components)
        M = sum(sum((w .* (sⁱ .- d.ξⁱˡ .* q))[(sⁱ .- d.ξⁱˡ .* q) .>= c]) for (w, sⁱ) in components)
        Pᵉ, Mᵉ = PPP.ice_excess_exceedance(d, c)
        @test Pᵉ ≈ P atol = 1e-4
        @test Mᵉ ≈ M atol = 1e-4 * s̄ⁱ
    end
    # The excess the vapor sinks of ice draw on: the positive part of the local excess, at least that of the mean
    for excess in (-2e-4, 0.0, 1e-4)
        available = PPP.available_ice_excess(d, s̄ⁱ - d.ξⁱˡ * qᶜˡ, excess)
        @test available ≥ max(0, excess)
    end

    # Statistics without the excess over ice give it the offsets and spreads of the excess over liquid
    prescribed = (; a = statistics.a, μ = statistics.μ, σ = statistics.σ, f⁺ = statistics.f⁺, ωᵐ = statistics.ωᵐ)
    dᵖ = ice_placement(prescribed, s̄, qᶜˡ, s̄ⁱ)
    @test all(dᵖ.k .== 1)
    @test all(dᵖ.mⁱ .- s̄ⁱ .≈ dᵖ.m .- s̄)
end

@testset "P3 subgrid ice: ice nucleates from the subgrid excess over ice of a cell subsaturated over ice on the mean" begin
    constants = ThermodynamicConstants(Float64)
    T = 250.0
    qᵛ⁺ˡ, qᵛ⁺ⁱ = cold_saturations(constants, all_processes; T)
    # Below ice saturation on the grid mean, so nothing nucleates there; the moist component is well above it
    qᵛ = 0.98 * qᵛ⁺ⁱ
    statistics = ice_mixture(0.15, 3e-4, -0.5e-4, 0.5e-4, 0.4e-4, 3.2e-4, -0.55e-4, 0.45e-4, 0.35e-4)
    narrow = ice_mixture(0.15, 0.0, 0.0, 1e-12, 1e-12, 0.0, 0.0, 1e-12, 1e-12)
    grid_mean = cold_rates(only_condensation, qᵛ, 0.0, 0.0, 0.0, constants; T, statistics)
    @test grid_mean.nucleation_number == 0
    @test cold_rates(all_processes, qᵛ, 0.0, 0.0, 0.0, constants; T, statistics = narrow).nucleation_number == 0
    subgrid = cold_rates(all_processes, qᵛ, 0.0, 0.0, 0.0, constants; T, statistics)
    @test subgrid.nucleation_number > 0
    @test subgrid.nucleation_mass > 0
end

@testset "P3 subgrid ice: the vapor growth of ice over the parts of the cell" begin
    constants = ThermodynamicConstants(Float64)
    T = 250.0
    qᵛ⁺ˡ, qᵛ⁺ⁱ = cold_saturations(constants, all_processes; T)
    qⁱ, nⁱ = 2e-5, 2e4
    statistics = ice_mixture(0.3, 2e-4, -0.9e-4, 0.6e-4, 0.5e-4, 2.2e-4, -1e-4, 0.55e-4, 0.45e-4; ωᵐ = 1e-3)

    # Without cloud liquid every part relaxes alike, and the growth is the grid-mean one
    qᵛ = qᵛ⁺ⁱ + 0.3 * (qᵛ⁺ˡ - qᵛ⁺ⁱ)
    grid_mean = cold_rates(only_condensation, qᵛ, 0.0, qⁱ, nⁱ, constants; T, statistics)
    subgrid = cold_rates(all_processes, qᵛ, 0.0, qⁱ, nⁱ, constants; T, statistics)
    @test grid_mean.deposition > 0
    @test subgrid.deposition ≈ grid_mean.deposition rtol = 1e-10

    # A partly cloudy cell, subsaturated over liquid on the mean: the cloud liquid of the saturated part holds its air
    # at liquid saturation, the ice of the subsaturated part takes that part's excess alone; the growth changes, and
    # stays finite and positive (the cell is supersaturated over ice)
    qᵛ = qᵛ⁺ˡ - 0.5e-4
    grid_mean = cold_rates(only_condensation, qᵛ, 5e-5, qⁱ, nⁱ, constants; T, statistics)
    subgrid = cold_rates(all_processes, qᵛ, 5e-5, qⁱ, nⁱ, constants; T, statistics)
    @test isfinite(subgrid.deposition) && subgrid.deposition > 0
    @test subgrid.deposition != grid_mean.deposition

    # The P3 tendencies conserve water
    ℳ = P3MicrophysicalState(5e-5, 1e8, 0.0, 0.0, qⁱ, nⁱ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    properties = PPP.p3_process_properties(all_processes, 1.0, ℳ)
    result = PPP.p3_tendencies(all_processes, 1.0, ℳ, properties, subgrid)
    water = result.tendency_ρqᵛ + result.tendency_ρqᶜˡ + result.tendency_ρqʳ + result.tendency_ρqⁱ + result.tendency_ρqʷⁱ
    @test abs(water) ≤ 10 * eps() * (abs(result.tendency_ρqᵛ) + abs(result.tendency_ρqⁱ))
end

#####
##### A cold column with the closure's distribution
#####

column_values(f) = Array(interior(f))[1, 1, :]

function cold_column(; ice = true, FT = Float64)
    grid = RectilinearGrid(CPU(), FT; size = 40, z = (0, 2000), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 102000, potential_temperature = 265)
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(FT; thermodynamic_constants = constants, condensation,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(FT; subgrid_distribution = DoubleGaussian(),
                                        static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), thermodynamic_constants = constants,
                            microphysics, closure)
    # The initial sounding of ISDAC (Ovchinnikov et al. 2014) below 1 km
    θˡ(z) = z < 400 ? 265 + 0.004 * (z - 400) : z < 825 ? 265.0 : 266 + (z - 825)^0.3
    qᵗ(z) = 1e-3 * (z < 400 ? 1.5 - 0.00075 * (z - 400) : z < 825 ? 1.5 : 1.2)
    set!(model; θ = θˡ, qᵗ, ρe = z -> z < 825 ? 0.12 : 1e-4)
    if ice
        set!(model.microphysical_fields.ρqⁱ, z -> 400 < z < 825 ? 1.3e-5 : 0.0)
        set!(model.microphysical_fields.ρnⁱ, z -> 400 < z < 825 ? 1.3e3 : 0.0)
    end
    return model
end

total_water(model) = (μ = model.microphysical_fields; ρ = column_values(model.dynamics.reference_state.density);
    Δz = diff(znodes(model.grid, Face()));
    sum((column_values(model.moisture_density) .+ column_values(μ.ρqᶜˡ) .+ column_values(μ.ρqʳ) .+
         column_values(μ.ρqⁱ) .+ column_values(μ.ρqʷⁱ)) .* Δz))

@testset "P3 subgrid ice: a mixed-phase column with the closure's distribution" begin
    model = cold_column()
    W₀ = total_water(model)
    simulation = Simulation(model; Δt = 10, stop_iteration = 30)
    run!(simulation)
    cl = model.closure_fields.subgrid_cloud.cloud
    ηⁱ = column_values(cl.ηⁱ)
    @test maximum(ηⁱ) > 0
    @test all(isfinite, column_values(cl.ΔN²ᴸ))
    @test all(0 .<= ηⁱ .<= 1)
    # The statistics carry the excess over ice, below the excess over liquid in the cold cloud
    stats = Breeze.AtmosphereModels.supersaturation_statistics(TurbulenceClosureStatistics(), 1, 1, 30, model.grid,
                                                              model.closure, model.closure_fields, nothing)
    @test all(stats.μⁱ .> stats.μ)
    # The water of the closed column is conserved
    @test total_water(model) ≈ W₀ rtol = 1e-10
    # The latent conversion the TKE receives at the centers is that of the faces (ice term included)
    grid = model.grid
    ρ = column_values(model.dynamics.reference_state.density)
    zᶜ = znodes(grid, Center()); zᶠ = znodes(grid, Face())
    Nz = size(grid, 3)
    ρᶠ = [(ρ[k] + ρ[k-1]) / 2 for k in 2:Nz]
    Kᶜ = column_values(model.closure_fields.Kᶜ)
    faces = sum(ρᶠ .* diff(zᶜ) .* Kᶜ[2:Nz] .* column_values(cl.ΔN²ᴸ)[2:Nz])
    centers = sum(ρ .* diff(zᶠ) .* column_values(cl.Bᴸ))
    @test centers ≈ faces rtol = 1e-10
end

@testset "P3 subgrid ice: without ice the latent response has no ice term" begin
    # (at the initial state: the cold column nucleates ice once it steps)
    model = cold_column(ice = false)
    @test all(column_values(model.closure_fields.subgrid_cloud.cloud.ηⁱ) .== 0)
end
