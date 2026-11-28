include(joinpath(@__DIR__, "setup.jl"))

#####
##### P3 with the subgrid distribution of the closure (`SubgridCondensation`, `AllProcesses`): the grid-mean limit,
##### conservation and positivity
#####

using Breeze
using Breeze.AtmosphereModels: PrescribedStatistics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, AerosolActivation, AerosolMode,
                                                       SubgridCondensation, CloudCondensationOnly, AllProcesses,
                                                       MixingEvaporationBalance, P3MicrophysicalState, LiquidRelaxationStage,
                                                       compute_p3_process_rates, subgrid_process_distribution
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Grids: znodes
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

@testset "P3 subgrid microphysics: construction" begin
    @test SubgridCondensation(PrescribedStatistics(1e-4)).processes === CloudCondensationOnly()
    @test AllProcesses().liquid_placement === MixingEvaporationBalance()
    @test_throws ArgumentError SubgridCondensation(PrescribedStatistics(1e-4); processes = :all)
    @test subgrid_process_distribution(only_condensation.condensation, gaussian(1e-3), 1e-2, 1e-2, 1e-4) === nothing

    # ℂᵐ is a coefficient of the closure's distribution, and sets the mixing rate it stores
    closure = TKEBasedTurbulenceClosure()
    @test closure_parameters(closure).ℂᵐ == 1
    @test with_parameters(closure; ℂᵐ = 3).subgrid_distribution.ℂᵐ == 3
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

@testset "P3 subgrid microphysics: rates" begin
    constants = ThermodynamicConstants(Float64)
    qᵛ⁺ = cell_saturation(9.5e-3, constants)

    # The vapor exchange of rain is unchanged by the placement
    qᵛ, qᶜˡ, qʳ = qᵛ⁺ - 2e-4, 6e-4, 1e-4
    statistics = mixture(0.1, 1.5e-3, -1.7e-4, 3e-4, 2e-4)
    only = cell_rates(only_condensation, qᵛ, qᶜˡ, constants; qʳ, statistics)
    rates = cell_rates(all_processes, qᵛ, qᶜˡ, constants; qʳ, statistics)
    @test isfinite(rates.autoconversion) && isfinite(rates.accretion) && isfinite(rates.condensation)
    @test rates.autoconversion != only.autoconversion
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

@testset "P3 subgrid microphysics: a column with the closure's distribution and mixing rate" begin
    grid = RectilinearGrid(CPU(); size=16, z=(0, 1600), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation)
    for ℂᵐ in (1, 4)
        closure = with_parameters(TKEBasedTurbulenceClosure(); ℂᵐ)
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
        closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(; variance_stability_functions))
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

column_values(f) = Array(interior(f))[1, 1, :]

function cold_column(; ice = true, FT = Float64)
    grid = RectilinearGrid(CPU(), FT; size = 40, z = (0, 2000), topology = (Flat, Flat, Bounded))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; base_pressure = 102000, potential_temperature = 265)
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(FT; thermodynamic_constants = constants, condensation,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(FT)
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
    @test all(abs.(stats.𝕏ⁱ) .≤ 1 + 1e-12)
    # The second moments of the ice and of the departure of the cloud liquid: realizable and finite
    m = model.closure_fields.subgrid_cloud.moments
    qⁱ′² = column_values(m.qⁱ′²)
    qⁱ′sⁱ′ = column_values(m.qⁱ′sⁱ′)
    @test all(isfinite, qⁱ′sⁱ′) && all(qⁱ′² .≥ 0) && all(column_values(m.qᵈ′²) .≥ 0)
    @test maximum(abs, qⁱ′sⁱ′) > 0
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
