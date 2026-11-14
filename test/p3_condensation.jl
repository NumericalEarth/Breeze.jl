include(joinpath(@__DIR__, "setup.jl"))

#####
##### The `condensation` option of P3, the supersaturation statistics interface, and the phase
##### relaxation rate ωᶜ
#####

using Breeze
using Breeze.AtmosphereModels: AtmosphereModels, NoSubgridCorrelation, PrescribedStatistics,
                               PrescribedPlumeStatistics, supersaturation_statistics, phase_relaxation_rate,
                               instantaneous_phase_relaxation
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, GridMeanCondensation,
                                                       SubgridCondensation, P3MicrophysicalState, CloudDroplets,
                                                       compute_p3_process_rates, p3_state_tendencies,
                                                       p3_phase_relaxation_rates, diagnose_cloud_dsd,
                                                       subgrid_vapor_excess_correlation, air_transport_properties
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density

using GPUArraysCore: @allowscalar
using Oceananigans
using Test

const PPP = PredictedParticleProperties

gaussian(μ, σ) = (; μ, σ, skewness = zero(σ), f⁺ = one(σ) / 2)

# ⟨max(0, x)⟩ - μ for x ~ N(μ, σ²) by the trapezoidal rule
function quadrature_correlation(μ, σ; n = 200_001)
    x = range(μ - 12σ, μ + 12σ; length = n)
    p = @. exp(-((x - μ) / σ)^2 / 2) / (σ * sqrt(2π))
    f = @. max(0, x) * p
    return step(x) * (sum(f) - (first(f) + last(f)) / 2) - μ
end

@testset "Subgrid vapor excess correlation Γ [$FT]" for FT in (Float32, Float64)
    for μ in FT.((-3e-4, -1e-5, 0, 1e-5, 3e-4)), σ in FT.((1e-6, 1e-5, 1e-4, 5e-4))
        Γ = subgrid_vapor_excess_correlation(gaussian(μ, σ), μ)
        @test Γ isa FT
        @test Γ ≥ 0
        FT === Float64 && @test Γ ≈ quadrature_correlation(μ, σ) rtol=1e-6 atol=1e-14
    end

    # σ → 0: Γ → max(0, -μ), so μ + Γ → max(0, μ)
    for μ in FT.((-3e-4, 2e-4))
        @test subgrid_vapor_excess_correlation(gaussian(μ, zero(FT)), μ) ≈ max(0, -μ)
        @test isfinite(subgrid_vapor_excess_correlation(gaussian(μ, zero(FT)), μ))
    end

    # Plumes: one plume is the single Gaussian, coincident plumes are too, and the plume means
    # enter only as offsets about the grid mean
    μ, σ = FT(-1e-4), FT(2e-4)
    single = subgrid_vapor_excess_correlation(gaussian(μ, σ), μ)
    @test subgrid_vapor_excess_correlation((; a = (one(FT),), μ = (FT(1e-3),), σ = (σ,)), μ) ≈ single
    @test subgrid_vapor_excess_correlation((; a = FT.((0.3, 0.7)), μ = FT.((1e-3, 1e-3)), σ = (σ, σ)), μ) ≈ single
    plumes = (; a = FT.((0.3, 0.7)), μ = FT.((6e-4, -2e-4)), σ = FT.((1e-4, 2e-4)))
    Γᵖ = subgrid_vapor_excess_correlation(plumes, μ)
    @test Γᵖ > 0
    shifted = (; plumes.a, μ = plumes.μ .+ FT(1e-3), plumes.σ)
    @test subgrid_vapor_excess_correlation(shifted, μ) ≈ Γᵖ
end

@testset "Supersaturation statistics interface" begin
    s = supersaturation_statistics(PrescribedStatistics(1f-4), 1, 1, 1, nothing, nothing, nothing, nothing)
    @test s === (; μ = 0f0, σ = 1f-4, skewness = 0f0, f⁺ = 0.5f0)
    plumes = PrescribedPlumeStatistics((0.3, 0.7), (6e-4, -2e-4), (1e-4, 2e-4))
    @test supersaturation_statistics(plumes, 1, 1, 1, nothing, nothing, nothing, nothing) ===
          (; a = (0.3, 0.7), μ = (6e-4, -2e-4), σ = (1e-4, 2e-4))
    @test isbits(SubgridCondensation(PrescribedStatistics(1e-4)))
end

@testset "P3 condensation keyword" begin
    @test P3Microphysics(Float64).condensation === GridMeanCondensation()
    subgrid = SubgridCondensation(PrescribedStatistics(1e-4))
    @test P3Microphysics(Float64; condensation = subgrid).condensation === subgrid

    @test P3Microphysics(Float64; condensation = NoSubgridCorrelation()).condensation === GridMeanCondensation()

    # P3 owns its cloud: a closure is never asked to set it
    @test AtmosphereModels.subgrid_correlation(P3Microphysics(Float64; condensation = subgrid)) === NoSubgridCorrelation()
end

# One P3 cell at temperature T and 900 hPa
function cell_state(qᵛ, qᶜˡ, constants; T = 285.0)
    FT = typeof(qᵛ)
    q = MoistureMassFractions(qᵛ, qᶜˡ, zero(FT))
    return with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(9e4)), FT(T), constants)
end

function cell_saturation(qᶜˡ, constants; T = 285.0)
    FT = typeof(qᶜˡ)
    𝒰 = cell_state(FT(1e-2), qᶜˡ, constants; T)
    return saturation_specific_humidity(FT(T), density(𝒰, constants), constants, PlanarLiquidSurface())
end

cell_microphysics(qᶜˡ, nᶜˡ) = P3MicrophysicalState(qᶜˡ, nᶜˡ, ntuple(_ -> zero(qᶜˡ), 10)...)

function cell_rates(p3, qᵛ, qᶜˡ, constants; statistics = nothing)
    FT = typeof(qᵛ)
    ρ = one(FT)
    𝒰 = cell_state(qᵛ, qᶜˡ, constants)
    ℳ = cell_microphysics(qᶜˡ, FT(2e8))
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, temperature(𝒰, constants),
                                    zero(FT), zero(FT), statistics)
end

using Breeze.Thermodynamics: temperature

@testset "P3 SubgridCondensation rates [$FT]" for FT in (Float32, Float64)
    constants = ThermodynamicConstants(FT)
    grid_mean = P3Microphysics(FT)
    p3 = P3Microphysics(FT; condensation = SubgridCondensation(PrescribedStatistics(zero(FT))))

    # σ → 0 recovers the grid-mean rate: condensation in cloud, evaporation of cloud, and
    # condensation onto the seed mass of a cloud-free supersaturated cell
    for (excess, qᶜˡ) in ((2e-4, 2e-4), (-1e-5, 3e-4), (2e-4, 0.0), (-2e-4, 0.0))
        qᵛ = cell_saturation(FT(qᶜˡ), constants) + FT(excess)
        reference = cell_rates(grid_mean, qᵛ, FT(qᶜˡ), constants)
        for σ in (zero(FT), FT(1e-9))
            rates = cell_rates(p3, qᵛ, FT(qᶜˡ), constants; statistics = gaussian(zero(FT), σ))
            @test rates.condensation ≈ reference.condensation rtol=sqrt(eps(FT)) atol=eps(FT)
            @test rates.ccn_activation_mass ≈ reference.ccn_activation_mass rtol=sqrt(eps(FT)) atol=eps(FT)
        end
    end

    # A subgrid spread condenses in a subsaturated cloud-free cell, where the grid mean does not,
    # and condenses more (evaporates less) in cloud; other processes are unchanged
    qᵛ = cell_saturation(FT(2e-4), constants) - FT(1e-4)
    reference = cell_rates(grid_mean, qᵛ, FT(2e-4), constants)
    spread = cell_rates(p3, qᵛ, FT(2e-4), constants; statistics = gaussian(zero(FT), FT(3e-4)))
    @test spread.condensation > reference.condensation
    @test spread.rain_evaporation === reference.rain_evaporation
    @test spread.autoconversion === reference.autoconversion
    dry = cell_saturation(zero(FT), constants) - FT(1e-4)
    @test cell_rates(grid_mean, dry, zero(FT), constants).ccn_activation_mass == 0
    @test cell_rates(p3, dry, zero(FT), constants; statistics = gaussian(zero(FT), FT(3e-4))).ccn_activation_mass > 0

    # Two plumes
    plumes = (; a = FT.((0.3, 0.7)), μ = FT.((3e-4, -1e-4)), σ = FT.((5e-5, 1e-4)))
    plume_rates = cell_rates(p3, qᵛ, FT(2e-4), constants; statistics = plumes)
    @test plume_rates.condensation isa FT
    @test plume_rates.condensation > reference.condensation
end

@testset "P3 SubgridCondensation conserves water and energy" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64; condensation = SubgridCondensation(PrescribedStatistics(3e-4)))
    for (excess, qᶜˡ) in ((2e-4, 2e-4), (-1e-4, 3e-4), (-1e-4, 0.0))
        qᵛ = cell_saturation(qᶜˡ, constants) + excess
        𝒰 = cell_state(qᵛ, qᶜˡ, constants)
        ρ = 1.1
        result = PPP.p3_tendency_compute(p3, ρ, cell_microphysics(qᶜˡ, 2e8), 𝒰, constants,
                                         PPP.p3_process_properties(p3, ρ, cell_microphysics(qᶜˡ, 2e8)),
                                         temperature(𝒰, constants), 0.0, 0.0,
                                         gaussian(0.0, 3e-4))
        water = result.tendency_ρqᵛ + result.tendency_ρqᶜˡ + result.tendency_ρqʳ + result.tendency_ρqⁱ + result.tendency_ρqʷⁱ
        @test abs(result.tendency_ρqᶜˡ) > 0
        @test abs(water) ≤ 1e-12 * abs(result.tendency_ρqᶜˡ)
        # The liquid-ice potential temperature is conserved under phase change: P3 adds no tendency to it
        @test PPP.p3_tendency_component(result, Val(:ρθ)) == 0
    end
end

@testset "Phase relaxation rate ωᶜ [$FT]" for FT in (Float32, Float64)
    constants = ThermodynamicConstants(FT)
    @test phase_relaxation_rate(1, 1, 1, RectilinearGrid(CPU(), FT; size = 1, z = (0, 1), topology = (Flat, Flat, Bounded)),
                                nothing, nothing, nothing, constants) === zero(FT)
    sa = SaturationAdjustment(FT)
    grid = RectilinearGrid(CPU(), FT; size = 1, z = (0, 1), topology = (Flat, Flat, Bounded))
    @test phase_relaxation_rate(1, 1, 1, grid, sa, nothing, nothing, constants) === FT(Inf)
    @test instantaneous_phase_relaxation(sa)
    @test !instantaneous_phase_relaxation(nothing)

    # N = 100 cm⁻³ droplets of about 10 µm radius: ωᶜ = 4π Dᵛ N r̄ / ξˡ, a time scale of seconds
    N = FT(1e8)
    p3 = P3Microphysics(FT; cloud = CloudDroplets(FT; number_concentration = N))
    ρ = one(FT)
    qᶜˡ = N * 4 * FT(π) / 3 * FT(1e-5)^3 * 1000 / ρ
    qᵛ = cell_saturation(qᶜˡ, constants)
    𝒰 = cell_state(qᵛ, qᶜˡ, constants)
    ℳ = cell_microphysics(qᶜˡ, N / ρ)
    ω = p3_phase_relaxation_rates(p3, ρ, ℳ, 𝒰, constants)
    @test ω.ωᶜ isa FT
    @test ω.ωⁱ == 0
    @test ω.ωᶜ == ω.ωˡ
    @test 1 < 1 / ω.ωᶜ < 30

    # The formula
    dsd = diagnose_cloud_dsd(p3, qᶜˡ, N / ρ, ρ)
    r̄ = (dsd.μᶜˡ + 1) / (2 * dsd.λᶜˡ)
    T = temperature(𝒰, constants)
    Dᵛ = air_transport_properties(T, Breeze.Thermodynamics.air_pressure(𝒰, constants), constants).Dᵛ
    ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(T, constants)
    Rᵛ = Breeze.Thermodynamics.vapor_gas_constant(constants)
    ξˡ = 1 + ℒˡ^2 * qᵛ / (Rᵛ * T^2 * constants.dry_air.heat_capacity)
    @test 2 < 1e6 * r̄ < 20
    @test ω.ωˡ ≈ 4 * FT(π) * Dᵛ * dsd.Nᶜˡ * r̄ / ξˡ rtol = 1e-3  # ξˡ here is evaluated at qᵛ ≈ qᵛ⁺ˡ

    # Without condensate, ωᶜ = 0
    @test p3_phase_relaxation_rates(p3, ρ, cell_microphysics(zero(FT), N / ρ), 𝒰, constants).ωᶜ == 0

    # Ice adds a deposition rate below freezing
    cold = with_temperature(LiquidIcePotentialTemperatureState(zero(FT), MoistureMassFractions(FT(2e-3), zero(FT), FT(1e-4)),
                                                               FT(1e5), FT(6e4)), FT(250), constants)
    iced = P3MicrophysicalState(zero(FT), N / ρ, zero(FT), zero(FT), FT(1e-4), FT(1e5), zero(FT), zero(FT),
                                zero(FT), zero(FT), zero(FT), zero(FT))
    ω_ice = p3_phase_relaxation_rates(p3, ρ, iced, cold, constants)
    @test ω_ice.ωⁱ > 0
    @test ω_ice.ωᶜ == ω_ice.ωˡ + ω_ice.ωⁱ
end

@testset "P3 SubgridCondensation in a column model [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = 16, z = (0, 1600), topology = (Flat, Flat, Bounded))

    function column_model(condensation)
        model = AtmosphereModel(grid; microphysics = P3Microphysics(FT; condensation))
        set!(model; θ = 290, qᵗ = z -> ifelse(z < 800, 12e-3, 4e-3), w = 0)
        return model
    end

    grid_mean = column_model(GridMeanCondensation())
    subgrid = column_model(SubgridCondensation(PrescribedStatistics(FT(5e-4))))
    plumes = column_model(SubgridCondensation(PrescribedPlumeStatistics(FT.((0.2, 0.8)), FT.((4e-4, -1e-4)), FT.((1e-4, 2e-4)))))

    # The statistics reach the process-rate diagnostics
    @test MicrophysicalProcessRate(subgrid, :condensation).kernel_function.condensation_fields isa Tuple

    for model in (grid_mean, subgrid, plumes), _ in 1:10
        time_step!(model, 2)
    end
    liquid(model) = @allowscalar Array(interior(model.microphysical_fields.ρqᶜˡ))
    @test all(isfinite, liquid(subgrid))
    @test all(isfinite, liquid(plumes))
    @test sum(liquid(subgrid)) > sum(liquid(grid_mean))

    # The grid-level phase relaxation rate
    k = 1
    𝒰 = @allowscalar Breeze.AtmosphereModels.diagnose_thermodynamic_state(1, 1, k, grid, subgrid.formulation, subgrid.dynamics,
                                                                           MoistureMassFractions(FT(1e-2)))
    ωᶜ = @allowscalar phase_relaxation_rate(1, 1, k, grid, subgrid.microphysics, subgrid.microphysical_fields, 𝒰,
                                             subgrid.thermodynamic_constants)
    @test ωᶜ isa FT
    @test ωᶜ ≥ 0
end
