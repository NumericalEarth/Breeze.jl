using Test
import Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant

using Breeze.Microphysics.PredictedParticleProperties:
    IceSizeDistributionState,
    chebyshev_gauss_nodes_weights,
    size_distribution,
    TabulatedFunction3D,
    TabulatedFunction4D,
    TabulatedFunction5D,
    TabulatedFunction6D,
    TabulatedFunction1D,
    P3ProcessRates,
    compute_p3_process_rates,
    consistent_rime_state,
    tendency_ρqᶜˡ,
    tendency_ρqʳ,
    tendency_ρnʳ,
    tendency_ρqⁱ,
    tendency_ρnⁱ,
    tendency_ρqᶠ,
    tendency_ρbᶠ,
    tendency_ρzⁱ,
    tendency_ρqʷⁱ,
    tendency_ρqᵛ,
    rain_autoconversion_rate,
    rain_accretion_rate,
    rain_evaporation_rate,
    rain_self_collection_rate,
    rain_breakup_rate,
    rain_terminal_velocity_mass_weighted,
    ventilation_enhanced_deposition,
    ice_melting_rate,
    ice_melting_rates,
    ice_aggregation_rate,
    cloud_riming_rate,
    cloud_warm_collection_rate,
    rain_riming_rate,
    rime_density,
    P3MicrophysicalState,
    RainMassWeightedVelocityEvaluator,
    RainNumberWeightedVelocityEvaluator,
    RainEvaporationVentilationEvaluator,
    tabulated_function_1d,
    homogeneous_freezing_cloud_rate,
    homogeneous_freezing_rain_rate,
    immersion_freezing_cloud_rate,
    immersion_freezing_rain_rate,
    air_transport_properties,
    psd_correction_spherical_volume,
    liu_daum_shape_parameter

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState

using Oceananigans: CPU, RectilinearGrid
using Oceananigans.Fields: interior

const PPP = Breeze.Microphysics.PredictedParticleProperties

@testset "P3 Integrals" begin

    @testset "Smoke tests - type construction" begin
        # Test main scheme construction
        p3 = PredictedParticlePropertiesMicrophysics()
        @test p3 isa PredictedParticlePropertiesMicrophysics
        @test p3.water_density == 1000.0
        @test p3.minimum_mass_mixing_ratio == 1e-14
        @test p3.minimum_number_mixing_ratio == 1e-16

        # Test alias
        p3_alias = P3Microphysics()
        @test p3_alias isa PredictedParticlePropertiesMicrophysics

        # Test with Float32
        p3_f32 = PredictedParticlePropertiesMicrophysics(Float32)
        @test p3_f32.water_density isa Float32
        @test p3_f32.minimum_mass_mixing_ratio isa Float32
        @test p3_f32.ice.fall_speed.reference_air_density isa Float32
    end

    @testset "AtmosphereModel initialization converts total moisture to vapor" begin
        FT = Float64
        grid = RectilinearGrid(CPU(); size=(1, 1, 1), extent=(1, 1, 1))

        constants = ThermodynamicConstants(FT)
        reference_state = Breeze.ReferenceState(grid, constants;
                                                surface_pressure = FT(101325),
                                                potential_temperature = FT(300))
        dynamics = Breeze.AnelasticDynamics(reference_state)
        model = Breeze.AtmosphereModel(grid; dynamics,
                                       thermodynamic_constants = constants,
                                       microphysics = PredictedParticlePropertiesMicrophysics(FT))

        qᵗ = FT(0.02)
        qᶜˡ = FT(0.005)
        qʳ = FT(0.001)
        qⁱ = FT(0.002)
        qʷⁱ = FT(0.0005)
        expected_qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ

        Breeze.set!(model; θ = FT(300), qᵗ, qᶜˡ, qʳ, qⁱ, qʷⁱ)

        qᵛ_actual = first(Array(interior(model.microphysical_fields.qᵛ)))
        @test qᵛ_actual ≈ expected_qᵛ
    end

    @testset "Ice properties construction" begin
        ice = IceProperties()
        @test ice isa IceProperties
        @test ice.minimum_rime_density == 50.0
        @test ice.maximum_rime_density == 900.0
        @test ice.maximum_shape_parameter == 20.0
        @test ice.minimum_reflectivity == 1e-35

        # Check all sub-containers exist
        @test ice.fall_speed isa IceFallSpeed
        @test ice.deposition isa IceDeposition
        @test ice.bulk_properties isa IceBulkProperties
        @test ice.collection isa IceCollection
        @test ice.sixth_moment isa IceSixthMoment
        @test ice.lambda_limiter isa IceLambdaLimiter
        @test ice.ice_rain isa IceRainCollection
    end

    @testset "Ice fall speed" begin
        fs = IceFallSpeed()
        constants = ThermodynamicConstants(Float64)
        Rᵈ = dry_air_gas_constant(constants)
        @test fs.reference_air_density ≈ 60000 / (Rᵈ * 253.15)

        # Skeleton form: integral fields are placeholders until tables load.
        @test isnothing(fs.number_weighted)
        @test isnothing(fs.mass_weighted)
        @test isnothing(fs.reflectivity_weighted)
    end

    @testset "Ice deposition" begin
        dep = IceDeposition()

        @test isnothing(dep.ventilation)
        @test isnothing(dep.ventilation_enhanced)
        @test isnothing(dep.small_ice_ventilation_constant)
        @test isnothing(dep.small_ice_ventilation_reynolds)
        @test isnothing(dep.large_ice_ventilation_constant)
        @test isnothing(dep.large_ice_ventilation_reynolds)
    end

    @testset "Ice bulk properties" begin
        bp = IceBulkProperties()
        @test bp.maximum_mean_diameter ≈ 2.0e-2
        @test bp.minimum_mean_diameter ≈ 2.0e-6

        @test isnothing(bp.effective_radius)
        @test isnothing(bp.mean_diameter)
        @test isnothing(bp.mean_density)
        @test isnothing(bp.reflectivity)
        @test isnothing(bp.slope)
        @test isnothing(bp.shape)
        @test isnothing(bp.shedding)
    end

    @testset "Ice collection" begin
        col = IceCollection()
        @test col.ice_rain_collection_efficiency ≈ 1.0

        @test isnothing(col.aggregation)
        @test isnothing(col.rain_collection)
    end

    @testset "Ice sixth moment" begin
        m6 = IceSixthMoment()
        @test isnothing(m6.rime)
        @test isnothing(m6.deposition)
        @test isnothing(m6.deposition1)
        @test isnothing(m6.melt1)
        @test isnothing(m6.melt2)
        @test isnothing(m6.aggregation)
        @test isnothing(m6.shedding)
        @test isnothing(m6.sublimation)
        @test isnothing(m6.sublimation1)
    end

    @testset "Ice lambda limiter" begin
        ll = IceLambdaLimiter()
        @test isnothing(ll.small_q)
        @test isnothing(ll.large_q)
    end

    @testset "Ice-rain collection" begin
        ir = IceRainCollection()
        @test isnothing(ir.mass)
        @test isnothing(ir.number)
        @test isnothing(ir.sixth_moment)
    end

    @testset "Rain properties" begin
        rain = RainProperties()
        @test rain.maximum_mean_diameter ≈ 2e-3
        @test rain.fall_speed_coefficient ≈ 841.99667
        @test rain.fall_speed_exponent ≈ 0.8

        @test isnothing(rain.shape_parameter)
        @test isnothing(rain.velocity_number)
        @test isnothing(rain.velocity_mass)
        @test isnothing(rain.evaporation)
    end

    @testset "Cloud droplet properties" begin
        cloud = CloudDropletProperties()
        @test cloud.number_concentration ≈ 200e6
        @test cloud.condensation_timescale ≈ 1.0

        # μ_c is diagnosed from Nc via Liu-Daum (2000) by default.
        # For Nc = 200e6 m⁻³ (200 cm⁻³): χ = 0.0005714*200 + 0.2714 = 0.38568,
        # μ_c = 1/0.38568² - 1 ≈ 5.72 (clamped to [2, 15])
        @test 2 ≤ cloud.shape_parameter ≤ 15
        @test cloud.shape_parameter ≈ liu_daum_shape_parameter(200e6)

        # Explicit shape_parameter overrides Liu-Daum
        cloud_override = CloudDropletProperties(Float64; shape_parameter=5)
        @test cloud_override.shape_parameter ≈ 5.0

        # Test custom parameters
        cloud_custom = CloudDropletProperties(Float64; number_concentration=50e6)
        @test cloud_custom.number_concentration ≈ 50e6
        # Marine Nc → higher μ_c than continental (fewer, larger, more uniform drops)
        @test cloud_custom.shape_parameter > cloud.shape_parameter
    end

    @testset "Water density is shared" begin
        # Water density should be at top level, shared by cloud and rain
        p3 = PredictedParticlePropertiesMicrophysics()
        @test p3.water_density ≈ 1000.0

        # Custom water density
        p3_custom = PredictedParticlePropertiesMicrophysics(Float64; water_density=998.0)
        @test p3_custom.water_density ≈ 998.0
    end

    @testset "Prognostic field names" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        names = prognostic_field_names(p3)

        # P3 cloud number should be prognostic.
        @test :ρqᶜˡ ∈ names
        @test :ρnᶜˡ ∈ names
        @test :ρqʳ ∈ names
        @test :ρnʳ ∈ names
        @test :ρqⁱ ∈ names
        @test :ρnⁱ ∈ names
        @test :ρqᶠ ∈ names
        @test :ρbᶠ ∈ names
        @test :ρz̃ⁱ ∈ names
        @test :ρzⁱ ∉ names
        @test :ρqʷⁱ ∈ names
    end

    @testset "Show methods" begin
        # Just test that show methods don't error
        p3 = PredictedParticlePropertiesMicrophysics()
        io = IOBuffer()
        show(io, p3)
        @test length(take!(io)) > 0

        show(io, p3.ice)
        @test length(take!(io)) > 0

        show(io, p3.ice.fall_speed)
        @test length(take!(io)) > 0

        show(io, p3.rain)
        @test length(take!(io)) > 0

        show(io, p3.cloud)
        @test length(take!(io)) > 0
    end

    @testset "Ice size distribution state" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        @test state.intercept ≈ 1e6
        @test state.shape ≈ 0.0
        @test state.slope ≈ 1000.0
        @test state.rime_fraction ≈ 0.0
        @test state.liquid_fraction ≈ 0.0
        @test state.rime_density ≈ 400.0

        # Test with rime
        state_rimed = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 2.0,
            slope = 500.0,
            rime_fraction = 0.5,
            rime_density = 600.0)

        @test state_rimed.rime_fraction ≈ 0.5
        @test state_rimed.rime_density ≈ 600.0

        # Test size distribution evaluation
        D = 100e-6  # 100 μm
        Np = size_distribution(D, state)
        @test Np > 0

        # N'(D) = N₀ D^μ exp(-λD) for μ=0: N₀ exp(-λD)
        expected = 1e6 * exp(-1000 * D)
        @test Np ≈ expected
    end

    @testset "Chebyshev-Gauss quadrature" begin
        nodes, weights = chebyshev_gauss_nodes_weights(Float64, 32)

        @test length(nodes) == 32
        @test length(weights) == 32

        # Nodes should be in [-1, 1]
        @test all(-1 ≤ x ≤ 1 for x in nodes)

        # Weights should sum to ≈2 (= ∫₋₁¹ dx, with √(1-x²) correction)
        @test sum(weights) ≈ 2 rtol=1e-2

        # Test Float32
        nodes32, weights32 = chebyshev_gauss_nodes_weights(Float32, 16)
        @test eltype(nodes32) == Float32
        @test eltype(weights32) == Float32
    end
end
