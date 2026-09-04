include(joinpath(@__DIR__, "setup.jl"))

using Test
import Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.AtmosphereModels: prognostic_field_names
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant

using Breeze.Microphysics.PredictedParticleProperties:
    chebyshev_gauss_nodes_weights,
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
    tendency_ρqʷⁱ,
    tendency_ρqᵛ,
    rain_autoconversion_rate,
    rain_accretion_rate,
    rain_evaporation_rate,
    rain_self_collection_rate,
    rain_breakup_rate,
    ice_melting_rate,
    ice_melting_rates,
    ice_aggregation_rate,
    cloud_riming_rate,
    cloud_warm_collection_rate,
    rain_riming_rate,
    rime_density,
    P3MicrophysicalState,
    RainMassWeightedVelocity,
    RainNumberWeightedVelocity,
    RainVelocityDiameterIntegral,
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
        @test p3.process_rates.liquid_water_density == 1000.0
        @test p3.minimum_mass_mixing_ratio == 1e-14
        @test p3.minimum_number_mixing_ratio == 1e-16

        # Test alias
        p3_alias = P3Microphysics()
        @test p3_alias isa PredictedParticlePropertiesMicrophysics

        # Test with Float32
        p3_f32 = PredictedParticlePropertiesMicrophysics(Float32)
        @test p3_f32.process_rates.liquid_water_density isa Float32
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
        ice = IceParticles()
        @test ice isa IceParticles
        @test ice.minimum_rime_density == 50.0
        @test ice.maximum_rime_density == 900.0
        @test ice.maximum_shape_parameter == 20.0

        # Check all sub-containers exist
        @test ice.fall_speed isa IceFallSpeed
        @test ice.deposition isa IceDeposition
        @test ice.bulk isa IceBulk
        @test ice.collection isa IceCollection
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
        bp = IceBulk()
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

        @test isnothing(col.aggregation)
        @test isnothing(col.cloud_collection)
        @test isnothing(col.cloud_aerosol_collection)
        @test isnothing(col.ice_aerosol_collection)
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
    end

    @testset "Rain properties" begin
        rain = RainDrops()

        # The active fall-speed law is the piecewise Gunn-Kinzer/Beard fit, not a single
        # power law: `RainDrops` carries its coefficients, and the ventilation pair
        # the runtime rates read.
        @test rain.fall_speed isa PPP.RainFallSpeed{Float64}
        @test rain.fall_speed.branch_velocity_scales == (4579.5, 49.62, 17.32)
        @test rain.fall_speed.branch_mass_exponents == (2/3, 1/3, 1/6)
        @test rain.fall_speed.transition_diameters == (134.43e-6, 1511.64e-6, 3477.84e-6)
        @test rain.fall_speed.plateau_velocity ≈ 9.17
        @test rain.ventilation isa PPP.RainVentilation{Float64}
        @test rain.ventilation.constant_coefficient ≈ 0.78
        @test rain.ventilation.reynolds_coefficient ≈ 0.32

        @test isnothing(rain.velocity_number)
        @test isnothing(rain.velocity_mass)
        @test isnothing(rain.evaporation)
    end

    @testset "Cloud droplet properties" begin
        cloud = CloudDroplets()
        @test cloud.number_concentration ≈ 200e6
        @test cloud.condensation_timescale ≈ 1.0

        # μᶜˡ is diagnosed from Nᶜˡ via Liu-Daum (2000) by default.
        # For Nᶜˡ = 200e6 m⁻³ (200 cm⁻³): χ = 0.0005714*200 + 0.2714 = 0.38568,
        # μᶜˡ = 1/0.38568² - 1 ≈ 5.72 (clamped to [2, 15])
        @test 2 ≤ cloud.shape_parameter ≤ 15
        @test cloud.shape_parameter ≈ liu_daum_shape_parameter(200e6)

        # Explicit shape_parameter overrides Liu-Daum
        cloud_override = CloudDroplets(Float64; shape_parameter=5)
        @test cloud_override.shape_parameter ≈ 5.0

        # The relation itself is a stored, configurable container
        @test cloud.shape isa PPP.CloudShape{Float64}
        @test cloud.shape.relative_dispersion_number_coefficient ≈ 5.714e-10
        @test cloud.shape.relative_dispersion_intercept ≈ 0.2714

        # Test custom parameters
        cloud_custom = CloudDroplets(Float64; number_concentration=50e6)
        @test cloud_custom.number_concentration ≈ 50e6
        # Marine Nᶜˡ → higher μᶜˡ than continental (fewer, larger, more uniform drops)
        @test cloud_custom.shape_parameter > cloud.shape_parameter
    end

    @testset "Thermodynamic constants are shared" begin
        # `process_rates` holds the single water density every rate reads.
        p3 = PredictedParticlePropertiesMicrophysics()
        @test p3.process_rates.liquid_water_density ≈ 1000.0

        default_constants = ThermodynamicConstants()
        custom_liquid = Breeze.CondensedPhase(Float64;
            reference_latent_heat = default_constants.liquid.reference_latent_heat,
            heat_capacity = default_constants.liquid.heat_capacity,
            density = 998)
        custom_ice = Breeze.CondensedPhase(Float64;
            reference_latent_heat = default_constants.ice.reference_latent_heat,
            heat_capacity = default_constants.ice.heat_capacity,
            density = 910)
        custom_constants = ThermodynamicConstants(Float64;
            dry_air_molar_mass = 0.03,
            liquid = custom_liquid,
            ice = custom_ice)
        p3_custom = PredictedParticlePropertiesMicrophysics(Float64;
                                                            thermodynamic_constants = custom_constants)

        @test p3_custom.process_rates.liquid_water_density == 998
        @test p3_custom.process_rates.pure_ice_density == 910
        @test p3_custom.process_rates.initial_rain_drop_mass ≈ 4π / 3 * 998 * (25e-6)^3
        @test p3_custom.process_rates.reference_air_density ≈
              100000 / (dry_air_gas_constant(custom_constants) * 273.15)
        @test p3_custom.ice.fall_speed.reference_air_density ≈
              60000 / (dry_air_gas_constant(custom_constants) * 253.15)
    end

    @testset "Prognostic field names" begin
        # ρnᶜˡ is not advected by default, because the prescribed-Nᶜˡ path takes
        # droplet number from `cloud.number_concentration`.
        p3 = PredictedParticlePropertiesMicrophysics()
        names = prognostic_field_names(p3)

        @test p3.process_rates.predict_supersaturation === false
        @test p3.process_rates isa ProcessRate{Float64, false}
        @test :ρqᶜˡ ∈ names
        @test :ρnᶜˡ ∉ names
        @test :ρnᵃ ∉ names
        @test :ρqʳ ∈ names
        @test :ρnʳ ∈ names
        @test :ρqⁱ ∈ names
        @test :ρnⁱ ∈ names
        @test :ρqᶠ ∈ names
        @test :ρbᶠ ∈ names
        @test :ρqʷⁱ ∈ names
        @test :ρsᵛ⁺ˡ ∉ names

        p3_supersaturation =
            PredictedParticlePropertiesMicrophysics(; predict_supersaturation = true)
        @test p3_supersaturation.process_rates.predict_supersaturation === true
        @test p3_supersaturation.process_rates isa ProcessRate{Float64, true}
        @test :ρsᵛ⁺ˡ ∈ prognostic_field_names(p3_supersaturation)

        # Aerosol activation adds the droplet-number and aerosol prognostics together.
        p3_aerosol = PredictedParticlePropertiesMicrophysics(;
            aerosol = AerosolActivation(AerosolMode(Float64)))
        names_aerosol = prognostic_field_names(p3_aerosol)
        @test :ρnᶜˡ ∈ names_aerosol
        @test :ρnᵃ ∈ names_aerosol
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
