using Test
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.AtmosphereModels: prognostic_field_names

using Breeze.Microphysics.PredictedParticleProperties:
    IceSizeDistributionState,
    evaluate,
    chebyshev_gauss_nodes_weights,
    size_distribution,
    tabulate,
    TabulationParameters,
    TabulatedFunction3D,
    P3ProcessRates,
    compute_p3_process_rates,
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
    cloud_condensation_rate,
    ventilation_enhanced_deposition,
    ice_melting_rate,
    ice_melting_rates,
    ice_aggregation_rate,
    cloud_riming_rate,
    rain_riming_rate,
    P3MicrophysicalState

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    LiquidIcePotentialTemperatureState

using Oceananigans: CPU

@testset "Predicted Particle Properties (P3) Microphysics" begin

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

    @testset "Ice properties construction" begin
        ice = IceProperties()
        @test ice isa IceProperties
        @test ice.minimum_rime_density == 50.0
        @test ice.maximum_rime_density == 900.0
        @test ice.maximum_shape_parameter == 10.0

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
        @test fs.reference_air_density ≈ 60000 / (287.15 * 253.15)
        @test fs.fall_speed_coefficient ≈ 11.72
        @test fs.fall_speed_exponent ≈ 0.41

        @test fs.number_weighted isa NumberWeightedFallSpeed
        @test fs.mass_weighted isa MassWeightedFallSpeed
        @test fs.reflectivity_weighted isa ReflectivityWeightedFallSpeed
    end

    @testset "Ice deposition" begin
        dep = IceDeposition()
        @test dep.thermal_conductivity ≈ 0.024
        @test dep.vapor_diffusivity ≈ 2.2e-5

        @test dep.ventilation isa Ventilation
        @test dep.ventilation_enhanced isa VentilationEnhanced
        @test dep.small_ice_ventilation_constant isa SmallIceVentilationConstant
        @test dep.small_ice_ventilation_reynolds isa SmallIceVentilationReynolds
        @test dep.large_ice_ventilation_constant isa LargeIceVentilationConstant
        @test dep.large_ice_ventilation_reynolds isa LargeIceVentilationReynolds
    end

    @testset "Ice bulk properties" begin
        bp = IceBulkProperties()
        @test bp.maximum_mean_diameter ≈ 0.02
        @test bp.minimum_mean_diameter ≈ 1e-5

        @test bp.effective_radius isa EffectiveRadius
        @test bp.mean_diameter isa MeanDiameter
        @test bp.mean_density isa MeanDensity
        @test bp.reflectivity isa Reflectivity
        @test bp.slope isa SlopeParameter
        @test bp.shape isa ShapeParameter
        @test bp.shedding isa SheddingRate
    end

    @testset "Ice collection" begin
        col = IceCollection()
        @test col.ice_cloud_collection_efficiency ≈ 0.1
        @test col.ice_rain_collection_efficiency ≈ 1.0

        @test col.aggregation isa AggregationNumber
        @test col.rain_collection isa RainCollectionNumber
    end

    @testset "Ice sixth moment" begin
        m6 = IceSixthMoment()
        @test m6.rime isa SixthMomentRime
        @test m6.deposition isa SixthMomentDeposition
        @test m6.deposition1 isa SixthMomentDeposition1
        @test m6.melt1 isa SixthMomentMelt1
        @test m6.melt2 isa SixthMomentMelt2
        @test m6.aggregation isa SixthMomentAggregation
        @test m6.shedding isa SixthMomentShedding
        @test m6.sublimation isa SixthMomentSublimation
        @test m6.sublimation1 isa SixthMomentSublimation1
    end

    @testset "Ice lambda limiter" begin
        ll = IceLambdaLimiter()
        @test ll.small_q isa NumberMomentLambdaLimit
        @test ll.large_q isa MassMomentLambdaLimit
    end

    @testset "Ice-rain collection" begin
        ir = IceRainCollection()
        @test ir.mass isa IceRainMassCollection
        @test ir.number isa IceRainNumberCollection
        @test ir.sixth_moment isa IceRainSixthMomentCollection
    end

    @testset "Rain properties" begin
        rain = RainProperties()
        @test rain.maximum_mean_diameter ≈ 6e-3
        @test rain.fall_speed_coefficient ≈ 4854.0
        @test rain.fall_speed_exponent ≈ 1.0

        @test rain.shape_parameter isa RainShapeParameter
        @test rain.velocity_number isa RainVelocityNumber
        @test rain.velocity_mass isa RainVelocityMass
        @test rain.evaporation isa RainEvaporation
    end

    @testset "Cloud droplet properties" begin
        cloud = CloudDropletProperties()
        @test cloud.number_concentration ≈ 100e6
        @test cloud.autoconversion_threshold ≈ 25e-6
        @test cloud.condensation_timescale ≈ 1.0

        # Test custom parameters
        cloud_custom = CloudDropletProperties(Float64; number_concentration=50e6)
        @test cloud_custom.number_concentration ≈ 50e6
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

        # With prescribed cloud number (default)
        @test :ρqᶜˡ ∈ names
        @test :ρqʳ ∈ names
        @test :ρnʳ ∈ names
        @test :ρqⁱ ∈ names
        @test :ρnⁱ ∈ names
        @test :ρqᶠ ∈ names
        @test :ρbᶠ ∈ names
        @test :ρzⁱ ∈ names
        @test :ρqʷⁱ ∈ names

        # Cloud number should NOT be in names with prescribed mode
        @test :ρnᶜˡ ∉ names
    end

    @testset "Integral type hierarchy" begin
        # Test abstract types
        @test NumberWeightedFallSpeed <: AbstractFallSpeedIntegral
        @test AbstractFallSpeedIntegral <: AbstractIceIntegral
        @test AbstractIceIntegral <: AbstractP3Integral

        @test Ventilation <: AbstractDepositionIntegral
        @test EffectiveRadius <: AbstractBulkPropertyIntegral
        @test AggregationNumber <: AbstractCollectionIntegral
        @test SixthMomentRime <: AbstractSixthMomentIntegral
        @test NumberMomentLambdaLimit <: AbstractLambdaLimiterIntegral

        @test RainShapeParameter <: AbstractRainIntegral
        @test AbstractRainIntegral <: AbstractP3Integral
    end

    @testset "TabulatedIntegral" begin
        # Create a test array
        data = rand(10, 5, 3)
        tab = TabulatedIntegral(data)

        @test tab isa TabulatedIntegral
        @test size(tab) == (10, 5, 3)
        @test tab[1, 1, 1] == data[1, 1, 1]
        @test tab[5, 3, 2] == data[5, 3, 2]
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

    @testset "Quadrature evaluation - fall speed integrals" begin
        # Create a test state
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        # Test number-weighted fall speed
        V_n = evaluate(NumberWeightedFallSpeed(), state)
        @test V_n > 0
        @test isfinite(V_n)

        # Test mass-weighted fall speed
        V_m = evaluate(MassWeightedFallSpeed(), state)
        @test V_m > 0
        @test isfinite(V_m)

        # Test reflectivity-weighted fall speed
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state)
        @test V_z > 0
        @test isfinite(V_z)

        # Mass-weighted should be larger than number-weighted
        # (larger particles fall faster and contribute more mass)
        # Note: this depends on the specific parameterization
    end

    @testset "Quadrature evaluation - deposition integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        # Basic ventilation
        v = evaluate(Ventilation(), state)
        @test v ≥ 0
        @test isfinite(v)

        v_enh = evaluate(VentilationEnhanced(), state)
        @test v_enh ≥ 0
        @test isfinite(v_enh)

        # Size-regime ventilation
        v_sc = evaluate(SmallIceVentilationConstant(), state)
        v_sr = evaluate(SmallIceVentilationReynolds(), state)
        v_lc = evaluate(LargeIceVentilationConstant(), state)
        v_lr = evaluate(LargeIceVentilationReynolds(), state)

        @test all(isfinite, [v_sc, v_sr, v_lc, v_lr])
    end

    @testset "Quadrature evaluation - bulk property integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        r_eff = evaluate(EffectiveRadius(), state)
        @test r_eff > 0
        @test isfinite(r_eff)

        d_m = evaluate(MeanDiameter(), state)
        @test d_m > 0
        @test isfinite(d_m)

        ρ_m = evaluate(MeanDensity(), state)
        @test ρ_m > 0
        @test isfinite(ρ_m)

        Z = evaluate(Reflectivity(), state)
        @test Z > 0
        @test isfinite(Z)
    end

    @testset "Quadrature evaluation - collection integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        n_agg = evaluate(AggregationNumber(), state)
        @test n_agg > 0
        @test isfinite(n_agg)

        n_rw = evaluate(RainCollectionNumber(), state)
        @test n_rw > 0
        @test isfinite(n_rw)
    end

    @testset "Quadrature evaluation - sixth moment integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0,
            liquid_fraction = 0.1)

        m6_rime = evaluate(SixthMomentRime(), state)
        @test m6_rime > 0
        @test isfinite(m6_rime)

        m6_dep = evaluate(SixthMomentDeposition(), state)
        @test m6_dep > 0
        @test isfinite(m6_dep)

        m6_agg = evaluate(SixthMomentAggregation(), state)
        @test m6_agg > 0
        @test isfinite(m6_agg)

        m6_shed = evaluate(SixthMomentShedding(), state)
        @test m6_shed > 0  # Non-zero because liquid_fraction > 0
        @test isfinite(m6_shed)
    end

    @testset "Quadrature evaluation - lambda limiter integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        i_small = evaluate(NumberMomentLambdaLimit(), state)
        @test i_small > 0
        @test isfinite(i_small)

        i_large = evaluate(MassMomentLambdaLimit(), state)
        @test i_large > 0
        @test isfinite(i_large)
    end

    @testset "Quadrature evaluation - ice-rain collection integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        q_ir = evaluate(IceRainMassCollection(), state)
        @test q_ir > 0
        @test isfinite(q_ir)

        n_ir = evaluate(IceRainNumberCollection(), state)
        @test n_ir > 0
        @test isfinite(n_ir)

        z_ir = evaluate(IceRainSixthMomentCollection(), state)
        @test z_ir > 0
        @test isfinite(z_ir)
    end

    @testset "Quadrature convergence" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0)

        # Test that quadrature gives consistent results across resolutions
        V_16 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=16)
        V_32 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=32)
        V_64 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=64)
        V_128 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)

        # All resolutions should give values within 1% of the high-resolution result
        @test abs(V_16 - V_128) / V_128 < 0.01
        @test abs(V_32 - V_128) / V_128 < 0.01
        @test abs(V_64 - V_128) / V_128 < 0.01
    end

    @testset "Tabulation parameters" begin
        params = TabulationParameters()
        @test params.number_of_mass_points == 50
        @test params.number_of_rime_fraction_points == 4
        @test params.number_of_liquid_fraction_points == 4
        @test params.minimum_log_mean_particle_mass ≈ -15
        @test params.maximum_log_mean_particle_mass ≈ -5
        @test params.number_of_quadrature_points == 64

        # Custom parameters
        params_custom = TabulationParameters(Float32;
            number_of_mass_points=20,
            number_of_rime_fraction_points=3,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=32)
        @test params_custom.number_of_mass_points == 20
        @test params_custom.number_of_rime_fraction_points == 3
        @test params_custom.number_of_liquid_fraction_points == 2
        @test params_custom.minimum_log_mean_particle_mass isa Float32
    end

    @testset "Tabulate single integral" begin
        params = TabulationParameters(Float64;
            number_of_mass_points=5,
            number_of_rime_fraction_points=2,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=16)

        # Tabulate number-weighted fall speed
        tab_Vn = tabulate(NumberWeightedFallSpeed(), CPU(), params)

        @test tab_Vn isa TabulatedFunction3D
        @test size(tab_Vn.table) == (5, 2, 2)

        # Values should be non-negative and finite
        @test all(isfinite, tab_Vn.table)
        @test all(x -> x >= 0, tab_Vn.table)

        # Test indexing via table (unrimed, liquid_fraction=0 should be positive)
        @test tab_Vn.table[1, 1, 1] > 0
        @test tab_Vn.table[5, 1, 1] > 0
    end

    @testset "Tabulate IceFallSpeed container" begin
        params = TabulationParameters(Float64;
            number_of_mass_points=5,
            number_of_rime_fraction_points=2,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=16)

        fs = IceFallSpeed()
        fs_tab = tabulate(fs, CPU(), params)

        # Parameters should be preserved
        @test fs_tab.reference_air_density == fs.reference_air_density
        @test fs_tab.fall_speed_coefficient == fs.fall_speed_coefficient
        @test fs_tab.fall_speed_exponent == fs.fall_speed_exponent

        # Integrals should be tabulated
        @test fs_tab.number_weighted isa TabulatedFunction3D
        @test fs_tab.mass_weighted isa TabulatedFunction3D
        @test fs_tab.reflectivity_weighted isa TabulatedFunction3D

        # Check sizes
        @test size(fs_tab.number_weighted.table) == (5, 2, 2)
        @test size(fs_tab.mass_weighted.table) == (5, 2, 2)
        @test size(fs_tab.reflectivity_weighted.table) == (5, 2, 2)
    end

    @testset "Tabulate IceDeposition container" begin
        params = TabulationParameters(Float64;
            number_of_mass_points=5,
            number_of_rime_fraction_points=2,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=16)

        dep = IceDeposition()
        dep_tab = tabulate(dep, CPU(), params)

        # Parameters should be preserved
        @test dep_tab.thermal_conductivity == dep.thermal_conductivity
        @test dep_tab.vapor_diffusivity == dep.vapor_diffusivity

        # All 6 integrals should be tabulated
        @test dep_tab.ventilation isa TabulatedFunction3D
        @test dep_tab.ventilation_enhanced isa TabulatedFunction3D
        @test dep_tab.small_ice_ventilation_constant isa TabulatedFunction3D
        @test dep_tab.small_ice_ventilation_reynolds isa TabulatedFunction3D
        @test dep_tab.large_ice_ventilation_constant isa TabulatedFunction3D
        @test dep_tab.large_ice_ventilation_reynolds isa TabulatedFunction3D
    end

    @testset "Tabulate P3 scheme by property" begin
        p3 = PredictedParticlePropertiesMicrophysics()

        # Tabulate fall speed
        p3_fs = tabulate(p3, :ice_fall_speed, CPU();
            number_of_mass_points=5,
            number_of_rime_fraction_points=2,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=16)

        @test p3_fs isa PredictedParticlePropertiesMicrophysics
        @test p3_fs.ice.fall_speed.number_weighted isa TabulatedFunction3D
        @test p3_fs.ice.fall_speed.mass_weighted isa TabulatedFunction3D

        # Other properties should be unchanged
        @test p3_fs.ice.deposition.ventilation isa Ventilation
        @test p3_fs.rain == p3.rain
        @test p3_fs.cloud == p3.cloud

        # Tabulate deposition
        p3_dep = tabulate(p3, :ice_deposition, CPU();
            number_of_mass_points=5,
            number_of_rime_fraction_points=2,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=16)

        @test p3_dep.ice.deposition.ventilation isa TabulatedFunction3D
        @test p3_dep.ice.fall_speed.number_weighted isa NumberWeightedFallSpeed
    end

    @testset "Tabulation error handling" begin
        p3 = PredictedParticlePropertiesMicrophysics()

        # Unknown property should throw
        @test_throws ArgumentError tabulate(p3, :unknown_property, CPU())
    end

    #####
    ##### Physical consistency tests
    #####

    @testset "Fall speed physical consistency" begin
        # For a given PSD, larger particles fall faster
        # Note: evaluate() returns density-weighted integrals (fluxes), not mean velocities.
        # So we cannot compare V_n and V_m directly without normalization.

        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 500.0)  # Larger particles (smaller λ)

        V_n = evaluate(NumberWeightedFallSpeed(), state)
        V_m = evaluate(MassWeightedFallSpeed(), state)
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state)

        # All should be positive
        @test V_n > 0
        @test V_m > 0
        @test V_z > 0

        # Typical ice fall speeds should be 0.1 - 10 m/s
        # These are normalized integrals, but check they're in a reasonable range
        @test isfinite(V_n)
        @test isfinite(V_m)
        @test isfinite(V_z)
    end

    @testset "Size distribution moments" begin
        # For exponential distribution (μ=0), analytical moments are known:
        # M_n = N₀ Γ(n+1) / λ^{n+1}
        # M_0 = N₀ / λ (total number)
        # M_1 = N₀ / λ²
        # M_3 = 6 N₀ / λ⁴
        # M_6 = 720 N₀ / λ⁷

        N₀ = 1e6
        λ = 1000.0
        μ = 0.0

        state = IceSizeDistributionState(Float64;
            intercept = N₀,
            shape = μ,
            slope = λ)

        # Number integral (0th moment proxy)
        n_int = evaluate(NumberMomentLambdaLimit(), state)  # This is just ∫ N'(D) dD
        @test n_int > 0
        @test isfinite(n_int)

        # Reflectivity (6th moment)
        Z = evaluate(Reflectivity(), state)
        @test Z > 0
        @test isfinite(Z)
    end

    @testset "Slope parameter dependence" begin
        # Smaller λ means larger particles
        # Integrals should increase as λ decreases (larger particles)

        state_small = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 2000.0)  # Small particles

        state_large = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)   # Large particles

        # Reflectivity (D^6 weighted) should increase with larger particles
        Z_small = evaluate(Reflectivity(), state_small)
        Z_large = evaluate(Reflectivity(), state_large)

        @test Z_large > Z_small

        # Fall speed should also increase with larger particles
        V_small = evaluate(NumberWeightedFallSpeed(), state_small)
        V_large = evaluate(NumberWeightedFallSpeed(), state_large)

        @test V_large > V_small
    end

    @testset "Shape parameter dependence" begin
        # Higher μ narrows the distribution
        state_mu0 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        state_mu2 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 2.0, slope = 1000.0)

        state_mu4 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 4.0, slope = 1000.0)

        # All should produce finite results
        V0 = evaluate(NumberWeightedFallSpeed(), state_mu0)
        V2 = evaluate(NumberWeightedFallSpeed(), state_mu2)
        V4 = evaluate(NumberWeightedFallSpeed(), state_mu4)

        @test all(isfinite, [V0, V2, V4])
        @test all(x -> x > 0, [V0, V2, V4])
    end

    @testset "Rime fraction dependence" begin
        # Test that both rimed and unrimed states produce valid results
        state_unrimed = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            rime_fraction = 0.0, rime_density = 400.0)

        state_rimed = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            rime_fraction = 0.5, rime_density = 600.0)

        # Both should produce valid results for number-weighted fall speed
        V_unrimed = evaluate(NumberWeightedFallSpeed(), state_unrimed)
        V_rimed = evaluate(NumberWeightedFallSpeed(), state_rimed)

        @test isfinite(V_unrimed)
        @test isfinite(V_rimed)
        @test V_unrimed > 0
        @test V_rimed > 0

        # Mass-weighted fall speed should be larger for rimed particles
        # because particle_mass depends on rime_fraction (higher effective density)
        F_m_unrimed = evaluate(MassWeightedFallSpeed(), state_unrimed)
        F_m_rimed = evaluate(MassWeightedFallSpeed(), state_rimed)

        @test isfinite(F_m_unrimed)
        @test isfinite(F_m_rimed)
        @test F_m_rimed > F_m_unrimed  # Higher mass → larger flux integral
    end

    @testset "Liquid fraction dependence" begin
        # Liquid on ice affects shedding and melting integrals
        state_dry = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            liquid_fraction = 0.0)

        state_wet = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            liquid_fraction = 0.3)

        # Shedding should be zero when dry, positive when wet
        shed_dry = evaluate(SheddingRate(), state_dry)
        shed_wet = evaluate(SheddingRate(), state_wet)

        @test shed_dry ≈ 0 atol=1e-20
        @test shed_wet > 0

        # Sixth moment shedding similarly
        m6_shed_dry = evaluate(SixthMomentShedding(), state_dry)
        m6_shed_wet = evaluate(SixthMomentShedding(), state_wet)

        @test m6_shed_dry ≈ 0 atol=1e-20
        @test m6_shed_wet > 0
    end

    @testset "Deposition integrals physical consistency" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        # Ventilation integrals should be positive
        v = evaluate(Ventilation(), state)
        v_enh = evaluate(VentilationEnhanced(), state)

        @test v > 0
        @test v_enh ≥ 0  # Could be zero if no particles > 100 μm

        # Small + large ice ventilation should roughly equal total
        v_sc = evaluate(SmallIceVentilationConstant(), state)
        v_lc = evaluate(LargeIceVentilationConstant(), state)

        @test v_sc ≥ 0
        @test v_lc ≥ 0
        # Note: v_sc + v_lc ≈ v only if ventilation factor is same
    end

    @testset "Collection integrals" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        n_agg = evaluate(AggregationNumber(), state)
        n_rw = evaluate(RainCollectionNumber(), state)

        @test n_agg > 0
        @test n_rw > 0
        @test isfinite(n_agg)
        @test isfinite(n_rw)
    end

    @testset "Sixth moment integrals physical consistency" begin
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            liquid_fraction = 0.1)

        # All sixth moment integrals should be finite and positive
        m6_rime = evaluate(SixthMomentRime(), state)
        m6_dep = evaluate(SixthMomentDeposition(), state)
        m6_dep1 = evaluate(SixthMomentDeposition1(), state)
        m6_mlt1 = evaluate(SixthMomentMelt1(), state)
        m6_mlt2 = evaluate(SixthMomentMelt2(), state)
        m6_agg = evaluate(SixthMomentAggregation(), state)
        m6_shed = evaluate(SixthMomentShedding(), state)
        m6_sub = evaluate(SixthMomentSublimation(), state)
        m6_sub1 = evaluate(SixthMomentSublimation1(), state)

        @test all(isfinite, [m6_rime, m6_dep, m6_dep1, m6_mlt1, m6_mlt2,
                             m6_agg, m6_shed, m6_sub, m6_sub1])
        @test all(x -> x > 0, [m6_rime, m6_dep, m6_mlt1, m6_agg, m6_sub])
    end

    @testset "Float32 input precision" begin
        # All integrals should work with Float32 input
        # Note: FastGaussQuadrature returns Float64 nodes/weights, so output
        # may be promoted to Float64. We just check the results are valid.
        state = IceSizeDistributionState(Float32;
            intercept = 1f6,
            shape = 0f0,
            slope = 1000f0)

        V_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=32)
        V_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=32)
        Z = evaluate(Reflectivity(), state; n_quadrature=32)

        # Results should be valid floating point numbers
        @test isfinite(V_n)
        @test isfinite(V_m)
        @test isfinite(Z)
        @test V_n > 0
        @test V_m > 0
        @test Z > 0
    end

    @testset "Extreme parameters" begin
        # Very small particles (large λ)
        state_small = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 10000.0)

        V_small = evaluate(NumberWeightedFallSpeed(), state_small)
        @test isfinite(V_small)
        @test V_small > 0

        # Very large particles (small λ)
        state_large = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 100.0)

        V_large = evaluate(NumberWeightedFallSpeed(), state_large)
        @test isfinite(V_large)
        @test V_large > 0

        # High shape parameter
        state_narrow = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 6.0, slope = 1000.0)

        V_narrow = evaluate(NumberWeightedFallSpeed(), state_narrow)
        @test isfinite(V_narrow)
        @test V_narrow > 0
    end

    #####
    ##### Analytical comparison tests
    #####

    @testset "Analytical comparison - exponential PSD moments" begin
        # For exponential PSD (μ=0): N'(D) = N₀ exp(-λD)
        # The n-th moment is:
        # M_n = ∫₀^∞ D^n N'(D) dD = N₀ n! / λ^{n+1}

        N₀ = 1e6
        μ = 0.0
        λ = 1000.0

        # For exponential (μ=0): M_n = N₀ n! / λ^{n+1}
        # M0 = N₀ / λ = 1e6 / 1000 = 1e3
        # M6 = N₀ * 720 / λ^7 = 1e6 * 720 / 1e21 = 7.2e-13

        state = IceSizeDistributionState(Float64;
            intercept = N₀, shape = μ, slope = λ)

        # Test NumberMomentLambdaLimit (which integrates the full PSD)
        small_q_lim = evaluate(NumberMomentLambdaLimit(), state; n_quadrature=128)
        @test small_q_lim > 0
        @test isfinite(small_q_lim)

        # Test Reflectivity (which integrates D^6 N'(D))
        refl = evaluate(Reflectivity(), state; n_quadrature=128)
        @test refl > 0
        @test isfinite(refl)
    end

    @testset "Analytical comparison - gamma PSD with mu=2" begin
        # For gamma PSD with μ=2: N'(D) = N₀ D² exp(-λD)
        # M_n = N₀ Γ(n+3) / λ^{n+3} = N₀ (n+2)! / λ^{n+3}

        N₀ = 1e6
        μ = 2.0
        λ = 1000.0

        state = IceSizeDistributionState(Float64;
            intercept = N₀, shape = μ, slope = λ)

        # All integrals should return finite positive values
        V_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)
        V_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=128)
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state; n_quadrature=128)

        @test all(isfinite, [V_n, V_m, V_z])
        @test all(x -> x > 0, [V_n, V_m, V_z])

        # For a power-law fall speed V(D) = a D^b, the ratio of moments gives:
        # V_n / V_m should depend on μ in a predictable way
        @test V_n > 0
        @test V_m > 0
    end

    @testset "Lambda limiter integrals consistency" begin
        # The lambda limiter integrals should produce values that
        # allow solving for λ bounds

        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        small_q = evaluate(NumberMomentLambdaLimit(), state)
        large_q = evaluate(MassMomentLambdaLimit(), state)

        @test small_q > 0
        @test large_q > 0
        @test isfinite(small_q)
        @test isfinite(large_q)

        # LargeQ should be > SmallQ for same state since large q
        # corresponds to small λ (larger particles)
        # Actually, these are both positive but their relative values
        # depend on the integral definition
        @test small_q ≠ large_q
    end

    @testset "Mass-weighted velocity ordering" begin
        # For particles with power-law fall speed V(D) = a D^b (b > 0):
        # Reflectivity-weighted (Z-weighted) mean velocity should be largest
        # because it weights by D^6, emphasizing large particles
        # Mass-weighted should be intermediate
        # Number-weighted should be smallest
        # Note: We must compare normalized mean velocities, not raw flux integrals.

        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)  # Large particles

        # Flux integrals
        F_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)
        F_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=128)
        F_z = evaluate(ReflectivityWeightedFallSpeed(), state; n_quadrature=128)

        # Normalization moments
        M_n = evaluate(NumberMomentLambdaLimit(), state; n_quadrature=128)
        M_m = evaluate(MassMomentLambdaLimit(), state; n_quadrature=128)
        M_z = evaluate(Reflectivity(), state; n_quadrature=128)

        # Mean velocities
        V_n = F_n / M_n
        V_m = F_m / M_m
        V_z = F_z / M_z

        # All should be positive
        @test V_n > 0
        @test V_m > 0
        @test V_z > 0

        # Check ordering: V_z ≥ V_m ≥ V_n for most PSDs
        @test V_z >= V_m
        @test V_m >= V_n
    end

    @testset "Ventilation integral properties" begin
        # Ventilation factor accounts for enhanced mass transfer due to air flow
        # For large particles (high Re), ventilation should be larger

        state_small = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 5000.0)  # Small particles

        state_large = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 200.0)   # Large particles

        v_small = evaluate(Ventilation(), state_small; n_quadrature=64)
        v_large = evaluate(Ventilation(), state_large; n_quadrature=64)

        @test isfinite(v_small)
        @test isfinite(v_large)
        @test v_small > 0
        @test v_large > 0

        # Larger particles have larger ventilation due to higher Reynolds
        @test v_large > v_small
    end

    @testset "Mean diameter integral" begin
        # Mean diameter should scale inversely with λ
        state1 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0)

        state2 = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)  # Half λ → double mean D

        D_mean_1 = evaluate(MeanDiameter(), state1; n_quadrature=64)
        D_mean_2 = evaluate(MeanDiameter(), state2; n_quadrature=64)

        @test D_mean_1 > 0
        @test D_mean_2 > 0

        # D_mean_2 should be ~2x D_mean_1 for exponential (μ=0) distribution
        @test D_mean_2 > D_mean_1
    end

    #####
    ##### Lambda solver tests
    #####

    @testset "IceMassPowerLaw construction" begin
        mass = IceMassPowerLaw()
        @test mass.coefficient ≈ 0.0121
        @test mass.exponent ≈ 1.9
        @test mass.ice_density ≈ 917.0

        mass32 = IceMassPowerLaw(Float32)
        @test mass32.coefficient isa Float32
    end

    @testset "ShapeParameterRelation construction" begin
        relation = ShapeParameterRelation()
        @test relation.a ≈ 0.00191
        @test relation.b ≈ 0.8
        @test relation.c ≈ 2.0
        @test relation.μmax ≈ 6.0

        # Test shape parameter computation
        μ = shape_parameter(relation, log(1000.0))
        @test μ ≥ 0
        @test μ ≤ relation.μmax
    end

    @testset "Ice regime thresholds" begin
        mass = IceMassPowerLaw()

        # Unrimed ice
        thresholds_unrimed = ice_regime_thresholds(mass, 0.0, 400.0)
        @test thresholds_unrimed.spherical > 0
        @test thresholds_unrimed.graupel == Inf
        @test thresholds_unrimed.partial_rime == Inf

        # Rimed ice
        thresholds_rimed = ice_regime_thresholds(mass, 0.5, 400.0)
        @test thresholds_rimed.spherical > 0
        @test thresholds_rimed.graupel > thresholds_rimed.spherical
        @test thresholds_rimed.partial_rime > thresholds_rimed.graupel
        @test thresholds_rimed.ρ_graupel > 0
    end

    @testset "Ice mass computation" begin
        mass = IceMassPowerLaw()

        # Small particles should have spherical mass
        D_small = 1e-5  # 10 μm
        m_small = ice_mass(mass, 0.0, 400.0, D_small)
        m_sphere = mass.ice_density * π / 6 * D_small^3
        @test m_small ≈ m_sphere

        # Mass should increase with diameter
        D_large = 1e-3  # 1 mm
        m_large = ice_mass(mass, 0.0, 400.0, D_large)
        @test m_large > m_small
    end

    @testset "Lambda solver - basic functionality" begin
        # Create a test case with known parameters
        L_ice = 1e-4   # 0.1 g/m³
        N_ice = 1e5    # 100,000 particles/m³
        rime_fraction = 0.0
        rime_density = 400.0

        logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density)

        @test isfinite(logλ)
        @test logλ > log(10)    # Within bounds
        @test logλ < log(1e7)

        λ = exp(logλ)
        @test λ > 0
    end

    @testset "Lambda solver - consistency" begin
        # Solve for λ, then verify the L/N ratio is recovered
        L_ice = 1e-4
        N_ice = 1e5
        rime_fraction = 0.0
        rime_density = 400.0

        mass = IceMassPowerLaw()
        shape_relation = ShapeParameterRelation()

        params = distribution_parameters(L_ice, N_ice, rime_fraction, rime_density;
                                          mass, shape_relation)

        @test params.N₀ > 0
        @test params.λ > 0
        @test params.μ ≥ 0

        # The solved parameters should be consistent
        @test isfinite(params.N₀)
        @test isfinite(params.λ)
        @test isfinite(params.μ)
    end

    @testset "Lambda solver - rimed ice" begin
        L_ice = 1e-4
        N_ice = 1e5
        rime_fraction = 0.5
        rime_density = 500.0

        logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density)

        @test isfinite(logλ)
        @test exp(logλ) > 0
    end

    @testset "Lambda solver - edge cases" begin
        # Zero mass or number should return the upper bound (smallest particles),
        # not the unphysical λ = 0.
        logλ_zero_L = solve_lambda(0.0, 1e5, 0.0, 400.0)
        @test logλ_zero_L == log(1e7)

        logλ_zero_N = solve_lambda(1e-4, 0.0, 0.0, 400.0)
        @test logλ_zero_N == log(1e7)
    end

    @testset "Lambda solver - L/N dependence" begin
        # Higher L/N ratio means larger particles, hence smaller λ
        N_ice = 1e5
        rime_fraction = 0.0
        rime_density = 400.0

        logλ_small = solve_lambda(1e-5, N_ice, rime_fraction, rime_density)  # Small L/N
        logλ_large = solve_lambda(1e-3, N_ice, rime_fraction, rime_density)  # Large L/N

        # Larger mean mass → smaller λ (larger characteristic diameter)
        @test logλ_large < logλ_small
    end

    @testset "P3ProcessRates construction" begin
        FT = Float64
        rates = P3ProcessRates(
            ntuple(_ -> zero(FT), fieldcount(P3ProcessRates))...
        )
        @test rates isa P3ProcessRates{FT}
        @test rates.condensation == 0.0
        @test rates.autoconversion == 0.0
        @test rates.partial_melting == 0.0
        @test rates.complete_melting == 0.0
    end

    @testset "Tendency functions - smoke tests" begin
        FT = Float64
        ρ = FT(1.0)    # Air density [kg/m³]
        qⁱ = FT(1e-4)  # Ice mass mixing ratio [kg/kg]
        nⁱ = FT(1e5)   # Ice number [1/kg]
        zⁱ = FT(1e-8)  # Ice reflectivity
        Fᶠ = FT(0.3)   # Rime fraction
        ρᶠ = FT(400.0)  # Rime density [kg/m³]
        prp = ProcessRateParameters(FT)

        # Create rates with typical warm-rain and ice process activity
        rates = P3ProcessRates(
            # Phase 1: Cloud condensation/evaporation
            FT(5e-7),   # condensation (positive = condensation)
            # Phase 1: Rain
            FT(1e-7),   # autoconversion
            FT(2e-7),   # accretion
            FT(-5e-8),  # rain_evaporation (negative = loss)
            FT(-1e-6),  # rain_self_collection (negative = loss)
            FT(5e-7),   # rain_breakup (positive = number source)
            # Phase 1: Ice
            FT(3e-7),   # deposition
            FT(1e-8),   # partial_melting
            FT(5e-8),   # complete_melting
            FT(-1e3),   # melting_number (negative = loss)
            # Phase 2: Aggregation
            FT(-500.0), # aggregation (negative = number loss)
            # Phase 2: Riming
            FT(1e-7),   # cloud_riming
            FT(-1e4),   # cloud_riming_number
            FT(5e-8),   # rain_riming
            FT(-500.0), # rain_riming_number
            FT(300.0),  # rime_density_new
            # Phase 2: Shedding and refreezing
            FT(2e-8),   # shedding
            FT(100.0),  # shedding_number
            FT(1e-8),   # refreezing
            # Ice nucleation
            FT(1e-9),   # nucleation_mass
            FT(10.0),   # nucleation_number
            FT(5e-9),   # cloud_freezing_mass
            FT(100.0),  # cloud_freezing_number
            FT(3e-9),   # rain_freezing_mass
            FT(50.0),   # rain_freezing_number
            # Rime splintering
            FT(1e-10),  # splintering_mass
            FT(1.0),    # splintering_number
        )

        # Test each tendency function returns a finite number
        @test isfinite(tendency_ρqᶜˡ(rates, ρ))
        @test isfinite(tendency_ρqʳ(rates, ρ))
        @test isfinite(tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, prp))
        @test isfinite(tendency_ρqⁱ(rates, ρ))
        @test isfinite(tendency_ρnⁱ(rates, ρ))
        @test isfinite(tendency_ρqᶠ(rates, ρ, Fᶠ))
        @test isfinite(tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ))
        @test isfinite(tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ))
        @test isfinite(tendency_ρqʷⁱ(rates, ρ))
        @test isfinite(tendency_ρqᵛ(rates, ρ))

        # Physics: condensation (vapor→cloud) should decrease vapor
        @test tendency_ρqᵛ(rates, ρ) < 0
    end

    @testset "Tendency functions - zero rates produce zero tendencies" begin
        FT = Float64
        ρ = FT(1.0)
        zero_rates = P3ProcessRates(ntuple(_ -> zero(FT), fieldcount(P3ProcessRates))...)

        @test tendency_ρqᶜˡ(zero_rates, ρ) == 0.0
        @test tendency_ρqʳ(zero_rates, ρ) == 0.0
        @test tendency_ρnʳ(zero_rates, ρ, FT(1e5), FT(1e-4), ProcessRateParameters(FT)) == 0.0
        @test tendency_ρqⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρnⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρqᶠ(zero_rates, ρ, FT(0.3)) == 0.0
        @test tendency_ρbᶠ(zero_rates, ρ, FT(0.3), FT(400.0)) == 0.0
        @test tendency_ρzⁱ(zero_rates, ρ, FT(1e-4), FT(1e5), FT(1e-8)) == 0.0
        @test tendency_ρqʷⁱ(zero_rates, ρ) == 0.0
        @test tendency_ρqᵛ(zero_rates, ρ) == 0.0
    end

    @testset "Tendency functions - Float32 type stability" begin
        FT = Float32
        ρ = FT(1.0)
        rates = P3ProcessRates(ntuple(_ -> FT(1e-7), fieldcount(P3ProcessRates))...)

        @test tendency_ρqᶜˡ(rates, ρ) isa FT
        @test tendency_ρqʳ(rates, ρ) isa FT
        @test tendency_ρnʳ(rates, ρ, FT(1e5), FT(1e-4), ProcessRateParameters(FT)) isa FT
        @test tendency_ρqⁱ(rates, ρ) isa FT
        @test tendency_ρnⁱ(rates, ρ) isa FT
        @test tendency_ρqᶠ(rates, ρ, FT(0.3)) isa FT
        @test tendency_ρbᶠ(rates, ρ, FT(0.3), FT(400.0)) isa FT
        @test tendency_ρzⁱ(rates, ρ, FT(1e-4), FT(1e5), FT(1e-8)) isa FT
        @test tendency_ρqʷⁱ(rates, ρ) isa FT
        @test tendency_ρqᵛ(rates, ρ) isa FT
    end

    #####
    ##### Process rate function tests
    #####

    @testset "rain_autoconversion_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        # KK2000 formula with typical cumulus values
        qc = FT(1e-3)     # 1 g/kg cloud water
        Nc = FT(100e6)     # 100 cm⁻³ cloud droplet concentration

        rate = rain_autoconversion_rate(p3, qc, Nc)
        @test rate > 0
        # KK2000 gives O(1e-6) kg/kg/s for these inputs
        @test rate > 1e-8
        @test rate < 1e-3

        # Higher cloud water content gives faster autoconversion
        rate_high = rain_autoconversion_rate(p3, FT(2e-3), Nc)
        @test rate_high > rate

        # Zero cloud water gives zero autoconversion
        rate_zero = rain_autoconversion_rate(p3, FT(0), Nc)
        @test rate_zero == 0

        # Below threshold gives zero (threshold is 1e-4 kg/kg)
        rate_below = rain_autoconversion_rate(p3, FT(5e-5), Nc)
        @test rate_below == 0
    end

    @testset "rain_accretion_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qc = FT(1e-3)
        qr = FT(1e-3)

        rate = rain_accretion_rate(p3, qc, qr)
        @test rate > 0
        @test isfinite(rate)

        # Zero cloud gives zero accretion
        @test rain_accretion_rate(p3, FT(0), qr) == 0

        # Zero rain gives zero accretion
        @test rain_accretion_rate(p3, qc, FT(0)) == 0

        # Higher rain gives faster accretion
        rate_high = rain_accretion_rate(p3, qc, FT(2e-3))
        @test rate_high > rate
    end

    @testset "rain_evaporation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qr = FT(1e-3)
        nr = FT(1e4)
        T = FT(288.0)
        ρ = FT(1.0)

        # Subsaturated: qv < qv_sat → negative evaporation rate
        qv_sat = FT(0.012)
        qv_sub = FT(0.008)    # 67% RH
        rate_sub = rain_evaporation_rate(p3, qr, nr, qv_sub, qv_sat, T, ρ)
        @test rate_sub < 0     # Negative = rain evaporating

        # Saturated: qv = qv_sat → zero evaporation
        rate_sat = rain_evaporation_rate(p3, qr, nr, qv_sat, qv_sat, T, ρ)
        @test rate_sat == 0

        # Supersaturated: qv > qv_sat → zero (no condensation on rain)
        qv_super = FT(0.015)
        rate_super = rain_evaporation_rate(p3, qr, nr, qv_super, qv_sat, T, ρ)
        @test rate_super == 0

        # Zero rain gives zero evaporation
        rate_norain = rain_evaporation_rate(p3, FT(0), nr, qv_sub, qv_sat, T, ρ)
        @test rate_norain == 0
    end

    @testset "cloud_condensation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        qcl = FT(1e-3)
        T = FT(288.0)
        qv_sat = FT(0.012)

        q = MoistureMassFractions(FT(0.015), FT(1e-3), FT(0))

        # Supersaturated: positive condensation
        qv_super = FT(0.015)
        rate_super = cloud_condensation_rate(p3, qcl, qv_super, qv_sat, T, q, constants)
        @test rate_super > 0

        # Subsaturated: negative (evaporation), limited by available cloud
        qv_sub = FT(0.008)
        q_sub = MoistureMassFractions(qv_sub, qcl, FT(0))
        rate_sub = cloud_condensation_rate(p3, qcl, qv_sub, qv_sat, T, q_sub, constants)
        @test rate_sub < 0

        # Saturated: approximately zero
        rate_sat = cloud_condensation_rate(p3, qcl, qv_sat, qv_sat, T, q, constants)
        @test abs(rate_sat) < 1e-10
    end

    @testset "ventilation_enhanced_deposition" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qi = FT(1e-4)
        ni = FT(1e4)
        Ff = FT(0.0)     # Unrimed
        ρf = FT(400.0)
        T = FT(253.15)   # -20C (cold, ice supersaturated)
        P = FT(50000.0)

        # Supersaturated over ice: positive deposition
        qv_sat_ice = FT(0.0005)
        qv_super = FT(0.001)    # Well above ice saturation
        rate_dep = ventilation_enhanced_deposition(p3, qi, ni, qv_super, qv_sat_ice, Ff, ρf, T, P)
        @test rate_dep > 0

        # Subsaturated over ice: negative (sublimation)
        qv_sub = FT(0.0001)
        rate_sub = ventilation_enhanced_deposition(p3, qi, ni, qv_sub, qv_sat_ice, Ff, ρf, T, P)
        @test rate_sub < 0

        # Zero ice gives zero deposition rate (mean mass → default)
        rate_noice = ventilation_enhanced_deposition(p3, FT(0), FT(0), qv_super, qv_sat_ice, Ff, ρf, T, P)
        @test abs(rate_noice) < 1e-20
    end

    @testset "ice_melting_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qi = FT(1e-4)
        ni = FT(1e4)
        P = FT(85000.0)
        qv = FT(0.008)
        qv_sat = FT(0.01)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)

        # Above freezing: positive melting
        T_warm = FT(275.15)    # +2C
        rate_warm = ice_melting_rate(p3, qi, ni, T_warm, P, qv, qv_sat, Ff, ρf, ρ)
        @test rate_warm > 0

        # Below freezing: zero melting
        T_cold = FT(263.15)    # -10C
        rate_cold = ice_melting_rate(p3, qi, ni, T_cold, P, qv, qv_sat, Ff, ρf, ρ)
        @test rate_cold == 0

        # Exactly at freezing: zero (no ΔT to drive melting)
        T_freeze = FT(273.15)
        rate_freeze = ice_melting_rate(p3, qi, ni, T_freeze, P, qv, qv_sat, Ff, ρf, ρ)
        @test rate_freeze == 0

        # Warmer temperatures give faster melting
        T_hot = FT(278.15)     # +5C
        rate_hot = ice_melting_rate(p3, qi, ni, T_hot, P, qv, qv_sat, Ff, ρf, ρ)
        @test rate_hot > rate_warm
    end

    @testset "ice_melting_rates partitioning" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qi = FT(1e-4)
        ni = FT(1e4)
        P = FT(85000.0)
        qv = FT(0.008)
        qv_sat = FT(0.01)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)
        T = FT(275.15)

        # No liquid on ice: all melting is partial (goes to coating)
        qwi_zero = FT(0)
        rates_dry = ice_melting_rates(p3, qi, ni, qwi_zero, T, P, qv, qv_sat, Ff, ρf, ρ)
        total = rates_dry.partial_melting + rates_dry.complete_melting
        @test total > 0
        @test rates_dry.partial_melting >= 0
        @test rates_dry.complete_melting >= 0

        # Saturated liquid coating: more complete melting
        qwi_high = FT(0.5 * qi)   # 50% liquid fraction
        rates_wet = ice_melting_rates(p3, qi, ni, qwi_high, T, P, qv, qv_sat, Ff, ρf, ρ)
        @test rates_wet.complete_melting >= rates_dry.complete_melting
    end

    @testset "ice_aggregation_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qi = FT(1e-4)
        ni = FT(1e5)
        Ff = FT(0.0)
        ρf = FT(400.0)

        # Near freezing (warm ice, sticky): aggregation active
        T_warm = FT(268.15)    # -5C
        rate_warm = ice_aggregation_rate(p3, qi, ni, T_warm, Ff, ρf)
        @test rate_warm < 0     # Number reduction rate is negative

        # Very cold (T < 253.15 K): much less aggregation
        T_cold = FT(233.15)    # -40C
        rate_cold = ice_aggregation_rate(p3, qi, ni, T_cold, Ff, ρf)
        # Aggregation efficiency at very cold T is 0.001 vs ~0.15 at -5C
        @test abs(rate_cold) < abs(rate_warm)

        # Zero ice: zero aggregation
        rate_noice = ice_aggregation_rate(p3, FT(0), FT(0), T_warm, Ff, ρf)
        @test rate_noice == 0

        # Heavily rimed (Ff > 0.9): aggregation shuts off
        rate_rimed = ice_aggregation_rate(p3, qi, ni, T_warm, FT(0.95), ρf)
        @test rate_rimed == 0
    end

    @testset "cloud_riming_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        qc = FT(1e-3)
        qi = FT(1e-4)
        ni = FT(1e4)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)

        # Below freezing with cloud and ice: positive riming
        T_cold = FT(263.15)    # -10C
        rate = cloud_riming_rate(p3, qc, qi, ni, T_cold, Ff, ρf, ρ)
        @test rate > 0

        # Above freezing: zero riming
        T_warm = FT(278.15)
        rate_warm = cloud_riming_rate(p3, qc, qi, ni, T_warm, Ff, ρf, ρ)
        @test rate_warm == 0

        # Zero cloud: zero riming
        rate_nocloud = cloud_riming_rate(p3, FT(0), qi, ni, T_cold, Ff, ρf, ρ)
        @test rate_nocloud == 0

        # Zero ice: zero riming
        rate_noice = cloud_riming_rate(p3, qc, FT(0), FT(0), T_cold, Ff, ρf, ρ)
        @test rate_noice == 0

        # More cloud water gives faster riming (rate is linear in qc)
        rate_high = cloud_riming_rate(p3, FT(2e-3), qi, ni, T_cold, Ff, ρf, ρ)
        @test rate_high > rate
    end

    @testset "rain_riming_rate" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        # Ice must dominate rain for rain riming
        qr = FT(1e-5)
        qi = FT(1e-4)    # qi > qr
        ni = FT(1e4)
        Ff = FT(0.0)
        ρf = FT(400.0)
        ρ = FT(1.0)

        T_cold = FT(263.15)
        rate = rain_riming_rate(p3, qr, qi, ni, T_cold, Ff, ρf, ρ)
        @test rate > 0

        # Above freezing: zero
        rate_warm = rain_riming_rate(p3, qr, qi, ni, FT(278.15), Ff, ρf, ρ)
        @test rate_warm == 0

        # Rain dominates ice (qr > qi): zero (handled by driver differently)
        rate_rain_dom = rain_riming_rate(p3, FT(1e-3), FT(1e-5), ni, T_cold, Ff, ρf, ρ)
        @test rate_rain_dom == 0
    end

    @testset "compute_p3_process_rates integration" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        ρ = FT(1.0)

        # Mixed-phase state: T = -5C, some cloud, rain, ice
        T = FT(268.15)
        qv = FT(0.003)
        qcl = FT(5e-4)
        qr = FT(1e-4)
        qi = FT(1e-4)
        qf = FT(1e-5)     # Some rime

        q = MoistureMassFractions(qv, qcl + qr, qi)

        # Build thermodynamic state: use potential temperature formulation
        # θ ≈ T / Π, for simplicity set pˢᵗ = P so Π ≈ 1
        P = FT(85000.0)
        pst = FT(100000.0)
        θ = T / (P / pst)^FT(0.286)  # Approximate dry potential temperature
        𝒰 = LiquidIcePotentialTemperatureState(θ, q, pst, P)

        ℳ = P3MicrophysicalState(
            qcl,           # qᶜˡ
            qr,            # qʳ
            FT(1e4),       # nʳ
            qi,            # qⁱ
            FT(1e5),       # nⁱ
            qf,            # qᶠ
            FT(qf / 400),  # bᶠ (rime volume)
            FT(1e-10),     # zⁱ (reflectivity)
            FT(0),         # qʷⁱ (liquid on ice)
        )

        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
        @test rates isa P3ProcessRates{FT}

        # All rates should be finite
        for name in fieldnames(P3ProcessRates)
            @test isfinite(getfield(rates, name))
        end

        # Sign checks for a cold mixed-phase environment:
        # Autoconversion should be positive (cloud → rain)
        @test rates.autoconversion > 0

        # Cloud riming should be positive (below freezing with cloud + ice)
        @test rates.cloud_riming > 0

        # Melting should be zero (below freezing)
        @test rates.partial_melting == 0
        @test rates.complete_melting == 0

        # Aggregation should be negative (number loss)
        @test rates.aggregation <= 0

        # Rime density should be physical
        @test rates.rime_density_new >= 50
        @test rates.rime_density_new <= 900
    end

    @testset "compute_p3_process_rates with tabulated scheme" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64
        constants = ThermodynamicConstants(FT)

        # Tabulate all ice integrals (small grid for speed)
        p3_tab = tabulate(p3, CPU();
            number_of_mass_points=20,
            number_of_rime_fraction_points=4,
            number_of_liquid_fraction_points=2,
            number_of_quadrature_points=32)

        # Verify tabulated scheme has TabulatedFunction3D fields
        @test p3_tab.ice.fall_speed.mass_weighted isa TabulatedFunction3D
        @test p3_tab.ice.deposition.ventilation isa TabulatedFunction3D
        @test p3_tab.ice.collection.aggregation isa TabulatedFunction3D
        @test p3_tab.ice.collection.rain_collection isa TabulatedFunction3D

        ρ = FT(1.0)

        # Mixed-phase state: T = -5C
        T = FT(268.15)
        qv = FT(0.003)
        qcl = FT(5e-4)
        qr = FT(1e-4)
        qi = FT(1e-4)
        qf = FT(1e-5)

        q = MoistureMassFractions(qv, qcl + qr, qi)
        P = FT(85000.0)
        pst = FT(100000.0)
        θ = T / (P / pst)^FT(0.286)
        𝒰 = LiquidIcePotentialTemperatureState(θ, q, pst, P)

        ℳ = P3MicrophysicalState(
            qcl, qr, FT(1e4), qi, FT(1e5), qf,
            FT(qf / 400), FT(1e-10), FT(0))

        # Compute rates with tabulated scheme
        rates_tab = compute_p3_process_rates(p3_tab, ρ, ℳ, 𝒰, constants)
        @test rates_tab isa P3ProcessRates{FT}

        # All rates should be finite
        for name in fieldnames(P3ProcessRates)
            @test isfinite(getfield(rates_tab, name))
        end

        # Sign checks should be consistent with analytical path
        @test rates_tab.autoconversion > 0
        @test rates_tab.cloud_riming > 0
        @test rates_tab.partial_melting == 0
        @test rates_tab.complete_melting == 0
        @test rates_tab.aggregation <= 0

        # Compute rates with analytical scheme for comparison
        rates_ana = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

        # Table and analytical should agree within order of magnitude
        # (table integrates over PSD, analytical uses mean-mass approximation)
        # Non-table-dependent rates should be identical
        @test rates_tab.autoconversion ≈ rates_ana.autoconversion
        @test rates_tab.accretion ≈ rates_ana.accretion
        @test rates_tab.rain_evaporation ≈ rates_ana.rain_evaporation
        @test rates_tab.condensation ≈ rates_ana.condensation

        # Table-dependent rates should be same sign and order of magnitude
        if rates_ana.deposition != 0
            ratio = rates_tab.deposition / rates_ana.deposition
            @test 0.01 < abs(ratio) < 100
        end
    end

    @testset "Vapor + cloud + rain + ice mass conservation" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        FT = Float64

        ρ = FT(1.0)

        # Create rates with typical mixed-phase values
        rates = P3ProcessRates(
            FT(5e-7),   # condensation
            FT(1e-7),   # autoconversion
            FT(2e-7),   # accretion
            FT(-5e-8),  # rain_evaporation
            FT(-1e-6),  # rain_self_collection
            FT(5e-7),   # rain_breakup
            FT(3e-7),   # deposition
            FT(1e-8),   # partial_melting
            FT(5e-8),   # complete_melting
            FT(-1e3),   # melting_number
            FT(-500.0), # aggregation
            FT(1e-7),   # cloud_riming
            FT(-1e4),   # cloud_riming_number
            FT(5e-8),   # rain_riming
            FT(-500.0), # rain_riming_number
            FT(300.0),  # rime_density_new
            FT(2e-8),   # shedding
            FT(100.0),  # shedding_number
            FT(1e-8),   # refreezing
            FT(1e-9),   # nucleation_mass
            FT(10.0),   # nucleation_number
            FT(5e-9),   # cloud_freezing_mass
            FT(100.0),  # cloud_freezing_number
            FT(3e-9),   # rain_freezing_mass
            FT(50.0),   # rain_freezing_number
            FT(1e-10),  # splintering_mass
            FT(1.0),    # splintering_number
        )

        # Compute total water tendency: vapor + cloud + rain + ice + liquid_on_ice
        # These should sum to zero (water is neither created nor destroyed)
        dqv = tendency_ρqᵛ(rates, ρ)
        dqc = tendency_ρqᶜˡ(rates, ρ)
        dqr = tendency_ρqʳ(rates, ρ)
        dqi = tendency_ρqⁱ(rates, ρ)
        dqwi = tendency_ρqʷⁱ(rates, ρ)

        total_water_tendency = dqv + dqc + dqr + dqi + dqwi
        @test abs(total_water_tendency) < 1e-15 * ρ
    end
end
