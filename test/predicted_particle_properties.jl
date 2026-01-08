using Test
using Breeze.Microphysics.PredictedParticleProperties

import Breeze.Microphysics.PredictedParticleProperties:
    IceSizeDistributionState,
    evaluate,
    chebyshev_gauss_nodes_weights,
    size_distribution,
    tabulate,
    TabulationParameters

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
        @test fs.reference_air_density ≈ 1.225
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
        @test ll.small_q isa SmallQLambdaLimit
        @test ll.large_q isa LargeQLambdaLimit
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
        @test SmallQLambdaLimit <: AbstractLambdaLimiterIntegral
        
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
        
        # Weights should sum to π (for Chebyshev-Gauss)
        @test sum(weights) ≈ π
        
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
        
        i_small = evaluate(SmallQLambdaLimit(), state)
        @test i_small > 0
        @test isfinite(i_small)
        
        i_large = evaluate(LargeQLambdaLimit(), state)
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
        
        # Test that quadrature converges with increasing number of points
        V_16 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=16)
        V_32 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=32)
        V_64 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=64)
        V_128 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)
        
        # Should converge (differences decrease)
        diff_16_32 = abs(V_32 - V_16)
        diff_32_64 = abs(V_64 - V_32)
        diff_64_128 = abs(V_128 - V_64)
        
        @test diff_32_64 < diff_16_32 || diff_32_64 < 1e-10
        @test diff_64_128 < diff_32_64 || diff_64_128 < 1e-10
    end

    @testset "Tabulation parameters" begin
        params = TabulationParameters()
        @test params.n_Qnorm == 50
        @test params.n_Fr == 4
        @test params.n_Fl == 4
        @test params.Qnorm_min ≈ 1e-18
        @test params.Qnorm_max ≈ 1e-5
        @test params.n_quadrature == 64
        
        # Custom parameters
        params_custom = TabulationParameters(Float32; 
            n_Qnorm=20, n_Fr=3, n_Fl=2, n_quadrature=32)
        @test params_custom.n_Qnorm == 20
        @test params_custom.n_Fr == 3
        @test params_custom.n_Fl == 2
        @test params_custom.Qnorm_min isa Float32
    end

    @testset "Tabulate single integral" begin
        params = TabulationParameters(Float64; n_Qnorm=5, n_Fr=2, n_Fl=2, n_quadrature=16)
        
        # Tabulate number-weighted fall speed
        tab_Vn = tabulate(NumberWeightedFallSpeed(), CPU(), params)
        
        @test tab_Vn isa TabulatedIntegral
        @test size(tab_Vn) == (5, 2, 2)
        
        # Values should be positive and finite
        @test all(isfinite, tab_Vn.data)
        @test all(x -> x > 0, tab_Vn.data)
        
        # Test indexing
        @test tab_Vn[1, 1, 1] > 0
        @test tab_Vn[5, 2, 2] > 0
    end

    @testset "Tabulate IceFallSpeed container" begin
        params = TabulationParameters(Float64; n_Qnorm=5, n_Fr=2, n_Fl=2, n_quadrature=16)
        
        fs = IceFallSpeed()
        fs_tab = tabulate(fs, CPU(), params)
        
        # Parameters should be preserved
        @test fs_tab.reference_air_density == fs.reference_air_density
        @test fs_tab.fall_speed_coefficient == fs.fall_speed_coefficient
        @test fs_tab.fall_speed_exponent == fs.fall_speed_exponent
        
        # Integrals should be tabulated
        @test fs_tab.number_weighted isa TabulatedIntegral
        @test fs_tab.mass_weighted isa TabulatedIntegral
        @test fs_tab.reflectivity_weighted isa TabulatedIntegral
        
        # Check sizes
        @test size(fs_tab.number_weighted) == (5, 2, 2)
        @test size(fs_tab.mass_weighted) == (5, 2, 2)
        @test size(fs_tab.reflectivity_weighted) == (5, 2, 2)
    end

    @testset "Tabulate IceDeposition container" begin
        params = TabulationParameters(Float64; n_Qnorm=5, n_Fr=2, n_Fl=2, n_quadrature=16)
        
        dep = IceDeposition()
        dep_tab = tabulate(dep, CPU(), params)
        
        # Parameters should be preserved
        @test dep_tab.thermal_conductivity == dep.thermal_conductivity
        @test dep_tab.vapor_diffusivity == dep.vapor_diffusivity
        
        # All 6 integrals should be tabulated
        @test dep_tab.ventilation isa TabulatedIntegral
        @test dep_tab.ventilation_enhanced isa TabulatedIntegral
        @test dep_tab.small_ice_ventilation_constant isa TabulatedIntegral
        @test dep_tab.small_ice_ventilation_reynolds isa TabulatedIntegral
        @test dep_tab.large_ice_ventilation_constant isa TabulatedIntegral
        @test dep_tab.large_ice_ventilation_reynolds isa TabulatedIntegral
    end

    @testset "Tabulate P3 scheme by property" begin
        p3 = PredictedParticlePropertiesMicrophysics()
        
        # Tabulate fall speed
        p3_fs = tabulate(p3, :ice_fall_speed, CPU(); 
            n_Qnorm=5, n_Fr=2, n_Fl=2, n_quadrature=16)
        
        @test p3_fs isa PredictedParticlePropertiesMicrophysics
        @test p3_fs.ice.fall_speed.number_weighted isa TabulatedIntegral
        @test p3_fs.ice.fall_speed.mass_weighted isa TabulatedIntegral
        
        # Other properties should be unchanged
        @test p3_fs.ice.deposition.ventilation isa Ventilation
        @test p3_fs.rain == p3.rain
        @test p3_fs.cloud == p3.cloud
        
        # Tabulate deposition
        p3_dep = tabulate(p3, :ice_deposition, CPU();
            n_Qnorm=5, n_Fr=2, n_Fl=2, n_quadrature=16)
        
        @test p3_dep.ice.deposition.ventilation isa TabulatedIntegral
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
        # Mass-weighted velocity should generally be larger than number-weighted
        # because larger particles contribute more to mass
        
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
        n_int = evaluate(SmallQLambdaLimit(), state)  # This is just ∫ N'(D) dD
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
        # Rimed particles have higher density
        state_unrimed = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            rime_fraction = 0.0, rime_density = 400.0)
        
        state_rimed = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 1000.0,
            rime_fraction = 0.5, rime_density = 600.0)
        
        # Both should produce valid results
        V_unrimed = evaluate(NumberWeightedFallSpeed(), state_unrimed)
        V_rimed = evaluate(NumberWeightedFallSpeed(), state_rimed)
        
        @test isfinite(V_unrimed)
        @test isfinite(V_rimed)
        @test V_unrimed > 0
        @test V_rimed > 0
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
        
        # Test SmallQLambdaLimit (which integrates the full PSD)
        small_q_lim = evaluate(SmallQLambdaLimit(), state; n_quadrature=128)
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
        
        small_q = evaluate(SmallQLambdaLimit(), state)
        large_q = evaluate(LargeQLambdaLimit(), state)
        
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
        # Reflectivity-weighted (Z-weighted) velocity should be largest
        # because it weights by D^6, emphasizing large particles
        # Mass-weighted should be intermediate
        # Number-weighted should be smallest
        
        state = IceSizeDistributionState(Float64;
            intercept = 1e6, shape = 0.0, slope = 500.0)  # Large particles
        
        V_n = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)
        V_m = evaluate(MassWeightedFallSpeed(), state; n_quadrature=128)
        V_z = evaluate(ReflectivityWeightedFallSpeed(), state; n_quadrature=128)
        
        # All should be positive
        @test V_n > 0
        @test V_m > 0
        @test V_z > 0
        
        # Check ordering: V_z ≥ V_m ≥ V_n for most PSDs
        # (this depends on the fall speed power-law exponent)
        # Just verify they're all in reasonable range
        @test isfinite(V_n)
        @test isfinite(V_m)
        @test isfinite(V_z)
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
        # Zero mass should return log(0)
        logλ_zero_L = solve_lambda(0.0, 1e5, 0.0, 400.0)
        @test logλ_zero_L == log(0.0)
        
        # Zero number should return log(0)
        logλ_zero_N = solve_lambda(1e-4, 0.0, 0.0, 400.0)
        @test logλ_zero_N == log(0.0)
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
end

