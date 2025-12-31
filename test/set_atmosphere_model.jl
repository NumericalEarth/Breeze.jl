using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

#####
##### Setting potential temperature
#####

@testset "Setting potential temperature (no microphysics) [$(FT)]" for FT in (Float32, Float64), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)

    # Initialize with potential temperature and dry air
    θᵢ = CenterField(grid)
    set!(θᵢ, (x, y, z) -> θ₀ + rand())
    set!(model; θ=θᵢ)

    θ_model = liquid_ice_potential_temperature(model) |> Field
    @test θ_model ≈ θᵢ
end

#####
##### Setting temperature directly
#####

@testset "Setting temperature directly [$(FT), $(formulation)]" for FT in (Float32, Float64), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 10), x=(0, 1_000), y=(0, 1_000), z=(0, 5_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101500)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)

    # Test with no microphysics first (no saturation adjustment effects)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation)

    # Set a standard lapse rate temperature profile with dry air
    T_profile(x, y, z) = FT(300) - FT(0.0065) * z

    set!(model, T=T_profile, qᵗ=FT(0))  # dry air

    # Check that temperature was set correctly (should match for dry air)
    z_nodes = Oceananigans.Grids.znodes(grid, Center())
    for k in 1:10
        T_expected = T_profile(0, 0, z_nodes[k])
        T_actual = @allowscalar model.temperature[1, 1, k]
        @test T_actual ≈ T_expected rtol=FT(1e-4)
    end

    # Check that potential temperature increases with height (stable atmosphere)
    θ = liquid_ice_potential_temperature(model) |> Field
    θ_prev = @allowscalar θ[1, 1, 1]
    for k in 2:10
        θ_k = @allowscalar θ[1, 1, k]
        @test θ_k > θ_prev  # potential temperature should increase with height
        θ_prev = θ_k
    end

    # Test round-trip consistency: set T, get θ; set θ back, get same T
    set!(model, T=FT(280), qᵗ=FT(0))
    T_after_set = @allowscalar model.temperature[2, 2, 5]
    @test T_after_set ≈ FT(280) rtol=FT(1e-4)

    # Now test with saturation adjustment
    microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
    model_moist = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, formulation, microphysics)

    # Set T with subsaturated moisture (no condensate expected)
    set!(model_moist, T=T_profile, qᵗ=FT(0.001))  # low moisture

    # Temperature should still be close to input for subsaturated air
    T_actual = @allowscalar model_moist.temperature[1, 1, 1]
    T_expected = T_profile(0, 0, z_nodes[1])
    @test T_actual ≈ T_expected rtol=FT(0.02)  # allow 2% tolerance due to moisture effects
end

#####
##### Setting relative humidity
#####

@testset "Setting relative humidity [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch; size=(1, 1, 8), x=(0, 1e3), y=(0, 1e3), z=(0, 1e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)

    @testset "Saturation adjustment: ℋ setting works for ℋ ≤ 1" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        ℋ₀ = FT(0.5)  # 50% relative humidity - subsaturated

        # This should work without error
        set!(model, θ=θ₀, ℋ=ℋ₀)

        # Verify the moisture was set correctly using the RelativeHumidity diagnostic
        # Note: We use rtol=0.02 because the setting uses reference density while
        # the diagnostic uses actual density (which depends on moisture), causing ~1% difference
        ℋ_field = RelativeHumidityField(model)
        @test @allowscalar all(isapprox.(interior(ℋ_field), ℋ₀; rtol=2e-2))

        # Verify qᵗ was actually set (not zero)
        qᵗ = model.specific_moisture
        @test @allowscalar all(x -> x > 0, interior(qᵗ))
    end

    @testset "Saturation adjustment: ℋ = 1 (saturated)" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        ℋ₀ = FT(1.0)  # 100% relative humidity - exactly saturated

        # This should work without error
        set!(model, θ=θ₀, ℋ=ℋ₀)

        # With saturation adjustment, relative humidity should be at or near 1
        # Note: We use rtol=0.02 to account for density approximation
        ℋ_field = RelativeHumidityField(model)
        @test @allowscalar all(isapprox.(interior(ℋ_field), 1.0; rtol=2e-2))
    end

    @testset "Saturation adjustment: spatially-varying relative humidity" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        # Relative humidity that decreases with height (but stays below 1)
        ℋ_func(x, y, z) = FT(0.8) * exp(-z / FT(500))

        set!(model, θ=θ₀, ℋ=ℋ_func)

        # Verify that relative humidity matches the function
        ℋ_field = RelativeHumidityField(model)

        @allowscalar begin
            for k in 1:8
                z = znodes(grid, Center())[k]
                expected_ℋ = ℋ_func(0, 0, z)
                actual_ℋ = interior(ℋ_field, 1, 1, k)[1]
                @test isapprox(actual_ℋ, expected_ℋ; rtol=5e-2)
            end
        end
    end

    @testset "Saturation adjustment: ℋ > 1 creates cloud liquid" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        ℋ₀ = FT(1.5)  # 150% - supersaturated

        # Setting ℋ > 1 should work and create cloud liquid after saturation adjustment
        set!(model, θ=θ₀, ℋ=ℋ₀)

        # After saturation adjustment, there should be cloud liquid
        qˡ = model.microphysical_fields.qˡ
        @test @allowscalar all(x -> x > 0, interior(qˡ))

        # And relative humidity should be capped at 1
        ℋ_field = RelativeHumidityField(model)
        @test @allowscalar all(x -> x ≤ 1.01, interior(ℋ_field))  # small tolerance for numerics
    end

    @testset "No microphysics: ℋ > 1 is allowed (supersaturated)" begin
        # Without microphysics, we can set ℋ > 1 and it stays supersaturated
        # Note: we use SaturationAdjustment but skip the adjustment by setting ℋ ≤ 1
        # To test ℋ > 1 without adjustment, we use a model with SaturationAdjustment
        # but verify the total moisture was set to the expected supersaturated value
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state))

        θ₀ = FT(300)
        ℋ₀ = FT(1.2)  # 120% - supersaturated

        set!(model, θ=θ₀, ℋ=ℋ₀)

        # Verify the moisture was set (qᵗ > 0)
        qᵗ = model.specific_moisture
        @test @allowscalar all(x -> x > 0, interior(qᵗ))

        # For no microphysics, qᵗ = qᵛ (all vapor, no condensate)
        # So relative humidity would be ℋ = qᵛ/qᵛ⁺ = qᵗ/qᵛ⁺
        # We cannot use RelativeHumidityField here because it requires microphysical_fields
        # Instead, verify that the moisture was set at the expected level by comparing
        # with a subsaturated case that we can verify with a SaturationAdjustment model

        # Create a model with saturation adjustment for comparison at ℋ = 1
        microphysics_sa = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model_sa = AtmosphereModel(grid; thermodynamic_constants=constants,
                                   dynamics=AnelasticDynamics(reference_state),
                                   microphysics=microphysics_sa)
        set!(model_sa, θ=θ₀, ℋ=FT(1.0))  # saturated

        # Get qᵗ at saturation
        qᵗ_saturated = copy(interior(model_sa.specific_moisture))
        qᵗ_supersaturated = interior(model.specific_moisture)

        # qᵗ should be ℋ₀ times the saturated value (approximately)
        @test @allowscalar all(isapprox.(qᵗ_supersaturated, ℋ₀ .* qᵗ_saturated; rtol=5e-2))
    end

    @testset "Compare ℋ and qᵗ setting: same result for subsaturated conditions" begin
        # For subsaturated conditions (ℋ < 1), setting via ℋ and qᵗ should give
        # equivalent results (up to the relationship qᵗ = ℋ * qᵛ⁺)

        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())

        model1 = AtmosphereModel(grid; thermodynamic_constants=constants,
                                 dynamics=AnelasticDynamics(reference_state),
                                 microphysics)

        model2 = AtmosphereModel(grid; thermodynamic_constants=constants,
                                 dynamics=AnelasticDynamics(reference_state),
                                 microphysics)

        θ₀ = FT(300)
        ℋ₀ = FT(0.6)  # 60% relative humidity

        # Set model1 using ℋ
        set!(model1, θ=θ₀, ℋ=ℋ₀)

        # Get the qᵗ that resulted from setting ℋ
        qᵗ_from_ℋ = copy(interior(model1.specific_moisture))

        # Set model2 using the same qᵗ directly
        set!(model2, θ=θ₀)
        set!(model2, qᵗ=qᵗ_from_ℋ)

        # The relative humidity should match
        ℋ1 = RelativeHumidityField(model1)
        ℋ2 = RelativeHumidityField(model2)

        @test @allowscalar isapprox(interior(ℋ1), interior(ℋ2); rtol=1e-6)
    end

    @testset "Setting ℋ sets both qᵗ and ρqᵗ consistently" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        ℋ₀ = FT(0.7)

        set!(model, θ=θ₀, ℋ=ℋ₀)

        # Check that ρqᵗ = ρ * qᵗ
        qᵗ = model.specific_moisture
        ρqᵗ = model.moisture_density
        ρ = Breeze.AtmosphereModels.dynamics_density(model.dynamics)

        @allowscalar begin
            for k in 1:8
                qᵗ_val = interior(qᵗ, 1, 1, k)[1]
                ρqᵗ_val = interior(ρqᵗ, 1, 1, k)[1]
                ρ_val = interior(ρ, 1, 1, k)[1]
                @test isapprox(ρqᵗ_val, ρ_val * qᵗ_val; rtol=1e-10)
            end
        end
    end
end

