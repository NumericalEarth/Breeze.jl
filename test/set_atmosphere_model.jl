using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

#####
##### Setting potential temperature
#####

@testset "Setting potential temperature (no microphysics) [$(FT)]" for FT in test_float_types(), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
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

@testset "Setting temperature directly [$(FT), $(formulation)]" for FT in test_float_types(), formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
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

@testset "Setting relative humidity [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch; size=(2, 2, 8), x=(0, 1e3), y=(0, 1e3), z=(0, 1e3))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())

    @testset "Scalar ℋ (subsaturated)" begin
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        set!(model, θ=FT(300), ℋ=FT(0.5))

        # Verify the moisture was set correctly using the RelativeHumidity diagnostic
        ℋ_field = RelativeHumidityField(model)
        @test @allowscalar all(isapprox.(interior(ℋ_field), FT(0.5); rtol=5e-2))
        @test @allowscalar all(x -> x > 0, interior(model.specific_moisture))
    end

    # Function inputs don't work on GPU (non-bitstype argument error)
    if default_arch isa CPU
        @testset "Function ℋ (spatially-varying)" begin
            model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                    dynamics=AnelasticDynamics(reference_state),
                                    microphysics)

            ℋ_func(x, y, z) = FT(0.8) * exp(-z / FT(500))
            set!(model, θ=FT(300), ℋ=ℋ_func)

            ℋ_field = RelativeHumidityField(model)
            @allowscalar for k in 1:8
                z = znodes(grid, Center())[k]
                @test isapprox(interior(ℋ_field, 1, 1, k)[1], ℋ_func(0, 0, z); rtol=5e-2)
            end
        end
    end

    @testset "Supersaturated ℋ creates cloud liquid" begin
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        set!(model, θ=FT(300), ℋ=FT(1.5))

        # After saturation adjustment, there should be cloud liquid
        @test @allowscalar all(x -> x > 0, interior(model.microphysical_fields.qˡ))
        # And relative humidity should be capped at 1
        ℋ_field = RelativeHumidityField(model)
        @test @allowscalar all(x -> x ≤ 1.01, interior(ℋ_field))
    end
end
