using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

using Breeze.Thermodynamics:
    saturation_vapor_pressure,
    dry_air_gas_constant,
    vapor_gas_constant,
    PlanarLiquidSurface

# Test setting relative humidity in AtmosphereModel

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

    @testset "Saturation adjustment: ℋ > 1 throws error (scalar)" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        ℋ₀ = FT(1.5)  # 150% - supersaturated

        set!(model, θ=θ₀)  # Set θ first
        @test_throws ArgumentError set!(model, ℋ=ℋ₀)
    end

    @testset "Saturation adjustment: ℋ > 1 throws error (array)" begin
        microphysics = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state),
                                microphysics)

        θ₀ = FT(300)
        # Array with one supersaturated value
        ℋ_array = fill(FT(0.8), 1, 1, 8)
        ℋ_array[1, 1, 4] = FT(1.1)  # One supersaturated point

        set!(model, θ=θ₀)
        @test_throws ArgumentError set!(model, ℋ=ℋ_array)
    end

    @testset "No microphysics: ℋ > 1 is allowed (no saturation adjustment)" begin
        # Without saturation adjustment, we should be able to set ℋ > 1
        # (though it's physically unrealistic without condensate removal)
        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=AnelasticDynamics(reference_state))

        θ₀ = FT(300)
        ℋ₀ = FT(1.2)  # 120% - supersaturated

        # This should NOT throw an error since there's no saturation adjustment
        set!(model, θ=θ₀, ℋ=ℋ₀)

        # Verify the moisture was set (qᵗ > 0)
        qᵗ = model.specific_moisture
        @test @allowscalar all(x -> x > 0, interior(qᵗ))
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
