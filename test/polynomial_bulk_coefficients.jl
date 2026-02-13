using Test
using Breeze
using Breeze.BoundaryConditions
using Breeze.BoundaryConditions: PolynomialBulkCoefficient,
                                 default_stability_function,
                                 neutral_coefficient_10m,
                                 adjust_coefficient_for_height,
                                 bulk_richardson_number
using Oceananigans.BoundaryConditions: BoundaryCondition

@testset "PolynomialBulkCoefficient" begin
    @testset "Constructor and defaults" begin
        # Test default constructor
        coef = PolynomialBulkCoefficient()
        @test coef.neutral_coefficients === nothing
        @test coef.roughness_length == 1.5e-4
        @test coef.stability_function === default_stability_function

        # Test explicit coefficients
        drag_coef = PolynomialBulkCoefficient(neutral_coefficients = (0.142, 0.076, 2.7))
        @test drag_coef.neutral_coefficients == (0.142, 0.076, 2.7)

        heat_coef = PolynomialBulkCoefficient(neutral_coefficients = (0.128, 0.068, 2.43))
        @test heat_coef.neutral_coefficients == (0.128, 0.068, 2.43)

        moisture_coef = PolynomialBulkCoefficient(neutral_coefficients = (0.120, 0.070, 2.55))
        @test moisture_coef.neutral_coefficients == (0.120, 0.070, 2.55)

        # Test custom coefficients
        custom_coef = PolynomialBulkCoefficient(
            neutral_coefficients = (1.0, 0.5, 0.1),
            roughness_length = 1e-3,
            stability_function = nothing
        )
        @test custom_coef.neutral_coefficients == (1.0, 0.5, 0.1)
        @test isnothing(custom_coef.stability_function)
    end

    @testset "Neutral coefficient computation" begin
        # Test Large & Yeager form at U = 10 m/s
        coeffs = (0.142, 0.076, 2.7)  # Large & Yeager (2009) drag coefficients
        U = 10.0
        U_min = 0.1
        C = neutral_coefficient_10m(coeffs, U, U_min)

        # C_D = (0.142 + 0.076*10 + 2.7/10) × 10⁻³
        #     = (0.142 + 0.76 + 0.27) × 10⁻³
        #     = 1.172 × 10⁻³
        @test C ≈ 1.172e-3 atol=1e-6

        # Test at low wind speed
        U_low = 1.0
        C_low = neutral_coefficient_10m(coeffs, U_low, U_min)
        # C_D = (0.142 + 0.076*1 + 2.7/1) × 10⁻³ = 2.918 × 10⁻³
        @test C_low ≈ 2.918e-3 atol=1e-6

        # Test at high wind speed
        U_high = 20.0
        C_high = neutral_coefficient_10m(coeffs, U_high, U_min)
        # C_D = (0.142 + 0.076*20 + 2.7/20) × 10⁻³
        #     = (0.142 + 1.52 + 0.135) × 10⁻³ = 1.797 × 10⁻³
        @test C_high ≈ 1.797e-3 atol=1e-6
    end

    @testset "Height adjustment" begin
        C₁₀ = 1.0e-3
        z₀ = 1.5e-4

        # At 10m, should return same value
        C_10m = adjust_coefficient_for_height(C₁₀, 10.0, z₀)
        @test C_10m ≈ C₁₀

        # At higher elevation, coefficient should be smaller (log profile)
        C_50m = adjust_coefficient_for_height(C₁₀, 50.0, z₀)
        @test C_50m < C₁₀

        # At lower elevation, coefficient should be larger
        C_5m = adjust_coefficient_for_height(C₁₀, 5.0, z₀)
        @test C_5m > C₁₀
    end

    @testset "Bulk Richardson number" begin
        z = 10.0
        θᵥ_sfc = 290.0
        U = 10.0
        U_min = 0.1

        # Unstable case: warm surface (θᵥ < θᵥ_sfc)
        θᵥ_unstable = 288.0
        Ri_unstable = bulk_richardson_number(z, θᵥ_unstable, θᵥ_sfc, U, U_min)
        @test Ri_unstable < 0

        # Neutral case: same temperature
        Ri_neutral = bulk_richardson_number(z, θᵥ_sfc, θᵥ_sfc, U, U_min)
        @test Ri_neutral ≈ 0.0 atol=1e-10

        # Stable case: cold surface (θᵥ > θᵥ_sfc)
        θᵥ_stable = 292.0
        Ri_stable = bulk_richardson_number(z, θᵥ_stable, θᵥ_sfc, U, U_min)
        @test Ri_stable > 0
    end

    @testset "Stability functions" begin
        # Unstable conditions (Ri < 0) enhance transfer
        Ri_unstable = -0.1
        ψ_unstable = default_stability_function(Ri_unstable)
        @test ψ_unstable > 1.0

        # Neutral conditions (Ri = 0) don't modify coefficient
        Ri_neutral = 0.0
        ψ_neutral = default_stability_function(Ri_neutral)
        @test ψ_neutral ≈ 1.0

        # Stable conditions (Ri > 0) reduce transfer
        Ri_stable = 0.1
        ψ_stable = default_stability_function(Ri_stable)
        @test ψ_stable < 1.0
    end

    @testset "Callable interface" begin
        # Test evaluation with no stability
        coef = PolynomialBulkCoefficient(
            neutral_coefficients = (0.142, 0.076, 2.7),
            stability_function = nothing
        )
        U = 10.0
        z = 10.0
        θᵥ = 288.0
        θᵥ₀ = 290.0
        constants = nothing
        C = coef(U, θᵥ, θᵥ₀, z, constants)
        @test C isa Number
        @test C > 0

        # Test with stability correction
        coef_stable = PolynomialBulkCoefficient(
            neutral_coefficients = (0.142, 0.076, 2.7),
            stability_function = default_stability_function
        )
        C_stable = coef_stable(U, θᵥ, θᵥ₀, z, constants)
        @test C_stable isa Number
        @test C_stable > 0
        # Unstable conditions should enhance transfer
        @test C_stable > C
    end

    @testset "Integration with BulkDrag" begin
        # Test that PolynomialBulkCoefficient works with BulkDrag constructor
        coef = PolynomialBulkCoefficient()
        bc = BulkDrag(coef, gustiness = 0.5)
        @test bc isa BoundaryCondition
        # Coefficient should have been materialized with momentum coefficients
        @test bc.condition.coefficient.neutral_coefficients == (0.142, 0.076, 2.7)
        @test bc.condition.gustiness == 0.5
    end

    @testset "Integration with BulkSensibleHeatFlux" begin
        # Test that PolynomialBulkCoefficient works with BulkSensibleHeatFlux
        coef = PolynomialBulkCoefficient()
        SST(x, y) = 300.0
        bc = BulkSensibleHeatFlux(coef, surface_temperature = SST)
        @test bc isa BoundaryCondition
        # Coefficient should have been materialized with sensible heat coefficients
        @test bc.condition.coefficient.neutral_coefficients == (0.128, 0.068, 2.43)
    end

    @testset "Integration with BulkVaporFlux" begin
        # Test that PolynomialBulkCoefficient works with BulkVaporFlux
        coef = PolynomialBulkCoefficient()
        SST(x, y) = 300.0
        bc = BulkVaporFlux(coef, surface_temperature = SST)
        @test bc isa BoundaryCondition
        # Coefficient should have been materialized with latent heat coefficients
        @test bc.condition.coefficient.neutral_coefficients == (0.120, 0.070, 2.55)
    end
end
