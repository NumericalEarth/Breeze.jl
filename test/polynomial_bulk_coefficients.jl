using Test
using Breeze
using Breeze.BoundaryConditions
using Breeze.BoundaryConditions: PolynomialCoefficient,
                                 FittedStabilityFunction,
                                 StabilityFunctionParameters,
                                 RichardsonNumberMapping,
                                 neutral_coefficient_10m,
                                 bulk_richardson_number,
                                 bulk_to_flux_richardson_number,
                                 integrated_stability_momentum,
                                 integrated_stability_scalar,
                                 stability_correction_factor
using Oceananigans
using Oceananigans.BoundaryConditions: BoundaryCondition

@testset "PolynomialCoefficient" begin
    @testset "Constructor and defaults" begin
        # Test default constructor — uses FittedStabilityFunction
        coef = PolynomialCoefficient()
        @test coef.polynomial === nothing
        @test coef.roughness_length == 1.5e-4
        @test coef.stability_function isa FittedStabilityFunction
        @test coef.stability_function.scalar_roughness_length ≈ 1.5e-4 / 7.3

        # Scalar roughness length follows roughness_length / 7.3
        coef_custom_rl = PolynomialCoefficient(roughness_length = 1e-3)
        @test coef_custom_rl.stability_function.scalar_roughness_length ≈ 1e-3 / 7.3

        # FittedStabilityFunction embeds parameter structs
        sf = coef.stability_function
        @test sf.richardson_number_mapping isa RichardsonNumberMapping
        @test sf.stability_function_parameters isa StabilityFunctionParameters
        @test sf.richardson_number_mapping.strongly_stable_transition ≈ 0.2

        # Test explicit coefficients
        drag_coef = PolynomialCoefficient(polynomial = (0.142, 0.076, 2.7))
        @test drag_coef.polynomial == (0.142, 0.076, 2.7)

        heat_coef = PolynomialCoefficient(polynomial = (0.128, 0.068, 2.43))
        @test heat_coef.polynomial == (0.128, 0.068, 2.43)

        moisture_coef = PolynomialCoefficient(polynomial = (0.120, 0.070, 2.55))
        @test moisture_coef.polynomial == (0.120, 0.070, 2.55)

        # Test custom coefficients with no stability
        custom_coef = PolynomialCoefficient(
            polynomial = (1.0, 0.5, 0.1),
            roughness_length = 1e-3,
            stability_function = nothing
        )
        @test custom_coef.polynomial == (1.0, 0.5, 0.1)
        @test isnothing(custom_coef.stability_function)
    end

    @testset "Parameter structs" begin
        # Default RichardsonNumberMapping has Li et al. (2010) values
        mapping = RichardsonNumberMapping()
        @test mapping.stable_unstable_transition ≈ 0.0
        @test mapping.strongly_stable_transition ≈ 0.2
        @test mapping.aᵘ₁₁ ≈ 0.0450
        @test mapping.aᵘ₂₂ ≈ 0.8845
        @test mapping.aˢ₁₁ ≈ 0.7529

        # Custom thresholds
        mapping_custom = RichardsonNumberMapping(strongly_stable_transition = 0.3)
        @test mapping_custom.strongly_stable_transition ≈ 0.3

        # Default StabilityFunctionParameters
        params = StabilityFunctionParameters()
        @test params.γᴰ ≈ 19.3
        @test params.γᵀ ≈ 11.6
        @test params.a ≈ 1.0
        @test params.b ≈ 2/3
        @test params.c ≈ 5.0
        @test params.d ≈ 0.35

        # Custom parameters
        params_custom = StabilityFunctionParameters(γᴰ = 16.0, γᵀ = 12.0)
        @test params_custom.γᴰ ≈ 16.0
        @test params_custom.γᵀ ≈ 12.0
    end

    @testset "Neutral coefficient computation" begin
        # Test Large & Yeager form at U = 10 m/s
        coeffs = (0.142, 0.076, 2.7)  # Large & Yeager (2009) drag coefficients
        U = 10.0
        U_min = 0.1
        C = neutral_coefficient_10m(coeffs, U, U_min)

        # Cᴰ = (0.142 + 0.076*10 + 2.7/10) × 10⁻³
        #     = (0.142 + 0.76 + 0.27) × 10⁻³
        #     = 1.172 × 10⁻³
        @test C ≈ 1.172e-3 atol=1e-6

        # Test at low wind speed
        U_low = 1.0
        C_low = neutral_coefficient_10m(coeffs, U_low, U_min)
        # Cᴰ = (0.142 + 0.076*1 + 2.7/1) × 10⁻³ = 2.918 × 10⁻³
        @test C_low ≈ 2.918e-3 atol=1e-6

        # Test at high wind speed
        U_high = 20.0
        C_high = neutral_coefficient_10m(coeffs, U_high, U_min)
        # Cᴰ = (0.142 + 0.076*20 + 2.7/20) × 10⁻³
        #     = (0.142 + 1.52 + 0.135) × 10⁻³ = 1.797 × 10⁻³
        @test C_high ≈ 1.797e-3 atol=1e-6
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

    @testset "Li et al. (2010) Riᴮ → ζ mapping" begin
        α = log(1e4)  # z/ℓ = 10⁴
        β = log(7.3)  # ℓ/ℓh = 7.3
        mapping = RichardsonNumberMapping()

        # Neutral: Riᴮ = 0 gives ζ = 0
        @test bulk_to_flux_richardson_number(0.0, α, β, mapping) ≈ 0.0 atol=1e-10

        # Unstable: Riᴮ < 0 gives ζ < 0
        @test bulk_to_flux_richardson_number(-0.5, α, β, mapping) < 0
        @test bulk_to_flux_richardson_number(-1.0, α, β, mapping) < 0

        # Stable: Riᴮ > 0 gives ζ > 0
        @test bulk_to_flux_richardson_number(0.1, α, β, mapping) > 0
        @test bulk_to_flux_richardson_number(0.5, α, β, mapping) > 0

        # Monotonicity: more unstable → more negative ζ
        @test bulk_to_flux_richardson_number(-1.0, α, β, mapping) < bulk_to_flux_richardson_number(-0.5, α, β, mapping)

        # Monotonicity: more stable → more positive ζ
        @test bulk_to_flux_richardson_number(0.5, α, β, mapping) > bulk_to_flux_richardson_number(0.1, α, β, mapping)

        # Test with β = 0 (ℓ = ℓh)
        β0 = 0.0
        @test bulk_to_flux_richardson_number(0.0, α, β0, mapping) ≈ 0.0 atol=1e-10
        @test bulk_to_flux_richardson_number(-1.0, α, β0, mapping) < 0
        @test bulk_to_flux_richardson_number(0.5, α, β0, mapping) > 0

        # Test across regime boundaries
        ζ_weakly = bulk_to_flux_richardson_number(0.19, α, β, mapping)
        ζ_strongly = bulk_to_flux_richardson_number(0.21, α, β, mapping)
        @test ζ_weakly > 0
        @test ζ_strongly > 0
    end

    @testset "Integrated stability functions Ψᴰ and Ψᵀ" begin
        params = StabilityFunctionParameters()

        # At ζ = 0, both Ψᴰ and Ψᵀ should be 0 (continuity)
        @test integrated_stability_momentum(0.0, params) ≈ 0.0 atol=1e-10
        @test integrated_stability_scalar(0.0, params) ≈ 0.0 atol=1e-10

        # Unstable (ζ < 0): Ψ > 0
        @test integrated_stability_momentum(-1.0, params) > 0
        @test integrated_stability_scalar(-1.0, params) > 0

        # Stable (ζ > 0): Ψ < 0
        @test integrated_stability_momentum(1.0, params) < 0
        @test integrated_stability_scalar(1.0, params) < 0

        # Monotonicity: more unstable → larger positive Ψ
        @test integrated_stability_momentum(-2.0, params) > integrated_stability_momentum(-1.0, params)
        @test integrated_stability_scalar(-2.0, params) > integrated_stability_scalar(-1.0, params)

        # Monotonicity: more stable → more negative Ψ
        @test integrated_stability_momentum(2.0, params) < integrated_stability_momentum(1.0, params)
        @test integrated_stability_scalar(2.0, params) < integrated_stability_scalar(1.0, params)

        # Known reference values (computed from Hogström/Beljaars-Holtslag formulas)
        @test integrated_stability_momentum(-1.0, params) ≈ 1.215 atol=0.01
        @test integrated_stability_scalar(-1.0, params) ≈ 1.644 atol=0.01
        @test integrated_stability_momentum(1.0, params) ≈ -4.282 atol=0.01
        @test integrated_stability_scalar(1.0, params) ≈ -4.434 atol=0.01
    end

    @testset "Stability correction factors" begin
        α = 10.0
        β = 2.0

        # Neutral (Ψᴰ = Ψᵀ = 0): factor = 1
        @test stability_correction_factor(α, β, 0.0, 0.0, Val(:momentum)) ≈ 1.0
        @test stability_correction_factor(α, β, 0.0, 0.0, Val(:scalar)) ≈ 1.0

        # Unstable (positive Ψ): factor > 1
        @test stability_correction_factor(α, β, 1.0, 1.0, Val(:momentum)) > 1.0
        @test stability_correction_factor(α, β, 1.0, 1.0, Val(:scalar)) > 1.0

        # Stable (negative Ψ): factor < 1
        @test stability_correction_factor(α, β, -2.0, -2.0, Val(:momentum)) < 1.0
        @test stability_correction_factor(α, β, -2.0, -2.0, Val(:scalar)) < 1.0

        # Exact momentum correction: [α/(α - Ψᴰ)]²
        Ψᴰ = 1.5
        expected_momentum = (α / (α - Ψᴰ))^2
        @test stability_correction_factor(α, β, Ψᴰ, 0.0, Val(:momentum)) ≈ expected_momentum

        # Exact scalar correction: [α/(α - Ψᴰ)] × [βh/(βh - Ψᵀ)]
        Ψᵀ = 1.0
        βh = α + β
        expected_scalar = (α / (α - Ψᴰ)) * (βh / (βh - Ψᵀ))
        @test stability_correction_factor(α, β, Ψᴰ, Ψᵀ, Val(:scalar)) ≈ expected_scalar

        # Momentum and scalar corrections differ
        f_m = stability_correction_factor(α, β, 1.0, 1.0, Val(:momentum))
        f_h = stability_correction_factor(α, β, 1.0, 1.0, Val(:scalar))
        @test f_m != f_h

        # Nothing transfer_type defaults to momentum
        @test stability_correction_factor(α, β, Ψᴰ, 0.0, nothing) ≈ expected_momentum
    end

    @testset "FittedStabilityFunction callable interface" begin
        sf = FittedStabilityFunction(1.5e-4 / 7.3)
        α = log(1e4)
        β = log(7.3)

        # Neutral: correction factor ≈ 1
        @test sf(0.0, α, β) ≈ 1.0 atol=1e-10

        # Unstable: correction factor > 1
        @test sf(-0.1, α, β) > 1.0

        # Stable: correction factor < 1
        @test sf(0.1, α, β) < 1.0

        # Scalar correction differs from momentum
        f_m = sf(-0.1, α, β, Val(:momentum))
        f_s = sf(-0.1, α, β, Val(:scalar))
        @test f_m != f_s
    end

    @testset "Callable interface" begin
        # Create a simple grid with first cell center at 10m
        grid = RectilinearGrid(size=(1, 1, 1), x=(0, 100), y=(0, 100), z=(0, 20))

        # Test evaluation with no stability correction
        coef = PolynomialCoefficient(
            polynomial = (0.142, 0.076, 2.7),
            stability_function = nothing
        )
        U = 10.0
        T₀ = 290.0
        C = coef(1, 1, grid, U, T₀)
        @test C isa Number
        @test C > 0

        # Test with FittedStabilityFunction — need a VPT field
        θᵥ_field = CenterField(grid)
        set!(θᵥ_field, 288.0)  # cooler than surface → unstable

        coef_fitted = PolynomialCoefficient(
            (0.142, 0.076, 2.7),     # polynomial
            1.5e-4,                   # roughness_length
            0.1,                      # minimum_wind_speed
            FittedStabilityFunction(1.5e-4 / 7.3),
            Breeze.PlanarLiquidSurface(),
            θᵥ_field,
            1e5,
            Breeze.Thermodynamics.ThermodynamicConstants(),
            Val(:momentum)
        )
        C_fitted = coef_fitted(1, 1, grid, U, T₀)
        @test C_fitted isa Number
        @test C_fitted > 0
        # Unstable conditions should enhance transfer
        @test C_fitted > C
    end

    @testset "Integration with BulkDrag" begin
        # Test that PolynomialCoefficient works with BulkDrag constructor
        coef = PolynomialCoefficient()
        SST(x, y) = 300.0
        bc = Breeze.BulkDrag(coefficient = coef, gustiness = 0.5, surface_temperature = SST)
        @test bc isa BoundaryCondition
        # Coefficient should have been materialized with momentum coefficients
        @test bc.condition.coefficient.polynomial == (0.142, 0.076, 2.7)
        @test bc.condition.gustiness == 0.5
        @test bc.condition.surface_temperature === SST
        @test bc.condition.coefficient.transfer_type === Val(:momentum)
    end

    @testset "Integration with BulkSensibleHeatFlux" begin
        # Test that PolynomialCoefficient works with BulkSensibleHeatFlux
        coef = PolynomialCoefficient()
        SST(x, y) = 300.0
        bc = Breeze.BulkSensibleHeatFlux(coefficient = coef, surface_temperature = SST)
        @test bc isa BoundaryCondition
        # Coefficient should have been materialized with sensible heat coefficients
        @test bc.condition.coefficient.polynomial == (0.128, 0.068, 2.43)
        @test bc.condition.coefficient.transfer_type === Val(:scalar)
    end

    @testset "Integration with BulkVaporFlux" begin
        # Test that PolynomialCoefficient works with BulkVaporFlux
        coef = PolynomialCoefficient()
        SST(x, y) = 300.0
        bc = Breeze.BulkVaporFlux(coefficient = coef, surface_temperature = SST)
        @test bc isa BoundaryCondition
        # Coefficient should have been materialized with latent heat coefficients
        @test bc.condition.coefficient.polynomial == (0.120, 0.070, 2.55)
        @test bc.condition.coefficient.transfer_type === Val(:scalar)
    end
end
