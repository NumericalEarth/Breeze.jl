using Breeze
using Test

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    TetensFormula,
    saturation_vapor_pressure,
    PlanarLiquidSurface,
    PlanarIceSurface,
    PlanarMixedPhaseSurface,
    absolute_zero_latent_heat,
    specific_heat_difference,
    vapor_gas_constant

function reference_mixed_surface_pressure(T, thermo, λ)
    ℒˡ₀ = absolute_zero_latent_heat(thermo, thermo.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(thermo, thermo.ice)
    Δcˡ = specific_heat_difference(thermo, thermo.liquid)
    Δcⁱ = specific_heat_difference(thermo, thermo.ice)

    ℒ₀ = λ * ℒˡ₀ + (one(λ) - λ) * ℒⁱ₀
    Δcᵝ = λ * Δcˡ + (one(λ) - λ) * Δcⁱ

    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    Rᵛ = vapor_gas_constant(thermo)

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵝ / Rᵛ) * exp((one(T) / Tᵗʳ - one(T) / T) * ℒ₀ / Rᵛ)
end

@testset "Saturation vapor pressure surfaces [$FT]" for FT in (Float32, Float64)
    thermo = ThermodynamicConstants(FT)
    Tᵗʳ = thermo.triple_point_temperature
    temperatures = FT.((Tᵗʳ * FT(0.9), Tᵗʳ, Tᵗʳ * FT(1.1)))

    liquid_surface = PlanarLiquidSurface()
    ice_surface = PlanarIceSurface()
    rtol = FT === Float64 ? 1e-12 : FT(1e-5)

    @testset "Planar homogeneous surfaces" begin
        for T in temperatures
            pˡ = saturation_vapor_pressure(T, thermo, thermo.liquid)
            pⁱ = saturation_vapor_pressure(T, thermo, thermo.ice)

            @test saturation_vapor_pressure(T, thermo, liquid_surface) ≈ pˡ rtol=rtol
            @test saturation_vapor_pressure(T, thermo, ice_surface) ≈ pⁱ rtol=rtol
        end
    end

    @testset "Planar mixed-phase surfaces" begin
        for λ in (zero(FT), FT(0.25), FT(0.5), FT(0.75), one(FT))
            surface = PlanarMixedPhaseSurface(λ)
            for T in temperatures
                p_surface = saturation_vapor_pressure(T, thermo, surface)
                p_reference = reference_mixed_surface_pressure(T, thermo, λ)

                @test p_surface ≈ p_reference rtol=rtol
            end
        end
    end
end

@testset "Tetens formula saturation vapor pressure [$FT]" for FT in (Float32, Float64)
    tetens = TetensFormula()
    thermo = ThermodynamicConstants(; saturation_vapor_pressure=tetens)

    # Test at reference temperature (273.15 K): should return reference pressure
    Tᵣ = FT(273.15)
    pᵛ⁺_ref = saturation_vapor_pressure(Tᵣ, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_ref ≈ FT(610) rtol=eps(FT)

    # Test monotonicity: pressure increases with temperature (liquid)
    T_warm = FT(300)
    T_cold = FT(250)
    pᵛ⁺_warm = saturation_vapor_pressure(T_warm, thermo, PlanarLiquidSurface())
    pᵛ⁺_cold = saturation_vapor_pressure(T_cold, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_warm > pᵛ⁺_ref > pᵛ⁺_cold

    # Test ice surface at reference temperature
    pⁱ_ref = saturation_vapor_pressure(Tᵣ, thermo, PlanarIceSurface())
    @test pⁱ_ref ≈ FT(610) rtol=eps(FT)

    # Test monotonicity for ice
    pⁱ_warm = saturation_vapor_pressure(T_warm, thermo, PlanarIceSurface())
    pⁱ_cold = saturation_vapor_pressure(T_cold, thermo, PlanarIceSurface())
    @test pⁱ_warm > pⁱ_ref > pⁱ_cold

    # Verify analytic expressions for liquid
    pᵣ = FT(610)
    aˡ = FT(17.27)
    δTˡ = FT(35.85)
    T_test = FT(288)
    expected_liquid = pᵣ * exp(aˡ * (T_test - Tᵣ) / (T_test - δTˡ))
    @test saturation_vapor_pressure(T_test, thermo, PlanarLiquidSurface()) ≈ expected_liquid rtol=eps(FT)

    # Verify analytic expressions for ice
    aⁱ = FT(21.875)
    δTⁱ = FT(7.65)
    expected_ice = pᵣ * exp(aⁱ * (T_test - Tᵣ) / (T_test - δTⁱ))
    @test saturation_vapor_pressure(T_test, thermo, PlanarIceSurface()) ≈ expected_ice rtol=eps(FT)

    # Test mixed-phase surface: linear interpolation between liquid and ice
    for λ in (FT(0), FT(0.5), FT(1))
        surface = PlanarMixedPhaseSurface(λ)
        pˡ = saturation_vapor_pressure(T_test, thermo, PlanarLiquidSurface())
        pⁱ = saturation_vapor_pressure(T_test, thermo, PlanarIceSurface())
        expected_mixed = λ * pˡ + (1 - λ) * pⁱ
        @test saturation_vapor_pressure(T_test, thermo, surface) ≈ expected_mixed rtol=eps(FT)
    end
end

@testset "Tetens vs Clausius-Clapeyron comparison [$FT]" for FT in (Float32, Float64)
    tetens = TetensFormula(FT)
    thermo_tetens = ThermodynamicConstants(FT; saturation_vapor_pressure=tetens)
    thermo_cc = ThermodynamicConstants(FT) # Default is Clausius-Clapeyron

    # Both formulas should agree reasonably well in the typical atmospheric range
    # Tetens is empirical; CC is thermodynamically derived but uses constant Δc approximation
    temperatures = FT.((260, 273.15, 285, 300))

    for T in temperatures
        pˡ_tetens = saturation_vapor_pressure(T, thermo_tetens, PlanarLiquidSurface())
        pˡ_cc = saturation_vapor_pressure(T, thermo_cc, PlanarLiquidSurface())
        # Expect agreement within ~5% for atmospheric conditions
        @test pˡ_tetens ≈ pˡ_cc rtol=FT(0.05)

        pⁱ_tetens = saturation_vapor_pressure(T, thermo_tetens, PlanarIceSurface())
        pⁱ_cc = saturation_vapor_pressure(T, thermo_cc, PlanarIceSurface())
        @test pⁱ_tetens ≈ pⁱ_cc rtol=FT(0.05)
    end
end
