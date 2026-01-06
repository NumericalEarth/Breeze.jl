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
    # Smoke test: verify Tetens formula computes saturation vapor pressure
    tetens = TetensFormula(FT)
    thermo = ThermodynamicConstants(FT; saturation_vapor_pressure=tetens)

    # Test at reference temperature (273 K): should return reference pressure
    T₀ = FT(273)
    pᵛ⁺_ref = saturation_vapor_pressure(T₀, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_ref ≈ FT(610) rtol=eps(FT)

    # Test at higher temperature: pressure should increase
    T_warm = FT(300)
    pᵛ⁺_warm = saturation_vapor_pressure(T_warm, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_warm > pᵛ⁺_ref

    # Test at lower temperature: pressure should decrease
    T_cold = FT(250)
    pᵛ⁺_cold = saturation_vapor_pressure(T_cold, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_cold < pᵛ⁺_ref

    # Verify the formula matches the expected analytic expression
    # pᵛ⁺(T) = p₀ * exp(a * (T - T₀) / (T - b))
    p₀ = FT(610)
    a = FT(17.27)
    T₀_param = FT(273)
    b = FT(36)
    T_test = FT(288)
    expected = p₀ * exp(a * (T_test - T₀_param) / (T_test - b))
    computed = saturation_vapor_pressure(T_test, thermo, PlanarLiquidSurface())
    @test computed ≈ expected rtol=eps(FT)
end
