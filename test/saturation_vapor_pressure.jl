using Breeze
using Test

using Breeze.Thermodynamics:
    ThermodynamicConstants,
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
