using Breeze
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    StaticEnergyState,
    temperature,
    mixture_heat_capacity

@testset "Thermodynamics" begin
    thermo = ThermodynamicConstants()

    # Test Saturation specific humidity calculation
    T = 293.15  # 20°C
    ρ = 1.2     # kg/m³
    q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
    @test q★ > 0
end

@testset "StaticEnergyState [$FT]" for FT in (Float32, Float64)
    T = FT(253.15)
    p = FT(101325)
    z = FT(1000)
    thermo = ThermodynamicConstants(FT)

    for qᵛ in 5e-3:5e-3:3e-2, qˡ in 0:5e-5:3e-4, qⁱ in 0:5e-5:3e-4
        qᵛ = convert(FT, qᵛ)
        qˡ = convert(FT, qˡ)
        qⁱ = convert(FT, qⁱ)
        q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
        cᵖᵐ = mixture_heat_capacity(q, thermo)
        g = thermo.gravitational_acceleration
        ℒˡᵣ = thermo.liquid.reference_latent_heat
        ℒⁱᵣ = thermo.ice.reference_latent_heat
        e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

        # Test with saturation adjustment
        𝒰 = StaticEnergyState(e, q, z, p)
        T★ = temperature(𝒰, thermo)
        @test T★ ≈ T
    end
end
