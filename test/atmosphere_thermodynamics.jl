using Breeze
using Oceananigans
using Test

@testset "Thermodynamics" begin
    thermo = ThermodynamicConstants()

    # Test Saturation specific humidity calculation
    T = 293.15  # 20°C
    ρ = 1.2     # kg/m³
    q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
    @test q★ > 0
end

@testset "MoistStaticEnergyState [$FT]" for FT in (Float32, Float64) begin
    T = FT(253.15)
    p = FT(101325)
    z = FT(1000)
    thermo = ThermodynamicConstants(FT)

    for qᵛ in 5e-3:5e-3:3e-2, qˡ in 0:5e-5:3e-4, qˡ in 0:5e-5:3e-4
        qᵛ = convert(FT, qᵛ)
        qˡ = convert(FT, qˡ)
        qⁱ = convert(FT, qⁱ)
        q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
        cᵖᵐ = mixture_heat_capacity(q, thermo)
        g = thermo.gravitational_acceleration
        e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

        # Test with saturation adjustment
        𝒰 = MoistStaticEnergyState(e, q, z, p)
        T★ = temperature(𝒰, thermo)
        @test T★ ≈ T
    end
end
