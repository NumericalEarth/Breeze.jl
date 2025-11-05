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
