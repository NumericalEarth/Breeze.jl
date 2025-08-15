using Test
using Breeze

@testset "Breeze.jl" begin
    @testset "Thermodynamics" begin
        thermo = AtmosphereThermodynamics()

        # Test saturation specific humidity calculation
        T = 293.15  # 20°C
        ρ = 1.2     # kg/m³
        q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.condensation)

        # Basic sanity checks
        @test q★ > 0
        @test q★ < 1  # specific humidity should be less than 1
    end
end

