using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: materialize_dynamics, dynamics_density, dynamics_pressure
using Oceananigans
using Test

@testset "CompressibleDynamics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()

    @testset "Constructor" begin
        dynamics = CompressibleDynamics(; constants)
        @test dynamics isa CompressibleDynamics
        @test dynamics.density === nothing  # Not materialized yet
    end

    @testset "materialize_dynamics" begin
        dynamics_stub = CompressibleDynamics(; constants)
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple())

        @test dynamics isa CompressibleDynamics
        @test dynamics.density isa Field
        @test dynamics.pressure isa Field
        @test dynamics_density(dynamics) === dynamics.density
        @test dynamics_pressure(dynamics) === dynamics.pressure
    end
end

