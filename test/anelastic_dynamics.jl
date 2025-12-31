using Breeze
using Breeze: ReferenceState, AnelasticDynamics
using Breeze.AtmosphereModels: materialize_dynamics, default_dynamics
using Breeze.AtmosphereModels: mean_pressure, pressure_anomaly, total_pressure
using Breeze.AtmosphereModels: dynamics_density, dynamics_pressure
using Oceananigans
using Test

@testset "AnelasticDynamics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()

    @testset "Constructor with ReferenceState" begin
        reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        dynamics = AnelasticDynamics(reference_state)

        @test dynamics isa AnelasticDynamics
        @test dynamics.reference_state === reference_state
        @test dynamics.pressure_anomaly === nothing  # Not materialized yet
    end

    @testset "default_dynamics" begin
        dynamics = default_dynamics(grid, constants)

        @test dynamics isa AnelasticDynamics
        @test dynamics.reference_state isa ReferenceState
        @test dynamics.pressure_anomaly === nothing
    end

    @testset "materialize_dynamics" begin
        reference_state = ReferenceState(grid, constants)
        dynamics_stub = AnelasticDynamics(reference_state)
        boundary_conditions = NamedTuple()
        
        dynamics = materialize_dynamics(dynamics_stub, grid, boundary_conditions)

        @test dynamics isa AnelasticDynamics
        @test dynamics.reference_state === reference_state
        @test dynamics.pressure_anomaly isa Field  # Now materialized
    end

    @testset "Pressure utilities" begin
        reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        dynamics_stub = AnelasticDynamics(reference_state)
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple())

        # Test mean_pressure
        p̄ = mean_pressure(dynamics)
        @test p̄ === reference_state.pressure

        # Test pressure_anomaly (returns an AbstractOperation)
        p′ = pressure_anomaly(dynamics)
        @test p′ isa Oceananigans.AbstractOperations.AbstractOperation

        # Test total_pressure (returns an AbstractOperation)
        p = total_pressure(dynamics)
        @test p isa Oceananigans.AbstractOperations.AbstractOperation
    end
end

