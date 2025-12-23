using Breeze
using Breeze: ReferenceState, AnelasticDynamics
using Breeze: StaticEnergyFormulation, LiquidIcePotentialTemperatureFormulation
using Breeze.AtmosphereModels: materialize_formulation, materialize_dynamics
using Breeze.AtmosphereModels: prognostic_thermodynamic_field_names
using Breeze.AtmosphereModels: additional_thermodynamic_field_names
using Breeze.AtmosphereModels: thermodynamic_density_name, thermodynamic_density
using Oceananigans
using Test

@testset "ThermodynamicFormulations [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)
    dynamics_stub = AnelasticDynamics(reference_state)
    dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple())

    # Boundary conditions needed for materialization
    boundary_conditions = (ρθ = FieldBoundaryConditions(), ρe = FieldBoundaryConditions())

    @testset "LiquidIcePotentialTemperature field naming (Symbol)" begin
        @test prognostic_thermodynamic_field_names(:LiquidIcePotentialTemperature) == (:ρθ,)
        @test additional_thermodynamic_field_names(:LiquidIcePotentialTemperature) == (:θ,)
        @test thermodynamic_density_name(:LiquidIcePotentialTemperature) == :ρθ
    end

    @testset "StaticEnergy field naming (Symbol)" begin
        @test prognostic_thermodynamic_field_names(:StaticEnergy) == (:ρe,)
        @test additional_thermodynamic_field_names(:StaticEnergy) == (:e,)
        @test thermodynamic_density_name(:StaticEnergy) == :ρe
    end

    @testset "materialize_formulation(:LiquidIcePotentialTemperature)" begin
        formulation = materialize_formulation(:LiquidIcePotentialTemperature, dynamics, grid, boundary_conditions)

        @test formulation isa LiquidIcePotentialTemperatureFormulation
        @test formulation.potential_temperature_density isa Field
        @test formulation.potential_temperature isa Field

        # Test struct methods
        @test prognostic_thermodynamic_field_names(formulation) == (:ρθ,)
        @test additional_thermodynamic_field_names(formulation) == (:θ,)
        @test thermodynamic_density_name(formulation) == :ρθ
        @test thermodynamic_density(formulation) === formulation.potential_temperature_density
    end

    @testset "materialize_formulation(:StaticEnergy)" begin
        formulation = materialize_formulation(:StaticEnergy, dynamics, grid, boundary_conditions)

        @test formulation isa StaticEnergyFormulation
        @test formulation.energy_density isa Field
        @test formulation.specific_energy isa Field

        # Test struct methods
        @test prognostic_thermodynamic_field_names(formulation) == (:ρe,)
        @test additional_thermodynamic_field_names(formulation) == (:e,)
        @test thermodynamic_density_name(formulation) == :ρe
        @test thermodynamic_density(formulation) === formulation.energy_density
    end

    @testset "Oceananigans.fields and prognostic_fields" begin
        θ_formulation = materialize_formulation(:LiquidIcePotentialTemperature, dynamics, grid, boundary_conditions)
        e_formulation = materialize_formulation(:StaticEnergy, dynamics, grid, boundary_conditions)

        # LiquidIcePotentialTemperature
        @test haskey(Oceananigans.fields(θ_formulation), :θ)
        @test haskey(Oceananigans.prognostic_fields(θ_formulation), :ρθ)

        # StaticEnergy
        @test haskey(Oceananigans.fields(e_formulation), :e)
        @test haskey(Oceananigans.prognostic_fields(e_formulation), :ρe)
    end
end

