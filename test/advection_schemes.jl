using Breeze
using Oceananigans
using Test

include("test_utils.jl")

@testset "Advection scheme configuration [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)

    @testset "Default advection schemes" begin
        static_energy_model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics,
                                              formulation=:StaticEnergy)
        potential_temperature_model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics, 
                                                      formulation=:LiquidIcePotentialTemperature)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa Centered
            @test model.advection.ρqᵗ isa Centered
            time_step!(model, 1)
        end
    end

    @testset "Unified advection parameter" begin
        static_energy_model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                              dynamics, formulation=:StaticEnergy, advection=WENO())

        potential_temperature_model= AtmosphereModel(grid; thermodynamic_constants=constants,
                                                     dynamics, formulation=:LiquidIcePotentialTemperature, advection=WENO())

        @test static_energy_model.advection.ρe isa WENO
        @test potential_temperature_model.advection.ρθ isa WENO

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa WENO
            time_step!(model, 1)
        end
    end

    @testset "Separate momentum and tracer advection" begin
        kw = (thermodynamic_constants=constants, momentum_advection = WENO(), scalar_advection = Centered(order=2))
        static_energy_model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, kw...)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa Centered
            time_step!(model, 1)
        end
    end

    @testset "FluxFormAdvection for momentum and tracers" begin
        advection = FluxFormAdvection(WENO(), WENO(), Centered(order=2))
        kw = (; thermodynamic_constants=constants, advection)
        static_energy_model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, kw...)

        @test static_energy_model.advection.ρe isa FluxFormAdvection
        @test static_energy_model.advection.ρe.x isa WENO
        @test static_energy_model.advection.ρe.y isa WENO
        @test static_energy_model.advection.ρe.z isa Centered

        @test potential_temperature_model.advection.ρθ isa FluxFormAdvection
        @test potential_temperature_model.advection.ρθ.x isa WENO
        @test potential_temperature_model.advection.ρθ.y isa WENO
        @test potential_temperature_model.advection.ρθ.z isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa FluxFormAdvection
            @test model.advection.ρqᵗ isa FluxFormAdvection
            @test model.advection.ρqᵗ.x isa WENO
            @test model.advection.ρqᵗ.y isa WENO
            @test model.advection.ρqᵗ.z isa Centered
            time_step!(model, 1)
        end
    end

    @testset "Tracer advection with user tracers" begin
        kw = (thermodynamic_constants=constants, tracers = :c, scalar_advection = UpwindBiased(order=1))
        static_energy_model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, kw...)

        @test static_energy_model.advection.ρe isa UpwindBiased
        @test potential_temperature_model.advection.ρθ isa UpwindBiased

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa Centered
            @test model.advection.ρqᵗ isa UpwindBiased
            @test model.advection.c isa UpwindBiased
            time_step!(model, 1)
        end
    end

    @testset "Mixed configuration with tracers" begin
        scalar_advection = (; c=Centered(order=2), ρqᵗ=WENO())
        kw = (thermodynamic_constants=constants, tracers = :c, momentum_advection = WENO(), scalar_advection)
        static_energy_model = AtmosphereModel(grid; dynamics, formulation=:StaticEnergy, kw...)
        potential_temperature_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature, kw...)

        @test static_energy_model.advection.ρe isa Centered
        @test potential_temperature_model.advection.ρθ isa Centered

        for model in (static_energy_model, potential_temperature_model)
            @test model.advection.momentum isa WENO
            @test model.advection.ρqᵗ isa WENO
            @test model.advection.c isa Centered
            time_step!(model, 1)
        end
    end
end

