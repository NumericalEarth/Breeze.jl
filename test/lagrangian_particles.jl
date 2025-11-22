using Breeze
using Oceananigans.Architectures: on_architecture, architecture
using Test

@testset "Lagrangian particles with AtmosphereModel [$(FT)]" for FT in (Float32, Float64)
    grid = RectilinearGrid(default_arch, FT; size=(8, 8, 8), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))
    thermo = ThermodynamicConstants(FT)

    arch = architecture(grid)
    P = 10

    # Test creating model with particles
    x₀ = on_architecture(arch, FT(500) * ones(P))
    y₀ = on_architecture(arch, FT(500) * ones(P))
    z₀ = on_architecture(arch, FT(500) * ones(P))

    lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀)
    @test lagrangian_particles isa LagrangianParticles
    @test length(lagrangian_particles) == P

    reference_state = ReferenceState(grid, thermo)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamics=thermo, formulation, particles=lagrangian_particles)

    # Test that particles are stored correctly
    @test model.lagrangian_particles isa LagrangianParticles
    @test length(model.lagrangian_particles) == P
    @test propertynames(model.lagrangian_particles.properties) == (:x, :y, :z)

    # Test that step_lagrangian_particles! can be called
    Δt = FT(0.1)
    step_lagrangian_particles!(model, Δt)

    # Test that particles still exist after stepping
    @test length(model.lagrangian_particles) == P
    @test propertynames(model.lagrangian_particles.properties) == (:x, :y, :z)

    # Test creating model without particles (default)
    model_no_particles = AtmosphereModel(grid; thermodynamics=thermo, formulation)
    @test isnothing(model_no_particles.lagrangian_particles)

    # Test that step_lagrangian_particles! works with nothing
    step_lagrangian_particles!(model_no_particles, Δt)
    @test isnothing(model_no_particles.lagrangian_particles)
end

