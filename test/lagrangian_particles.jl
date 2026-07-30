using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Test

function uniform_flow_particles(FT)
    x = on_architecture(default_arch, FT[100])
    y = on_architecture(default_arch, FT[200])
    z = on_architecture(default_arch, FT[1000])
    return LagrangianParticles(; x, y, z)
end

particle_position(particles) = @allowscalar (particles.properties.x[1],
                                             particles.properties.y[1],
                                             particles.properties.z[1])

@testset "Lagrangian particles in AtmosphereModel [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 4000), y=(0, 4000), z=(0, 4000))

    @testset "Construction and show" begin
        particles = uniform_flow_particles(FT)
        model = AtmosphereModel(grid; particles)
        @test model.particles === particles
        @test occursin("particles: 1 LagrangianParticles", sprint(show, model))

        # Without particles the show method is unchanged
        model = AtmosphereModel(grid)
        @test isnothing(model.particles)
        @test !occursin("particles", sprint(show, model))
    end

    @testset "Advection by uniform flow (anelastic, SSP RK3)" begin
        particles = uniform_flow_particles(FT)
        model = AtmosphereModel(grid; particles)
        set!(model, u=10)

        Δt = 10
        for _ in 1:5
            time_step!(model, Δt)
        end

        # Particles are advected once per step with the full Δt
        x, y, z = particle_position(particles)
        @test x ≈ 100 + 10 * Δt * 5
        @test y == 200
        @test z == 1000
    end

    @testset "Advection by uniform flow (compressible, acoustic RK3)" begin
        particles = uniform_flow_particles(FT)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; dynamics, particles)
        ref = model.dynamics.reference_state
        set!(model; θ=300, u=10, qᵗ=0, ρ=ref.density)

        Δt = 1
        for _ in 1:5
            time_step!(model, Δt)
        end

        x, y, z = particle_position(particles)
        @test isapprox(x, 100 + 10 * Δt * 5, rtol=1e-3)
        @test y == 200
        @test isapprox(z, 1000, atol=1)
    end
end
