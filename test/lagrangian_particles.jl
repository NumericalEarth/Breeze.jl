include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: adiabatic_balance_twin
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: rnode, xnode, ynode, znode
using Oceananigans.TimeSteppers: step_lagrangian_particles!
using Test

function particles_at(FT, x, y, z; restitution = 1)
    x = on_architecture(default_arch, FT[x])
    y = on_architecture(default_arch, FT[y])
    z = on_architecture(default_arch, FT[z])
    return LagrangianParticles(; x, y, z, restitution=FT(restitution))
end

uniform_flow_particles(FT) = particles_at(FT, 100, 200, 1000)

particle_position(particles) = @allowscalar (particles.properties.x[1],
                                             particles.properties.y[1],
                                             particles.properties.z[1])

# Terrain-following grids are only supported by the compressible dynamics (the
# anelastic pressure solver assumes horizontally uniform Δz), so every
# terrain test below builds a compressible model.
terrain_dynamics(FT) = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                            reference_potential_temperature=FT(300))

function terrain_grid(FT, formulation; grid_size, x_extent, y_extent = nothing, z_top, topography)
    Nz = last(grid_size)
    r_faces = collect(range(FT(0), z_top, length=Nz+1))
    z_coordinate = TerrainFollowingVerticalDiscretization(r_faces; formulation)

    # Oceananigans requires halo ≤ size in every non-Flat direction.
    halo = map(N -> min(5, N), grid_size)

    grid = if isnothing(y_extent)
        RectilinearGrid(default_arch; size=grid_size, halo,
                        x=x_extent, z=z_coordinate, topology=(Periodic, Flat, Bounded))
    else
        RectilinearGrid(default_arch; size=grid_size, halo,
                        x=x_extent, y=y_extent, z=z_coordinate,
                        topology=(Periodic, Periodic, Bounded))
    end

    materialize_terrain!(grid, topography)
    return grid
end

@testset "Lagrangian particles in AtmosphereModel [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 4000), y=(0, 4000), z=(0, 4000))

    @testset "Construction and show" begin
        particles = uniform_flow_particles(FT)
        model = AtmosphereModel(grid; particles)
        @test model.particles === particles
        @test occursin("particles: 1 LagrangianParticles", sprint(show, model))

        twin = adiabatic_balance_twin(model)
        @test isnothing(twin.particles)
        @test model.particles === particles

        # Without particles, microphysics closes the summary tree
        model = AtmosphereModel(grid)
        @test isnothing(model.particles)
        summary_lines = split(sprint(show, model), '\n')
        @test startswith(last(summary_lines), "└── microphysics: ")
        @test !any(line -> occursin("particles", line), summary_lines)
    end

    @testset "Terrain-following physical coordinates" begin
        Nx, Ny, Nz = 16, 4, 8
        Lx, Ly, Lz = FT(4000), FT(4000), FT(2000)

        h₀ = FT(300)
        hill_width = FT(800)
        topography(x, y) = h₀ * exp(-(x^2 + y^2) / hill_width^2)

        tf_grid = terrain_grid(FT, LinearDecay(); grid_size=(Nx, Ny, Nz), x_extent=(-Lx/2, Lx/2),
                               y_extent=(-Ly/2, Ly/2), z_top=Lz, topography)
        dynamics = terrain_dynamics(FT)

        # u = z / 100 is linear in physical height. A particle at z = 1000
        # must therefore move 10 m in one second even though its reference
        # coordinate r is lower over the mountain.
        x₀ = FT(0)
        y₀ = FT(0)
        z₀ = FT(1000)
        particles = particles_at(FT, x₀, y₀, z₀)
        model = AtmosphereModel(tf_grid; dynamics, particles)
        set!(model.velocities.u, (x, y, z) -> z / FT(100))
        step_lagrangian_particles!(model, FT(1))

        x, y, z = particle_position(particles)
        @test x ≈ x₀ + z₀ / FT(100)
        @test y ≈ y₀
        @test z ≈ z₀

        # The lower particle wall follows the local terrain rather than r = 0.
        i = Nx ÷ 2
        j = Ny ÷ 2
        x₀ = @allowscalar xnode(i, tf_grid, Center())
        y₀ = @allowscalar ynode(j, tf_grid, Center())
        z_surface = topography(x₀, y₀)
        particles = particles_at(FT, x₀, y₀, z_surface + FT(10); restitution=0)
        model = AtmosphereModel(tf_grid; dynamics, particles)
        set!(model.velocities.w, -FT(20))
        step_lagrangian_particles!(model, FT(1))

        x, y, z = particle_position(particles)
        @test x ≈ x₀
        @test y ≈ y₀
        @test z ≈ z_surface

        # The upper wall is the flat lid z = r_top, not h + z_top.
        particles = particles_at(FT, x₀, y₀, Lz - FT(10); restitution=0)
        model = AtmosphereModel(tf_grid; dynamics, particles)
        set!(model.velocities.w, FT(20))
        step_lagrangian_particles!(model, FT(100))
        @test last(particle_position(particles)) ≈ Lz

        # Horizontal wrapping happens in physical coordinates, and the terrain
        # wall is then evaluated over the far side of the domain.
        x_start = Lx/2 - FT(100)
        particles = particles_at(FT, x_start, y₀, FT(1000))
        model = AtmosphereModel(tf_grid; dynamics, particles)
        set!(model.velocities.u, FT(30))
        # A particle this close to the edge interpolates across the last x-face,
        # whose halo `set!` does not touch. `compute_velocities!` fills these on
        # every `update_state!`, so do the same here.
        fill_halo_regions!(model.velocities.u)
        step_lagrangian_particles!(model, FT(10))
        x, _, z = particle_position(particles)
        @test x ≈ x_start + 300 - Lx
        @test z ≈ FT(1000)
    end

    @testset "Two-level terrain coordinate inversion" begin
        Nx, Nz = 12, 8
        Lx, Lz = FT(4000), FT(2000)
        formulation() = TwoLevelDecay(large_scale_height = FT(1400),
                                      small_scale_height = FT(400))
        dynamics = terrain_dynamics(FT)

        # Constant terrain: the SLEVE split puts everything in the large-scale
        # component, so this isolates the b₁ branch of the inversion.
        flat_terrain = terrain_grid(FT, formulation(); grid_size=(Nx, Nz), x_extent=(-Lx/2, Lx/2),
                                    z_top=Lz, topography = x -> FT(300))

        i = Nx ÷ 2 + 1
        k = Nz ÷ 2
        x₀ = @allowscalar xnode(i, flat_terrain, Face())
        r₀ = @allowscalar rnode(k, flat_terrain, Center())
        z₀ = @allowscalar znode(i, 1, k, flat_terrain, Face(), Center(), Center())
        @test z₀ > r₀

        particles = particles_at(FT, x₀, 0, z₀)
        model = AtmosphereModel(flat_terrain; dynamics, particles)
        set!(model.velocities.u, (x, z) -> z / FT(100))
        step_lagrangian_particles!(model, FT(1))

        x, _, z = particle_position(particles)
        @test x ≈ x₀ + z₀ / FT(100)
        @test z ≈ z₀

        # A narrow hill leaves a large small-scale residual h₂, so both SLEVE
        # bases contribute to the Newton residual and its derivative.
        hill(x) = FT(400) * exp(-x^2 / FT(400)^2)
        hilly = terrain_grid(FT, formulation(); grid_size=(Nx, Nz), x_extent=(-Lx/2, Lx/2),
                             z_top=Lz, topography=hill)
        h₂ = hilly.z.formulation.h₂
        @test maximum(abs, parent(h₂)) > 100   # h₂ genuinely nonzero

        i = Nx ÷ 2 + 1
        for k in (1, Nz ÷ 2, Nz)
            x₀ = @allowscalar xnode(i, hilly, Face())
            z₀ = @allowscalar znode(i, 1, k, hilly, Face(), Center(), Center())
            particles = particles_at(FT, x₀, 0, z₀)
            model = AtmosphereModel(hilly; dynamics, particles)
            set!(model.velocities.u, (x, z) -> z / FT(100))
            step_lagrangian_particles!(model, FT(1))

            # Exact only because (x₀, z₀) is a u node: recovering u = z₀/100 means
            # the inversion returned rnode(k) rather than treating z as r.
            x, _, z = particle_position(particles)
            @test x ≈ x₀ + z₀ / FT(100)
            @test z ≈ z₀
        end
    end

    @testset "Terrain-following advection inside time_step!" begin
        Nx, Nz = 12, 8
        Lx, Lz = FT(4000), FT(2000)
        U = FT(5)
        Δt = FT(0.5)
        steps = 4

        # Uniform terrain: the coordinate surfaces are flat but offset from r, so
        # z ≠ r while the dynamics stay at rest under a uniform wind. The particle
        # displacement is therefore exactly U Δt per step.
        h₀ = FT(300)
        tf_grid = terrain_grid(FT, LinearDecay(); grid_size=(Nx, Nz), x_extent=(-Lx/2, Lx/2),
                               z_top=Lz, topography = x -> h₀)

        z₀ = FT(1200)
        particles = particles_at(FT, FT(0), 0, z₀)
        model = AtmosphereModel(tf_grid; dynamics=terrain_dynamics(FT), particles)
        set!(model, θ=300, ρ=model.dynamics.reference_state.density, u=U, w=0)

        for _ in 1:steps
            time_step!(model, Δt)
        end

        x, _, z = particle_position(particles)
        @test x ≈ U * Δt * steps rtol=1e-5
        @test z ≈ z₀ rtol=1e-6

        # Over a hill the flow is no longer trivial, but the particle must stay
        # within the domain and above the local terrain surface.
        hill(x) = FT(200) * exp(-x^2 / FT(800)^2)
        hilly = terrain_grid(FT, LinearDecay(); grid_size=(Nx, Nz), x_extent=(-Lx/2, Lx/2),
                             z_top=Lz, topography=hill)
        particles = particles_at(FT, -Lx/4, 0, z₀)
        model = AtmosphereModel(hilly; dynamics=terrain_dynamics(FT), particles)
        set!(model, θ=300, ρ=model.dynamics.reference_state.density, u=U, w=0)

        for _ in 1:steps
            time_step!(model, Δt)
        end

        x, _, z = particle_position(particles)
        @test isfinite(x) && isfinite(z)
        @test -Lx/2 ≤ x ≤ Lx/2
        @test hill(x) < z < Lz
        @test x > -Lx/4   # advected downstream
    end

    @testset "Particles are rejected on an immersed terrain-following grid" begin
        Nx, Nz = 8, 6
        Lx, Lz = FT(4000), FT(2000)
        tf_grid = terrain_grid(FT, LinearDecay(); grid_size=(Nx, Nz), x_extent=(-Lx/2, Lx/2),
                               z_top=Lz, topography = x -> FT(100))
        immersed = ImmersedBoundaryGrid(tf_grid, GridFittedBottom((x) -> FT(400)))
        particles = particles_at(FT, FT(0), 0, FT(1000))
        @test_throws ArgumentError AtmosphereModel(immersed; dynamics=terrain_dynamics(FT), particles)
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

        # A uniform wind over flat ground is an exact steady state, so the
        # trajectory is exact to roundoff rather than merely close.
        x, y, z = particle_position(particles)
        @test x ≈ 100 + 10 * Δt * 5
        @test y == 200
        @test isapprox(z, 1000, atol=100 * eps(FT(1000)))
    end
end
