using Breeze
using Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization, rnode, xnode
using Test

@testset "TerrainFollowingDiscretization" begin
    @testset "follow_terrain! with function topography" begin
        Nx, Nz = 32, 10
        Lx, Lz = 100000.0, 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 500.0
        a = 2000.0
        h(x, y) = h₀ * exp(-x^2 / a^2)

        metrics = follow_terrain!(grid, h)

        # Check that metrics are returned
        @test metrics isa TerrainMetrics

        # Check sigma: at any column, σ = (Lz - h) / Lz
        for i in 1:Nx
            x = xnode(i, grid, Center())
            h_expected = h₀ * exp(-x^2 / a^2)
            σ_expected = (Lz - h_expected) / Lz
            @test grid.z.σᶜᶜⁿ[i, 1, 1] ≈ σ_expected rtol=1e-10
        end

        # Check eta: η = h at all columns
        for i in 1:Nx
            x = xnode(i, grid, Center())
            h_expected = h₀ * exp(-x^2 / a^2)
            @test grid.z.ηⁿ[i, 1, 1] ≈ h_expected rtol=1e-10
        end

        # Check that σᶜᶜ⁻ was also set
        @test parent(grid.z.σᶜᶜ⁻) == parent(grid.z.σᶜᶜⁿ)

        # Check z_top
        @test metrics.z_top ≈ Lz
    end

    @testset "follow_terrain! terrain slopes" begin
        Nx, Nz = 64, 10
        Lx, Lz = 100000.0, 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        # Gaussian mountain: h(x) = h₀ * exp(-x² / a²), so ∂h/∂x = -2x/a² * h
        h₀ = 500.0
        a = 10000.0
        h(x, y) = h₀ * exp(-x^2 / a^2)

        metrics = follow_terrain!(grid, h)

        Δx = Lx / Nx

        # Check the slope ∂h/∂x via centered finite differences
        # Only test the interior where the Gaussian is well-resolved
        # (avoid near periodic boundary where the wrap-around contaminates)
        for i in Nx÷4:3Nx÷4
            x = xnode(i, grid, Face())
            analytical_slope = -2 * x / a^2 * h₀ * exp(-x^2 / a^2)
            @test metrics.∂x_h[i, 1, 1] ≈ analytical_slope rtol=0.05 atol=1e-4
        end
    end

    @testset "Terrain-following grid with CompressibleDynamics" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x, y) = 200 * exp(-x^2 / 2000^2)
        metrics = follow_terrain!(grid, h)

        # Check that we can construct a compressible model on this grid
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(ExplicitTimeStepping()))

        @test model isa AtmosphereModel

        # Check that we can set initial conditions and take a time step
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        constants = model.thermodynamic_constants
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)
        set!(model, ρ=ρᵢ, θ=θ₀)

        Δt = 0.1
        time_step!(model, Δt)

        # Model should not blow up
        @test isfinite(maximum(abs, model.velocities.w))
    end
end
