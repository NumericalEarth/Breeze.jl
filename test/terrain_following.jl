using Breeze
using Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization, rnode, xnode, ynode, znode, λnode, φnode
using Breeze.Thermodynamics: hydrostatic_pressure
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

        # Check that σᶠᶜ differs from σᶜᶜ (staggered interpolation)
        # σᶠᶜ at face i should be the average of σᶜᶜ at i-1 and i
        for i in 2:Nx
            σᶠᶜ_expected = (grid.z.σᶜᶜⁿ[i-1, 1, 1] + grid.z.σᶜᶜⁿ[i, 1, 1]) / 2
            @test grid.z.σᶠᶜⁿ[i, 1, 1] ≈ σᶠᶜ_expected rtol=1e-10
        end

        # Check that physical z-nodes reflect terrain
        # At the surface (k=1, Face), z should equal h(x)
        for i in 1:Nx
            x = xnode(i, grid, Center())
            h_expected = h₀ * exp(-x^2 / a^2)
            z_surface = znode(i, 1, 1, grid, Center(), Center(), Face())
            @test z_surface ≈ h_expected rtol=1e-10
        end

        # At the top (k=Nz+1, Face), z should equal Lz
        for i in 1:Nx
            z_top_computed = znode(i, 1, Nz+1, grid, Center(), Center(), Face())
            @test z_top_computed ≈ Lz rtol=1e-10
        end
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

    @testset "Terrain-following grid with CompressibleDynamics (no terrain physics)" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x, y) = 200 * exp(-x^2 / 2000^2)
        metrics = follow_terrain!(grid, h)

        # Without terrain_metrics in dynamics, uses standard physics
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(ExplicitTimeStepping()))
        @test model isa AtmosphereModel
        @test model.dynamics.terrain_metrics === nothing

        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        constants = model.thermodynamic_constants
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)
        set!(model, ρ=ρᵢ, θ=θ₀)

        Δt = 0.1
        time_step!(model, Δt)
        @test isfinite(maximum(abs, model.velocities.w))
    end

    @testset "Terrain-following CompressibleDynamics with terrain physics" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x, y) = 200 * exp(-x^2 / 2000^2)
        metrics = follow_terrain!(grid, h)

        # With terrain_metrics, physics includes terrain corrections
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); terrain_metrics=metrics)
        model = AtmosphereModel(grid; dynamics)

        @test model isa AtmosphereModel
        @test model.dynamics.terrain_metrics isa TerrainMetrics
        @test model.dynamics.contravariant_vertical_velocity !== nothing
        @test model.dynamics.contravariant_vertical_momentum !== nothing

        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        constants = model.thermodynamic_constants
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)
        set!(model, ρ=ρᵢ, θ=θ₀)

        Δt = 0.1
        time_step!(model, Δt)
        @test isfinite(maximum(abs, model.velocities.w))
        @test isfinite(maximum(abs, model.dynamics.contravariant_vertical_velocity))
    end

    @testset "Terrain-following CompressibleDynamics on a LatitudeLongitudeGrid" begin
        # Spherical-geometry insurance: the same terrain physics must run on a
        # LatitudeLongitudeGrid, where the metric terms carry cos(φ) factors and
        # the topography is sampled at (λ, φ) rather than (x, y).
        Nλ, Nφ, Nz = 8, 8, 8
        Lz = 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = LatitudeLongitudeGrid(CPU(); size=(Nλ, Nφ, Nz),
                                     longitude=(-2, 2), latitude=(40, 44), z=z_faces)

        h(λ, φ) = 200 * exp(-(λ^2 + (φ - 42)^2) / 0.5)
        metrics = follow_terrain!(grid, h)

        dynamics = CompressibleDynamics(ExplicitTimeStepping(); terrain_metrics=metrics)
        model = AtmosphereModel(grid; dynamics)

        @test model isa AtmosphereModel
        @test model.dynamics.terrain_metrics isa TerrainMetrics
        @test model.dynamics.contravariant_vertical_velocity !== nothing

        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        constants = model.thermodynamic_constants
        ρᵢ(λ, φ, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)
        set!(model, ρ=ρᵢ, θ=θ₀)

        Δt = 0.1
        time_step!(model, Δt)
        @test isfinite(maximum(abs, model.velocities.w))
        @test isfinite(maximum(abs, model.dynamics.contravariant_vertical_velocity))
    end

    @testset "Contravariant velocity for horizontal flow over terrain" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 200.0
        a = 2000.0
        h(x, y) = h₀ * exp(-x^2 / a^2)
        metrics = follow_terrain!(grid, h)

        dynamics = CompressibleDynamics(ExplicitTimeStepping(); terrain_metrics=metrics)
        model = AtmosphereModel(grid; dynamics)

        constants = model.thermodynamic_constants
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

        U₀ = 10.0
        set!(model, ρ=ρᵢ, θ=θ₀, u=U₀)

        # Take one step to trigger computation of Ω̃
        time_step!(model, 0.1)

        # Ω̃ should be nonzero near the mountain (terrain slopes are nonzero)
        # and near zero far from the mountain (terrain slopes ≈ 0)
        Ω̃ = model.dynamics.contravariant_vertical_velocity
        @test maximum(abs, Ω̃) > 0

        # At the model top (k = Nz+1), terrain slopes decay to zero so Ω̃ ≈ w
        # (the decay factor is 1 - ζ/z_top = 0 at the top)
        w = model.velocities.w
        for i in 1:Nx
            @test Ω̃[i, 1, Nz+1] ≈ w[i, 1, Nz+1] atol=1e-10
        end
    end

    @testset "Terrain reference state matches continuous hydrostatic profile" begin
        # The terrain reference state p_ref(i,j,k) must equal the continuous
        # hydrostatic pressure evaluated at the local physical height z(i,j,k).
        # A bug that initializes every column from sea-level pressure creates
        # O(ρgh) errors over terrain.
        Nx, Nz = 16, 8
        Lx, Lz = 100000.0, 10000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 1000.0
        a = 10000.0
        h(x, y) = h₀ * exp(-x^2 / a^2)
        metrics = follow_terrain!(grid, h)

        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        terrain_metrics=metrics,
                                        reference_potential_temperature=θ₀)
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants

        p_ref = model.dynamics.terrain_reference_pressure

        # At each grid point, p_ref should match the continuous profile
        # to within the discretization error of the Exner integration (O(Δz²))
        for i in 1:Nx, k in 1:Nz
            z_phys = znode(i, 1, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z_phys, p₀, θ₀, pˢᵗ, constants)
            # Discrete Exner integration has O(Δz²) error; with Δz ≈ 1250 m
            # the accumulated error at the top is ~0.5%, so use 1% tolerance
            @test p_ref[i, 1, k] ≈ p_exact rtol=1e-2
        end

        # Critical check: at a given k-level, p_ref must NOT be constant across
        # columns (it should vary because physical heights differ). But at the
        # SAME physical height, values from different columns should agree closely.
        # Compare the flat column (i at domain edge) vs the mountain-top column.
        i_flat = 1    # far from mountain
        i_peak = Nx÷2 # near mountain peak
        z_flat_1 = znode(i_flat, 1, 1, grid, Center(), Center(), Center())
        z_peak_1 = znode(i_peak, 1, 1, grid, Center(), Center(), Center())

        # Physical heights differ, so p_ref at k=1 should differ
        @test z_peak_1 > z_flat_1 + 100  # mountain is at least 100 m higher
        @test p_ref[i_peak, 1, 1] < p_ref[i_flat, 1, 1]  # higher altitude → lower pressure
    end

    @testset "Terrain reference state with θ(z) profile (Function dispatch)" begin
        # Same test but with a non-constant potential temperature profile,
        # exercising the numerically_integrated_hydrostatic_pressure path.
        Nx, Nz = 16, 16
        Lx, Lz = 100000.0, 10000.0

        z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 1000.0
        a = 10000.0
        h(x, y) = h₀ * exp(-x^2 / a^2)
        metrics = follow_terrain!(grid, h)

        g_val = 9.80665
        N² = 1e-4
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        θ_of_z(z) = θ₀ * exp(N² * z / g_val)

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        terrain_metrics=metrics,
                                        reference_potential_temperature=θ_of_z)
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants

        p_ref = model.dynamics.terrain_reference_pressure

        # At each grid point, p_ref should match the continuous profile
        for i in 1:Nx, k in 1:Nz
            z_phys = znode(i, 1, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z_phys, p₀, θ_of_z, pˢᵗ, constants)
            # Finer grid (Nz=16) so tighter tolerance than Nz=8 test
            @test p_ref[i, 1, k] ≈ p_exact rtol=5e-3
        end

        # Mountain-top column should have lower p_ref at k=1 than flat column
        i_flat = 1
        i_peak = Nx÷2
        @test p_ref[i_peak, 1, 1] < p_ref[i_flat, 1, 1]
    end

    @testset "set_topography! horizontal coordinates (Flat, 3D, lat-lon)" begin
        # The horizontal coordinates passed to topography(...) come from
        # ξnode/ηnode (see follow_terrain.jl::set_topography!). This guards two
        # properties that a naive `node(...)[1:2]` would break:
        #   (a) on a 2D x–z (Flat) grid the second argument is the degenerate
        #       meridional node `nothing`, NOT the vertical coordinate z;
        #   (b) the coordinates generalize to (λ, φ) on a LatitudeLongitudeGrid.

        # (a) Flat grid: probe returns a sentinel iff the second arg is `nothing`.
        #     If z ever leaked into that slot, the probe would return 2.0 instead.
        Nx, Nz = 8, 6
        zf = MutableVerticalDiscretization(collect(range(0, 500.0, length=Nz+1)))
        flat_grid = RectilinearGrid(CPU(); size=(Nx, Nz), x=(-500, 500), z=zf,
                                    topology=(Periodic, Flat, Bounded))
        h_probe(x, y) = y === nothing ? 1.0 : 2.0
        m_flat = follow_terrain!(flat_grid, h_probe)
        @test all(m_flat.topography[i, 1, 1] == 1.0 for i in 1:Nx)

        # (b) 3D grid: topography receives the true (x, y).
        Nx3, Ny3, Nz3 = 5, 4, 3
        zf3 = MutableVerticalDiscretization(collect(range(0, 300.0, length=Nz3+1)))
        grid3 = RectilinearGrid(CPU(); size=(Nx3, Ny3, Nz3), x=(0, 100), y=(0, 80),
                                z=zf3, topology=(Periodic, Periodic, Bounded))
        h_xy(x, y) = x + 2y
        m3 = follow_terrain!(grid3, h_xy)
        for i in 1:Nx3, j in 1:Ny3
            expected = xnode(i, grid3, Center()) + 2 * ynode(j, grid3, Center())
            @test m3.topography[i, j, 1] ≈ expected
        end

        # (c) LatitudeLongitudeGrid: topography receives (λ, φ) in degrees.
        Nλ, Nφ, Nzll = 6, 5, 4
        zfll = MutableVerticalDiscretization(collect(range(0, 4000.0, length=Nzll+1)))
        llg = LatitudeLongitudeGrid(CPU(); size=(Nλ, Nφ, Nzll),
                                    longitude=(-5, 5), latitude=(40, 50), z=zfll)
        h_λφ(λ, φ) = 100.0 + λ + 2φ
        m_ll = follow_terrain!(llg, h_λφ)
        for i in 1:Nλ, j in 1:Nφ
            expected = h_λφ(λnode(i, llg, Center()), φnode(j, llg, Center()))
            @test m_ll.topography[i, j, 1] ≈ expected
        end
    end
end
