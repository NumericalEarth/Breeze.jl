using Breeze
using Oceananigans.Architectures: CPU

# Run under `default_arch` when the test runner provides it (which routes to
# GPU when CUDA is functional, matching project convention); otherwise fall
# back to CPU() so this file can also be included directly with
# `julia --project=. test/<this file>.jl` during development.
@isdefined(default_arch) || (default_arch = CPU())

using CUDA: @allowscalar

using Breeze.CompressibleEquations: compute_contravariant_velocity!
using Oceananigans
using Oceananigans.Grids: znode, λnode, φnode
using Breeze.Thermodynamics: hydrostatic_pressure
using Test

@allowscalar begin
@testset "TerrainFollowing on LatitudeLongitudeGrid" begin
    @testset "materialize_terrain! on LatitudeLongitudeGrid" begin
        Nλ, Nφ, Nz = 36, 15, 8
        Lz = 30000.0

        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1));
            formulation = LinearDecay())

        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz),
                                     halo = (5, 5, 5),
                                     longitude = (0, 360),
                                     latitude = (-75, 75),
                                     z = z_faces)

        h₀ = 200.0
        λ_center = 180.0
        φ_center = 40.0
        h(λ, φ) = h₀ * exp(-((λ - λ_center)^2 + (φ - φ_center)^2) / 20^2)

        materialize_terrain!(grid, h)

        # Surface z should equal h(λ, φ) at each cell center
        for i in 1:Nλ, j in 1:Nφ
            λ = λnode(i, grid, Center())
            φ = φnode(j, grid, Center())
            h_expected = h(λ, φ)
            z_surface = znode(i, j, 1, grid, Center(), Center(), Face())
            @test z_surface ≈ h_expected rtol=1e-10
        end

        # Top should be flat at Lz
        for i in 1:Nλ, j in 1:Nφ
            z_top = znode(i, j, Nz+1, grid, Center(), Center(), Face())
            @test z_top ≈ Lz rtol=1e-10
        end
    end

    @testset "LLG: materialize_terrain! slopes and TwoLevelDecay" begin
        Nλ, Nφ, Nz = 36, 30, 6
        Lz = 10000.0

        # LinearDecay: check slopes
        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)

        mountain(λ, φ) = 200 * exp(-((λ - 180)^2 + φ^2) / 20^2)
        materialize_terrain!(grid, mountain)
        metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

        @test maximum(abs, metrics.∂x_h) > 0
        @test maximum(abs, metrics.∂y_h) > 0
        @test maximum(metrics.∂y_h) > 0   # positive slope south of equator
        @test minimum(metrics.∂y_h) < 0   # negative slope north of equator

        # TwoLevelDecay: smoothing decomposes terrain into large+small scale
        z_faces_2 = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=5));
            formulation = TwoLevelDecay(large_scale_height=8000, small_scale_height=2000))
        grid_2 = LatitudeLongitudeGrid(default_arch;
                                        size = (12, 10, 4), halo = (5, 5, 5),
                                        longitude = (0, 360), latitude = (-60, 60),
                                        z = z_faces_2)
        materialize_terrain!(grid_2, mountain)
        @test maximum(abs, grid_2.z.formulation.h₂) > 0
    end

    @testset "LLG: explicit time stepping with terrain" begin
        Nλ, Nφ, Nz = 12, 10, 6
        Lz = 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)
        materialize_terrain!(grid, (λ, φ) -> 200 * exp(-((λ - 180)^2 + (φ - 30)^2) / 20^2))

        dynamics = CompressibleDynamics(ExplicitTimeStepping())
        model = AtmosphereModel(grid; dynamics)

        @test model.dynamics.terrain_metrics isa TerrainMetrics
        @test model.dynamics.contravariant_vertical_velocity !== nothing

        constants = model.thermodynamic_constants
        ρᵢ(λ, φ, z) = adiabatic_hydrostatic_density(z, 101325.0, 300.0, 1e5, constants)
        set!(model, ρ=ρᵢ, θ=300.0, u=10.0)
        time_step!(model, 0.1)

        w̃ = model.dynamics.contravariant_vertical_velocity
        @test isfinite(maximum(abs, model.velocities.w))
        @test maximum(abs, w̃) > 0

        # At the model top, terrain slopes decay to zero so w̃ ≈ w
        for i in 1:Nλ, j in 1:Nφ
            @test w̃[i, j, Nz+1] ≈ model.velocities.w[i, j, Nz+1] atol=1e-10
        end
    end

    @testset "LLG: split-explicit terrain transport, mass conservation, kinematic BC" begin
        Nλ, Nφ, Nz = 12, 10, 6
        Lz = 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)
        materialize_terrain!(grid, (λ, φ) -> 100 * exp(-((λ - 180)^2 + (φ - 30)^2) / 20^2))

        # Covers: adaptive substeps, slope-inside stencil, divergence damping
        damping = ThermalDivergenceDamping(coefficient=0.05, damp_vertical=true)
        dynamics = CompressibleDynamics(
            SplitExplicitTimeDiscretization(acoustic_cfl=0.5; damping=damping);
            slope_stencil = SlopeInsideInterpolation(),
            reference_potential_temperature = 300)
        model = AtmosphereModel(grid; dynamics)
        set!(model, θ=300, ρ=model.dynamics.reference_state.density, u=0, w=0)

        @test model.dynamics.terrain_metrics.pressure_gradient_stencil isa SlopeInsideInterpolation
        @test model.timestepper.substepper.substeps === nothing  # adaptive

        initial_mass = sum(interior(model.dynamics.dry_density))
        time_step!(model, 0.1)
        final_mass = sum(interior(model.dynamics.dry_density))

        w̃ = model.dynamics.contravariant_vertical_velocity
        ρw̃ = model.dynamics.contravariant_vertical_momentum

        @test isfinite(maximum(abs, w̃))
        @test isapprox(maximum(abs, interior(model.velocities.w)), 0; atol = 1e-12)
        @test isapprox(maximum(abs, interior(w̃)), 0; atol = 1e-12)
        @test abs(final_mass - initial_mass) / initial_mass <= 1e-13

        # Kinematic BC: w̃ = 0 at the surface
        for i in 1:Nλ, j in 1:Nφ
            @test isapprox(w̃[i, j, 1], 0; atol = 1e-12)
            @test isapprox(ρw̃[i, j, 1], 0; atol = 1e-12)
        end
    end

    @testset "LLG: zero terrain recovers height-coordinate results" begin
        Nλ, Nφ, Nz = 12, 10, 6
        Lz = 10000.0

        # Explicit: w̃ ≡ w, ρw̃ ≡ ρw when h = 0
        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)
        materialize_terrain!(grid, (λ, φ) -> 0)

        dynamics = CompressibleDynamics(ExplicitTimeStepping())
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants
        ρᵢ(λ, φ, z) = adiabatic_hydrostatic_density(z, 101325.0, 300.0, 1e5, constants)
        set!(model, ρ=ρᵢ, θ=300.0, u=10, w=1)
        compute_contravariant_velocity!(model)

        @test isapprox(interior(model.dynamics.contravariant_vertical_velocity),
                       interior(model.velocities.w); atol=1e-12)
        @test isapprox(interior(model.dynamics.contravariant_vertical_momentum),
                       interior(model.momentum.ρw); atol=1e-12)

        # Split-explicit: TFVD(h=0) matches plain LLG after one step
        function llg_flat_model(terrain)
            g = if terrain
                zf = TerrainFollowingVerticalDiscretization(
                    collect(range(0, Lz, length=Nz+1)); formulation=LinearDecay())
                g = LatitudeLongitudeGrid(default_arch; size=(Nλ, Nφ, Nz), halo=(5, 5, 5),
                                          longitude=(0, 360), latitude=(-60, 60), z=zf)
                materialize_terrain!(g, (λ, φ) -> 0)
                g
            else
                LatitudeLongitudeGrid(default_arch; size=(Nλ, Nφ, Nz), halo=(5, 5, 5),
                                      longitude=(0, 360), latitude=(-60, 60), z=(0, Lz))
            end
            m = AtmosphereModel(g;
                dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6)))
            set!(m, ρ=1, θ=300, u=0.1, w=0.01)
            return m
        end

        height_model  = llg_flat_model(false)
        terrain_model = llg_flat_model(true)
        time_step!(height_model, 0.01)
        time_step!(terrain_model, 0.01)

        tol = 1e-14
        height_ρθ  = Breeze.AtmosphereModels.thermodynamic_density(height_model.formulation)
        terrain_ρθ = Breeze.AtmosphereModels.thermodynamic_density(terrain_model.formulation)

        @test isapprox(interior(height_model.dynamics.dry_density),
                       interior(terrain_model.dynamics.dry_density); atol=tol)
        @test isapprox(interior(height_ρθ), interior(terrain_ρθ); atol=tol)
        @test isapprox(interior(height_model.momentum.ρu),
                       interior(terrain_model.momentum.ρu); atol=tol)
        @test isapprox(interior(height_model.momentum.ρw),
                       interior(terrain_model.momentum.ρw); atol=tol)
    end

    @testset "LLG: terrain reference state matches hydrostatic profile" begin
        Nλ, Nφ, Nz = 12, 10, 16
        Lz = 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)

        h₀ = 1000.0
        materialize_terrain!(grid, (λ, φ) -> h₀ * exp(-((λ - 180)^2 + (φ - 30)^2) / 20^2))

        g_val = 9.80665
        N² = 1e-4
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        θ_of_z(z) = θ₀ * exp(N² * z / g_val)

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature=θ_of_z)
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants
        pᵣ = model.dynamics.reference_state.pressure

        for i in 1:Nλ, j in 1:Nφ, k in 1:Nz
            z = znode(i, j, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z, p₀, θ_of_z, pˢᵗ, constants)
            @test pᵣ[i, j, k] ≈ p_exact rtol=5e-3
        end

        # Mountain-top column should have lower pressure than a flat column
        z_max, i_peak, j_peak = -Inf, 1, 1
        for i in 1:Nλ, j in 1:Nφ
            z_sfc = znode(i, j, 1, grid, Center(), Center(), Face())
            if z_sfc > z_max
                z_max = z_sfc
                i_peak, j_peak = i, j
            end
        end
        @test z_max > znode(1, 1, 1, grid, Center(), Center(), Face()) + 100
        @test pᵣ[i_peak, j_peak, 1] < pᵣ[1, 1, 1]
    end

    @testset "LLG: terrain_amg_operators chain-rule dispatch coverage" begin
        Nλ, Nφ, Nz = 12, 10, 6
        Lz = 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)
        materialize_terrain!(grid, (λ, φ) -> 200 * exp(-((λ - 180)^2 + (φ - 30)^2) / 20^2))

        i, j, k = 4, 4, 3
        tol = 1e-10 * Lz

        # Slope operators should return finite values on a sloped grid
        slope_ops = (
            Oceananigans.Operators.∂x_zᶠᶜᶜ, Oceananigans.Operators.∂x_zᶜᶜᶜ,
            Oceananigans.Operators.∂x_zᶠᶜᶠ, Oceananigans.Operators.∂x_zᶜᶠᶜ,
            Oceananigans.Operators.∂x_zᶠᶠᶜ, Oceananigans.Operators.∂x_zᶜᶜᶠ,
            Oceananigans.Operators.∂y_zᶜᶠᶜ, Oceananigans.Operators.∂y_zᶜᶜᶜ,
            Oceananigans.Operators.∂y_zᶜᶠᶠ, Oceananigans.Operators.∂y_zᶠᶜᶜ,
            Oceananigans.Operators.∂y_zᶠᶠᶜ, Oceananigans.Operators.∂y_zᶜᶜᶠ,
        )
        for op in slope_ops
            @test isfinite(op(i, j, k, grid))
        end

        # Derivative of a constant is exactly zero
        number_ops = (
            Oceananigans.Operators.∂xᶠᶜᶜ, Oceananigans.Operators.∂xᶜᶜᶜ,
            Oceananigans.Operators.∂xᶠᶜᶠ, Oceananigans.Operators.∂xᶜᶠᶜ,
            Oceananigans.Operators.∂xᶠᶠᶜ,
            Oceananigans.Operators.∂yᶜᶠᶜ, Oceananigans.Operators.∂yᶜᶜᶜ,
            Oceananigans.Operators.∂yᶜᶠᶠ, Oceananigans.Operators.∂yᶠᶜᶜ,
            Oceananigans.Operators.∂yᶠᶠᶜ,
        )
        for op in number_ops
            @test op(i, j, k, grid, 1.0) == 0
        end

        # Function-arg: ϕ = znode(physical) → chain-rule should give ≈ 0
        function_arg_cases = (
            (Oceananigans.Operators.∂xᶠᶜᶜ, Center(), Center(), Center()),
            (Oceananigans.Operators.∂xᶜᶜᶜ, Face(),   Center(), Center()),
            (Oceananigans.Operators.∂xᶠᶜᶠ, Center(), Center(), Face()),
            (Oceananigans.Operators.∂xᶜᶠᶜ, Face(),   Face(),   Center()),
            (Oceananigans.Operators.∂xᶠᶠᶜ, Center(), Face(),   Center()),
            (Oceananigans.Operators.∂yᶜᶠᶜ, Center(), Center(), Center()),
            (Oceananigans.Operators.∂yᶜᶜᶜ, Center(), Face(),   Center()),
            (Oceananigans.Operators.∂yᶜᶠᶠ, Center(), Center(), Face()),
            (Oceananigans.Operators.∂yᶠᶜᶜ, Face(),   Face(),   Center()),
            (Oceananigans.Operators.∂yᶠᶠᶜ, Face(),   Center(), Center()),
        )
        for (op, ℓx, ℓy, ℓz) in function_arg_cases
            v = op(i, j, k, grid, Oceananigans.Grids.znode, ℓx, ℓy, ℓz)
            @test abs(v) < tol
        end

        # Field-arg: fill Field with physical z, assert chain-rule ≈ 0
        field_cases = (
            (Oceananigans.Operators.∂xᶠᶜᶜ, Center, Center, Center),
            (Oceananigans.Operators.∂xᶜᶜᶜ, Face,   Center, Center),
            (Oceananigans.Operators.∂xᶠᶜᶠ, Center, Center, Face),
            (Oceananigans.Operators.∂xᶜᶠᶜ, Face,   Face,   Center),
            (Oceananigans.Operators.∂xᶠᶠᶜ, Center, Face,   Center),
            (Oceananigans.Operators.∂yᶜᶠᶜ, Center, Center, Center),
            (Oceananigans.Operators.∂yᶜᶜᶜ, Center, Face,   Center),
            (Oceananigans.Operators.∂yᶜᶠᶠ, Center, Center, Face),
            (Oceananigans.Operators.∂yᶠᶜᶜ, Face,   Face,   Center),
            (Oceananigans.Operators.∂yᶠᶠᶜ, Face,   Center, Center),
        )
        for (op, LX, LY, LZ) in field_cases
            ϕ = Field{LX, LY, LZ}(grid)
            set!(ϕ, (λ, φ, z) -> z)
            v = op(i, j, k, grid, ϕ)
            @test abs(v) < tol
        end
    end

    @testset "LLG: on_architecture round-trip preserves materialised terrain" begin
        Nλ, Nφ, Nz = 12, 10, 8
        Lz = 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(
            collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = LatitudeLongitudeGrid(default_arch;
                                     size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60),
                                     z = z_faces)

        mountain(λ, φ) = 300 * exp(-((λ - 180)^2 + (φ - 30)^2) / 20^2)
        materialize_terrain!(grid, mountain)

        cpu_arch = Oceananigans.Architectures.CPU()
        rebuilt = Oceananigans.Architectures.on_architecture(cpu_arch, grid)

        @test rebuilt.z isa TerrainFollowingVerticalDiscretization
        @test rebuilt.z.formulation.h !== nothing
        @test rebuilt.z.formulation.∂x_h !== nothing

        @allowscalar for i in 1:Nλ, j in 1:Nφ
            λ = λnode(i, rebuilt, Center())
            φ = φnode(j, rebuilt, Center())
            h_expected = mountain(λ, φ)
            @test znode(i, j, 1,    rebuilt, Center(), Center(), Face()) ≈ h_expected rtol=1e-10
            @test znode(i, j, Nz+1, rebuilt, Center(), Center(), Face()) ≈ Lz          rtol=1e-10
        end
    end

end
end
