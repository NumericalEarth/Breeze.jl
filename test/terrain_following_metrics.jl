using Breeze
using Breeze.AtmosphereModels: x_pressure_gradient
using Oceananigans.Architectures: CPU

# Run under `default_arch` when the test runner provides it (which routes to
# GPU when CUDA is functional, matching project convention); otherwise fall
# back to CPU() so this file can also be included directly with
# `julia --project=. test/<this file>.jl` during development.
@isdefined(default_arch) || (default_arch = CPU())

using CUDA: @allowscalar

using Breeze.CompressibleEquations: compute_contravariant_velocity!, sponge_rhs, sponge_term_diag
using Oceananigans
using Oceananigans.Grids: rnode, xnode, znode
using Oceananigans.Operators: divᶜᶜᶜ
using Test

@allowscalar begin
@testset "TerrainFollowing metrics and explicit dynamics" begin
    @testset "materialize_terrain! with function topography" begin
        Nx, Nz = 32, 10
        Lx, Lz = 100000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 500.0
        a = 2000.0
        h(x) = h₀ * exp(-x^2 / a^2)

        materialize_terrain!(grid, h)
        metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

        # Check that metrics are returned
        @test metrics isa TerrainMetrics

        # Check z_top
        @test metrics.z_top ≈ Lz

        # Check that physical z-nodes reflect terrain (TFVD: z = r + h(x)·b(r)).
        # At the surface (k=1, Face), z should equal h(x).
        for i in 1:Nx
            x = xnode(i, grid, Center())
            h_expected = h₀ * exp(-x^2 / a^2)
            z_surface = znode(i, 1, 1, grid, Center(), Center(), Face())
            @test z_surface ≈ h_expected rtol=1e-10
        end

        # At the top (k=Nz+1, Face), z should equal Lz.
        for i in 1:Nx
            z_top_computed = znode(i, 1, Nz+1, grid, Center(), Center(), Face())
            @test z_top_computed ≈ Lz rtol=1e-10
        end
    end

    @testset "materialize_terrain! terrain slopes" begin
        Nx, Nz = 64, 10
        Lx, Lz = 100000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        # Gaussian mountain: h(x) = h₀ * exp(-x² / a²), so ∂h/∂x = -2x/a² * h
        h₀ = 500.0
        a = 10000.0
        h(x) = h₀ * exp(-x^2 / a^2)

        materialize_terrain!(grid, h)
        metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

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

    @testset "TwoLevelDecay smoothing separates y-dependent terrain" begin
        Nx, Ny, Nz = 8, 8, 4
        Lx, Ly, Lz = 10000.0, 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                          formulation = TwoLevelDecay(large_scale_height = 4000,
                                                                                      small_scale_height = 1000))
        grid = RectilinearGrid(default_arch; size=(Nx, Ny, Nz),
                               x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=z_faces,
                               topology=(Periodic, Periodic, Bounded))

        h₀ = 100.0
        h(x, y) = h₀ * sin(2π * y / Ly)

        materialize_terrain!(grid, h)

        maximum_h₂ = 0.0
        maximum_∂y_h₂ = 0.0
        for i in 1:Nx, j in 1:Ny
            maximum_h₂ = max(maximum_h₂, abs(grid.z.formulation.h₂[i, j, 1]))
            maximum_∂y_h₂ = max(maximum_∂y_h₂, abs(grid.z.formulation.∂y_h₂[i, j, 1]))
        end

        @test maximum_h₂ > 0.1h₀
        @test maximum_∂y_h₂ > 0
    end

    @testset "Non-TFVD grid leaves terrain_metrics nothing" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        # Plain height-coordinate RectilinearGrid (no TFVD): `CompressibleDynamics`
        # should leave terrain_metrics === nothing regardless of any kwarg.
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=(0, Lz),
                               topology=(Periodic, Flat, Bounded))

        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(ExplicitTimeStepping()))
        @test model isa AtmosphereModel
        @test model.dynamics.terrain_metrics === nothing
        @test model.dynamics.contravariant_vertical_velocity === nothing
        @test model.dynamics.contravariant_vertical_momentum === nothing

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

    @testset "CompressibleDynamics auto-builds TerrainMetrics on TFVD grid" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 200 * exp(-x^2 / 2000^2))

        # Default: stencil is SlopeOutsideInterpolation()
        model_default = AtmosphereModel(grid;
            dynamics=CompressibleDynamics(ExplicitTimeStepping()))
        @test model_default.dynamics.terrain_metrics isa TerrainMetrics
        @test model_default.dynamics.terrain_metrics.pressure_gradient_stencil isa SlopeOutsideInterpolation

        # Override via slope_stencil kwarg
        model_inside = AtmosphereModel(grid;
            dynamics=CompressibleDynamics(ExplicitTimeStepping(); slope_stencil = SlopeInsideInterpolation()))
        @test model_inside.dynamics.terrain_metrics.pressure_gradient_stencil isa SlopeInsideInterpolation

        # Escape hatch: pass a pre-built TerrainMetrics; it should pass through verbatim
        pre_built = build_terrain_metrics(grid, SlopeInsideInterpolation())
        model_explicit = AtmosphereModel(grid;
            dynamics=CompressibleDynamics(ExplicitTimeStepping(); terrain_metrics = pre_built))
        @test model_explicit.dynamics.terrain_metrics === pre_built
    end

    @testset "Terrain-following CompressibleDynamics with terrain physics" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 200 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)

        # With terrain_metrics, physics includes terrain corrections
        dynamics = CompressibleDynamics(ExplicitTimeStepping())
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

    @testset "Contravariant velocity for horizontal flow over terrain" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 200.0
        a = 2000.0
        h(x) = h₀ * exp(-x^2 / a^2)
        materialize_terrain!(grid, h)

        dynamics = CompressibleDynamics(ExplicitTimeStepping())
        model = AtmosphereModel(grid; dynamics)

        constants = model.thermodynamic_constants
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

        U₀ = 10.0
        set!(model, ρ=ρᵢ, θ=θ₀, u=U₀)

        # Take one step to trigger computation of w̃
        time_step!(model, 0.1)

        # w̃ should be nonzero near the mountain (terrain slopes are nonzero)
        # and near zero far from the mountain (terrain slopes ≈ 0)
        w̃ = model.dynamics.contravariant_vertical_velocity
        @test maximum(abs, w̃) > 0

        # At the model top (k = Nz+1), terrain slopes decay to zero so w̃ ≈ w
        # (the decay factor is 1 - r/z_top = 0 at the top)
        w = model.velocities.w
        for i in 1:Nx
            @test w̃[i, 1, Nz+1] ≈ w[i, 1, Nz+1] atol=1e-10
        end
    end

    @testset "Zero terrain recovers Cartesian vertical transport" begin
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        materialize_terrain!(grid, x -> 0)

        dynamics = CompressibleDynamics(ExplicitTimeStepping())
        model = AtmosphereModel(grid; dynamics)

        constants = model.thermodynamic_constants
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

        set!(model, ρ=ρᵢ, θ=θ₀, u=10, w=1)
        compute_contravariant_velocity!(model)

        # On TFVD with h ≡ 0, the basis functions still get evaluated but the
        # slope factor is zero, so ρw̃ ≈ ρw to machine precision (not bit-equal).
        w̃ = model.dynamics.contravariant_vertical_velocity
        ρw̃ = model.dynamics.contravariant_vertical_momentum
        @test isapprox(interior(w̃),  interior(model.velocities.w); atol = 1e-12)
        @test isapprox(interior(ρw̃), interior(model.momentum.ρw);  atol = 1e-12)
    end

    @testset "Terrain metric identities for constant fields" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 200 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)
        # This testset checks raw metric identities of the PGF operator (e.g. a horizontally
        # constant *total* pressure has zero gradient), which hold for the full-pressure form.
        # Disable the terrain reference (on by default for terrain grids) so the operator
        # differences the full pressure rather than a perturbation about pᵣ(z).
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_state=nothing)
        model = AtmosphereModel(grid; dynamics)

        set!(model, ρ=1, θ=300, u=1, w=0)
        compute_contravariant_velocity!(model)

        maximum_interior_divergence = 0.0
        for i in 1:Nx, k in 2:Nz
            divergence = divᶜᶜᶜ(i, 1, k, grid,
                                 model.momentum.ρu,
                                 model.momentum.ρv,
                                 model.dynamics.contravariant_vertical_momentum)
            maximum_interior_divergence = max(maximum_interior_divergence, abs(divergence))
        end
        @test maximum_interior_divergence <= 1e-12

        fill!(parent(model.dynamics.pressure), 100000)
        maximum_pressure_gradient = 0.0
        for i in 1:Nx, k in 1:Nz
            pressure_gradient = x_pressure_gradient(i, 1, k, grid, model.dynamics)
            maximum_pressure_gradient = max(maximum_pressure_gradient, abs(pressure_gradient))
        end
        @test maximum_pressure_gradient == 0

        function terrain_pressure_gradient_error(pressure_function, expected_gradient, stencil)
            z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
            grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                                   x=(-Lx/2, Lx/2), z=z_faces,
                                   topology=(Periodic, Flat, Bounded))
            materialize_terrain!(grid, h)
            dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                            slope_stencil = stencil,
                                            reference_state = nothing)
            model = AtmosphereModel(grid; dynamics)

            for i in 1:Nx, k in 1:Nz
                x = xnode(i, grid, Center())
                z = znode(i, 1, k, grid, Center(), Center(), Center())
                model.dynamics.pressure[i, 1, k] = pressure_function(x, z)
            end

            maximum_error = 0.0
            for i in 2:Nx-1, k in 2:Nz-1
                pressure_gradient = x_pressure_gradient(i, 1, k, grid, model.dynamics)
                maximum_error = max(maximum_error, abs(pressure_gradient - expected_gradient))
            end

            return maximum_error
        end

        for stencil in (SlopeOutsideInterpolation(), SlopeInsideInterpolation())
            @test terrain_pressure_gradient_error((x, z) -> x, 1, stencil) == 0
        end

        # For p = z, the physical horizontal pressure gradient at constant z is zero.
        # The outside-interpolation stencil delegates to Oceananigans' generalized
        # derivative and cancels this manufactured metric term to roundoff.
        @test terrain_pressure_gradient_error((x, z) -> z, 0,
                                              SlopeOutsideInterpolation()) <= 1e-12

        function slope_inside_metric_cancellation_error(Nx, Nz)
            Lx, Lz = 10000.0, 5000.0
            z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
            grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                                   x=(-Lx/2, Lx/2), z=z_faces,
                                   topology=(Periodic, Flat, Bounded))

            h(x) = 200 * exp(-x^2 / 2000^2)
            materialize_terrain!(grid, h)
            dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                            slope_stencil = SlopeInsideInterpolation(),
                                            reference_state = nothing)
            model = AtmosphereModel(grid; dynamics)

            for i in 1:Nx, k in 1:Nz
                z = znode(i, 1, k, grid, Center(), Center(), Center())
                model.dynamics.pressure[i, 1, k] = z
            end

            maximum_error = 0.0
            for i in 3:Nx-2, k in 3:Nz-2
                pressure_gradient = x_pressure_gradient(i, 1, k, grid, model.dynamics)
                maximum_error = max(maximum_error, abs(pressure_gradient))
            end

            return maximum_error
        end

        coarse_error = slope_inside_metric_cancellation_error(16, 8)
        medium_error = slope_inside_metric_cancellation_error(32, 16)
        fine_error = slope_inside_metric_cancellation_error(64, 32)

        @test medium_error < coarse_error / 2
        @test fine_error < medium_error / 2

        function terrain_divergence_error(Nx, Nz)
            Lx, Lz = 10000.0, 5000.0
            h₀ = 200.0
            a = 2000.0

            h(x) = h₀ * exp(-x^2 / a^2)
            ∂x_h(x) = -2 * x / a^2 * h(x)
            σ(x) = (Lz - h(x)) / Lz
            ∂x_σ(x) = -∂x_h(x) / Lz

            z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
            grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                                   x=(-Lx/2, Lx/2), z=z_faces,
                                   topology=(Periodic, Flat, Bounded))
            materialize_terrain!(grid, h)

            ρu = XFaceField(grid)
            ρv = YFaceField(grid)
            ρw̃ = ZFaceField(grid)

            for i in 1:Nx+1, k in 1:Nz
                x = xnode(i, grid, Face())
                ρu[i, 1, k] = sin(2π * x / Lx)
            end

            for i in 1:Nx, k in 1:Nz+1
                r = rnode(k, grid, Face())
                ρw̃[i, 1, k] = cos(π * r / Lz)
            end

            set!(ρv, 0)

            maximum_error = 0.0
            for i in 3:Nx-2, k in 2:Nz-1
                x = xnode(i, grid, Center())
                r = rnode(k, grid, Center())

                horizontal_flux = sin(2π * x / Lx)
                horizontal_flux_gradient = (2π / Lx) * cos(2π * x / Lx)
                vertical_flux_gradient = -(π / Lz) * sin(π * r / Lz)
                expected_divergence = (∂x_σ(x) * horizontal_flux +
                                       σ(x) * horizontal_flux_gradient +
                                       vertical_flux_gradient) / σ(x)

                divergence = divᶜᶜᶜ(i, 1, k, grid, ρu, ρv, ρw̃)
                maximum_error = max(maximum_error, abs(divergence - expected_divergence))
            end

            return maximum_error
        end

        coarse_divergence_error = terrain_divergence_error(16, 8)
        medium_divergence_error = terrain_divergence_error(32, 16)
        fine_divergence_error = terrain_divergence_error(64, 32)

        @test medium_divergence_error < coarse_divergence_error / 2
        @test fine_divergence_error < medium_divergence_error / 2
    end

    @testset "UpperSponge uses terrain-following vertical coordinate" begin
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 500 * exp(-x^2 / 1000^2)
        materialize_terrain!(grid, h)

        sponge = UpperSponge(damping_rate=0.2, depth=2000, ramp=LinearRamp())
        δτᵐ⁺ = 3.0
        δτˢ⁻ = 2.0
        k = Nz
        i_flat = 1
        i_peak = Nx ÷ 2
        old_ρw = ZFaceField(grid)
        set!(old_ρw, 4)

        reference_z = rnode(k, grid, Face())
        expected_diag = δτᵐ⁺ * sponge.damping_rate *
                        sponge.ramp(reference_z, grid.Lz, sponge.depth)
        expected_rhs = δτˢ⁻ * sponge.damping_rate *
                       sponge.ramp(reference_z, grid.Lz, sponge.depth) * 4

        @test znode(i_flat, 1, k, grid, Center(), Center(), Face()) !=
              znode(i_peak, 1, k, grid, Center(), Center(), Face())
        @test sponge_term_diag(i_flat, 1, k, grid, sponge, δτᵐ⁺) ≈ expected_diag
        @test sponge_term_diag(i_peak, 1, k, grid, sponge, δτᵐ⁺) ≈ expected_diag
        @test sponge_rhs(i_flat, 1, k, grid, sponge, δτˢ⁻, old_ρw) ≈ expected_rhs
        @test sponge_rhs(i_peak, 1, k, grid, sponge, δτˢ⁻, old_ρw) ≈ expected_rhs
    end

    @testset "3D TFVD with non-Flat y: explicit step + ∂y operators" begin
        Nx, Ny, Nz = 8, 8, 6
        Lx, Ly, Lz = 10000.0, 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                          formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                               x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=z_faces,
                               topology=(Periodic, Periodic, Bounded))

        # 3D bell mountain — non-trivial slope in BOTH x and y so ∂y operators fire
        h₀, a = 200.0, 2000.0
        h(x, y) = h₀ / (1 + (x/a)^2 + (y/a)^2)^1.5
        materialize_terrain!(grid, h)
        metrics = build_terrain_metrics(grid, SlopeOutsideInterpolation())

        # Both ∂x_h and ∂y_h should have non-zero interior values
        @test maximum(abs, metrics.∂x_h) > 1e-3
        @test maximum(abs, metrics.∂y_h) > 1e-3

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature = 300)
        model = AtmosphereModel(grid; dynamics)
        ρᵢ(x, y, z) = adiabatic_hydrostatic_density(z, 101325.0, 300.0, 1e5,
                                                    model.thermodynamic_constants)
        set!(model, ρ=ρᵢ, θ=300, u=1.0, v=0.5)

        @test model.dynamics.terrain_metrics isa TerrainMetrics
        @test model.dynamics.contravariant_vertical_velocity !== nothing

        Δt = 0.1
        time_step!(model, Δt)
        @test isfinite(maximum(abs, model.velocities.w))
        @test isfinite(maximum(abs, model.velocities.v))
        @test isfinite(maximum(abs, model.dynamics.contravariant_vertical_velocity))
    end

    @testset "terrain_amg_operators: full chain-rule dispatch coverage" begin
        # Drives every AMG-mirror operator on a TerrainFollowingGrid: the slope operators
        # (`∂x_z*`, `∂y_z*`), the `c::Number` disambiguators, and both the
        # Field-arg and Function-arg chain-rule overloads across every
        # supported stagger. The chain-rule identity says that for a field
        # equal to physical altitude z, `(∂ϕ/∂x)|_z = 0` because
        # constant-altitude surfaces are flat in physical x by definition.
        # With SlopeOutsideInterpolation the operators cancel discretely to
        # machine precision, so we can assert this directly.

        Nx, Ny, Nz = 8, 8, 6
        Lx, Ly, Lz = 10000.0, 10000.0, 5000.0
        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                          formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                               x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=z_faces,
                               topology=(Periodic, Periodic, Bounded))
        materialize_terrain!(grid, (x, y) -> 200 * exp(-(x^2 + y^2) / 2000^2))

        # Pick an interior point well away from boundaries.
        i, j, k = 4, 4, 3
        tol = 1e-10 * Lz   # relative to typical znode magnitude

        # --- Slope operators (∂x_z*, ∂y_z*) — finite on a sloped grid ---
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

        # --- Number disambiguators: derivative of a constant is exactly zero ---
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

        # --- Function-arg chain-rule overloads: ϕ = znode(physical) → ≈ 0 ---
        # Operand stagger is X-flipped from the operator's result stagger:
        # ∂xᶠ** ⇒ operand at C in x; ∂xᶜ** ⇒ operand at F in x; same for ∂y**.
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

        # --- Field-arg chain-rule overloads: build a Field at each operand
        # stagger, fill with physical z via set!, assert ≈ 0. ---
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
            # set! with (x, y, z) -> z evaluates `z` at physical altitude via
            # the TFVD `node()` override.
            set!(ϕ, (x, y, z) -> z)
            v = op(i, j, k, grid, ϕ)
            @test abs(v) < tol
        end
    end

    @testset "on_architecture round-trip preserves materialised terrain" begin
        # Mirrors the CPU mirror that `set_to_function!` builds when called on a
        # GPU TFVD grid. Without `cpu_face_constructor_z(::TerrainFollowingGrid)` + the
        # materialised-arrays branch of `allocate_formulation`, the rebuild
        # discards `formulation.h`/`∂x_h`/`∂y_h` and the `node()` override
        # returns r instead of physical altitude.
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                          formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        h₀ = 300.0
        materialize_terrain!(grid, x -> h₀ * exp(-x^2 / 2000^2))

        cpu_arch = Oceananigans.Architectures.CPU()
        rebuilt = Oceananigans.Architectures.on_architecture(cpu_arch, grid)

        @test rebuilt.z isa TerrainFollowingVerticalDiscretization
        @test rebuilt.z.formulation.h !== nothing
        @test rebuilt.z.formulation.∂x_h !== nothing

        # Compare znode (physical altitude) at the surface (k=1, Face) and at
        # the top (k=Nz+1, Face) against the analytic terrain.
        @allowscalar for i in 1:Nx
            x = xnode(i, rebuilt, Center())
            h_expected = h₀ * exp(-x^2 / 2000^2)
            @test znode(i, 1, 1,    rebuilt, Center(), Center(), Face()) ≈ h_expected rtol=1e-10
            @test znode(i, 1, Nz+1, rebuilt, Center(), Center(), Face()) ≈ Lz          rtol=1e-10
        end
    end

    @testset "TwoLevelDecay precomputed decay basis matches analytic bₙ(r)" begin
        TFD = Breeze.TerrainFollowingDiscretization
        Nx, Nz = 8, 16
        Lz = 10000.0
        s₁, s₂ = 6000.0, 1500.0
        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz+1));
                      formulation = TwoLevelDecay(large_scale_height = s₁, small_scale_height = s₂))
        grid = RectilinearGrid(default_arch; size = (Nx, Nz), halo = (5, 5),
                               x = (0, 1000), z = z_faces, topology = (Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 100.0)
        f = grid.z.formulation
        z_top = f.z_top

        # The materialized basis must reproduce the analytic bₙ(r), bₙ′(r) at the
        # Center and Face reference nodes (same b evaluated at the same r).
        for k in 1:Nz
            rc = rnode(k, grid, Center())
            @test @allowscalar(f.basis.b₁ᶜ[1, 1, k])  ≈ TFD.b_two_level(rc, z_top, s₁)  rtol=1e-5
            @test @allowscalar(f.basis.b₂ᶜ[1, 1, k])  ≈ TFD.b_two_level(rc, z_top, s₂)  rtol=1e-5
            @test @allowscalar(f.basis.∂b₁ᶜ[1, 1, k]) ≈ TFD.b′_two_level(rc, z_top, s₁) rtol=1e-5
            @test @allowscalar(f.basis.∂b₂ᶜ[1, 1, k]) ≈ TFD.b′_two_level(rc, z_top, s₂) rtol=1e-5
        end
        for k in 1:Nz+1
            rf = rnode(k, grid, Face())
            @test @allowscalar(f.basis.b₁ᶠ[1, 1, k])  ≈ TFD.b_two_level(rf, z_top, s₁)  rtol=1e-5
            @test @allowscalar(f.basis.∂b₂ᶠ[1, 1, k]) ≈ TFD.b′_two_level(rf, z_top, s₂) rtol=1e-5
        end

        # b(r) decays from 1 at the surface (r=0) toward 0 at the model top.
        @test @allowscalar(f.basis.b₁ᶠ[1, 1, 1]) ≈ 1 rtol=1e-6
        @test abs(@allowscalar(f.basis.b₁ᶠ[1, 1, Nz+1])) < 1e-6
    end

end
end
