using Breeze
using Breeze.AtmosphereModels: transport_velocities, x_pressure_gradient
using Oceananigans.Architectures: CPU

# Run under `default_arch` when the test runner provides it (which routes to
# GPU when CUDA is functional, matching project convention); otherwise fall
# back to CPU() so this file can also be included directly with
# `julia --project=. test/terrain_following.jl` during development.
@isdefined(default_arch) || (default_arch = CPU())

using CUDA: @allowscalar

using Breeze.CompressibleEquations: assemble_slow_vertical_momentum_tendency!,
                                    compute_acoustic_substeps,
                                    compute_contravariant_velocity!,
                                    freeze_linearization_state!,
                                    linearized_pressure_perturbation,
                                    outer_step_start_transport_velocities,
                                    sponge_rhs,
                                    sponge_term_diag,
                                    terrain_horizontal_linearized_pressure_gradient_correction,
                                    z_linearized_pressure_gradient
using Breeze.TimeSteppers: compute_slow_momentum_tendencies!,
                           compute_slow_scalar_tendencies!
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: rnode, xnode, znode, λnode, φnode
using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, divᶜᶜᶜ, ∂zᶜᶜᶠ
using Breeze.Thermodynamics: hydrostatic_pressure, dry_air_gas_constant, vapor_gas_constant
using Test

# The terrain physics testsets that close the TwoLevelDecay-through-substepper gap are
# parametrized over both formulations. Both are skeleton instances (terrain components
# filled per-grid by materialize_terrain!), so the same constant is reused across grids.
const TERRAIN_FORMULATIONS = (LinearDecay(),
                              TwoLevelDecay(large_scale_height = 2500.0,
                                            small_scale_height = 1250.0))

@allowscalar begin
@testset "TerrainFollowingDiscretization" begin
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

    @testset "Split-explicit zero terrain matches height coordinates" begin
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0

        function flat_split_explicit_model(terrain; damping=NoDivergenceDamping())
            # `terrain=false`: plain RectilinearGrid (no TFVD) → terrain_metrics === nothing
            #                  via the auto-build path (only triggers on TFVD grids).
            # `terrain=true`:  TFVD grid with h ≡ 0 → metrics auto-built; terrain physics
            #                  runs but the slope factor is zero, so output should match.
            grid = if terrain
                z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
                g = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                                    x=(-Lx/2, Lx/2), z=z_faces,
                                    topology=(Periodic, Flat, Bounded))
                materialize_terrain!(g, x -> 0)
                g
            else
                RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                                x=(-Lx/2, Lx/2), z=(0, Lz),
                                topology=(Periodic, Flat, Bounded))
            end
            time_discretization = SplitExplicitTimeDiscretization(substeps=6,
                                                                  damping=damping)
            dynamics = CompressibleDynamics(time_discretization)
            model = AtmosphereModel(grid; dynamics)
            set!(model,
                 ρ=1,
                 θ=300,
                 u=(x, z) -> 0.1 * sin(2π * x / Lx),
                 w=(x, z) -> 0.01 * sin(π * z / Lz))
            return model
        end

        height_substeps = compute_acoustic_substeps(flat_split_explicit_model(false).grid,
                                                    0.01, ThermodynamicConstants(), 0.5)
        terrain_substeps = compute_acoustic_substeps(flat_split_explicit_model(true).grid,
                                                     0.01, ThermodynamicConstants(), 0.5)
        @test height_substeps == terrain_substeps

        function assert_matching_prognostics(height_model, terrain_model)
            height_ρθ = Breeze.AtmosphereModels.thermodynamic_density(height_model.formulation)
            terrain_ρθ = Breeze.AtmosphereModels.thermodynamic_density(terrain_model.formulation)
            # The "height" branch uses a plain RectilinearGrid while the "terrain"
            # branch uses TFVD with h ≡ 0. The two grid types route through different
            # spacing/operator code paths whose floating-point results agree only to
            # ~machine precision (~10⁻²³ here in normalised units), so the comparison
            # tolerance is machine-precision-floor, not the exact-zero bound used when
            # both branches were the same TFVD grid.
            zero_terrain_tolerance = 1e-14

            @test isapprox(interior(height_model.dynamics.density),
                           interior(terrain_model.dynamics.density);
                           atol=zero_terrain_tolerance)
            @test isapprox(interior(height_ρθ),
                           interior(terrain_ρθ);
                           atol=zero_terrain_tolerance)
            @test isapprox(interior(height_model.momentum.ρu),
                           interior(terrain_model.momentum.ρu);
                           atol=zero_terrain_tolerance)
            @test isapprox(interior(height_model.momentum.ρv),
                           interior(terrain_model.momentum.ρv);
                           atol=zero_terrain_tolerance)
            @test isapprox(interior(height_model.momentum.ρw),
                           interior(terrain_model.momentum.ρw);
                           atol=zero_terrain_tolerance)
        end

        # One-step increment equivalence.
        height_model = flat_split_explicit_model(false)
        terrain_model = flat_split_explicit_model(true)
        time_step!(height_model, 0.01)
        time_step!(terrain_model, 0.01)
        assert_matching_prognostics(height_model, terrain_model)

        # Ten-step trajectory equivalence.
        height_model = flat_split_explicit_model(false)
        terrain_model = flat_split_explicit_model(true)
        for _ in 1:10
            time_step!(height_model, 0.01)
            time_step!(terrain_model, 0.01)
        end
        assert_matching_prognostics(height_model, terrain_model)

        # The vertical divergence damping path modifies the implicit column
        # coefficients; it should still reduce exactly to height coordinates
        # when the terrain is flat.
        damping = ThermalDivergenceDamping(coefficient=0.05, damp_vertical=true)
        height_model = flat_split_explicit_model(false; damping)
        terrain_model = flat_split_explicit_model(true; damping)
        time_step!(height_model, 0.01)
        time_step!(terrain_model, 0.01)
        assert_matching_prognostics(height_model, terrain_model)
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
        dynamics = CompressibleDynamics(ExplicitTimeStepping())
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
            dynamics = CompressibleDynamics(ExplicitTimeStepping(); slope_stencil = stencil)
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
            dynamics = CompressibleDynamics(ExplicitTimeStepping(); slope_stencil = SlopeInsideInterpolation())
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

    @testset "Split-explicit terrain transport uses w̃ [$(nameof(typeof(formulation)))]" for formulation in TERRAIN_FORMULATIONS
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation)
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 100 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; dynamics)
        set!(model, θ=300, ρ=model.dynamics.terrain_reference_density, u=0, w=0)

        @test model.timestepper isa AcousticRungeKutta3
        @test transport_velocities(model).w === model.timestepper.substepper.time_averaged_velocities.w

        substepper = model.timestepper.substepper
        freeze_linearization_state!(substepper, model)
        compute_slow_momentum_tendencies!(model)
        compute_slow_scalar_tendencies!(model)
        assemble_slow_vertical_momentum_tendency!(substepper, model)
        @test isapprox(maximum(abs, interior(substepper.slow_vertical_momentum_tendency)), 0; atol = 1e-12)

        initial_mass = sum(interior(model.dynamics.density))
        time_step!(model, 0.1)
        final_mass = sum(interior(model.dynamics.density))

        w̃ = model.dynamics.contravariant_vertical_velocity
        ρw̃ = model.dynamics.contravariant_vertical_momentum
        w̃_transport = transport_velocities(model).w

        @test isfinite(maximum(abs, model.velocities.w))
        @test isfinite(maximum(abs, w̃))
        @test isfinite(maximum(abs, ρw̃))
        @test isfinite(maximum(abs, w̃_transport))
        @test isapprox(maximum(abs, interior(model.velocities.w)), 0; atol = 1e-12)
        @test isapprox(maximum(abs, interior(w̃)), 0; atol = 1e-12)
        @test abs(final_mass - initial_mass) / initial_mass <= 1e-13

        # The kinematic terrain BC makes the contravariant w̃ vanish at the surface.
        # With the boundary-condition approach this is a computed cancellation
        # (ρw̃ = ρw - slope·ρu with ρw|₁ = slope·ρu): exact in CPU Float64, but at
        # machine epsilon on GPU (non-associative FP / FMA), so compare ≈ 0.
        for i in 1:Nx
            @test isapprox(w̃[i, 1, 1], 0; atol = 1e-12)
            @test isapprox(ρw̃[i, 1, 1], 0; atol = 1e-12)
            @test isapprox(w̃_transport[i, 1, 1], 0; atol = 1e-12)
        end

        bottom_w̃ = [w̃[i, 1, 1] for i in 1:Nx]
        bottom_ρw̃ = [ρw̃[i, 1, 1] for i in 1:Nx]
        fill_halo_regions!(w̃, ρw̃, w̃_transport)
        compute_contravariant_velocity!(model)

        @test sum(abs, [ρw̃[i, 1, 1] for i in 1:Nx]) / initial_mass <= 1e-13
        @test [w̃[i, 1, 1] for i in 1:Nx] == bottom_w̃
        @test [ρw̃[i, 1, 1] for i in 1:Nx] == bottom_ρw̃
    end

    @testset "Adaptive acoustic substeps with terrain metrics" begin
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0

        function adaptive_grid(terrain)
            z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
            grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                                   x=(-Lx/2, Lx/2), z=z_faces,
                                   topology=(Periodic, Flat, Bounded))
            terrain && materialize_terrain!(grid, x -> 100 * exp(-x^2 / 2000^2))
            return grid
        end

        height_grid = adaptive_grid(false)
        terrain_grid = adaptive_grid(true)
        constants = ThermodynamicConstants()
        Δt = 0.1
        acoustic_cfl = 0.5

        height_substeps = compute_acoustic_substeps(height_grid, Δt, constants, acoustic_cfl)
        terrain_substeps = compute_acoustic_substeps(terrain_grid, Δt, constants, acoustic_cfl)
        @test terrain_substeps == height_substeps
        @test terrain_substeps ≥ 1

        model_grid = adaptive_grid(false)
        materialize_terrain!(model_grid, x -> 100 * exp(-x^2 / 2000^2))
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(acoustic_cfl=acoustic_cfl);
                                        reference_potential_temperature=300)
        model = AtmosphereModel(model_grid; dynamics)
        set!(model, θ=300, ρ=model.dynamics.terrain_reference_density, u=0, w=0)

        @test model.timestepper.substepper.substeps === nothing
        time_step!(model, Δt)
        @test model.clock.iteration == 1
        @test isfinite(maximum(abs, interior(model.velocities.w)))
        @test isapprox(maximum(abs, interior(model.velocities.w)), 0; atol = 1e-12)
    end

    @testset "Split-explicit terrain acoustic stability diagnostics [$(nameof(typeof(formulation)))]" for formulation in TERRAIN_FORMULATIONS
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0
        Δt = 2.0
        acoustic_cfl = 0.5
        constants = ThermodynamicConstants()
        Rᵈ = Breeze.dry_air_gas_constant(constants)
        cᵖᵈ = constants.dry_air.heat_capacity
        γᵈ = cᵖᵈ / (cᵖᵈ - Rᵈ)
        sound_speed = sqrt(γᵈ * Rᵈ * 300)

        for h₀ in (0.0, 100.0, 300.0)
            z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation)
            grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                                   x=(-Lx/2, Lx/2), z=z_faces,
                                   topology=(Periodic, Flat, Bounded))
            materialize_terrain!(grid, x -> h₀ * exp(-x^2 / 2000^2))

            dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(acoustic_cfl=acoustic_cfl);
                                            reference_potential_temperature=300)
            model = AtmosphereModel(grid; dynamics)
            set!(model,
                 θ=300,
                 ρ=model.dynamics.terrain_reference_density,
                 u=(x, z) -> 1 + 0.1 * sin(2π * x / Lx),
                 w=0)

            Nτ = compute_acoustic_substeps(grid, Δt, constants, acoustic_cfl)
            acoustic_CFL = (Δt / Nτ) * sound_speed / (Lx / Nx)
            @test acoustic_CFL <= acoustic_cfl

            maximum_advective_CFL = 0.0
            maximum_contravariant_CFL = 0.0

            for _ in 1:3
                time_step!(model, Δt)
                w̃ = model.dynamics.contravariant_vertical_velocity

                advective_CFL = Δt * maximum(abs, interior(model.velocities.u)) / (Lx / Nx)
                contravariant_CFL = Δt * maximum(abs, interior(w̃)) / (Lz / Nz)
                maximum_advective_CFL = max(maximum_advective_CFL, advective_CFL)
                maximum_contravariant_CFL = max(maximum_contravariant_CFL, contravariant_CFL)

                @test isfinite(maximum(abs, interior(model.dynamics.density)))
                @test isfinite(maximum(abs, interior(model.momentum.ρu)))
                @test isfinite(maximum(abs, interior(model.momentum.ρw)))
                @test isfinite(maximum(abs, interior(w̃)))
            end

            @test maximum_advective_CFL < acoustic_cfl
            @test maximum_contravariant_CFL < acoustic_cfl
        end
    end

    @testset "Cheap bell mountain-wave response smoke" begin
        Nx, Nz = 8, 4
        Lx, Lz = 100e3, 15e3
        Δt = 2.0
        U = 20.0
        N = 0.01
        h₀ = 1.0
        a = 10e3
        θ₀ = 300.0
        p₀ = 100000.0
        pˢᵗ = 1e5

        constants = ThermodynamicConstants(Float64)
        g = constants.gravitational_acceleration
        cᵖᵈ = constants.dry_air.heat_capacity
        Rᵈ = Breeze.dry_air_gas_constant(constants)
        β = g / (Rᵈ * θ₀)
        N² = N^2
        k★ = sqrt(max(0, N² / U^2 - β^2 / 4))
        horizontal_wavelength = 2π / k★

        θ_of_z(z) = θ₀ * exp(N² * z / g)
        hill(x) = h₀ / (1 + (x / a)^2)
        ĥ(k) = π * a * h₀ * exp(-a * abs(k))
        m²(k) = N² / U^2 - β^2 / 4 - k^2

        function w_linear(x, z; nk = 64)
            k_max = max(10 / a, 10 * k★)
            k = range(0, k_max, length = nk)
            Δk = step(k)

            integral = zero(Float64)
            for n in eachindex(k)
                kⁿ = k[n]
                weight = (n == firstindex(k) || n == lastindex(k)) ? 0.5 : 1.0
                m²ⁿ = m²(kⁿ)
                m_abs = sqrt(abs(m²ⁿ))
                phase = ifelse(m²ⁿ >= 0,
                               sin(m_abs * z + kⁿ * x),
                               exp(-m_abs * z) * sin(kⁿ * x))
                integral += weight * kⁿ * ĥ(kⁿ) * phase
            end

            return -(U / π) * exp(β * z / 2) * Δk * integral
        end

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, hill)

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(acoustic_cfl=0.5,
                                                                        sponge=UpperSponge(damping_rate=0.1,
                                                                                           depth=Lz/3));
                                        slope_stencil = SlopeInsideInterpolation(),
                                        reference_potential_temperature=θ_of_z,
                                        surface_pressure=p₀,
                                        standard_pressure=pˢᵗ)
        model = AtmosphereModel(grid; dynamics,
                                thermodynamic_constants=constants)

        set!(model,
             ρ = model.dynamics.terrain_reference_density,
             θ = (x, z) -> θ_of_z(z),
             u = U,
             v = 0,
             w = 0,
             enforce_mass_conservation = false)

        time_step!(model, Δt)

        squared_error = 0.0
        squared_reference = 0.0
        maximum_w = 0.0
        maximum_reference_w = 0.0
        maximum_w_x = 0.0
        maximum_reference_w_x = 0.0

        for i in 1:Nx, k in 2:Nz
            z = znode(i, 1, k, grid, Center(), Center(), Face())
            z > 0.75Lz && continue
            x = xnode(i, grid, Center())
            simulated_w = model.velocities.w[i, 1, k]
            reference_w = w_linear(x, z)

            squared_error += (simulated_w - reference_w)^2
            squared_reference += reference_w^2

            if abs(simulated_w) > maximum_w
                maximum_w = abs(simulated_w)
                maximum_w_x = x
            end

            if abs(reference_w) > maximum_reference_w
                maximum_reference_w = abs(reference_w)
                maximum_reference_w_x = x
            end
        end

        normalized_rmse = sqrt(squared_error / max(eps(), squared_reference))
        amplitude_error = abs(maximum_w - maximum_reference_w) / max(eps(), maximum_reference_w)
        phase_error_wavelengths = abs(maximum_w_x - maximum_reference_w_x) / horizontal_wavelength

        @test isfinite(normalized_rmse)
        @test isfinite(amplitude_error)
        @test isfinite(phase_error_wavelengths)
        @test maximum_w > 0
        @test maximum_reference_w > 0
        @test normalized_rmse < 1.2
        @test amplitude_error < 1.0
        # The phase metric uses the location of a single maximum on an 8×4
        # smoke grid, so it is much less robust than the RMSE/amplitude checks.
        @test phase_error_wavelengths < 2.5
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

    @testset "Terrain acoustic substepper supports vertical divergence damping" begin
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 100 * exp(-x^2 / 2000^2))

        damping = ThermalDivergenceDamping(coefficient=0.05, damp_vertical=true)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=4,
                                                                        damping=damping);
                                        slope_stencil = SlopeInsideInterpolation(),
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; dynamics)
        set!(model,
             θ=300,
             ρ=model.dynamics.terrain_reference_density,
             u=(x, z) -> 1 + 1e-3 * sin(2π * x / Lx),
             v=0,
             w=0)

        time_step!(model, 0.1)
        compute_contravariant_velocity!(model)

        @test isfinite(maximum(abs, model.velocities.u))
        @test isfinite(maximum(abs, model.velocities.w))
        @test isfinite(maximum(abs, model.dynamics.contravariant_vertical_velocity))

        # w̃ = 0 at the surface is a computed cancellation under the kinematic terrain
        # BC (exact in CPU Float64, machine-epsilon on GPU), so compare ≈ 0.
        for i in 1:Nx
            @test isapprox(model.dynamics.contravariant_vertical_velocity[i, 1, 1], 0; atol = 1e-12)
            @test isapprox(model.dynamics.contravariant_vertical_momentum[i, 1, 1], 0; atol = 1e-12)
        end
    end

    @testset "Split-explicit terrain supports slope-inside pressure-gradient stencil" begin
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 100 * exp(-x^2 / 2000^2))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                        slope_stencil = SlopeInsideInterpolation(),
                                        reference_potential_temperature=300)
        model = AtmosphereModel(grid; dynamics)
        set!(model,
             θ=300,
             ρ=model.dynamics.terrain_reference_density,
             u=(x, z) -> 0.1 * sin(2π * x / Lx),
             w=0)

        @test model.dynamics.terrain_metrics.pressure_gradient_stencil isa SlopeInsideInterpolation
        time_step!(model, 0.01)
        @test model.clock.iteration == 1
        @test isfinite(maximum(abs, interior(model.momentum.ρu)))
        @test isfinite(maximum(abs, interior(model.dynamics.density)))
    end

    @testset "Acoustic substep gates terrain ρw̃ slope correction [$(nameof(typeof(formulation)))]" for formulation in TERRAIN_FORMULATIONS
        # The contravariant vertical-momentum perturbation ρw̃ = ρw − slopeₓ·ρu − slopeᵧ·ρv
        # carries a horizontal slope correction slopeₓ·∂ₓ(Cᴸ(ρθ)′) in its acoustic
        # pressure-gradient force. Because ρw̃ and ρu are tied by that relation, the
        # correction must respect the SAME MPAS first-small-step gate that
        # `_explicit_horizontal_step!` applies to ρu's perturbation PGF — otherwise the
        # two are out of phase on substep 1 of a multi-substep stage. The gate factor
        # therefore scales ONLY the horizontal slope correction inside
        # `z_linearized_pressure_gradient`; the vertical ∂z(Cᴸ(ρθ)′) part is always
        # applied (the vertical acoustic mode is solved implicitly every substep).
        Nx, Nz = 16, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation)
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 300 * exp(-x^2 / 2000^2))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                        slope_stencil = SlopeOutsideInterpolation(),
                                        reference_potential_temperature = 300)
        model = AtmosphereModel(grid; dynamics)
        d = model.dynamics

        # Linearization coefficients and a non-constant ρθ′ perturbation (varies in x
        # so the horizontal slope correction is genuinely nonzero over the mountain).
        ρθ′  = CenterField(grid)
        Πᴸ   = CenterField(grid)
        γRᵐᴸ = CenterField(grid)
        for i in 1:Nx, k in 1:Nz
            x = xnode(i, grid, Center())
            z = znode(i, 1, k, grid, Center(), Center(), Center())
            ρθ′[i, 1, k]  = sin(2π * x / Lx) * (1 + z / Lz)
            Πᴸ[i, 1, k]   = 1.0
            γRᵐᴸ[i, 1, k] = 287.0 * 1.4
        end
        fill_halo_regions!(ρθ′)
        fill_halo_regions!(Πᴸ)
        fill_halo_regions!(γRᵐᴸ)

        correction_seen = false
        for i in 3:Nx-2, k in 2:Nz-1
            ∂z_p′      = ∂zᶜᶜᶠ(i, 1, k, grid, linearized_pressure_perturbation, ρθ′, Πᴸ, γRᵐᴸ)
            correction = terrain_horizontal_linearized_pressure_gradient_correction(i, 1, k, grid, d, ρθ′, Πᴸ, γRᵐᴸ)

            z_gated = z_linearized_pressure_gradient(i, 1, k, grid, d, ρθ′, Πᴸ, γRᵐᴸ, 0.0)
            z_full  = z_linearized_pressure_gradient(i, 1, k, grid, d, ρθ′, Πᴸ, γRᵐᴸ, 1.0)

            # Gate off ⇒ pure vertical gradient, no horizontal slope correction.
            @test z_gated == ∂z_p′
            # Gate on ⇒ vertical gradient minus the full horizontal slope correction.
            @test z_full == ∂z_p′ - correction

            correction_seen = correction_seen || (abs(correction) > 1e-10)
        end

        # The gate is meaningful: the slope correction is genuinely nonzero over the mountain.
        @test correction_seen
    end

    @testset "Terrain reference state matches continuous hydrostatic profile" begin
        # The terrain reference state pᵣ(i,j,k) must equal the continuous
        # hydrostatic pressure evaluated at the local physical height z(i,j,k).
        # A bug that initializes every column from sea-level pressure creates
        # O(ρgh) errors over terrain.
        Nx, Nz = 16, 8
        Lx, Lz = 100000.0, 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 1000.0
        a = 10000.0
        h(x) = h₀ * exp(-x^2 / a^2)
        materialize_terrain!(grid, h)

        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature=θ₀)
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants

        pᵣ = model.dynamics.terrain_reference_pressure

        # At each grid point, pᵣ should match the continuous profile
        # to within the discretization error of the Exner integration (O(Δz²))
        for i in 1:Nx, k in 1:Nz
            z = znode(i, 1, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
            # Discrete Exner integration has O(Δz²) error; with Δz ≈ 1250 m
            # the accumulated error at the top is ~0.5%, so use 1% tolerance
            @test pᵣ[i, 1, k] ≈ p_exact rtol=1e-2
        end

        # Critical check: at a given k-level, pᵣ must NOT be constant across
        # columns (it should vary because physical heights differ). But at the
        # SAME physical height, values from different columns should agree closely.
        # Compare the flat column (i at domain edge) vs the mountain-top column.
        i_flat = 1    # far from mountain
        i_peak = Nx÷2 # near mountain peak
        z_flat_1 = znode(i_flat, 1, 1, grid, Center(), Center(), Center())
        z_peak_1 = znode(i_peak, 1, 1, grid, Center(), Center(), Center())

        # Physical heights differ, so pᵣ at k=1 should differ
        @test z_peak_1 > z_flat_1 + 100  # mountain is at least 100 m higher
        @test pᵣ[i_peak, 1, 1] < pᵣ[i_flat, 1, 1]  # higher altitude → lower pressure
    end

    @testset "Terrain reference state with θ(z) profile (Function dispatch)" begin
        # Same test but with a non-constant potential temperature profile,
        # exercising the numerically_integrated_hydrostatic_pressure path.
        Nx, Nz = 16, 16
        Lx, Lz = 100000.0, 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 1000.0
        a = 10000.0
        h(x) = h₀ * exp(-x^2 / a^2)
        materialize_terrain!(grid, h)

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

        pᵣ = model.dynamics.terrain_reference_pressure

        # At each grid point, pᵣ should match the continuous profile
        for i in 1:Nx, k in 1:Nz
            z = znode(i, 1, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z, p₀, θ_of_z, pˢᵗ, constants)
            # Finer grid (Nz=16) so tighter tolerance than Nz=8 test
            @test pᵣ[i, 1, k] ≈ p_exact rtol=5e-3
        end

        # Mountain-top column should have lower pᵣ at k=1 than flat column
        i_flat = 1
        i_peak = Nx÷2
        @test pᵣ[i_peak, 1, 1] < pᵣ[i_flat, 1, 1]
    end

    #####
    ##### 3D TFVD (non-Flat y) — exercises the y-direction chain-rule operators
    ##### (`∂y_zᶜᶠᶜ`, `∂yᶜᶠᶜ`, etc.) and 3D node()/contravariant-velocity paths
    ##### that 2D (Periodic, Flat, Bounded) tests cannot reach.
    #####

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

    @testset "3D TFVD with non-Flat y: split-explicit + ∂y pressure gradient" begin
        Nx, Ny, Nz = 8, 8, 6
        Lx, Ly, Lz = 10000.0, 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                          formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                               x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=z_faces,
                               topology=(Periodic, Periodic, Bounded))
        materialize_terrain!(grid, (x, y) -> 100 * exp(-(x^2 + y^2) / 2000^2))

        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=6);
                                        reference_potential_temperature = 300)
        model = AtmosphereModel(grid; dynamics)
        ρᵢ(x, y, z) = adiabatic_hydrostatic_density(z, 101325.0, 300.0, 1e5,
                                                    model.thermodynamic_constants)
        set!(model, ρ=ρᵢ, θ=300,
             u=(x, y, z) -> 0.1 * sin(2π * x / Lx),
             v=(x, y, z) -> 0.1 * sin(2π * y / Ly))

        # Step the substepper. This routes through both ∂xᶠᶜᶜ and ∂yᶜᶠᶜ chain-rule
        # operators (the y branch is unreachable on the 2D Flat-y tests above).
        time_step!(model, 0.1)
        @test isfinite(maximum(abs, model.velocities.w))
        @test isfinite(maximum(abs, model.velocities.v))
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

    #####
    ##### LatitudeLongitudeGrid terrain following
    #####
    ##### These verify that terrain-following coordinates work correctly on
    ##### spherical grids where ξnode → λnode and ηnode → φnode.
    #####

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
        set!(model, θ=300, ρ=model.dynamics.terrain_reference_density, u=0, w=0)

        @test model.dynamics.terrain_metrics.pressure_gradient_stencil isa SlopeInsideInterpolation
        @test model.timestepper.substepper.substeps === nothing  # adaptive

        initial_mass = sum(interior(model.dynamics.density))
        time_step!(model, 0.1)
        final_mass = sum(interior(model.dynamics.density))

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

        @test isapprox(interior(height_model.dynamics.density),
                       interior(terrain_model.dynamics.density); atol=tol)
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
        pᵣ = model.dynamics.terrain_reference_pressure

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

    @testset "Constant moist terrain reference state satisfies discrete hydrostatic balance" begin
        Nx, Nz = 8, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 200 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)

        θ_reference = 300.0
        qᵛ_reference = 0.012

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature = θ_reference,
                                        reference_vapor_mass_fraction = qᵛ_reference)
        model = AtmosphereModel(grid; dynamics)

        p_ref = model.dynamics.terrain_reference_pressure
        ρ_ref = model.dynamics.terrain_reference_density
        constants = model.thermodynamic_constants
        g = constants.gravitational_acceleration
        p₀ = dynamics.surface_pressure
        pˢᵗ = dynamics.standard_pressure
        Rᵈ = dry_air_gas_constant(constants)
        Rᵛ = vapor_gas_constant(constants)
        cᵖᵈ = constants.dry_air.heat_capacity
        cᵖᵛ = constants.vapor.heat_capacity

        @test p_ref !== nothing
        @test ρ_ref !== nothing

        qᵛ_surface = qᵛ_reference
        qᵈ_surface = 1 - qᵛ_surface
        Rᵐ_surface = qᵈ_surface * Rᵈ + qᵛ_surface * Rᵛ
        cᵖᵐ_surface = qᵈ_surface * cᵖᵈ + qᵛ_surface * cᵖᵛ
        κ_surface = Rᵐ_surface / cᵖᵐ_surface
        T_surface₀ = θ_reference * (p₀ / pˢᵗ)^κ_surface

        for i in 1:Nx
            z_surface = znode(i, 1, 1, grid, Center(), Center(), Face())
            p_surface = p₀ * (1 - g * z_surface / (cᵖᵐ_surface * T_surface₀))^(cᵖᵐ_surface / Rᵐ_surface)
            T_surface = θ_reference * (p_surface / pˢᵗ)^κ_surface
            ρ_surface = p_surface / (Rᵐ_surface * T_surface)

            # Surface (bottom face) to first cell center spans half a cell.
            hydrostatic_residual = (p_ref[i, 1, 1] - p_surface) / (Δzᶜᶜᶜ(i, 1, 1, grid) / 2) +
                                   g * (ρ_ref[i, 1, 1] + ρ_surface) / 2
            @test abs(hydrostatic_residual) <= 1e-6
        end

        for i in 1:Nx, k in 2:Nz
            hydrostatic_residual = (p_ref[i, 1, k] - p_ref[i, 1, k - 1]) / Δzᶜᶜᶠ(i, 1, k, grid) +
                                   g * (ρ_ref[i, 1, k] + ρ_ref[i, 1, k - 1]) / 2
            @test abs(hydrostatic_residual) <= 1e-8
        end

        i_flat = 1
        i_peak = Nx ÷ 2
        @test p_ref[i_peak, 1, 1] < p_ref[i_flat, 1, 1]
    end

    @testset "Variable moist terrain reference state satisfies interior discrete hydrostatic balance" begin
        Nx, Nz = 8, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 200 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)

        θ_reference(z) = 300.0 + 0.01 * z
        qᵛ_reference(z) = 0.012 * exp(-z / 1000)

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature = θ_reference,
                                        reference_vapor_mass_fraction = qᵛ_reference)
        model = AtmosphereModel(grid; dynamics)

        p_ref = model.dynamics.terrain_reference_pressure
        ρ_ref = model.dynamics.terrain_reference_density
        g = model.thermodynamic_constants.gravitational_acceleration

        @test p_ref !== nothing
        @test ρ_ref !== nothing

        for i in 1:Nx, k in 2:Nz
            hydrostatic_residual = (p_ref[i, 1, k] - p_ref[i, 1, k - 1]) / Δzᶜᶜᶠ(i, 1, k, grid) +
                                   g * (ρ_ref[i, 1, k] + ρ_ref[i, 1, k - 1]) / 2
            @test abs(hydrostatic_residual) <= 1e-8
        end

        i_flat = 1
        i_peak = Nx ÷ 2
        @test p_ref[i_peak, 1, 1] < p_ref[i_flat, 1, 1]
    end
end
end
