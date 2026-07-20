using Breeze
using Breeze.AtmosphereModels: transport_velocities
using Oceananigans.Architectures: CPU

# Run under `default_arch` when the test runner provides it (which routes to
# GPU when CUDA is functional, matching project convention); otherwise fall
# back to CPU() so this file can also be included directly with
# `julia --project=. test/<this file>.jl` during development.
@isdefined(default_arch) || (default_arch = CPU())

using CUDA: @allowscalar

using Breeze.CompressibleEquations: assemble_slow_vertical_momentum_tendency!, compute_acoustic_substeps, compute_contravariant_velocity!, freeze_linearization_state!, δpᴸ, terrain_horizontal_linearized_pressure_gradient_correction, ∇ᶻp′
using Breeze.TimeSteppers: compute_slow_momentum_tendencies!, compute_slow_scalar_tendencies!
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: xnode, znode
using Oceananigans.Operators: ∂zᶜᶜᶠ
using Breeze.Thermodynamics: dry_air_gas_constant
using Test

# The terrain physics testsets that close the TwoLevelDecay-through-substepper gap are
# parametrized over both formulations. Both are skeleton instances (terrain components
# filled per-grid by materialize_terrain!), so the same constant is reused across grids.
const TERRAIN_FORMULATIONS = (LinearDecay(),
                              TwoLevelDecay(large_scale_height = 2500.0,
                                            small_scale_height = 1250.0))

@allowscalar begin
@testset "TerrainFollowing split-explicit dynamics" begin
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
            # Isolate the pure slope-term reduction: disable the terrain reference (on by default
            # for terrain grids) so both branches difference the full pressure and the flat
            # (h ≡ 0) terrain path matches the height path to machine precision.
            dynamics = CompressibleDynamics(time_discretization; terrain_reference=false)
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

            @test isapprox(interior(height_model.dynamics.dry_density),
                           interior(terrain_model.dynamics.dry_density);
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

        initial_mass = sum(interior(model.dynamics.dry_density))
        time_step!(model, 0.1)
        final_mass = sum(interior(model.dynamics.dry_density))

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

                @test isfinite(maximum(abs, interior(model.dynamics.dry_density)))
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
        @test isfinite(maximum(abs, interior(model.dynamics.dry_density)))
    end

    @testset "Acoustic substep gates terrain ρw̃ slope correction [$(nameof(typeof(formulation)))]" for formulation in TERRAIN_FORMULATIONS
        # The contravariant vertical-momentum perturbation ρw̃ = ρw − slopeₓ·ρu − slopeᵧ·ρv
        # carries a horizontal slope correction slopeₓ·∂ₓ(Cᴸ(ρθ)′) in its acoustic
        # pressure-gradient force. Because ρw̃ and ρu are tied by that relation, the
        # correction must respect the SAME MPAS first-small-step gate that
        # `_explicit_horizontal_step!` applies to ρu's perturbation PGF — otherwise the
        # two are out of phase on substep 1 of a multi-substep stage. The gate factor
        # therefore scales ONLY the horizontal slope correction inside
        # `∇ᶻp′`; the vertical ∂z(Cᴸ(ρθ)′) part is always
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
            ∂z_p′      = ∂zᶜᶜᶠ(i, 1, k, grid, δpᴸ, ρθ′, Πᴸ, γRᵐᴸ)
            correction = terrain_horizontal_linearized_pressure_gradient_correction(i, 1, k, grid, d, ρθ′, Πᴸ, γRᵐᴸ)

            z_gated = ∇ᶻp′(i, 1, k, grid, d, ρθ′, Πᴸ, γRᵐᴸ, 0.0)
            z_full  = ∇ᶻp′(i, 1, k, grid, d, ρθ′, Πᴸ, γRᵐᴸ, 1.0)

            # Gate off ⇒ pure vertical gradient, no horizontal slope correction.
            @test z_gated == ∂z_p′
            # Gate on ⇒ vertical gradient minus the full horizontal slope correction.
            @test z_full == ∂z_p′ - correction

            correction_seen = correction_seen || (abs(correction) > 1e-10)
        end

        # The gate is meaningful: the slope correction is genuinely nonzero over the mountain.
        @test correction_seen
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

end
end
