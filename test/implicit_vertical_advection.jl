using Breeze
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization, needs_implicit_solver
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: volume
using Oceananigans.Solvers: BatchedTridiagonalSolver
using Oceananigans.TimeSteppers: implicit_step!
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, VerticallyImplicitTimeDiscretization
using Oceananigans.Units: kilometers
using Breeze.CompressibleEquations: CompressibleDynamics, SplitExplicitTimeDiscretization
using Test

import Breeze.AtmosphereModels as AM

@testset "Adaptive implicit vertical advection [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 16), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)
    dynamics = AnelasticDynamics(reference_state)

    aiva() = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5))

    @testset "z-Face implicit solve (ρw): explicit limit, conservation, positivity" begin
        # Pointwise/host checks of the Breeze-owned z-Face coefficients, evaluated on a CPU grid.
        cpu_grid = RectilinearGrid(CPU(); size=(1, 1, 16), x=(0, 100), y=(0, 100), z=(0, 1000))
        Δt = FT(50)
        td = AdaptiveVerticallyImplicitDiscretization(FT; cfl=FT(0.3))
        scheme = WENO(FT; time_discretization=td)

        # Strong interior updraft, vanishing near the boundaries so interior fluxes telescope.
        w = Field{Center, Center, Face}(cpu_grid)
        set!(w, (x, y, z) -> 8 * sinpi(z / 1000)^2)
        fill_halo_regions!(w)

        ρ = CenterField(cpu_grid)
        set!(ρ, (x, y, z) -> 1.2 * exp(-z / 500))
        fill_halo_regions!(ρ)

        solver = implicit_diffusion_solver(VerticallyImplicitTimeDiscretization(), cpu_grid)
        clock = Clock(cpu_grid)
        Nz = size(cpu_grid, 3)

        q₀(x, y, z) = exp(-(z - 500)^2 / (2 * 150^2))
        column_momentum(q) = sum(volume(1, 1, k, cpu_grid, Center(), Center(), Face()) * q[1, 1, k] for k in 1:Nz)

        # Explicit limit: with the vertical CFL below target everywhere the implicit velocity
        # vanishes and the solve reduces to the identity.
        q = Field{Center, Center, Face}(cpu_grid)
        set!(q, q₀)
        fill_halo_regions!(q)
        td.Δt[] = FT(1//1000)
        before = Array(interior(q))
        implicit_step!(q, solver, nothing, nothing, nothing, clock, (;), FT(1//1000),
                       AM.VerticalMomentumImplicitAdvection(scheme), (; w), ρ)
        @test Array(interior(q)) == before

        # Strong splitting: the density-weighted upwind system must stay finite, conserve the
        # column momentum ∑ Vᶜᶜᶠ ρw (interior fluxes telescope; w ≈ 0 at boundary-adjacent
        # centers), preserve positivity (I - ΔtL is an M-matrix, so its inverse is
        # nonnegative), and transport momentum in the upwind direction — upward here, since
        # the advecting velocity is an updraft. (No max principle applies: flux-form
        # transport into thinner air legitimately amplifies the specific velocity ρw/ρ.)
        set!(q, q₀)
        fill_halo_regions!(q)
        td.Δt[] = Δt
        momentum₀ = column_momentum(q)
        z_face(k) = Oceananigans.Grids.znode(1, 1, k, cpu_grid, Center(), Center(), Face())
        momentum_height(q) = sum(volume(1, 1, k, cpu_grid, Center(), Center(), Face()) * q[1, 1, k] * z_face(k)
                                 for k in 1:Nz) / column_momentum(q)
        height₀ = momentum_height(q)
        implicit_step!(q, solver, nothing, nothing, nothing, clock, (;), Δt,
                       AM.VerticalMomentumImplicitAdvection(scheme), (; w), ρ)
        @test all(isfinite, Array(interior(q)))
        @test column_momentum(q) ≈ momentum₀ rtol=sqrt(eps(FT))
        @test minimum(q[1, 1, k] for k in 1:Nz) ≥ -sqrt(eps(FT))
        @test momentum_height(q) > height₀
    end

    @testset "Construction wires the implicit solver and detects AIVA" begin
        model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                tracers=:ρc, scalar_advection=(; ρc=aiva()))
        @test needs_implicit_solver(model.advection.ρc)
        @test model.timestepper.implicit_solver isa BatchedTridiagonalSolver

        momentum_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                         momentum_advection=aiva())
        @test needs_implicit_solver(momentum_model.advection.momentum)
        @test momentum_model.timestepper.implicit_solver isa BatchedTridiagonalSolver
    end

    @testset "Momentum AIVA reduces to explicit advection below the CFL threshold" begin
        θ₀ = reference_state.potential_temperature
        θᵢ(x, y, z) = θ₀ + 2 * exp(-((x - 50)^2 + (y - 50)^2 + (z - 300)^2) / (2 * 80^2))
        uᵢ(x, y, z) = sinpi(z / 1000)

        explicit_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                         momentum_advection=WENO(FT))
        adaptive_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                         momentum_advection=aiva())

        for model in (explicit_model, adaptive_model)
            set!(model; θ=θᵢ, u=uᵢ)
            # A small Δt keeps the vertical CFL below the target, so the adaptive scheme
            # must reproduce the explicit scheme: the flux scale is 1 and the implicit
            # velocity is 0 everywhere.
            for _ in 1:3
                time_step!(model, 1)
            end
        end

        for name in (:ρu, :ρv, :ρw, :ρθ)
            explicit_field = Array(interior(Oceananigans.fields(explicit_model)[name]))
            adaptive_field = Array(interior(Oceananigans.fields(adaptive_model)[name]))
            @test isapprox(explicit_field, adaptive_field; rtol=sqrt(eps(FT)))
        end
    end

    @testset "Stable above the explicit vertical CFL (all variables)" begin
        θ₀ = reference_state.potential_temperature
        model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                tracers=:ρc,
                                momentum_advection=aiva(),
                                scalar_advection=(; ρθ=aiva(), ρc=aiva()))
        set!(model; θ = (x, y, z) -> θ₀ + 2 * exp(-((x-50)^2 + (y-50)^2 + (z-300)^2) / (2*80^2)),
                    ρc = (x, y, z) -> exp(-(z-300)^2 / (2*100^2)))

        # A large Δt drives a large vertical CFL; the run must stay finite.
        for _ in 1:20
            time_step!(model, 30)
        end
        @test all(isfinite, Array(interior(model.tracers.ρc)))
        for name in (:ρu, :ρv, :ρw, :ρθ)
            @test all(isfinite, Array(interior(Oceananigans.fields(model)[name])))
        end
    end

    @testset "Works with the acoustic substepper (compressible)" begin
        cgrid = RectilinearGrid(default_arch; size=(8, 8, 8), halo=(5, 5, 5),
                                x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers),
                                topology=(Periodic, Periodic, Bounded))
        cdyn() = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)

        # AIVA on a transport scalar (tracer) with the acoustic substepper: those scalars are
        # advanced by the generic implicit step (`scalar_substep!`).
        model = AtmosphereModel(cgrid; dynamics=cdyn(), timestepper=:AcousticRungeKutta3,
                                tracers=:ρc, scalar_advection=(; ρc=aiva()))
        @test needs_implicit_solver(model.advection.ρc)
        @test model.timestepper.implicit_solver isa BatchedTridiagonalSolver

        ref = model.dynamics.reference_state
        set!(model; θ = (x, y, z) -> 300 + 2 * exp(-((x-4kilometers)^2 + (z-4kilometers)^2) / (2*(1kilometers)^2)),
                    ρ = ref.density,
                    ρc = (x, y, z) -> exp(-(z - 4kilometers)^2 / (2*(1kilometers)^2)))
        for _ in 1:5
            time_step!(model, 1)
        end
        @test all(isfinite, Array(interior(model.tracers.ρc)))

        # Momentum and thermodynamic-variable AIVA with the acoustic substepper: the implicit
        # remainder is applied once per RK stage after the substep loop
        # (`implicit_advection_substep!`). Below the CFL threshold the flux scale is 1 and the
        # implicit velocity is 0, so the adaptive scheme must reproduce the explicit one.
        θᵢ(x, y, z) = 300 + 2 * exp(-((x-4kilometers)^2 + (y-4kilometers)^2 + (z-4kilometers)^2) / (2*(1kilometers)^2))

        explicit_model = AtmosphereModel(cgrid; dynamics=cdyn(), timestepper=:AcousticRungeKutta3,
                                         momentum_advection=WENO(FT), scalar_advection=(; ρθ=WENO(FT)))
        adaptive_model = AtmosphereModel(cgrid; dynamics=cdyn(), timestepper=:AcousticRungeKutta3,
                                         momentum_advection=aiva(), scalar_advection=(; ρθ=aiva()))

        for model in (explicit_model, adaptive_model)
            set!(model; θ=θᵢ, ρ=model.dynamics.reference_state.density)
            for _ in 1:3
                time_step!(model, 1)
            end
        end

        for name in (:ρu, :ρv, :ρw, :ρθ)
            explicit_field = Array(interior(Oceananigans.fields(explicit_model)[name]))
            adaptive_field = Array(interior(Oceananigans.fields(adaptive_model)[name]))
            @test isapprox(explicit_field, adaptive_field; rtol=sqrt(eps(FT)))
        end
    end

    @testset "Acoustic substepper is stable above the explicit vertical advective CFL" begin
        tall_grid = RectilinearGrid(default_arch; size=(8, 8, 32), halo=(5, 5, 5),
                                    x=(0, 4kilometers), y=(0, 4kilometers), z=(0, 2kilometers),
                                    topology=(Periodic, Periodic, Bounded))
        model = AtmosphereModel(tall_grid;
                                dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300),
                                timestepper=:AcousticRungeKutta3,
                                momentum_advection=aiva(), scalar_advection=(; ρθ=aiva()))

        ref = model.dynamics.reference_state
        set!(model; θ = (x, y, z) -> 300 + 2 * exp(-((x-2kilometers)^2 + (y-2kilometers)^2 + (z-700)^2) / (2*300^2)),
                    ρ = ref.density)
        set!(model; w = (x, y, z) -> 10 * exp(-(z-700)^2 / (2*200^2)))

        # Δt = 10 gives a vertical advective CFL of max|w| Δt / Δz ≈ 10 ⋅ 10 / 62.5 = 1.6; the
        # acoustic substep count adapts to the acoustic CFL automatically.
        for _ in 1:5
            time_step!(model, 10)
        end
        for name in (:ρu, :ρv, :ρw, :ρθ)
            @test all(isfinite, Array(interior(Oceananigans.fields(model)[name])))
        end
    end

    @testset "Terrain-following dynamics (TFVD, acoustic)" begin
        Nx, Nz = 16, 16
        Lx, Lz = 10000.0, 4000.0
        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation=LinearDecay())
        terrain_grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                                       x=(-Lx/2, Lx/2), z=z_faces,
                                       topology=(Periodic, Flat, Bounded))
        materialize_terrain!(terrain_grid, x -> 200 * exp(-x^2 / 2000^2))

        terrain_dynamics() = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)

        p₀, θ₀, pˢᵗ = 101325.0, 300.0, 100000.0
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)
        θᵢ(x, z) = θ₀ + 2 * exp(-(x^2 + (z - 1500)^2) / (2 * 500^2))

        explicit_model = AtmosphereModel(terrain_grid; dynamics=terrain_dynamics(), timestepper=:AcousticRungeKutta3,
                                         momentum_advection=WENO(FT), scalar_advection=(; ρθ=WENO(FT)))
        adaptive_model = AtmosphereModel(terrain_grid; dynamics=terrain_dynamics(), timestepper=:AcousticRungeKutta3,
                                         momentum_advection=aiva(), scalar_advection=(; ρθ=aiva()))

        # On terrain-following grids the adaptive-implicit split partitions the contravariant velocity.
        @test AM.advecting_vertical_velocity(adaptive_model.dynamics, adaptive_model.velocities) ===
              adaptive_model.dynamics.contravariant_vertical_velocity

        # Below the CFL threshold the adaptive scheme reproduces the explicit one over terrain too.
        for model in (explicit_model, adaptive_model)
            set!(model; ρ=ρᵢ, θ=θᵢ, u=10)
            for _ in 1:3
                time_step!(model, 1//2)
            end
        end
        for name in (:ρu, :ρv, :ρw, :ρθ)
            explicit_field = Array(interior(Oceananigans.fields(explicit_model)[name]))
            adaptive_field = Array(interior(Oceananigans.fields(adaptive_model)[name]))
            @test isapprox(explicit_field, adaptive_field; rtol=sqrt(eps(FT)))
        end

        # Above the explicit vertical advective CFL: re-seed a strong updraft (α ≈ 10 ⋅ 30 / 250 ≈ 1.2)
        # and take large steps; the acoustic substep count adapts automatically.
        set!(adaptive_model; w = (x, z) -> 10 * exp(-(z - 1500)^2 / (2 * 400^2)))
        for _ in 1:3
            time_step!(adaptive_model, 30)
        end
        for name in (:ρu, :ρv, :ρw, :ρθ)
            @test all(isfinite, Array(interior(Oceananigans.fields(adaptive_model)[name])))
        end
    end
end
