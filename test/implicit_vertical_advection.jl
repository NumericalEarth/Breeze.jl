include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans: TendencyCallsite
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization,
                              cell_advection_timescale, AdaptiveImplicitVerticalAdvection,
                              vertical_scheme, FluxFormAdvection
using Oceananigans.BoundaryConditions: fill_halo_regions!, needs_implicit_solver
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Operators: volume
using Oceananigans.Solvers: BatchedTridiagonalSolver
using Oceananigans.Simulations: TimeStepWizard
using Oceananigans.TimeSteppers: implicit_step!, time_discretization
using Oceananigans.TurbulenceClosures: implicit_diffusion_solver, VerticallyImplicitTimeDiscretization,
                                       HorizontalFormulation, ThreeDimensionalFormulation
using Oceananigans.Units: kilometers
using Oceananigans.Utils: IterationInterval, NormalDivision
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

    # The explicit fraction of the split, s = min(1, cfl Δz / (|w| Δt)). `s ≡ 1` means the
    # split is inert, so equality tests in that regime say nothing about it; every testset
    # below states which regime it is in.
    function minimum_explicit_fraction(grid, scheme, w, Δt)
        vertical = vertical_scheme(scheme)
        vertical isa AdaptiveImplicitVerticalAdvection || return one(FT)
        cfl = time_discretization(vertical).cfl
        Δz = Array(interior(compute!(Field(zspacings(grid, Center(), Center(), Face())))))
        α = maximum(abs.(Array(interior(w))) .* Δt ./ Δz)
        return α > 0 ? min(one(cfl), cfl / α) : one(cfl)
    end

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

        # Engaged: α = 8 ⋅ 50 / 62.5 = 6.4 against cfl = 0.3, so s ≈ 0.047.
        @test minimum_explicit_fraction(cpu_grid, scheme, w, Δt) < 1//10

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
                       scheme, (; w), ρ)
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
                       scheme, (; w), ρ)
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

        # Explicit limit: s ≡ 1, so this equality pins the reduction to explicit advection and
        # says nothing about the split itself.
        @test minimum_explicit_fraction(grid, adaptive_model.advection.momentum,
                                        adaptive_model.velocities.w, 1) == 1

        for name in (:ρu, :ρv, :ρw, :ρθ)
            explicit_field = Array(interior(Oceananigans.fields(explicit_model)[name]))
            adaptive_field = Array(interior(Oceananigans.fields(adaptive_model)[name]))
            @test isapprox(explicit_field, adaptive_field; rtol=sqrt(eps(FT)))
        end
    end

    # A smoke test, not a stability claim: plain explicit WENO5 also survives this configuration
    # (measured, 20 steps of Δt = 30: explicit max|w| = 4.6980, AIVA 3.2487), so finiteness here
    # says only that the AIVA path runs, not that it extends the stable step.
    @testset "Runs with AIVA on every prognostic above the explicit vertical CFL" begin
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
        # (`implicit_substep!`). Below the CFL threshold the flux scale is 1 and the
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

        # Explicit limit: s ≡ 1 for momentum and for the thermodynamic variable alike.
        @test minimum_explicit_fraction(cgrid, adaptive_model.advection.momentum,
                                        adaptive_model.velocities.w, 1) == 1
        @test minimum_explicit_fraction(cgrid, adaptive_model.advection.ρθ,
                                        adaptive_model.velocities.w, 1) == 1

        for name in (:ρu, :ρv, :ρw, :ρθ)
            explicit_field = Array(interior(Oceananigans.fields(explicit_model)[name]))
            adaptive_field = Array(interior(Oceananigans.fields(adaptive_model)[name]))
            @test isapprox(explicit_field, adaptive_field; rtol=sqrt(eps(FT)))
        end

        # A `FluxFormAdvection` whose z-scheme is adaptive-implicit is itself an
        # `AdaptiveImplicitVerticalAdvection`, so it matches both the Breeze and the Oceananigans
        # `update_advection_timestep!` methods. Without the disambiguation this `set!` throws
        # `MethodError: ... is ambiguous`.
        flux_form = FluxFormAdvection(WENO(FT), WENO(FT), aiva())
        @test flux_form isa AdaptiveImplicitVerticalAdvection
        flux_form_model = AtmosphereModel(cgrid; dynamics=cdyn(), timestepper=:AcousticRungeKutta3,
                                          tracers=:ρc, momentum_advection=aiva(),
                                          scalar_advection=(; ρc=flux_form))
        set!(flux_form_model; θ=θᵢ, ρ=flux_form_model.dynamics.reference_state.density,
                              ρc = (x, y, z) -> exp(-(z - 4kilometers)^2 / (2*(1kilometers)^2)))
        for _ in 1:3
            time_step!(flux_form_model, 1)
        end
        @test all(isfinite, Array(interior(flux_form_model.tracers.ρc)))
    end

    # A smoke test, not a stability claim: explicit WENO5 also survives this configuration.
    # AIVA on ρθ throws a DomainError at t = 500–530 s (upstream, issue #897), so only
    # momentum carries AIVA here.
    @testset "Acoustic substepper runs with AIVA on every acoustic prognostic" begin
        tall_grid = RectilinearGrid(default_arch; size=(8, 8, 32), halo=(5, 5, 5),
                                    x=(0, 4kilometers), y=(0, 4kilometers), z=(0, 2kilometers),
                                    topology=(Periodic, Periodic, Bounded))
        # The vertically-implicit closure exercises the combined diffusion + advection solve:
        # both contributions land in the same tridiagonal system for each acoustic prognostic.
        model = AtmosphereModel(tall_grid;
                                dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300),
                                timestepper=:AcousticRungeKutta3,
                                closure=ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1, κ=1),
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

    @testset "Advecting-state cache pairs the implicit half with the slow tendencies" begin
        Nx, Nz = 16, 24
        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, 2400.0, length=Nz+1)); formulation=LinearDecay())
        cache_grid = RectilinearGrid(default_arch; size=(Nx, Nz), halo=(5, 5),
                                     x=(-8000.0, 8000.0), z=z_faces,
                                     topology=(Periodic, Flat, Bounded))
        materialize_terrain!(cache_grid, x -> 200 * exp(-x^2 / 2000^2))
        cache_dynamics() = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)

        # No adaptive-implicit scheme ⇒ no cache is allocated and closure-only solves read live state.
        explicit_model = AtmosphereModel(cache_grid; dynamics=cache_dynamics(), timestepper=:AcousticRungeKutta3,
                                         momentum_advection=WENO(FT), scalar_advection=(; ρθ=WENO(FT)))
        @test explicit_model.timestepper.substepper.vertical_velocity_cache === nothing
        @test explicit_model.timestepper.substepper.density_cache === nothing

        adaptive_model = AtmosphereModel(cache_grid; dynamics=cache_dynamics(), timestepper=:AcousticRungeKutta3,
                                         momentum_advection=aiva(), scalar_advection=(; ρθ=WENO(FT)))
        substepper = adaptive_model.timestepper.substepper
        @test substepper.vertical_velocity_cache isa Field
        @test substepper.density_cache isa Field

        p₀, θ₀, pˢᵗ = 101325.0, 300.0, 100000.0
        set!(adaptive_model; ρ=(x, z) -> adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants), θ=θ₀, u=10)
        for _ in 1:3
            time_step!(adaptive_model, 1)
        end

        # The cache must hold exactly the state the slow tendencies split, immune to anything
        # written to the live fields between tendency computation and the implicit solve.
        Breeze.TimeSteppers.prepare_acoustic_cache!(substepper, adaptive_model)
        Breeze.TimeSteppers.cache_advecting_state!(adaptive_model)
        w_live = AM.advecting_vertical_velocity(adaptive_model.dynamics, adaptive_model.velocities)
        ρ_live = AM.dynamics_density(adaptive_model.dynamics)
        w_at_tendency = Array(parent(w_live))
        ρ_at_tendency = Array(parent(ρ_live))
        parent(w_live) .*= 3
        parent(ρ_live) .*= 2
        w_used, ρ_used = Breeze.TimeSteppers.advecting_state(adaptive_model)
        @test Array(parent(w_used)) == w_at_tendency
        @test Array(parent(ρ_used)) == ρ_at_tendency
        parent(w_live) ./= 3
        parent(ρ_live) ./= 2

        # Split identity wᴸ = wᵉ + wⁱ through the real interpolation, at a Δt that saturates
        # the split (s < 1) — the identity is vacuous where s ≡ 1.
        scheme = Oceananigans.Advection.vertical_scheme(adaptive_model.advection.momentum)
        td = Oceananigans.TimeSteppers.time_discretization(scheme)
        td.Δt[] = 500
        w_cpu = Oceananigans.on_architecture(CPU(), w_used)
        cpu_grid = Oceananigans.on_architecture(CPU(), cache_grid)
        residual = zero(FT)
        saturated = 0
        for i in 1:Nx, k in 2:Nz
            s_face = Oceananigans.Advection.explicit_velocity_scaleᶜᶜᶠ(i, 1, k, cpu_grid, scheme, td, w_cpu)
            wⁱ = Oceananigans.Advection.implicit_vertical_velocityᶜᶜᶠ(i, 1, k, cpu_grid, scheme, td, w_cpu)
            wᴸ = @inbounds w_cpu[i, 1, k]
            residual = max(residual, abs(wᴸ - (s_face * wᴸ + wⁱ)))
            saturated += s_face < 1
        end
        @test saturated > 0
        @test residual < 10 * eps(FT)
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
        # Explicit limit: s ≡ 1 over terrain too.
        @test minimum_explicit_fraction(terrain_grid, adaptive_model.advection.momentum,
                                        adaptive_model.velocities.w, 1//2) == 1

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

    #####
    ##### Vertically-implicit closure diffusion survives AIVA on the thermodynamic variable:
    ##### below the split threshold (s ≡ 1) the combined tridiagonal solve must reduce to the
    ##### diffusion the explicit twin gets.
    #####
    @testset "Below threshold, AIVA does not disturb vertically-implicit diffusion" begin
        Nz, Lz = 64, 4kilometers
        diffusion_grid = RectilinearGrid(default_arch; size=(8, Nz), halo=(5, 5),
                                         x=(0, 2kilometers), z=(0, Lz), topology=(Periodic, Flat, Bounded))

        θ₀ = 300
        θᵢ(x, z) = θ₀ + 4 * exp(-(z - 2kilometers)^2 / (2 * 200^2))
        ρᵢ(x, z) = adiabatic_hydrostatic_density(z, 101325, θ₀, 100000, constants)
        wᵢ(x, z) = 4 * sinpi(z / Lz)
        κ = 200

        function diffusive_model(θ_advection, κ)
            model = AtmosphereModel(diffusion_grid;
                                    dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                                  reference_potential_temperature=300),
                                    timestepper=:AcousticRungeKutta3,
                                    closure=ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=κ, κ),
                                    momentum_advection=WENO(FT), scalar_advection=(; ρθ=θ_advection))
            set!(model; ρ=ρᵢ, θ=θᵢ)
            set!(model; w=wᵢ)
            return model
        end

        Δt = 1
        θ(m) = Array(interior(m.formulation.potential_temperature))
        stepped(m) = (for _ in 1:8; time_step!(m, Δt); end; m)

        explicit_diffusive = stepped(diffusive_model(WENO(FT), κ))
        explicit_inviscid  = stepped(diffusive_model(WENO(FT), 0))
        adaptive_diffusive = stepped(diffusive_model(aiva(), κ))
        adaptive_inviscid  = stepped(diffusive_model(aiva(), 0))

        # Explicit limit by construction: κ Δt / Δz² = 0.05 per step, α = |w| Δt / Δz = 0.06.
        @test minimum_explicit_fraction(diffusion_grid, adaptive_diffusive.advection.ρθ,
                                        adaptive_diffusive.velocities.w, Δt) == 1

        # Non-vacuous: the closure really does move θ over this window.
        diffusion_signal = maximum(abs, θ(explicit_diffusive) .- θ(explicit_inviscid))
        @test diffusion_signal > 1//10

        # The adaptive scheme reproduces the explicit one — with the closure as well as without it.
        @test isapprox(θ(adaptive_inviscid),  θ(explicit_inviscid);  rtol=sqrt(eps(FT)))
        @test isapprox(θ(adaptive_diffusive), θ(explicit_diffusive); rtol=sqrt(eps(FT)))

        # Stated so a failure names the cause: AIVA must not swallow the diffusion.
        @test maximum(abs, θ(adaptive_diffusive) .- θ(adaptive_inviscid)) > diffusion_signal / 2
    end

    #####
    ##### An engaged split transports exactly once.
    #####
    ##### The explicit fraction and the implicit remainder must sum to one conservative
    ##### transport. With uniform ρ and uniform w the exact answer is a rigid translation, and
    ##### the first moment of a conservative flux-form operator translates at the advecting
    ##### velocity whatever the reconstruction — so the centroid displacement measures how much
    ##### transport the *pair* of halves applied, independently of accuracy. This mirrors
    ##### `scalar_substep!`: explicit `div_ρUc` into a forward-Euler update, then `implicit_step!`
    ##### with the same scheme, the same velocities and the same density.
    #####
    @testset "Explicit fraction and implicit remainder sum to one transport" begin
        Nz, Lz = 200, 10000
        Δz = Lz / Nz
        w₀, Δt, cfl = 5, 30, 1//2
        transport_grid = RectilinearGrid(CPU(); size=(6, 6, Nz), halo=(5, 5, 5),
                                         x=(0, 600), y=(0, 600), z=(0, Lz),
                                         topology=(Periodic, Periodic, Bounded))

        w = Field{Center, Center, Face}(transport_grid)
        set!(w, (x, y, z) -> w₀)          # uniform ⇒ divergence free ⇒ exact rigid translation
        fill_halo_regions!(w)
        u = XFaceField(transport_grid)
        v = YFaceField(transport_grid)
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        U = (; u, v, w)

        ρ = CenterField(transport_grid)
        set!(ρ, 1)
        fill_halo_regions!(ρ)

        td = AdaptiveVerticallyImplicitDiscretization(FT; cfl)
        td.Δt[] = Δt
        scheme = WENO(FT; weight_computation=NormalDivision, order=5, time_discretization=td)
        solver = implicit_diffusion_solver(VerticallyImplicitTimeDiscretization(), transport_grid)
        transport_clock = Clock(transport_grid)

        z_center = [znode(1, 1, k, transport_grid, Center(), Center(), Center()) for k in 1:Nz]
        blob(x, y, z) = exp(-(z - 4000)^2 / (2 * 400^2))
        centroid(Q) = (q = Array(interior(Q, 2, 2, :)); sum(q .* z_center) / sum(q))

        Q = CenterField(transport_grid)
        set!(Q, blob)
        fill_halo_regions!(Q)
        z₀ = centroid(Q)

        # Engaged: the split withholds five sixths of the vertical flux at this Δt.
        s = min(1, cfl / (w₀ * Δt / Δz))
        @test s < 1//4
        @test needs_implicit_solver(scheme)

        G = CenterField(transport_grid)
        c = CenterField(transport_grid)
        interior(c) .= interior(Q) ./ interior(ρ)
        fill_halo_regions!(c)
        for k in 1:Nz, j in 1:6, i in 1:6
            G[i, j, k] = AM.div_ρUc(i, j, k, transport_grid, scheme, ρ, U, c)
        end
        interior(Q) .-= Δt .* interior(G)
        fill_halo_regions!(Q)

        # The explicit half alone carries only the fraction s.
        @test (centroid(Q) - z₀) / (w₀ * Δt) ≈ s rtol=1//100

        implicit_step!(Q, solver, nothing, nothing, nothing, transport_clock, (;), Δt, scheme, U, ρ)
        fill_halo_regions!(Q)

        # Together the two halves carry the whole transport.
        @test centroid(Q) - z₀ ≈ w₀ * Δt rtol=1//100
    end

    #####
    ##### The split time step is the stage interval of the stage that consumes it.
    #####
    ##### `update_advection_timestep!` runs inside `update_state!`, before the tendencies of the
    ##### *upcoming* stage are computed, and it is the only writer of the split for the whole
    ##### stage: the explicit fraction frozen into `Gⁿ` and the implicit remainder that completes
    ##### it therefore read one value. The Δt ladder both shrinks and grows, because a stale
    ##### second writer is invisible at constant Δt.
    #####
    @testset "Split time step tracks the Wicker–Skamarock stage interval" begin
        ladder_grid = RectilinearGrid(default_arch; size=(8, 8), halo=(5, 5),
                                      x=(0, 4kilometers), z=(0, 4kilometers),
                                      topology=(Periodic, Flat, Bounded))

        function ladder_model()
            model = AtmosphereModel(ladder_grid;
                                    dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                                  reference_potential_temperature=300),
                                    timestepper=:AcousticRungeKutta3,
                                    tracers=:ρc, momentum_advection=aiva(),
                                    scalar_advection=(; ρθ=aiva(), ρc=aiva()))
            set!(model; ρ=model.dynamics.reference_state.density, θ=300, ρc=1)
            return model
        end

        split_timestep(scheme) = time_discretization(vertical_scheme(scheme)).Δt[]

        model = ladder_model()
        ts = model.timestepper
        β = (ts.β₁, ts.β₂, ts.β₃)

        # `TendencyCallsite` callbacks fire at the end of `compute_tendencies!`, just after
        # `update_state!` refreshed the split — so a callback sees the value the upcoming stage
        # (`clock.stage` of step `clock.iteration + 1`) builds its tendencies with and then hands
        # to its own implicit solve.
        observed = Dict{Tuple{Int, Int}, FT}()
        record(m) = (observed[(m.clock.iteration + 1, m.clock.stage)] = split_timestep(m.advection.ρc); nothing)
        callbacks = [Callback(record, IterationInterval(1); callsite=TendencyCallsite())]

        Δts = map(FT, (10, 4, 7, 21))   # shrinks then grows; `align_time_step` does both
        for Δt in Δts
            time_step!(model, Δt; callbacks)
        end

        # One split, written once per stage: the acoustic prognostics and the tracers all read it.
        @test split_timestep(model.advection.momentum) == split_timestep(model.advection.ρθ)
        @test split_timestep(model.advection.ρc) == split_timestep(model.advection.ρθ)

        # Never `Inf` (a fully implicit split from a healthy state) and never zero.
        @test all(t -> isfinite(t) && t > 0, values(observed))

        # Stages 2 and 3 run with their own stage interval β Δt.
        for (step, Δt) in enumerate(Δts), stage in (2, 3)
            @test observed[(step, stage)] ≈ β[stage] * Δt rtol=sqrt(eps(FT))
        end

        # Stage 1 of step n is computed at the end of step n-1, before Δtₙ is known, so it runs
        # with β₁ Δtₙ₋₁. That is deliberate: what has to hold is that both halves of a stage read
        # the *same* value, not that the value is fresh. Moving this write into the stage to
        # freshen it is exactly what desynchronized the tracer split.
        for step in 2:length(Δts)
            @test observed[(step, 1)] ≈ β[1] * Δts[step-1] rtol=sqrt(eps(FT))
        end

        # The first step has no previous stage increment to invert; `maybe_prepare_first_time_step!`
        # seeds the clock from the Δt it was handed, so stage 1 still gets its own interval.
        @test observed[(1, 1)] ≈ β[1] * Δts[1] rtol=sqrt(eps(FT))

        # After a completed step the split is pinned to stage 1 of the next one.
        @test split_timestep(model.advection.ρθ) ≈ β[1] * Δts[end] rtol=sqrt(eps(FT))

        # A clock that has forgotten its stage increment — a restart, or any state not built by
        # `time_step!` — must still enter stage 1 with a finite split. Nothing writes the split
        # before stage 1 unless the first-step preparation seeds the clock, so the observable is
        # whether a stage-1 tendency computation happened at all.
        restarted = ladder_model()
        restarted.clock.iteration = 7
        restarted.clock.last_Δt = FT(Inf)
        restarted.clock.last_stage_Δt = FT(Inf)
        after_restart = Tuple{Int, FT}[]
        restart_callbacks = [Callback(m -> (push!(after_restart, (m.clock.stage, split_timestep(m.advection.ρc))); nothing),
                                      IterationInterval(1); callsite=TendencyCallsite())]
        time_step!(restarted, FT(10); callbacks=restart_callbacks)
        @test first(after_restart)[1] == 1
        @test first(after_restart)[2] ≈ β[1] * FT(10) rtol=sqrt(eps(FT))
        @test all(t -> isfinite(t[2]) && t[2] > 0, after_restart)
    end

    #####
    ##### The explicit and implicit halves of a scalar update split ONE velocity.
    #####
    ##### `update_state!` builds the moisture and tracer tendencies from the substepper's
    ##### time-averaged transport velocity, and the next acoustic loop resets and rebuilds that
    ##### field before `scalar_substep!` applies them. The implicit remainder therefore reads the
    ##### copy `cache_transport_velocity!` froze, not the live field — otherwise the two halves
    ##### split velocities from consecutive acoustic loops and no longer add back to one
    ##### transport operator.
    #####
    @testset "Tracer split pairs one transport velocity across the acoustic loop" begin
        pair_grid = RectilinearGrid(default_arch; size=(8, 8, 8), halo=(5, 5, 5),
                                    x=(0, 4kilometers), y=(0, 4kilometers), z=(0, 4kilometers),
                                    topology=(Periodic, Periodic, Bounded))

        # A tiny explicit-CFL target puts the split in its active regime for a gentle, stable
        # updraft; clipping at cfl = 0.5 would take a violent one.
        split_aiva() = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=FT(1//100)))

        pair_model = AtmosphereModel(pair_grid;
                                     dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                                     reference_potential_temperature=300),
                                     timestepper = :AcousticRungeKutta3,
                                     tracers = :ρc,
                                     momentum_advection = split_aiva(),
                                     scalar_advection = (; ρθ=split_aiva(), ρc=split_aiva()))

        # An updraft vanishing at both lids (a uniform `w` fights impenetrability and wrecks the
        # state), and a vertically varying tracer so vertical transport has something to act on.
        set!(pair_model; ρ=pair_model.dynamics.reference_state.density, θ=300,
             w=(x, y, z) -> sinpi(z / 4kilometers),
             ρc=(x, y, z) -> 1 + z / 4kilometers)

        pair_substepper = pair_model.timestepper.substepper

        # `TendencyCallsite` fires at the end of `compute_tendencies!`: `live` is the velocity the
        # tendencies were just built from, and `cache` the copy frozen after the *previous*
        # tendency computation — the one the stage in between actually split.
        observed = NamedTuple[]
        function record_transport(m)
            push!(observed, (live = Array(interior(AM.transport_velocities(m).w)),
                             cache = Array(interior(pair_substepper.time_averaged_vertical_velocity_cache)),
                             split_Δt = time_discretization(vertical_scheme(m.advection.ρc)).Δt[],
                             Gρc = maximum(abs, Array(interior(m.timestepper.Gⁿ.ρc)))))
            return nothing
        end
        transport_callbacks = [Callback(record_transport, IterationInterval(1); callsite=TendencyCallsite())]

        for _ in 1:2
            time_step!(pair_model, FT(30); callbacks=transport_callbacks)
        end

        # Cold start: `maybe_prepare_first_time_step!` seeds the transport velocity *before* the
        # first tendency computation, so stage 1 of step 1 transports tracers with a physical
        # velocity instead of the substepper's zero-initialized field.
        @test maximum(abs, first(observed).live) > 0
        @test first(observed).Gρc > 0

        # The pairing invariant: the velocity a stage's implicit remainder splits is exactly the
        # one the explicit fraction in its `Gⁿ` was scaled by.
        for k in 2:length(observed)
            @test observed[k].cache == observed[k-1].live
        end

        # Not vacuous: the live field really is rebuilt between stages, so splitting it live
        # would have used a velocity that `Gⁿ` was never built with.
        @test any(observed[k].live != observed[k-1].live for k in 2:length(observed))

        # And the split is active, so the pairing is load-bearing: most of the vertical transport
        # goes through the implicit remainder at the tightest stage.
        Δz_pair = 4kilometers / 8
        pair_cfl = time_discretization(vertical_scheme(pair_model.advection.ρc)).cfl
        fractions = [min(one(FT), pair_cfl * Δz_pair / (maximum(abs, o.live) * o.split_Δt))
                     for o in observed if maximum(abs, o.live) > 0]
        @test minimum(fractions) < 1
    end

    #####
    ##### The engaged thermodynamic split survives under acoustic substepping: the withheld
    ##### transport is folded into the loop (base through the CN predictors, perturbation via
    ##### the predictor solve), so the pressure never differences an inconsistent ρθ. The
    ##### post-loop placement died within ~5 steps on both of these cases (issue #897).
    #####
    @testset "Engaged ρθ-AIVA under AcousticRungeKutta3 (fold-in, #897)" begin
        engaged_aiva() = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=FT(1//100)))
        z16 = TerrainFollowingVerticalDiscretization(collect(range(0, 4000, length=17)); formulation=LinearDecay())

        for (hillheight, θᵢ, name) in ((0,   (x, z) -> 300 + 5 * exp(-(x^2 + (z - 1000)^2) / (2 * 600^2)), "flat bubble"),
                                       (600, (x, z) -> 300.0, "hill uniform θ"))
            fold_grid = RectilinearGrid(default_arch; size=(32, 16), halo=(5, 5),
                                        x=(-10kilometers, 10kilometers), z=z16,
                                        topology=(Periodic, Flat, Bounded))
            materialize_terrain!(fold_grid, x -> hillheight * exp(-x^2 / 2000^2))
            fold_model = AtmosphereModel(fold_grid;
                                         dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                                         reference_potential_temperature=300),
                                         timestepper = :AcousticRungeKutta3,
                                         momentum_advection = engaged_aiva(),
                                         scalar_advection = (; ρθ=engaged_aiva()))
            ρ₃₀₀(x, z) = adiabatic_hydrostatic_density(z, 101325.0, 300.0, 100000.0, constants)
            set!(fold_model; ρ=ρ₃₀₀, θ=θᵢ, u=10)
            for _ in 1:60
                time_step!(fold_model, 5)
            end
            ρᵈ = fold_model.dynamics.dry_density
            ρθ = fold_model.formulation.potential_temperature_density
            @test all(isfinite, Array(interior(ρᵈ)))
            @test minimum(interior(ρᵈ)) > 0
            @test all(isfinite, Array(interior(ρθ)))
            if name == "hill uniform θ"
                # Split-consistency invariant: uniform θ stays uniform to millikelvin.
                θdev = maximum(abs, Array(interior(ρθ)) ./ Array(interior(ρᵈ)) .- 300)
                @test θdev < 0.01
            end
        end
    end

    @testset "Advective timescale drops the vertical term under AIVA" begin
        Δx = 100 / 4
        Δz = 1000 / 16

        # A strong updraft so the vertical advective CFL dominates the 3D timescale. Set the
        # velocity fields directly (the anelastic model would otherwise re-diagnose w from
        # continuity); the timescale reads `model.velocities`.
        function seed_flow!(model)
            set!(model.velocities.u, 2)
            set!(model.velocities.w, 20)
            fill_halo_regions!(model.velocities.u)
            fill_halo_regions!(model.velocities.w)
            return model
        end

        τ_horizontal = Δx / 2                        # 1 / (|u|/Δx)
        τ_three_dim  = 1 / (2/Δx + 20/Δz)            # 1 / (|u|/Δx + |w|/Δz)

        # Explicit direction control via the callable, independent of the advection schemes.
        explicit_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                         momentum_advection=WENO(FT))
        seed_flow!(explicit_model)

        τ3 = CellAdvectionTimescale(ThreeDimensionalFormulation())(explicit_model)
        τh = CellAdvectionTimescale(HorizontalFormulation())(explicit_model)
        @test τ3 ≈ τ_three_dim rtol=1e-6
        @test τh ≈ τ_horizontal rtol=1e-6
        @test τh > τ3
        # The three-dimensional method reproduces Oceananigans' plain timescale.
        @test τ3 ≈ cell_advection_timescale(grid, explicit_model.velocities) rtol=1e-6
        # With explicit advection the automatic default keeps the vertical term.
        @test cell_advection_timescale(explicit_model) ≈ τ_three_dim rtol=1e-6

        # All-AIVA model: every vertically-advected prognostic is implicit, so the automatic
        # default drops the vertical term and the wizard would float Δt on the horizontal CFL.
        aiva_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                     momentum_advection=aiva(),
                                     scalar_advection=(; ρθ=aiva(), ρqᵛ=aiva()))
        seed_flow!(aiva_model)
        @test cell_advection_timescale(aiva_model) ≈ τ_horizontal rtol=1e-6

        # A single explicit scalar (ρqᵛ here) re-imposes the vertical CFL through the shared w.
        mixed_model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                      momentum_advection=aiva(),
                                      scalar_advection=(; ρθ=aiva()))  # ρqᵛ defaults to explicit
        seed_flow!(mixed_model)
        @test cell_advection_timescale(mixed_model) ≈ τ_three_dim rtol=1e-6

        # The wizard's hook calls our method: at cfl the picked Δt rides the horizontal CFL for
        # the all-AIVA model, leaving a vertical advective CFL well above 1 for AIVA to absorb.
        wizard = TimeStepWizard(cfl=FT(0.5))
        @test wizard.cell_advection_timescale === cell_advection_timescale
        Δt_wizard = wizard.cfl * wizard.cell_advection_timescale(aiva_model)
        @test Δt_wizard ≈ FT(0.5) * τ_horizontal rtol=1e-6
        @test Δt_wizard * (20 / Δz) > 1   # vertical CFL the wizard no longer clamps
    end
end
