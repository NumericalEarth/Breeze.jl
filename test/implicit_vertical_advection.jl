using Breeze
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization, needs_implicit_solver,
                              implicit_advection_upper_diagonal, implicit_advection_lower_diagonal,
                              implicit_advection_diagonal
using Oceananigans.Solvers: BatchedTridiagonalSolver
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

    @testset "Density-weighted coefficients reduce to Oceananigans at ρ ≡ 1" begin
        # This is a pointwise, architecture-independent comparison of the coefficient functions, so
        # it is evaluated on a CPU grid (the loop below indexes fields from the host).
        cpu_grid = RectilinearGrid(CPU(); size=(4, 4, 16), x=(0, 100), y=(0, 100), z=(0, 1000))
        Δt = FT(50)
        scheme = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=FT(0.3)))
        scheme.time_discretization.Δt[] = Δt   # populate the Ref as the time loop would

        w = Field{Center, Center, Face}(cpu_grid)
        set!(w, (x, y, z) -> 5 * sin(2π * z / 1000))   # strong w that violates the explicit CFL
        ρ1 = CenterField(cpu_grid)
        set!(ρ1, 1)

        maxerr = zero(FT)
        for k in 2:15, j in 1:4, i in 1:4
            bu = AM.breeze_implicit_advection_upper_diagonal(i, j, k, cpu_grid, scheme, w, ρ1, Δt)
            ou = implicit_advection_upper_diagonal(i, j, k, cpu_grid, scheme, w, Δt, Center(), Center())
            bl = AM.breeze_implicit_advection_lower_diagonal(i, j, k, cpu_grid, scheme, w, ρ1, Δt)
            ol = implicit_advection_lower_diagonal(i, j, k, cpu_grid, scheme, w, Δt, Center(), Center())
            bd = AM.breeze_implicit_advection_diagonal(i, j, k, cpu_grid, scheme, w, ρ1, Δt)
            od = implicit_advection_diagonal(i, j, k, cpu_grid, scheme, w, Δt, Center(), Center())
            maxerr = max(maxerr, abs(bu - ou), abs(bl - ol), abs(bd - od))
        end
        @test maxerr == 0
    end

    @testset "Construction wires the implicit solver and detects AIVA" begin
        model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                tracers=:ρc, scalar_advection=(; ρc=aiva()))
        @test needs_implicit_solver(model.advection.ρc)
        @test model.timestepper.implicit_solver isa BatchedTridiagonalSolver
    end

    @testset "Adaptive implicit vertical advection is rejected for momentum" begin
        @test_throws ArgumentError AtmosphereModel(grid; dynamics,
                                                   formulation=:LiquidIcePotentialTemperature,
                                                   momentum_advection=aiva())
    end

    @testset "Stable above the explicit vertical CFL" begin
        θ₀ = reference_state.potential_temperature
        model = AtmosphereModel(grid; dynamics, formulation=:LiquidIcePotentialTemperature,
                                tracers=:ρc, scalar_advection=(; ρc=aiva()))
        set!(model; θ = (x, y, z) -> θ₀ + 2 * exp(-((x-50)^2 + (y-50)^2 + (z-300)^2) / (2*80^2)),
                    ρc = (x, y, z) -> exp(-(z-300)^2 / (2*100^2)))

        # A large Δt drives a large vertical CFL; the run must stay finite.
        for _ in 1:20
            time_step!(model, 30)
        end
        @test all(isfinite, Array(interior(model.tracers.ρc)))
    end

    @testset "Works with the acoustic substepper (compressible)" begin
        cgrid = RectilinearGrid(default_arch; size=(8, 8, 8), halo=(5, 5, 5),
                                x=(0, 8kilometers), y=(0, 8kilometers), z=(0, 8kilometers),
                                topology=(Periodic, Periodic, Bounded))
        cdyn = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)

        # AIVA on a transport scalar (tracer) is supported with the acoustic substepper: those
        # scalars are advanced by the generic implicit step (`scalar_substep!`).
        model = AtmosphereModel(cgrid; dynamics=cdyn, timestepper=:AcousticRungeKutta3,
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

        # AIVA on the thermodynamic variable is rejected with the acoustic substepper, which
        # integrates it inside the acoustic substep loop rather than the generic implicit step.
        @test_throws ArgumentError AtmosphereModel(cgrid; dynamics=cdyn, timestepper=:AcousticRungeKutta3,
                                                   scalar_advection=(; ρθ=aiva()))
    end
end
