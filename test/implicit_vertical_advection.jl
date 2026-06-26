using Breeze
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization, needs_implicit_solver,
                              implicit_advection_upper_diagonal, implicit_advection_lower_diagonal,
                              implicit_advection_diagonal
using Oceananigans.Solvers: BatchedTridiagonalSolver
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
        Δt = FT(50)
        scheme = WENO(FT; time_discretization=AdaptiveVerticallyImplicitDiscretization(FT; cfl=FT(0.3)))
        scheme.time_discretization.Δt[] = Δt   # populate the Ref as the time loop would

        w = Field{Center, Center, Face}(grid)
        set!(w, (x, y, z) -> 5 * sin(2π * z / 1000))   # strong w that violates the explicit CFL
        ρ1 = CenterField(grid)
        set!(ρ1, 1)

        maxerr = zero(FT)
        for k in 2:15, j in 1:4, i in 1:4
            bu = AM.breeze_implicit_advection_upper_diagonal(i, j, k, grid, scheme, w, ρ1, Δt)
            ou = implicit_advection_upper_diagonal(i, j, k, grid, scheme, w, Δt, Center(), Center())
            bl = AM.breeze_implicit_advection_lower_diagonal(i, j, k, grid, scheme, w, ρ1, Δt)
            ol = implicit_advection_lower_diagonal(i, j, k, grid, scheme, w, Δt, Center(), Center())
            bd = AM.breeze_implicit_advection_diagonal(i, j, k, grid, scheme, w, ρ1, Δt)
            od = implicit_advection_diagonal(i, j, k, grid, scheme, w, Δt, Center(), Center())
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
        @test all(isfinite, interior(model.tracers.ρc))
    end
end
