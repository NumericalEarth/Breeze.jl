using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans: Oceananigans
using Test

@testset "Open boundary conditions with PerturbationAdvection [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    @testset "Uniform flow stability with open x-boundaries [$FT]" begin
        Nx, Nz = 32, 8
        grid = RectilinearGrid(default_arch;
                               size = (Nx, Nz),
                               x = (0, 1000),
                               z = (0, 1000),
                               halo = (5, 5),
                               topology = (Bounded, Flat, Bounded))

        U = FT(10)

        # Build exterior value as ρᵣ * U (momentum density, not velocity)
        tmp_model = AtmosphereModel(grid; advection = WENO())
        ρᵣ = tmp_model.dynamics.reference_state.density
        ρu_mean = Field{Face, Nothing, Center}(grid)
        set!(ρu_mean, ρᵣ * U)

        scheme = PerturbationAdvection()
        ρu_bcs = FieldBoundaryConditions(
            west = OpenBoundaryCondition(ρu_mean; scheme),
            east = OpenBoundaryCondition(ρu_mean; scheme))

        model = AtmosphereModel(grid;
                                advection = WENO(),
                                boundary_conditions = (; ρu = ρu_bcs))

        θ₀ = model.dynamics.reference_state.potential_temperature
        set!(model; u = U, θ = θ₀)

        Δt = FT(0.01)
        Nsteps = 100
        for _ in 1:Nsteps
            time_step!(model, Δt)
        end

        # The uniform flow should remain uniform: PMA should not corrupt it
        u_max = @allowscalar maximum(abs, interior(model.velocities.u))
        u_min = @allowscalar minimum(abs, interior(model.velocities.u))
        @test u_max < U + FT(1e-4)
        @test u_min > U - FT(1e-4)
    end

    @testset "Thermal bubble with mean flow and open boundaries [$FT]" begin
        Nx, Nz = 16, 16
        grid = RectilinearGrid(default_arch;
                               size = (Nx, Nz),
                               x = (0, 2000),
                               z = (0, 2000),
                               halo = (5, 5),
                               topology = (Bounded, Flat, Bounded))

        U = FT(10)

        # Build exterior value as ρᵣ * U (momentum density, not velocity)
        tmp_model = AtmosphereModel(grid; advection = WENO())
        ρᵣ = tmp_model.dynamics.reference_state.density
        ρu_mean = Field{Face, Nothing, Center}(grid)
        set!(ρu_mean, ρᵣ * U)

        scheme = PerturbationAdvection(; outflow_timescale = 1)
        ρu_bcs = FieldBoundaryConditions(
            west = OpenBoundaryCondition(ρu_mean; scheme),
            east = OpenBoundaryCondition(ρu_mean; scheme))

        model = AtmosphereModel(grid;
                                advection = WENO(),
                                boundary_conditions = (; ρu = ρu_bcs))

        # Warm bubble initial condition with mean flow
        xc, zc, R = FT(1000), FT(1000), FT(300)
        θ₀ = model.dynamics.reference_state.potential_temperature

        θᵢ(x, z) = θ₀ + ifelse((x - xc)^2 + (z - zc)^2 < R^2, FT(2), FT(0))
        set!(model; u = U, θ = θᵢ)

        Δt = FT(0.1)
        for _ in 1:10
            time_step!(model, Δt)
        end

        # The model should not blow up
        @test all(isfinite, interior(model.velocities.u))
        @test all(isfinite, interior(model.velocities.w))
    end
end
