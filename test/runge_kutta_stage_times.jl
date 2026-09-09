include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: dynamics_density
using Breeze.ParcelModels: ParcelDynamics
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: time_step!
using Test

#####
##### Stage abscissae of the SSP RK3 stepper
#####
##### Shu-Osher has Butcher abscissae c = (0, 1, 1/2): u^(2) approximates the solution at
##### tⁿ + Δt/2, so the third stage's tendency belongs at the midpoint. At tⁿ + Δt instead the
##### order conditions fail (Σbᵢcᵢ = 5/6 ≠ 1/2) and the scheme drops to first order for a
##### right-hand side that depends explicitly on time.
#####

"""
Step a model whose only active tendency is a forcing `F(t)` on a spatially uniform tracer,
and return the resulting tracer concentration.

Every spatial operator vanishes for a uniform tracer, so `∂ₜ(ρ c) = ρ F(t)` exactly and one
step returns the Runge-Kutta quadrature of `F` over `[0, Δt]`, abscissae included.
"""
function stepped_tracer_quadrature(grid, F, Δt)
    model = AtmosphereModel(grid; tracers = (:c,), forcing = (; c = F))
    ρ = Array(interior(dynamics_density(model.dynamics)))
    time_step!(model, Δt)
    ρc = Array(interior(model.tracers.c))
    return ρc[1, 1, 1] / ρ[1, 1, 1]
end

@testset "SSP RK3 stage abscissae [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 1), y=(0, 1), z=(0, 1))
    Δt = FT(1//2)
    rtol = FT === Float64 ? 1e-12 : 1e-5

    # b = (1/6, 1/6, 2/3) at c = (0, 1, 1/2) is Simpson's rule, exact for cubics. At c₃ = 1
    # the same weights integrate F = t to 5Δt²/6 instead of Δt²/2.
    @testset "quadrature of a time-only forcing" begin
        for (F, exact) in (((x, y, z, t) -> t,   Δt^2 / 2),
                           ((x, y, z, t) -> t^2, Δt^3 / 3),
                           ((x, y, z, t) -> t^3, Δt^4 / 4))
            @test isapprox(stepped_tracer_quadrature(grid, F, Δt), exact; rtol)
        end
    end

    # Each `update_state!` builds the tendency its stage consumes, so the clock it sees is
    # that stage's abscissa. The trace also pins `clock.stage`, the key per-stage work such
    # as the filtered surface state deduplicates on: it must reach 3.
    @testset "clock time and stage at each tendency evaluation" begin
        model = AtmosphereModel(grid)
        trace = Tuple{Float64, Int}[]
        record_clock(m) = (push!(trace, (m.clock.time, m.clock.stage)); nothing)

        simulation = Simulation(model; Δt, stop_iteration=1, verbose=false)
        add_callback!(simulation, record_clock, IterationInterval(1); callsite = UpdateStateCallsite())
        run!(simulation)

        # (tⁿ, stage 1) from the pre-step preparation, the two interior abscissae, then
        # (tⁿ⁺¹, stage 1) for the next step's first stage.
        half_Δt = Float64(Δt) / 2
        @test trace == [(0.0, 1), (Float64(Δt), 2), (half_Δt, 3), (Float64(Δt), 1)]
        @test model.clock.time == Δt
        @test model.clock.iteration == 1
        @test model.clock.stage == 1
    end
end

@testset "Parcel SSP RK3 clock bookkeeping" begin
    grid = RectilinearGrid(default_arch; size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics=ParcelDynamics())

    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))
    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    # The parcel stepper ticks to the same abscissae and closes each step on tⁿ⁺¹ exactly.
    Δt = 0.5
    for iteration in 1:4
        time_step!(model, Δt)
        @test model.clock.time == iteration * Δt
        @test model.clock.iteration == iteration
        @test model.clock.stage == 1
    end
end
