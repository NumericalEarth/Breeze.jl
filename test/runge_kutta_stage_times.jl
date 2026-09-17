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
##### Stage abscissae of the SSP Runge-Kutta steppers
#####
##### SSP RK3 has Butcher abscissae c = (0, 1, 1/2): u^(2) approximates the solution at
##### tⁿ + Δt/2, so the third stage's tendency belongs at the midpoint. At tⁿ + Δt instead the
##### order conditions fail (Σbᵢcᵢ = 5/6 ≠ 1/2) and the scheme drops to first order for a
##### right-hand side that depends explicitly on time. SSP RK(4,3) has c = (0, 1/2, 1, 1/2),
##### so its clock likewise steps back to the midpoint before the last stage.
#####

"""
Step a model whose only active tendency is a forcing `F(t)` on a spatially uniform tracer,
and return the resulting tracer concentration.

Every spatial operator vanishes for a uniform tracer, so `∂ₜ(ρ c) = ρ F(t)` exactly and one
step returns the Runge-Kutta quadrature of `F` over `[0, Δt]`, abscissae included.
"""
function stepped_tracer_quadrature(grid, F, Δt, timestepper)
    model = AtmosphereModel(grid; timestepper, tracers = (:c,), forcing = (; c = F))
    ρ = Array(interior(dynamics_density(model.dynamics)))
    time_step!(model, Δt)
    ρc = Array(interior(model.tracers.c))
    return ρc[1, 1, 1] / ρ[1, 1, 1]
end

"""
Step a spatially uniform tracer with the linear relaxation `∂ₜ c = -λ c` from `c = 1`, and
return the amplification factor after one step: the stepper's stability polynomial at
`z = -λ Δt`.
"""
function stepped_tracer_amplification(grid, λ, Δt, timestepper)
    relaxation(x, y, z, t, c) = -λ * c
    forcing = Forcing(relaxation, field_dependencies = :c)
    model = AtmosphereModel(grid; timestepper, tracers = (:c,), forcing = (; c = forcing))
    set!(model, c = 1)
    time_step!(model, Δt)
    c = Array(interior(model.tracers.c))
    return c[1, 1, 1]
end

# Stability polynomials R(z) of the two schemes, from their Butcher tableaux.
stability_polynomial(::Val{:SSPRungeKutta3}, z)  = 1 + z + z^2 / 2 + z^3 / 6
stability_polynomial(::Val{:SSPRungeKutta43}, z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 48

# The clock trace expected from one step: (tⁿ, stage 1) from the pre-step preparation, the
# interior abscissae, then (tⁿ⁺¹, stage 1) for the next step's first stage.
expected_clock_trace(::Val{:SSPRungeKutta3}, Δt)  = [(0.0, 1), (Δt, 2), (Δt / 2, 3), (Δt, 1)]
expected_clock_trace(::Val{:SSPRungeKutta43}, Δt) = [(0.0, 1), (Δt / 2, 2), (Δt, 3), (Δt / 2, 4), (Δt, 1)]

@testset "$(timestepper) stage abscissae [$(FT)]" for timestepper in (:SSPRungeKutta3, :SSPRungeKutta43),
                                                       FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 1), y=(0, 1), z=(0, 1))
    Δt = FT(1//2)
    rtol = FT === Float64 ? 1e-12 : 1e-5

    # SSP RK3 has b = (1/6, 1/6, 2/3) at c = (0, 1, 1/2), and SSP RK(4,3) has
    # b = (1/6, 1/6, 1/6, 1/2) at c = (0, 1/2, 1, 1/2): both are Simpson's rule, exact for
    # cubics. At c = 1 for the last stage instead, the same weights integrate F = t to
    # 5Δt²/6 (RK3) or 3Δt²/4 (RK(4,3)) instead of Δt²/2.
    @testset "quadrature of a time-only forcing" begin
        for (F, exact) in (((x, y, z, t) -> t,   Δt^2 / 2),
                           ((x, y, z, t) -> t^2, Δt^3 / 3),
                           ((x, y, z, t) -> t^3, Δt^4 / 4))
            @test isapprox(stepped_tracer_quadrature(grid, F, Δt, timestepper), exact; rtol)
        end
    end

    # A state-dependent right-hand side exercises the full tableau rather than its row sums:
    # one step of ∂ₜc = -λc multiplies c by the stability polynomial R(-λΔt), which is
    # 1 + z + z²/2 + z³/6 for any third-order three-stage scheme but carries the
    # scheme-specific z⁴/48 for SSP RK(4,3).
    @testset "stability polynomial of a linear relaxation" begin
        λ = FT(3//2)
        z = -λ * Δt
        @test isapprox(stepped_tracer_amplification(grid, λ, Δt, timestepper),
                       stability_polynomial(Val(timestepper), z); rtol)
    end

    # Each `update_state!` builds the tendency its stage consumes, so the clock it sees is
    # that stage's abscissa. The trace also pins `clock.stage`, the key per-stage work such
    # as the filtered surface state deduplicates on: it must reach the number of stages.
    @testset "clock time and stage at each tendency evaluation" begin
        model = AtmosphereModel(grid; timestepper)
        trace = Tuple{Float64, Int}[]
        record_clock(m) = (push!(trace, (m.clock.time, m.clock.stage)); nothing)

        simulation = Simulation(model; Δt, stop_iteration=1, verbose=false)
        add_callback!(simulation, record_clock, IterationInterval(1); callsite = UpdateStateCallsite())
        run!(simulation)

        @test trace == expected_clock_trace(Val(timestepper), Float64(Δt))
        @test model.clock.time == Δt
        @test model.clock.iteration == 1
        @test model.clock.stage == 1
    end
end

@testset "Parcel SSP RK3 clock bookkeeping" begin
    # The parcel stepper interpolates the environment at the parcel position on the host,
    # so it runs on the CPU like the other parcel tests.
    grid = RectilinearGrid(CPU(); size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
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
