include(joinpath(@__DIR__, "setup.jl"))

#####
##### Time stepping on a calendar clock: `clock.time` is a `DateTime`, `Δt` stays in seconds.
#####

using Breeze
using Breeze.AtmosphereModels: adiabatic_balance_twin
using Breeze.ParcelModels: ParcelDynamics
using Dates: DateTime, Millisecond
using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using Test

const t₀ = DateTime(2024, 6, 21, 12)

function anelastic_model()
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 1), y=(0, 1), z=(0, 1))
    return AtmosphereModel(grid; clock = Clock(time = t₀))
end

function compressible_model()
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 1_000), y=(0, 1_000), z=(0, 1_000), topology=(Periodic, Periodic, Bounded))
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics, clock = Clock(time = t₀))
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300)
    return model
end

function parcel_model()
    # The parcel stepper interpolates the environment on the host, so it runs on the CPU.
    grid = RectilinearGrid(CPU(); size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics = ParcelDynamics(), clock = Clock(time = t₀))
    set!(model, T = z -> 288 - 0.0065z, p = z -> 101325 * exp(-z / 8500), ρ = z -> 1.2 * exp(-z / 8500), z=0.0, w=1.0)
    return model
end

# Every stepper computes tⁿ⁺¹ up front and closes the step on it exactly. That sum is the one
# place a `DateTime` clock differs from a numeric one, so each stepper must land on the instant.
@testset "Each stepper closes a step on the exact instant [$name]" for (name, build) in
        (("SSP RK3", anelastic_model), ("acoustic RK3", compressible_model), ("parcel", parcel_model))
    model = build()
    Δt = 0.3
    for n in 1:3
        time_step!(model, Δt)
        @test model.clock.time == t₀ + Millisecond(round(Int, 1000n * Δt))
        @test model.clock.iteration == n
        @test model.clock.stage == 1
    end
    @test model.clock.last_Δt == Δt
end

# The balance steps its twin forward and backward and then resets the twin's clock. A calendar
# clock has no zero, so the initial time is restored instead.
@testset "Adiabatic balance restores the initial time" begin
    grid = RectilinearGrid(default_arch; size=(8, 8, 16), halo=(5, 5, 5),
                           x=(0, 100e3), y=(0, 100e3), z=(0, 10e3), topology=(Periodic, Periodic, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics, coriolis = nothing, clock = Clock(time = t₀))
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300)

    twin = adiabatic_balance_twin(model)
    @test twin.clock.time == t₀

    balance_adiabatically!(twin; Δt = 0.5, cycles = 1)
    @test twin.clock.time == t₀
    @test twin.clock.iteration == 0
    @test isinf(twin.clock.last_Δt)
end
