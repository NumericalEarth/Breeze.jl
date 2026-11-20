include(joinpath(@__DIR__, "setup.jl"))

#####
##### Time stepping on a calendar clock: `clock.time` is a `DateTime` and `Δt` stays in seconds.
#####
##### Every stepper computes the end of the step up front, `tⁿ⁺¹`, to close the step on it exactly.
##### A `DateTime` cannot take a `Float64` directly, so that sum has to go through `next_time`;
##### these tests run each stepper on a calendar clock and check the clock lands where it should.
#####

using Breeze
using Breeze: continuity_density
using Breeze.AtmosphereModels: adiabatic_balance_twin, AdiabaticBalancer, thermodynamic_density
using Breeze.ParcelModels: ParcelDynamics
using Dates: DateTime, Millisecond
using Oceananigans
using Oceananigans: UpdateStateCallsite
using Oceananigans.TimeSteppers: time_step!, update_state!
using Test

const t₀ = DateTime(2024, 6, 21, 12)

@testset "SSP RK3 on a DateTime clock" begin
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 1), y=(0, 1), z=(0, 1))
    model = AtmosphereModel(grid; clock = Clock(time = t₀))
    trace = Tuple{DateTime, Int}[]
    record_clock(m) = (push!(trace, (m.clock.time, m.clock.stage)); nothing)

    Δt = 0.5
    simulation = Simulation(model; Δt, stop_iteration = 1, verbose = false)
    add_callback!(simulation, record_clock, IterationInterval(1); callsite = UpdateStateCallsite())
    run!(simulation)

    # The same abscissae as the numeric clock in runge_kutta_stage_times.jl: tⁿ, tⁿ + Δt, tⁿ + Δt/2,
    # then tⁿ⁺¹ for the next step's first stage.
    @test trace == [(t₀, 1), (t₀ + Millisecond(500), 2), (t₀ + Millisecond(250), 3), (t₀ + Millisecond(500), 1)]
    @test model.clock.time == t₀ + Millisecond(500)
    @test model.clock.iteration == 1
    @test model.clock.stage == 1
    @test model.clock.last_Δt == Δt
end

@testset "Acoustic RK3 on a DateTime clock" begin
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), halo=(5, 5, 5),
                           x=(0, 1_000), y=(0, 1_000), z=(0, 1_000),
                           topology=(Periodic, Periodic, Bounded))
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics, clock = Clock(time = t₀))
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300)

    Δt = 0.4
    for iteration in 1:3
        time_step!(model, Δt)
        @test model.clock.time == t₀ + Millisecond(round(Int, 1000 * iteration * Δt))
        @test model.clock.iteration == iteration
        @test model.clock.stage == 1
    end
    @test model.clock.last_Δt == Δt
end

@testset "Parcel SSP RK3 on a DateTime clock" begin
    # The parcel stepper interpolates the environment at the parcel position on the host,
    # so it runs on the CPU like the other parcel tests.
    grid = RectilinearGrid(CPU(); size=10, z=(0, 1000), topology=(Flat, Flat, Bounded))
    model = AtmosphereModel(grid; dynamics = ParcelDynamics(), clock = Clock(time = t₀))

    T(z) = 288.0 - 0.0065 * z
    p(z) = 101325.0 * exp(-z / 8500)
    ρ(z) = p(z) / (287.0 * T(z))
    set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)

    Δt = 0.5
    for iteration in 1:4
        time_step!(model, Δt)
        @test model.clock.time == t₀ + Millisecond(round(Int, 1000 * iteration * Δt))
        @test model.clock.iteration == iteration
        @test model.clock.stage == 1
    end
end

#####
##### The adiabatic balance steps a stripped twin forward and backward and then resets its clock.
##### A calendar clock has no zero to reset to, so the balance restores the initial time instead,
##### and the twin starts at the production time so the boundary conditions see the right instant.
#####

const T₀_dc  = 250.0
const G_dc   = 9.80665
const CPD_dc = 1005.0
θ_iso_dc(z) = T₀_dc * exp(G_dc * z / (CPD_dc * T₀_dc))

function production_model_on_calendar_clock(arch; Nx = 8, Ny = 8, Nz = 16, Lz = 10e3, Lh = 100e3)
    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lh), y = (0, Lh), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(sponge = UpperSponge(damping_rate = 1/5, depth = 2000.0))
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ_iso_dc,
                               base_pressure = 1e5, standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn, thermodynamic_constants = constants,
                           coriolis = nothing, timestepper = :AcousticRungeKutta3,
                           clock = Clock(time = t₀))
end

# Discrete-balanced rest state (mirrors adiabatic_balance_set.jl)
function set_discrete_rest!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.dry_density) .= parent(ref.density)
    parent(thermodynamic_density(model.formulation)) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
    return nothing
end

@testset "Adiabatic balance on a DateTime clock" begin
    model = production_model_on_calendar_clock(default_arch)
    set_discrete_rest!(model)

    twin = adiabatic_balance_twin(model)
    @test twin.clock !== model.clock
    @test twin.clock.time == t₀
    @test twin.clock.iteration == 0

    ρ₀ = deepcopy(continuity_density(model.dynamics))
    set!(model; balancer = AdiabaticBalancer(Δt = 0.5, cycles = 1))

    @test continuity_density(model.dynamics) ≈ ρ₀ rtol=1e-8
    @test isapprox(maximum(abs, model.momentum.ρw), 0; atol=1e-7)
    @test model.clock.time == t₀
    @test model.clock.iteration == 0
    @test model.clock.stage == 1
    @test isinf(model.clock.last_Δt)
end
