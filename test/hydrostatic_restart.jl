include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps, ForwardBackwardSubsteps
using Oceananigans
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

#####
##### Restart of HydrostaticDynamics from a checkpoint at a completed time step
#####
#
# A run of 6 + 4 time steps is compared with a run of 6 steps that writes a checkpoint with
# Oceananigans' `Checkpointer`, followed by a new model, built from scratch, that picks the
# checkpoint up and takes the remaining 4 steps.
#
# Expectation, written before the first run: every prognostic field is equal BITWISE. The
# checkpoint holds the clock and the prognostic fields (pˢ, u, v, ρθ, ρqᵛ and the tracers).
# Neither time stepper keeps anything else from one time step to the next: SSP RK3 and the
# split-explicit stepper copy the state at the start of each step, the split stepper freezes
# its linearization and computes its number of substeps from that state, and every diagnostic
# is recomputed from the prognostic state by `update_state!`. The surface geopotential and the
# hybrid coefficients are not checkpointed; they belong to the construction of the model, which
# the restart repeats. A difference of even one bit would mean that something that is not
# checkpointed carries over between time steps.
#
# The flow has a mountain, hybrid levels, rotation, a wave in the surface pressure, shear,
# horizontal divergence, two tracers (one advected by WENO) and a time-dependent forcing, so
# that every term is active and the clock matters. Only a restart at a completed time step is
# tested: a restart in the middle of a time step is not supported.

const spin_up_steps = 6
const restart_steps = 4
const Δt = 300

function build(; kw...)
    Nx, Ny, Nz = 8, 6, 6
    grid = RectilinearGrid(default_arch, FT; size=(Nx, Ny, Nz), x=(0, 8e5), y=(0, 6e5), z=(0, 1))
    A, B = hybrid_coefficients(Nz)
    kˣ, kʸ = 2π / 8e5, 2π / 6e5

    mountain(x, y) = g * 1000 * exp(-((x - 4e5) / 1.5e5)^2 - ((y - 3e5) / 1.5e5)^2)
    dynamics = HydrostaticDynamics(surface_geopotential=mountain, pressure_coefficient=A, surface_pressure_coefficient=B)
    acceleration(x, y, z, t) = 1e-4 * sin(kˣ * x) * (1 + t / 1000)

    model = hydrostatic_model(grid; dynamics, coriolis=FPlane(FT; f=1e-4), tracers=(:c, :d),
                              scalar_advection=(; d=WENO(FT; order=5)), forcing=(; u=acceleration),
                              thermodynamic_constants=reference_model_constants(), kw...)

    set!(model; pˢ = (x, y) -> isentropic_surface_pressure(mountain(x, y), 300) + 200 * cos(kˣ * x) * cos(kʸ * y),
                θ = (x, y, z) -> 290 + 50 * z + 2 * sin(kˣ * x) * sin(kʸ * y),
                u = (x, y, z) -> 5 * (1 + z) * (1 + sin(kˣ * x) / 2),
                v = (x, y, z) -> 2 * z * sin(kˣ * x) * cos(kʸ * y))

    ρ = continuity_density(model.dynamics)
    blob(x, y, z) = exp(-((x - 3e5) / 1.5e5)^2 - ((y - 3e5) / 1.5e5)^2) * (1 + z)
    mixing_ratio = CenterField(grid)
    set!(mixing_ratio, blob)
    set!(model; c = mixing_ratio * ρ, d = mixing_ratio * ρ)

    return model
end

snapshot(model) = Dict(string(name) => Array(interior(field))
                       for (name, field) in pairs(Oceananigans.prognostic_fields(model)))

function run_continuously(; kw...)
    model = build(; kw...)
    initial = snapshot(model)
    simulation = Simulation(model; Δt, stop_iteration=spin_up_steps + restart_steps, verbose=false)
    run!(simulation)
    return snapshot(model), initial, model.clock
end

function run_with_restart(dir; kw...)
    model = build(; kw...)
    simulation = Simulation(model; Δt, stop_iteration=spin_up_steps, verbose=false)
    simulation.output_writers[:checkpointer] = Checkpointer(model; schedule=IterationInterval(spin_up_steps), dir)
    run!(simulation)

    checkpoint = joinpath(dir, "checkpoint_iteration$(spin_up_steps).jld2")
    @test isfile(checkpoint)
    at_checkpoint = snapshot(model)

    model = build(; kw...)
    simulation = Simulation(model; Δt, stop_iteration=spin_up_steps + restart_steps, verbose=false)
    @test_logs (:info, r"Picking up simulation from checkpoint file") match_mode=:any run!(simulation; pickup=checkpoint)
    return snapshot(model), at_checkpoint, model.clock
end

const restart_timesteppers = (("SSPRungeKutta3", (;)),
                              ("MIS3C + VerletSubsteps()", split_explicit(:MIS3C, VerletSubsteps())),
                              ("MIS4a + ForwardBackwardSubsteps(0.1)", split_explicit(:MIS4a, ForwardBackwardSubsteps(divergence_damping=0.1))),
                              ("WickerSkamarock3 + VerletSubsteps(), 6 substeps", split_explicit(:WickerSkamarock3, VerletSubsteps(); substeps=6)))

@testset "Restart from a checkpoint [$name]" for (name, timestepper) in restart_timesteppers
    continuous, initial, clock = run_continuously(; timestepper...)

    restarted, at_checkpoint, restarted_clock = mktempdir() do dir
        run_with_restart(dir; timestepper...)
    end

    @test restarted_clock.iteration == clock.iteration == spin_up_steps + restart_steps
    @test restarted_clock.time == clock.time

    @test keys(continuous) == keys(restarted)

    for name in sort(collect(keys(continuous)))
        difference = maximum(abs, restarted[name] .- continuous[name])
        scale = maximum(abs, continuous[name])
        evolved_since_checkpoint = maximum(abs, continuous[name] .- at_checkpoint[name])
        @info "Restart, $name: largest difference from the uninterrupted run $difference " *
              "(largest value $scale, change over the restarted steps $evolved_since_checkpoint)"

        @test all(isfinite, restarted[name])
        @test restarted[name] == continuous[name] # bitwise

        # The comparison is not vacuous: the fields evolve over the restarted steps (the moisture is zero)
        name == "ρqᵛ" || @test evolved_since_checkpoint > 1e-6 * scale
    end
end

#####
##### Restart on a pressure-level grid
#####
#
# The density of a `PressureLevelGrid` is history: it converts the prognostic densities to specific
# values and is re-diagnosed from them, so the checkpoint carries it with the dynamics. Expected,
# before the first run: bitwise, as above. Measured: not bitwise. The pickup calls `update_state!`
# once more than the uninterrupted run, and the re-diagnosis of the density is idempotent only to
# a rounding (the specific values (ρ⁺ / ρ) ρc / ρ⁺ and ρc / ρ differ in the last place), which the
# four restarted steps amplify to a few hundred eps (largest: u, 1.6e-13 relative). The tolerance
# is 4096 eps of each field's largest value; a missing density in the checkpoint gives differences
# of order one.

using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization

function build_pressure_level(; kw...)
    Nz = 6
    A, B = hybrid_coefficients(Nz)
    grid = RectilinearGrid(default_arch, FT; size=(8, 6, Nz), halo=(3, 3, 3), x=(0, 8e5), y=(0, 6e5),
                           z=PressureLevelVerticalDiscretization(A, B; gravitational_acceleration=g),
                           topology=(Periodic, Periodic, Bounded))
    kˣ, kʸ = 2π / 8e5, 2π / 6e5
    mountain(x, y) = g * 1000 * exp(-((x - 4e5) / 1.5e5)^2 - ((y - 3e5) / 1.5e5)^2)
    model = hydrostatic_model(grid; dynamics=HydrostaticDynamics(surface_geopotential=mountain), coriolis=FPlane(FT; f=1e-4),
                              tracers=(:c,), thermodynamic_constants=reference_model_constants(), kw...)
    set!(model; pˢ = (x, y) -> isentropic_surface_pressure(mountain(x, y), 300) + 200 * cos(kˣ * x) * cos(kʸ * y),
                θ = (x, y, z) -> 290 + 5e-3 * z + 2 * sin(kˣ * x) * sin(kʸ * y),
                u = (x, y, z) -> 5 * (1 + sin(kˣ * x) / 2),
                v = (x, y, z) -> 2 * sin(kˣ * x) * cos(kʸ * y))
    set!(model; c = Field(1 * model.dynamics.density))
    return model
end

function run_pressure_level(dir, restart; kw...)
    model = build_pressure_level(; kw...)
    stop = restart ? spin_up_steps : spin_up_steps + restart_steps
    simulation = Simulation(model; Δt, stop_iteration=stop, verbose=false)
    restart && (simulation.output_writers[:checkpointer] = Checkpointer(model; schedule=IterationInterval(spin_up_steps), dir))
    run!(simulation)
    restart || return snapshot(model), Array(interior(model.dynamics.density))

    model = build_pressure_level(; kw...)
    simulation = Simulation(model; Δt, stop_iteration=spin_up_steps + restart_steps, verbose=false)
    run!(simulation; pickup=joinpath(dir, "checkpoint_iteration$(spin_up_steps).jld2"))
    return snapshot(model), Array(interior(model.dynamics.density))
end

@testset "Restart from a checkpoint on a PressureLevelGrid [$name]" for (name, timestepper) in restart_timesteppers[1:2]
    continuous, ρ = mktempdir(dir -> run_pressure_level(dir, false; timestepper...))
    restarted, ρʳ = mktempdir(dir -> run_pressure_level(dir, true; timestepper...))
    @test keys(continuous) == keys(restarted)
    for name in sort(collect(keys(continuous)))
        @info "Restart on a PressureLevelGrid, $name: largest difference $(maximum(abs, restarted[name] .- continuous[name]))"
        restart_rounding = 4096 * eps(FT) * maximum(abs, continuous[name])
        @test maximum(abs, restarted[name] .- continuous[name]) ≤ restart_rounding
    end
    @test maximum(abs, ρʳ .- ρ) ≤ 4096 * eps(FT) * maximum(abs, ρ)
end
