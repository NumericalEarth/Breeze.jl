#####
##### validation/substepping/10_bubble_stage_refresh_test.jl
#####
##### Re-run the dry thermal bubble with the *modified* substepper source
##### (acoustic_substepping.jl `convert_slow_tendencies!` uses current state
##### rather than U⁰). Everything else matches 07_dry_thermal_bubble_wizard.jl.
#####
##### Output lands in out/dry_thermal_bubble_wizard/stage_refresh.jld2 so it
##### plots next to the existing runs.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using JLD2

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")
isdir(OUTDIR) || mkpath(OUTDIR)

const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

build_grid() = RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * max(0, 1 - sqrt((x - x₀)^2 + (z - z₀)^2) / r₀)
end

function build_compressible(grid; damping = PressureProjectionDamping(coefficient = 0.5),
                            forward_weight = 0.8)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; damping, forward_weight)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

grid  = build_grid()
model = build_compressible(grid)
ref   = model.dynamics.reference_state
set!(model; θ = θᵢ_builder(grid), ρ = ref.density)

sim = Simulation(model; Δt = 0.5, stop_time = 500.0, verbose = false)
conjure_time_step_wizard!(sim; cfl = 0.3)

function progress(sim)
    @info @sprintf("[stage_refresh] iter=%5d t=%6.1fs Δt=%.3fs max|w|=%.2f",
                   iteration(sim), sim.model.clock.time, sim.Δt,
                   maximum(abs, interior(sim.model.velocities.w)))
end
add_callback!(sim, progress, IterationInterval(200))

outputs = (; w = model.velocities.w)
sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                       filename = joinpath(OUTDIR, "stage_refresh.jld2"),
                                       schedule = TimeInterval(10seconds),
                                       overwrite_existing = true)

function _go!(sim)
    t0 = time()
    ok = true; err = ""
    try
        run!(sim)
    catch e
        ok = false
        err = first(sprint(showerror, e), 400)
    end
    return (; wall = time() - t0, ok, err)
end
result = _go!(sim)

@info "[stage_refresh] done" result.ok iter = iteration(sim) result.wall max_w_final = maximum(abs, interior(model.velocities.w)) result.err
