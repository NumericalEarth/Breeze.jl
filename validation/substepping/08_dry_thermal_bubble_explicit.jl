#####
##### validation/substepping/08_dry_thermal_bubble_explicit.jl
#####
##### Fully-explicit compressible ground-truth for the dry thermal bubble.
##### Reuses the IC from 07_dry_thermal_bubble_wizard.jl, but uses
##### ExplicitTimeStepping at Δt = 0.1 s (well below the acoustic CFL of
##### Δz/c_s ≈ 0.22 s). Saves to out/dry_thermal_bubble_wizard/explicit.jld2
##### so the 3-panel animation script can pick it up next to
##### anelastic.jld2 and compressible.jld2.
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

# Same physical setup as 07.
const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r = sqrt((x - x₀)^2 + (z - z₀)^2)
        return θᵇᵍ(z) + Δθ * max(0, 1 - r / r₀)
    end
end

function build_explicit_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants)
end

# Acoustic CFL: Δz = 78 m, c_s ≈ 347 m/s → Δt_max ≈ 0.22 s. Use 0.1 s fixed.
const Δt     = 0.1
const STOP_T = 1000.0    # seconds — enough to cover the t = 400 s ring event
                         # and the subsequent rebound plateau. Not the full
                         # 1500 s because explicit is slow.

function run_explicit()
    grid = build_grid()
    model = build_explicit_model(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ_builder(grid), ρ = ref.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)

    counter = Ref(0)
    function _progress(sim)
        counter[] += 1
        if counter[] % 10 == 0
            @info @sprintf("[explicit] iter=%6d t=%7.1fs Δt=%.3fs max|w|=%.2f max|u|=%.2f",
                           iteration(sim), sim.model.clock.time, sim.Δt,
                           maximum(abs, interior(sim.model.velocities.w)),
                           maximum(abs, interior(sim.model.velocities.u)))
        end
    end
    add_callback!(sim, _progress, IterationInterval(500))

    outputs = (; w = model.velocities.w, T = model.temperature)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "explicit.jld2"),
                                           schedule = TimeInterval(10seconds),
                                           overwrite_existing = true)

    res = timed_run!(sim; label = "explicit")
    s = summarize_result("explicit", res, model)

    @info "[explicit] done" s
    return s
end

s = run_explicit()

# Append to result.jld2
resfile = joinpath(OUTDIR, "result.jld2")
existing = isfile(resfile) ? load(resfile) : Dict{String,Any}()
jldsave(resfile; explicit = s, existing...)
