#####
##### validation/substepping/40_hydrostatic_balance.jl
#####
##### Hydrostatic-balance regression test for the acoustic substepper.
#####
##### A stratified atmosphere initialized in exact hydrostatic balance with
##### zero velocity should stay at rest. `max|w|` must remain at machine-zero
##### (ε ~ 1e-12 m/s for Float64) over many outer steps. Any drift signals a
##### bookkeeping bug in the slow / fast tendency split (sign error, missing
##### reference subtraction, off-centering applied to the wrong term, etc.).
#####
##### We sweep Ns ∈ {6, 12, 24, 48} at fixed Δt to make sure the balance is
##### independent of the substep count — Baldauf 2010 eq. (14) consistency.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "hydrostatic_balance"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 600.0      # 10 simulated minutes
const Δt     = 1.0
const g      = 9.80665

# Realistic stably-stratified background: θ̄(z) = θ₀ exp(N²z/g)
const θ₀ = 300.0
const N² = 1e-4           # mid-troposphere stratification
θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z)        # 2-arg form for set! on (Periodic, Flat, Bounded)

function build_grid()
    RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

function build_substepped_model(; Ns,
                                forward_weight = 0.55,
                                damping = NoDivergenceDamping())
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(substeps = Ns; forward_weight, damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns)
    model = build_substepped_model(; Ns)
    ref   = model.dynamics.reference_state

    # Start in EXACT hydrostatic balance: θ = θ̄(z), ρ = ρ̄(z), velocities = 0.
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)

    drift = Float64[]
    times = Float64[]
    function _track(sim)
        w = sim.model.velocities.w
        wmax = Float64(maximum(abs, interior(w)))
        umax = Float64(maximum(abs, interior(sim.model.velocities.u)))
        push!(drift, wmax)
        push!(times, Float64(sim.model.clock.time))
        @info @sprintf("[%s] iter=%4d t=%5.1fs max|w|=%.3e max|u|=%.3e",
                       label, iteration(sim), sim.model.clock.time, wmax, umax)
        return nothing
    end
    add_callback!(sim, _track, IterationInterval(20))

    t0 = time()
    status = :ok; err = ""
    try
        run!(sim)
    catch e
        status = :crashed; err = sprint(showerror, e)
    end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    return (; label, Ns,
              t = Float64(model.clock.time),
              wmax_final, has_nan, elapsed, status, err,
              times, drift)
end

results = NamedTuple[]
for Ns in (6, 12, 24, 48)
    label = "Ns$(Ns)"
    @info "=== $label ==="
    push!(results, run_one(label; Ns))
end

@info "=== SUMMARY (max|w| should be ≈ 1e-12 m/s for Float64) ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-6s t=%5.1fs final max|w|=%.3e  %5.1fs",
                   mark, r.label, r.t, r.wmax_final, r.elapsed)
end

jldsave(joinpath(OUTDIR, "summary.jld2"); results)

# Quick figure: max|w|(t) for each Ns
let
    fig = Figure(size = (900, 400))
    ax = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
              yscale = log10,
              title = "Hydrostatic-balance drift (rest atmosphere, all w should stay near machine zero)")
    for r in results
        # Replace zeros so log scale doesn't choke; mark NaN runs distinctly.
        d = map(x -> x == 0 ? 1e-16 : x, r.drift)
        lines!(ax, r.times, d; label = r.label, linewidth = 2)
    end
    axislegend(ax, position = :rb)
    save(joinpath(OUTDIR, "drift.png"), fig)
    @info "wrote drift.png"
end
