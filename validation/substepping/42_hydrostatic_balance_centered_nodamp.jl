#####
##### validation/substepping/42_hydrostatic_balance_centered_nodamp.jl
#####
##### Diagnostic with damping disabled and centered time-discretization
##### (ω = 0.5, ε = 0). For a correct linearization of the inviscid acoustic
##### system this is the trapezoidal rule and should be neutrally stable —
##### `max|w|` must stay at machine epsilon for all Ns.
#####
##### Used together with code edits that toggle individual terms in the
##### Schur tridiag (e.g. zeroing `buoyancy_linearization_coefficient`) to
##### isolate which term breaks consistency.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "hydrostatic_balance_centered_nodamp"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 600.0
const Δt     = 1.0
const g      = 9.80665
const θ₀ = 300.0
const N² = 1e-4
θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z)

build_grid() = RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

function build_substepped_model(; Ns)
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = Ns,
                                         forward_weight = 0.5,
                                         damping = NoDivergenceDamping())
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns)
    model = build_substepped_model(; Ns)
    ref   = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)

    drift = Float64[]; times = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(drift, wmax)
        push!(times, Float64(sim.model.clock.time))
        if mod(iteration(sim), 100) == 0
            @info @sprintf("[%s] iter=%4d t=%5.1fs max|w|=%.3e",
                           label, iteration(sim), sim.model.clock.time, wmax)
        end
    end
    add_callback!(sim, _track, IterationInterval(20))

    t0 = time(); status = :ok; err = ""
    try; run!(sim); catch e; status = :crashed; err = sprint(showerror, e); end
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
    label = @sprintf("Ns%02d", Ns)
    @info "=== $label ==="
    push!(results, run_one(label; Ns))
end

@info "=== SUMMARY (ω=0.5, NoDivergenceDamping; should be machine epsilon for all Ns) ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-6s t=%5.1fs final max|w|=%.3e  (%5.1fs)",
                   mark, r.label, r.t, r.wmax_final, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)
