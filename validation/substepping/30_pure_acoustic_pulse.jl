#####
##### validation/substepping/30_pure_acoustic_pulse.jl
#####
##### Minimal test for the substepper's acoustic core: centered CN, no damping,
##### no gravity. A small Gaussian (ρθ)′ perturbation should propagate as a
##### linear acoustic wave at c_s = √(γRT_ref) ≈ 347 m/s with no growth.
#####
##### If centered CN (ω = 0.5) with NoDivergenceDamping is unstable on this
##### test, the bug is in the core acoustic algorithm (Schur substitution,
##### off-centering, sign in some discrete operator). If it's stable here but
##### unstable on the bubble, the bug is in buoyancy coupling.
#####
##### Setup: 2D (Periodic, Flat, Bounded), g = 0, uniform reference, very small
##### Gaussian (ρθ)' perturbation. Run 60s — enough for the wave to traverse
##### the domain twice.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Statistics
using Printf
using JLD2

CUDA.functional() || error("GPU required")
const arch = GPU()

const CASE = "pure_acoustic_pulse"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 60.0
const Δt     = 1.0

const θ₀ = 300.0
const Δθ = 0.01      # 0.01 K — tiny, purely linear acoustic regime
const σ_pulse = 1e3
const x₀ = 0.0
const z₀ = 5e3

θᵢ(x, z) = θ₀ + Δθ * exp(-((x - x₀)^2 + (z - z₀)^2) / (2σ_pulse^2))

function build_grid()
    RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

function build_compressible_model(; Ns, ω, damping)
    grid = build_grid()
    # g = 0 — pure acoustic, no buoyancy.
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = 0.0)
    td  = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = ω, damping = damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = z -> θ₀)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = Centered(order = 2),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns, ω, damping)
    model = build_compressible_model(; Ns, ω, damping)
    set!(model; θ = θᵢ, ρ = model.dynamics.reference_state.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    function _progress(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        @info @sprintf("[%s] iter=%4d t=%5.1fs max|w|=%.3g", label,
                       iteration(sim), sim.model.clock.time, wmax)
    end
    add_callback!(sim, _progress, IterationInterval(10))

    sim.output_writers[:jld2] = JLD2Writer(model, (; w = model.velocities.w);
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(2.0),
                                           overwrite_existing = true)

    t0 = time()
    status = :ok; err = ""
    try
        run!(sim)
    catch e
        status = :crashed; err = string(typeof(e))
    end
    elapsed = time() - t0
    wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
    has_nan = any(isnan, interior(sim.model.velocities.w))
    return (; label, t = sim.model.clock.time, wmax, has_nan, elapsed, status, err)
end

results = NamedTuple[]
for (label, ω, damping) in [
    ("CN_nodamp",     0.5, NoDivergenceDamping()),
    ("CN_pproj_0.1",  0.5, PressureProjectionDamping(coefficient = 0.1)),
    ("ω0.7_nodamp",   0.7, NoDivergenceDamping()),
    ("ω0.8_nodamp",   0.8, NoDivergenceDamping()),
    ("ω1.0_nodamp",   1.0, NoDivergenceDamping()),
]
    @info "=== $label ==="
    push!(results, run_one(label; Ns = 12, ω = ω, damping = damping))
end

@info "=== SUMMARY ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-15s t=%5.1f wmax=%.3g  %4.1fs",
                   mark, r.label, r.t, r.wmax, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)
