#####
##### validation/substepping/22_acoustic_pulse.jl
#####
##### Pure acoustic pulse test with NO gravity. Uniform isothermal atmosphere
##### with a small Gaussian θ perturbation at the domain centre; the resulting
##### acoustic wave should propagate outward at c_s = √(γRᵈT) with amplitude
##### decaying only through numerical dissipation and the acoustic-damping
##### kernel. No buoyancy, no stratification, no advection of the bubble —
##### just the column-solve's acoustic response in isolation.
#####
##### With g = 0:
#####   - `buoyancy_coefficient(g) = 0`
#####   - `buoyancy_linearization_coefficient ∝ g = 0`
##### so the tridiagonal reduces to the PGF-θ Schur coupling only. A bug in
##### the Δτᵋ-scaling of the remaining terms will show up cleanly as either
##### explosive amplification or wrong wave speed as Ns increases.
#####
##### Sweep Ns ∈ {6, 12, 24, 48, 96} at fixed outer Δt = 1 s.
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

const CASE = "acoustic_pulse_no_gravity"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

# A 60 s sim lets the sound wave (c_s ≈ 347 m/s) cross a 20 km wide domain twice.
const STOP_T = 60.0
const Δt     = 1.0            # outer step; same as the 4-way comparison

# Domain 20 km × 10 km; 64×64 so each diagnostic variant runs in seconds.
function build_grid()
    RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

const θ₀   = 300.0
const Δθ   = 0.01        # 0.01 K — genuinely small pulse, purely linear acoustic response
const σ    = 1e3         # Gaussian width (m)
const x₀   = 0.0
const z₀   = 5e3

θᵢ(x, z) = θ₀ + Δθ * exp(-((x - x₀)^2 + (z - z₀)^2) / (2σ^2))

function build_model(; Ns)
    grid = build_grid()
    # Zero gravity → isolate the acoustic tridiagonal.
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = 0.0)
    td  = SplitExplicitTimeDiscretization(substeps = Ns,
                                          forward_weight = 0.8,
                                          damping = PressureProjectionDamping(coefficient = 0.5))
    # Uniform reference: θᵣ(z) = 300 K, uniform ρ, uniform Π.
    dyn = CompressibleDynamics(td; reference_potential_temperature = z -> θ₀)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(; Ns)
    model = build_model(; Ns)
    set!(model; θ = θᵢ, ρ = model.dynamics.reference_state.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    function _progress(sim)
        @info @sprintf("[Ns=%d] iter=%4d t=%5.1fs max|w|=%.3g  max|θ-300|=%.3g",
                       Ns, iteration(sim), sim.model.clock.time,
                       Float64(maximum(abs, interior(sim.model.velocities.w))),
                       Float64(maximum(abs, interior(model.tracers.ρ) .* 0 .+
                               interior(PotentialTemperature(model)) .- 300)))
    end
    add_callback!(sim, _progress, IterationInterval(10))

    outputs = (; w = model.velocities.w,
                 θ = PotentialTemperature(model),
                 ρ = dynamics_density(model.dynamics))
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "Ns_$(Ns).jld2"),
                                           schedule = TimeInterval(2.0),
                                           overwrite_existing = true)

    t0 = time()
    status = :ok
    err = ""
    try
        run!(sim)
    catch e
        status = :crashed
        err = string(typeof(e))
    end
    elapsed = time() - t0

    wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
    has_nan = any(isnan, interior(sim.model.velocities.w))

    return (; Ns, iter = iteration(sim), t = sim.model.clock.time,
            wmax, has_nan, elapsed, status, err)
end

results = NamedTuple[]
for Ns in [6, 12, 24, 48, 96]
    @info "=== Ns=$Ns ==="
    push!(results, run_one(; Ns))
end

@info "=== ACOUSTIC PULSE SWEEP SUMMARY ==="
for r in results
    mark = r.status == :ok && !r.has_nan ? "✓" :
           r.status == :ok && r.has_nan  ? "NaN" : "✗"
    @info @sprintf("  %3s Ns=%-3d iter=%4d t=%5.1f wmax=%.3g elapsed=%.1fs",
                   mark, r.Ns, r.iter, r.t, r.wmax, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)
@info "jld2 outputs in $OUTDIR"
