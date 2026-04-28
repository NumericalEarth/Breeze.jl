#####
##### Diagnostic: hydrostatic-balance test with g = 0. Isolates whether the
##### centered-CN instability (test 40) is in the buoyancy part of the
##### vertical Schur or in the sound part. With g = 0, only the sound
##### coupling acts; if Ns=48 still grows, the sound coefficients are buggy;
##### if it stays at machine zero, the buoyancy coefficients are buggy.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = joinpath(@__DIR__, "out", "hydrostatic_no_gravity")
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 600.0
const Δt = 1.0
const θ₀ = 300.0

θᵇᵍ(z) = θ₀
θᵢ(x, z) = θ₀

build_grid() = RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

function build_substepped_model(; Ns, gravity = 9.80665, ω = 0.5)
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = gravity)
    td = SplitExplicitTimeDiscretization(substeps = Ns,
                                         forward_weight = ω,
                                         damping = NoDivergenceDamping())
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ₀)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns, gravity = 9.80665, ω = 0.5)
    model = build_substepped_model(; Ns, gravity, ω)
    ref   = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)

    drift = Float64[]; times = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(drift, wmax); push!(times, Float64(sim.model.clock.time))
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
    return (; label, Ns, t = Float64(model.clock.time),
              wmax_final, has_nan, elapsed, status, err)
end

results = NamedTuple[]
for ω in (0.5, 0.55, 0.6, 0.7)
    for Ns in (12, 48)
        label = @sprintf("g_full_w%.2f_Ns%02d", ω, Ns)
        @info "=== $label ==="
        push!(results, run_one(label; Ns, ω))
    end
end

@info "=== SUMMARY (uniform θ̄=$(θ₀); rest atmosphere) ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-12s t=%5.1fs final max|w|=%.3e  (%5.1fs)",
                   mark, r.label, r.t, r.wmax_final, r.elapsed)
end
