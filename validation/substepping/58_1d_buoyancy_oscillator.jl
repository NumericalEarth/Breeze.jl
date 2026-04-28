#####
##### validation/substepping/58_1d_buoyancy_oscillator.jl
#####
##### 1D vertical buoyancy oscillator: stratified column at rest, Flat
##### horizontal, with a small θ perturbation. The system reduces to a
##### pure buoyancy oscillation at frequency ω_BV.
#####
##### Pass criterion: substepper max|w| matches explicit-compressible
##### closely. Identifies whether the bubble's bug is in vertical
##### buoyancy alone (1D test) or only with horizontal coupling.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "1d_buoyancy_oscillator"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Lz   = 10e3
const Nz   = 64
const θ₀   = 300.0
const N²   = 1e-4
const g    = 9.80665
const Δθ_pulse = 1e-4         # tiny — linear regime
const σ_z  = 500.0
const z_pulse = Lz / 2

θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z) + Δθ_pulse * exp(-((z - z_pulse) / σ_z)^2)
θᵢ(z) = θᵢ(0, z)

const STOP_T = 600.0  # roughly 1 BV period
const Δt_expl  = 0.05
const Δt_subst = 1.0

build_grid() = RectilinearGrid(arch; size = (Nz,), halo = (5,),
                               z = (0, Lz),
                               topology = (Flat, Flat, Bounded))

function build_explicit_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = Centered(order = 2),
                           thermodynamic_constants = constants)
end

function build_substepped_model(grid; ω = 0.55)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = ω,
                                          damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = Centered(order = 2),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt, stop_time = STOP_T)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)
    sim = Simulation(model; Δt, stop_time, verbose = false)

    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, wmax)
    end
    add_callback!(sim, _track, IterationInterval(20))

    t0 = time(); status = :ok
    try; run!(sim); catch e; status = :crashed; end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    # Find max-over-time |w|
    wmax_overall = isempty(wmax_log) ? wmax_final : maximum(wmax_log)

    return (; label, wmax_final, wmax_overall, has_nan, elapsed, status, times, wmax_log)
end

@info "=== explicit ==="
expl = run_one("explicit", build_explicit_model; Δt = Δt_expl)

results = NamedTuple[]
push!(results, expl)
for ω in (0.5, 0.55, 0.7)
    label = "subst_w$(@sprintf("%.2f", ω))"
    @info "=== $label ==="
    push!(results, run_one(label, grid -> build_substepped_model(grid; ω); Δt = Δt_subst))
end

@info "=== SUMMARY ==="
for r in results
    mark = r.has_nan ? "NaN" : "✓"
    @info @sprintf("  %3s %-15s wmax_overall=%.4e wmax_final=%.4e (%5.1fs)",
                   mark, r.label, r.wmax_overall, r.wmax_final, r.elapsed)
end
ratio = filter(r -> r.label != "explicit" && !r.has_nan, results)
expl_w = expl.wmax_overall
@info "=== RATIO (sub/expl, max-over-time amplitude) ==="
for r in ratio
    @info @sprintf("  %-15s ratio=%.3f", r.label, r.wmax_overall / expl_w)
end
