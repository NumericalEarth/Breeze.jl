#####
##### validation/substepping/20_gpu_fast_sweep.jl
#####
##### Fast GPU-based diagnostic sweeps. Uses a modest resolution (64×64) so
##### each bubble case runs in seconds on an H200. A single julia invocation
##### runs many variants and writes summary results.
#####
##### Sweeps:
#####   A. substep count Ns ∈ {3, 6, 12, 24, 48, 96} at fixed outer Δt
#####      — confirms the Δτᵋ-scaling bug from CPU run (13_bubble_substeps_sweep).
#####   B. forward_weight ω ∈ {0.55, 0.60, 0.70, 0.80, 0.90, 0.95}
#####      — maps the stability envelope in ω.
#####   C. PressureProjectionDamping coefficient β ∈ {0.0, 0.1, 0.25, 0.5, 1.0}
#####      — how much damping is actually needed.
#####
##### Each variant reports iteration reached, max|w| at crash (or end), crash
##### iteration, and whether it completed.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Statistics
using Printf
using JLD2

CUDA.functional() || error("CUDA not functional — this driver requires a GPU")
const arch = GPU()

const OUTDIR = joinpath(@__DIR__, "out", "gpu_fast_sweep")
isdir(OUTDIR) || mkpath(OUTDIR)

# Low-res for speed. Diagnosing the checkerboard doesn't need 128².
const Nx = 64
const Nz = 64
const STOP_T = 5minutes    # enough to see the stiffness develop

const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
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

function run_variant(; label,
                       substeps      = nothing,
                       forward_weight = 0.8,
                       damping        = PressureProjectionDamping(coefficient = 0.5),
                       fixed_Δt       = 1.0,
                       use_wizard     = false)
    grid  = build_grid()
    constants = ThermodynamicConstants(eltype(grid))
    td    = SplitExplicitTimeDiscretization(; substeps, forward_weight, damping)
    dyn   = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    model = AtmosphereModel(grid; dynamics = dyn,
                            advection = WENO(order = 9),
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)
    set!(model; θ = θᵢ_builder(grid), ρ = model.dynamics.reference_state.density)

    sim = Simulation(model; Δt = fixed_Δt, stop_time = STOP_T, verbose = false)
    use_wizard && conjure_time_step_wizard!(sim; cfl = 0.3)

    max_w = Ref(0.0)
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        max_w[] = max(max_w[], wmax)
    end
    add_callback!(sim, _track, IterationInterval(5))

    t0 = time()
    status = :ok
    err_str = ""
    try
        run!(sim)
    catch e
        status = :crashed
        err_str = string(typeof(e)) * ": " * sprint(showerror, e)[1:min(end, 100)]
    end
    elapsed = time() - t0

    # Check for NaNs/Infs
    has_nan = any(isnan, interior(sim.model.velocities.w)) ||
              any(isnan, interior(sim.model.velocities.u))

    return (; label,
            iteration = iteration(sim),
            t_reached = sim.model.clock.time,
            max_w_seen = max_w[],
            wmax_final = Float64(maximum(abs, interior(sim.model.velocities.w))),
            elapsed,
            status,
            err = err_str,
            has_nan)
end

results = Vector{NamedTuple}()

@info "=============== SWEEP A: substeps (fixed Δt=1s) ==============="
for Ns in [3, 6, 12, 24, 48, 96]
    r = run_variant(label = "Ns=$Ns", substeps = Ns, fixed_Δt = 1.0)
    push!(results, r)
    @info @sprintf("  Ns=%-3d  iter=%5d  t=%6.1f s  max|w|=%6.2f  elapsed=%5.1fs  %s",
                   Ns, r.iteration, r.t_reached, r.max_w_seen, r.elapsed,
                   r.status == :ok ? "OK" : "CRASH $(r.err)")
end

@info "=============== SWEEP B: forward_weight (Ns auto, wizard) ==============="
for ω in [0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95]
    r = run_variant(label = "ω=$ω", forward_weight = ω, use_wizard = true, fixed_Δt = 0.5)
    push!(results, r)
    @info @sprintf("  ω=%.2f  iter=%5d  t=%6.1f s  max|w|=%6.2f  elapsed=%5.1fs  %s",
                   ω, r.iteration, r.t_reached, r.max_w_seen, r.elapsed,
                   r.status == :ok ? "OK" : "CRASH $(r.err)")
end

@info "=============== SWEEP C: damping coefficient (ω=0.8, wizard) ==============="
for β in [0.0, 0.1, 0.25, 0.5, 1.0]
    damping = β == 0.0 ? NoDivergenceDamping() :
                         PressureProjectionDamping(coefficient = β)
    r = run_variant(label = "β=$β",
                    damping = damping,
                    use_wizard = true,
                    fixed_Δt = 0.5)
    push!(results, r)
    @info @sprintf("  β=%.2f  iter=%5d  t=%6.1f s  max|w|=%6.2f  elapsed=%5.1fs  %s",
                   β, r.iteration, r.t_reached, r.max_w_seen, r.elapsed,
                   r.status == :ok ? "OK" : "CRASH $(r.err)")
end

@info "=============== SUMMARY ==============="
for r in results
    mark = r.status == :ok ? "✓" : "✗"
    @info @sprintf("  %s  %-12s  iter=%5d t=%6.1f max|w|=%6.2f  %4.1fs",
                   mark, r.label, r.iteration, r.t_reached, r.max_w_seen, r.elapsed)
end

jldsave(joinpath(OUTDIR, "fast_sweep_results.jld2"); results)
@info "saved $(joinpath(OUTDIR, "fast_sweep_results.jld2"))"
