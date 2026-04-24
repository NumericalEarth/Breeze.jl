#####
##### validation/substepping/21_gpu_4way.jl
#####
##### Four-case side-by-side on GPU at 128² (matches 07):
#####   1. anelastic            Δt = 1.0 s
#####   2. compressible Ns=12   Δt = 1.0 s  (ω=0.8, PressureProjectionDamping(0.5))
#####   3. compressible Ns=48   Δt = 1.0 s
#####   4. compressible explicit  Δt = 0.1 s
#####
##### All save w, θ, ρ (anelastic: no ρ since ρ ≡ ρ_ref). Runs to the prescribed
##### STOP_T but each case stops early on NaN; FieldTimeSeries picks up whatever
##### frames were written.
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

const CASE = "dry_thermal_bubble_4way_dt1"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 7minutes

# Physics — same as 07.
const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(arch; size = (128, 128), halo = (5, 5),
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

function run_one(; label, build_model, Δt, stop_t = STOP_T)
    grid = build_grid()
    model = build_model(grid)

    θᵢ = θᵢ_builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ)
        outputs = (; w = model.velocities.w, θ = PotentialTemperature(model))
    else
        set!(model; θ = θᵢ, ρ = model.dynamics.reference_state.density)
        outputs = (; w = model.velocities.w,
                     θ = PotentialTemperature(model),
                     ρ = dynamics_density(model.dynamics))
    end

    sim = Simulation(model; Δt, stop_time = stop_t, verbose = false)

    function _progress(sim)
        @info @sprintf("[%s] iter=%5d t=%6.1fs Δt=%.3fs max|w|=%.2f",
                       label, iteration(sim), sim.model.clock.time, sim.Δt,
                       maximum(abs, interior(sim.model.velocities.w)))
    end
    add_callback!(sim, _progress, IterationInterval(100))

    # Lower output cadence for the explicit run so we don't write 10× more frames.
    schedule = label == "explicit" ? TimeInterval(10seconds) : TimeInterval(10seconds)

    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule,
                                           overwrite_existing = true)

    t0 = time()
    status = :ok
    err_str = ""
    try
        run!(sim)
    catch e
        status = :crashed
        err_str = string(typeof(e))
    end
    elapsed = time() - t0
    return (; label, iter = iteration(sim), t = sim.model.clock.time,
            wmax = Float64(maximum(abs, interior(sim.model.velocities.w))),
            elapsed, status, err_str)
end

# Builders
function build_anelastic(grid)
    constants = ThermodynamicConstants(eltype(grid))
    ref = ReferenceState(grid, constants; potential_temperature = θᵇᵍ)
    dyn = AnelasticDynamics(ref)
    return AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9))
end

function build_compressible_substepper(grid, Ns)
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(substeps = Ns,
                                          forward_weight = 0.8,
                                          damping = PressureProjectionDamping(coefficient = 0.5))
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function build_compressible_explicit(grid)
    constants = ThermodynamicConstants(eltype(grid))
    dyn = CompressibleDynamics(ExplicitTimeStepping();
                               reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants)
end

results = NamedTuple[]

@info "=== 1/4: anelastic Δt=1.0 ==="
push!(results, run_one(label = "anelastic", build_model = build_anelastic, Δt = 1.0))

@info "=== 2/4: substepper Ns=12 Δt=1.0 ==="
push!(results, run_one(label = "Ns12", Δt = 1.0,
                       build_model = g -> build_compressible_substepper(g, 12)))

@info "=== 3/4: substepper Ns=48 Δt=1.0 ==="
push!(results, run_one(label = "Ns48", Δt = 1.0,
                       build_model = g -> build_compressible_substepper(g, 48)))

@info "=== 4/4: compressible explicit Δt=0.1 ==="
push!(results, run_one(label = "explicit", build_model = build_compressible_explicit, Δt = 0.1))

@info "=== SUMMARY ==="
for r in results
    @info @sprintf("  %-10s  iter=%5d t=%6.1f max|w|=%6.2f  elapsed=%5.1fs  %s",
                   r.label, r.iter, r.t, r.wmax, r.elapsed,
                   r.status == :ok ? "OK" : "CRASH $(r.err_str)")
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)
@info "jld2 outputs in $OUTDIR"
