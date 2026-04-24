#####
##### validation/substepping/23_bubble_g_sweep.jl
#####
##### Two follow-ups to the acoustic-pulse finding:
#####   A. The dry thermal bubble with NO gravity (g=0). If this is stable
#####      at Ns=96, the substepper bug is conclusively gravity-induced.
#####   B. Vary g from 0 to 9.81 at fixed Ns=48 to map the threshold at
#####      which the column-solve goes unstable.
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

const OUTDIR = joinpath(@__DIR__, "out", "bubble_g_sweep")
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 60.0   # short — we just need to see whether it crashes early
const Δt     = 1.0    # outer step

const θ₀_ref = 300.0
const N²     = 1e-6   # weak stratification (matches 07)
const r₀     = 2e3
const Δθ     = 10.0

# θᵇᵍ depends on g via the stratification — for g=0 the exp() collapses to 1.
make_θᵇᵍ(g) = z -> g == 0 ? θ₀_ref : θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

function θᵢ_builder(grid, θᵇᵍ)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r = sqrt((x - x₀)^2 + (z - z₀)^2)
        return θᵇᵍ(z) + Δθ * max(0, 1 - r / r₀)
    end
end

function run_one(; Ns, g)
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = g)
    θᵇᵍ = make_θᵇᵍ(g)
    td  = SplitExplicitTimeDiscretization(substeps = Ns,
                                          forward_weight = 0.8,
                                          damping = PressureProjectionDamping(coefficient = 0.5))
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    model = AtmosphereModel(grid; dynamics = dyn,
                            advection = WENO(order = 9),
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)
    set!(model; θ = θᵢ_builder(grid, θᵇᵍ), ρ = model.dynamics.reference_state.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    function _progress(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        @info @sprintf("[Ns=%d g=%.2f] iter=%4d t=%5.1f max|w|=%.3g",
                       Ns, g, iteration(sim), sim.model.clock.time, wmax)
    end
    add_callback!(sim, _progress, IterationInterval(10))

    t0 = time()
    status = :ok; err = ""
    try
        run!(sim)
    catch e
        status = :crashed; err = string(typeof(e))
    end
    elapsed = time() - t0
    has_nan = any(isnan, interior(sim.model.velocities.w))
    wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
    return (; Ns, g, iter = iteration(sim), t = sim.model.clock.time,
              wmax, has_nan, elapsed, status, err)
end

results = NamedTuple[]

@info "================ A: bubble with g=0, Ns sweep ================"
for Ns in [6, 12, 24, 48, 96]
    push!(results, run_one(; Ns, g = 0.0))
end

@info "================ B: bubble at Ns=48, g sweep ================"
for g in [0.0, 1.0, 3.0, 5.0, 9.80665]
    push!(results, run_one(; Ns = 48, g = g))
end

@info "================ SUMMARY ================"
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s Ns=%-3d g=%5.2f iter=%4d t=%5.1f wmax=%.3g %4.1fs",
                   mark, r.Ns, r.g, r.iter, r.t, r.wmax, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)
