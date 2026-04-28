#####
##### Find optimal (Δt, Klemp coef) combinations for the bubble.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz, Nx, Nz = 20e3, 10e3, 64, 64
const θ₀, N², r₀, Δθ, g_phys = 300.0, 1e-4, 2e3, 0.001, 9.80665
θᵇᵍ(z) = θ₀ * exp(N² * z / g_phys)
const STOP_T = 300.0

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

θᵢ_builder(grid) = let
    x₀ = mean(xnodes(grid, Center())); z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * exp(-((x-x₀)^2 + (z-z₀)^2) / r₀^2)
end

function run_substepped(; Δt, coef)
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55,
                                         damping = KlempDivergenceDamping(coefficient = coef))
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    m = AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9),
                        timestepper = :AcousticRungeKutta3)
    set!(m; θ = θᵢ_builder(grid), ρ = m.dynamics.reference_state.density)
    sim = Simulation(m; Δt, stop_time = STOP_T, verbose = false)
    wmax = Ref(0.0)
    function _track(sim)
        wmax[] = max(wmax[], Float64(maximum(abs, interior(sim.model.velocities.w))))
        return nothing
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 30.0/Δt))))
    try; run!(sim); catch; end
    return (; wmax_overall = wmax[],
              has_nan = any(isnan, parent(m.velocities.w)))
end

# Explicit ground truth
let
    grid = build_grid()
    expl = AtmosphereModel(grid;
        dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θᵇᵍ),
        advection = WENO(order = 9))
    set!(expl; θ = θᵢ_builder(grid), ρ = expl.dynamics.reference_state.density)
    sim = Simulation(expl; Δt = 0.05, stop_time = STOP_T, verbose = false)
    global wmax_expl_ref = Ref(0.0)
    function _track_expl(sim)
        wmax_expl_ref[] = max(wmax_expl_ref[], Float64(maximum(abs, interior(sim.model.velocities.w))))
        return nothing
    end
    add_callback!(sim, _track_expl, IterationInterval(60))
    run!(sim)
    @info @sprintf("explicit wmax_overall=%.4e", wmax_expl_ref[])
end

@info "=== (Δt, coef) sweep ==="
@info @sprintf("  %5s | %5s | %s", "Δt", "coef", "result")
for Δt in (0.5, 1.0, 1.5, 2.0, 3.0)
    for coef in (0.1, 0.2, 0.5, 1.0)
        r = run_substepped(; Δt, coef)
        if r.has_nan
            @info @sprintf("  %5.2f | %5.2f | NaN", Δt, coef)
        else
            ratio = r.wmax_overall / wmax_expl_ref[]
            @info @sprintf("  %5.2f | %5.2f | ✓ ratio=%.3f", Δt, coef, ratio)
        end
    end
end
