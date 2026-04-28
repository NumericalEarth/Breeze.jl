#####
##### Test if proper Klemp 2018 divergence damping stabilizes the bubble at Δt=1
##### (where the substepper would otherwise NaN due to soft outer-step CFL).
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz, Nx, Nz = 20e3, 10e3, 64, 64
const θ₀, N², r₀, Δθ, g_phys = 300.0, 1e-4, 2e3, 0.001, 9.80665
θᵇᵍ(z) = θ₀ * exp(N² * z / g_phys)
const STOP_T, Δt_expl, Δt_subst = 300.0, 0.05, 1.0

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

θᵢ_builder(grid) = let
    x₀ = mean(xnodes(grid, Center())); z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * exp(-((x-x₀)^2 + (z-z₀)^2) / r₀^2)
end

build_explicit(grid) = AtmosphereModel(grid;
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θᵇᵍ),
    advection = WENO(order = 9))

function build_substepped(grid; ω = 0.55, damping = NoDivergenceDamping())
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = ω, damping = damping)
    AtmosphereModel(grid;
        dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ),
        advection = WENO(order = 9), timestepper = :AcousticRungeKutta3)
end

function run_one(builder; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, Float64(maximum(abs, interior(sim.model.velocities.w))))
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 30.0/Δt))))
    try; run!(sim); catch; end
    w = model.velocities.w
    return (; wmax_final = Float64(maximum(abs, interior(w))),
              wmax_overall = isempty(wmax_log) ? 0.0 : maximum(wmax_log),
              has_nan = any(isnan, parent(w)))
end

@info "=== explicit ground truth ==="
expl = run_one(build_explicit; Δt = Δt_expl)
@info @sprintf("  wmax_final=%.4e  wmax_overall=%.4e", expl.wmax_final, expl.wmax_overall)

@info "=== Δt=1 with Klemp damping (sweep coefficient) ==="
for coef in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    damp = KlempDivergenceDamping(coefficient = coef)
    r = run_one(grid -> build_substepped(grid; ω = 0.55, damping = damp); Δt = Δt_subst)
    if r.has_nan
        @info @sprintf("  coef=%-8.1e | NaN", coef)
    else
        ratio_fin = r.wmax_final / expl.wmax_final
        ratio_ov  = r.wmax_overall / expl.wmax_overall
        @info @sprintf("  coef=%-8.1e | ✓  fin=%.3f  ov=%.3f", coef, ratio_fin, ratio_ov)
    end
end

@info "=== Δt=1, no damping (baseline) ==="
r = run_one(grid -> build_substepped(grid; ω = 0.55); Δt = Δt_subst)
mark = r.has_nan ? "NaN" : "✓"
@info @sprintf("  %s wmax_final=%.4e", mark, r.wmax_final)
