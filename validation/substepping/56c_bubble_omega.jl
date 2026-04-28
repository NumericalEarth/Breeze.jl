#####
##### Bubble run with various ω (and Klemp damping). See if stable for 300s.
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

function build_substepped(grid; ω, damping = NoDivergenceDamping())
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = ω, damping = damping)
    AtmosphereModel(grid;
        dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ),
        advection = WENO(order = 9), timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt)
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
    return (; label, wmax_final = Float64(maximum(abs, interior(w))),
              has_nan = any(isnan, parent(w)), times, wmax_log)
end

@info "=== explicit ==="
expl = run_one("expl", build_explicit; Δt = Δt_expl)
@info @sprintf("  expl wmax_final=%.4e", expl.wmax_final)

for ω in (0.55, 0.6, 0.65, 0.7)
    label = "ω=$(ω) NoDamp"
    @info "=== $label ==="
    r = run_one(label, grid -> build_substepped(grid; ω); Δt = Δt_subst)
    mark = r.has_nan ? "NaN" : "✓"
    ratio = isnan(r.wmax_final) ? NaN : r.wmax_final / expl.wmax_final
    @info @sprintf("  %s wmax_final=%.4e ratio=%.3f", mark, r.wmax_final, ratio)
    for (i, t) in enumerate(r.times)
        @info @sprintf("    t=%5.1f w=%.4e", t, r.wmax_log[i])
    end
end

for damp_coef in (0.05, 0.1, 0.2)
    label = "ω=0.55 Klemp(d=$damp_coef)"
    @info "=== $label ==="
    damping = KlempDivergenceDamping(coefficient = damp_coef)
    r = run_one(label, grid -> build_substepped(grid; ω = 0.55, damping); Δt = Δt_subst)
    mark = r.has_nan ? "NaN" : "✓"
    ratio = isnan(r.wmax_final) ? NaN : r.wmax_final / expl.wmax_final
    @info @sprintf("  %s wmax_final=%.4e ratio=%.3f", mark, r.wmax_final, ratio)
end
