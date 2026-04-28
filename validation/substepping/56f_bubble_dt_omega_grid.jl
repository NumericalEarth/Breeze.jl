#####
##### Bubble: search Δt × ω space for STABLE substepper config.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz, Nx, Nz = 20e3, 10e3, 64, 64
const θ₀, N², r₀, Δθ, g_phys = 300.0, 1e-4, 2e3, 0.001, 9.80665
θᵇᵍ(z) = θ₀ * exp(N² * z / g_phys)
const STOP_T, Δt_expl = 300.0, 0.05

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

function build_substepped(grid; Ns = 12, ω = 0.55)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = ω, damping = NoDivergenceDamping())
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
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 50.0/Δt))))
    try; run!(sim); catch; end
    w = model.velocities.w
    return (; wmax_final = Float64(maximum(abs, interior(w))),
              wmax_overall = isempty(wmax_log) ? 0.0 : maximum(wmax_log),
              has_nan = any(isnan, parent(w)))
end

@info "=== explicit ground truth ==="
expl = run_one(build_explicit; Δt = Δt_expl)
@info @sprintf("  wmax_final=%.4e  wmax_overall=%.4e", expl.wmax_final, expl.wmax_overall)

@info "=== Δt × ω stability grid (Ns=12) ==="
@info @sprintf("  %5s | %5s | %s", "Δt", "ω", "result")
for Δt in (0.1, 0.25, 0.5, 1.0)
    for ω in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
        r = run_one(grid -> build_substepped(grid; ω); Δt = Δt)
        if r.has_nan
            @info @sprintf("  %5.2f | %5.2f | NaN", Δt, ω)
        else
            ratio_fin = r.wmax_final / expl.wmax_final
            ratio_ov  = r.wmax_overall / expl.wmax_overall
            @info @sprintf("  %5.2f | %5.2f | ✓  fin_ratio=%.3f  ov_ratio=%.3f", Δt, ω, ratio_fin, ratio_ov)
        end
    end
end
