#####
##### Test the Δt × cs / Δx CFL hypothesis: at coarser Δx, larger Δt
##### should be stable.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Statistics, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz = 20e3, 10e3
const θ₀, N², r₀, Δθ, g_phys = 300.0, 1e-4, 2e3, 0.001, 9.80665
const cs = sqrt(1.4 * 287.0 * 300)
θᵇᵍ(z) = θ₀ * exp(N² * z / g_phys)

const STOP_T, Δt_expl = 300.0, 0.05

build_grid(Nx, Nz) = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                                     x = (-Lx/2, Lx/2), z = (0, Lz),
                                     topology = (Periodic, Flat, Bounded))

θᵢ_builder(grid) = let
    x₀ = mean(xnodes(grid, Center())); z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * exp(-((x-x₀)^2 + (z-z₀)^2) / r₀^2)
end

build_explicit(grid) = AtmosphereModel(grid;
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θᵇᵍ),
    advection = WENO(order = 9))

function build_substepped(grid)
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = 0.55, damping = NoDivergenceDamping())
    AtmosphereModel(grid;
        dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ),
        advection = WENO(order = 9), timestepper = :AcousticRungeKutta3)
end

function run_one(builder, grid; Δt)
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

@info "=== Test: Δt × cs / Δx > 1 → instability ==="
@info @sprintf("  %-9s | %-7s | %s", "grid", "CFL", "result")
for (Nx, Nz, Δt) in [(64, 64, 1.0), (32, 32, 1.0), (32, 32, 2.0), (32, 32, 4.0),
                     (16, 16, 4.0), (16, 16, 8.0)]
    Δx = Lx / Nx
    cfl = cs * Δt / Δx
    grid = build_grid(Nx, Nz)
    expl = run_one(build_explicit, grid; Δt = Δt_expl)
    subs = run_one(build_substepped, grid; Δt = Δt)
    if subs.has_nan
        @info @sprintf("  %2d×%2d Δt=%4.1f | %.2f | NaN  (expl=%.4e)", Nx, Nz, Δt, cfl, expl.wmax_final)
    else
        ratio = subs.wmax_overall / expl.wmax_overall
        @info @sprintf("  %2d×%2d Δt=%4.1f | %.2f | ✓ ratio=%.3f  (expl=%.4e)", Nx, Nz, Δt, cfl, ratio, expl.wmax_overall)
    end
end
