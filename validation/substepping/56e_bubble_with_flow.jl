#####
##### Bubble + background flow (like IGW). Test whether Doppler shift
##### stabilizes the late-time instability.
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

function build_substepped(grid; ω = 0.55)
    td = SplitExplicitTimeDiscretization(substeps = 12, forward_weight = ω, damping = NoDivergenceDamping())
    AtmosphereModel(grid;
        dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ),
        advection = WENO(order = 9), timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder, U; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ_builder(grid), ρ = ref.density, u = (x,z) -> U)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, Float64(maximum(abs, interior(sim.model.velocities.w))))
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 20.0/Δt))))
    try; run!(sim); catch; end
    w = model.velocities.w
    return (; label,
              wmax_final = Float64(maximum(abs, interior(w))),
              wmax_overall = isempty(wmax_log) ? 0.0 : maximum(wmax_log),
              has_nan = any(isnan, parent(w)), times, wmax_log)
end

for U in (0.0, 5.0, 10.0, 20.0)
    @info "============= U₀=$U m/s ============="
    expl = run_one("expl_U$U", build_explicit, U; Δt = Δt_expl)
    subs = run_one("subs_U$U", build_substepped, U; Δt = Δt_subst)
    mark = subs.has_nan ? "NaN" : "✓"
    rfin = isnan(subs.wmax_final) ? NaN : subs.wmax_final / expl.wmax_final
    rov = isnan(subs.wmax_overall) ? NaN : subs.wmax_overall / expl.wmax_overall
    @info @sprintf("  expl wmax_final=%.4e wmax_overall=%.4e", expl.wmax_final, expl.wmax_overall)
    @info @sprintf("  %s subs wmax_final=%.4e (ratio=%.3f)  wmax_overall=%.4e (ratio=%.3f)",
                   mark, subs.wmax_final, rfin, subs.wmax_overall, rov)
    for (i, t) in enumerate(subs.times)
        @info @sprintf("    t=%5.1f  subs=%.4e", t, subs.wmax_log[i])
    end
end
