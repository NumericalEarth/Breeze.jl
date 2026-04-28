#####
##### IGW with various ω to see if the ~12% under-prediction is off-centering damping.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lx, Lz, Nx, Nz = 300e3, 10e3, 300, 50
const θ₀, N², g_phys = 300.0, 1e-4, 9.80665
const σ_pulse_x = 5e3
const U₀ = 20.0
const Δθ_pulse = 1e-2
θ̄(z) = θ₀ * exp(N² * z / g_phys)

const STOP_T = 600.0
const Δt_subst, Δt_expl = 1.0, 0.05

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

θᵢ(x, z) = θ̄(z) + Δθ_pulse * sin(π * z / Lz) / (1 + (x / σ_pulse_x)^2)

build_explicit(grid) = AtmosphereModel(grid;
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θ̄),
    advection = WENO(order = 5))

function build_substepped(grid; Ns = 12, ω)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = ω, damping = NoDivergenceDamping())
    AtmosphereModel(grid;
        dynamics = CompressibleDynamics(td; reference_potential_temperature = θ̄),
        advection = WENO(order = 5), timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density, u = (x,z) -> U₀)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, Float64(maximum(abs, interior(sim.model.velocities.w))))
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 60.0/Δt))))
    try; run!(sim); catch; end
    w = model.velocities.w
    return (; label, wmax_final = Float64(maximum(abs, interior(w))),
              has_nan = any(isnan, parent(w)), times, wmax_log,
              wmax_overall = isempty(wmax_log) ? 0.0 : maximum(wmax_log))
end

@info "=== explicit ==="
expl = run_one("expl", build_explicit; Δt = Δt_expl)
@info @sprintf("  expl wmax_final=%.4e wmax_overall=%.4e", expl.wmax_final, expl.wmax_overall)

for ω in (0.50, 0.51, 0.52, 0.53, 0.55, 0.6, 0.7)
    @info "=== ω=$ω ==="
    r = run_one("ω=$ω", grid -> build_substepped(grid; ω); Δt = Δt_subst)
    mark = r.has_nan ? "NaN" : "✓"
    rfin = isnan(r.wmax_final) ? NaN : r.wmax_final / expl.wmax_final
    rov = isnan(r.wmax_overall) ? NaN : r.wmax_overall / expl.wmax_overall
    @info @sprintf("  %s wmax_final=%.4e (ratio=%.3f)  wmax_overall=%.4e (ratio=%.3f)",
                   mark, r.wmax_final, rfin, r.wmax_overall, rov)
end
