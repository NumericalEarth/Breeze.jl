#####
##### Short bubble run (60s) — capture w(t) and crash time, compare with explicit
##### Goal: see whether the new horizontal PGF drive gives the right bubble
##### amplitude before any blow-up.
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

function build_explicit_model(grid)
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θᵇᵍ)
    AtmosphereModel(grid; dynamics, advection = WENO(order = 9))
end

function build_substepped_model(grid; Ns = 12, ω = 0.55)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = ω, damping = NoDivergenceDamping())
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    AtmosphereModel(grid; dynamics, advection = WENO(order = 9), timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    times = Float64[]; wmax_log = Float64[]; umax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        umax = Float64(maximum(abs, interior(sim.model.velocities.u)))
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, wmax)
        push!(umax_log, umax)
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 10.0/Δt))))
    try; run!(sim); catch e; @info "  crashed: $(sprint(showerror, e))"; end
    return (; label, times, wmax_log, umax_log)
end

@info "=== explicit (ground truth, 60s) ==="
expl = run_one("expl", build_explicit_model; Δt = Δt_expl)
@info "=== substepped (60s, 1s outer) ==="
subs = run_one("subs", build_substepped_model; Δt = Δt_subst)

@info "=== TIME SERIES (every ~1s) ==="
@info @sprintf("  %5s  %14s %14s    %14s %14s",
               "t", "expl_w", "expl_u", "subs_w", "subs_u")
let n = min(length(expl.times), length(subs.times))
    for i in 1:n
        # Find matching expl entry
        ie = findfirst(t -> isapprox(t, subs.times[i]; atol=0.5), expl.times)
        ie === nothing && continue
        @info @sprintf("  %5.1f  %.4e  %.4e   %.4e  %.4e",
                       subs.times[i], expl.wmax_log[ie], expl.umax_log[ie],
                       subs.wmax_log[i], subs.umax_log[i])
    end
end
