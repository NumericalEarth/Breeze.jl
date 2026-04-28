#####
##### 1D buoyancy oscillator: sweep Ns and Δt to find the source of the 5× suppression.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf

const arch = CUDA.functional() ? GPU() : CPU()
const Lz, Nz = 10e3, 64
const θ₀, N², g_phys = 300.0, 1e-4, 9.80665
const Δθ_pulse, σ_z, z_pulse = 1e-4, 500.0, 5e3
θ̄(z) = θ₀ * exp(N² * z / g_phys)
θᵢ(x, z) = θ̄(z) + Δθ_pulse * exp(-((z - z_pulse) / σ_z)^2)
θᵢ(z) = θᵢ(0, z)

const STOP_T, Δt_expl = 600.0, 0.05

build_grid() = RectilinearGrid(arch; size = (Nz,), halo = (5,),
                               z = (0, Lz), topology = (Flat, Flat, Bounded))

build_explicit(grid) = AtmosphereModel(grid;
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θ̄),
    advection = Centered(order = 2))

function build_substepped(grid; Ns = 12, ω = 0.55)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = ω, damping = NoDivergenceDamping())
    AtmosphereModel(grid;
        dynamics = CompressibleDynamics(td; reference_potential_temperature = θ̄),
        advection = Centered(order = 2), timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, Float64(maximum(abs, interior(sim.model.velocities.w))))
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 5.0/Δt))))
    try; run!(sim); catch; end
    w = model.velocities.w
    return (; label,
              wmax_final = Float64(maximum(abs, interior(w))),
              wmax_overall = isempty(wmax_log) ? 0.0 : maximum(wmax_log),
              has_nan = any(isnan, parent(w)),
              times, wmax_log)
end

@info "=== explicit ==="
expl = run_one("expl", build_explicit; Δt = Δt_expl)
@info @sprintf("  expl wmax_overall=%.4e wmax_final=%.4e", expl.wmax_overall, expl.wmax_final)

@info "=== Ns sweep at Δt=1, ω=0.55 ==="
for Ns in (6, 12, 24, 48, 96)
    r = run_one("Ns=$Ns", grid -> build_substepped(grid; Ns); Δt = 1.0)
    mark = r.has_nan ? "NaN" : "✓"
    rov = r.wmax_overall / expl.wmax_overall
    @info @sprintf("  %s Ns=%2d wmax_overall=%.4e ratio=%.3f", mark, Ns, r.wmax_overall, rov)
end

@info "=== Δt sweep at Ns=12, ω=0.55 ==="
for Δt in (0.05, 0.1, 0.5, 1.0, 2.0)
    r = run_one("Δt=$Δt", grid -> build_substepped(grid); Δt = Δt)
    mark = r.has_nan ? "NaN" : "✓"
    rov = isnan(r.wmax_overall) ? NaN : r.wmax_overall / expl.wmax_overall
    @info @sprintf("  %s Δt=%.2f wmax_overall=%.4e ratio=%.3f", mark, Δt, r.wmax_overall, rov)
end

@info "=== ω sweep at Ns=12, Δt=1 ==="
for ω in (0.55, 0.6, 0.7, 0.8)
    r = run_one("ω=$ω", grid -> build_substepped(grid; ω); Δt = 1.0)
    mark = r.has_nan ? "NaN" : "✓"
    rov = isnan(r.wmax_overall) ? NaN : r.wmax_overall / expl.wmax_overall
    @info @sprintf("  %s ω=%.2f wmax_overall=%.4e ratio=%.3f", mark, ω, r.wmax_overall, rov)
end

@info "=== Time series at Ns=12, Δt=1, ω=0.55 ==="
sub = run_one("default", grid -> build_substepped(grid); Δt = 1.0)
for i in 1:5:length(sub.times)
    ie = findfirst(t -> isapprox(t, sub.times[i]; atol=2.5), expl.times)
    if ie !== nothing
        @info @sprintf("  t=%5.1f  expl=%.4e  subs=%.4e  ratio=%.3f",
                       sub.times[i], expl.wmax_log[ie], sub.wmax_log[i],
                       sub.wmax_log[i]/max(expl.wmax_log[ie], 1e-30))
    end
end
