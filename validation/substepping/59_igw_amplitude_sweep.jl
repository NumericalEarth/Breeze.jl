#####
##### validation/substepping/59_igw_amplitude_sweep.jl
#####
##### Skamarock-Klemp 1994 IGW with sweep over perturbation amplitude.
##### Compare substepper vs explicit-compressible vs anelastic at each Δθ.
##### Goal: ratio sub/expl ≈ 1.0 across the sweep, AND no late-time NaN.
#####

include("common.jl")

using Breeze, Oceananigans, Oceananigans.Units, CUDA, Printf

const arch = CUDA.functional() ? GPU() : CPU()

const Lx, Lz, Nx, Nz = 300e3, 10e3, 300, 50
const θ₀, N², g_phys = 300.0, 1e-4, 9.80665
const σ_pulse_x = 5e3
const U₀ = 20.0
θ̄(z) = θ₀ * exp(N² * z / g_phys)

const STOP_T = 600.0
const Δt_subst, Δt_expl = 1.0, 0.05

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

θᵢ_builder(Δθ) = (x, z) -> θ̄(z) + Δθ * sin(π * z / Lz) / (1 + (x / σ_pulse_x)^2)

function build_explicit(grid)
    dyn = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature = θ̄)
    AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 5))
end

function build_substepped(grid; Ns = 12, ω = 0.55)
    td = SplitExplicitTimeDiscretization(substeps = Ns, forward_weight = ω, damping = NoDivergenceDamping())
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ̄)
    AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 5),
                    timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder, Δθ; Δt)
    grid = build_grid()
    model = builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ_builder(Δθ), ρ = ref.density, u = (x,z) -> U₀)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, Float64(maximum(abs, interior(sim.model.velocities.w))))
    end
    add_callback!(sim, _track, IterationInterval(max(1, round(Int, 60.0/Δt))))
    try; run!(sim); catch; end
    w = model.velocities.w
    return (; label, Δθ,
              wmax_final = Float64(maximum(abs, interior(w))),
              has_nan = any(isnan, parent(w)), times, wmax_log)
end

results = NamedTuple[]
for Δθ in (1e-2, 1e-1, 1.0, 5.0)
    @info "=========== Δθ = $Δθ ==========="
    @info "  --- explicit ---"
    push!(results, run_one("expl_$(Δθ)", build_explicit, Δθ; Δt = Δt_expl))
    @info "  --- substepped ---"
    push!(results, run_one("subs_$(Δθ)", build_substepped, Δθ; Δt = Δt_subst))
end

@info "=== SUMMARY ==="
for Δθ in (1e-2, 1e-1, 1.0, 5.0)
    e = filter(r -> r.label == "expl_$(Δθ)" && !r.has_nan, results)
    s = filter(r -> r.label == "subs_$(Δθ)" && !r.has_nan, results)
    en = filter(r -> r.label == "expl_$(Δθ)" &&  r.has_nan, results)
    sn = filter(r -> r.label == "subs_$(Δθ)" &&  r.has_nan, results)
    if length(e) == 1 && length(s) == 1
        ratio = s[1].wmax_final / e[1].wmax_final
        @info @sprintf("  Δθ=%6.3f  expl=%.4e  subs=%.4e  ratio=%.3f", Δθ, e[1].wmax_final, s[1].wmax_final, ratio)
    elseif length(e) == 1 && !isempty(sn)
        @info @sprintf("  Δθ=%6.3f  expl=%.4e  subs=NaN", Δθ, e[1].wmax_final)
    end
end

@info "=== TIME SERIES (substepper) ==="
for r in filter(rr -> startswith(rr.label, "subs_"), results)
    @info "  $(r.label):"
    for (i, t) in enumerate(r.times)
        @info @sprintf("    t=%5.1f  w=%.4e", t, r.wmax_log[i])
    end
end
