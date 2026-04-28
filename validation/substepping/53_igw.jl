#####
##### validation/substepping/53_igw.jl
#####
##### Test (iv) on the new-substepper validation ladder: linear inertia-
##### gravity wave (IGW) with buoyancy and slow advection. Adds the buoyancy
##### coupling on top of test (iii).
#####
##### Setup follows Skamarock & Klemp 1994:
#####   - 2D channel, x ∈ [-150 km, 150 km], z ∈ [0, 10 km]
#####   - Stably stratified background: θ̄(z) = θ₀ exp(N² z / g), N² = 1e-4
#####   - Small θ′ perturbation centered at (x = 0, z = Lz/2)
#####   - Light background u = 20 m/s for non-trivial slow advection
#####
##### Pass criterion (centered CN, no damping):
#####   - max|w| stays bounded; gravity-wave amplitude consistent with linear
#####     theory (~few cm/s for a 1e-2 K perturbation in N²=1e-4 air).
#####   - Both Centered and WENO advection give comparable wave fields.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "igw"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Lx = 300e3
const Lz = 10e3
const Nx = 300
const Nz = 50
const θ₀ = 300.0
const N² = 1e-4
const g  = 9.80665
const Δθ_pulse = 1e-2        # K — small perturbation, linear regime
const σ_pulse_x = 5e3
const z_pulse = Lz / 2
const U₀ = 20.0              # background flow

const STOP_T = 600.0         # 10 minutes for diagnostic
const Δt = 1.0

θ̄(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θ̄(z) + Δθ_pulse * sin(π * z / Lz) / (1 + (x / σ_pulse_x)^2)
uᵢ(x, z) = U₀

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

function build_substepped_model(; Ns, ω = 0.55, advection,
                                damping = NoDivergenceDamping())
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(substeps = Ns; forward_weight = ω, damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ̄)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = advection,
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns, ω = 0.55, advection,
                 damping = NoDivergenceDamping(),
                 Δt = Δt, stop_time = STOP_T)
    model = build_substepped_model(; Ns, ω, advection, damping)
    ref   = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density, u = uᵢ)

    sim = Simulation(model; Δt, stop_time, verbose = false)

    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(times, Float64(sim.model.clock.time)); push!(wmax_log, wmax)
        if mod(iteration(sim), 20) == 0
            @info @sprintf("[%s] iter=%4d t=%6.1fs max|w|=%.3e",
                           label, iteration(sim), sim.model.clock.time, wmax)
        end
    end
    add_callback!(sim, _track, IterationInterval(2))

    sim.output_writers[:jld2] = JLD2Writer(model,
        (; w = model.velocities.w);
        filename = joinpath(OUTDIR, "$(label).jld2"),
        schedule = TimeInterval(60.0),
        overwrite_existing = true)

    t0 = time(); status = :ok; err = ""
    try; run!(sim); catch e; status = :crashed; err = sprint(showerror, e); end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    return (; label, Ns, ω, advection_kind = string(typeof(advection).name.name),
              t = Float64(model.clock.time),
              wmax_final, has_nan, elapsed, status, err, times, wmax_log)
end

results = NamedTuple[]
for advection in (Centered(order = 2), WENO(order = 5))
    adv_name = string(typeof(advection).name.name)
    for Ns in (12, 24)
        label = @sprintf("%s_Ns%02d", adv_name, Ns)
        @info "=== $label ==="
        push!(results, run_one(label; Ns, advection))
    end
end

@info "=== SUMMARY ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-25s Ns=%2d %s  final max|w|=%.3e  (%6.1fs)",
                   mark, r.label, r.Ns, r.advection_kind, r.wmax_final, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)

let
    fig = Figure(size = (1100, 450))
    ax = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
              title = "IGW — centered CN, no damping (Skamarock-Klemp 1994 setup)")
    for r in results
        lines!(ax, r.times, r.wmax_log; label = r.label, linewidth = 1.6)
    end
    axislegend(ax, position = :rt, framevisible = false)
    save(joinpath(OUTDIR, "wmax_vs_t.png"), fig)
    @info "wrote wmax_vs_t.png"
end
