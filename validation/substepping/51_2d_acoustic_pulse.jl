#####
##### validation/substepping/51_2d_acoustic_pulse.jl
#####
##### Test (ii) on the new-substepper validation ladder: 2D Gaussian
##### acoustic pulse in (Periodic, Flat, Bounded) topology, g = 0, no
##### damping. Adds horizontal acoustic coupling on top of test (i).
#####
##### Pass criterion (centered CN, no damping):
#####   - max|w| stays bounded — no exponential growth.
#####   - Ns-consistency: final max|w| converges as Ns → ∞ (Baldauf 2010 eq. 14).
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "2d_acoustic_pulse"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Lx = 20e3
const Lz = 10e3
const Nx = 64
const Nz = 64
const θ₀ = 300.0
const Δθ_pulse = 1e-3        # tiny — purely linear acoustic regime
const σ_pulse = 1e3
const x_pulse = 0.0
const z_pulse = Lz / 2

const c_s = sqrt(1.4 * 287.0 * θ₀)
const STOP_T = 60.0
const Δt = 0.5

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

θᵢ(x, z) = θ₀ + Δθ_pulse * exp(-((x - x_pulse)^2 + (z - z_pulse)^2) / (2σ_pulse^2))

function build_substepped_model(; Ns, ω = 0.5, damping = NoDivergenceDamping())
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = 0.0)
    td  = SplitExplicitTimeDiscretization(substeps = Ns; forward_weight = ω, damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = z -> θ₀)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = Centered(order = 2),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns, ω = 0.5, damping = NoDivergenceDamping(),
                 Δt = Δt, stop_time = STOP_T)
    model = build_substepped_model(; Ns, ω, damping)
    ref   = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt, stop_time, verbose = false)

    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(times, Float64(sim.model.clock.time)); push!(wmax_log, wmax)
        if mod(iteration(sim), 20) == 0
            @info @sprintf("[%s] iter=%4d t=%6.2fs max|w|=%.3e",
                           label, iteration(sim), sim.model.clock.time, wmax)
        end
    end
    add_callback!(sim, _track, IterationInterval(2))

    sim.output_writers[:jld2] = JLD2Writer(model, (; w = model.velocities.w);
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(2.0),
                                           overwrite_existing = true)

    t0 = time(); status = :ok; err = ""
    try; run!(sim); catch e; status = :crashed; err = sprint(showerror, e); end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    return (; label, Ns, ω, t = Float64(model.clock.time),
              wmax_final, has_nan, elapsed, status, err, times, wmax_log)
end

results = NamedTuple[]
for Ns in (6, 12, 24, 48)
    label = @sprintf("Ns%02d_centered_nodamp", Ns)
    @info "=== $label ==="
    push!(results, run_one(label; Ns))
end

@info "=== SUMMARY (centered CN, no damping; expect bounded max|w|, Ns-converged) ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-30s Ns=%2d ω=%.2f t=%6.2fs final max|w|=%.3e  (%5.1fs)",
                   mark, r.label, r.Ns, r.ω, r.t, r.wmax_final, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)

let
    fig = Figure(size = (1000, 450))
    ax = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
              title = "2D Gaussian acoustic pulse — centered CN, no damping")
    for r in results
        lines!(ax, r.times, r.wmax_log; label = @sprintf("Ns=%d", r.Ns), linewidth = 2)
    end
    axislegend(ax, position = :rt)
    save(joinpath(OUTDIR, "wmax_vs_t.png"), fig)
    @info "wrote wmax_vs_t.png"
end
