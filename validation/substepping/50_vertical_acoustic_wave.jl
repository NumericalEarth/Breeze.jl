#####
##### validation/substepping/50_vertical_acoustic_wave.jl
#####
##### Test (i) on the new-substepper validation ladder: 1D vertical acoustic
##### wave. (Flat, Flat, Bounded) topology so only the implicit vertical
##### Schur core is exercised — no horizontal coupling, no slow advection,
##### no buoyancy.
#####
##### Setup:
#####   - 1D column, z ∈ [0, 10 km], Nz = 128, Bounded vertical with
#####     impenetrable w = 0 at top and bottom.
#####   - Uniform isothermal background, g = 0 (pure acoustic, no
#####     stratification interference).
#####   - Initial perturbation: small Gaussian (ρθ)′ centered at z = 5 km.
#####   - Run for one transit time so the wave reflects off the top and
#####     bottom and returns near its origin.
#####
##### Pass criterion (centered CN, no damping):
#####   - max|w| stays bounded (no exponential growth).
#####   - Energy in the perturbation is conserved to within machine tolerance.
#####   - The wave shape returns close to the initial pulse after a full
#####     round trip.
#####
##### This test is the cleanest possible exercise of the implicit Schur
##### tridiag for ρw–π coupling. If it fails, the Schur assembly is wrong.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "vertical_acoustic_wave"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Lz   = 10e3        # 10 km column
const Nz   = 128
const θ₀   = 300.0       # uniform background potential temperature (K)
const Δθ_pulse = 1e-3    # tiny perturbation, fully linear regime (K)
const σ_z  = 500.0       # Gaussian half-width (m)
const z_pulse = Lz / 2   # pulse center (m)

# Sound speed at the background state. With g = 0, T = θ Π and Π depends
# only on p / pˢᵗ. For a uniform isothermal state, c_s = √(γ R T) ≈ 347 m/s.
const c_s = sqrt(1.4 * 287.0 * θ₀)
# Round-trip transit time for the perturbation
const STOP_T = 2 * Lz / c_s    # ~57.6 s — out and back

build_grid() = RectilinearGrid(arch; size = (Nz,), halo = (5,),
                               z = (0, Lz),
                               topology = (Flat, Flat, Bounded))

# Tiny pulse on top of constant background θ₀
θᵢ(z) = θ₀ + Δθ_pulse * exp(-((z - z_pulse) / σ_z)^2)
θᵢ(x, z) = θᵢ(z)         # 2-arg form for set!

function build_substepped_model(; Ns, ω = 0.5,
                                damping = NoDivergenceDamping())
    grid = build_grid()
    # g = 0 so we test pure acoustic, no buoyancy
    constants = ThermodynamicConstants(eltype(grid); gravitational_acceleration = 0.0)
    td  = SplitExplicitTimeDiscretization(substeps = Ns; forward_weight = ω, damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = z -> θ₀)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = Centered(order = 2),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns, ω = 0.5, damping = NoDivergenceDamping(),
                 Δt = 0.05, stop_time = STOP_T)
    model = build_substepped_model(; Ns, ω, damping)
    ref   = model.dynamics.reference_state

    # Initial condition: θ = θ₀ + small Gaussian, ρ = ref.density (so the state
    # is a tiny perturbation off the reference column at rest).
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt, stop_time, verbose = false)

    times    = Float64[]
    wmax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, wmax)
        if mod(iteration(sim), 50) == 0
            @info @sprintf("[%s] iter=%4d t=%6.2fs max|w|=%.3e",
                           label, iteration(sim), sim.model.clock.time, wmax)
        end
    end
    add_callback!(sim, _track, IterationInterval(5))

    sim.output_writers[:jld2] = JLD2Writer(model, (; w = model.velocities.w);
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(1.0),
                                           overwrite_existing = true)

    t0 = time(); status = :ok; err = ""
    try; run!(sim); catch e; status = :crashed; err = sprint(showerror, e); end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    return (; label, Ns, ω,
              t = Float64(model.clock.time),
              wmax_final, has_nan, elapsed, status, err,
              times, wmax_log)
end

# Sweep Ns to test consistency. Run at ω=0.5 (centered CN) with no damping —
# this is the strict diagnostic. If any of these blow up, the scheme is
# wrong, period.
results = NamedTuple[]
for Ns in (6, 12, 24, 48)
    label = @sprintf("Ns%02d_centered_nodamp", Ns)
    @info "=== $label ==="
    push!(results, run_one(label; Ns, ω = 0.5, damping = NoDivergenceDamping()))
end

@info "=== SUMMARY (centered CN, no damping; expect bounded max|w|) ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-30s Ns=%2d ω=%.2f t=%6.2fs final max|w|=%.3e  (%5.1fs)",
                   mark, r.label, r.Ns, r.ω, r.t, r.wmax_final, r.elapsed)
end

jldsave(joinpath(OUTDIR, "summary.jld2"); results)

# Plot max|w|(t) for each Ns
let
    fig = Figure(size = (1000, 450))
    ax = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
              title = "1D vertical acoustic wave — centered CN, no damping (max|w| should stay bounded)")
    for r in results
        lines!(ax, r.times, r.wmax_log; label = @sprintf("Ns=%d", r.Ns), linewidth = 2)
    end
    axislegend(ax, position = :rt)
    save(joinpath(OUTDIR, "wmax_vs_t.png"), fig)
    @info "wrote wmax_vs_t.png"
end
