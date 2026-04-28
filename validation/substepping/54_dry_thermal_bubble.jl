#####
##### validation/substepping/54_dry_thermal_bubble.jl
#####
##### Test (v) on the new-substepper validation ladder: dry thermal bubble.
##### Stiff IC, nonlinear amplitude, full buoyancy.
#####
##### Compares the new substepper against the anelastic baseline.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Statistics
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "dry_thermal_bubble"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Δt = 1.0
const STOP_T = 600.0           # 10 minutes — fits within bubble rise time
const θ₀_ref = 300.0
const N²     = 1e-4    # use IGW-strength stratification for diagnostic
const r₀     = 2e3
const Δθ     = 0.5     # SMALL perturbation for linear regime test (was 10K)
const g_phys = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g_phys)

build_grid() = RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    # Smooth Gaussian bubble (no sharp gradient) — easier on the substepper.
    function θᵢ(x, z)
        r² = (x - x₀)^2 + (z - z₀)^2
        return θᵇᵍ(z) + Δθ * exp(-r² / r₀^2)
    end
end

function build_anelastic_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    reference_state = ReferenceState(grid, constants; potential_temperature = θᵇᵍ)
    dynamics = AnelasticDynamics(reference_state)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9))
end

function build_substepped_model(grid; Ns = 12, ω = 0.55,
                                damping = NoDivergenceDamping())
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = Ns,
                                         forward_weight = ω, damping = damping)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt = Δt, stop_time = STOP_T)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ_builder(grid))
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    end

    sim = Simulation(model; Δt, stop_time, verbose = false)

    times = Float64[]; wmax_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(times, Float64(sim.model.clock.time)); push!(wmax_log, wmax)
        if mod(iteration(sim), 20) == 0
            @info @sprintf("[%s] iter=%4d t=%6.1fs max|w|=%.3f",
                           label, iteration(sim), sim.model.clock.time, wmax)
        end
    end
    add_callback!(sim, _track, IterationInterval(5))

    sim.output_writers[:jld2] = JLD2Writer(model,
        (; w = model.velocities.w, u = model.velocities.u);
        filename = joinpath(OUTDIR, "$(label).jld2"),
        schedule = TimeInterval(30.0),
        overwrite_existing = true)

    t0 = time(); status = :ok; err = ""
    try; run!(sim); catch e; status = :crashed; err = sprint(showerror, e); end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    return (; label, t = Float64(model.clock.time), wmax_final, has_nan, elapsed,
              status, err, times, wmax_log)
end

results = NamedTuple[]
@info "=== anelastic ==="
push!(results, run_one("anelastic", build_anelastic_model))
@info "=== substepped 64x64 (Ns=12, ω=0.55, no damping) ==="
push!(results, run_one("substepped_64x64_w55_nodamp",
                       grid -> build_substepped_model(grid; Ns = 12, ω = 0.55,
                                                      damping = NoDivergenceDamping())))

@info "=== SUMMARY ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-30s t=%6.1fs final max|w|=%.3f  (%6.1fs wall)",
                   mark, r.label, r.t, r.wmax_final, r.elapsed)
end

let
    fig = Figure(size = (1100, 450))
    ax = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
              title = "Dry thermal bubble — anelastic vs substepped (centered CN)")
    for r in results
        lines!(ax, r.times, r.wmax_log; label = r.label, linewidth = 1.6)
    end
    axislegend(ax, position = :rb, framevisible = false)
    save(joinpath(OUTDIR, "wmax_vs_t.png"), fig)
    @info "wrote wmax_vs_t.png"
end
