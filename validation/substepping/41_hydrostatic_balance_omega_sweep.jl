#####
##### validation/substepping/41_hydrostatic_balance_omega_sweep.jl
#####
##### Follow-up to test 40: sweep `forward_weight` ω at the marginally-stable
##### (Ns=24) and broken (Ns=48) substep counts on the hydrostatic-balance
##### regression. Disambiguates:
#####   - Off-centering interaction:  ω=0.5 (centered, ε=0) more stable than ω=0.7
#####   - Core-scheme bug:            ω=0.5 unstable too → bug independent of ε
#####
##### Each (Ns, ω) case runs the same rest atmosphere as test 40 for 600s and
##### records the final max|w|. A correct discretization keeps it at machine
##### epsilon (~1e-13) for Float64.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()

const CASE   = "hydrostatic_balance_omega_sweep"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 600.0
const Δt     = 1.0
const g      = 9.80665
const θ₀ = 300.0
const N² = 1e-4
θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z)

build_grid() = RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

function build_substepped_model(; Ns, ω,
                                damping = PressureProjectionDamping(coefficient = 0.5))
    grid = build_grid()
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization(substeps = Ns; forward_weight = ω, damping)
    dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics = dyn,
                           advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label; Ns, ω)
    model = build_substepped_model(; Ns, ω)
    ref   = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)

    drift = Float64[]; times = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        push!(drift, wmax)
        push!(times, Float64(sim.model.clock.time))
        if mod(iteration(sim), 100) == 0
            @info @sprintf("[%s] iter=%4d t=%5.1fs max|w|=%.3e",
                           label, iteration(sim), sim.model.clock.time, wmax)
        end
    end
    add_callback!(sim, _track, IterationInterval(20))

    t0 = time()
    status = :ok; err = ""
    try
        run!(sim)
    catch e
        status = :crashed; err = sprint(showerror, e)
    end
    elapsed = time() - t0

    w = model.velocities.w
    wmax_final = Float64(maximum(abs, interior(w)))
    has_nan = any(isnan, parent(w))

    return (; label, Ns, ω,
              t = Float64(model.clock.time),
              wmax_final, has_nan, elapsed, status, err,
              times, drift)
end

results = NamedTuple[]
for Ns in (24, 48)
    for ω in (0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
        label = @sprintf("Ns%02d_w%.2f", Ns, ω)
        @info "=== $label ==="
        push!(results, run_one(label; Ns, ω))
    end
end

@info "=== SUMMARY ==="
@info @sprintf("  %-20s  %-8s  %-9s  %s", "label", "Ns", "ω", "final max|w|")
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-15s Ns=%2d ω=%.2f final max|w|=%.3e  (%5.1fs)",
                   mark, r.label, r.Ns, r.ω, r.wmax_final, r.elapsed)
end
jldsave(joinpath(OUTDIR, "summary.jld2"); results)

# Drift vs t, log-y, panel per Ns, line per ω
let
    fig = Figure(size = (1300, 460))
    for (col, Ns_val) in enumerate((24, 48))
        ax = Axis(fig[1, col]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
                  yscale = log10,
                  title = "Hydrostatic-balance drift, Ns=$(Ns_val)")
        for r in results
            r.Ns == Ns_val || continue
            d = map(x -> x == 0 ? 1e-16 : x, r.drift)
            lines!(ax, r.times, d; label = @sprintf("ω=%.2f", r.ω), linewidth = 2)
        end
        axislegend(ax, position = :rb)
    end
    save(joinpath(OUTDIR, "drift_vs_t.png"), fig)
    @info "wrote drift_vs_t.png"
end
