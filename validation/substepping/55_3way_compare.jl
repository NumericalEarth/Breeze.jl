#####
##### validation/substepping/55_3way_compare.jl
#####
##### Three-way comparison: anelastic, fully explicit compressible, and
##### substepped compressible. Small-amplitude bubble in a stratified
##### atmosphere. The three should agree closely (O(δθ/θ)² relative
##### error) — anything more is a substepper bug.
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

const CASE   = "3way_compare"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

# Small bubble in moderately-stratified atmosphere on a coarse grid.
const Lx = 20e3
const Lz = 10e3
const Nx = 64
const Nz = 64
const θ₀_ref = 300.0
const N²     = 1e-4
const r₀     = 2e3
const Δθ     = 0.1                # small — linear-ish regime
const g_phys = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g_phys)

const STOP_T = 300.0
const Δt_anelastic = 1.0
const Δt_substep   = 1.0
const Δt_explicit  = 0.05         # acoustic-CFL bound for explicit

build_grid() = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                               x = (-Lx/2, Lx/2), z = (0, Lz),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
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

function build_explicit_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                    reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants)
end

function build_substepped_model(grid; ω = 0.55, damping = NoDivergenceDamping())
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = 12,
                                         forward_weight = ω,
                                         damping = damping)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_one(label, builder; Δt, stop_time = STOP_T)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ_builder(grid))
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ_builder(grid), ρ = ref.density)
    end

    sim = Simulation(model; Δt, stop_time, verbose = false)

    times = Float64[]; wmax_log = Float64[]; w_at_center_log = Float64[]
    function _track(sim)
        wmax = Float64(maximum(abs, interior(sim.model.velocities.w)))
        # Sample w at the bubble center
        wc = let w = sim.model.velocities.w
            iw = Nx ÷ 2 + 1; kw = Int(round(0.3 * Nz)) + 1
            arr = Array(interior(w))
            Float64(arr[iw, 1, kw])
        end
        push!(times, Float64(sim.model.clock.time))
        push!(wmax_log, wmax)
        push!(w_at_center_log, wc)
        if mod(iteration(sim), 100) == 0
            @info @sprintf("[%s] iter=%5d t=%6.1fs max|w|=%.4f w_c=%+.4f",
                           label, iteration(sim), sim.model.clock.time, wmax, wc)
        end
    end
    add_callback!(sim, _track, IterationInterval(50))

    sim.output_writers[:jld2] = JLD2Writer(model,
        (; w = model.velocities.w);
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
              status, err, times, wmax_log, w_at_center_log)
end

results = NamedTuple[]
@info "=== anelastic ==="
push!(results, run_one("anelastic", build_anelastic_model; Δt = Δt_anelastic))
@info "=== explicit-compressible ==="
push!(results, run_one("explicit", build_explicit_model; Δt = Δt_explicit))
for (lbl, ω, damping) in [
    ("subst_w55_nodamp", 0.55, NoDivergenceDamping()),
    ("subst_w7_nodamp",  0.7,  NoDivergenceDamping()),
    ("subst_w9_nodamp",  0.9,  NoDivergenceDamping()),
    ("subst_w55_klemp01", 0.55, KlempDivergenceDamping(coefficient = 0.1)),
    ("subst_w55_klemp1",  0.55, KlempDivergenceDamping(coefficient = 1.0)),
]
    @info "=== $lbl ==="
    push!(results, run_one(lbl,
                           grid -> build_substepped_model(grid; ω, damping);
                           Δt = Δt_substep))
end

@info "=== SUMMARY ==="
for r in results
    mark = r.has_nan ? "NaN" : (r.status == :ok ? "✓" : "✗")
    @info @sprintf("  %3s %-16s t=%6.1fs final max|w|=%.5f  (%6.1fs wall)",
                   mark, r.label, r.t, r.wmax_final, r.elapsed)
end

let
    fig = Figure(size = (1100, 700))
    ax1 = Axis(fig[1, 1]; xlabel = "t (s)", ylabel = "max |w| (m/s)",
               title = "Three-way: anelastic vs explicit vs substepped (Δθ=$(Δθ)K)")
    ax2 = Axis(fig[2, 1]; xlabel = "t (s)", ylabel = "w at bubble center (m/s)")
    for r in results
        lines!(ax1, r.times, r.wmax_log; label = r.label, linewidth = 1.6)
        lines!(ax2, r.times, r.w_at_center_log; label = r.label, linewidth = 1.6)
    end
    axislegend(ax1, position = :rb, framevisible = false)
    axislegend(ax2, position = :rb, framevisible = false)
    save(joinpath(OUTDIR, "compare.png"), fig)
    @info "wrote compare.png"
end
