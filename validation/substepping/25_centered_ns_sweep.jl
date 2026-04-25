#####
##### validation/substepping/25_centered_ns_sweep.jl
#####
##### Centered(2) advection, Ns sweep on the dry bubble.
##### Tests whether Centered(2) actually CURES the Ns-scaling instability
##### or just MITIGATES it. Includes Ns=96 as a stress.
#####
##### All in one Julia process so animation runs without re-precompile.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using CUDA
using CairoMakie
using Statistics
using Printf
using JLD2

CUDA.functional() || error("GPU required")
const arch = GPU()

const CASE = "centered_ns_sweep"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 60.0
const ADVECTION = Centered(order = 2)

const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(arch; size = (64, 64), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r = sqrt((x - x₀)^2 + (z - z₀)^2)
        return θᵇᵍ(z) + Δθ * max(0, 1 - r / r₀)
    end
end

function build_anelastic(grid)
    constants = ThermodynamicConstants(eltype(grid))
    ref = ReferenceState(grid, constants; potential_temperature = θᵇᵍ)
    dyn = AnelasticDynamics(ref)
    return AtmosphereModel(grid; dynamics = dyn, advection = ADVECTION)
end

function build_substepper(Ns)
    return function(grid)
        constants = ThermodynamicConstants(eltype(grid))
        td = SplitExplicitTimeDiscretization(substeps = Ns,
                                             forward_weight = 0.8,
                                             damping = PressureProjectionDamping(coefficient = 0.5))
        dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
        return AtmosphereModel(grid; dynamics = dyn, advection = ADVECTION,
                               thermodynamic_constants = constants,
                               timestepper = :AcousticRungeKutta3)
    end
end

function run_case(label, Δt, build_model)
    grid  = build_grid()
    model = build_model(grid)
    θᵢ    = θᵢ_builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ)
    else
        set!(model; θ = θᵢ, ρ = model.dynamics.reference_state.density)
    end
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    add_callback!(sim, s -> @info(@sprintf("[%s] iter=%4d t=%.0fs max|w|=%.3g",
        label, iteration(s), s.model.clock.time,
        Float64(maximum(abs, interior(s.model.velocities.w))))), IterationInterval(20))
    sim.output_writers[:jld2] = JLD2Writer(model, (; w = model.velocities.w);
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(2.0),
                                           overwrite_existing = true)
    t0 = time()
    try
        run!(sim)
    catch e
        @warn "$label CRASH: $(typeof(e))"
    end
    return (; label, iter = iteration(sim), t = sim.model.clock.time,
              wmax = Float64(maximum(abs, interior(sim.model.velocities.w))),
              elapsed = time() - t0)
end

results = NamedTuple[]
push!(results, run_case("anelastic", 1.0, build_anelastic))
for Ns in (12, 24, 48, 96)
    push!(results, run_case("Ns$Ns", 1.0, build_substepper(Ns)))
end

@info "=== SUMMARY (Centered(order=2)) ==="
for r in results
    @info @sprintf("  %-9s wmax=%6.3f  elapsed=%4.1fs", r.label, r.wmax, r.elapsed)
end

#####
##### Animation: 5 panels (anelastic + 4 substepper Ns values).
#####

@info "=== Rendering 5-panel animation ==="
labels = ("anelastic", "Ns12", "Ns24", "Ns48", "Ns96")
titles = ("Anelastic\nΔt=1.0s",
          "Ns=12\nΔt=1.0s, ω=0.8, β=0.5",
          "Ns=24\nΔt=1.0s",
          "Ns=48\nΔt=1.0s",
          "Ns=96\nΔt=1.0s")

ws = [FieldTimeSeries(joinpath(OUTDIR, "$l.jld2"), "w") for l in labels]
Nt = minimum(length.(getfield.(ws, :times)))
@info "Animating $Nt frames"
grid = ws[1].grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3
vmax = maximum(maximum(abs, interior(ws[1][i])) for i in 1:Nt)  # anelastic baseline

n = Observable(1)
slices = [@lift Array(interior(ws[k][$n]))[:, 1, :] for k in 1:5]

fig = Figure(size = (2400, 600), fontsize = 14)
title_node = @lift @sprintf("Centered(2) Ns sweep — t = %.1f s", ws[1].times[$n])
fig[0, 1:6] = Label(fig, title_node, fontsize = 20, tellwidth = false)

local hm
for (k, (t, s)) in enumerate(zip(titles, slices))
    ax = Axis(fig[1, k]; title = t, xlabel = "x (km)",
              ylabel = k == 1 ? "z (km)" : "", aspect = DataAspect())
    hm = heatmap!(ax, x_km, z_km, s; colormap = :balance, colorrange = (-vmax, vmax))
    k == 5 && Colorbar(fig[1, 6], hm; label = "w (m/s)")
end

mp4 = joinpath(OUTDIR, "centered_ns_sweep.mp4")
CairoMakie.record(fig, mp4, 1:Nt; framerate = 10) do i
    n[] = i
end
@info "wrote $mp4"

n[] = Nt
save(joinpath(OUTDIR, "centered_ns_sweep_final.png"), fig)
@info "wrote centered_ns_sweep_final.png"
