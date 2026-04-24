#####
##### validation/substepping/24_centered_advection_test.jl
#####
##### Tests the user's hypothesis: the bug is a stencil mismatch between the
##### slow tendency's WENO-9 and the substepper's arithmetic-mean θ-flux.
##### Switch advection to Centered(order=2) so both the slow tendency and
##### the substepper use the SAME 2nd-order arithmetic mean. If the
##### Ns-scaling instability disappears, hypothesis confirmed.
#####
##### Single-job design: runs all 4 cases, then renders the animation in the
##### same Julia process to avoid the ~80 s re-precompilation overhead.
#####
#####   1. Anelastic            Δt = 1.0 s, Centered(2)
#####   2. Substepper Ns=12     Δt = 1.0 s, Centered(2), ω=0.8, β=0.5
#####   3. Substepper Ns=48     Δt = 1.0 s, Centered(2)
#####   4. Compressible explicit Δt = 0.1 s, Centered(2)
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

const CASE = "centered_advection_test"
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

function run_case(; label, Δt, build_model)
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
    status = :ok
    try
        run!(sim)
    catch e
        status = :crashed
        @warn "$label crashed: $(typeof(e))"
    end
    elapsed = time() - t0
    return (; label, iter = iteration(sim), t = sim.model.clock.time,
              wmax = Float64(maximum(abs, interior(sim.model.velocities.w))),
              elapsed, status)
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
        td  = SplitExplicitTimeDiscretization(substeps = Ns,
                                              forward_weight = 0.8,
                                              damping = PressureProjectionDamping(coefficient = 0.5))
        dyn = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
        return AtmosphereModel(grid; dynamics = dyn, advection = ADVECTION,
                               thermodynamic_constants = constants,
                               timestepper = :AcousticRungeKutta3)
    end
end

function build_explicit(grid)
    constants = ThermodynamicConstants(eltype(grid))
    dyn = CompressibleDynamics(ExplicitTimeStepping();
                               reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics = dyn, advection = ADVECTION,
                           thermodynamic_constants = constants)
end

@info "=== Running 4 cases with Centered(order=2) advection ==="

results = NamedTuple[]
push!(results, run_case(label = "anelastic", Δt = 1.0, build_model = build_anelastic))
push!(results, run_case(label = "Ns12",      Δt = 1.0, build_model = build_substepper(12)))
push!(results, run_case(label = "Ns48",      Δt = 1.0, build_model = build_substepper(48)))
push!(results, run_case(label = "explicit",  Δt = 0.1, build_model = build_explicit))

@info "=== SUMMARY ==="
for r in results
    @info @sprintf("  %-9s iter=%5d t=%5.1f wmax=%6.2f  %4.1fs  %s",
                   r.label, r.iter, r.t, r.wmax, r.elapsed, r.status)
end

#####
##### Animation in the same process — avoids re-precompile overhead.
#####

@info "=== Rendering 4-way animation ==="
labels  = ("anelastic", "Ns12", "Ns48", "explicit")
titles  = ("Anelastic\nΔt=1.0s, Centered(2)",
           "Substepper Ns=12\nΔt=1.0s, Centered(2)",
           "Substepper Ns=48\nΔt=1.0s, Centered(2)",
           "Compressible explicit\nΔt=0.1s, Centered(2)")
ws = [FieldTimeSeries(joinpath(OUTDIR, "$l.jld2"), "w") for l in labels]

Nt = minimum(length.(getfield.(ws, :times)))
@info "Animating $Nt frames"
grid = ws[1].grid
x_km = collect(xnodes(grid, Center())) ./ 1e3
z_km = collect(znodes(grid, Face()))   ./ 1e3

# Color range from the anelastic baseline.
vmax = maximum(maximum(abs, interior(ws[1][i])) for i in 1:Nt)
vmax = vmax > 0 ? vmax : 1.0
@info "color range ±$vmax m/s"

n = Observable(1)
slices = [@lift Array(interior(ws[k][$n]))[:, 1, :] for k in 1:4]

fig = Figure(size = (2000, 600), fontsize = 14)
title_node = @lift @sprintf("Centered(2) — t = %.1f s", ws[1].times[$n])
fig[0, 1:5] = Label(fig, title_node, fontsize = 20, tellwidth = false)

local hm
for (k, (t, s)) in enumerate(zip(titles, slices))
    ax = Axis(fig[1, k]; title = t,
              xlabel = "x (km)", ylabel = k == 1 ? "z (km)" : "",
              aspect = DataAspect())
    hm = heatmap!(ax, x_km, z_km, s; colormap = :balance, colorrange = (-vmax, vmax))
    k == 4 && Colorbar(fig[1, 5], hm; label = "w (m/s)")
end

mp4 = joinpath(OUTDIR, "bubble_centered2.mp4")
CairoMakie.record(fig, mp4, 1:Nt; framerate = 10) do i
    n[] = i
end
@info "wrote $mp4"

n[] = Nt
save(joinpath(OUTDIR, "bubble_centered2_final.png"), fig)
@info "wrote bubble_centered2_final.png"
