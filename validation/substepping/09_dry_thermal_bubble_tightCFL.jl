#####
##### validation/substepping/09_dry_thermal_bubble_tightCFL.jl
#####
##### Substepper at *tight* CFL to isolate whether the ring artifact that
##### shows up around t = 400 s is truncation (vanishes as Δt → 0) or a
##### formulation bug (persists).
#####
##### Two runs, same IC, stopped at t = 500 s so we cover the first ring event:
#####   tight_dt01:  Δt = 0.1 s fixed (== explicit Δt!). N = 6 adaptive.
#####   tight_dt025: Δt = 0.25 s fixed. N = 12 forced (Δτ ≈ 0.02 s, very safe).
#####
#####
##### Both outputs land next to anelastic.jld2 / explicit.jld2 / compressible.jld2.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using JLD2

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard")
isdir(OUTDIR) || mkpath(OUTDIR)

const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

build_grid() = RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                               x = (-10e3, 10e3), z = (0, 10e3),
                               topology = (Periodic, Flat, Bounded))

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    (x, z) -> θᵇᵍ(z) + Δθ * max(0, 1 - sqrt((x - x₀)^2 + (z - z₀)^2) / r₀)
end

function build_compressible_model(grid; substeps = nothing,
                                  damping = PressureProjectionDamping(coefficient = 0.5),
                                  forward_weight = 0.8)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; substeps, damping, forward_weight)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

const STOP_T = 500.0  # enough to cover the t = 400 s ring event

function run_case(label, builder; Δt)
    grid = build_grid()
    model = builder(grid)
    set!(model; θ = θᵢ_builder(grid), ρ = model.dynamics.reference_state.density)
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    outputs = (; w = model.velocities.w, T = model.temperature)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(10seconds),
                                           overwrite_existing = true)
    t0 = time()
    ok = true; err=""
    try
        run!(sim)
    catch e
        ok = false; err = sprint(showerror, e)
    end
    wall = time() - t0
    w = model.velocities.w
    @info @sprintf("[%s] Δt=%.3f ok=%s iter=%d t=%.1fs wall=%.1fs max|w|=%.2f",
                   label, Δt, ok, iteration(sim), sim.model.clock.time, wall,
                   maximum(abs, interior(w)))
    return nothing
end

@info "[tight_dt01] Δt=0.1 adaptive N (≈N=6)"
run_case("tight_dt01",  grid -> build_compressible_model(grid; substeps = nothing); Δt = 0.1)

@info "[tight_dt025] Δt=0.25 N=12"
run_case("tight_dt025", grid -> build_compressible_model(grid; substeps = 12);      Δt = 0.25)
