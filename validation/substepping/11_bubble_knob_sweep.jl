#####
##### validation/substepping/11_bubble_knob_sweep.jl
#####
##### Runtime-knob sweep to isolate what causes the ring in the substepper.
##### Each run reaches t = 420 s (just past the ring event) or crashes. All use
##### wizard cfl = 0.3 (reproducing the wizard run from 07).
#####
##### Variables:
#####   - damping coefficient β_d ∈ {0.1, 0.25, 0.5}
#####   - forward_weight ω ∈ {0.6, 0.7, 0.8, 0.9}
#####   - damping strategy {Pressure, Thermo}
#####
##### Metric: max|w| at t=400s compared to anelastic/explicit ≈ 29.5. Any run
##### that gives max|w| within 5% of reference *without* the ring visible in
##### the field plot is a "no ring" winner.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using JLD2

const OUTDIR = joinpath(@__DIR__, "out", "dry_thermal_bubble_wizard", "knob_sweep")
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

function run_case(label; damping, forward_weight, cfl = 0.3, stop_time = 420.0)
    grid = build_grid()
    td = SplitExplicitTimeDiscretization(; damping, forward_weight)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                            timestepper = :AcousticRungeKutta3)
    set!(model; θ = θᵢ_builder(grid), ρ = model.dynamics.reference_state.density)

    sim = Simulation(model; Δt = 0.5, stop_time, verbose = false)
    conjure_time_step_wizard!(sim; cfl)
    outputs = (; w = model.velocities.w)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(10seconds),
                                           overwrite_existing = true)
    t0 = time()
    ok = true; err = ""
    try
        run!(sim)
    catch e
        ok = false
        err = first(sprint(showerror, e), 160)
    end
    wall = time() - t0
    wmax = maximum(abs, interior(model.velocities.w))
    @info @sprintf("[%-24s] ok=%s wall=%.1fs iter=%d t=%.1f max|w|=%.2f %s",
                   label, ok, wall, iteration(sim), sim.model.clock.time, wmax,
                   ok ? "" : err)
    return (; ok, wall, iter = iteration(sim), t = sim.model.clock.time, wmax, err)
end

# Sweep
results = Dict{String,Any}()
results["Press0.5_fw0.8"] = run_case("Press0.5_fw0.8"; damping = PressureProjectionDamping(coefficient = 0.5), forward_weight = 0.8)
results["Press0.5_fw0.6"] = run_case("Press0.5_fw0.6"; damping = PressureProjectionDamping(coefficient = 0.5), forward_weight = 0.6)
results["Press0.5_fw0.7"] = run_case("Press0.5_fw0.7"; damping = PressureProjectionDamping(coefficient = 0.5), forward_weight = 0.7)
results["Press0.5_fw0.9"] = run_case("Press0.5_fw0.9"; damping = PressureProjectionDamping(coefficient = 0.5), forward_weight = 0.9)
results["Press0.1_fw0.8"] = run_case("Press0.1_fw0.8"; damping = PressureProjectionDamping(coefficient = 0.1), forward_weight = 0.8)
results["Press0.1_fw0.6"] = run_case("Press0.1_fw0.6"; damping = PressureProjectionDamping(coefficient = 0.1), forward_weight = 0.6)
results["Press0.25_fw0.8"] = run_case("Press0.25_fw0.8"; damping = PressureProjectionDamping(coefficient = 0.25), forward_weight = 0.8)
results["Thermo0.5_fw0.8"] = run_case("Thermo0.5_fw0.8"; damping = ThermodynamicDivergenceDamping(coefficient = 0.5), forward_weight = 0.8)
results["Thermo0.1_fw0.8"] = run_case("Thermo0.1_fw0.8"; damping = ThermodynamicDivergenceDamping(coefficient = 0.1), forward_weight = 0.8)
results["NoDamp_fw0.8"]   = run_case("NoDamp_fw0.8";   damping = NoDivergenceDamping(),                            forward_weight = 0.8)

jldsave(joinpath(OUTDIR, "sweep_results.jld2"); results)
@info "sweep done"
