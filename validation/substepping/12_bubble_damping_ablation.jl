#####
##### validation/substepping/12_bubble_damping_ablation.jl
#####
##### A/B the divergence-damping strategy on the dry thermal bubble stress
##### case to see whether the bubble-top checkerboard observed in the 3-way
##### movie is CAUSED by the damping kernel (i.e. damping is mis-signed or
##### over-amplified and exciting a 2Δ mode) or is a vertical-column-solve
##### residual that the damping fails to fully suppress.
#####
##### Same grid, same IC, same forward_weight = 0.8, same cfl = 0.3 wizard.
##### The only thing that varies is the `damping` strategy:
#####
#####   baseline  = PressureProjectionDamping(coefficient = 0.5)   # the 07 choice
#####   no_damp   = NoDivergenceDamping()
#####   strong    = PressureProjectionDamping(coefficient = 0.1)   # mild projection
#####   thermo    = ThermodynamicDivergenceDamping(coefficient = 0.1)
#####
##### Saves `w` at t = 400s for all four runs so we can see, side-by-side,
##### whether the ringing at the bubble top is present in every variant
##### (→ tridiagonal), absent without damping (→ damping kernel), or changes
##### with the damping form (→ projection-specific artefact).
#####
##### Note: without damping the run may blow up — if so, the instability mode
##### at blowup is itself informative (grid-scale → tridiagonal, gravity-wave
##### → damping was suppressing something physical).
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using JLD2

const CASE = "dry_thermal_bubble_damping_ablation"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 7minutes  # Enough to see bubble-top ringing develop (artefact visible by t=400s in 07).
const CFL    = 0.3

const θ₀_ref = 300.0
const N²     = 1e-6
const r₀     = 2e3
const Δθ     = 10.0
const g      = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function build_grid()
    RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
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

function build_compressible_model(grid, damping)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(forward_weight = 0.8, damping = damping)
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_case(label, damping)
    grid = build_grid()
    model = build_compressible_model(grid, damping)
    θᵢ = θᵢ_builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt = 0.5, stop_time = STOP_T, verbose = false)
    conjure_time_step_wizard!(sim; cfl = CFL)

    progress_counter = Ref(0)
    function _progress(sim)
        progress_counter[] += 1
        if progress_counter[] % 5 == 0
            @info @sprintf("[%s] iter=%5d t=%7.1fs Δt=%.3fs max|w|=%.3f",
                           label, iteration(sim), sim.model.clock.time, sim.Δt,
                           maximum(abs, interior(sim.model.velocities.w)))
        end
    end
    add_callback!(sim, _progress, IterationInterval(50))

    outputs = (; w = model.velocities.w)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(10seconds),
                                           overwrite_existing = true)

    try
        res = timed_run!(sim; label)
        @info "[$label] DONE: $(summarize_result(label, res, model))"
        return :ok
    catch e
        @warn "[$label] CRASHED" exception = e
        return :crashed
    end
end

cases = [
    ("baseline_pressproj_0.5", PressureProjectionDamping(coefficient = 0.5)),
    ("mild_pressproj_0.1",     PressureProjectionDamping(coefficient = 0.1)),
    ("thermo_0.1",             ThermodynamicDivergenceDamping(coefficient = 0.1)),
    ("no_damping",             NoDivergenceDamping()),
]

statuses = Dict{String, Symbol}()
for (label, damping) in cases
    @info "[$CASE] Running '$label' with damping = $damping"
    statuses[label] = run_case(label, damping)
end

@info "[$CASE] ABLATION SUMMARY:"
for (label, _) in cases
    @info "  $label → $(statuses[label])"
end
@info "[$CASE] done. jld2 outputs in $OUTDIR"
