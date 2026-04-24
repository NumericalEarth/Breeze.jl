#####
##### validation/substepping/13_bubble_substeps_sweep.jl
#####
##### Sweep the number of acoustic substeps per outer step on the dry
##### thermal bubble, holding the outer Δt fixed. Tests whether the
##### bubble-top 2Δ checkerboard observed in 07 responds to acoustic-CFL
##### reduction (i.e., smaller Δτᵋ = Δt/N).
#####
##### If the artefact decays with N → it's a Δτᵋ-dependent instability
##### (not enough resolution of the acoustic mode in the column solve).
##### If the artefact is insensitive to N → it's a coefficient-level bug
##### in the tridiagonal assembly that more substeps cannot fix.
#####
##### Fixed outer Δt = 1.0 s. Wizard is disabled so substep count is the
##### only variable. ProportionalSubsteps distribution. PressureProjection-
##### Damping(0.5) and forward_weight = 0.8 held fixed at the 07 values.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using Printf
using JLD2

const CASE = "dry_thermal_bubble_substeps_sweep"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const STOP_T = 7minutes
const FIXED_DT = 1.0            # outer Δt, held fixed across all runs
const SUBSTEP_COUNTS = [6, 12, 24, 48]

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

function build_compressible_model(grid, Ns)
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(substeps = Ns,
                                         forward_weight = 0.8,
                                         damping = PressureProjectionDamping(coefficient = 0.5))
    dynamics = CompressibleDynamics(td; reference_potential_temperature = θᵇᵍ)
    return AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function run_case(label, Ns)
    grid = build_grid()
    model = build_compressible_model(grid, Ns)
    θᵢ = θᵢ_builder(grid)
    ref = model.dynamics.reference_state
    set!(model; θ = θᵢ, ρ = ref.density)

    sim = Simulation(model; Δt = FIXED_DT, stop_time = STOP_T, verbose = false)
    # NOTE: No wizard — fixed outer Δt across all variants.

    progress_counter = Ref(0)
    function _progress(sim)
        progress_counter[] += 1
        if progress_counter[] % 20 == 0
            @info @sprintf("[%s/Ns=%d] iter=%5d t=%7.1fs Δt=%.3fs max|w|=%.3f",
                           label, Ns, iteration(sim), sim.model.clock.time, sim.Δt,
                           maximum(abs, interior(sim.model.velocities.w)))
        end
    end
    add_callback!(sim, _progress, IterationInterval(20))

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

statuses = Dict{String, Symbol}()
for Ns in SUBSTEP_COUNTS
    label = "Ns_$(Ns)"
    @info "[$CASE] Running '$label' (substeps = $Ns, outer Δt = $FIXED_DT s)"
    statuses[label] = run_case(label, Ns)
end

@info "[$CASE] SUBSTEPS-SWEEP SUMMARY:"
for Ns in SUBSTEP_COUNTS
    label = "Ns_$(Ns)"
    @info "  Ns=$(Ns) → $(statuses[label])"
end
@info "[$CASE] done. jld2 outputs in $OUTDIR"
