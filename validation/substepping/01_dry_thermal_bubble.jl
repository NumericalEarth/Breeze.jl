#####
##### validation/substepping/01_dry_thermal_bubble.jl
#####
##### Mirror of examples/dry_thermal_bubble.jl comparing AnelasticDynamics with
##### CompressibleDynamics(SplitExplicitTimeDiscretization()).
#####
##### Departure from the original example: the original uses formulation=:StaticEnergy,
##### but the MPAS acoustic substepper is presently wired for ρθ recovery. We run both
##### anelastic and compressible with :LiquidIcePotentialTemperature (the default) so
##### the comparison is apples-to-apples at the formulation level.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using JLD2

const CASE = "dry_thermal_bubble"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Δt     = 2.0
const STOP_T = 25minutes

function build_grid()
    RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                    x = (-10e3, 10e3), z = (0, 10e3),
                    topology = (Periodic, Flat, Bounded))
end

θ₀_ref = 300.0
N²     = 1e-6
r₀     = 2e3
Δθ     = 10.0
const g = 9.80665
θᵇᵍ(z) = θ₀_ref * exp(N² * z / g)

function θᵢ_builder(grid)
    x₀ = mean(xnodes(grid, Center()))
    z₀ = 0.3 * grid.Lz
    function θᵢ(x, z)
        r = sqrt((x - x₀)^2 + (z - z₀)^2)
        θ′ = Δθ * max(0, 1 - r / r₀)
        return θᵇᵍ(z) + θ′
    end
end

function build_anelastic_model(grid)
    constants = ThermodynamicConstants(eltype(grid))
    reference_state = ReferenceState(grid, constants; potential_temperature = θᵇᵍ)
    dynamics = AnelasticDynamics(reference_state)
    advection = WENO(order=9)
    model = AtmosphereModel(grid; dynamics, advection)
    return model
end

function build_compressible_model(grid; damping = ThermodynamicDivergenceDamping(coefficient = 0.1),
                                  advection = WENO(order=9))
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; damping)
    dynamics = CompressibleDynamics(td;
                                    reference_potential_temperature = θᵇᵍ)
    model = AtmosphereModel(grid; dynamics, advection,
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)
    return model
end

function run_case(label, model_builder; Δt_local = Δt, stop_time = STOP_T)
    grid = build_grid()
    model = model_builder(grid)

    θᵢ = θᵢ_builder(grid)

    if label == "anelastic"
        set!(model; θ = θᵢ)
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, ρ = ref.density)
    end

    simulation = Simulation(model; Δt = Δt_local, stop_time, verbose = false)

    outputs = merge(model.velocities, (; θ = Breeze.PotentialTemperature(model)))
    simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                                  filename = joinpath(OUTDIR, "$(label).jld2"),
                                                  schedule = TimeInterval(60seconds),
                                                  overwrite_existing = true)

    res = timed_run!(simulation; label)
    s = summarize_result(label, res, model)
    return s, model, grid
end

@info "[$CASE] Anelastic run…"
a_sum, a_model, grid = run_case("anelastic", build_anelastic_model)
@info "[$CASE] Compressible run…"
c_sum, c_model, _    = run_case("compressible", build_compressible_model)

# Build a side-by-side final-state figure from the saved outputs. If compressible
# crashed mid-run we still try to plot whatever was written.
function maybe_heatmap(label)
    path = joinpath(OUTDIR, "$(label).jld2")
    isfile(path) || return nothing
    try
        fts = FieldTimeSeries(path, "w")
        return fts[end]
    catch e
        @warn "Could not open $path: $(sprint(showerror, e))"
        return nothing
    end
end

fig_path = joinpath(OUTDIR, "summary.png")
let wa = maybe_heatmap("anelastic"), wc = maybe_heatmap("compressible")
    if wa !== nothing && wc !== nothing
        two_column_figure(fig_path, wa, wc;
                          title_a = "anelastic w (m/s)",
                          title_b = "compressible w (m/s)",
                          label = "w (m/s)")
        @info "wrote $fig_path"
    else
        @warn "skipping figure for $CASE (missing output)"
    end
end

open(joinpath(OUTDIR, "result.jld2"), "w") do _
end
jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a_sum, compressible = c_sum,
        case = CASE, setup = (; Δt, stop_time = STOP_T))

io = IOBuffer()
report_case(io, CASE,
            "2D bubble, 128×128, Δt = $(Δt)s, stop = $(STOP_T)s, CPU, WENO(9)",
            a_sum, c_sum)
report_path = joinpath(OUTDIR, "report.md")
write(report_path, take!(io))
@info "[$CASE] done" report_path
