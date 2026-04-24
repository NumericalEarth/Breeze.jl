#####
##### validation/substepping/02_cloudy_thermal_bubble.jl
#####
##### Only the "dry" segment of examples/cloudy_thermal_bubble.jl is rewritten —
##### the moist/precip variants reuse the same setup so if the dry one works,
##### the rest is a separate microphysics exercise.
#####

include("common.jl")

using Breeze
using Oceananigans
using Oceananigans.Units
using Statistics
using JLD2

const CASE = "cloudy_thermal_bubble"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

const Δt     = 2.0
const STOP_T = 1000.0  # seconds (original stop_time)

const θ₀  = 300.0
const r₀  = 2e3
const z₀  = 2e3
const Δθ  = 2.0
const p₀  = 1e5

function build_grid()
    RectilinearGrid(CPU();
                    size = (128, 128), halo = (5, 5),
                    x = (-10e3, 10e3),
                    z = (0, 10e3),
                    topology = (Bounded, Flat, Bounded))
end

function θᵢ(x, z)
    r = sqrt((x / r₀)^2 + ((z - z₀) / r₀)^2)
    return θ₀ + Δθ * cos(π * min(1, r) / 2)^2
end

function build_anelastic(grid)
    constants = ThermodynamicConstants(eltype(grid))
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = p₀, potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)
    AtmosphereModel(grid; dynamics, advection = WENO(order=9), thermodynamic_constants = constants)
end

function build_compressible(grid; damping = PressureProjectionDamping(coefficient = 0.1))
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; damping)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = p₀,
                                    reference_potential_temperature = θ₀)
    AtmosphereModel(grid; dynamics, advection = WENO(order=9),
                    thermodynamic_constants = constants,
                    timestepper = :AcousticRungeKutta3)
end

function run_case(label, builder; Δt_local = Δt, stop_time = STOP_T)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θᵢ)
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θᵢ, ρ = ref.density)
    end
    sim = Simulation(model; Δt = Δt_local, stop_time, verbose = false)
    outputs = (; θ = Breeze.PotentialTemperature(model), w = model.velocities.w)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(50seconds),
                                           overwrite_existing = true)
    res = timed_run!(sim; label)
    s = summarize_result(label, res, model)
    return s
end

@info "[$CASE] Anelastic run…"
a = run_case("anelastic", build_anelastic)
@info "[$CASE] Compressible run…"
c = run_case("compressible", build_compressible)

function read_last(path, name)
    isfile(path) || return nothing
    try
        fts = FieldTimeSeries(path, name)
        return fts[end]
    catch e
        @warn "Could not read $(name) from $path" exception=e
        return nothing
    end
end

wa = read_last(joinpath(OUTDIR, "anelastic.jld2"), "w")
wc = read_last(joinpath(OUTDIR, "compressible.jld2"), "w")
if wa !== nothing && wc !== nothing
    two_column_figure(joinpath(OUTDIR, "summary.png"), wa, wc;
                      title_a = "anelastic w (m/s)", title_b = "compressible w (m/s)",
                      label = "w (m/s)")
end

jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a, compressible = c,
        case = CASE, Δt = Δt, stop_time = STOP_T)

io = IOBuffer()
report_case(io, CASE,
            "2D bubble, 128×128 Bounded/Flat/Bounded, Δt = $(Δt)s, stop = $(STOP_T)s, CPU, WENO(9), Δθ = $(Δθ)K.",
            a, c)
write(joinpath(OUTDIR, "report.md"), take!(io))
@info "[$CASE] done"
