#####
##### validation/substepping/03_prescribed_sst.jl
#####
##### 2D moist convection driven by a bulk-flux prescribed SST pattern.
##### Starts from θ = θ₀ uniform + small u = 1 m/s — no initial bubble shock.
#####

include("common.jl")

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Oceananigans
using Oceananigans.Units
using Statistics
using JLD2

const CASE = "prescribed_sst"
const OUTDIR = joinpath(@__DIR__, "out", CASE)
isdir(OUTDIR) || mkpath(OUTDIR)

# Original Δt=10, stop=4hours. We cap to 30 min to keep timing tractable.
const Δt     = 10.0
const STOP_T = 30minutes
const p₀     = 101325.0
const θ₀     = 285.0
const ΔT     = 4.0
const Uᵍ     = 1e-2

function build_grid()
    RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                    x = (-10kilometers, 10kilometers),
                    z = (0, 10kilometers),
                    topology = (Periodic, Flat, Bounded))
end

T₀(x) = θ₀ + ΔT / 2 * sign(cos(2π * x / 20e3))

function build_bcs(grid)
    filtered_velocities = FilteredSurfaceVelocities(grid; filter_timescale = 1hour)
    coef = PolynomialCoefficient(roughness_length = 1.5e-4)
    ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = coef; gustiness = Uᵍ, surface_temperature = T₀, filtered_velocities))
    ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = coef; gustiness = Uᵍ, surface_temperature = T₀, filtered_velocities))
    ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = coef; gustiness = Uᵍ, surface_temperature = T₀, filtered_velocities))
    ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = coef; gustiness = Uᵍ, surface_temperature = T₀, filtered_velocities))
    return (; ρu = ρu_bcs, ρv = ρv_bcs, ρe = ρe_bcs, ρqᵉ = ρqᵉ_bcs)
end

function build_anelastic(grid)
    constants = ThermodynamicConstants(eltype(grid))
    reference_state = ReferenceState(grid, constants; surface_pressure = p₀, potential_temperature = θ₀)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                    momentum_advection = WENO(order=9), scalar_advection = WENO(order=5),
                    microphysics, boundary_conditions = build_bcs(grid))
end

function build_compressible(grid; damping = PressureProjectionDamping(coefficient = 0.1))
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; damping)
    dynamics = CompressibleDynamics(td;
                                    surface_pressure = p₀,
                                    reference_potential_temperature = θ₀)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    AtmosphereModel(grid; dynamics, thermodynamic_constants = constants,
                    momentum_advection = WENO(order=9), scalar_advection = WENO(order=5),
                    microphysics, boundary_conditions = build_bcs(grid),
                    timestepper = :AcousticRungeKutta3)
end

function run_case(label, builder)
    grid = build_grid()
    model = builder(grid)
    if label == "anelastic"
        set!(model; θ = θ₀, u = 1.0)
    else
        ref = model.dynamics.reference_state
        set!(model; θ = θ₀, u = 1.0, ρ = ref.density)
    end
    sim = Simulation(model; Δt, stop_time = STOP_T, verbose = false)
    outputs = (; w = model.velocities.w, T = model.temperature)
    sim.output_writers[:jld2] = JLD2Writer(model, outputs;
                                           filename = joinpath(OUTDIR, "$(label).jld2"),
                                           schedule = TimeInterval(2minutes),
                                           overwrite_existing = true)
    res = timed_run!(sim; label)
    return summarize_result(label, res, model)
end

@info "[$CASE] Anelastic run…"
a = run_case("anelastic", build_anelastic)
@info "[$CASE] Compressible run…"
c = run_case("compressible", build_compressible)

wa = try; FieldTimeSeries(joinpath(OUTDIR, "anelastic.jld2"), "w")[end]; catch; nothing; end
wc = try; FieldTimeSeries(joinpath(OUTDIR, "compressible.jld2"), "w")[end]; catch; nothing; end
wa !== nothing && wc !== nothing && two_column_figure(joinpath(OUTDIR, "summary.png"), wa, wc;
    title_a = "anelastic w (m/s)", title_b = "compressible w (m/s)", label = "w (m/s)")

jldsave(joinpath(OUTDIR, "result.jld2"); anelastic = a, compressible = c, case = CASE, Δt, stop_time = STOP_T)
io = IOBuffer()
report_case(io, CASE,
            "2D moist convection over SST front, 128×128, Δt=$(Δt)s, stop=$(STOP_T)s, CPU, WENO(9)/WENO(5), BulkDrag/Sensible/Vapor, SatAdj.",
            a, c)
write(joinpath(OUTDIR, "report.md"), take!(io))
@info "[$CASE] done"
