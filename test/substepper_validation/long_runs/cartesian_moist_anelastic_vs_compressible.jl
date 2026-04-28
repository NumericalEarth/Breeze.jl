#####
##### Cartesian moist convection — anelastic vs compressible-substepper.
#####
##### A 2-D Cartesian (Periodic, Flat, Bounded) box with the SAME moist
##### physics as the moist baroclinic wave: NonEquilibriumCloudFormation
##### with τ_relax=200s, OneMomentCloudMicrophysics, surface fluxes over
##### a warm SST that drives boundary-layer convection.
#####
##### Both anelastic and compressible-substepper are run on identical
##### grids and physics. If anelastic is stable but compressible-
##### substepper NaNs at any reasonable Δt, the substepper is the
##### binding factor. If both fail at the same Δt, the moist physics
##### implementation is the issue.
#####

using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2
using CloudMicrophysics
using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "cartesian_moist_anelastic_vs_compressible"

Oceananigans.defaults.FloatType = Float32

const constants = ThermodynamicConstants(;
    gravitational_acceleration = 9.80665,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

const Rᵈ  = Breeze.dry_air_gas_constant(constants)
const cᵖᵈ = constants.dry_air.heat_capacity
const g   = constants.gravitational_acceleration
const p₀  = 101325.0   # Pa surface pressure
const θ₀  = 300.0      # K background θ (≈300K isothermal-ish)

# SST: warm uniform surface to drive convection (no horizontal gradient
# so the test isolates moist convective overturning, not advection).
const T_surface = 305.0
# 2-D Flat-y grid: surface BC takes a single horizontal coordinate (x).
T_surface_field(x) = T_surface

const Cᴰ = 1e-3
const Uᵍ = 1e-2

# Grid: 2-D Cartesian box, 100km × 30km, 64×128 cells (Δx=1.6km, Δz=234m)
function build_grid()
    RectilinearGrid(arch; size = (64, 128), halo = (5, 5),
                    x = (-50kilometers, 50kilometers),
                    z = (0, 30kilometers),
                    topology = (Periodic, Flat, Bounded))
end

function make_bcs()
    return (
        ρu  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_field)),
        ρθ  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_field)),
        ρqᵛ = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_field)),
    )
end

# Initial moisture profile — exponentially decaying with z (similar to
# moist BCI tropical column).
function specific_humidity(x, z)
    q₀_surf = 0.012   # kg/kg surface
    qₜ      = 1e-12
    Hq      = 3000.0   # m e-folding scale
    return q₀_surf * exp(-z/Hq) + qₜ
end

# ============================================================================
# Compressible-substepper builder
# ============================================================================

function build_compressible(; substepper_kw = NamedTuple())
    grid = build_grid()
    θᵇᵍ(z) = θ₀ * exp(g * z / (cᵖᵈ * θ₀))   # isothermal-T₀=300K stratification

    td  = SplitExplicitTimeDiscretization(; substepper_kw...)
    dyn = CompressibleDynamics(td;
                               surface_pressure = p₀,
                               reference_potential_temperature = θᵇᵍ)

    BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    τ_relax = 200.0
    relaxation = ConstantRateCondensateFormation(1/τ_relax)
    cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
    microphysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics(; cloud_formation)

    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))

    bcs = make_bcs()
    model = AtmosphereModel(grid;
                            dynamics = dyn,
                            momentum_advection = weno,
                            scalar_advection = (ρθ = weno, ρqᵛ = bp_weno,
                                                ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                                                ρqʳ = bp_weno, ρqˢ = bp_weno),
                            microphysics = microphysics,
                            thermodynamic_constants = constants,
                            boundary_conditions = bcs,
                            timestepper = :AcousticRungeKutta3)

    ref = model.dynamics.reference_state
    set!(model; θ = (x, z) -> θᵇᵍ(z), ρ = ref.density,
                qᵛ = (x, z) -> specific_humidity(x, z))
    return model
end

# ============================================================================
# Anelastic builder
# ============================================================================

function build_anelastic()
    grid = build_grid()

    BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    τ_relax = 200.0
    relaxation = ConstantRateCondensateFormation(1/τ_relax)
    cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
    microphysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics(; cloud_formation)

    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = p₀,
                                     potential_temperature = θ₀)
    dyn = AnelasticDynamics(reference_state)

    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))

    bcs = make_bcs()
    model = AtmosphereModel(grid;
                            dynamics = dyn,
                            momentum_advection = weno,
                            scalar_advection = (ρθ = weno, ρqᵛ = bp_weno,
                                                ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                                                ρqʳ = bp_weno, ρqˢ = bp_weno),
                            microphysics = microphysics,
                            thermodynamic_constants = constants,
                            boundary_conditions = bcs)

    # Stratified initial θ: SAME profile the compressible run uses, so the
    # comparison is on identical IC modulo dynamics.
    θᵇᵍ(z) = θ₀ * exp(g * z / (cᵖᵈ * θ₀))
    set!(model; θ = (x, z) -> θᵇᵍ(z), qᵛ = (x, z) -> specific_humidity(x, z))
    return model
end

# ============================================================================
# Run with adaptive Δt + diagnostics + JLD2 snapshot
# ============================================================================

function run_with_diagnostics(label, model;
                               Δt_init, max_Δt, cfl, stop_time, sample_every = 200)
    sim = Simulation(model; Δt = Δt_init, stop_time)
    conjure_time_step_wizard!(sim; cfl, max_Δt, max_change = 1.05)

    # Diagnostics
    iters = Int[]; ts = Float64[]; dts = Float64[]
    wmax = Float64[]; umax = Float64[]
    qcl_max = Float64[]; qv_max = Float64[]
    walls = Float64[]
    wall0 = Ref(time_ns())

    function diag_cb(s)
        m = s.model
        u, w = m.velocities.u, m.velocities.w
        wm = Float64(maximum(abs, interior(w)))
        um = Float64(maximum(abs, interior(u)))
        ρqᶜˡ_max = if hasproperty(m, :microphysical_fields)
            Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
        else
            0.0
        end
        ρqᵛ_max = if hasproperty(m, :moisture_density)
            Float64(maximum(interior(m.moisture_density)))
        else
            0.0
        end
        push!(iters, iteration(s)); push!(ts, time(s)); push!(dts, s.Δt)
        push!(wmax, wm); push!(umax, um); push!(qcl_max, ρqᶜˡ_max); push!(qv_max, ρqᵛ_max)
        push!(walls, (time_ns() - wall0[]) / 1e9)
        @info @sprintf("[%s] iter=%6d t=%6.2fh Δt=%5.2fs max|u|=%.2f max|w|=%.3e max(ρqcl)=%.2e wall=%.0fs",
                       label, iteration(s), time(s)/3600, s.Δt, um, wm, ρqᶜˡ_max, walls[end])
        flush(stdout)
        return nothing
    end
    add_callback!(sim, diag_cb, IterationInterval(sample_every))

    wall0[] = time_ns()
    crashed = false
    try
        run!(sim)
    catch err
        crashed = true
        @error "[$label] CRASHED" err
    end

    return (; label, iters, ts, dts, wmax, umax, qcl_max, qv_max, walls,
             crashed = crashed, final_t = time(sim), final_iter = iteration(sim))
end

# ============================================================================
# Main: run both
# ============================================================================

stop_time  = 24hours
cfl        = 0.7

# Compressible-substepper run: adaptive Δt
@info "===== Compressible-substepper run ====="
model_comp = build_compressible()
result_comp = run_with_diagnostics("comp", model_comp;
    Δt_init = 1.0, max_Δt = 60.0, cfl, stop_time, sample_every = 200)

# Anelastic run: adaptive Δt (no acoustic CFL, so larger max_Δt allowed)
@info "===== Anelastic run ====="
model_anel = build_anelastic()
result_anel = run_with_diagnostics("anel", model_anel;
    Δt_init = 1.0, max_Δt = 60.0, cfl, stop_time, sample_every = 200)

# Save
jldsave(joinpath(OUTDIR, RUN_LABEL * "_results.jld2");
        comp = result_comp, anel = result_anel,
        config = (; stop_time, cfl))

@info "Done. Output: $(joinpath(OUTDIR, RUN_LABEL))_results.jld2"
