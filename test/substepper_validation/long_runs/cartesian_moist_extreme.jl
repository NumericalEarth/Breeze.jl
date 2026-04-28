#####
##### Test B — Extreme Cartesian moist test, matching lat-lon BW dynamism.
#####
##### The original `cartesian_moist_v2.jl` ran cleanly but in a far milder
##### regime than the lat-lon BW: Δt=1s, 4K SST contrast, 5g/kg uniform qᵛ
##### and Δx=156m. To stress-test the substepper against a regime that
##### matches the failing lat-lon moist BW, this test uses:
#####
#####   - Δx = 20 km   (matches the lat-lon polar minimum spacing)
#####   - Δt = 20 s    (the failing lat-lon BW value)
#####   - SST gradient 240→310 K (matches BW pole-to-equator)
#####   - qᵛ profile peaked at warm regions, BW-style (Gaussian in x × pressure)
#####   - Same NonEquilibriumCloudFormation + τ_relax=200s microphysics
#####   - Surface fluxes ON
#####   - 2-D Periodic-Flat-Bounded box, no Coriolis (just stress test)
#####
##### If THIS Cartesian test ALSO fails the same way as the lat-lon BW,
##### the curvilinear-grid hypothesis is wrong; the issue is the moist
##### substepper at production Δt with strong forcings, regardless of
##### geometry. If it succeeds, the lat-lon-specific code path matters.
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
const RUN_LABEL = "cartesian_moist_extreme"
const STEM = joinpath(OUTDIR, RUN_LABEL)

Oceananigans.defaults.FloatType = Float32

const constants = ThermodynamicConstants(;
    gravitational_acceleration = 9.80665,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)
const Rᵈ = Breeze.dry_air_gas_constant(constants)
const cᵖᵈ = constants.dry_air.heat_capacity
const g = constants.gravitational_acceleration
const κ = Rᵈ / cᵖᵈ
const p₀ = 1e5
const θ₀ = 250.0   # isothermal-T₀=250K reference, matching BW
const Lx = 4000kilometers
const Lz = 30kilometers
const Nx = 200      # Δx = 20 km, matches lat-lon polar
const Nz = 64       # Δz ≈ 470 m, matches BW

# SST gradient: 240 K at one end, 310 K at the other — matches BW pole-to-equator
const T_low = 240.0
const T_high = 310.0
function T_surface_field(x)
    # Smooth half-cosine over Lx so SST varies from T_low at x=0 to
    # T_high at x=Lx/2 and back, mimicking pole-equator-pole.
    s = sin(π * (x + Lx/2) / Lx)
    return T_low + (T_high - T_low) * s^2
end

# qᵛ profile: BW-style Gaussian-in-x × Gaussian-in-pressure
function specific_humidity(x, z)
    q₀_max = 0.018; qₜ = 1e-12
    pʷ = 34000.0
    # Hydrostatic-ish pressure for the q profile (approx, isothermal at T₀_ref=250)
    p_approx = p₀ * exp(-g*z/(Rᵈ * θ₀))
    η = p_approx / p₀
    # x-Gaussian: peaked where T_surface is warm
    s = sin(π * (x + Lx/2) / Lx); xfrac = s^2
    q_trop = q₀_max * xfrac * exp(-((η-1)*p₀/pʷ)^2)
    return ifelse(η > 0.1, q_trop, qₜ)
end

# Stratified θ background (isothermal-T₀=θ₀)
θᵇᵍ(z) = θ₀ * exp(g * z / (cᵖᵈ * θ₀))

const Cᴰ = 1e-3
const Uᵍ = 1e-2

function build_grid()
    RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                    x = (-Lx/2, Lx/2),
                    z = (0, Lz),
                    topology = (Periodic, Flat, Bounded))
end

function make_bcs()
    return (
        ρu  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_field)),
        ρθ  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_field)),
        ρqᵛ = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T_surface_field)),
    )
end

function build_microphysics()
    BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    τ_relax = 200.0
    relaxation = ConstantRateCondensateFormation(1/τ_relax)
    cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
    return BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics(; cloud_formation)
end

function build_compressible()
    grid = build_grid()
    td = SplitExplicitTimeDiscretization()
    dyn = CompressibleDynamics(td;
                               surface_pressure = p₀,
                               reference_potential_temperature = θᵇᵍ)
    weno = WENO(order = 5)
    bp_weno = WENO(order = 5, bounds = (0, 1))
    bcs = make_bcs()
    model = AtmosphereModel(grid;
                            momentum_advection = weno,
                            scalar_advection   = (ρθ = weno, ρqᵛ = bp_weno,
                                                  ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                                                  ρqʳ = bp_weno, ρqˢ = bp_weno),
                            microphysics = build_microphysics(),
                            thermodynamic_constants = constants,
                            dynamics = dyn,
                            boundary_conditions = bcs,
                            timestepper = :AcousticRungeKutta3)
    ref = model.dynamics.reference_state
    set!(model; θ = (x, z) -> θᵇᵍ(z),
                ρ = ref.density,
                qᵛ = (x, z) -> specific_humidity(x, z))
    return model
end

# Diagnostics
diag_iters = Int[]; diag_t = Float64[]; diag_wmax = Float64[]; diag_umax = Float64[]
diag_qcl_max = Float64[]; diag_qv_max = Float64[]; diag_psurf = Float64[]
diag_M = Float64[]; diag_walls = Float64[]
M0_ref = Ref(0.0)
wall0 = Ref(time_ns())

function diag_cb(sim)
    m = sim.model
    push!(diag_iters, iteration(sim)); push!(diag_t, time(sim))
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    qcm = Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
    qvm = Float64(maximum(interior(m.moisture_density)))
    pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
    ρ_field = Breeze.AtmosphereModels.dynamics_density(m.dynamics)
    M = Float64(sum(interior(ρ_field)))
    if iteration(sim) == 0
        M0_ref[] = M
    end
    push!(diag_wmax, wm); push!(diag_umax, um); push!(diag_qcl_max, qcm)
    push!(diag_qv_max, qvm); push!(diag_psurf, pm); push!(diag_M, M)
    push!(diag_walls, (time_ns()-wall0[])/1e9)
    @info @sprintf("[cart-extreme] iter=%6d t=%6.3fd Δt=%5.1fs max|u|=%.2f max|w|=%.3e max(qcl)=%.2e max(qv)=%.3e p_surf=%.0f ΔM/M0=%.2e wall=%.0fs",
                   iteration(sim), time(sim)/86400, sim.Δt, um, wm, qcm, qvm, pm,
                   (M-M0_ref[])/M0_ref[], diag_walls[end])
    flush(stdout)
end

# RUN
@info "===== Extreme Cartesian moist test ====="
@info "  Δx = $(Lx/Nx/1000) km, Δz = $(Lz/Nz) m, Δt = 20 s"
@info "  SST: $T_low – $T_high K"
@info "  qᵛ_max ≈ 18 g/kg at warm SST"
@info "  Run length: 5 days"

model = build_compressible()
Δt = 20seconds
stop_time = 5days
sim = Simulation(model; Δt, stop_time)
add_callback!(sim, diag_cb, IterationInterval(200))

wall0[] = time_ns()
crashed = false
try
    run!(sim)
    @info "[cart-extreme] RUN COMPLETED"
catch e
    crashed = true
    @error "[cart-extreme] RUN FAILED" e
end

jldsave(STEM*"_diagnostics.jld2";
        iters=diag_iters, t=diag_t, wmax=diag_wmax, umax=diag_umax,
        qcl_max=diag_qcl_max, qv_max=diag_qv_max,
        psurf_min=diag_psurf, total_mass=diag_M, wall=diag_walls,
        M0=M0_ref[], crashed=crashed,
        config=(; Lx, Lz, Nx, Nz, T_low, T_high, Δt, stop_time))

@info "Done. final t=$(time(sim))s ($(time(sim)/86400)d), crashed=$crashed"
