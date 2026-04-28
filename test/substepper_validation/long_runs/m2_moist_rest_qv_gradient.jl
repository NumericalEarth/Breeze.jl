#####
##### M2 — Moist rest atmosphere with horizontal qᵛ gradient
#####
##### Acceptance test from MOIST_SUBSTEPPER_STRATEGY.md.
#####
##### Setup: 32×32×64 3-D Cartesian box, Lz=30 km. Isothermal-T₀=250 K
##### moist hydrostatic background. qᵛ varies cosine in x:
#####   qᵛ(x) = qᵛ_max · max(0, cos(π x / Lh))
##### with qᵛ_max = 10 g/kg. State is at rest; pressure profile chosen
##### so the local moist hydrostatic balance ∂z p = −ρ g holds in every
##### column.
#####
##### Pass: max|w| envelope ≤ 1e-10 m/s.
#####
##### M2 vs M1: M2 has a *spatial* qᵛ gradient. The dry substepper's
##### per-cell γᵈRᵈ vs γᵐRᵐ mismatch becomes a *spatially varying*
##### bias that projects onto horizontal acoustic and gravity-wave
##### modes. Predicted current behaviour: drifts faster than M1.
##### Predicted post-fix: ε.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2
using CloudMicrophysics

const arch    = CUDA.functional() ? GPU() : CPU()
const OUTDIR  = @__DIR__
const STEM    = joinpath(OUTDIR, "m2_moist_rest_qv_gradient")

Oceananigans.defaults.FloatType = Float64
Oceananigans.defaults.gravitational_acceleration = 9.80665

const T₀     = 250.0
const p₀     = 1.0e5
const qᵛ_max = 10.0e-3   # 10 g/kg peak vapor mass fraction

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

const g   = constants.gravitational_acceleration
const Rᵈ  = Breeze.dry_air_gas_constant(constants)
const Rᵛ  = 461.5
const cᵖᵈ = constants.dry_air.heat_capacity
const κᵈ  = Rᵈ / cᵖᵈ

const Nx = 32
const Ny = 32
const Nz = 64
const Lh = 100e3
const Lz = 30e3

# Cosine half-bump in qᵛ: zero at the periodic seam, peak at x = Lh/2.
qᵛ_xyz(x, y, z) = qᵛ_max * max(0.0, cos(π * (x - Lh/2) / Lh))

# Local mixture R given the column-local qᵛ
function R_mixture(x, y, z)
    q = qᵛ_xyz(x, y, z)
    return (1 - q) * Rᵈ + q * Rᵛ
end

# Each (x, y) column has its own moist scale height. Pressure profile:
# p(x, y, z) = p₀ exp(−z / H(x,y)) with H(x,y) = R_m(x,y) T₀ / g.
function pressure_xyz(x, y, z)
    H = R_mixture(x, y, z) * T₀ / g
    return p₀ * exp(-z / H)
end
density_xyz(x, y, z) = pressure_xyz(x, y, z) / (R_mixture(x, y, z) * T₀)

# Dry potential temperature θ = T (p₀/p)^κᵈ.
function θ_xyz(x, y, z)
    p = pressure_xyz(x, y, z)
    return T₀ * (p₀ / p)^κᵈ
end

#####
##### Grid + model
#####

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lh), y = (0, Lh), z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

T₀_ref = T₀
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

τ_relax = 200.0
relaxation = ConstantRateCondensateFormation(1 / τ_relax)
cloud_formation = NonEquilibriumCloudFormation(relaxation, relaxation)
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

weno = WENO()
bp_weno = WENO(order = 5, bounds = (0, 1))

model = AtmosphereModel(grid; dynamics, microphysics,
                        thermodynamic_constants = constants,
                        momentum_advection = weno,
                        scalar_advection = (ρθ = weno, ρqᵛ = bp_weno,
                                            ρqᶜˡ = bp_weno, ρqᶜⁱ = bp_weno,
                                            ρqʳ  = bp_weno, ρqˢ  = bp_weno),
                        timestepper = :AcousticRungeKutta3)

set!(model, θ = θ_xyz, ρ = density_xyz, qᵛ = qᵛ_xyz)

#####
##### Run + diagnostics
#####

const Δt = 20.0
const n_outer_steps = 100
const sample_every  = 5

simulation = Simulation(model; Δt, stop_iteration = n_outer_steps)

iters = Int[]; tvec = Float64[]
wmax  = Float64[]; umax = Float64[]; vmax = Float64[]
qcl_max = Float64[]; M_total = Float64[]
wall0 = Ref(time_ns())

ρ_field = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
M0 = Float64(sum(interior(ρ_field)))
qᵛ_max0 = Float64(maximum(interior(model.microphysical_fields.qᵛ)))

function diag_cb(sim)
    m = sim.model
    push!(iters, iteration(sim)); push!(tvec, time(sim))
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    vm = Float64(maximum(abs, interior(m.velocities.v)))
    qm = Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
    M  = Float64(sum(interior(ρ_field)))
    push!(wmax, wm); push!(umax, um); push!(vmax, vm)
    push!(qcl_max, qm); push!(M_total, M)
    @info @sprintf("[m2] iter=%4d t=%6.1fs max|w|=%.4e max|u|=%.4e max|v|=%.4e ρqcl=%.2e ΔM/M0=%+.2e",
                   iteration(sim), time(sim), wm, um, vm, qm, M / M0 - 1)
    flush(stdout)
end
add_callback!(simulation, diag_cb, IterationInterval(sample_every))

@info @sprintf("M2 moist rest with qᵛ gradient: 32×32×64 Lz=30km, qᵛ_max=%.1f g/kg, Δt=%.1fs, %d outer steps",
               1000 * qᵛ_max, Δt, n_outer_steps)
@info @sprintf("Substepper defaults: ω=%.2f  damping=%s",
               model.timestepper.substepper.forward_weight,
               typeof(model.timestepper.substepper.damping).name.name)
@info @sprintf("Initial qᵛ_max in domain: %.4e (target %.4e)", qᵛ_max0, qᵛ_max)

t_start = time()
crashed = false
try
    run!(simulation)
catch e
    @warn "[m2] crashed: $(sprint(showerror, e))"
    crashed = true
end
elapsed = time() - t_start

#####
##### Pass / fail
#####

env = isempty(wmax) ? NaN : maximum(wmax)
final_w = isempty(wmax) ? NaN : last(wmax)

@info @sprintf("=== M2 result ===")
@info @sprintf("envelope max|w|         = %.4e m/s  (pass threshold 1e-10)", env)
@info @sprintf("envelope max|u|         = %.4e m/s", isempty(umax) ? NaN : maximum(umax))
@info @sprintf("envelope max|v|         = %.4e m/s", isempty(vmax) ? NaN : maximum(vmax))
@info @sprintf("mass drift M[end]/M0-1  = %+.4e", M_total[end] / M0 - 1)
@info @sprintf("wall time               = %.1f s (%d steps)", elapsed, length(iters))

passed = !crashed && isfinite(env) && env <= 1e-10
if passed
    @info "✓ M2 PASS"
else
    @warn @sprintf("✗ M2 FAIL: envelope %.4e exceeds 1e-10 (or crashed=%s)", env, crashed)
end

jldsave(STEM * "_results.jld2";
        iters, tvec, wmax, umax, vmax, qcl_max, M_total, M0,
        Δt, n_outer_steps, qᵛ_max, T₀, Lz, env, final_w, passed, crashed)

@info "Done."
