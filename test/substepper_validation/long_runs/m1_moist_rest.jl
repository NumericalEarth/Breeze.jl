#####
##### M1 — Moist rest atmosphere with uniform qᵛ
#####
##### Acceptance test from MOIST_SUBSTEPPER_STRATEGY.md.
#####
##### Setup: 32×32×64 3-D Cartesian box, Lz=30 km. Isothermal-T₀=250 K
##### moist hydrostatic background with uniform qᵛ = 1 g/kg. State is
##### bit-quiet — u=v=w=0, no microphysical activity (sub-saturated
##### everywhere). Run 100 outer steps at Δt=20 s and measure the
##### envelope of max|w|.
#####
##### Pass: max|w| envelope ≤ 1e-10 m/s (matches the dry M0 bound).
#####
##### Predicted CURRENT result (dry substepper on moist state): drift
##### well above the bound. The substepper closes the linearised PGF
##### with γᵈRᵈ but the actual EoS uses γᵐRᵐ, so the mismatch creates
##### a per-outer-step injection.
#####
##### Predicted POST-FIX result (Phase 2 of MOIST_SUBSTEPPER_STRATEGY.md
##### / PRISTINE §A3/B1/B2): drift at machine ε, matching M0.
#####
##### This is the cheapest single-knob test of moist-substepper
##### correctness. Cartesian box, no curvilinear metric, no qᵛ
##### gradient, no surface fluxes, no condensation.
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
const STEM    = joinpath(OUTDIR, "m1_moist_rest")

Oceananigans.defaults.FloatType = Float64
Oceananigans.defaults.gravitational_acceleration = 9.80665

const T₀  = 250.0       # Isothermal background temperature (K)
const p₀  = 1.0e5       # Surface pressure (Pa)
const qᵛ_const = 1.0e-3 # 1 g/kg uniform vapor mass fraction

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

const g   = constants.gravitational_acceleration
const Rᵈ  = Breeze.dry_air_gas_constant(constants)
const Rᵛ  = 461.5
const cᵖᵈ = constants.dry_air.heat_capacity
const κᵈ  = Rᵈ / cᵖᵈ

# Moist mixture gas constant for the uniform-qᵛ background
const Rᵐ_bg = (1 - qᵛ_const) * Rᵈ + qᵛ_const * Rᵛ
const H_m   = Rᵐ_bg * T₀ / g   # Moist scale height

# Moist hydrostatic profile (∂z p = −ρ g, p = ρ Rᵐ T):
pressure_profile(z) = p₀ * exp(-z / H_m)
density_profile(z)  = pressure_profile(z) / (Rᵐ_bg * T₀)

# Dry potential temperature θ = T (p₀/p)^κᵈ. With T = T₀ isothermal
# and the moist exponential pressure profile, this rises with z.
function θ_profile(λ_or_x, φ_or_y, z)
    p = pressure_profile(z)
    return T₀ * (p₀ / p)^κᵈ
end
density_xyz(x, y, z) = density_profile(z)
qᵛ_xyz(x, y, z)      = qᵛ_const

#####
##### Grid + model
#####

const Nx = 32
const Ny = 32
const Nz = 64
const Lh = 100e3
const Lz = 30e3

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lh), y = (0, Lh), z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

# Reference state uses the dry θ profile (isothermal-T₀ scale height).
# Note: this is intentionally inconsistent with the moist hydrostatic IC
# in the basic state's pressure scale height — that mismatch is part of
# what the dry substepper is being asked to absorb. Phase 2 should still
# keep max|w| at ε regardless, because the substepper's frozen `p⁰` is
# read from the EoS at U⁰, not from the reference state.
T₀_ref = T₀
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

# Microphysics is required to expose ρqᵛ as a prognostic. Use the same
# OneMomentCloudMicrophysics + NonEquilibriumCloudFormation as the moist
# BW; at qᵛ=1 g/kg the column is sub-saturated everywhere (sat qᵛ at
# T=250 K, p=1 bar is ~3.7 g/kg) so no condensation fires.
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

set!(model, θ = θ_profile, ρ = density_xyz, qᵛ = qᵛ_xyz)

#####
##### Run + diagnostics
#####

const Δt = 20.0
const n_outer_steps = 100
const sample_every  = 5

simulation = Simulation(model; Δt, stop_iteration = n_outer_steps)

iters = Int[]; tvec = Float64[]
wmax  = Float64[]; umax = Float64[]
qcl_max = Float64[]; psurf = Float64[]; M_total = Float64[]
wall0 = Ref(time_ns())

ρ_field = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
M0 = Float64(sum(interior(ρ_field)))

function diag_cb(sim)
    m = sim.model
    push!(iters, iteration(sim)); push!(tvec, time(sim))
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    qm = Float64(maximum(interior(m.microphysical_fields.ρqᶜˡ)))
    pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
    M  = Float64(sum(interior(ρ_field)))
    push!(wmax, wm); push!(umax, um); push!(qcl_max, qm)
    push!(psurf, pm); push!(M_total, M)
    @info @sprintf("[m1] iter=%4d t=%6.1fs max|w|=%.4e max|u|=%.4e max(ρqcl)=%.2e ΔM/M0=%+.2e",
                   iteration(sim), time(sim), wm, um, qm, M / M0 - 1)
    flush(stdout)
end
add_callback!(simulation, diag_cb, IterationInterval(sample_every))

@info @sprintf("M1 moist rest: 32×32×64 Lz=30km, qᵛ=%.1f g/kg, Δt=%.1fs, %d outer steps",
               1000 * qᵛ_const, Δt, n_outer_steps)
@info @sprintf("Substepper defaults: ω=%.2f  damping=%s",
               model.timestepper.substepper.forward_weight,
               typeof(model.timestepper.substepper.damping).name.name)

t_start = time()
crashed = false
try
    run!(simulation)
catch e
    @warn "[m1] crashed: $(sprint(showerror, e))"
    crashed = true
end
elapsed = time() - t_start

#####
##### Pass / fail
#####

env = isempty(wmax) ? NaN : maximum(wmax)
final_w = isempty(wmax) ? NaN : last(wmax)
growth = if length(wmax) > 1 && wmax[1] > 0
    (last(wmax) / first(filter(>(0), wmax))) ^ (1 / (length(wmax) - 1))
else
    NaN
end

@info @sprintf("=== M1 result ===")
@info @sprintf("envelope max|w|         = %.4e m/s  (pass threshold 1e-10)", env)
@info @sprintf("final max|w|            = %.4e m/s", final_w)
@info @sprintf("per-sample growth ratio = %.4f", growth)
@info @sprintf("mass drift M[end]/M0-1  = %+.4e", M_total[end] / M0 - 1)
@info @sprintf("wall time               = %.1f s (%d steps)", elapsed, length(iters))

passed = !crashed && isfinite(env) && env <= 1e-10
if passed
    @info "✓ M1 PASS"
else
    @warn @sprintf("✗ M1 FAIL: envelope %.4e exceeds 1e-10 (or crashed=%s)", env, crashed)
end

jldsave(STEM * "_results.jld2";
        iters, tvec, wmax, umax, qcl_max, psurf, M_total, M0,
        Δt, n_outer_steps, qᵛ_const, T₀, Lz, env, final_w, growth, passed, crashed)

@info "Done."
