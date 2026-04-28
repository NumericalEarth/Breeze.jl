#####
##### M3 — Acoustic pulse in moist atmosphere
#####
##### Acceptance test from MOIST_SUBSTEPPER_STRATEGY.md.
#####
##### Setup: 2-D Cartesian box, Lx=80 km, Lz=10 km, 512×64 cells
##### (Δx ≈ 156 m). Uniform moist hydrostatic background:
##### T = T₀ = 300 K, qᵛ = 10 g/kg, no condensate. A small Gaussian
##### (ρθ)′ bump (FWHM ~ 2 km) is placed at x = Lx/2. The bump splits
##### into left- and right-going acoustic pulses. We track the
##### right-going pulse position over 80 simulated seconds and fit
##### its propagation speed.
#####
##### Expected speeds (T₀ = 300 K):
#####   cs_dry   = √(γᵈ Rᵈ T₀) = √(1.4 · 287 · 300) ≈ 347.2 m/s
#####   cs_moist = √(γᵐ Rᵐ T₀) ≈ 349.0 m/s   (qᵛ=10 g/kg)
##### Difference: ~1.8 m/s, or 0.52 %.
#####
##### Pass: measured cs is within 1 % of cs_moist.
#####
##### Predicted CURRENT (dry substepper): measured cs ≈ cs_dry; off by
##### ~ 0.5 % below cs_moist — close enough that a 1 % bound just barely
##### fails. This test is sensitive to the linearised PGF coefficient
##### γᵈRᵈ vs γᵐRᵐ.
#####
##### Predicted POST-FIX: measured cs matches cs_moist within ~ 0.1 %
##### (limited by the resolution of the pulse-tracking).
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
const STEM    = joinpath(OUTDIR, "m3_moist_acoustic_pulse")

Oceananigans.defaults.FloatType = Float64
Oceananigans.defaults.gravitational_acceleration = 9.80665

const T₀     = 300.0      # Background temperature (K)
const p₀     = 1.0e5
const qᵛ_const = 10.0e-3  # 10 g/kg uniform vapor

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

const g   = constants.gravitational_acceleration
const Rᵈ  = Breeze.dry_air_gas_constant(constants)
const Rᵛ  = 461.5
const cᵖᵈ = constants.dry_air.heat_capacity
const cᵖᵛ = 1850.0
const κᵈ  = Rᵈ / cᵖᵈ

# Mixture quantities for the uniform background
const Rᵐ_bg = (1 - qᵛ_const) * Rᵈ + qᵛ_const * Rᵛ
const cᵖᵐ_bg = (1 - qᵛ_const) * cᵖᵈ + qᵛ_const * cᵖᵛ
const cᵛᵐ_bg = cᵖᵐ_bg - Rᵐ_bg
const γᵐ_bg  = cᵖᵐ_bg / cᵛᵐ_bg
const cᵛᵈ    = cᵖᵈ - Rᵈ
const γᵈ     = cᵖᵈ / cᵛᵈ
const cs_dry   = sqrt(γᵈ * Rᵈ * T₀)
const cs_moist = sqrt(γᵐ_bg * Rᵐ_bg * T₀)

const H_m = Rᵐ_bg * T₀ / g
pressure_profile(z) = p₀ * exp(-z / H_m)
density_profile(z)  = pressure_profile(z) / (Rᵐ_bg * T₀)

#####
##### Initial condition: hydrostatic + Gaussian (ρθ)′ pulse
#####

const Lx = 80e3
const Lz = 10e3
const Nx = 512
const Nz = 64

const x_pulse = Lx / 2
const σ_pulse = 1.0e3      # 1 km Gaussian width
const A_pulse = 0.5        # K — small θ perturbation for linearity

# Background dry potential temperature θ̄ = T₀ (p₀/p)^κᵈ
function θ_background(x, y, z)
    p = pressure_profile(z)
    return T₀ * (p₀ / p)^κᵈ
end
function θ_initial(x, y, z)
    base = θ_background(x, y, z)
    bump = A_pulse * exp(-((x - x_pulse) / σ_pulse)^2)
    return base + bump
end
density_xyz(x, y, z) = density_profile(z)
qᵛ_xyz(x, y, z)      = qᵛ_const

#####
##### Grid + model
#####

# 2-D: Periodic in x, Flat in y, Bounded in z
grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       halo = (5, 5),
                       x = (0, Lx), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

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

set!(model, θ = θ_initial, ρ = density_xyz, qᵛ = qᵛ_xyz)

#####
##### Tracking
#####
##### After the initial split, the right-going pulse front travels at cs.
##### Track it via the location of max(ρθ)′ in the right half-domain
##### x ∈ (x_pulse, Lx) at each diagnostic step.
#####

const Δt = 1.0
const n_outer_steps = 80    # 80 simulated seconds
const sample_every  = 2

simulation = Simulation(model; Δt, stop_iteration = n_outer_steps)

x_centers = collect(LinRange(Lx / (2Nx), Lx * (2Nx - 1) / (2Nx), Nx))
i_pulse_init = argmin(abs.(x_centers .- x_pulse))

iters = Int[]; tvec = Float64[]
xpos_R = Float64[]; xpos_L = Float64[]
amp_R  = Float64[]; amp_L  = Float64[]
wmax = Float64[]

ρ_field   = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
ρθ_field  = model.formulation.potential_temperature_density

# Capture the IC θ̄(x, z) profile (z-dependent) so we can subtract it cleanly.
θ_bg_xz = Array(interior(ρθ_field) ./ interior(ρ_field))[:, 1, :]   # (Nx, Nz)

function diag_cb(sim)
    m = sim.model
    push!(iters, iteration(sim)); push!(tvec, time(sim))
    ρ  = Array(interior(ρ_field))[:, 1, :]
    ρθ = Array(interior(ρθ_field))[:, 1, :]
    θ  = ρθ ./ ρ
    θp = θ .- θ_bg_xz
    # Search for the right-going pulse: column-max of |θ′| over k, restricted to
    # x ∈ (x_pulse + 0.5 km, Lx). Symmetric for left-going.
    mask_R = x_centers .> (x_pulse + 500.0)
    mask_L = x_centers .< (x_pulse - 500.0)
    col_amp = vec(maximum(abs.(θp), dims = 2))
    iR = findmax(col_amp .* mask_R)[2]
    iL = findmax(col_amp .* mask_L)[2]
    push!(xpos_R, x_centers[iR]); push!(amp_R, col_amp[iR])
    push!(xpos_L, x_centers[iL]); push!(amp_L, col_amp[iL])
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    push!(wmax, wm)
    @info @sprintf("[m3] iter=%3d t=%5.1fs xR=%6.2f km xL=%6.2f km ampR=%.3e ampL=%.3e max|w|=%.3e",
                   iteration(sim), time(sim),
                   x_centers[iR] / 1e3, x_centers[iL] / 1e3,
                   col_amp[iR], col_amp[iL], wm)
    flush(stdout)
end
add_callback!(simulation, diag_cb, IterationInterval(sample_every))

@info @sprintf("M3 moist acoustic pulse: 2-D %d×%d, Lx=%.0fkm, Lz=%.0fkm, qᵛ=%.1f g/kg",
               Nx, Nz, Lx / 1e3, Lz / 1e3, 1000 * qᵛ_const)
@info @sprintf("Predicted cs_dry = %.3f m/s, cs_moist = %.3f m/s (Δ = %.3f m/s, %.3f %%)",
               cs_dry, cs_moist, cs_moist - cs_dry, 100 * (cs_moist - cs_dry) / cs_moist)
@info @sprintf("Substepper: ω=%.2f  damping=%s",
               model.timestepper.substepper.forward_weight,
               typeof(model.timestepper.substepper.damping).name.name)

t_start = time()
crashed = false
try
    run!(simulation)
catch e
    @warn "[m3] crashed: $(sprint(showerror, e))"
    crashed = true
end
elapsed = time() - t_start

#####
##### Linear fit cs from the right-going pulse trajectory
#####

if length(tvec) >= 4
    # Skip the first sample (initial-pulse split is non-linear); use a
    # window where the pulse has separated from the IC and hasn't wrapped.
    skip_first = max(1, length(tvec) ÷ 6)
    keep = (skip_first + 1):length(tvec)
    t_fit = tvec[keep] .- tvec[skip_first]
    x_fit_R = xpos_R[keep] .- xpos_R[skip_first]
    x_fit_L = xpos_L[keep] .- xpos_L[skip_first]
    # Least-squares slope through origin: cs = sum(t·x)/sum(t²)
    cs_R = sum(t_fit .* x_fit_R) / sum(t_fit .^ 2)
    cs_L = -sum(t_fit .* x_fit_L) / sum(t_fit .^ 2)
    cs_avg = (cs_R + cs_L) / 2
    err_R = abs(cs_R - cs_moist) / cs_moist
    err_L = abs(cs_L - cs_moist) / cs_moist
    err_avg = abs(cs_avg - cs_moist) / cs_moist

    @info @sprintf("=== M3 result ===")
    @info @sprintf("right-going pulse cs = %.3f m/s  (err vs cs_moist = %.3f %%)", cs_R, 100*err_R)
    @info @sprintf("left-going  pulse cs = %.3f m/s  (err vs cs_moist = %.3f %%)", cs_L, 100*err_L)
    @info @sprintf("avg               cs = %.3f m/s  (err vs cs_moist = %.3f %%)", cs_avg, 100*err_avg)
    @info @sprintf("expected cs_dry      = %.3f m/s, cs_moist = %.3f m/s", cs_dry, cs_moist)

    passed = !crashed && isfinite(cs_avg) && err_avg <= 0.01
    if passed
        @info "✓ M3 PASS"
    else
        @warn @sprintf("✗ M3 FAIL: |cs - cs_moist| / cs_moist = %.3f %% > 1 %% (or crashed=%s)",
                       100 * err_avg, crashed)
    end

    jldsave(STEM * "_results.jld2";
            iters, tvec, xpos_R, xpos_L, amp_R, amp_L, wmax,
            Δt, n_outer_steps, qᵛ_const, T₀,
            cs_dry, cs_moist, cs_R, cs_L, cs_avg, err_avg, passed, crashed)
else
    @warn "Insufficient samples to fit cs"
    jldsave(STEM * "_results.jld2";
            iters, tvec, xpos_R, xpos_L, amp_R, amp_L, wmax,
            Δt, n_outer_steps, qᵛ_const, T₀,
            cs_dry, cs_moist, passed = false, crashed = crashed)
end

@info @sprintf("wall time = %.1f s", elapsed)
@info "Done."
