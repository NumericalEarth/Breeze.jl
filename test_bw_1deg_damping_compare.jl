#!/usr/bin/env julia
#
# 1° DCMIP2016 baroclinic wave: damping strategy comparison at Δt=1400.
#
# Runs the canonical DCMIP2016 baroclinic wave (Ullrich et al. 2014) at
# Δt=1400 for 14 days with each of the four AcousticDampingStrategy variants.
# For each strategy, we save:
#   - per-step timeseries: max|u|, max|v|, max|w|, min(p_bot)
#   - crash time (or "ran to completion")
#
# Saved JLD2 fields are named so that downstream plotting can load all four
# in one shot.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

Nλ, Nφ, Nz = 360, 170, 15
H = 30kilometers

arch = CUDA.functional() ? GPU() : CPU()

# DCMIP2016 BW initial condition (Ullrich et al. 2014)
Tᴱ = 310.0; Tᴾ = 240.0; Tₘ = (Tᴱ + Tᴾ) / 2
Γ = 0.005; K = 3; b_ = 2

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g; η = z / (b_ * Hₛ); e = exp(-η^2)
    A = (Tₘ - Tᴾ) / (Tₘ * Tᴾ); C = (K + 2) / 2 * (Tᴱ - Tᴾ) / (Tᴱ * Tᴾ)
    τ₁ = exp(Γ * z / Tₘ) / Tₘ + A * (1 - 2η^2) * e
    τ₂ = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ * z / Tₘ) - 1) / Γ + A * z * e
    ∫τ₂ = C * z * e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ) = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)
T_ic(λ, φ, z) = 1 / (τ_and_integrals(z)[1] - τ_and_integrals(z)[2] * F(φ))
p_ic(λ, φ, z) = p₀ * exp(-g / Rᵈ * (τ_and_integrals(z)[3] - τ_and_integrals(z)[4] * F(φ)))
ρ_ic(λ, φ, z) = p_ic(λ, φ, z) / (Rᵈ * T_ic(λ, φ, z))
θ_ic(λ, φ, z) = T_ic(λ, φ, z) * (p₀ / p_ic(λ, φ, z))^κ

function u_ic(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z); T = T_ic(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * T
    rcosφ = a * cosd(φ); Ωrcosφ = Ω * rcosφ
    u_bal = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
    uₚ = 1.0; rₚ = 0.1; λₚ = π / 9; φₚ = 2π / 9; zₚ = 15000.0
    φʳ = deg2rad(φ); λʳ = deg2rad(λ)
    gc = acos(sin(φₚ) * sin(φʳ) + cos(φₚ) * cos(φʳ) * cos(λʳ - λₚ)) / rₚ
    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, 0.0)
    u_pert = ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
    return u_bal + u_pert
end

T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

# ── Run one strategy ──────────────────────────────────────────────────────────
function run_one_strategy(damping, label, savefile)
    @printf("\n========================================\n")
    @printf("=== %s ===\n", label)
    @printf("========================================\n")

    grid = LatitudeLongitudeGrid(arch; size=(Nλ, Nφ, Nz), halo=(5, 5, 5),
                                 longitude=(0, 360), latitude=(-85, 85), z=(0, H))

    coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
    td = SplitExplicitTimeDiscretization(;
        damping = damping,
        acoustic_damping_coefficient = 0.5,
        substep_distribution = ProportionalSubsteps())
    dynamics = CompressibleDynamics(td;
                                    surface_pressure=p₀,
                                    reference_potential_temperature=θ_ref)
    model = AtmosphereModel(grid; dynamics, coriolis,
                            thermodynamic_constants=constants, advection=WENO(),
                            timestepper=:AcousticRungeKutta3)
    set!(model; θ=θ_ic, u=u_ic, ρ=ρ_ic)

    Δt = 1400.0    # the comparison Δt
    n_days = 14
    n_steps = round(Int, n_days * 86400 / Δt)

    @printf("Grid: %d×%d×%d  Δt=%.0fs  %d outer steps to day %d\n",
            Nλ, Nφ, Nz, Δt, n_steps, n_days)
    @printf("Initial: max|u|=%.2f m/s, max|w|=%.2e\n",
            maximum(abs, interior(model.velocities.u)),
            maximum(abs, interior(model.velocities.w)))

    function bottom_pressure_min(model)
        p = interior(model.dynamics.pressure, :, :, 1)
        return minimum(p)
    end

    ts_t   = Float64[]
    ts_w   = Float64[]
    ts_u   = Float64[]
    ts_v   = Float64[]
    ts_psm = Float64[]

    t0 = time()
    crashed_at = 0
    crash_reason = ""
    for step in 1:n_steps
        try
            time_step!(model, Δt)
        catch err
            crashed_at = step
            crash_reason = sprint(showerror, err)
            @printf("EXCEPTION at outer step %d (t=%.2f days): %s\n",
                    step, model.clock.time/86400, crash_reason)
            break
        end
        if any(isnan, parent(model.dynamics.density))
            crashed_at = step
            crash_reason = "NaN in density"
            @printf("NaN at outer step %d (t=%.2f days)\n", step, model.clock.time/86400)
            break
        end

        push!(ts_t,   model.clock.time / 86400)
        push!(ts_w,   maximum(abs, interior(model.velocities.w)))
        push!(ts_u,   maximum(abs, interior(model.velocities.u)))
        push!(ts_v,   maximum(abs, interior(model.velocities.v)))
        push!(ts_psm, bottom_pressure_min(model))

        # Daily progress print
        if step % round(Int, 86400/Δt) == 0 || step == n_steps
            elapsed = time() - t0
            day = model.clock.time / 86400
            @printf("day %5.1f  max|w|=%.3e  max|u|=%5.1f  max|v|=%5.2f  min(p_bot)=%.1f hPa  [wall %.0fs]\n",
                    day, ts_w[end], ts_u[end], ts_v[end], ts_psm[end]/100, elapsed)
            flush(stdout)
        end
    end

    wall = time() - t0
    @printf("\n--- %s done: %.0fs wall ---\n", label, wall)
    if crashed_at == 0
        @printf("ran to day %.1f, max|u| reached %.1f, max|w| reached %.4f, min(p_bot) %.1f hPa\n",
                n_days, maximum(ts_u), maximum(ts_w), minimum(ts_psm)/100)
    end

    jldsave(savefile;
            label = label,
            ts_t, ts_w, ts_u, ts_v, ts_psm,
            crashed_at, crash_reason,
            wall_seconds = wall,
            Nlambda = Nλ, Nphi = Nφ, Nz = Nz, dt = Δt, n_days = n_days)
    @printf("Saved %s\n", savefile)
    flush(stdout)
    return crashed_at, ts_w, ts_psm, wall
end

# ── Run all four strategies ───────────────────────────────────────────────────
strategies = [
    (NoDivergenceDamping(),                              "NoDivergenceDamping",                  "bw_dt1400_none.jld2"),
    (ThermodynamicDivergenceDamping(coefficient=0.1),    "ThermodynamicDivergenceDamping(0.1)",  "bw_dt1400_thermo.jld2"),
    (ConservativeProjectionDamping(coefficient=0.1),     "ConservativeProjectionDamping(0.1)",   "bw_dt1400_cons.jld2"),
    (PressureProjectionDamping(coefficient=0.1),         "PressureProjectionDamping(0.1)",       "bw_dt1400_press.jld2"),
]

results = []
for (damping, label, savefile) in strategies
    cr, ts_w, ts_psm, wall = run_one_strategy(damping, label, savefile)
    push!(results, (label, cr, length(ts_w) > 0 ? maximum(ts_w) : NaN,
                            length(ts_psm) > 0 ? minimum(ts_psm)/100 : NaN, wall))
end

@printf("\n\n================ SUMMARY ================\n")
@printf("%-40s %12s %14s %14s %10s\n", "strategy", "crashed?", "max|w|", "min(p_bot) hPa", "wall (s)")
for r in results
    crash_str = r[2] == 0 ? "—" : "step $(r[2])"
    @printf("%-40s %12s %14.4e %14.1f %10.0f\n", r[1], crash_str, r[3], r[4], r[5])
end
@printf("==========================================\n")
