#!/usr/bin/env julia
#
# 1° DCMIP2016 baroclinic wave comparison at advective CFL ≈ 0.7.
#
# Avoids the polar Δx_min trap by using latitude=(-80, 80) instead of
# (-85, 85). The smallest cell on this grid is at lat 80°:
#
#   Δx_min = a · cos(80°) · (2π / Nλ) ≈ 19.3 km
#
# At the BCI peak (U_max ≈ 60 m/s) this gives CFL = U Δt / Δx_min = 0.7
# at Δt = 225 s. Below CFL = 0.7 the WS-RK3 + WENO5 advective scheme is
# stable; any noise we see is *acoustic*, not advective. So this is the
# right configuration for comparing damping strategies as such.

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

# Domain capped at ±80° to keep Δx_min sane on a lat-lon grid.
Nλ, Nφ, Nz = 360, 160, 15
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

# Δt chosen so the BCI peak (U ≈ 60 m/s) hits CFL = 0.7 against the
# polar Δx_min = a · cos(80°) · (2π / 360) ≈ 19.3 km.
Δt = 225.0
n_days = 7
n_steps = round(Int, n_days * 86400 / Δt)
@printf("Δt = %.0f s, %d outer steps to day %d\n", Δt, n_steps, n_days)
Δx_min = a * cosd(80.0) * 2π / Nλ
@printf("Δx_min at lat 80° ≈ %.2f km, advective CFL @ U=60 m/s = %.2f\n",
        Δx_min/1000, 60.0 * Δt / Δx_min)

function run_one(damping, label, savefile)
    @printf("\n========================================\n")
    @printf("=== %s ===\n", label)
    @printf("========================================\n")

    grid = LatitudeLongitudeGrid(arch; size=(Nλ, Nφ, Nz), halo=(5, 5, 5),
                                 longitude=(0, 360), latitude=(-80, 80), z=(0, H))
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

    function bottom_pressure_min(model)
        p = interior(model.dynamics.pressure, :, :, 1)
        return minimum(p)
    end

    ts_t   = Float64[]; ts_w = Float64[]; ts_u = Float64[]
    ts_v   = Float64[]; ts_psm = Float64[]

    t0 = time()
    crashed_at = 0
    daily_print_every = max(1, round(Int, 86400 / Δt))
    for step in 1:n_steps
        try
            time_step!(model, Δt)
        catch err
            crashed_at = step
            @printf("EXCEPTION at outer step %d (t=%.2f days): %s\n",
                    step, model.clock.time/86400, first(sprint(showerror, err), 200))
            break
        end
        if any(isnan, parent(model.dynamics.density))
            crashed_at = step
            @printf("NaN at outer step %d (t=%.2f days)\n", step, model.clock.time/86400)
            break
        end

        push!(ts_t,   model.clock.time / 86400)
        push!(ts_w,   maximum(abs, interior(model.velocities.w)))
        push!(ts_u,   maximum(abs, interior(model.velocities.u)))
        push!(ts_v,   maximum(abs, interior(model.velocities.v)))
        push!(ts_psm, bottom_pressure_min(model))

        if step % daily_print_every == 0 || step == n_steps
            elapsed = time() - t0
            day = model.clock.time / 86400
            @printf("day %5.2f  max|w|=%.3e  max|u|=%5.1f  max|v|=%5.2f  min(p_bot)=%.1f hPa  [wall %.0fs]\n",
                    day, ts_w[end], ts_u[end], ts_v[end], ts_psm[end]/100, elapsed)
            flush(stdout)
        end
    end

    wall = time() - t0
    final_day = length(ts_t) > 0 ? ts_t[end] : 0.0
    @printf("\n--- %s done: %.0fs wall, reached day %.2f ---\n", label, wall, final_day)
    jldsave(savefile;
            label = label, dt = Δt,
            ts_t, ts_w, ts_u, ts_v, ts_psm,
            crashed_at, wall_seconds = wall, final_day = final_day)
    @printf("Saved %s\n", savefile)
    flush(stdout)

    # Pull max|w| over the trajectory excluding the first 5 startup steps
    skip = 5
    body_max_w = length(ts_w) > skip ? maximum(@view ts_w[skip+1:end]) : NaN
    return (label = label,
            crashed_at = crashed_at,
            final_day = final_day,
            full_max_w = length(ts_w) > 0 ? maximum(ts_w) : NaN,
            body_max_w = body_max_w,
            day7_w = length(ts_w) > 0 ? ts_w[end] : NaN,
            min_psm_hPa = length(ts_psm) > 0 ? minimum(ts_psm)/100 : NaN,
            wall = wall)
end

# ── Six runs ─────────────────────────────────────────────────────────────────
configs = [
    (NoDivergenceDamping(),                              "NoDivergenceDamping",                  "bw_cfl07_none.jld2"),
    (ThermodynamicDivergenceDamping(coefficient=0.1),    "ThermodynamicDivergenceDamping(0.1)",  "bw_cfl07_thermo01.jld2"),
    (ThermodynamicDivergenceDamping(coefficient=0.5),    "ThermodynamicDivergenceDamping(0.5)",  "bw_cfl07_thermo05.jld2"),
    (ConservativeProjectionDamping(coefficient=0.1),     "ConservativeProjectionDamping(0.1)",   "bw_cfl07_cons01.jld2"),
    (PressureProjectionDamping(coefficient=0.1),         "PressureProjectionDamping(0.1)",       "bw_cfl07_press01.jld2"),
    (PressureProjectionDamping(coefficient=0.5),         "PressureProjectionDamping(0.5)",       "bw_cfl07_press05.jld2"),
]

results = []
for (damping, label, savefile) in configs
    push!(results, run_one(damping, label, savefile))
end

@printf("\n\n================================ SUMMARY ================================\n")
@printf("%-40s %12s %14s %14s %14s\n",
        "strategy", "crashed at", "final day", "body max|w|", "day-7 max|w|")
for r in results
    crash_str = r.crashed_at == 0 ? "—" : "step $(r.crashed_at)"
    @printf("%-40s %12s %14.2f %14.4e %14.4e\n",
            r.label, crash_str, r.final_day, r.body_max_w, r.day7_w)
end
@printf("=========================================================================\n")
