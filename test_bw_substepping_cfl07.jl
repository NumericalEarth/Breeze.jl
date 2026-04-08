#!/usr/bin/env julia
#
# DCMIP2016 baroclinic wave with substepping at advective CFL = 0.7.
#
# Domain spans latitude=(-80, 80) to avoid the polar Δx_min trap. The outer
# time step is set so the BCI-peak jet (U_max ≈ 60 m/s) hits CFL = 0.7
# against the polar minimum cell width:
#
#     Δx_min = a · cos(80°) · 2π / Nλ
#     Δt_max = 0.7 · Δx_min / U_max
#
# Substepping uses Breeze's default damping
# (PressureProjectionDamping(coefficient = 0.1), the WRF/CM1 standard).
#
# This is the post-merge follow-up to the test_bw_cfl07_compare.jl 4-strategy
# test (which used latitude=(-80, 80) at 1° resolution and Δt = 225 s, the
# corresponding CFL=0.7 time step). Here we use the canonical 2° resolution
# (matching examples/baroclinic_wave.jl) and let Δt = 450 s.

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

# 1° resolution at latitude=(-80, 80). Nλ=360 / Nφ=160 keeps Δλ = Δφ = 1°.
Nλ, Nφ, Nz = 360, 160, 30
H = 30kilometers

arch = CUDA.functional() ? GPU() : CPU()
grid = LatitudeLongitudeGrid(arch; size=(Nλ, Nφ, Nz), halo=(5, 5, 5),
                             longitude=(0, 360), latitude=(-80, 80), z=(0, H))

Δx_min = a * cosd(80.0) * 2π / Nλ
U_BCI_peak = 60.0  # max|u| once the BCI is fully developed
Δt = round(0.7 * Δx_min / U_BCI_peak)
@printf "Δx_min at lat 80° (1° grid) = %.2f km\n" (Δx_min/1000)
@printf "Δt for CFL=0.7 at U=%.0f m/s: %.0f s\n" U_BCI_peak Δt
@printf "(That's %.0f× larger than the canonical explicit Δt=2 s)\n" (Δt/2)

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

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)
td = SplitExplicitTimeDiscretization()
@printf "\nDefault damping: %s(coefficient=%.2f)\n" typeof(td.damping).name.name td.damping.coefficient

dynamics = CompressibleDynamics(td;
                                surface_pressure=p₀,
                                reference_potential_temperature=θ_ref)
model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants=constants, advection=WENO(),
                        timestepper=:AcousticRungeKutta3)
set!(model; θ=θ_ic, u=u_ic, ρ=ρ_ic)

n_days = 14
n_steps = round(Int, n_days * 86400 / Δt)
@printf "Grid: %d×%d×%d, latitude=(-80, 80)\n" Nλ Nφ Nz
@printf "Δt = %.0fs, %d outer steps to day %d\n" Δt n_steps n_days
@printf "Initial: max|u|=%.2f m/s, max|w|=%.2e\n\n" maximum(abs, interior(model.velocities.u)) maximum(abs, interior(model.velocities.w))

ts_t = Float64[]; ts_w = Float64[]; ts_u = Float64[]; ts_v = Float64[]; ts_psm = Float64[]

function bottom_pressure_min(model)
    p = interior(model.dynamics.pressure, :, :, 1)
    return minimum(p)
end

t0 = time()
crashed_at = 0
for step in 1:n_steps
    try
        time_step!(model, Δt)
    catch err
        crashed_at = step
        @printf "EXCEPTION at outer step %d (t=%.2f days): %s\n" step (model.clock.time/86400) first(sprint(showerror, err), 200)
        break
    end
    if any(isnan, parent(model.dynamics.density))
        crashed_at = step
        @printf "NaN at outer step %d (t=%.2f days)\n" step (model.clock.time/86400)
        break
    end

    push!(ts_t, model.clock.time / 86400)
    push!(ts_w, maximum(abs, interior(model.velocities.w)))
    push!(ts_u, maximum(abs, interior(model.velocities.u)))
    push!(ts_v, maximum(abs, interior(model.velocities.v)))
    push!(ts_psm, bottom_pressure_min(model))

    if step % round(Int, 86400/Δt) == 0 || step == n_steps
        elapsed = time() - t0
        day = model.clock.time / 86400
        @printf "day %5.2f  max|w|=%.3e  max|u|=%5.1f  max|v|=%5.2f  min(p_bot)=%.1f hPa  [wall %.0fs]\n" day ts_w[end] ts_u[end] ts_v[end] (ts_psm[end]/100) elapsed
        flush(stdout)
    end
end

wall = time() - t0
@printf "\n=== DONE ===\n"
@printf "wall: %.0f s\n" wall
if crashed_at == 0
    @printf "Reached day %d cleanly\n" n_days
    @printf "max|u| over run: %.2f m/s\n" maximum(ts_u)
    @printf "max|w| over run: %.4f m/s\n" maximum(ts_w)
    @printf "min(p_bot) over run: %.1f hPa\n" (minimum(ts_psm)/100)
else
    @printf "Crashed at step %d (day %.2f)\n" crashed_at (length(ts_t) > 0 ? ts_t[end] : 0.0)
    @printf "max|w| reached: %.2f m/s\n" (length(ts_w) > 0 ? maximum(ts_w) : NaN)
end

jldsave("bw_substepping_cfl07.jld2";
        Nlambda=Nλ, Nphi=Nφ, Nz=Nz, dt=Δt, n_days=n_days,
        ts_t, ts_w, ts_u, ts_v, ts_psm, crashed_at, wall_seconds=wall)
@printf "Saved bw_substepping_cfl07.jld2\n"
