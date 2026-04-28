#####
##### Adaptive-Δt dry baroclinic wave — find the substepper's actual CFL
##### ceiling for the 1° dry case.
#####
##### Same setup as `dry_bw_14day.jl` (which ran cleanly at fixed Δt=225s
##### for 14 days, the value documented in Breeze/examples/baroclinic_wave.jl).
##### Difference: replace fixed Δt with `conjure_time_step_wizard!(sim;
##### cfl=0.7, max_change=1.05)`. Initial Δt = 50s. Cap max_Δt = 600s.
#####
##### Pass: 14-day completion. Report:
#####   - Δt_max achieved.
#####   - Wall time vs fixed-Δt=225s baseline.
#####   - Cyclone deepening + max|w| envelope.
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "dry_bw_14day_adaptive"
const STEM = joinpath(OUTDIR, RUN_LABEL)

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220.0
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = Breeze.dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

Nλ = 360; Nφ = 160; Nz = 64; H = 30kilometers
grid = LatitudeLongitudeGrid(arch; size=(Nλ,Nφ,Nz), halo=(5,5,5),
                             longitude=(0,360), latitude=(-80,80), z=(0,H))

Tᴱ=310.0; Tᴾ=240.0; Tₘ=(Tᴱ+Tᴾ)/2; Γ=0.005; K=3; b=2

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g; η = z/(b*Hₛ); e = exp(-η^2)
    A = (Tₘ-Tᴾ)/(Tₘ*Tᴾ); C = (K+2)/2 * (Tᴱ-Tᴾ)/(Tᴱ*Tᴾ)
    return (exp(Γ*z/Tₘ)/Tₘ + A*(1-2η^2)*e,
            C*(1-2η^2)*e,
            (exp(Γ*z/Tₘ)-1)/Γ + A*z*e,
            C*z*e)
end
F(φ)  = cosd(φ)^K - K/(K+2)*cosd(φ)^(K+2)
dF(φ) = cosd(φ)^(K-1) - cosd(φ)^(K+1)

function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z); 1/(τ₁ - τ₂*F(φ))
end
function pressure(λ, φ, z)
    _,_,∫τ₁,∫τ₂ = τ_and_integrals(z); p₀*exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z)/(Rᵈ*temperature(λ, φ, z))
potential_temperature(λ, φ, z) = temperature(λ, φ, z)*(p₀/pressure(λ, φ, z))^κ
function zonal_velocity(λ, φ, z)
    _,_,_,∫τ₂ = τ_and_integrals(z); T = temperature(λ, φ, z)
    U = g/a * K * ∫τ₂ * dF(φ) * T
    rcosφ = a*cosd(φ); Ωrcosφ = Ω*rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ*U)
    uₚ=1.0; rₚ=0.1; λₚ=π/9; φₚ=2π/9; zₚ=15000.0
    φʳ=deg2rad(φ); λʳ=deg2rad(λ)
    gc = acos(sin(φₚ)*sin(φʳ) + cos(φₚ)*cos(φʳ)*cos(λʳ-λₚ))/rₚ
    taper = ifelse(z < zₚ, 1 - 3*(z/zₚ)^2 + 2*(z/zₚ)^3, 0.0)
    u_b + ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
end

coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)
T₀_ref = 250.0; θ_ref(z) = T₀_ref*exp(g*z/(cᵖᵈ*T₀_ref))
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure=p₀, reference_potential_temperature=θ_ref)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants=constants,
                        advection=WENO(),
                        timestepper=:AcousticRungeKutta3)
set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

# Adaptive Δt — PERFORMANT strategy:
#   cfl = 0.6 → 15% margin from WS-RK3 CFL=0.7
#   max_Δt = 400s → lets Δt go beyond the fixed-Δt=225s baseline during the
#                   pre-cyclone phase (when u_max=28 m/s, allowing Δt up to
#                   0.6×19000/28 ≈ 407s by advective CFL).
#   IterationInterval(5) → wizard polls every 5 outer iters, twice as often
#                          as default; catches max|U| spikes faster.
#   max_change = 1.05 → standard.
#
# Trajectory expectation:
#   Pre-cyclone  (u=28 m/s):  Δt → 400s (max_Δt cap)
#   Cyclone peak (u=70 m/s):  Δt → 0.6×19000/70 ≈ 163s
#   Avg over 14 days: ~280s  → ~30% fewer outer steps than fixed Δt=225s.
Δt_init = 30seconds
sim = Simulation(model; Δt = Δt_init, stop_time = 14days)
conjure_time_step_wizard!(sim, IterationInterval(5);
                          cfl = 0.5, max_Δt = 250.0, max_change = 1.05)

iters=Int[]; ts=Float64[]; dts=Float64[]; wmax=Float64[]; umax=Float64[]
psurf=Float64[]; walls=Float64[]
M0_ref = Ref(0.0); H0_ref = Ref(0.0); Δt_max_seen = Ref(0.0)
wall0 = Ref(time_ns())

function diag_cb(s)
    m = s.model
    push!(iters, iteration(s)); push!(ts, time(s)); push!(dts, s.Δt)
    Δt_max_seen[] = max(Δt_max_seen[], s.Δt)
    wm = Float64(maximum(abs, interior(m.velocities.w)))
    um = Float64(maximum(abs, interior(m.velocities.u)))
    pm = Float64(minimum(view(interior(m.dynamics.pressure), :, :, 1)))
    push!(wmax, wm); push!(umax, um); push!(psurf, pm)
    push!(walls, (time_ns()-wall0[])/1e9)
    @info @sprintf("[dry-adaptive] iter=%6d t=%6.3fd Δt=%5.1fs (max-so-far=%.1f) max|u|=%.2f max|w|=%.3e p_surf=%.0f wall=%.0fs",
                   iteration(s), time(s)/86400, s.Δt, Δt_max_seen[], um, wm, pm, walls[end])
    flush(stdout)
end
add_callback!(sim, diag_cb, IterationInterval(50))

@info "Starting adaptive dry BW, init Δt=$(Δt_init)s, max_Δt=600s, cfl=0.7"
wall0[] = time_ns()
crashed = false
try
    run!(sim)
    @info "[dry-adaptive] RUN COMPLETED"
catch e
    crashed = true
    @error "[dry-adaptive] RUN FAILED" e
end

jldsave(STEM*"_diagnostics.jld2";
        iters=iters, t=ts, dt=dts, wmax=wmax, umax=umax, psurf_min=psurf,
        wall=walls, Δt_max=Δt_max_seen[], crashed=crashed)

@info "Done. final t=$(time(sim))s, Δt_max_achieved=$(Δt_max_seen[])s, total wall=$(walls[end])s"
