#####
##### F2 — Dry baroclinic wave, 14-day end-to-end validation
#####
##### DCMIP-2016 Test 4-1 (Jablonowski-Williamson 2006).
##### Parameters from `Breeze/examples/baroclinic_wave.jl` —
##### isothermal-T₀=250 K reference state, 1° lat-lon (Nλ=360, Nφ=160,
##### Nz=64), Δt=225 s, latitude=(-80, 80), Lz=30 km.
#####
##### Tracks: max|w|(t), max|u|(t), surface pressure minimum, mass
##### conservation, ρθ conservation. Saves snapshots every 6 hours.
#####
##### Pass criteria (from SUBSTEPPER_TEST_PLAN.md F2):
##### (a) Run completes 14 days without NaN.
##### (b) Surface-pressure minimum at day 9 within ±5 hPa of JW06 (~940 hPa).
##### (c) Mass drift over 14 days ≤ 1e-10.
##### (d) max|w| stays bounded (< 5 m/s for healthy BCI).
#####

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using JLD2
using Statistics

const arch = CUDA.functional() ? GPU() : CPU()
const OUTDIR = @__DIR__
const RUN_LABEL = "dry_bw_14day"
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

Nλ = 360
Nφ = 160
Nz = 64
H  = 30kilometers

grid = LatitudeLongitudeGrid(arch;
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude  = (-80, 80),
                             z         = (0, H))

# DCMIP-2016 / JW06 analytic balanced jet
Tᴱ = 310.0; Tᴾ = 240.0; Tₘ = (Tᴱ + Tᴾ)/2; Γ = 0.005; K = 3; b = 2

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g
    η  = z / (b * Hₛ); e = exp(-η^2)
    A = (Tₘ - Tᴾ)/(Tₘ * Tᴾ); C = (K + 2)/2 * (Tᴱ - Tᴾ)/(Tᴱ * Tᴾ)
    τ₁  = exp(Γ * z / Tₘ)/Tₘ + A*(1 - 2η^2)*e
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ*z/Tₘ) - 1)/Γ + A*z*e
    ∫τ₂ = C*z*e
    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K/(K+2)*cosd(φ)^(K+2)
dF(φ) = cosd(φ)^(K-1) - cosd(φ)^(K+1)

function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1/(τ₁ - τ₂*F(φ))
end
function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g/Rᵈ*(∫τ₁ - ∫τ₂*F(φ)))
end
density(λ, φ, z) = pressure(λ, φ, z)/(Rᵈ * temperature(λ, φ, z))
potential_temperature(λ, φ, z) = temperature(λ, φ, z) * (p₀/pressure(λ, φ, z))^κ

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = temperature(λ, φ, z)
    U = g/a * K * ∫τ₂ * dF(φ) * T
    rcosφ = a*cosd(φ); Ωrcosφ = Ω*rcosφ
    u_b = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ*U)
    uₚ=1.0; rₚ=0.1; λₚ=π/9; φₚ=2π/9; zₚ=15000.0
    φʳ=deg2rad(φ); λʳ=deg2rad(λ)
    gc = acos(sin(φₚ)*sin(φʳ) + cos(φₚ)*cos(φʳ)*cos(λʳ-λₚ))/rₚ
    taper = ifelse(z < zₚ, 1 - 3*(z/zₚ)^2 + 2*(z/zₚ)^3, 0.0)
    u_p = ifelse(gc < 1, uₚ * taper * exp(-gc^2), 0.0)
    return u_b + u_p
end

coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)
T₀_ref   = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θ_ref)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO(),
                        timestepper = :AcousticRungeKutta3)

set!(model, θ = potential_temperature, u = zonal_velocity, ρ = density)

Δt = 225seconds
stop_time = 14days

simulation = Simulation(model; Δt, stop_time)

# Diagnostics arrays
diag_iters = Int[]
diag_t     = Float64[]
diag_wmax  = Float64[]
diag_umax  = Float64[]
diag_ρmin  = Float64[]
diag_ρmax  = Float64[]
diag_psurf_min = Float64[]
diag_total_mass = Float64[]
diag_total_ρθ   = Float64[]
diag_wall      = Float64[]
wall_start = Ref(time_ns())

# Reference initial diagnostics for conservation
ρ_field0 = Breeze.AtmosphereModels.dynamics_density(model.dynamics)
M0 = Float64(sum(interior(ρ_field0)))
ρθ_field0 = model.formulation.potential_temperature_density
H0 = Float64(sum(interior(ρθ_field0)))

function diag_cb(sim)
    m = sim.model
    u, v, w = m.velocities
    p = m.dynamics.pressure
    ρ = Breeze.AtmosphereModels.dynamics_density(m.dynamics)
    ρθ = m.formulation.potential_temperature_density

    wmax = Float64(maximum(abs, interior(w)))
    umax = Float64(maximum(abs, interior(u)))
    ρmin = Float64(minimum(interior(ρ)))
    ρmax = Float64(maximum(interior(ρ)))
    p_surf = view(interior(p), :, :, 1)
    psurf_min = Float64(minimum(p_surf))
    M = Float64(sum(interior(ρ)))
    H = Float64(sum(interior(ρθ)))

    push!(diag_iters, iteration(sim))
    push!(diag_t, time(sim))
    push!(diag_wmax, wmax)
    push!(diag_umax, umax)
    push!(diag_ρmin, ρmin)
    push!(diag_ρmax, ρmax)
    push!(diag_psurf_min, psurf_min)
    push!(diag_total_mass, M)
    push!(diag_total_ρθ, H)
    push!(diag_wall, (time_ns() - wall_start[]) / 1e9)

    @info @sprintf("[dry] iter=%5d  t=%5.2fd  Δt=%.0fs  max|u|=%.2f  max|w|=%.2e  p_surf_min=%.0f Pa  ΔM/M0=%.2e  ΔH/H0=%.2e  wall=%.0fs",
                   iteration(sim), time(sim)/86400, sim.Δt, umax, wmax, psurf_min,
                   (M-M0)/M0, (H-H0)/H0, diag_wall[end])
    flush(stdout); flush(stderr)

    if isnan(wmax)
        @error "[dry] NaN in w at iter $(iteration(sim))"
        return false
    end
    return nothing
end

add_callback!(simulation, diag_cb, IterationInterval(50))

# Periodic JLD2 snapshot of full state at 12-hour intervals
sim_outputs = merge(model.velocities,
                    (; ρθ = model.formulation.potential_temperature_density,
                       p  = model.dynamics.pressure))
simulation.output_writers[:state] = JLD2Writer(model, sim_outputs;
    filename = STEM * "_state.jld2",
    schedule = TimeInterval(12hours),
    overwrite_existing = true)

@info "Starting dry-BW 14-day run, Δt=$(Δt)s, $(stop_time/86400) days, $(Int(stop_time/Δt)) outer steps"
wall_start[] = time_ns()

try
    run!(simulation)
    @info "[dry] RUN COMPLETED"
catch e
    @error "[dry] RUN FAILED" e
end

jldsave(STEM * "_diagnostics.jld2";
        iters = diag_iters, t = diag_t, wmax = diag_wmax, umax = diag_umax,
        ρmin = diag_ρmin, ρmax = diag_ρmax, psurf_min = diag_psurf_min,
        total_mass = diag_total_mass, total_ρθ = diag_total_ρθ, wall = diag_wall,
        M0 = M0, H0 = H0)

@info "Dry-BW 14-day run complete. Diagnostics: $(STEM)_diagnostics.jld2"
