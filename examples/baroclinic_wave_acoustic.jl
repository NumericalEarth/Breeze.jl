# Baroclinic wave: Acoustic substepping (MPAS-style) test
#
# Uses AcousticRungeKutta3 with SplitExplicitTimeDiscretization.
# Isothermal T₀=250K reference state (matching MPAS convention).

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA

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

Nλ = 90; Nφ = 42; Nz = 15
H  = 30kilometers

arch = CUDA.functional() ? GPU() : CPU()
grid = LatitudeLongitudeGrid(arch; size=(Nλ, Nφ, Nz), halo=(5, 5, 5),
                             longitude=(0, 360), latitude=(-85, 85), z=(0, H))

## DCMIP2016 ICs (same as baroclinic_wave.jl)
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

coriolis = HydrostaticSphericalCoriolis(rotation_rate=Ω)

## MPAS isothermal T₀=250K reference state: θ₀(z) = T₀ exp(gz/(cₚT₀))
T₀_ref = 250.0
θ_ref(z) = T₀_ref * exp(g * z / (cᵖᵈ * T₀_ref))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure=p₀,
                                reference_potential_temperature=θ_ref)

## Time step limited by outer-loop gravity-wave CFL, not acoustic CFL.
## Acoustic substepping removes the acoustic CFL constraint, giving ~5×
## speedup over the explicit solver (Δt=2s).
Δt = 10.0
stop_time = 1days

model = AtmosphereModel(grid; dynamics, coriolis,
                         thermodynamic_constants=constants, advection=WENO(),
                         timestepper=:AcousticRungeKutta3)
set!(model; θ=θ_ic, u=u_ic, ρ=ρ_ic)

simulation = Simulation(model; Δt, stop_time, verbose=false)
add_polar_filter!(simulation; threshold_latitude=60)

add_callback!(simulation, sim -> @info(@sprintf("t=%s, max|w|=%.3f, max|u|=%.1f",
    prettytime(sim), maximum(abs, sim.model.velocities.w),
    maximum(abs, sim.model.velocities.u))), IterationInterval(500))

run!(simulation)

@printf("\nAcoustic RK3 (Δt=%.0fs): max|w| = %.4f m/s\n",
        Δt, maximum(abs, interior(model.velocities.w)))
