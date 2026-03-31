# Baroclinic wave — vertically implicit time stepping test
#
# Copy of baroclinic_wave.jl using VerticallyImplicitTimeStepping instead of
# ExplicitTimeStepping. Reduced resolution and shorter run for debugging.
# The implicit treatment of vertical acoustics should allow ~30x larger Δt.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf

# DCMIP2016 parameters

Oceananigans.defaults.FloatType = Float64
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

# Reduced resolution for testing
Nλ = 90
Nφ = 42
Nz = 15
H  = 30kilometers

grid = LatitudeLongitudeGrid(CPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-85, 85),
                             z = (0, H))

# DCMIP2016 initial condition parameters

Tᴱ = 310.0
Tᴾ = 240.0
Tₘ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005
K  = 3
b  = 2

function τ_and_integrals(z)
    Hₛ = Rᵈ * Tₘ / g
    η  = z / (b * Hₛ)
    e  = exp(-η^2)

    A = (Tₘ - Tᴾ) / (Tₘ * Tᴾ)
    C = (K + 2) / 2 * (Tᴱ - Tᴾ) / (Tᴱ * Tᴾ)

    τ₁  = exp(Γ * z / Tₘ) / (Γ * Tₘ) + A * (1 - 2η^2) * e
    τ₂  = C * (1 - 2η^2) * e
    ∫τ₁ = (exp(Γ * z / Tₘ) - 1) / Γ + A * z * e
    ∫τ₂ = C * z * e

    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

function temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * temperature(λ, φ, z))

function potential_temperature(λ, φ, z)
    T = temperature(λ, φ, z)
    p = pressure(λ, φ, z)
    return T * (p₀ / p)^κ
end

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    T = temperature(λ, φ, z)

    U = g / a * K * ∫τ₂ * dF(φ) * T
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    u_balanced = -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)

    uₚ = 1.0
    rₚ = 0.1
    λₚ = π / 9
    φₚ = 2π / 9
    zₚ = 15000.0

    φʳ = deg2rad(φ)
    λʳ = deg2rad(λ)
    great_circle = acos(sin(φₚ) * sin(φʳ) + cos(φₚ) * cos(φʳ) * cos(λʳ - λₚ)) / rₚ

    taper = ifelse(z < zₚ, 1 - 3 * (z / zₚ)^2 + 2 * (z / zₚ)^3, 0.0)
    u_perturbation = ifelse(great_circle < 1, uₚ * taper * exp(-great_circle^2), 0.0)

    return u_balanced + u_perturbation
end

# Model with VerticallyImplicitTimeStepping

coriolis = HydrostaticSphericalCoriolis()

dynamics = CompressibleDynamics(VerticallyImplicitTimeStepping();
                                surface_pressure = p₀)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO())

set!(model, θ=potential_temperature, u=zonal_velocity, ρ=density)

# Time-stepping: VITS removes vertical acoustic CFL; start with Δt=10s
Δt = 10
stop_time = 1days

simulation = Simulation(model; Δt, stop_time)

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

run!(simulation)

@info "Simulation complete."
