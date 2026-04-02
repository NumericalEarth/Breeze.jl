# IGW comparison: Anelastic vs Compressible (explicit) vs IMEX SSP3332
#
# Skamarock & Klemp (1994) inertia-gravity wave test case.
# Compares three formulations at t = 3000s to verify the IMEX-ARK
# time stepper correctly captures gravity wave propagation.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

## Problem parameters
p₀ = 100000
θ₀ = 300
U  = 20
N² = 0.01^2

Nx, Nz = 300, 10
Lx, Lz = 300kilometers, 10kilometers

grid = RectilinearGrid(CPU(), size=(Nx, Nz), halo=(5, 5),
                       x=(0, Lx), z=(0, Lz),
                       topology=(Periodic, Flat, Bounded))

Δθ = 0.01
a  = 5000
x₀ = Lx / 3

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration
pˢᵗ = 1e5

θᵇᵍ(z) = θ₀ * exp(N² * z / g)
θᵢ(x, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

advection = WENO()
surface_pressure = p₀

## CFL parameters
Δx, Δz_val = Lx / Nx, Lz / Nz
Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
ℂᵃᶜ = sqrt(cᵖᵈ / (cᵖᵈ - Rᵈ) * Rᵈ * θ₀)
Δt_advective = 0.5 * min(Δx, Δz_val) / U
Δt_compressible = 0.5 * min(Δx, Δz_val) / (ℂᵃᶜ + U)

@printf("Δx = %.0fm, Δz = %.0fm, ℂ = %.0f m/s\n", Δx, Δz_val, ℂᵃᶜ)
@printf("Δt_advective = %.1fs, Δt_compressible = %.2fs\n", Δt_advective, Δt_compressible)

stop_time = 3000.0

## Background θ for perturbation diagnostic
θᵇᵍ_field = CenterField(grid)
set!(θᵇᵍ_field, (x, z) -> θᵇᵍ(z))

## === Case 1: Anelastic (reference) ===
println("\n=== Anelastic ===")
reference_state = ReferenceState(grid, constants; surface_pressure, potential_temperature=θ₀)
model_anel = AtmosphereModel(grid; advection, dynamics=AnelasticDynamics(reference_state))
set!(model_anel; θ=θᵢ, u=U)
sim_anel = Simulation(model_anel; Δt=Δt_advective, stop_time, verbose=false)
run!(sim_anel)
θ′_op = PotentialTemperature(model_anel) - θᵇᵍ_field
θ′_f = compute!(Field(θ′_op))
θ′_anel = interior(θ′_f, :, 1, :) |> Array
w_anel = interior(model_anel.velocities.w, :, 1, :) |> Array
@printf("  Done. max|θ′| = %.4e, max|w| = %.4f\n", maximum(abs, θ′_anel), maximum(abs, w_anel))

## === Case 2: Compressible explicit ===
@printf("\n=== Compressible explicit (Δt=%.2fs) ===\n", Δt_compressible)
dynamics_exp = CompressibleDynamics(ExplicitTimeStepping();
                                     surface_pressure,
                                     reference_potential_temperature=θᵇᵍ)
model_exp = AtmosphereModel(grid; advection, dynamics=dynamics_exp)
ref_exp = model_exp.dynamics.reference_state
set!(model_exp; θ=θᵢ, u=U, qᵗ=0, ρ=ref_exp.density)
sim_exp = Simulation(model_exp; Δt=Δt_compressible, stop_time, verbose=false)
run!(sim_exp)
θ′_f = compute!(Field(PotentialTemperature(model_exp) - θᵇᵍ_field))
θ′_exp = interior(θ′_f, :, 1, :) |> Array
w_exp = interior(model_exp.velocities.w, :, 1, :) |> Array
@printf("  Done. max|θ′| = %.4e, max|w| = %.4f\n", maximum(abs, θ′_exp), maximum(abs, w_exp))

## === Case 3: IMEX SSP3332 at compressible Δt ===
@printf("\n=== IMEX SSP3332 (Δt=%.2fs) ===\n", Δt_compressible)
dynamics_ssp = CompressibleDynamics(VerticallyImplicitTimeStepping();
                                     surface_pressure,
                                     reference_potential_temperature=θᵇᵍ)
model_ssp = AtmosphereModel(grid; advection, dynamics=dynamics_ssp,
                             timestepper=:IMEXRungeKuttaSSP3332)
ref_ssp = model_ssp.dynamics.reference_state
set!(model_ssp; θ=θᵢ, u=U, qᵗ=0, ρ=ref_ssp.density)
sim_ssp = Simulation(model_ssp; Δt=Δt_compressible, stop_time, verbose=false)
run!(sim_ssp)
θ′_f = compute!(Field(PotentialTemperature(model_ssp) - θᵇᵍ_field))
θ′_ssp = interior(θ′_f, :, 1, :) |> Array
w_ssp = interior(model_ssp.velocities.w, :, 1, :) |> Array
@printf("  Done. max|θ′| = %.4e, max|w| = %.4f\n", maximum(abs, θ′_ssp), maximum(abs, w_ssp))

## === Plot ===
x_km = range(0, Lx / 1000, length=Nx)
z_km = range(0, Lz / 1000, length=Nz)
z_mid = Nz ÷ 2
levels = range(-Δθ / 2, stop=Δθ / 2, length=21)

fig = Figure(size=(1400, 900))

for (row, (data, title)) in enumerate([
    (θ′_anel, "Anelastic (Δt=$(round(Δt_advective, digits=1))s)"),
    (θ′_exp,  "Compressible explicit (Δt=$(round(Δt_compressible, digits=2))s)"),
    (θ′_ssp,  "IMEX SSP3332 (Δt=$(round(Δt_compressible, digits=2))s)")
])
    ax = Axis(fig[row, 1]; title, xlabel=row==3 ? "x (km)" : "", ylabel="z (km)")
    row < 3 && hidexdecorations!(ax, grid=false)
    cf = contourf!(ax, x_km, z_km, data; colormap=:balance, levels)
    row == 1 && Colorbar(fig[1:3, 2], cf; label="θ′ (K)")
end

ax_cross = Axis(fig[1:3, 3]; xlabel="x (km)", ylabel="θ′ (K)",
                title="θ′ at z = $(round(z_km[z_mid], digits=1)) km")
lines!(ax_cross, x_km, θ′_anel[:, z_mid]; label="Anelastic", linewidth=2)
lines!(ax_cross, x_km, θ′_exp[:, z_mid]; label="Explicit", linewidth=2, linestyle=:dash)
lines!(ax_cross, x_km, θ′_ssp[:, z_mid]; label="SSP3332", linewidth=2, linestyle=:dot)
axislegend(ax_cross, position=:rt)

save("igw_hevi_comparison.png", fig)
@info "Saved igw_hevi_comparison.png"
