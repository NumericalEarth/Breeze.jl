# # Acoustic Wave in Compressible Dynamics
#
# This example demonstrates the propagation of an acoustic wave using
# the fully compressible Euler equations. Acoustic waves are pressure/density
# perturbations that propagate at the speed of sound.
#
# The compressible dynamics directly time-steps density as a prognostic variable
# and computes pressure from the Poisson equation for potential temperature.
#
# ## Physics
#
# In a compressible atmosphere, small perturbations in pressure and density
# propagate as acoustic waves with speed:
#
# ```math
# c_s = \sqrt{\gamma R^d T}
# ```
#
# where `γ ≈ 1.4` is the heat capacity ratio for dry air.
# For `T = 300 K`, the speed of sound is approximately 347 m/s.

using Oceananigans
using Breeze
using CairoMakie

# ## Model setup
#
# We use a quasi-1D domain to simulate acoustic wave propagation.

Nx = 64
Nz = 4
Lx = 1000  # m
Lz = 100   # m

grid = RectilinearGrid(CPU(),
                       size = (Nx, 1, Nz),
                       x = (0, Lx),
                       y = (0, 10),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

# ## Compressible Dynamics
#
# We create a `CompressibleDynamics` model which time-steps density
# directly and computes pressure from the Poisson equation.

dynamics = CompressibleDynamics()
model = AtmosphereModel(grid; dynamics)

println("Model created: ", typeof(model.dynamics))

# ## Initial conditions
#
# We initialize with a uniform background state plus a small
# Gaussian density perturbation.

T₀ = 300.0   # Background temperature (K)
ρ₀ = 1.2     # Background density (kg/m³)
δρ = 0.001   # Small density perturbation (kg/m³)
x₀ = Lx / 2  # Center of perturbation
σ = 30.0     # Width of Gaussian perturbation (m)

# For dry air at T = 300 K, the potential temperature is approximately equal
# to the temperature (since p ≈ p₀ at the surface). 
# θ ≈ T for small pressure deviations from p₀.
θ₀ = T₀

# Gaussian density perturbation (uniform in y and z)
ρᵢ(x, y, z) = ρ₀ + δρ * exp(-(x - x₀)^2 / (2σ^2))

# Set prognostic fields
set!(model, ρ=ρᵢ, θ=θ₀)
ρᵢ_field = deepcopy(model.dynamics.density)

println("Initial density range: ", extrema(model.dynamics.density))

# ## Time stepping
#
# For acoustic wave propagation, we need a time step that satisfies
# the acoustic CFL condition: `Δt < Δx / c_s`

Δx = Lx / Nx
c_s = 347.0  # Speed of sound (approximate)
Δt = 0.05 * Δx / c_s  # CFL = 0.05 for stability

simulation = Simulation(model, Δt=Δt, stop_iteration=1000)

println("Time step: ", Δt, " s")
println("Acoustic CFL: ", Δt * c_s / Δx)

# Run a few time steps
run!(simulation)

println("Simulation completed after ", simulation.model.clock.iteration, " iterations!")
println("Final time: ", prettytime(simulation.model.clock.time))

# Check for NaNs
if any(isnan, interior(model.dynamics.density))
    println("WARNING: NaN detected in density field!")
else
    println("Density field OK, range: ", extrema(model.dynamics.density))
end

if any(isnan, interior(model.momentum.ρu))
    println("WARNING: NaN detected in ρu field!")
else
    println("ρu field OK, range: ", extrema(model.momentum.ρu))
end

println("Pressure range: ", extrema(model.dynamics.pressure))

# ## Visualization
#
# Let's visualize the density perturbation.

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1],
          xlabel="x (m)",
          ylabel="Density (kg/m³)",
          title="Acoustic wave after $(prettytime(model.clock.time))")

lines!(ax, view(model.dynamics.density, :, 1, 1), color=:dodgerblue, linewidth=2, label="Final")

# Initial condition for comparison
lines!(ax, view(ρᵢ_field, :, 1, 1), color=:gray, linewidth=1, linestyle=:dash, label="Initial")

axislegend(ax)

save("acoustic_wave.png", fig)
println("Figure saved to acoustic_wave.png")
