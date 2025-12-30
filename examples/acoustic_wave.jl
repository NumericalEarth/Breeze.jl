# # Acoustic Wave Refraction by Wind Shear
#
# This example demonstrates how wind shear refracts acoustic waves using
# the fully compressible Euler equations. When wind speed increases with height,
# sound waves traveling with the wind are bent downward toward the surface,
# while sound traveling against the wind is bent upward and away.
#
# This phenomenon explains why sounds carry farther downwind than upwind —
# the refraction traps acoustic energy near the surface in the downwind direction.
#
# ## Physics
#
# The effective sound speed for a wave traveling in direction ``\hat{n}`` is
#
# ```math
# c_{\rm eff} = c_s + \mathbf{u} \cdot \hat{n}
# ```
#
# where ``c_s = \sqrt{\gamma R^d T}`` is the thermodynamic sound speed
# and ``\mathbf{u}`` is the wind velocity. When wind speed increases with height,
# ``c_{\rm eff}`` increases with height for downwind propagation. By Snell's law,
# wavefronts tilt and rays bend toward regions of lower effective sound speed —
# i.e., downward for downwind propagation.

using Oceananigans
using Breeze
using CairoMakie

# ## Model Setup
#
# We use a 2D domain (x-z plane) large enough to observe the refraction effect.

Nx = 256
Nz = 64
Lx = 2000  # m
Lz = 500   # m

grid = RectilinearGrid(CPU(),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## Compressible Dynamics

dynamics = CompressibleDynamics()
model = AtmosphereModel(grid; dynamics)

# ## Initial Conditions
#
# We set up a logarithmic wind profile (typical of the atmospheric surface layer)
# with wind increasing from zero at the surface. The sound source is a localized
# Gaussian pressure/density pulse near the left side of the domain.

T₀ = 300    # Background temperature (K)
ρ₀ = 1.2    # Background density (kg/m³)
θ₀ = T₀    # Potential temperature ≈ temperature for weak pressure perturbations

# Wind profile parameters
u★ = 2.0     # Friction velocity (m/s)
z₀ = 0.1     # Roughness length (m)
κ = 0.4      # von Kármán constant

# Logarithmic wind profile: u(z) = (u★/κ) * log((z + z₀) / z₀)
# This gives ~20 m/s wind at z = 500 m

# Acoustic pulse parameters
δρ = 0.01    # Density perturbation amplitude (kg/m³)
x₀ = 200     # Pulse center x-position (m)
z₀_pulse = 100  # Pulse center z-position (m)
σ = 30       # Pulse width (m)

# Initial conditions as functions
uᵢ(x, z) = (u★ / κ) * log((z + z₀) / z₀)
ρᵢ(x, z) = ρ₀ + δρ * exp(-((x - x₀)^2 + (z - z₀_pulse)^2) / (2σ^2))

set!(model, u=uᵢ, ρ=ρᵢ, θ=θ₀)

# Save initial density for comparison
ρ_initial = deepcopy(model.dynamics.density)

# ## Time Stepping
#
# For acoustic waves, we need a small time step satisfying the CFL condition
# for sound speed plus the maximum wind speed.

Δx = Lx / Nx
Δz = Lz / Nz

constants = model.thermodynamic_constants
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
cₛ = sqrt(γ * Rᵈ * T₀)

u_max = uᵢ(0, Lz)  # Maximum wind speed at domain top
c_eff_max = cₛ + u_max

Δt = 0.1 * min(Δx, Δz) / c_eff_max

# Run long enough for the wave to propagate across a significant portion of the domain
stop_time = 3.0  # seconds

simulation = Simulation(model; Δt, stop_time)

@info "Sound speed: $(round(cₛ, digits=1)) m/s"
@info "Max wind speed: $(round(u_max, digits=1)) m/s"
@info "Max effective sound speed: $(round(c_eff_max, digits=1)) m/s"
@info "Time step: $(round(Δt * 1000, digits=3)) ms"

run!(simulation)

@info "Simulation completed at t = $(prettytime(model.clock.time))"

# ## Visualization
#
# We plot the density perturbation to visualize how the acoustic wave
# has been refracted by the wind shear. The wave traveling to the right
# (with the wind) should curve downward, while the wave traveling left
# (against the wind) curves upward and away from the surface.

ρ′ = model.dynamics.density .- ρ₀  # Density perturbation

fig = Figure(size=(900, 500), fontsize=14)

ax = Axis(fig[1, 1],
          xlabel = "x (m)",
          ylabel = "z (m)",
          title = "Acoustic wave refraction by wind shear (t = $(prettytime(model.clock.time)))",
          aspect = Lx / Lz)

# Plot density perturbation
hm = heatmap!(ax, view(ρ′, :, 1, :),
              colormap = :balance,
              colorrange = (-δρ/2, δρ/2))

Colorbar(fig[1, 2], hm, label = "ρ′ (kg/m³)")

# Add annotation for wind direction
text!(ax, Lx - 200, Lz - 50, text = "wind →", fontsize = 16)

save("acoustic_wave.png", fig)
@info "Figure saved to acoustic_wave.png"
