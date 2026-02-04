# # Acoustic wave refraction by wind shear (2D horizontal)
#
# This example simulates an acoustic pulse propagating through a horizontal wind shear layer
# using the fully compressible [Euler equations](https://en.wikipedia.org/wiki/Euler_equations_(fluid_dynamics)).
# When wind speed varies across the domain, sound waves are refracted: waves traveling **with**
# the wind bend toward regions of lower wind speed, while waves traveling **against**
# the wind bend toward regions of higher wind speed.
#
# The sound speed for a wave traveling in direction ``\hat{\boldsymbol{n}}`` is
# ```math
# 𝕌ˢ = 𝕌ˢⁱ + \boldsymbol{u} \cdot \hat{\boldsymbol{n}}
# ```
# where ``𝕌ˢⁱ`` is the intrinsic sound speed and ``\boldsymbol{u}`` is the wind velocity.
# This causes wavefronts to tilt toward regions of lower effective sound speed.
#
# This is a 2D horizontal slice simulation with doubly-periodic boundary conditions.

using Breeze
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid and model setup

Nx, Ny = 256, 128
Lx, Ly = 1000.0, 200.0  # meters

grid = RectilinearGrid(size = (Nx, Ny), extent = (Lx, Ly),
                       topology = (Periodic, Periodic, Flat))

model = AtmosphereModel(grid; dynamics = CompressibleDynamics())

# ## Background state
#
# For a 2D horizontal slice, we use uniform thermodynamic properties.
# The reference density is computed from the ideal gas law at the given pressure and temperature.

constants = model.thermodynamic_constants

θ₀ = 300.0      # Reference potential temperature (K)
p₀ = 101325.0   # Surface pressure (Pa)

# Compute thermodynamic constants
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)

# Reference density from ideal gas law: ρ = p / (R * T)
# At surface, T ≈ θ₀ (potential temperature equals temperature when p = p_ref)
ρ₀ = p₀ / (Rᵈ * θ₀)

# The sound speed determines the acoustic wave propagation speed
𝕌ˢⁱ = sqrt(γ * Rᵈ * θ₀)

# The wind profile varies linearly across the y-direction to create horizontal shear.

U₀ = 20.0 # Maximum velocity (m/s)

# Wind increases from U₀/2 at y=0 to 3U₀/2 at y=Ly
Uᵢ(y) = U₀ * (0.5 + y / Ly)

# ## Initial conditions
#
# We initialize a localized Gaussian density pulse representing an acoustic disturbance.
# No velocity perturbation - only the background wind shear.

δρ = 0.001        # Density perturbation amplitude (kg/m³) - small for linear acoustics
σ = 50.0          # Pulse width (m)
x₀ = Lx / 2       # Gaussian center x-position (domain center)
y₀ = Ly / 2       # Gaussian center y-position (domain center)

# Gaussian density perturbation, no velocity perturbation
set!(model, 
     ρ = (x, y) -> ρ₀ + δρ * exp(-((x - x₀)^2 + (y - y₀)^2) / (2σ^2)),
     θ = θ₀, 
     u = (x, y) -> Uᵢ(y))


# ## Simulation setup
#
# Acoustic waves travel fast (``𝕌ˢⁱ ≈ 347`` m/s), so we need a small time step.
# The [Courant–Friedrichs–Lewy (CFL) condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition) is based on the effective sound speed ``𝕌ˢ = 𝕌ˢⁱ + \mathrm{max}(U)``.

Δx, Δy = Lx / Nx, Ly / Ny
𝕌ˢ = 𝕌ˢⁱ + U₀ * 1.5  # max wind speed
Δt = 0.5 * min(Δx, Δy) / 𝕌ˢ
nsteps = 36*36 # number of time steps

simulation = Simulation(model; Δt, stop_iteration = nsteps)

progress(sim) = @info @sprintf("Iter: %d, t: %s, max|u|: %.2f m/s, max|v|: %.2f m/s",
                               iteration(sim), prettytime(sim),
                               maximum(abs, sim.model.velocities.u), 
                               maximum(abs, sim.model.velocities.v))

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# We perturbation fields for density and x-velocity for visualization.

ρ = model.dynamics.density
u, v, w = model.velocities

ρᵇᵍ = CenterField(grid)
uᵇᵍ = XFaceField(grid)

set!(ρᵇᵍ, (x, y) -> ρ₀)
set!(uᵇᵍ, (x, y) -> Uᵢ(y))

ρ′ = Field(ρ - ρᵇᵍ)
u′ = Field(u - uᵇᵍ)

U = Average(u, dims = 1)
R = Average(ρ, dims = 1)
V² = Average(v^2, dims = 1)

filename = "acoustic_wave.jld2"
outputs = (; ρ′, u′, v, U, R, V²)

simulation.output_writers[:jld2] = JLD2Writer(model, outputs; filename,
                                              including = [:grid],
                                              schedule = IterationInterval(1),
                                              overwrite_existing = true)

run!(simulation)

# ## Visualization
#
# Load the saved perturbation fields and create a snapshot.

ρ′ts = FieldTimeSeries(filename, "ρ′")
u′ts = FieldTimeSeries(filename, "u′")
vts = FieldTimeSeries(filename, "v")
Uts = FieldTimeSeries(filename, "U")
Rts = FieldTimeSeries(filename, "R")
V²ts = FieldTimeSeries(filename, "V²")

times = ρ′ts.times
Nt = length(times)

fig = Figure(size = (900, 600), fontsize = 12)

aspect_ratio = Lx / Ly
axρ = Axis(fig[1, 2]; aspect = aspect_ratio, ylabel = "y (m)")
axv = Axis(fig[2, 2]; aspect = aspect_ratio, ylabel = "y (m)")
axu = Axis(fig[3, 2]; aspect = aspect_ratio, xlabel = "x (m)", ylabel = "y (m)")
axR = Axis(fig[1, 1]; xlabel = "⟨ρ⟩ (kg/m³)")
axV = Axis(fig[2, 1]; xlabel = "⟨v²⟩ (m²/s²)", limits = (extrema(V²ts), nothing))
axU = Axis(fig[3, 1]; xlabel = "⟨u⟩ (m/s)")

hidexdecorations!(axρ)
hidexdecorations!(axv)
colsize!(fig.layout, 1, Relative(0.2))

n = Observable(Nt)
ρ′n = @lift ρ′ts[$n]
u′n = @lift u′ts[$n]
vn = @lift vts[$n]
Un = @lift Uts[$n]
Rn = @lift Rts[$n]
V²n = @lift V²ts[$n]

ρlim = δρ / 2  # Colorrange based on density perturbation amplitude
ulim = 0.1     # Small colorrange for velocity (no initial perturbation)

hmρ = heatmap!(axρ, ρ′n; colormap = :balance, colorrange = (-ρlim, ρlim))
hmv = heatmap!(axv, vn; colormap = :balance, colorrange = (-ulim, ulim))
hmu = heatmap!(axu, u′n; colormap = :balance, colorrange = (-ulim, ulim))

lines!(axR, Rn)
lines!(axV, V²n)
lines!(axU, Un)

Colorbar(fig[1, 3], hmρ; label = "ρ′ (kg/m³)")
Colorbar(fig[2, 3], hmv; label = "v (m/s)")
Colorbar(fig[3, 3], hmu; label = "u′ (m/s)")

title = @lift "Acoustic wave in horizontal shear — t = $(prettytime(times[$n])), nsteps=$nsteps"
fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

output_filename = "acoustic_wave_nsteps$(nsteps).mp4"
CairoMakie.record(fig, output_filename, 1:Nt, framerate = 18) do nn
    n[] = nn
end
nothing #hide

# ![](acoustic_wave.mp4)
