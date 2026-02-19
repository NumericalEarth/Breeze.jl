# # Mountain waves with terrain-following coordinates
#
# This example simulates steady-state [mountain waves](https://en.wikipedia.org/wiki/Lee_wave)
# using the fully compressible equations on a terrain-following grid.
#
# A uniform wind blows over a Gaussian ridge, generating gravity waves that propagate
# vertically. With stable stratification (``N > 0``), the phase lines tilt upstream
# with height. This classic test case is well-described by [Durran (2010)](@cite Durran2010)
# and the linear theory of [Queney (1948)](@cite Queney1948).
#
# Instead of using an [`ImmersedBoundaryGrid`](@ref) with partial cells, we deform the
# computational grid itself using [`follow_terrain!`](@ref), which applies a basic
# terrain-following (Gal-Chen) coordinate transformation. The physics are then
# corrected with [`TerrainMetrics`](@ref) to account for the tilted coordinate surfaces.

using Breeze
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid setup
#
# We build a 2D grid on a periodic domain with a [`MutableVerticalDiscretization`](@ref)
# so the vertical coordinate can be deformed to follow the terrain.

Nx, Nz = 256, 64
Lx = 100kilometers
Lz = 20kilometers

z_faces = MutableVerticalDiscretization(collect(range(0, Lz, length=Nz+1)))
grid = RectilinearGrid(size = (Nx, Nz),
                       x = (-Lx/2, Lx/2), z = z_faces,
                       topology = (Periodic, Flat, Bounded))

# ## Terrain
#
# A bell-shaped mountain (Gaussian ridge) centered at the origin:

h₀ = 500meters
a = 5kilometers
h(x, y) = h₀ * exp(-x^2 / a^2)

# Apply the terrain to the grid and retrieve the metric terms.

metrics = follow_terrain!(grid, h)

# ## Model construction
#
# Build a compressible model with explicit time-stepping and terrain corrections.
# Passing `terrain_metrics` to [`CompressibleDynamics`](@ref) activates the
# terrain-following physics: contravariant vertical velocity, corrected pressure
# gradient, and terrain-aware divergence.

dynamics = CompressibleDynamics(ExplicitTimeStepping(); terrain_metrics=metrics)
model = AtmosphereModel(grid; dynamics)

# ## Initial conditions
#
# We start from hydrostatic balance with a stably stratified potential temperature
# profile ``θ(z) = θ₀ \exp(N^2 z / g)`` and a uniform horizontal wind.

constants = model.thermodynamic_constants
g = constants.gravitational_acceleration

θ₀ = 300      # Surface potential temperature (K)
p₀ = 101325   # Surface pressure (Pa)
pˢᵗ = 1e5     # Standard pressure (Pa)
N² = 1e-4     # Brunt–Väisälä frequency squared (s⁻²)

θᵢ(x, z) = θ₀ * exp(N² * z / g)
ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

U₀ = 10  # Background wind speed (m/s)

set!(model, ρ=ρᵢ, θ=θᵢ, u=U₀)

# ## Time-stepping
#
# Acoustic waves require a CFL condition based on the sound speed.

Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
ℂᵃᶜ = sqrt(γ * Rᵈ * θ₀)

Δx = Lx / Nx
Δz = Lz / Nz
Δt = 0.4 * min(Δx, Δz) / (ℂᵃᶜ + U₀)

stop_time = 2000  # seconds — enough for waves to develop

simulation = Simulation(model; Δt, stop_time)

function progress(sim)
    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: %d, t: %s, max|u|: %.2f, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim),
                   maximum(abs, u), maximum(abs, w))
    @info msg
end

add_callback!(simulation, progress, IterationInterval(500))

# ## Output
#
# Save vertical velocity and the contravariant vertical velocity for visualization.

w = model.velocities.w
Ω̃ = model.dynamics.Ω̃

filename = "terrain_following_mountain_wave.jld2"
outputs = (; w, Ω̃)

simulation.output_writers[:jld2] = JLD2Writer(model, outputs; filename,
                                              schedule = TimeInterval(100),
                                              overwrite_existing = true)

run!(simulation)

# ## Visualization

wts = FieldTimeSeries(filename, "w")
Ω̃ts = FieldTimeSeries(filename, "Ω̃")
times = wts.times
Nt = length(times)

fig = Figure(size = (900, 500), fontsize = 12)

axw = Axis(fig[1, 1]; ylabel = "z (m)", title = "w (m/s)")
axΩ = Axis(fig[2, 1]; ylabel = "z (m)", xlabel = "x (km)", title = "Ω̃ (m/s)")
hidexdecorations!(axw, grid = false)

n = Observable(Nt)
wn = @lift wts[$n]
Ω̃n = @lift Ω̃ts[$n]

wlim = maximum(abs, wts[Nt]) / 2
wlim = max(wlim, 1e-6)

hmw = heatmap!(axw, wn; colormap = :balance, colorrange = (-wlim, wlim))
hmΩ = heatmap!(axΩ, Ω̃n; colormap = :balance, colorrange = (-wlim, wlim))

Colorbar(fig[1, 2], hmw)
Colorbar(fig[2, 2], hmΩ)

title = @lift "Mountain wave — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title; fontsize = 16, tellwidth = false)

CairoMakie.record(fig, "terrain_following_mountain_wave.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](terrain_following_mountain_wave.mp4)
