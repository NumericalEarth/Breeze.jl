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
# terrain-following ([Gal-Chen and Somerville (1975)](@cite GalChen1975)) coordinate
# transformation. The physics are then corrected with [`TerrainMetrics`](@ref) to account
# for the tilted coordinate surfaces.

using Breeze
using Oceananigans.Grids: MutableVerticalDiscretization, znode
using Oceananigans.Operators: Δzᶜᶜᶠ
using Oceananigans.Units
using Breeze.Thermodynamics: ThermodynamicConstants, dry_air_gas_constant
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
# A Gaussian ridge centered at the origin. The mountain height ``h_0`` and half-width
# ``a`` are chosen so that the non-dimensional mountain height ``N h_0 / U`` is in
# the weakly nonlinear regime (``\approx 0.25``), following the test cases reviewed
# by [Klemp (2011)](@cite Klemp2011).

h₀ = 250meters
a = 10kilometers
h(x, y) = h₀ * exp(-x^2 / a^2)

# Apply the terrain to the grid and retrieve the metric terms.

metrics = follow_terrain!(grid, h)

# ## Physical parameters
#
# Define the atmospheric state before building the model, since the
# `reference_potential_temperature` profile is needed at construction time.

constants = ThermodynamicConstants(Float64)
g = constants.gravitational_acceleration

θ₀ = 300      # Surface potential temperature (K)
p₀ = 101325   # Surface pressure (Pa)
pˢᵗ = 1e5     # Standard pressure (Pa)
N² = 1e-4     # Brunt-Väisälä frequency squared (s⁻²)

Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ = Rᵈ / cᵖᵈ
π_surface = (p₀ / pˢᵗ)^κ

# Potential temperature increases exponentially with height for constant ``N^2``.
θ_of_z(z) = θ₀ * exp(N² * z / g)

# ## Sponge layer
#
# A Rayleigh damping layer near the domain top absorbs upward-propagating waves
# and prevents spurious reflections from the rigid lid.

sponge_width = Lz / 4
sponge_mask(x, z) = exp(-(z - Lz)^2 / sponge_width^2)
ρw_sponge = Relaxation(rate=1/10, mask=sponge_mask)

# ## Model construction
#
# Build a compressible model with explicit time-stepping, WENO advection, and terrain
# corrections. Passing `terrain_metrics` to [`CompressibleDynamics`](@ref) activates
# the terrain-following physics: contravariant vertical velocity, corrected pressure
# gradient, and terrain-aware divergence. The `reference_potential_temperature` enables
# a perturbation pressure approach for the horizontal pressure gradient that reduces
# the truncation error inherent in terrain-following coordinates.

dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                terrain_metrics = metrics,
                                reference_potential_temperature = θ_of_z)
model = AtmosphereModel(grid; dynamics, advection=WENO(), forcing=(; ρw=ρw_sponge))

# ## Initial conditions
#
# We initialize the atmosphere in discrete hydrostatic balance using Exner function
# integration. This is essential for compressible models on terrain-following grids:
# the equation of state alone does not produce a pressure field in discrete
# hydrostatic balance, so column-by-column Exner integration is needed to avoid
# spurious vertical accelerations.

# Initialize density and potential temperature density column by column
# via discrete Exner integration from the surface pressure.
ρ_field = model.dynamics.density
ρθ_field = model.formulation.potential_temperature_density

for i in 1:Nx
    πₖ = π_surface
    for k in 1:Nz
        z_phys = znode(i, 1, k, grid, Center(), Center(), Center())
        θₖ = θ_of_z(z_phys)

        if k > 1
            z_below = znode(i, 1, k - 1, grid, Center(), Center(), Center())
            θ_face = (θₖ + θ_of_z(z_below)) / 2
            πₖ = πₖ - g * Δzᶜᶜᶠ(i, 1, k, grid) / (cᵖᵈ * θ_face)
        end

        pₖ = pˢᵗ * πₖ^(1 / κ)
        ρₖ = pₖ / (Rᵈ * θₖ * πₖ)

        ρ_field[i, 1, k] = ρₖ
        ρθ_field[i, 1, k] = ρₖ * θₖ
    end
end

U₀ = 10  # Background wind speed (m/s)

set!(model, u=U₀)

# ## Time-stepping
#
# Acoustic waves require a CFL condition based on the sound speed.

γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
ℂᵃᶜ = sqrt(γ * Rᵈ * θ₀)

Δx = Lx / Nx
Δz = Lz / Nz
Δt = 0.4 * min(Δx, Δz) / (ℂᵃᶜ + U₀)

stop_time = 2000  # seconds — enough for waves to develop

simulation = Simulation(model; Δt, stop_time)

function progress(sim)
    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: %d, t: %s, max|u|: %.2f, max|w|: %.4f m/s",
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
