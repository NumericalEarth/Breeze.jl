# # Thermal bubble
#
# This example sets up, runs, and visualizes a "thermal bubble" (just a circular
# region of warm air) rising through a stably-stratified background.

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

# ## A simple model on a RectilinearGrid

grid = RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                       x = (-10e3, 10e3), z = (0, 10e3),
                       topology = (Periodic, Flat, Bounded))

advection = WENO(order=9)
model = AtmosphereModel(grid; advection)

# ## Moist static energy perturbation
#
# We add a localized potential temperature perturbation that translates into a
# moist static energy anomaly.

r₀ = 2e3
Δθ = 10 # K
N² = 1e-6
θ₀ = model.formulation.reference_state.potential_temperature
g = model.thermodynamic_constants.gravitational_acceleration

function θᵢ(x, z;
            x₀ = mean(xnodes(grid, Center())),
            z₀ = 0.3*grid.Lz,
            N² = N²)

    θ̄ = θ₀ * exp(N² * z / g)
    r = sqrt((x - x₀)^2 + (z - z₀)^2)
    θ′ = Δθ * max(0, 1 - r / r₀)

    return θ̄ + θ′
end

set!(model, θ = θᵢ)

ρe = model.formulation.thermodynamics.energy_density
ρE = Field(Average(ρe, dims=1))
ρe′ = Field(ρe - ρE)

# ## Initial energy perturbation visualization
#
# Plot the initial moist static energy perturbation to ensure the bubble looks
# as expected.

fig = Figure()
ax = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial energy perturbation ρe′ (J / kg)")
hm = heatmap!(ax, ρe′)
Colorbar(fig[1, 2], hm, label = "ρe′ (J/kg)")
fig

# ## Simulation rising

simulation = Simulation(model; Δt=2, stop_time=25minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    ρe = sim.model.formulation.thermodynamics.energy_density
    u, v, w = sim.model.velocities

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, extrema(ρe): (%.2f, %.2f) J/kg, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   minimum(ρe), maximum(ρe),
                   maximum(abs, u), maximum(abs, w))

    @info msg
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1minute))

u, v, w = model.velocities
T = model.temperature

outputs = merge(model.velocities, model.tracers, (; ρe′, ρe, T))

filename = "thermal_bubble.jld2"
writer = JLD2Writer(model, outputs; filename,
                    schedule = TimeInterval(10seconds),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

run!(simulation)

# ## Visualization
#
# Visualize the moist static energy perturbation and the vertical velocity through
# time and create an animation.

@info "Creating visualization..."

ρe′t = FieldTimeSeries(filename, "ρe′")
wt = FieldTimeSeries(filename, "w")

times = ρe′t.times
Nt = length(ρe′t)

fig = Figure(size = (800, 800), fontsize = 12)
axρ = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Energy perturbation ρe′ (J / kg)")
axw = Axis(fig[2, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Vertical velocity w (m / s)")

n = Observable(Nt)

ρe′n = @lift ρe′t[$n]
wn = @lift wt[$n]

title = @lift "Thermal bubble evolution — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

ρe′_range = (minimum(ρe′t), maximum(ρe′t))
w_range = maximum(abs, wt)

hmρ = heatmap!(axρ, ρe′n, colorrange = ρe′_range, colormap = :balance)
hmw = heatmap!(axw, wn, colorrange = (-w_range, w_range), colormap = :balance)

Colorbar(fig[1, 2], hmρ, label = "ρe′ (J/kg)", vertical = true)
Colorbar(fig[2, 2], hmw, label = "w (m/s)", vertical = true)

CairoMakie.record(fig, "thermal_bubble.mp4", 1:Nt, framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](thermal_bubble.mp4)
