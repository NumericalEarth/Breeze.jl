# # Cloudy thermal bubble
#
# This example sets up, runs, and visualizes simulations of "thermal bubbles"
# (just circular regions of warm air) rising through a neutral background.
# We run both a dry simulation and a "cloudy" simulation. In the cloudy case,
# we simulate a pocket of warm air rising in a saturated, condensate-laden environment.

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

# ## Dry thermal bubble
#
# We first set up a dry thermal bubble simulation without moisture processes.
# This serves as a baseline for comparison with the moist case.

grid = RectilinearGrid(CPU();
                       size = (128, 128), halo = (5, 5),
                       x = (-10e3, 10e3),
                       z = (0, 10e3),
                       topology = (Bounded, Flat, Bounded))

thermodynamic_constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermodynamic_constants, surface_pressure=1e5, potential_temperature=300)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)
advection = WENO(order=9)
model = AtmosphereModel(grid; formulation, thermodynamic_constants, advection)

# ## Potential temperature perturbation
#
# We add a localized potential temperature perturbation for the dry bubble.
# In the dry case, this perturbation directly affects buoyancy without any
# moisture-related effects.

r₀ = 2e3
z₀ = 2e3
Δθ = 2 # K
θ₀ = model.formulation.reference_state.potential_temperature
g = model.thermodynamic_constants.gravitational_acceleration

function θᵢ(x, z)
    r = sqrt((x / r₀)^2 + ((z - z₀) / r₀)^2)
    return θ₀ + Δθ * cos(π * min(1, r) / 2)^2
end

set!(model, θ=θᵢ)

# ## Initial dry bubble visualization
#
# Plot the initial potential temperature to visualize the dry thermal bubble.

θ = liquid_ice_potential_temperature(model)
E = total_energy(model)
∫E = Integral(E) |> Field

fig = Figure()
ax = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial potential temperature θ (K)")
hm = heatmap!(ax, θ)
Colorbar(fig[1, 2], hm, label = "ρe′ (J/kg)")
fig

# ## Simulation rising

simulation = Simulation(model; Δt=2, stop_time=1000)
conjure_time_step_wizard!(simulation, cfl=0.7)
θ = liquid_ice_potential_temperature(model)

function progress(sim)
    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: % 4d, t: % 14s, Δt: % 14s, ⟨E⟩: %.8e J, extrema(θ): (%.2f, %.2f) K, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), mean(E), extrema(θ)..., maximum(abs, w))
    @info msg
    return nothing
end

add_callback!(simulation, progress, TimeInterval(100))

u, v, w = model.velocities
outputs = (; θ, w)

filename = "dry_thermal_bubble.jld2"
writer = JLD2Writer(model, outputs; filename,
                    schedule = TimeInterval(10seconds),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

run!(simulation)

fig = Figure()
axθ = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[2, 1], aspect=2, xlabel="x (m)", ylabel="z (m)")

hmθ = heatmap!(axθ, θ)
hmw = heatmap!(axw, w, colorrange = (-20, 20), colormap = :balance)
w_levels = -20 : 2 : 20
contour!(axw, w, levels = w_levels, color = :black, linewidth = 0.5)

Colorbar(fig[1, 2], hmθ, label = "θ (K) at t = $(prettytime(simulation.model.clock.time))")
Colorbar(fig[2, 2], hmw, label = "w (m/s) at t = $(prettytime(simulation.model.clock.time))")

fig
save("dry_thermal.png", fig)