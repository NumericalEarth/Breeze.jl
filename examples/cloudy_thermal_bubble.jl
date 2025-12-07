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
reference_state = ReferenceState(grid, thermodynamic_constants, base_pressure=1e5, potential_temperature=300)
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
hmw = heatmap!(axw, w)

Colorbar(fig[1, 2], hmθ, label = "θ (K) at t = $(prettytime(simulation.model.clock.time))")
Colorbar(fig[2, 2], hmw, label = "w (m/s) at t = $(prettytime(simulation.model.clock.time))")

fig

# Just running to t=1000 is pretty boring, Let's run the simulation for a longer time, just for fun!

# simulation.stop_time = 30minutes
# run!(simulation)

# ## Visualization
#
# Visualize the potential temperature and the vertical velocity through
# time and create an animation.

θt = FieldTimeSeries(filename, "θ")
wt = FieldTimeSeries(filename, "w")

times = θt.times
fig = Figure(size = (800, 800), fontsize = 12)
axθ = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[2, 1], aspect=2, xlabel="x (m)", ylabel="z (m)")

n = Observable(length(θt))
θn = @lift θt[$n]
wn = @lift wt[$n]

title = @lift "Dry thermal bubble evolution — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

θ_range = (minimum(θt), maximum(θt))
w_range = maximum(abs, wt)

hmθ = heatmap!(axθ, θn, colorrange = θ_range, colormap = :thermal)
hmw = heatmap!(axw, wn, colorrange = (-w_range, w_range), colormap = :balance)

Colorbar(fig[1, 2], hmθ, label = "θ (K)", vertical = true)
Colorbar(fig[2, 2], hmw, label = "w (m/s)", vertical = true)

CairoMakie.record(fig, "dry_thermal_bubble.mp4", 1:length(θt), framerate = 12) do nn
    n[] = nn
end

nothing #hide

# ![](dry_thermal_bubble.mp4)

# ## Moist thermal bubble with warm-phase saturation adjustment
#
# Now we set up a moist thermal bubble simulation with warm-phase saturation adjustment,
# following the methodology described by Bryan and Fritsch (2002). This simulation
# includes moisture processes, where excess water vapor condenses to liquid water,
# releasing latent heat that enhances the buoyancy of the rising bubble.
# 
# For pedagogical purposes, we build a new model with warm-phase saturation adjustment microphysics.
# (We coudl have also used this model for the dry simulation):

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
moist_model = AtmosphereModel(grid; formulation, thermodynamic_constants, advection, microphysics)

# ## Moist thermal bubble initial conditions
#
# For the moist bubble, we initialize both temperature and moisture perturbations.
# The bubble is warm and moist, leading to condensation and latent heat release
# as it rises and cools. First, we set the potential temperature to match the dry case,
# then we use the diagnostic saturation specific humidity field to set the moisture.

# Set potential temperature to match the dry bubble initially
set!(moist_model, θ=θᵢ, qᵗ=0.025)

# Compute saturation specific humidity using the diagnostic field,
# and adjust the buoyancy to match the dry bubble
# Note, this isn't quite right and needs to be fixed.

using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant

qᵛ⁺ = SaturationSpecificHumidityField(moist_model, :equilibrium)
θᵈ = liquid_ice_potential_temperature(moist_model) # note, current state is dry
Rᵈ = dry_air_gas_constant(thermodynamic_constants)
Rᵛ = vapor_gas_constant(thermodynamic_constants)
Rᵐ = Rᵈ * (1 - qᵛ⁺) + Rᵛ * qᵛ⁺
θᵐ = θᵈ * Rᵈ / Rᵐ

set!(moist_model, θ=θᵐ)

# ## Simulation

moist_simulation = Simulation(moist_model; Δt=2, stop_time=1hours)
conjure_time_step_wizard!(moist_simulation, cfl=0.7)

E = total_energy(moist_model)
θ = liquid_ice_potential_temperature(moist_model)

function progress_moist(sim)
    ρqᵗ = sim.model.moisture_density
    u, v, w = sim.model.velocities

    msg = @sprintf("Iter: % 4d, t: % 14s, Δt: % 14s, ⟨E⟩: %.8e J, extrema(θ): (%.2f, %.2f) K \n",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), mean(E), extrema(θ)...)

    msg *= @sprintf("   extrema(qᵗ): (%.2e, %.2e), max(qˡ): %.2e, max|w|: %.2f m/s, mean(qᵗ): %.2e",
                    extrema(ρqᵗ)..., maximum(qˡ), maximum(abs, w), mean(ρqᵗ))

    @info msg
    return nothing
end

add_callback!(moist_simulation, progress_moist, TimeInterval(100))

θ = liquid_ice_potential_temperature(moist_model)
u, v, w = moist_model.velocities
qᵗ = moist_model.specific_moisture
qˡ = moist_model.microphysical_fields.qˡ
qˡ′ = qˡ - Field(Average(qˡ, dims=1))
moist_outputs = (; θ, w, qˡ′)

moist_filename = "moist_thermal_bubble.jld2"
moist_writer = JLD2Writer(moist_model, moist_outputs; filename=moist_filename,
                          schedule = TimeInterval(10seconds),
                          overwrite_existing = true)

moist_simulation.output_writers[:jld2] = moist_writer

run!(moist_simulation)

# ## Visualization of moist thermal bubble

θt = FieldTimeSeries(moist_filename, "θ")
wt = FieldTimeSeries(moist_filename, "w")
qˡ′t = FieldTimeSeries(moist_filename, "qˡ′")

times = θt.times
fig = Figure(size = (1800, 800), fontsize = 12)
axθ = Axis(fig[1, 2], aspect=2, xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[1, 3], aspect=2, xlabel="x (m)", ylabel="z (m)")
axl = Axis(fig[2, 2:3], aspect=2, xlabel="x (m)", ylabel="z (m)")

θ_range = (minimum(θt), maximum(θt))
w_range = maximum(abs, wt)
qˡ′_range = (minimum(qˡ′t), maximum(qˡ′t))

n = Observable(length(θt))
θn = @lift θt[$n]
wn = @lift wt[$n]
qˡ′n = @lift qˡ′t[$n]

hmθ = heatmap!(axθ, θn, colorrange = θ_range, colormap = :thermal)
hmw = heatmap!(axw, wn, colorrange = (-w_range, w_range), colormap = :balance)
hml = heatmap!(axl, qˡ′n, colorrange = qˡ′_range, colormap = :balance)

Colorbar(fig[1, 1], hmθ, label = "θ (K)", vertical = true)
Colorbar(fig[1, 4], hmw, label = "w (m/s)", vertical = true)
Colorbar(fig[2, 4], hml, label = "qˡ (kg/kg)", vertical = true)

CairoMakie.record(fig, "cloudy_thermal_bubble.mp4", 1:length(θt), framerate = 24) do nn
    n[] = nn
end
nothing #hide

# ![](cloudy_thermal_bubble.mp4)
