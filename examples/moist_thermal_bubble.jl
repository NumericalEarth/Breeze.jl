# # Thermal bubbles -- moist and dry
#
# This example sets up, runs, and visualizes simulations of "thermal bubbles"
# (just circular regions of warm air) rising through a neutral background.
# We run both a dry and a moist simulation, and compare the results.

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

# ## Dry thermal bubble
#
# We first set up a dry thermal bubble simulation without moisture processes.
# This serves as a baseline for comparison with the moist case.

grid = RectilinearGrid(CPU(); size = (128, 128), halo = (5, 5),
                       x = (-10e3, 10e3),
                       z = (0, 10e3),
                       topology = (Bounded, Flat, Bounded))

thermodynamic_constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermodynamic_constants, base_pressure=1e5, potential_temperature=300)
formulation = AnelasticFormulation(reference_state, thermodynamics=:PotentialTemperature)
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

θ = potential_temperature(model)

fig = Figure()
ax = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial potential temperature θ (K)")
hm = heatmap!(ax, θ)
Colorbar(fig[1, 2], hm, label = "ρe′ (J/kg)")
fig

# ## Simulation rising

simulation = Simulation(model; Δt=2, stop_time=1000)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    θ = potential_temperature(sim.model)
    u, v, w = sim.model.velocities

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, extrema(θ): (%.2f, %.2f) K, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), extrema(θ)...,
                   maximum(abs, u), maximum(abs, w))

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

simulation.stop_time = 30minutes
run!(simulation)

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

# Create a new model with warm-phase saturation adjustment microphysics
microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
moist_model = AtmosphereModel(grid; formulation, thermodynamic_constants,
                              advection, microphysics)

# ## Moist thermal bubble initial conditions
#
# For the moist bubble, we initialize both temperature and moisture perturbations.
# The bubble is warm and moist, leading to condensation and latent heat release
# as it rises and cools. First, we set the potential temperature to match the dry case,
# then we use the diagnostic saturation specific humidity field to set the moisture.

# Set potential temperature to match the dry bubble initially
set!(moist_model, θ=θᵢ)

# Compute saturation specific humidity using the diagnostic field
using Breeze.AtmosphereModels: SaturationSpecificHumidityField

qᵛ⁺ = SaturationSpecificHumidityField(moist_model)
θᵈ = potential_temperature(moist_model) # note, current state is dry
Rᵈ = thermodynamic_constants.dry_air.gas_constant
Rᵛ = thermodynamic_constants.vapor.gas_constant
Rᵐ = Rᵈ * (1 - qᵛ⁺) + Rᵛ * qᵛ⁺
θᵐ = θᵈ * Rᵈ / Rᵐ

set!(moist_model, θ=θᵐ, qᵛ=qᵛ⁺)

#=
# Set total moisture to saturation specific humidity
set!(moist_model, qᵗ=qᵛ⁺)

# Compute adjusted potential temperature to maintain buoyancy (Bryan and Fritsch 2002)
# The virtual potential temperature is ϑ = Rᵐ / Rᵈ * θ
# To maintain the same buoyancy (same virtual potential temperature), we need:
# θ_moist = Rᵈ / Rᵐ * θ_dry
using Breeze.Thermodynamics: mixture_gas_constant, dry_air_gas_constant, MoistureMassFractions

θ_dry_field = potential_temperature(model)
qᵗ_field = moist_model.specific_moisture
T_field = moist_model.temperature

# Compute adjusted potential temperature to maintain buoyancy (Bryan and Fritsch 2002)
# The virtual potential temperature is ϑ = Rᵐ / Rᵈ * θ
# To maintain the same buoyancy (same virtual potential temperature), we need:
# θ_moist = Rᵈ / Rᵐ * θ_dry
compute!(θ_dry_field)
compute!(qᵗ_field)

# Create adjusted potential temperature field
θ_adjusted_field = CenterField(grid)

# Compute adjusted potential temperature
using GPUArraysCore: @allowscalar
θ_dry_data = parent(θ_dry_field)
qᵗ_data = parent(qᵗ_field)
θ_adjusted_data = parent(θ_adjusted_field)

@allowscalar begin
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        θ_dry = θ_dry_data[i, j, k]
        qᵗ = qᵗ_data[i, j, k]
        q = MoistureMassFractions(qᵗ)
        Rᵐ = mixture_gas_constant(q, thermodynamic_constants)
        Rᵈ = dry_air_gas_constant(thermodynamic_constants)
        θ_adjusted_data[i, j, k] = Rᵈ / Rᵐ * θ_dry
    end
end

# Set the adjusted potential temperature
set!(moist_model, θ=θ_adjusted_field)

# ## Initial moist bubble visualization

θ_moist = potential_temperature(moist_model)
qᵗ_moist = moist_model.specific_moisture

fig = Figure()
axθ = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial potential temperature θ (K)")
axq = Axis(fig[2, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial total moisture qᵗ (kg/kg)")
hmθ = heatmap!(axθ, θ_moist)
hmq = heatmap!(axq, qᵗ_moist)
Colorbar(fig[1, 2], hmθ, label = "θ (K)")
Colorbar(fig[2, 2], hmq, label = "qᵗ (kg/kg)")
fig

# ## Moist bubble simulation

simulation_moist = Simulation(moist_model; Δt=2, stop_time=1000)
conjure_time_step_wizard!(simulation_moist, cfl=0.7)

function progress_moist(sim)
    θ = potential_temperature(sim.model)
    qᵗ = sim.model.specific_moisture
    u, v, w = sim.model.velocities
    
    msg = @sprintf("Iter: %d, t: %s, Δt: %s, extrema(θ): (%.2f, %.2f) K, extrema(qᵗ): (%.2e, %.2e) kg/kg, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), extrema(θ)...,
                   extrema(qᵗ)..., maximum(abs, u), maximum(abs, w))
    
    @info msg
    return nothing
end

add_callback!(simulation_moist, progress_moist, TimeInterval(100))

u_moist, v_moist, w_moist = moist_model.velocities
θ_moist = potential_temperature(moist_model)
qᵗ_moist = moist_model.specific_moisture
qˡ_moist = moist_model.microphysical_fields.qˡ

outputs_moist = (; θ=θ_moist, w=w_moist, qᵗ=qᵗ_moist, qˡ=qˡ_moist)

filename_moist = "moist_thermal_bubble.jld2"
writer_moist = JLD2Writer(moist_model, outputs_moist; filename=filename_moist,
                          schedule = TimeInterval(10seconds),
                          overwrite_existing = true)

simulation_moist.output_writers[:jld2] = writer_moist

run!(simulation_moist)

simulation_moist.stop_time = 30minutes
run!(simulation_moist)

# ## Visualization of moist thermal bubble

θt_moist = FieldTimeSeries(filename_moist, "θ")
wt_moist = FieldTimeSeries(filename_moist, "w")
qᵗt_moist = FieldTimeSeries(filename_moist, "qᵗ")

times_moist = θt_moist.times
fig = Figure(size = (1200, 800), fontsize = 12)
axθ = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[2, 1], aspect=2, xlabel="x (m)", ylabel="z (m)")
axq = Axis(fig[1, 2], aspect=2, xlabel="x (m)", ylabel="z (m)")

n_moist = Observable(length(θt_moist))
θn_moist = @lift θt_moist[$n_moist]
wn_moist = @lift wt_moist[$n_moist]
qᵗn_moist = @lift qᵗt_moist[$n_moist]

title_moist = @lift "Moist thermal bubble evolution — t = $(prettytime(times_moist[$n_moist]))"
fig[0, :] = Label(fig, title_moist, fontsize = 16, tellwidth = false)

θ_range_moist = (minimum(θt_moist), maximum(θt_moist))
w_range_moist = maximum(abs, wt_moist)
qᵗ_range_moist = (minimum(qᵗt_moist), maximum(qᵗt_moist))

hmθ_moist = heatmap!(axθ, θn_moist, colorrange = θ_range_moist, colormap = :thermal)
hmw_moist = heatmap!(axw, wn_moist, colorrange = (-w_range_moist, w_range_moist), colormap = :balance)
hmq_moist = heatmap!(axq, qᵗn_moist, colorrange = qᵗ_range_moist, colormap = :viridis)

Colorbar(fig[1, 3], hmθ_moist, label = "θ (K)", vertical = true)
Colorbar(fig[2, 3], hmw_moist, label = "w (m/s)", vertical = true)
Colorbar(fig[1, 0], hmq_moist, label = "qᵗ (kg/kg)", vertical = true)

CairoMakie.record(fig, "moist_thermal_bubble.mp4", 1:length(θt_moist), framerate = 12) do nn
    n_moist[] = nn
end
nothing #hide

# ![](moist_thermal_bubble.mp4)
=#