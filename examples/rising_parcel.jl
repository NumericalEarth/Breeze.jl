# # Rising parcel: dry adiabatic ascent
#
# This example demonstrates the `ParcelDynamics` module, which simulates
# Lagrangian air parcels rising through a prescribed atmospheric sounding.
# As the parcel ascends, it cools adiabatically following the dry adiabatic
# lapse rate.
#
# The key feature is that `ParcelDynamics` works with `AtmosphereModel`,
# using the same `Simulation` interface as grid-based simulations.
#
# ## Physics overview
#
# A rising parcel undergoes adiabatic expansion as pressure decreases with
# height. For a dry adiabat, temperature decreases at approximately 9.8 K/km
# (the dry adiabatic lapse rate). The parcel conserves its potential temperature
# or static energy during this process.

using Oceananigans
using Oceananigans: interpolate
using Oceananigans.Units
using Breeze
using Breeze.ParcelModels: ParcelDynamics
using Breeze.Thermodynamics: temperature, density, saturation_specific_humidity, PlanarLiquidSurface
using CairoMakie

# ## Create the model
#
# The grid defines the vertical domain for the parcel trajectory.
# ParcelDynamics works with the standard AtmosphereModel interface.

grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
model = AtmosphereModel(grid; dynamics=ParcelDynamics())

# ## Set environmental profiles and initial parcel position
#
# The `set!` function configures the environmental sounding and
# initializes the parcel at the specified height.

reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                 surface_pressure = 101325,
                                 potential_temperature = 300)

p = reference_state.pressure
ρ = reference_state.density
θ₀ = reference_state.potential_temperature

# Humidity profile (exponential decay with height)
qᵗ₀ = 0.015    # Surface specific humidity [kg/kg]
Hq = 2500      # Humidity scale height [m]
qᵗ(z) = qᵗ₀ * exp(-z / Hq)

set!(model, θ=θ₀, p=p, ρ=ρ, qᵗ=qᵗ, z=0, w=1)

@info "Model created" model.dynamics

# ## Set up the simulation
#
# We create a `Simulation` and use a callback to record parcel state.

simulation = Simulation(model; Δt=1.0, stop_time=30minutes)

# Storage for time series
times = Float64[]
heights = Float64[]
T_parcel = Float64[]
T_environment = Float64[]
supersaturations = Float64[]

# Callback to record parcel state at each iteration
function record_parcel_state!(sim)
    model = sim.model
    constants = model.thermodynamic_constants
    state = model.dynamics.state

    push!(times, model.clock.time)
    push!(heights, state.z)

    Tn = temperature(state.𝒰, constants)
    push!(T_parcel, Tn)

    z = state.z
    Te = interpolate((z,), model.temperature)
    push!(T_environment, Te)

    ρn = density(state.𝒰, constants)
    qᵛ⁺ = saturation_specific_humidity(Tn, ρn, constants, PlanarLiquidSurface())
    Sn = (state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺) - 1
    push!(supersaturations, Sn)

    return nothing
end

add_callback!(simulation, record_parcel_state!, IterationInterval(1))

# Run the simulation
run!(simulation)

@info "Simulation complete" model.clock.time model.dynamics.state.z

# Convert heights to km for plotting
heights_km = heights ./ 1000
times_min = times ./ 60

# ## Visualization

set_theme!(fontsize=14, linewidth=2)
fig = Figure(size=(900, 400))

# Panel 1: Height vs Temperature
ax1 = Axis(fig[1, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Parcel ascent: adiabatic cooling")

lines!(ax1, T_parcel, heights_km; color=:magenta, label="Parcel T")

# Add environmental temperature for comparison
lines!(ax1, T_environment, heights_km; color=:gray, linestyle=:dash, label="Environment T")

axislegend(ax1; position=:lt)

# Panel 2: Supersaturation evolution
ax2 = Axis(fig[1, 2];
    xlabel = "Height (km)",
    ylabel = "Supersaturation (%)",
    title = "Approach to saturation")

lines!(ax2, heights_km, supersaturations .* 100; color=:purple)
hlines!(ax2, [0]; color=:gray, linestyle=:dash, label="Saturation")

axislegend(ax2; position=:lb)

fig

# ## Discussion
#
# The parcel rises at 1 m/s through the environmental profile, using the
# standard `time_step!` interface. The key points demonstrated:
#
# 1. **AtmosphereModel integration**: ParcelDynamics works with AtmosphereModel,
#    using the same `time_step!` function as grid-based simulations.
#
# 2. **Grid defines the domain**: The grid specifies the vertical extent
#    through which the parcel can travel.
#
# 3. **set! interface**: Environmental profiles and initial conditions are
#    set using the familiar `set!` function.
#
# 4. **Adiabatic cooling**: As the parcel ascends, pressure drops and temperature
#    decreases following the dry adiabatic lapse rate (~9.8 K/km).
#
# 5. **Approach to saturation**: The supersaturation panel shows the parcel
#    becoming increasingly supersaturated as it cools. With microphysics enabled,
#    condensation would begin once S > 0.
