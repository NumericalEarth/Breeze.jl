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
using Oceananigans.Units
using Breeze
using Breeze.ParcelModels: ParcelDynamics
using Breeze.Thermodynamics: temperature, density, saturation_specific_humidity, PlanarLiquidSurface
using CairoMakie

# ## Environmental sounding
#
# We prescribe a simple environmental profile:
# - Temperature: Standard atmosphere lapse rate (6.5 K/km)
# - Pressure: Hydrostatic pressure from ideal gas
# - Humidity: Decreasing with height
# - Updraft: Constant 1 m/s vertical velocity

g = 9.81
Rᵈ = 287.0
T₀ = 288.15      # Surface temperature [K]
p₀ = 101325.0    # Surface pressure [Pa]
Γ = 0.0065       # Environmental temperature lapse rate [K/m]
qᵗ₀ = 0.015      # Surface specific humidity [kg/kg]
Hq = 2500.0      # Humidity scale height [m]
w_updraft = 1.0  # Updraft velocity [m/s]

# Temperature profile (standard atmosphere)
T(z) = T₀ - Γ * z

# Pressure profile (hypsometric equation for constant lapse rate)
p(z) = p₀ * (T(z) / T₀)^(g / (Rᵈ * Γ))

# Density from ideal gas law
ρ(z) = p(z) / (Rᵈ * T(z))

# Humidity profile (exponential decay)
qᵗ(z) = qᵗ₀ * exp(-z / Hq)

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

set!(model, T=T, p=p, ρ=ρ, qᵗ=qᵗ, z=0.0, w=w_updraft)

@info "Model created" model.dynamics

# ## Set up the simulation
#
# We create a `Simulation` and use a callback to record parcel state.

simulation = Simulation(model; Δt=1.0, stop_time=30minutes)

# Storage for time series
times = Float64[]
heights = Float64[]
temperatures = Float64[]
supersaturations = Float64[]

# Callback to record parcel state at each iteration
function record_parcel_state!(sim)
    model = sim.model
    constants = model.thermodynamic_constants
    state = model.dynamics.state

    push!(times, model.clock.time)
    push!(heights, state.z)

    Tₙ = temperature(state.𝒰, constants)
    ρₙ = density(state.𝒰, constants)
    push!(temperatures, Tₙ)

    qᵛ⁺ = saturation_specific_humidity(Tₙ, ρₙ, constants, PlanarLiquidSurface())
    S = (state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺) - 1
    push!(supersaturations, S)

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

lines!(ax1, temperatures, heights_km; color=:magenta, label="Parcel T")

# Add environmental temperature for comparison
z_range = range(0, stop=maximum(heights), length=100)
T_env_profile = T.(z_range)
lines!(ax1, T_env_profile, z_range./1000; color=:gray, linestyle=:dash, label="Environment T")

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
