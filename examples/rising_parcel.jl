# # Rising parcel: dry adiabatic ascent
#
# This example demonstrates the `ParcelDynamics` module, which simulates
# Lagrangian air parcels rising through a prescribed atmospheric sounding.
# As the parcel ascends, it cools adiabatically following the dry adiabatic
# lapse rate.
#
# The key feature is that `ParcelDynamics` works with `AtmosphereModel`,
# using the same `time_step!` interface as grid-based simulations.
#
# ## Physics overview
#
# A rising parcel undergoes adiabatic expansion as pressure decreases with
# height. For a dry adiabat, temperature decreases at approximately 9.8 K/km
# (the dry adiabatic lapse rate). The parcel conserves its potential temperature
# or static energy during this process.

using Breeze
using Breeze.ParcelDynamics: ParcelDynamics, ParcelState, EnvironmentalProfile
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions,
    temperature, density, saturation_specific_humidity,
    PlanarLiquidSurface, mixture_heat_capacity
using Breeze.AtmosphereModels: NothingMicrophysicalState
using Oceananigans: set!
using Oceananigans.TimeSteppers: time_step!
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
T_env(z) = T₀ - Γ * z

# Pressure profile (hypsometric equation for constant lapse rate)
p_env(z) = p₀ * (T_env(z) / T₀)^(g / (Rᵈ * Γ))

# Density from ideal gas law
ρ_env(z) = p_env(z) / (Rᵈ * T_env(z))

# Humidity profile (exponential decay)
qᵗ_env(z) = qᵗ₀ * exp(-z / Hq)

# Create the environmental profile
profile = EnvironmentalProfile(
    temperature = T_env,
    pressure = p_env,
    density = ρ_env,
    specific_humidity = qᵗ_env,
    u = z -> 0.0,
    v = z -> 0.0,
    w = z -> w_updraft
)

# ## Initialize parcel at surface
#
# The parcel starts at z = 0 with environmental conditions.
# We use `StaticEnergyState` for the thermodynamic formulation.

constants = ThermodynamicConstants()

z₀ = 0.0
T_init = T_env(z₀)
p_init = p_env(z₀)
ρ_init = ρ_env(z₀)
qᵗ_init = qᵗ_env(z₀)

# Initial moisture: all vapor (no condensate)
q_init = MoistureMassFractions(qᵗ_init)

# Static energy: e = cᵖᵐ * T + g * z
cᵖᵐ = mixture_heat_capacity(q_init, constants)
e_init = cᵖᵐ * T_init + g * z₀

# Create thermodynamic state
𝒰_init = StaticEnergyState(e_init, q_init, z₀, p_init)

# No microphysics for this dry example
ℳ_init = NothingMicrophysicalState(Float64)

# Create initial parcel state
state₀ = ParcelState(0.0, 0.0, z₀, ρ_init, qᵗ_init, 𝒰_init, ℳ_init)

# ## Create AtmosphereModel with ParcelDynamics
#
# ParcelDynamics works with AtmosphereModel, enabling the use of
# `set!` to initialize the state and `time_step!` to evolve it.

dynamics = ParcelDynamics(profile, state₀)
model = AtmosphereModel(dynamics; thermodynamic_constants=constants)

# Check the model type
@info "Created model" typeof(model) model.dynamics

# ## Run the parcel simulation
#
# We integrate for 30 minutes with a 1 second time step,
# using the standard `time_step!` interface.

Δt = 1.0         # Time step [s]
stop_time = 1800.0  # 30 minutes
n_steps = Int(stop_time / Δt)

# Storage for time series
times = Float64[0.0]
heights = Float64[model.dynamics.state.z]
T_initial = temperature(model.dynamics.state.𝒰, constants)
temperatures = Float64[T_initial]

# Compute initial supersaturation
ρ_initial = density(model.dynamics.state.𝒰, constants)
qᵛ⁺_initial = saturation_specific_humidity(T_initial, ρ_initial, constants, PlanarLiquidSurface())
S_initial = (model.dynamics.state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺_initial) - 1
supersaturations = Float64[S_initial]

# Time loop using the standard time_step! interface
for n in 1:n_steps
    time_step!(model, Δt)

    # Record state
    push!(times, model.clock.time)
    push!(heights, model.dynamics.state.z)

    T = temperature(model.dynamics.state.𝒰, constants)
    ρ = density(model.dynamics.state.𝒰, constants)
    push!(temperatures, T)

    # Supersaturation
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    S = (model.dynamics.state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺) - 1
    push!(supersaturations, S)
end

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
T_env_profile = T_env.(z_range)
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
# 2. **Adiabatic cooling**: As the parcel ascends, pressure drops and temperature
#    decreases following the dry adiabatic lapse rate (~9.8 K/km).
#
# 3. **Approach to saturation**: The supersaturation panel shows the parcel
#    becoming increasingly supersaturated as it cools. With microphysics enabled,
#    condensation would begin once S > 0.
#
# 4. **Clock tracking**: The model's clock automatically tracks simulation time,
#    just like grid-based AtmosphereModels.
