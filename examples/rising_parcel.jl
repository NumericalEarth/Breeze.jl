# # Rising parcel: dry adiabatic ascent
#
# This example demonstrates the `ParcelDynamics` module, which simulates
# Lagrangian air parcels rising through a prescribed atmospheric sounding.
# As the parcel ascends, it cools adiabatically following the dry adiabatic
# lapse rate.
#
# ## Physics overview
#
# A rising parcel undergoes adiabatic expansion as pressure decreases with
# height. For a dry adiabat, temperature decreases at approximately 9.8 K/km
# (the dry adiabatic lapse rate). The parcel conserves its potential temperature
# or static energy during this process.
#
# This example shows how `ParcelDynamics` correctly:
# 1. Evolves parcel position through the environmental velocity field
# 2. Applies adiabatic adjustment as pressure changes
# 3. Tracks thermodynamic state through the ascent

using Breeze
using Breeze.ParcelDynamics: ParcelDynamics, ParcelState, EnvironmentalProfile,
    adiabatic_adjustment, environmental_velocity, environmental_pressure, environmental_density
using Breeze.Thermodynamics: StaticEnergyState, MoistureMassFractions,
    temperature, density, saturation_specific_humidity,
    PlanarLiquidSurface, mixture_heat_capacity, with_moisture
using Breeze.AtmosphereModels: NothingMicrophysicalState
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

# ## Create ParcelDynamics with initial state

dynamics = ParcelDynamics(profile, state₀)

# ## Run the parcel simulation
#
# We integrate for 30 minutes with a 1 second time step.

Δt = 1.0         # Time step [s]
stop_time = 1800.0  # 30 minutes

# Storage for time series
times = Float64[0.0]
heights = Float64[dynamics.state.z]
T_initial = temperature(dynamics.state.𝒰, constants)
temperatures = Float64[T_initial]

# Compute initial supersaturation
ρ_initial = density(dynamics.state.𝒰, constants)
qᵛ⁺_initial = saturation_specific_humidity(T_initial, ρ_initial, constants, PlanarLiquidSurface())
S_initial = (dynamics.state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺_initial) - 1
supersaturations = Float64[S_initial]

# Time stepping function for dry adiabatic parcel
function step_dry_parcel!(dynamics, Δt, constants)
    state = dynamics.state
    profile = dynamics.profile

    x, y, z = state.x, state.y, state.z
    qᵗ = state.qᵗ
    𝒰 = state.𝒰
    ℳ = state.ℳ

    # Get environmental velocity
    u, v, w = environmental_velocity(profile, z)

    # Update position (Forward Euler)
    x_new = x + u * Δt
    y_new = y + v * Δt
    z_new = z + w * Δt

    # Environmental conditions at new height
    p_new = environmental_pressure(profile, z_new)
    ρ_new = environmental_density(profile, z_new)

    # Adiabatic adjustment of thermodynamic state
    𝒰_new = adiabatic_adjustment(𝒰, z_new, p_new, constants)

    # Update state
    dynamics.state = ParcelState(x_new, y_new, z_new, ρ_new, qᵗ, 𝒰_new, ℳ)
    return nothing
end

# Time loop
for n in 1:Int(stop_time / Δt)
    step_dry_parcel!(dynamics, Δt, constants)

    # Record state
    push!(times, n * Δt)
    push!(heights, dynamics.state.z)

    T = temperature(dynamics.state.𝒰, constants)
    ρ = density(dynamics.state.𝒰, constants)
    push!(temperatures, T)

    # Supersaturation
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    S = (dynamics.state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺) - 1
    push!(supersaturations, S)
end

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
# The parcel rises at 1 m/s through the environmental profile.
# As it ascends, pressure drops and the parcel cools adiabatically.
#
# For a dry adiabat with static energy conservation, the temperature
# decreases at the dry adiabatic lapse rate:
#
# ```math
# \Gamma_d = \frac{g}{c_p^m} \approx 9.8 \text{ K/km}
# ```
#
# Since the environmental lapse rate (6.5 K/km) is less steep than
# the dry adiabatic lapse rate, the parcel becomes increasingly
# cooler than its environment as it rises. This would make it
# negatively buoyant in a real atmosphere.
#
# The supersaturation panel shows that as the parcel cools, it
# approaches saturation (S → 0). With microphysics enabled,
# condensation would begin once S > 0, releasing latent heat
# and slowing the cooling rate.
