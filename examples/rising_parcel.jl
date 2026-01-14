# # Rising parcel with cloud formation
#
# This example demonstrates the new `ParcelDynamics` module, which simulates
# Lagrangian air parcels rising through a prescribed atmospheric sounding.
# As the parcel ascends, it cools adiabatically and eventually reaches
# saturation, triggering cloud formation.
#
# The key insight is that **microphysics tendencies are purely local** -
# they depend only on the parcel's thermodynamic state, not on neighboring
# grid cells. This enables the same microphysics code to work for both
# grid-based LES and Lagrangian parcel models without duplication.
#
# ## Physics overview
#
# A rising parcel undergoes:
# 1. **Adiabatic cooling**: Temperature decreases as pressure drops (~10 K/km)
# 2. **Supersaturation**: As T drops, saturation vapor pressure drops faster than actual vapor pressure
# 3. **Condensation**: Excess vapor condenses onto cloud droplets (relaxation timescale τ ~ 10 s)
# 4. **Latent heating**: Condensation releases heat, partially offsetting cooling
# 5. **Precipitation formation**: Cloud droplets grow and eventually rain out
#
# This is the classic "adiabatic parcel model" used to understand cloud microphysics,
# dating back to the foundational work of [Köhler1921](@citet).

using Breeze
using Breeze.ParcelDynamics: ParcelModel, ParcelState, EnvironmentalProfile,
    step_parcel!, adiabatic_adjustment, compute_moisture_fractions
using Breeze.Thermodynamics: StaticEnergyState, LiquidIcePotentialTemperatureState,
    MoistureMassFractions, temperature, density, saturation_specific_humidity,
    PlanarLiquidSurface, mixture_heat_capacity
using CloudMicrophysics
using CairoMakie

# ## Environmental sounding
#
# We prescribe a simple environmental profile:
# - Temperature: Standard atmosphere lapse rate (6.5 K/km)
# - Pressure: Hydrostatic pressure from ideal gas
# - Humidity: Decreasing with height (relative humidity ~ 80% at surface)
# - Updraft: Constant 1 m/s vertical velocity

const g = 9.81
const Rᵈ = 287.0
const T₀ = 288.15  # Surface temperature [K]
const p₀ = 101325.0  # Surface pressure [Pa]
const Γ = 0.0065  # Temperature lapse rate [K/m]
const qᵗ₀ = 0.015  # Surface specific humidity [kg/kg]
const H_q = 2500.0  # Humidity scale height [m]
const w_updraft = 1.0  # Updraft velocity [m/s]

# Temperature profile (standard atmosphere)
T_env(z) = T₀ - Γ * z

# Pressure profile (hypsometric equation for constant lapse rate)
# p(z) = p₀ * (T(z)/T₀)^(g/(Rᵈ*Γ))
p_env(z) = p₀ * (T_env(z) / T₀)^(g / (Rᵈ * Γ))

# Density from ideal gas law (dry approximation for environmental profile)
ρ_env(z) = p_env(z) / (Rᵈ * T_env(z))

# Humidity profile (exponential decay)
qᵗ_env(z) = qᵗ₀ * exp(-z / H_q)

# Calm horizontal winds, constant updraft
profile = EnvironmentalProfile(
    temperature = T_env,
    pressure = p_env,
    density = ρ_env,
    specific_humidity = qᵗ_env,
    u = z -> 0.0,
    v = z -> 0.0,
    w = z -> w_updraft
)

# ## Microphysics scheme
#
# We use the one-moment warm-phase non-equilibrium scheme.
# Cloud liquid and rain are prognostic; condensation uses relaxation toward saturation.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics
WarmPhaseOneMomentState = BreezeCloudMicrophysicsExt.WarmPhaseOneMomentState

microphysics = OneMomentCloudMicrophysics()
constants = ThermodynamicConstants()

# ## Initialize parcel at surface
#
# The parcel starts at z = 0 with environmental conditions.
# We use `StaticEnergyState` for the thermodynamic formulation.

z₀ = 0.0
T_init = T_env(z₀)
p_init = p_env(z₀)
ρ_init = ρ_env(z₀)
qᵗ_init = qᵗ_env(z₀)

# Initial moisture: all vapor, no cloud or rain
q_init = MoistureMassFractions(qᵗ_init)

# Static energy: e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ
cᵖᵐ = mixture_heat_capacity(q_init, constants)
e_init = cᵖᵐ * T_init + g * z₀

# Create thermodynamic state
𝒰_init = StaticEnergyState(e_init, q_init, z₀, p_init)

# Initial microphysical state: no cloud or rain
ℳ_init = WarmPhaseOneMomentState(0.0, 0.0)

# Create parcel state
parcel = ParcelState(0.0, 0.0, z₀, ρ_init, qᵗ_init, 𝒰_init, ℳ_init)

# ## Create parcel model

model = ParcelModel(profile, microphysics, constants)

# ## Run the parcel simulation
#
# We integrate for 30 minutes with a 1 second time step.

Δt = 1.0  # Time step [s]
stop_time = 1800.0  # 30 minutes

# Storage for time series
times = Float64[0.0]
heights = Float64[parcel.z]
temperatures = Float64[temperature(parcel.thermodynamic_state, constants)]
qᵛ_series = Float64[parcel.thermodynamic_state.moisture_mass_fractions.vapor]
qᶜˡ_series = Float64[parcel.microphysical_state.qᶜˡ]
qʳ_series = Float64[parcel.microphysical_state.qʳ]
supersaturations = Float64[]

# Compute initial supersaturation
T_curr = temperature(parcel.thermodynamic_state, constants)
ρ_curr = density(parcel.thermodynamic_state, constants)
qᵛ⁺ = saturation_specific_humidity(T_curr, ρ_curr, constants, PlanarLiquidSurface())
S_init = (parcel.thermodynamic_state.moisture_mass_fractions.vapor / qᵛ⁺) - 1
push!(supersaturations, S_init)

# Time loop
t = 0.0
current_parcel = parcel

while t < stop_time
    global t, current_parcel

    # Step the parcel forward
    current_parcel = step_parcel!(current_parcel, model, Δt)
    t += Δt

    # Record state
    push!(times, t)
    push!(heights, current_parcel.z)

    T_curr = temperature(current_parcel.thermodynamic_state, constants)
    ρ_curr = density(current_parcel.thermodynamic_state, constants)
    push!(temperatures, T_curr)

    q = current_parcel.thermodynamic_state.moisture_mass_fractions
    push!(qᵛ_series, q.vapor)
    push!(qᶜˡ_series, current_parcel.microphysical_state.qᶜˡ)
    push!(qʳ_series, current_parcel.microphysical_state.qʳ)

    # Supersaturation
    qᵛ⁺ = saturation_specific_humidity(T_curr, ρ_curr, constants, PlanarLiquidSurface())
    S = (q.vapor / qᵛ⁺) - 1
    push!(supersaturations, S)
end

# Convert heights to km for plotting
heights_km = heights ./ 1000

# ## Visualization
#
# We plot the parcel's journey through thermodynamic space:
# height vs temperature (showing adiabatic cooling), and
# the evolution of moisture and supersaturation.

set_theme!(fontsize=14, linewidth=2)
fig = Figure(size=(1000, 800))

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

# Panel 2: Moisture evolution with height
ax2 = Axis(fig[1, 2];
    xlabel = "Mixing ratio (g/kg)",
    ylabel = "Height (km)",
    title = "Moisture partition")

lines!(ax2, qᵛ_series .* 1000, heights_km; color=:dodgerblue, label="qᵛ (vapor)")
lines!(ax2, qᶜˡ_series .* 1000, heights_km; color=:lime, label="qᶜˡ (cloud)")
lines!(ax2, qʳ_series .* 1000, heights_km; color=:orangered, label="qʳ (rain)")

axislegend(ax2; position=:rt)

# Panel 3: Time series of moisture
ax3 = Axis(fig[2, 1];
    xlabel = "Time (min)",
    ylabel = "Mixing ratio (g/kg)",
    title = "Moisture evolution")

times_min = times ./ 60
lines!(ax3, times_min, qᵛ_series .* 1000; color=:dodgerblue, label="qᵛ")
lines!(ax3, times_min, qᶜˡ_series .* 1000; color=:lime, label="qᶜˡ")
lines!(ax3, times_min, qʳ_series .* 1000; color=:orangered, label="qʳ")

axislegend(ax3; position=:rt)

# Panel 4: Supersaturation
ax4 = Axis(fig[2, 2];
    xlabel = "Time (min)",
    ylabel = "Supersaturation (%)",
    title = "Supersaturation evolution")

lines!(ax4, times_min, supersaturations .* 100; color=:purple)
hlines!(ax4, [0]; color=:gray, linestyle=:dash)

fig

# ## Discussion
#
# The parcel rises at 1 m/s, cooling adiabatically at roughly 10 K/km.
# Initially, all moisture is vapor. As temperature drops, the saturation
# vapor pressure decreases, and eventually the parcel becomes supersaturated.
#
# Once supersaturated, vapor condenses onto cloud droplets following the
# non-equilibrium relaxation:
#
# ```math
# \frac{dq^{cl}}{dt} = \frac{q^v - q^{v*}}{\Gamma \tau}
# ```
#
# where τ ≈ 10 s is the relaxation timescale and Γ is a thermodynamic
# adjustment factor accounting for latent heating.
#
# The supersaturation panel shows the parcel becoming increasingly
# supersaturated as it rises, with condensation working to bring
# supersaturation back toward zero.
#
# This simple parcel model demonstrates that the new `ParcelDynamics`
# infrastructure correctly:
# 1. Evolves parcel position through the environmental profile
# 2. Applies adiabatic adjustment as pressure changes
# 3. Computes microphysics tendencies using the same scalar-state functions
#    used by the grid-based `AtmosphereModel`
