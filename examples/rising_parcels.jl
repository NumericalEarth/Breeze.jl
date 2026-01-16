# # Rising parcels: from dry adiabats to precipitating clouds
#
# This example demonstrates `ParcelDynamics`, which simulates Lagrangian air parcels
# ascending through a prescribed atmospheric sounding. We explore two regimes:
#
# 1. **Dry adiabatic ascent**: A rising parcel cools at ~9.8 K/km, conserving
#    potential temperature. Vapor increases toward saturation as temperature drops.
#
# 2. **Cloudy ascent with precipitation**: A moist parcel rises through the
#    lifting condensation level, forming cloud via condensation, then rain via
#    autoconversion. We use two-moment microphysics [(Seifert and Beheng, 2006)](@citet SeifertBeheng2006)
#    to track both mass and number concentration.
#
# The parcel model works with `AtmosphereModel`, using the standard `Simulation` interface.

using Oceananigans
using Oceananigans: interpolate
using Oceananigans.Units
using Breeze
using Breeze.ParcelModels: ParcelDynamics
using Breeze.Thermodynamics: temperature, density, saturation_specific_humidity, PlanarLiquidSurface
using CloudMicrophysics
using CairoMakie

# ## Part 1: Dry adiabatic ascent
#
# A parcel rising through the troposphere experiences decreasing pressure,
# causing adiabatic expansion and cooling. Without moisture condensation,
# the parcel follows the dry adiabatic lapse rate Γd ≈ 9.8 K/km.

grid = RectilinearGrid(size=100, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
model = AtmosphereModel(grid; dynamics=ParcelDynamics())

reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                 surface_pressure = 101325,
                                 potential_temperature = 300)

# Set up environmental profiles with moisture that increases toward saturation with height
qᵗ₀ = 0.015    # Surface specific humidity [kg/kg]
Hq = 2500      # Humidity scale height [m]
qᵗ(z) = qᵗ₀ * exp(-z / Hq)

set!(model,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density,
     qᵗ = qᵗ,
     z = 0, w = 1)

simulation = Simulation(model; Δt=1.0, stop_time=30minutes)

# Record parcel trajectory
dry_times = Float64[]
dry_heights = Float64[]
dry_T_parcel = Float64[]
dry_T_environment = Float64[]
dry_supersaturations = Float64[]

function record_dry_state!(sim)
    state = sim.model.dynamics.state
    constants = sim.model.thermodynamic_constants
    push!(dry_times, sim.model.clock.time)
    push!(dry_heights, state.z)

    T = temperature(state.𝒰, constants)
    push!(dry_T_parcel, T)
    push!(dry_T_environment, interpolate((state.z,), sim.model.temperature))

    ρ = density(state.𝒰, constants)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    S = (state.𝒰.moisture_mass_fractions.vapor / qᵛ⁺) - 1
    push!(dry_supersaturations, S)
    return nothing
end

add_callback!(simulation, record_dry_state!, IterationInterval(1))
run!(simulation)

@info "Dry parcel reached" model.dynamics.state.z

# ## Part 2: Cloudy parcel with two-moment microphysics
#
# Now we simulate a moist parcel that rises through the lifting condensation level (LCL),
# triggering condensation and eventually precipitation. The two-moment scheme tracks
# both mass and number concentration, enabling realistic autoconversion rates
# that depend on droplet size.
#
# We initialize with a high droplet number concentration (as if the parcel
# formed on many CCN) to observe gradual cloud-to-rain conversion.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

microphysics = TwoMomentCloudMicrophysics(; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
cloudy_model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

# Use the same reference state
set!(cloudy_model,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density,
     qᵗ = qᵗ,
     z = 0, w = 1)

# Activate droplets at the surface: the parcel carries potential CCN that activate
# as it rises and supersaturation develops. We prescribe an initial droplet
# number concentration that represents aerosol activation.
nᶜˡ₀ = 100e6  # 100 droplets per mg of air [1/kg]
cloudy_model.dynamics.state.μ = (; ρqᶜˡ=0.0, ρnᶜˡ=1.2 * nᶜˡ₀, ρqʳ=0.0, ρnʳ=0.0)

cloudy_simulation = Simulation(cloudy_model; Δt=1.0, stop_time=120minutes)

# Storage for cloudy parcel trajectory
cloudy_times = Float64[]
cloudy_heights = Float64[]
cloudy_T = Float64[]
cloudy_qᵛ = Float64[]
cloudy_qᶜˡ = Float64[]
cloudy_qʳ = Float64[]
cloudy_nᶜˡ = Float64[]
cloudy_nʳ = Float64[]
cloudy_supersaturations = Float64[]

function record_cloudy_state!(sim)
    state = sim.model.dynamics.state
    constants = sim.model.thermodynamic_constants
    μ = state.μ
    ρ = state.ρ

    push!(cloudy_times, sim.model.clock.time)
    push!(cloudy_heights, state.z)

    T = temperature(state.𝒰, constants)
    push!(cloudy_T, T)

    q = state.𝒰.moisture_mass_fractions
    push!(cloudy_qᵛ, q.vapor)
    push!(cloudy_qᶜˡ, μ.ρqᶜˡ / ρ)
    push!(cloudy_qʳ, μ.ρqʳ / ρ)
    push!(cloudy_nᶜˡ, μ.ρnᶜˡ / ρ)
    push!(cloudy_nʳ, μ.ρnʳ / ρ)

    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    S = (q.vapor / qᵛ⁺) - 1
    push!(cloudy_supersaturations, S)
    return nothing
end

add_callback!(cloudy_simulation, record_cloudy_state!, IterationInterval(10))
run!(cloudy_simulation)

@info "Cloudy parcel reached" cloudy_model.dynamics.state.z

# ## Visualization
#
# We create a comprehensive figure showing:
# - Dry ascent: adiabatic cooling and approach to saturation
# - Cloudy ascent: condensation onset, cloud development, and precipitation formation

set_theme!(fontsize=14, linewidth=2.5)
fig = Figure(size=(1000, 800))

# Color palette
c_vapor = :dodgerblue
c_cloud = :lime
c_rain = :orangered
c_temp = :magenta
c_number = :darkorchid

## Row 1: Dry adiabatic ascent (condensed panels)
Label(fig[1, 1:2], "Dry adiabatic ascent", fontsize=16)

heights_km = dry_heights ./ 1000

ax1a = Axis(fig[2, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Adiabatic cooling")
lines!(ax1a, dry_T_parcel, heights_km; color=c_temp, label="Parcel")
lines!(ax1a, dry_T_environment, heights_km; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax1a; position=:lt)

ax1b = Axis(fig[2, 2];
    xlabel = "Height (km)",
    ylabel = "Supersaturation (%)",
    title = "Approach to saturation")
lines!(ax1b, heights_km, dry_supersaturations .* 100; color=c_vapor)
hlines!(ax1b, [0]; color=:gray, linestyle=:dash)
text!(ax1b, 1.2, -30; text="Saturation (S = 0)", fontsize=12, color=:gray)

## Row 2: Cloudy parcel - condensation and cloud formation
Label(fig[3, 1:2], "Cloudy ascent with two-moment microphysics", fontsize=16)

cloudy_heights_km = cloudy_heights ./ 1000

ax2a = Axis(fig[4, 1];
    xlabel = "Height (km)",
    ylabel = "Mixing ratio (g/kg)",
    title = "Moisture evolution")
lines!(ax2a, cloudy_heights_km, cloudy_qᵛ .* 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax2a, cloudy_heights_km, cloudy_qᶜˡ .* 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax2a, cloudy_heights_km, cloudy_qʳ .* 1000; color=c_rain, label="Rain qʳ")
axislegend(ax2a; position=:rt)

ax2b = Axis(fig[4, 2];
    xlabel = "Height (km)",
    ylabel = "Supersaturation (%)",
    title = "Supersaturation evolution")
lines!(ax2b, cloudy_heights_km, cloudy_supersaturations .* 100; color=c_vapor)
hlines!(ax2b, [0]; color=:gray, linestyle=:dash)

## Row 3: Number concentrations and precipitation
ax3a = Axis(fig[5, 1];
    xlabel = "Height (km)",
    ylabel = "Number (#/mg)",
    title = "Droplet number evolution")
lines!(ax3a, cloudy_heights_km, cloudy_nᶜˡ ./ 1e6; color=c_cloud, label="Cloud nᶜˡ")
lines!(ax3a, cloudy_heights_km, cloudy_nʳ ./ 1e3; color=c_rain, label="Rain nʳ × 10³")
axislegend(ax3a; position=:rt)

ax3b = Axis(fig[5, 2];
    xlabel = "Height (km)",
    ylabel = "Mean droplet mass (μg)",
    title = "Mean droplet size evolution")

# Compute mean droplet mass: q/n gives mass per droplet
mean_cloud_mass = cloudy_qᶜˡ ./ max.(cloudy_nᶜˡ, 1e-10) .* 1e12  # Convert to μg
mean_rain_mass = cloudy_qʳ ./ max.(cloudy_nʳ, 1e-10) .* 1e9  # Convert to μg (rain drops are larger)

# Mask out near-zero values
cloud_mask = cloudy_qᶜˡ .> 1e-10
rain_mask = cloudy_qʳ .> 1e-10

if any(cloud_mask)
    lines!(ax3b, cloudy_heights_km[cloud_mask], mean_cloud_mass[cloud_mask]; color=c_cloud, label="Cloud")
end
if any(rain_mask)
    lines!(ax3b, cloudy_heights_km[rain_mask], mean_rain_mass[rain_mask] ./ 1e3; color=c_rain, label="Rain (mg)")
end
axislegend(ax3b; position=:rt)

rowsize!(fig.layout, 1, Relative(0.04))
rowsize!(fig.layout, 3, Relative(0.04))

fig

# ## Discussion
#
# ### Dry adiabatic ascent (top row)
#
# The parcel cools at the dry adiabatic lapse rate (~9.8 K/km) as it rises.
# Supersaturation steadily increases because:
# 1. Temperature drops, reducing the saturation vapor pressure
# 2. Total moisture is conserved (in the absence of microphysics)
#
# The parcel approaches but does not reach saturation (S = 0%) because we
# stop before the lifting condensation level (LCL).
#
# ### Cloudy ascent (middle and bottom rows)
#
# With two-moment microphysics, the parcel exhibits rich cloud physics:
#
# 1. **Condensation onset**: As the parcel rises and cools, supersaturation
#    develops. The non-equilibrium scheme relaxes supersaturation by converting
#    vapor to cloud liquid, with a timescale of ~10 s.
#
# 2. **Cloud development**: Cloud liquid water content grows as condensation
#    continues. The droplet number concentration slowly decreases due to
#    self-collection (droplets merging).
#
# 3. **Precipitation formation**: When cloud droplets grow large enough,
#    autoconversion transfers mass from cloud to rain. The [Seifert and Beheng (2006)](@citet SeifertBeheng2006)
#    scheme derives autoconversion rates from the evolving size distribution:
#    - Fewer, larger droplets → faster autoconversion
#    - Many small droplets → suppressed precipitation
#
# 4. **Mean droplet mass**: The ratio q/n reveals how droplet size evolves.
#    Cloud droplets grow by condensation and self-collection. Rain drops
#    form via autoconversion and grow via accretion (collecting cloud droplets).
#
# This example illustrates the fundamental connection between aerosols and
# precipitation: more CCN → more cloud droplets → smaller drops → delayed
# rain formation (the cloud lifetime effect).
