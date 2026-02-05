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
#    autoconversion. We use two-moment microphysics [SeifertBeheng2006](@citet)
#    to track both mass and number concentration.
#
# The parcel model works with `AtmosphereModel`, using the standard `Simulation` interface.

using Oceananigans
using Oceananigans: interpolate
using Oceananigans.Units
using Breeze
using Breeze: initial_aerosol_number
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

# Store parcel snapshots: (time, height, thermodynamic state, density)
dry_snapshots = []

function record_dry_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(dry_snapshots, (; t, z=state.z, 𝒰=state.𝒰, ρ=state.ρ))
    return nothing
end

add_callback!(simulation, record_dry_state!, IterationInterval(1))
run!(simulation)

@info "Dry parcel reached" model.dynamics.state.z

# Extract time series from snapshots
constants = model.thermodynamic_constants
dry_t = [s.t for s in dry_snapshots]
dry_z = [s.z for s in dry_snapshots]
dry_T = [temperature(s.𝒰, constants) for s in dry_snapshots]
dry_S = [supersaturation(temperature(s.𝒰, constants), s.ρ, s.𝒰.moisture_mass_fractions,
                         constants, PlanarLiquidSurface()) for s in dry_snapshots]
nothing #hide

# Environmental temperature at each parcel height
dry_Tₑ = [interpolate((s.z,), model.temperature) for s in dry_snapshots]
nothing #hide

# ## Part 2: Cloudy parcel with two-moment microphysics
#
# Now we simulate a moist parcel that rises through the lifting condensation level (LCL),
# triggering condensation and eventually precipitation. The two-moment scheme tracks
# both mass and number concentration, enabling realistic autoconversion rates
# that depend on droplet size.
#
# Cloud droplets form via **aerosol activation** when the parcel becomes supersaturated.
# The default aerosol population (~100 cm⁻³ continental aerosol) provides CCN that
# activate as the parcel rises. This physical process is handled automatically by
# `TwoMomentCloudMicrophysics`.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

microphysics = TwoMomentCloudMicrophysics()
cloudy_model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

# Use the same reference state. Cloud droplet number starts at zero and
# will grow via aerosol activation when the parcel becomes supersaturated.
set!(cloudy_model,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density,
     qᵗ = qᵗ,
     z = 0, w = 1)

# Initialize microphysical prognostics: cloud and rain start at zero,
# but aerosol number is set from the aerosol distribution parameters.
# The parcel carries available CCN that deplete as droplets activate.
Nᵃ₀ = initial_aerosol_number(microphysics)  # 100e6 m⁻³ for default continental aerosol
@info "Initial aerosol number concentration" Nᵃ₀
cloudy_model.dynamics.state.μ = (; ρqᶜˡ=0.0, ρnᶜˡ=0.0, ρqʳ=0.0, ρnʳ=0.0, ρnᵃ=Nᵃ₀)

cloudy_simulation = Simulation(cloudy_model; Δt=0.1, stop_time=120minutes)

# Store cloudy parcel snapshots
cloudy_snapshots = []

function record_cloudy_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(cloudy_snapshots, (; t, z=state.z, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(cloudy_simulation, record_cloudy_state!, IterationInterval(10))
run!(cloudy_simulation)

@info "Cloudy parcel reached" cloudy_model.dynamics.state.z

# Extract time series from cloudy snapshots
cloudy_constants = cloudy_model.thermodynamic_constants
cloudy_t = [s.t for s in cloudy_snapshots]
cloudy_z = [s.z for s in cloudy_snapshots]
cloudy_T = [temperature(s.𝒰, cloudy_constants) for s in cloudy_snapshots]
cloudy_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in cloudy_snapshots]
cloudy_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in cloudy_snapshots]
cloudy_qʳ = [s.μ.ρqʳ / s.ρ for s in cloudy_snapshots]
cloudy_nᶜˡ = [s.μ.ρnᶜˡ / s.ρ for s in cloudy_snapshots]
cloudy_nʳ = [s.μ.ρnʳ / s.ρ for s in cloudy_snapshots]
cloudy_nᵃ = [s.μ.ρnᵃ / s.ρ for s in cloudy_snapshots]
cloudy_S = [supersaturation(temperature(s.𝒰, cloudy_constants), s.ρ,
                            s.𝒰.moisture_mass_fractions, cloudy_constants,
                            PlanarLiquidSurface()) for s in cloudy_snapshots]
nothing #hide

# ## Visualization
#
# We create a comprehensive figure showing:
# - Dry ascent: adiabatic cooling and approach to saturation
# - Cloudy ascent: condensation onset, cloud development, and precipitation formation

set_theme!(fontsize=14, linewidth=2.5)
fig = Figure(size=(1000, 800))
nothing #hide

# Color palette
c_vapor = :dodgerblue
c_cloud = :lime
c_rain = :orangered
c_temp = :magenta

## Row 1: Dry adiabatic ascent
Label(fig[1, 1:2], "Dry adiabatic ascent", fontsize=16)

ax1a = Axis(fig[2, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Adiabatic cooling")
lines!(ax1a, dry_T, dry_z / 1000; color=c_temp, label="Parcel")
lines!(ax1a, dry_Tₑ, dry_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax1a; position=:lt)

ax1b = Axis(fig[2, 2];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Approach to saturation")
lines!(ax1b, dry_S, dry_z / 1000; color=c_vapor)
vlines!(ax1b, [0]; color=:gray, linestyle=:dash)

## Row 2: Cloudy parcel - condensation and cloud formation
Label(fig[3, 1:2], "Cloudy ascent with two-moment microphysics", fontsize=16)

ax2a = Axis(fig[4, 1];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax2a, cloudy_qᵛ, cloudy_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax2a, cloudy_qᶜˡ, cloudy_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax2a, cloudy_qʳ, cloudy_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax2a; position=:rt)

ax2b = Axis(fig[4, 2];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Supersaturation evolution")
lines!(ax2b, cloudy_S, cloudy_z / 1000; color=c_vapor)
vlines!(ax2b, [0]; color=:gray, linestyle=:dash)

## Row 3: Number concentrations and mean droplet size
# Filter out zero/near-zero values for log scale plotting
nᶜˡ_mask = cloudy_nᶜˡ .> 1e-3
nʳ_mask = cloudy_nʳ .> 1e-3
nᵃ_mask = cloudy_nᵃ .> 1e-3

z_max_km = maximum(cloudy_z) / 1000

ax3a = Axis(fig[5, 1];
    xlabel = "Number concentration (1/kg)",
    ylabel = "Height (km)",
    xscale = log10,
    limits = (nothing, nothing, 0, z_max_km),
    title = "Number concentration evolution")

if any(nᵃ_mask)
    lines!(ax3a, cloudy_nᵃ[nᵃ_mask], cloudy_z[nᵃ_mask] / 1000; color=:gray, label="Aerosol nᵃ")
end
if any(nᶜˡ_mask)
    lines!(ax3a, cloudy_nᶜˡ[nᶜˡ_mask], cloudy_z[nᶜˡ_mask] / 1000; color=c_cloud, label="Cloud nᶜˡ")
end
if any(nʳ_mask)
    lines!(ax3a, cloudy_nʳ[nʳ_mask], cloudy_z[nʳ_mask] / 1000; color=c_rain, label="Rain nʳ")
end
axislegend(ax3a; position=:rt)

# Mean droplet mass: q/n gives mass per droplet (kg)
mean_cloud_mass = cloudy_qᶜˡ ./ max.(cloudy_nᶜˡ, 1e-20)
mean_rain_mass = cloudy_qʳ ./ max.(cloudy_nʳ, 1e-20)

ax3b = Axis(fig[5, 2];
    xlabel = "Mean droplet mass (kg)",
    ylabel = "Height (km)",
    limits = (nothing, nothing, 0, z_max_km),
    title = "Mean droplet size evolution")

cloud_mask = cloudy_qᶜˡ .> 1e-10
rain_mask = cloudy_qʳ .> 1e-10

if any(cloud_mask)
    lines!(ax3b, mean_cloud_mass[cloud_mask], cloudy_z[cloud_mask] / 1000; color=c_cloud, label="Cloud")
end
if any(rain_mask)
    lines!(ax3b, mean_rain_mass[rain_mask], cloudy_z[rain_mask] / 1000; color=c_rain, label="Rain")
end
axislegend(ax3b; position=:rt)

rowsize!(fig.layout, 1, Relative(0.04))
rowsize!(fig.layout, 3, Relative(0.04))

fig
save("rising_parcels.png", fig)

# ## Discussion
#
# ### Dry adiabatic ascent (top row)
#
# The parcel cools at the dry adiabatic lapse rate (~9.8 K/km) as it rises.
# Supersaturation steadily increases because:
# 1. Temperature drops, reducing the saturation vapor pressure
# 2. Total moisture is conserved (in the absence of microphysics)
#
#
# ### Cloudy ascent (middle and bottom rows)
#
# With two-moment microphysics, the parcel exhibits rich cloud physics:
#
# 1. **Aerosol activation**: As the parcel rises and becomes supersaturated,
#    aerosol particles activate into cloud droplets. The activation rate depends
#    on updraft velocity and supersaturation, following the [AbdulRazzakGhan2000](@citet)
#    parameterization. Cloud droplet number increases from zero as activation occurs.
#
# 2. **Condensation**: Supersaturation drives vapor-to-liquid conversion.
#    The non-equilibrium scheme relaxes supersaturation toward zero with a
#    timescale of ~10 s, growing cloud droplets.
#
# 3. **Cloud development**: Cloud liquid water content grows as condensation
#    continues. The droplet number concentration may decrease due to
#    self-collection (droplets merging into larger drops).
#
# 4. **Precipitation formation**: When cloud droplets grow large enough,
#    autoconversion transfers mass from cloud to rain. The [SeifertBeheng2006](@citet)
#    scheme derives autoconversion rates from the evolving size distribution:
#    - Fewer, larger droplets → faster autoconversion
#    - Many small droplets → suppressed precipitation
#
# 5. **Mean droplet mass**: The ratio q/n reveals how droplet size evolves.
#    Cloud droplets grow by condensation and self-collection. Rain drops
#    form via autoconversion and grow via accretion (collecting cloud droplets).
#
# This example illustrates the fundamental connection between aerosols and
# precipitation: more CCN → more cloud droplets → smaller drops → delayed
# rain formation (the cloud lifetime effect).
