# # Rising dry and cloudy parcels
#
# This example demonstrates the `ParcelDynamics` mode for `AtmosphereModel`,
# which enables Lagrangian simulations of air parcels moving through a
# prescribed background atmosphere. The example simulates three parcels:
#
# 1. **Ascending dry adiabatic parcel**: A rising parcel cools at ~9.8 K/km, conserving
#    potential temperature. Vapor increases toward saturation as temperature drops.
#
# 2. **Ascending cloudy parcel with CliMA one-moment microphysics)**: A moist parcel rises through the
#    lifting condensation level, forming cloud via condensation, then rain via
#    autoconversion. We use one-moment microphysics with non-equilibrium cloud
#    formation [Morrison2008novel](@citet) to track cloud liquid and rain mass.
#
# 3. **Ascending cloudy parcel with Kessler microphysics**: The same moist parcel, but using
#    the DCMIP2016 Kessler warm-rain scheme [Kessler1969](@citet). This scheme includes
#    autoconversion, accretion, saturation adjustment, and rain evaporation.
#
# The parcel model works with `AtmosphereModel`, using the standard `Simulation` interface.

using Oceananigans
using Oceananigans: interpolate
using Oceananigans.Units
using Breeze
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

set!(model, qᵗ = qᵗ, z = 0, w = 1,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

simulation = Simulation(model; Δt=1, stop_time=30minutes)

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
dry_Tₑ = [interpolate(s.z, model.temperature) for s in dry_snapshots]
nothing #hide

# ## Part 2: Cloudy parcel with one-moment microphysics
#
# Now we simulate a moist parcel that rises through the lifting condensation level (LCL),
# triggering condensation and eventually precipitation. The one-moment scheme tracks
# cloud liquid and rain mass, using non-equilibrium cloud formation where
# supersaturation relaxes toward zero on a characteristic timescale (~10 s).

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics

microphysics = OneMomentCloudMicrophysics()
cloudy_model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

# Use the same reference state. The one-moment scheme initializes
# with zero cloud liquid and rain; condensation begins when supersaturation develops.
set!(cloudy_model, qᵗ = qᵗ, z = 0, w = 1,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

cloudy_simulation = Simulation(cloudy_model; Δt=1, stop_time=120minutes)

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
cloudy_S = [supersaturation(temperature(s.𝒰, cloudy_constants), s.ρ,
                            s.𝒰.moisture_mass_fractions, cloudy_constants,
                            PlanarLiquidSurface()) for s in cloudy_snapshots]
nothing #hide

# Environmental temperature at each parcel height
cloudy_Tₑ = [interpolate(s.z, cloudy_model.temperature) for s in cloudy_snapshots]
nothing #hide

# ## Part 3: Cloudy parcel with Kessler microphysics
#
# Now we simulate the same moist parcel using the DCMIP2016 Kessler warm-rain scheme.
# This scheme includes autoconversion, accretion, saturation adjustment, and rain
# evaporation, following Klemp and Wilhelmson (1978). Unlike the one-moment scheme
# which uses a relaxation approach, Kessler performs direct saturation adjustment.
#
# Note: The DCMIP2016 Kessler scheme uses `TetensFormula` for saturation vapor
# pressure. We pass it explicitly via `thermodynamic_constants`.

using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula, ThermodynamicConstants

microphysics = DCMIP2016KesslerMicrophysics()
constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics, constants)

# Create reference state with the Tetens-based thermodynamic constants
reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                 surface_pressure = 101325,
                                 potential_temperature = 300)

# Use the Kessler-specific reference state for initial conditions
set!(model, qᵗ = qᵗ, z = 0, w = 1,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

kessler_simulation = Simulation(model; Δt=1, stop_time=120minutes)

# Store Kessler parcel snapshots
kessler_snapshots = []

function record_kessler_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(kessler_snapshots, (; t, z=state.z, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(kessler_simulation, record_kessler_state!, IterationInterval(10))
run!(kessler_simulation)

@info "Kessler parcel reached" model.dynamics.state.z

# Extract time series from Kessler snapshots
kessler_constants = model.thermodynamic_constants
kessler_t = [s.t for s in kessler_snapshots]
kessler_z = [s.z for s in kessler_snapshots]
kessler_T = [temperature(s.𝒰, kessler_constants) for s in kessler_snapshots]
kessler_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in kessler_snapshots]
kessler_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in kessler_snapshots]
kessler_qʳ = [s.μ.ρqʳ / s.ρ for s in kessler_snapshots]
kessler_S = [supersaturation(temperature(s.𝒰, kessler_constants), s.ρ,
                             s.𝒰.moisture_mass_fractions, kessler_constants,
                             PlanarLiquidSurface()) for s in kessler_snapshots]
nothing #hide

# Environmental temperature at each parcel height
kessler_Tₑ = [interpolate(s.z, model.temperature) for s in kessler_snapshots]
nothing #hide

# ## Visualization
#
# We create a figure showing:
# - Dry ascent: adiabatic cooling and approach to saturation
# - One-moment cloudy ascent: condensation onset, cloud development, and precipitation formation
# - Kessler cloudy ascent: the same physics with the DCMIP2016 Kessler scheme

set_theme!(fontsize=14, linewidth=2.5)
fig = Figure(size=(1200, 900))
nothing #hide

# Color palette
c_vapor = :dodgerblue
c_cloud = :lime
c_rain = :orangered
c_temp = :magenta

## Row 1: Dry adiabatic ascent
Label(fig[1, 1:3], "Dry adiabatic ascent", fontsize=16)

ax1a = Axis(fig[2, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Adiabatic cooling")
lines!(ax1a, dry_T, dry_z / 1000; color=c_temp, label="Parcel")
lines!(ax1a, dry_Tₑ, dry_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax1a; position=:lb, backgroundcolor=(:white, 0.8))

ax1b = Axis(fig[2, 2];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Approach to saturation")
lines!(ax1b, dry_S, dry_z / 1000; color=c_vapor)
vlines!(ax1b, [0]; color=:gray, linestyle=:dash)

## Row 2: Cloudy parcel - one-moment microphysics
Label(fig[3, 1:3], "Cloudy ascent with one-moment microphysics", fontsize=16)

ax2a = Axis(fig[4, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax2a, cloudy_T, cloudy_z / 1000; color=c_temp, label="Parcel")
lines!(ax2a, cloudy_Tₑ, cloudy_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax2a; position=:lb, backgroundcolor=(:white, 0.8))

ax2b = Axis(fig[4, 2];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Supersaturation evolution")
lines!(ax2b, cloudy_S, cloudy_z / 1000; color=c_vapor)
vlines!(ax2b, [0]; color=:gray, linestyle=:dash)

ax2c = Axis(fig[4, 3];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax2c, cloudy_qᵛ, cloudy_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax2c, cloudy_qᶜˡ, cloudy_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax2c, cloudy_qʳ, cloudy_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax2c; position=:rt, backgroundcolor=(:white, 0.8))

## Row 3: Cloudy parcel - Kessler microphysics
Label(fig[5, 1:3], "Cloudy ascent with Kessler microphysics", fontsize=16)

ax3a = Axis(fig[6, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax3a, kessler_T, kessler_z / 1000; color=c_temp, label="Parcel")
lines!(ax3a, kessler_Tₑ, kessler_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax3a; position=:lb, backgroundcolor=(:white, 0.8))

ax3b = Axis(fig[6, 2];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Supersaturation evolution")
lines!(ax3b, kessler_S, kessler_z / 1000; color=c_vapor)
vlines!(ax3b, [0]; color=:gray, linestyle=:dash)

ax3c = Axis(fig[6, 3];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax3c, kessler_qᵛ, kessler_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax3c, kessler_qᶜˡ, kessler_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax3c, kessler_qʳ, kessler_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax3c; position=:rt, backgroundcolor=(:white, 0.8))

rowsize!(fig.layout, 1, Relative(0.04))
rowsize!(fig.layout, 3, Relative(0.04))
rowsize!(fig.layout, 5, Relative(0.04))

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
#
# ### Cloudy ascent with one-moment microphysics (middle row)
#
# With one-moment non-equilibrium microphysics, the parcel exhibits key cloud physics:
#
# 1. **Dry to moist adiabatic transition**: Initially, the parcel cools at the
#    dry adiabatic lapse rate (~9.8 K/km). Once the parcel reaches saturation,
#    condensation releases latent heat, and the parcel transitions to the smaller
#    moist adiabatic lapse rate (~6 K/km). This is visible in the temperature
#    panel as a change in slope.
#
# 2. **Condensation onset**: As the parcel rises and cools, supersaturation
#    develops. The non-equilibrium scheme relaxes supersaturation by converting
#    vapor to cloud liquid, with a characteristic timescale (~10 s).
#
# 3. **Cloud development**: Cloud liquid water content grows as condensation
#    continues. The one-moment scheme tracks only mass, not number concentration.
#
# 4. **Precipitation formation**: Autoconversion transfers mass from cloud liquid
#    to rain based on a parameterized rate that depends on the cloud liquid
#    water content. Once rain forms, accretion (rain collecting cloud droplets)
#    accelerates precipitation development.
#
#
# ### Cloudy ascent with Kessler microphysics (bottom row)
#
# The DCMIP2016 Kessler scheme produces similar results to the one-moment scheme,
# but with some notable differences:
#
# 1. **Single-step saturation adjustment**: The Kessler scheme performs a single-step
#    saturation adjustment rather than the relaxation-based approach of the one-moment
#    scheme. This aims to keep supersaturation at zero when cloud is present.
#
# 2. **Similar precipitation formation**: Both schemes use the same fundamental
#    processes (autoconversion and accretion) to convert cloud water to rain,
#    though with different parameterizations.
#
# 3. **Rain evaporation**: The Kessler scheme explicitly includes rain evaporation
#    into subsaturated air, following Klemp and Wilhelmson (1978).
#
#
# ### Why supersaturation remains slightly negative in the Kessler scheme
#
# You may notice that the Kessler scheme shows small negative supersaturation
# even as cloud forms. This is expected behavior due to
# the interaction between the single-step saturation adjustment and the parcel
# model's energy-conserving thermodynamics.
#
# The explanation is as follows:
#
# 1. **Saturation adjustment at temperature T₀**: The Kessler scheme computes
#    how much vapor to condense based on the current temperature T₀.
#
# 2. **Latent heat release**: When condensation occurs, latent heat is released.
#    The parcel model conserves static energy, so the temperature automatically
#    increases to T₁ > T₀.
#
# 3. **Higher saturation vapor pressure at T₁**: At the new temperature T₁, the saturation vapor
#    pressure is higher than at T₀.
#
# 4. **Residual subsaturation**: The vapor was adjusted to match saturation at T₀,
#    but at T₁ it is now slightly below saturation.
#
# For exact equilibrium, an iterative approach (like `SaturationAdjustment`)
# would be needed, but the single-step method is computationally efficient and
# the resulting cloud formation is not too bad.
#
# This example demonstrates the basic thermodynamic and microphysical processes
# governing cloud formation in a rising air parcel, and shows how different
# microphysics schemes produce qualitatively similar but quantitatively different
# results.
