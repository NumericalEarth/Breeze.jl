# # Buoyancy-driven rising parcels
#
# This example demonstrates **buoyancy-driven** `ParcelDynamics`, where
# each parcel carries a prognostic vertical velocity driven by the density
# difference between the parcel and its environment:
#
# ```math
# \frac{dw}{dt} = B = -g \frac{\rho_\mathrm{parcel} - \rho_\mathrm{env}}{\rho_\mathrm{env}}, \qquad
# \frac{dz}{dt} = w.
# ```
#
# We launch four parcels with an initial upward velocity of 1 m/s:
#
# 1. **Dry adiabatic parcel**: Cools at ~9.8 K/km, conserving potential temperature.
#    Buoyancy decelerates the parcel as it becomes denser than the environment.
#
# 2. **Cloudy parcel (one-moment microphysics)**: Condensation releases latent heat,
#    providing positive buoyancy that sustains or accelerates the ascent.
#    Uses [Morrison (2008)](@citet Morrison2008novel) non-equilibrium cloud formation.
#
# 3. **Cloudy parcel (Kessler microphysics)**: Same initial conditions with the
#    DCMIP2016 Kessler warm-rain scheme [Kessler1969](@citet).
#
# 4. **Cloudy parcel (two-moment microphysics)**: The [Seifert and Beheng (2006)](@cite
#    SeifertBeheng2006) two-moment scheme tracks mass and number concentration.
#    Cloud droplets form via aerosol activation [Abdul-Razzak and Ghan (2000)](@cite
#    AbdulRazzakGhan2000).
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
# A buoyancy-driven parcel launched upward at 1 m/s decelerates as adiabatic
# cooling makes it denser than the isentropic environment. Without moisture
# condensation, the parcel follows the dry adiabatic lapse rate Γd ≈ 9.8 K/km.

grid = RectilinearGrid(size=300, z=(0, 30kilometers), topology=(Flat, Flat, Bounded))
dynamics = ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity())
model = AtmosphereModel(grid; dynamics)

reference_state = ReferenceState(grid, model.thermodynamic_constants,
                                 surface_pressure = 101325,
                                 potential_temperature = 300)

# Set up environmental profiles with moisture that increases toward saturation with height
qᵗ₀ = 0.015    # Surface specific humidity [kg/kg]
Hq = 2500      # Humidity scale height [m]
qᵗ(z) = qᵗ₀ * exp(-z / Hq)

set!(model, qᵗ = qᵗ, z = 0, w_parcel = 1.0,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

simulation = Simulation(model; Δt=1, stop_time=30minutes)

# Store parcel snapshots: (time, height, thermodynamic state, density)
dry_snapshots = []

function record_dry_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(dry_snapshots, (; t, z=state.z, w=state.w, 𝒰=state.𝒰, ρ=state.ρ))
    return nothing
end

add_callback!(simulation, record_dry_state!, IterationInterval(1))
run!(simulation)

@info "Dry parcel reached" model.dynamics.state.z

# Extract time series from snapshots
constants = model.thermodynamic_constants
dry_t = [s.t for s in dry_snapshots]
dry_z = [s.z for s in dry_snapshots]
dry_w = [s.w for s in dry_snapshots]
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
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

microphysics = OneMomentCloudMicrophysics()
cloudy_model = AtmosphereModel(grid; dynamics=ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity()),
                               microphysics)

# Use the same reference state. The one-moment scheme initializes
# with zero cloud liquid and rain; condensation begins when supersaturation develops.
set!(cloudy_model, qᵗ = qᵗ, z = 0, w_parcel = 1.0,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

cloudy_simulation = Simulation(cloudy_model; Δt=1, stop_time=120minutes)

# Store cloudy parcel snapshots
cloudy_snapshots = []

function record_cloudy_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(cloudy_snapshots, (; t, z=state.z, w=state.w, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(cloudy_simulation, record_cloudy_state!, IterationInterval(10))
run!(cloudy_simulation)

@info "Cloudy parcel reached" cloudy_model.dynamics.state.z

# Extract time series from cloudy snapshots
cloudy_constants = cloudy_model.thermodynamic_constants
cloudy_t = [s.t for s in cloudy_snapshots]
cloudy_z = [s.z for s in cloudy_snapshots]
cloudy_w = [s.w for s in cloudy_snapshots]
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
kessler_constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
kessler_model = AtmosphereModel(grid; dynamics=ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity()),
                                microphysics, thermodynamic_constants=kessler_constants)

# Create reference state with the Tetens-based thermodynamic constants
kessler_reference_state = ReferenceState(grid, kessler_model.thermodynamic_constants,
                                         surface_pressure = 101325,
                                         potential_temperature = 300)

# Use the Kessler-specific reference state for initial conditions
set!(kessler_model, qᵗ = qᵗ, z = 0, w_parcel = 1.0,
     θ = kessler_reference_state.potential_temperature,
     p = kessler_reference_state.pressure,
     ρ = kessler_reference_state.density)

kessler_simulation = Simulation(kessler_model; Δt=1, stop_time=120minutes)

# Store Kessler parcel snapshots
kessler_snapshots = []

function record_kessler_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(kessler_snapshots, (; t, z=state.z, w=state.w, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(kessler_simulation, record_kessler_state!, IterationInterval(10))
run!(kessler_simulation)

@info "Kessler parcel reached" kessler_model.dynamics.state.z

# Extract time series from Kessler snapshots
kessler_constants = kessler_model.thermodynamic_constants
kessler_t = [s.t for s in kessler_snapshots]
kessler_z = [s.z for s in kessler_snapshots]
kessler_w = [s.w for s in kessler_snapshots]
kessler_T = [temperature(s.𝒰, kessler_constants) for s in kessler_snapshots]
kessler_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in kessler_snapshots]
kessler_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in kessler_snapshots]
kessler_qʳ = [s.μ.ρqʳ / s.ρ for s in kessler_snapshots]
kessler_S = [supersaturation(temperature(s.𝒰, kessler_constants), s.ρ,
                             s.𝒰.moisture_mass_fractions, kessler_constants,
                             PlanarLiquidSurface()) for s in kessler_snapshots]
nothing #hide

# Environmental temperature at each parcel height
kessler_Tₑ = [interpolate(s.z, kessler_model.temperature) for s in kessler_snapshots]
nothing #hide

# ## Part 4: Cloudy parcel with two-moment microphysics
#
# Finally, we simulate the same moist parcel using the [Seifert and Beheng (2006)](@cite
# SeifertBeheng2006) two-moment scheme. Unlike the one-moment schemes above, this tracks
# both mass *and* number concentration for cloud liquid and rain. Cloud droplets form via
# **aerosol activation** when the parcel becomes supersaturated — the default aerosol
# population (~100 cm⁻³ continental aerosol) provides the CCN.

twom_microphysics = TwoMomentCloudMicrophysics()
twom_model = AtmosphereModel(grid; dynamics=ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity()),
                             microphysics=twom_microphysics)

# Use the same reference state. Aerosol number is automatically initialized
# from the default aerosol distribution.
set!(twom_model, qᵗ = qᵗ, z = 0, w_parcel = 1.0,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

twom_simulation = Simulation(twom_model; Δt=0.1, stop_time=120minutes)

# Store two-moment parcel snapshots
twom_snapshots = []

function record_twom_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(twom_snapshots, (; t, z=state.z, w=state.w, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(twom_simulation, record_twom_state!, IterationInterval(100))
run!(twom_simulation)

@info "Two-moment parcel reached" twom_model.dynamics.state.z

# Extract time series from two-moment snapshots
twom_constants = twom_model.thermodynamic_constants
twom_t = [s.t for s in twom_snapshots]
twom_z = [s.z for s in twom_snapshots]
twom_w = [s.w for s in twom_snapshots]
twom_T = [temperature(s.𝒰, twom_constants) for s in twom_snapshots]
twom_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in twom_snapshots]
twom_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in twom_snapshots]
twom_qʳ = [s.μ.ρqʳ / s.ρ for s in twom_snapshots]
twom_nᶜˡ = [s.μ.ρnᶜˡ / s.ρ for s in twom_snapshots]
twom_nʳ = [s.μ.ρnʳ / s.ρ for s in twom_snapshots]
twom_nᵃ = [s.μ.ρnᵃ / s.ρ for s in twom_snapshots]
twom_S = [supersaturation(temperature(s.𝒰, twom_constants), s.ρ,
                          s.𝒰.moisture_mass_fractions, twom_constants,
                          PlanarLiquidSurface()) for s in twom_snapshots]
nothing #hide

# Environmental temperature at each parcel height
twom_Tₑ = [interpolate(s.z, twom_model.temperature) for s in twom_snapshots]
nothing #hide

# ## Visualization
#
# Each row shows temperature, velocity, supersaturation, and (for cloudy parcels)
# moisture or number evolution. The velocity panels reveal how buoyancy
# accelerates or decelerates the parcel — the signature of buoyancy-driven ascent.

set_theme!(fontsize=14, linewidth=2.5)
fig = Figure(size=(1600, 1200))
nothing #hide

# Color palette
c_vapor = :dodgerblue
c_cloud = :lime
c_rain = :orangered
c_temp = :magenta
c_vel = :black

## Row 1: Dry adiabatic ascent
Label(fig[1, 1:4], "Dry adiabatic ascent", fontsize=16)

ax1a = Axis(fig[2, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Adiabatic cooling")
lines!(ax1a, dry_T, dry_z / 1000; color=c_temp, label="Parcel")
lines!(ax1a, dry_Tₑ, dry_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax1a; position=:lb, backgroundcolor=(:white, 0.8))

ax1b = Axis(fig[2, 2];
    xlabel = "Velocity (m/s)",
    ylabel = "Height (km)",
    title = "Vertical velocity")
lines!(ax1b, dry_w, dry_z / 1000; color=c_vel)

ax1c = Axis(fig[2, 3];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Approach to saturation")
lines!(ax1c, dry_S, dry_z / 1000; color=c_vapor)
vlines!(ax1c, [0]; color=:gray, linestyle=:dash)

## Row 2: Cloudy parcel - one-moment microphysics
Label(fig[3, 1:4], "Cloudy ascent with one-moment microphysics", fontsize=16)

ax2a = Axis(fig[4, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax2a, cloudy_T, cloudy_z / 1000; color=c_temp, label="Parcel")
lines!(ax2a, cloudy_Tₑ, cloudy_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax2a; position=:lb, backgroundcolor=(:white, 0.8))

ax2b = Axis(fig[4, 2];
    xlabel = "Velocity (m/s)",
    ylabel = "Height (km)",
    title = "Vertical velocity")
lines!(ax2b, cloudy_w, cloudy_z / 1000; color=c_vel)

ax2c = Axis(fig[4, 3];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Supersaturation")
lines!(ax2c, cloudy_S, cloudy_z / 1000; color=c_vapor)
vlines!(ax2c, [0]; color=:gray, linestyle=:dash)

ax2d = Axis(fig[4, 4];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax2d, cloudy_qᵛ, cloudy_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax2d, cloudy_qᶜˡ, cloudy_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax2d, cloudy_qʳ, cloudy_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax2d; position=:rt, backgroundcolor=(:white, 0.8))

## Row 3: Cloudy parcel - Kessler microphysics
Label(fig[5, 1:4], "Cloudy ascent with Kessler microphysics", fontsize=16)

ax3a = Axis(fig[6, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax3a, kessler_T, kessler_z / 1000; color=c_temp, label="Parcel")
lines!(ax3a, kessler_Tₑ, kessler_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax3a; position=:lb, backgroundcolor=(:white, 0.8))

ax3b = Axis(fig[6, 2];
    xlabel = "Velocity (m/s)",
    ylabel = "Height (km)",
    title = "Vertical velocity")
lines!(ax3b, kessler_w, kessler_z / 1000; color=c_vel)

ax3c = Axis(fig[6, 3];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Supersaturation")
lines!(ax3c, kessler_S, kessler_z / 1000; color=c_vapor)
vlines!(ax3c, [0]; color=:gray, linestyle=:dash)

ax3d = Axis(fig[6, 4];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax3d, kessler_qᵛ, kessler_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax3d, kessler_qᶜˡ, kessler_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax3d, kessler_qʳ, kessler_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax3d; position=:rt, backgroundcolor=(:white, 0.8))

## Row 4: Cloudy parcel - two-moment microphysics
Label(fig[7, 1:4], "Cloudy ascent with two-moment microphysics", fontsize=16)

ax4a = Axis(fig[8, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax4a, twom_T, twom_z / 1000; color=c_temp, label="Parcel")
lines!(ax4a, twom_Tₑ, twom_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax4a; position=:lb, backgroundcolor=(:white, 0.8))

ax4b = Axis(fig[8, 2];
    xlabel = "Velocity (m/s)",
    ylabel = "Height (km)",
    title = "Vertical velocity")
lines!(ax4b, twom_w, twom_z / 1000; color=c_vel)

ax4c = Axis(fig[8, 3];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax4c, twom_qᵛ, twom_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax4c, twom_qᶜˡ, twom_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax4c, twom_qʳ, twom_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax4c; position=:rt, backgroundcolor=(:white, 0.8))

ax4d = Axis(fig[8, 4];
    xlabel = "Number concentration (1/kg)",
    ylabel = "Height (km)",
    xscale = log10,
    title = "Number concentration")

nᶜˡ_mask = twom_nᶜˡ .> 1e-3
nʳ_mask = twom_nʳ .> 1e-3
nᵃ_mask = twom_nᵃ .> 1e-3

if any(nᵃ_mask)
    lines!(ax4d, twom_nᵃ[nᵃ_mask], twom_z[nᵃ_mask] / 1000; color=:gray, label="Aerosol nᵃ")
end
if any(nᶜˡ_mask)
    lines!(ax4d, twom_nᶜˡ[nᶜˡ_mask], twom_z[nᶜˡ_mask] / 1000; color=c_cloud, label="Cloud nᶜˡ")
end
if any(nʳ_mask)
    lines!(ax4d, twom_nʳ[nʳ_mask], twom_z[nʳ_mask] / 1000; color=c_rain, label="Rain nʳ")
end
axislegend(ax4d; position=:rt, backgroundcolor=(:white, 0.8))

rowsize!(fig.layout, 1, Relative(0.03))
rowsize!(fig.layout, 3, Relative(0.03))
rowsize!(fig.layout, 5, Relative(0.03))
rowsize!(fig.layout, 7, Relative(0.03))

fig
save("rising_parcels.png", fig)

# ## Discussion
#
# ### Buoyancy-driven dry ascent (top row)
#
# The dry parcel launched at 1 m/s decelerates as adiabatic cooling makes it
# denser than the isentropic environment. The velocity panel shows the parcel
# slowing down, since without latent heating there is no sustained source of
# positive buoyancy. Supersaturation increases steadily as the temperature drops
# and the saturation vapor pressure falls below the (conserved) vapor content.
#
#
# ### Cloudy ascent with one-moment microphysics (second row)
#
# Condensation releases latent heat, warming the parcel relative to its
# environment and generating positive buoyancy. The velocity panel reveals
# the contrast with the dry case: the cloudy parcel accelerates once condensation
# begins. Key processes:
#
# 1. **Dry-to-moist transition**: Below the lifting condensation level the parcel
#    cools at ~9.8 K/km; above it, latent heating reduces the lapse rate to
#    ~6 K/km, visible as a slope change in the temperature panel.
#
# 2. **Non-equilibrium condensation**: Supersaturation relaxes toward zero on a
#    ~10 s timescale as vapor converts to cloud liquid.
#
# 3. **Precipitation formation**: Autoconversion transfers cloud mass to rain;
#    accretion then accelerates rain growth.
#
#
# ### Cloudy ascent with Kessler microphysics (third row)
#
# The DCMIP2016 Kessler scheme performs single-step saturation adjustment rather
# than relaxation, aiming to keep supersaturation at zero. The velocity evolution
# is qualitatively similar to the one-moment case — latent heating sustains
# positive buoyancy — but quantitative differences arise from the different
# condensation treatment.
#
# The Kessler scheme may show small negative supersaturation even when cloud is
# present. This is because the adjustment is computed at temperature T₀ but
# latent heating raises the temperature to T₁ > T₀, where the saturation vapor
# pressure is higher. An iterative approach (like `SaturationAdjustment`) would
# eliminate this residual, but the single-step method is efficient and the cloud
# formation is reasonable.
#
#
# ### Cloudy ascent with two-moment microphysics (bottom row)
#
# The [Seifert and Beheng (2006)](@cite SeifertBeheng2006) two-moment scheme
# tracks both mass and number concentration, enabling size-dependent process
# rates. Aerosol activation following [Abdul-Razzak and Ghan (2000)](@cite
# AbdulRazzakGhan2000) converts CCN into cloud droplets when the parcel becomes
# supersaturated. The number concentration panel reveals aerosol depletion,
# cloud droplet self-collection, autoconversion, and accretion — processes
# invisible to one-moment schemes.
#
# Comparing velocity across all four rows highlights the role of latent heating:
# dry parcels decelerate, while moist parcels with active microphysics can
# sustain or even increase their vertical velocity through buoyancy generated
# by condensational warming.
