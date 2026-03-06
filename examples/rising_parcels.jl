# # Rising dry and cloudy parcels
#
# This example demonstrates the `ParcelDynamics` mode for `AtmosphereModel`,
# which enables Lagrangian simulations of air parcels moving through a
# prescribed background atmosphere. The example simulates seven parcels — five with
# prescribed vertical velocity and two with buoyancy-driven ascent:
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
# 4. **Ascending cloudy parcel with two-moment microphysics**: The moist parcel again, now
#    using the [Seifert and Beheng (2006)](@cite SeifertBeheng2006) two-moment scheme, which
#    tracks both mass and number concentration for cloud liquid and rain. Cloud droplets
#    form via aerosol activation using the [Abdul-Razzak and Ghan (2000)](@cite
#    AbdulRazzakGhan2000) scheme when the parcel becomes supersaturated.
#
# 5. **Buoyancy-driven cloudy ascent**: The same two-moment microphysics as case 4, but with
#    `PrognosticVerticalVelocity` — the parcel's vertical velocity responds to the
#    buoyancy force dw/dt = −g(ρ_parcel − ρ_env)/ρ_env, rather than being prescribed.
#
# 6. **Ascending cloudy parcel with P3 microphysics**: The Predicted Particle Properties
#    (P3) scheme — a three-moment ice scheme with continuously predicted particle
#    properties (rime fraction, rime density, liquid fraction). Unlike traditional
#    schemes that use discrete ice categories, P3 uses a single ice category with
#    evolving characteristics. Cloud liquid condensation uses a relaxation-to-saturation
#    approach.
#
# 7. **Buoyancy-driven P3 ascent**: The P3 scheme with `PrognosticVerticalVelocity`,
#    directly comparable to case 5 to show how the ice microphysics coupling
#    affects the buoyancy-driven dynamics.
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

grid = RectilinearGrid(size=700, z=(0, 7kilometers), topology=(Flat, Flat, Bounded))
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
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

microphysics = OneMomentCloudMicrophysics()
cloudy_model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

# Use the same reference state. The one-moment scheme initializes
# with zero cloud liquid and rain; condensation begins when supersaturation develops.
set!(cloudy_model, qᵗ = qᵗ, z = 0, w = 1,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

cloudy_simulation = Simulation(cloudy_model; Δt=1, stop_time=90minutes)

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
kessler_constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
kessler_model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics,
                                thermodynamic_constants=kessler_constants)

# Create reference state with the Tetens-based thermodynamic constants
kessler_reference_state = ReferenceState(grid, kessler_model.thermodynamic_constants,
                                         surface_pressure = 101325,
                                         potential_temperature = 300)

# Use the Kessler-specific reference state for initial conditions
set!(kessler_model, qᵗ = qᵗ, z = 0, w = 1,
     θ = kessler_reference_state.potential_temperature,
     p = kessler_reference_state.pressure,
     ρ = kessler_reference_state.density)

kessler_simulation = Simulation(kessler_model; Δt=1, stop_time=90minutes)

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

@info "Kessler parcel reached" kessler_model.dynamics.state.z

# Extract time series from Kessler snapshots
kessler_constants = kessler_model.thermodynamic_constants
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
twom_model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics=twom_microphysics)

# Use the same reference state. Aerosol number is automatically initialized
# from the default aerosol distribution.
set!(twom_model, qᵗ = qᵗ, z = 0, w = 1,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

twom_simulation = Simulation(twom_model; Δt=1, stop_time=90minutes)

# Store two-moment parcel snapshots
twom_snapshots = []

function record_twom_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(twom_snapshots, (; t, z=state.z, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(twom_simulation, record_twom_state!, IterationInterval(100))
run!(twom_simulation)

@info "Two-moment parcel reached" twom_model.dynamics.state.z

# Extract time series from two-moment snapshots
twom_constants = twom_model.thermodynamic_constants
twom_t = [s.t for s in twom_snapshots]
twom_z = [s.z for s in twom_snapshots]
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

# ## Part 5: Buoyancy-driven cloudy ascent
#
# In all four cases above, the vertical velocity is prescribed: the parcel
# rises at a constant 1 m/s regardless of its thermodynamic state.
# Now we switch to **buoyancy-driven** ascent using `PrognosticVerticalVelocity`,
# where the parcel carries a prognostic vertical velocity driven by buoyancy:
# dw/dt = −g(ρ_parcel − ρ_env)/ρ_env. We reuse the two-moment microphysics
# so the reader can directly compare rows 4 and 5.

buoyant_dynamics = ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity())
buoyant_model = AtmosphereModel(grid; dynamics=buoyant_dynamics,
                                microphysics=TwoMomentCloudMicrophysics())

set!(buoyant_model, qᵗ = qᵗ, z = 0, w_parcel = 1.0,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

buoyant_simulation = Simulation(buoyant_model; Δt=1, stop_time=5minutes)

buoyant_snapshots = []

function record_buoyant_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(buoyant_snapshots, (; t, z=state.z, w=state.w, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(buoyant_simulation, record_buoyant_state!, IterationInterval(10))
run!(buoyant_simulation)

@info "Buoyancy-driven parcel reached" buoyant_model.dynamics.state.z

# Extract time series
buoyant_constants = buoyant_model.thermodynamic_constants
buoyant_t = [s.t for s in buoyant_snapshots]
buoyant_z = [s.z for s in buoyant_snapshots]
buoyant_w = [s.w for s in buoyant_snapshots]
buoyant_T = [temperature(s.𝒰, buoyant_constants) for s in buoyant_snapshots]
buoyant_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in buoyant_snapshots]
buoyant_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in buoyant_snapshots]
buoyant_qʳ = [s.μ.ρqʳ / s.ρ for s in buoyant_snapshots]
buoyant_nᶜˡ = [s.μ.ρnᶜˡ / s.ρ for s in buoyant_snapshots]
buoyant_nʳ = [s.μ.ρnʳ / s.ρ for s in buoyant_snapshots]
buoyant_nᵃ = [s.μ.ρnᵃ / s.ρ for s in buoyant_snapshots]
nothing #hide

# Environmental temperature at each parcel height
buoyant_Tₑ = [interpolate(s.z, buoyant_model.temperature) for s in buoyant_snapshots]
nothing #hide

# ## Part 6: P3 microphysics — prescribed velocity
#
# The Predicted Particle Properties (P3) scheme represents ice with a single
# category whose properties (rime fraction, rime density) evolve continuously.
# P3 tracks 9 microphysical prognostic variables: cloud liquid mass; rain mass
# and number; ice mass, number, rime mass, rime volume, reflectivity, and
# liquid-on-ice mass. Cloud condensation uses relaxation-to-saturation with a
# psychrometric correction factor.

p3_model = AtmosphereModel(grid; dynamics=ParcelDynamics(),
                           microphysics=PredictedParticlePropertiesMicrophysics())

set!(p3_model, qᵗ = qᵗ, z = 0, w = 1,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

p3_simulation = Simulation(p3_model; Δt=1, stop_time=90minutes)

p3_snapshots = []

function record_p3_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(p3_snapshots, (; t, z=state.z, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(p3_simulation, record_p3_state!, IterationInterval(10))
run!(p3_simulation)

@info "P3 parcel reached" p3_model.dynamics.state.z

p3_constants = p3_model.thermodynamic_constants
p3_t = [s.t for s in p3_snapshots]
p3_z = [s.z for s in p3_snapshots]
p3_T = [temperature(s.𝒰, p3_constants) for s in p3_snapshots]
p3_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in p3_snapshots]
p3_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in p3_snapshots]
p3_qʳ = [s.μ.ρqʳ / s.ρ for s in p3_snapshots]
p3_qⁱ = [s.μ.ρqⁱ / s.ρ for s in p3_snapshots]
p3_S = [supersaturation(temperature(s.𝒰, p3_constants), s.ρ,
                        s.𝒰.moisture_mass_fractions, p3_constants,
                        PlanarLiquidSurface()) for s in p3_snapshots]
nothing #hide

p3_Tₑ = [interpolate(s.z, p3_model.temperature) for s in p3_snapshots]
nothing #hide

# ## Part 7: P3 microphysics — buoyancy-driven ascent
#
# The same P3 scheme, but now with prognostic vertical velocity driven by
# buoyancy. This tests the full coupling: condensation → latent heating →
# positive buoyancy → acceleration → more condensation.

p3_buoyant_dynamics = ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity())
p3_buoyant_model = AtmosphereModel(grid; dynamics=p3_buoyant_dynamics,
                                   microphysics=PredictedParticlePropertiesMicrophysics())

set!(p3_buoyant_model, qᵗ = qᵗ, z = 0, w_parcel = 1.0,
     θ = reference_state.potential_temperature,
     p = reference_state.pressure,
     ρ = reference_state.density)

p3_buoyant_simulation = Simulation(p3_buoyant_model; Δt=1, stop_time=5minutes)

p3_buoyant_snapshots = []

function record_p3_buoyant_state!(sim)
    state = sim.model.dynamics.state
    t = sim.model.clock.time
    push!(p3_buoyant_snapshots, (; t, z=state.z, w=state.w, ρ=state.ρ, 𝒰=state.𝒰, μ=state.μ))
    return nothing
end

add_callback!(p3_buoyant_simulation, record_p3_buoyant_state!, IterationInterval(10))
run!(p3_buoyant_simulation)

@info "P3 buoyancy-driven parcel reached" p3_buoyant_model.dynamics.state.z

p3b_constants = p3_buoyant_model.thermodynamic_constants
p3b_t = [s.t for s in p3_buoyant_snapshots]
p3b_z = [s.z for s in p3_buoyant_snapshots]
p3b_w = [s.w for s in p3_buoyant_snapshots]
p3b_T = [temperature(s.𝒰, p3b_constants) for s in p3_buoyant_snapshots]
p3b_qᵛ = [s.𝒰.moisture_mass_fractions.vapor for s in p3_buoyant_snapshots]
p3b_qᶜˡ = [s.μ.ρqᶜˡ / s.ρ for s in p3_buoyant_snapshots]
p3b_qʳ = [s.μ.ρqʳ / s.ρ for s in p3_buoyant_snapshots]
p3b_qⁱ = [s.μ.ρqⁱ / s.ρ for s in p3_buoyant_snapshots]
nothing #hide

p3b_Tₑ = [interpolate(s.z, p3_buoyant_model.temperature) for s in p3_buoyant_snapshots]
nothing #hide

# ## Visualization
#
# We create a figure showing all seven regimes:
# - Dry ascent: adiabatic cooling and approach to saturation
# - One-moment cloudy ascent: condensation onset, cloud development, and precipitation formation
# - Kessler cloudy ascent: the same physics with the DCMIP2016 Kessler scheme
# - Two-moment cloudy ascent: mass and number evolution with aerosol activation
# - Buoyancy-driven ascent: two-moment microphysics with prognostic vertical velocity
# - P3 cloudy ascent: three-moment ice with cloud condensation and warm rain
# - Buoyancy-driven P3 ascent: P3 microphysics with prognostic vertical velocity

set_theme!(fontsize=14, linewidth=2.5)
fig = Figure(size=(1200, 1550))
nothing #hide

# Color palette
c_vapor = :dodgerblue
c_cloud = :lime
c_rain = :orangered
c_ice = :cyan
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

## Row 4: Cloudy parcel - two-moment microphysics
Label(fig[7, 1:3], "Cloudy ascent with two-moment microphysics", fontsize=16)

ax4a = Axis(fig[8, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax4a, twom_T, twom_z / 1000; color=c_temp, label="Parcel")
lines!(ax4a, twom_Tₑ, twom_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax4a; position=:lb, backgroundcolor=(:white, 0.8))

ax4b = Axis(fig[8, 2];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax4b, twom_qᵛ, twom_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax4b, twom_qᶜˡ, twom_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax4b, twom_qʳ, twom_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax4b; position=:rt, backgroundcolor=(:white, 0.8))

ax4c = Axis(fig[8, 3];
    xlabel = "Number concentration (1/kg)",
    ylabel = "Height (km)",
    xscale = log10,
    title = "Number concentration")

nᶜˡ_mask = twom_nᶜˡ .> 1e-3
nʳ_mask = twom_nʳ .> 1e-3
nᵃ_mask = twom_nᵃ .> 1e-3

if any(nᵃ_mask)
    lines!(ax4c, twom_nᵃ[nᵃ_mask], twom_z[nᵃ_mask] / 1000; color=:gray, label="Aerosol nᵃ")
end
if any(nᶜˡ_mask)
    lines!(ax4c, twom_nᶜˡ[nᶜˡ_mask], twom_z[nᶜˡ_mask] / 1000; color=c_cloud, label="Cloud nᶜˡ")
end
if any(nʳ_mask)
    lines!(ax4c, twom_nʳ[nʳ_mask], twom_z[nʳ_mask] / 1000; color=c_rain, label="Rain nʳ")
end
axislegend(ax4c; position=:rt, backgroundcolor=(:white, 0.8))

## Row 5: Buoyancy-driven cloudy ascent
Label(fig[9, 1:4], "Buoyancy-driven ascent with two-moment microphysics", fontsize=16)

ax5a = Axis(fig[10, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax5a, buoyant_T, buoyant_z / 1000; color=c_temp, label="Parcel")
lines!(ax5a, buoyant_Tₑ, buoyant_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax5a; position=:lb, backgroundcolor=(:white, 0.8))

ax5b = Axis(fig[10, 2];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax5b, buoyant_qᵛ, buoyant_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax5b, buoyant_qᶜˡ, buoyant_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax5b, buoyant_qʳ, buoyant_z / 1000; color=c_rain, label="Rain qʳ")
axislegend(ax5b; position=:rt, backgroundcolor=(:white, 0.8))

ax5c = Axis(fig[10, 3];
    xlabel = "Number concentration (1/kg)",
    ylabel = "Height (km)",
    xscale = log10,
    title = "Number concentration")

bnᶜˡ_mask = buoyant_nᶜˡ .> 1e-3
bnʳ_mask = buoyant_nʳ .> 1e-3
bnᵃ_mask = buoyant_nᵃ .> 1e-3

if any(bnᵃ_mask)
    lines!(ax5c, buoyant_nᵃ[bnᵃ_mask], buoyant_z[bnᵃ_mask] / 1000; color=:gray, label="Aerosol nᵃ")
end
if any(bnᶜˡ_mask)
    lines!(ax5c, buoyant_nᶜˡ[bnᶜˡ_mask], buoyant_z[bnᶜˡ_mask] / 1000; color=c_cloud, label="Cloud nᶜˡ")
end
if any(bnʳ_mask)
    lines!(ax5c, buoyant_nʳ[bnʳ_mask], buoyant_z[bnʳ_mask] / 1000; color=c_rain, label="Rain nʳ")
end
axislegend(ax5c; position=:rt, backgroundcolor=(:white, 0.8))

ax5d = Axis(fig[10, 4];
    xlabel = "Velocity (m/s)",
    ylabel = "Height (km)",
    title = "Vertical velocity")
lines!(ax5d, buoyant_w, buoyant_z / 1000; color=:black)

## Row 6: P3 microphysics - prescribed velocity
Label(fig[11, 1:3], "Cloudy ascent with P3 microphysics", fontsize=16)

ax6a = Axis(fig[12, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax6a, p3_T, p3_z / 1000; color=c_temp, label="Parcel")
lines!(ax6a, p3_Tₑ, p3_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax6a; position=:lb, backgroundcolor=(:white, 0.8))

ax6b = Axis(fig[12, 2];
    xlabel = "Supersaturation",
    ylabel = "Height (km)",
    title = "Supersaturation evolution")
lines!(ax6b, p3_S, p3_z / 1000; color=c_vapor)
vlines!(ax6b, [0]; color=:gray, linestyle=:dash)

ax6c = Axis(fig[12, 3];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax6c, p3_qᵛ, p3_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax6c, p3_qᶜˡ, p3_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax6c, p3_qʳ, p3_z / 1000; color=c_rain, label="Rain qʳ")
lines!(ax6c, p3_qⁱ, p3_z / 1000; color=c_ice, label="Ice qⁱ")
axislegend(ax6c; position=:rt, backgroundcolor=(:white, 0.8))

## Row 7: P3 buoyancy-driven ascent
Label(fig[13, 1:4], "Buoyancy-driven ascent with P3 microphysics", fontsize=16)

ax7a = Axis(fig[14, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Temperature evolution")
lines!(ax7a, p3b_T, p3b_z / 1000; color=c_temp, label="Parcel")
lines!(ax7a, p3b_Tₑ, p3b_z / 1000; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax7a; position=:lb, backgroundcolor=(:white, 0.8))

ax7b = Axis(fig[14, 2];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Moisture evolution")
lines!(ax7b, p3b_qᵛ, p3b_z / 1000; color=c_vapor, label="Vapor qᵛ")
lines!(ax7b, p3b_qᶜˡ, p3b_z / 1000; color=c_cloud, label="Cloud qᶜˡ")
lines!(ax7b, p3b_qʳ, p3b_z / 1000; color=c_rain, label="Rain qʳ")
lines!(ax7b, p3b_qⁱ, p3b_z / 1000; color=c_ice, label="Ice qⁱ")
axislegend(ax7b; position=:rt, backgroundcolor=(:white, 0.8))

ax7c = Axis(fig[14, 3];
    xlabel = "Velocity (m/s)",
    ylabel = "Height (km)",
    title = "Vertical velocity")
lines!(ax7c, p3b_w, p3b_z / 1000; color=:black)

for ax in [ax2a, ax2b, ax2c,
           ax3a, ax3b, ax3c,
           ax4a, ax4b, ax4c,
           ax5a, ax5b, ax5c, ax5d,
           ax6a, ax6b, ax6c,
           ax7a, ax7b, ax7c]
    ylims!(ax, 0, 5)
end

rowsize!(fig.layout, 1, Relative(0.02))
rowsize!(fig.layout, 3, Relative(0.02))
rowsize!(fig.layout, 5, Relative(0.02))
rowsize!(fig.layout, 7, Relative(0.02))
rowsize!(fig.layout, 9, Relative(0.02))
rowsize!(fig.layout, 11, Relative(0.02))
rowsize!(fig.layout, 13, Relative(0.02))

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
# ### Cloudy ascent with one-moment microphysics (second row)
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
# ### Cloudy ascent with Kessler microphysics (third row)
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
#
# ### Cloudy ascent with two-moment microphysics (fourth row)
#
# The [Seifert and Beheng (2006)](@cite SeifertBeheng2006) two-moment scheme
# adds a crucial dimension: number concentration. This enables physically-based
# precipitation formation rates that depend on droplet size:
#
# 1. **Aerosol activation**: As the parcel rises and becomes supersaturated,
#    aerosol particles activate into cloud droplets following the
#    [Abdul-Razzak and Ghan (2000)](@cite AbdulRazzakGhan2000) parameterization.
#    Cloud droplet number increases from zero as activation occurs, while aerosol
#    number decreases.
#
# 2. **Condensation with number tracking**: Like the one-moment scheme,
#    supersaturation drives vapor-to-liquid conversion. But the two-moment scheme
#    also knows how many droplets share the condensed water, enabling size-aware
#    process rates.
#
# 3. **Number concentration panel**: The right panel reveals processes invisible
#    to one-moment schemes: aerosol depletion by activation, and
#    collision-coalescence processes that reshape the size distribution —
#    cloud droplet self-collection, autoconversion of cloud to rain,
#    accretion of cloud by rain, rain self-collection, and rain breakup.
#
# ### Buoyancy-driven cloudy ascent (fifth row)
#
# Row 5 repeats the two-moment microphysics from row 4 but switches
# from prescribed to **prognostic vertical velocity**. The parcel starts at
# 1 m/s and its velocity evolves according to the buoyancy force
# B = −g(ρ_parcel − ρ_env)/ρ_env. Below the lifting condensation level,
# the parcel decelerates just as in a dry adiabatic ascent. Once condensation
# begins, latent heat release warms the parcel relative to its environment,
# generating positive buoyancy that accelerates the ascent. The velocity panel
# shows this transition directly — a key signature of buoyancy-driven dynamics
# that is invisible when velocity is prescribed. Comparing the temperature and
# moisture panels between rows 4 and 5 reveals how the ascent rate shapes
# the cloud and precipitation evolution.
#
# ### P3 microphysics (sixth row)
#
# The Predicted Particle Properties (P3) scheme uses a single ice category
# with continuously predicted properties — a fundamentally different approach
# from traditional bulk microphysics schemes. In this warm-rain parcel test,
# the key behavior to verify is:
#
# 1. **Relaxation-to-saturation condensation**: P3 uses the same non-equilibrium
#    condensation approach as the one-moment scheme, but with a psychrometric
#    correction factor that accounts for latent heating during phase change.
#    Supersaturation should remain small and positive above the LCL.
#
# 2. **Warm-rain processes**: Autoconversion and accretion convert cloud liquid
#    to rain, similar to the other schemes. The P3 autoconversion uses a
#    Kessler-like parameterization with a threshold cloud water content.
#
# 3. **Temperature evolution**: The transition from dry to moist adiabatic
#    lapse rate should be visible, confirming that latent heat release from
#    condensation is correctly captured by the static energy formulation.
#
# At these warm temperatures (T > 273 K), ice processes are inactive.
# Ice nucleation, deposition, riming, and aggregation activate at colder
# temperatures in deeper convection scenarios.
#
# ### Buoyancy-driven P3 ascent (seventh row)
#
# Comparing rows 6 and 7 demonstrates the same prescribed-vs-prognostic
# velocity contrast as rows 4 and 5, but with P3 microphysics. The
# buoyancy-velocity coupling is qualitatively the same: condensation →
# latent heating → buoyancy → acceleration → more condensation. Differences
# between the two-moment and P3 velocity evolution reflect the different
# condensation parameterizations and warm-rain process rates.
#
# This example demonstrates the basic thermodynamic and microphysical processes
# governing cloud formation in a rising air parcel, and shows how different
# microphysics schemes produce qualitatively similar but quantitatively different
# results. The buoyancy-driven cases further demonstrate the coupling between
# microphysics and dynamics: condensation → latent heating → buoyancy →
# acceleration → more condensation.
