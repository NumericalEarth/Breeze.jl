# # Single column radiation: RRTMGP versus ecCKD
#
# Breeze has two full-spectrum radiative transfer formulations. [`ClearSkyOptics`](@ref)
# and [`AllSkyOptics`](@ref) solve the column with RRTMGP's correlated-``k`` tables
# through RRTMGP.jl; [`EcCKDOptics`](@ref) solves it with the ecCKD gas optics of
# [HoganMatricardi2022](@citet) through NumericalRadiation.jl. This example puts both on
# one column with an idealized tropical profile and compares them in three scenarios:
# clear sky with present-day CO₂ (420 ppm), clear sky with doubled CO₂ (840 ppm), and all
# sky with a liquid cloud between 1 and 2 km. The question is where the two formulations
# agree, where they do not, and by how much.

using Breeze
using Oceananigans.Units
using CairoMakie
using Printf
using Dates

using NCDatasets  # For the RRTMGP and ecCKD lookup tables
using RRTMGP
using NumericalRadiation: NumericalRadiation  # Loads the ecCKD extension; NumericalRadiation exports its own ThermodynamicConstants

# ## Grid
#
# We create a single column spanning 20 km with 64 layers at a particular place.

Nz = 64
λ, φ = -76.13, 39.48

grid = RectilinearGrid(size=Nz, x=λ, y=φ, z=(0, 20kilometers),
                       topology=(Flat, Flat, Bounded))

# ## Thermodynamics and a tropical reference state
#
# The column holds an idealized tropical temperature profile: 300 K at the surface, cooling
# at 6.5 K/km to a 196 K tropopause at 16 km, and isothermal above. The anelastic reference
# state supplies the pressure and density that the radiation solvers see, so it must follow
# the same profile: a 300 K dry adiabat would reach 107 K at 20 km and put half the real
# mass of air (and of ozone and CO₂) in the stratosphere. `ReferenceState` integrates the
# hydrostatic balance for a potential temperature profile `θᵣ(z)`, which we form from the
# analytic hydrostatic pressure of the temperature profile.

surface_temperature = 300
tropopause_temperature = 196
lapse_rate = 6.5e-3  # K/m
Tᵢ(z) = max(surface_temperature - lapse_rate * z, tropopause_temperature)

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity  # J/(kg·K)

base_pressure = 101325
standard_pressure = 1e5
z_tropopause = (surface_temperature - tropopause_temperature) / lapse_rate
p_tropopause = base_pressure * (tropopause_temperature / surface_temperature)^(g / (Rᵈ * lapse_rate))

# Hydrostatic pressure of the temperature profile: a power law under the constant lapse rate
# and an exponential in the isothermal stratosphere
pᵢ(z) = z < z_tropopause ? base_pressure * (Tᵢ(z) / surface_temperature)^(g / (Rᵈ * lapse_rate)) :
                           p_tropopause * exp(-g * (z - z_tropopause) / (Rᵈ * tropopause_temperature))

θᵢ(z) = Tᵢ(z) * (standard_pressure / pᵢ(z))^(Rᵈ / cᵖᵈ)

reference_state = ReferenceState(grid, constants; base_pressure, standard_pressure,
                                 potential_temperature = θᵢ)

dynamics = AnelasticDynamics(reference_state)

# ## Six radiative transfer models
#
# Every model shares the surface properties and the solar constant, and the two all-sky
# models share the effective radii of the cloud particles. The solar zenith angle is
# computed from the model clock and the grid location.

surface = (; surface_temperature, surface_emissivity = 0.98, surface_albedo = 0.1, solar_constant = 1361)
effective_radii = (liquid_effective_radius = ConstantRadiusParticles(10e-6),
                   ice_effective_radius = ConstantRadiusParticles(30e-6))

high_co2_atmosphere = BackgroundAtmosphere(CO₂ = 840e-6)

# The three RRTMGP models treat the 20 km grid top as the top of the atmosphere.

rrtmgp = (clear_sky = RadiativeTransferModel(grid, ClearSkyOptics(), constants; surface...),
          high_co2  = RadiativeTransferModel(grid, ClearSkyOptics(), constants; surface...,
                                             background_atmosphere = high_co2_atmosphere),
          all_sky   = RadiativeTransferModel(grid, AllSkyOptics(), constants; surface..., effective_radii...))

# By default an ecCKD model extends the radiation column above the grid top with a
# [`ColumnExtension`](@ref). Here we switch it off with `column_extension = nothing`, so
# that both formulations solve the same problem and every difference below is a difference
# in the optics. The all-sky ecCKD model takes its cloud optics from the Mie droplet and
# Baum ice scattering tables selected by [`CloudScatteringTables`](@ref).

ecckd = (clear_sky = RadiativeTransferModel(grid, EcCKDOptics(), constants; surface...,
                                            column_extension = nothing),
         high_co2  = RadiativeTransferModel(grid, EcCKDOptics(), constants; surface...,
                                            background_atmosphere = high_co2_atmosphere,
                                            column_extension = nothing),
         all_sky   = RadiativeTransferModel(grid, EcCKDOptics(clouds = CloudScatteringTables()), constants;
                                            surface..., effective_radii..., column_extension = nothing))

formulations = (; rrtmgp, ecckd)

# ## Atmosphere models
#
# Build one atmosphere model per radiation model, with saturation adjustment microphysics.

clock = Clock(time=DateTime(1950, 11, 1, 17, 0, 0))  # local noon at λ = -76°
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

build_model(radiation) = AtmosphereModel(grid; clock, dynamics, microphysics, radiation)
models = map(formulation -> map(build_model, formulation), formulations)

# ## Initial condition: the tropical profile with a cloud
#
# We set the temperature to the tropical profile `Tᵢ` of the reference state. The relative
# humidity is 80 % throughout, except for a layer between 1 and 2 km that we supersaturate
# slightly so that saturation adjustment produces a cloud for the all-sky scenario.

ℋᵢ(z) = ifelse(1kilometer < z < 2kilometers, 1.05, 0.8)

for formulation in models, model in formulation
    set!(model; T=Tᵢ, ℋ=ℋᵢ)
end

# ## The atmospheric state and the fluxes
#
# After `set!`, the radiation has been computed. We build Fields and
# AbstractOperations to visualize the atmospheric state and radiative fluxes.

T = models.rrtmgp.clear_sky.temperature
qᵛ = specific_humidity(models.rrtmgp.clear_sky)
ℋ = RelativeHumidityField(models.rrtmgp.clear_sky)
qˡ = models.rrtmgp.all_sky.microphysical_fields.qˡ

# The net flux is the sum of all four components. Because downwelling fluxes are
# stored with a negative sign, the sum is already "up minus down". The upwelling
# shortwave carries the radiation scattered back to space by air and clouds and
# reflected by the surface, so leaving it out would misstate the net flux, and with
# it the heating rate, wherever the shortwave scatters or reflects.

net_longwave(radiation) = radiation.upwelling_longwave_flux + radiation.downwelling_longwave_flux
net_shortwave(radiation) = radiation.upwelling_shortwave_flux + radiation.downwelling_shortwave_flux
net_flux(radiation) = net_longwave(radiation) + net_shortwave(radiation)

# The `RadiativeTransferModel` computes the heating tendency `Q = -dF_net/dz` (W/m³) from
# the radiative flux divergence. We convert to K/day using `dT/dt = Q / (ρᵣ cₚ)` with the
# reference density and dry-air heat capacity; the model tendencies use the local mixture
# heat capacity and are therefore slightly different in moist and cloudy layers.

ρᵣ = reference_state.density
to_K_per_day = 86400 / cᵖᵈ
heating_rate(radiation) = to_K_per_day * radiation.flux_divergence / ρᵣ

set_theme!(fontsize=14, linewidth=2.5)

# Format altitude ticks in km (but keep internal units in meters).
z_ticks_km = 0:5:20
z_ticks_m = ((z_ticks_km .* 1000), string.(z_ticks_km))

# Colors distinguish the scenarios; line styles the formulations
# (solid for RRTMGP, dashed for ecCKD).
colors = (clear_sky = :dodgerblue, high_co2 = :orangered, all_sky = :lime)
scenario_labels = (clear_sky = "Clear sky (420 ppm)", high_co2 = "2×CO₂ (840 ppm)", all_sky = "All sky (cloudy)")
linestyles = (rrtmgp = :solid, ecckd = :dash)

fig = Figure(size=(1800, 800), fontsize=14)
nothing #hide

# Atmospheric state panels (top row)
ax_T = Axis(fig[1, 1]; xlabel="Temperature (K)", ylabel="Altitude (km)",
            yticks=z_ticks_m, xticks=200:25:300)
ax_q = Axis(fig[1, 2]; xlabel="Specific humidity (kg/kg)", yticks=z_ticks_m)
ax_H = Axis(fig[1, 3]; xlabel="Relative humidity (%)", yticks=z_ticks_m)
ax_ql = Axis(fig[1, 4]; xlabel="Cloud liquid (g/kg)", yticks=z_ticks_m)

# Radiation panels (bottom row) - one per component
ax_lw_up = Axis(fig[2, 1]; xlabel="LW ↑ (W/m²)", ylabel="Altitude (km)", yticks=z_ticks_m)
ax_lw_dn = Axis(fig[2, 2]; xlabel="LW ↓ (W/m²)", yticks=z_ticks_m)
ax_sw_dn = Axis(fig[2, 3]; xlabel="SW ↓ (W/m²)", yticks=z_ticks_m)
ax_sw_up = Axis(fig[2, 4]; xlabel="SW ↑ (W/m²)", yticks=z_ticks_m)
ax_net = Axis(fig[2, 5]; xlabel="Net flux (W/m²)", yticks=z_ticks_m)

# Hide y-axis decorations on inner panels
[hideydecorations!(ax, grid=false) for ax in (ax_q, ax_H, ax_ql, ax_lw_dn, ax_sw_dn, ax_sw_up, ax_net)]

# Atmospheric state
lines!(ax_T, T; color=:gray30)
lines!(ax_q, qᵛ; color=:gray30)
lines!(ax_H, 100ℋ; color=:gray30)
lines!(ax_ql, 1000qˡ; color=:lime)  # Convert to g/kg

# Downwelling fluxes are negative, so we negate them for display.
function plot_fluxes!(formulation, scenario)
    rtm = formulations[formulation][scenario]
    style = (color=colors[scenario], linestyle=linestyles[formulation])
    lines!(ax_lw_up,  rtm.upwelling_longwave_flux;    style...)
    lines!(ax_lw_dn, -rtm.downwelling_longwave_flux;  style...)
    lines!(ax_sw_dn, -rtm.downwelling_shortwave_flux; style...)
    lines!(ax_sw_up,  rtm.upwelling_shortwave_flux;   style...)
    lines!(ax_net,    net_flux(rtm);                  style...)
    return nothing
end

for formulation in keys(formulations), scenario in keys(rrtmgp)
    plot_fluxes!(formulation, scenario)
end

scenario_handles = [LineElement(color=c, linewidth=3) for c in colors]
formulation_handles = [LineElement(color=:gray50, linewidth=3, linestyle=s) for s in linestyles]
Legend(fig[1, 5], [scenario_handles, formulation_handles],
       [collect(scenario_labels), ["RRTMGP", "ecCKD"]], ["Scenario", "Formulation"];
       framevisible=false, tellwidth=false)

fig

# The dashed curves sit on the solid ones almost everywhere: at this scale the two
# formulations are indistinguishable in all three scenarios. The comparison lives in the
# differences.

# ## The differences
#
# The ecCKD minus RRTMGP profiles of the four flux components and of the heating rate,
# for each scenario, show where the formulations part.

fig_diff = Figure(size=(1800, 450), fontsize=14)
nothing #hide

difference_axes = (lw_up   = Axis(fig_diff[1, 1]; xlabel="Δ LW ↑ (W/m²)", ylabel="Altitude (km)", yticks=z_ticks_m),
                   lw_dn   = Axis(fig_diff[1, 2]; xlabel="Δ LW ↓ (W/m²)", yticks=z_ticks_m),
                   sw_dn   = Axis(fig_diff[1, 3]; xlabel="Δ SW ↓ (W/m²)", yticks=z_ticks_m),
                   sw_up   = Axis(fig_diff[1, 4]; xlabel="Δ SW ↑ (W/m²)", yticks=z_ticks_m),
                   heating = Axis(fig_diff[1, 5]; xlabel="Δ heating rate (K/day)", yticks=z_ticks_m))

[hideydecorations!(ax, grid=false) for ax in Tuple(difference_axes)[2:end]]

function plot_differences!(scenario)
    e, r = ecckd[scenario], rrtmgp[scenario]
    style = (color=colors[scenario], label=scenario_labels[scenario])
    lines!(difference_axes.lw_up,    e.upwelling_longwave_flux    - r.upwelling_longwave_flux;    style...)
    lines!(difference_axes.lw_dn,   -e.downwelling_longwave_flux  + r.downwelling_longwave_flux;  style...)
    lines!(difference_axes.sw_dn,   -e.downwelling_shortwave_flux + r.downwelling_shortwave_flux; style...)
    lines!(difference_axes.sw_up,    e.upwelling_shortwave_flux   - r.upwelling_shortwave_flux;   style...)
    lines!(difference_axes.heating,  heating_rate(e) - heating_rate(r);                           style...)
    return nothing
end

foreach(plot_differences!, keys(rrtmgp))

for ax in difference_axes
    vlines!(ax, 0; color=:gray50, linestyle=:dash, linewidth=1)
    xlims!(ax, -6, 6)
end

xlims!(difference_axes.heating, -3, 3)
axislegend(difference_axes.sw_up, position=:lb)

fig_diff

# ## Boundary fluxes and the CO₂ forcing
#
# Four numbers per scenario summarize the figures: the upwelling longwave flux at the top of
# the column (the outgoing longwave radiation, OLR), the downwelling longwave and shortwave
# fluxes at the surface, and the shortwave reflected out of the top. The 2×CO₂ forcing is
# the reduction in the net upward flux between the 420 ppm and 840 ppm scenarios, at the top
# of the column and at the surface, split into its longwave and shortwave parts.

top = Nz + 1

print_header(title) = @printf("%-22s %8s %8s %16s\n", title, "RRTMGP", "ecCKD", "ecCKD − RRTMGP")
print_row(label, r, e) = @printf("  %-20s %8.1f %8.1f %16.1f\n", label, r, e, e - r)

for scenario in keys(rrtmgp)
    r, e = rrtmgp[scenario], ecckd[scenario]
    print_header(scenario_labels[scenario])
    print_row("OLR",          r.upwelling_longwave_flux[1, 1, top],    e.upwelling_longwave_flux[1, 1, top])
    print_row("LW ↓ surface", -r.downwelling_longwave_flux[1, 1, 1],   -e.downwelling_longwave_flux[1, 1, 1])
    print_row("SW ↓ surface", -r.downwelling_shortwave_flux[1, 1, 1],  -e.downwelling_shortwave_flux[1, 1, 1])
    print_row("SW ↑ top",     r.upwelling_shortwave_flux[1, 1, top],   e.upwelling_shortwave_flux[1, 1, top])
end

forcing(net, formulation, k) = net(formulation.clear_sky)[1, 1, k] - net(formulation.high_co2)[1, 1, k]

print_header("2×CO₂ forcing (W/m²)")
print_row("longwave, top",      forcing(net_longwave, rrtmgp, top),  forcing(net_longwave, ecckd, top))
print_row("longwave, surface",  forcing(net_longwave, rrtmgp, 1),    forcing(net_longwave, ecckd, 1))
print_row("shortwave, top",     forcing(net_shortwave, rrtmgp, top), forcing(net_shortwave, ecckd, top))
print_row("shortwave, surface", forcing(net_shortwave, rrtmgp, 1),   forcing(net_shortwave, ecckd, 1))

# Where the two formulations solve the same problem, they agree. In clear sky the ecCKD
# OLR is 1.4 W/m² higher than RRTMGP's (257.9 versus 256.5), the surface receives 1.8 W/m²
# more longwave (411.8 versus 410.0) and 0.3 W/m² more sunlight, and 0.9 W/m² less
# shortwave is reflected out of the top (96.2 versus 97.0): the spread expected between two
# correlated-``k`` models fit independently to line-by-line references. The CO₂ forcing
# agrees to 0.1 W/m²: doubling CO₂ cuts the OLR by 4.1 W/m² (RRTMGP) and 4.0 W/m² (ecCKD),
# adds 0.8 to 0.9 W/m² of longwave at the surface, and takes 0.9 W/m² of sunlight from the
# surface through the near-infrared bands of CO₂, so that the net surface forcing nearly
# vanishes in both. Below 17 km the clear-sky heating rates differ by 0.17 K/day (root mean
# square), most in the lowest cell, which cools 1.0 K/day with ecCKD and is neutral with
# RRTMGP.
#
# The cloud is where they part, because the cloud optics differ in kind: RRTMGP's all-sky
# lookup tables against the Mie and Baum scattering tables mapped onto the ecCKD g-points.
# The same liquid water reflects 3.5 W/m² more sunlight with ecCKD (493.0 versus
# 489.5 W/m²), emits 2.9 W/m² more longwave to the surface (447.3 versus 444.4) and lets
# 3.1 W/m² more longwave out of the top, twice the clear-sky difference. In the cloud-top
# cell at 1.7 km, where longwave cooling outruns shortwave heating even at local noon, the
# net cooling is 11.0 K/day with RRTMGP and 13.6 K/day with ecCKD. The other place the
# heating rates disagree is the top cell: with the atmosphere ending at 20 km, both
# formulations absorb the ultraviolet that ozone above would have absorbed in that one cell,
# RRTMGP at 6.1 K/day and ecCKD at 7.5 K/day.

# ## The column extension
#
# One thing the RRTMGP formulation cannot represent is the atmosphere above the grid. An
# ecCKD model built with the default [`ColumnExtension`](@ref) continues the column above
# 20 km with 40 layers up to 65 km following the U.S. Standard Atmosphere, so the top face
# receives longwave radiation from the stratosphere, the sunlight entering the grid has
# already crossed the ozone layer, and the ozone heating that the models above put in their
# top cell is spread over the extension where it belongs.

extended_radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants; surface...)
extended_model = build_model(extended_radiation)
set!(extended_model; T=Tᵢ, ℋ=ℋᵢ)

@printf("LW ↓ top face:     %6.1f W/m² with the extension, %6.1f without\n",
        -extended_radiation.downwelling_longwave_flux[1, 1, top],
        -ecckd.clear_sky.downwelling_longwave_flux[1, 1, top])
@printf("SW ↓ surface:      %6.1f W/m² with the extension, %6.1f without\n",
        -extended_radiation.downwelling_shortwave_flux[1, 1, 1],
        -ecckd.clear_sky.downwelling_shortwave_flux[1, 1, 1])
@printf("Top cell heating:  %6.2f K/day with the extension, %6.2f without\n",
        heating_rate(extended_radiation)[1, 1, Nz],
        heating_rate(ecckd.clear_sky)[1, 1, Nz])

# The extension supplies 10.2 W/m² of downwelling longwave on the top face, where the
# no-extension model (and any RRTMGP model) has none; it absorbs 8.6 W/m² of the sunlight
# above the grid, so the surface receives 563.7 W/m² instead of 572.3; and the ozone heating
# of the top cell falls from 7.5 K/day to 2.4 K/day.
