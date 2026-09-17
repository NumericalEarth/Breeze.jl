# # Single column radiation (gray, clear-sky, and all-sky)
#
# This example sets up a single-column atmospheric model with an idealized
# temperature and moisture profile. We compute radiative fluxes using RRTMGP's
# gray atmosphere solver with the optical thickness parameterization
# by [OGormanSchneider2008](@citet), and compare against clear-sky full-spectrum
# gas optics, doubled CO₂, and all-sky (cloudy) radiation. Each full-spectrum
# case is solved twice: with RRTMGP's correlated-``k`` tables, and with the ecCKD
# gas optics of [HoganMatricardi2022](@citet) through NumericalRadiation.jl.

using Breeze
using Oceananigans.Units
using CairoMakie
using Printf

using NCDatasets  # For the RRTMGP and ecCKD lookup tables
using RRTMGP
using NumericalRadiation: NumericalRadiation  # Loads the ecCKD extension; NumericalRadiation exports its own ThermodynamicConstants

# ## Grid and thermodynamics
#
# We create a single column spanning 20 km with 64 layers at a particular place.

Nz = 64
λ, φ = -76.13, 39.48

grid = RectilinearGrid(size=Nz, x=λ, y=φ, z=(0, 20kilometers),
                       topology=(Flat, Flat, Bounded))

# Set up the thermodynamic constants and reference state.
surface_temperature = 300
constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 base_pressure = 101325,
                                 potential_temperature = surface_temperature)

dynamics = AnelasticDynamics(reference_state)

# ## Radiative transfer models
#
# We create a gray radiative transfer model using the [OGormanSchneider2008](@citet)
# optical thickness parameterization. The solar zenith angle is computed from the
# model clock and grid location. We also create clear-sky full-spectrum models
# with present-day and doubled CO₂ concentrations.

using Dates

gray_radiation = RadiativeTransferModel(grid, GrayOptics(), constants;
                                        surface_temperature,
                                        surface_emissivity = 0.98,
                                        surface_albedo = 0.1,
                                        solar_constant = 1361)        # W/m²

# Clear-sky with default CO₂ (~420 ppm)
clear_sky_radiation = RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                             surface_temperature,
                                             surface_emissivity = 0.98,
                                             surface_albedo = 0.1,
                                             solar_constant = 1361)    # W/m²

# Clear-sky with doubled CO₂ (~840 ppm) to show the radiative forcing effect
high_co2_atmosphere = BackgroundAtmosphere(CO₂ = 840e-6)
high_co2_radiation = RadiativeTransferModel(grid, ClearSkyOptics(), constants;
                                            background_atmosphere = high_co2_atmosphere,
                                            surface_temperature,
                                            surface_emissivity = 0.98,
                                            surface_albedo = 0.1,
                                            solar_constant = 1361)    # W/m²

# All-sky with cloud scattering optics
all_sky_radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                           surface_temperature,
                                           surface_emissivity = 0.98,
                                           surface_albedo = 0.1,
                                           solar_constant = 1361,
                                           liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                           ice_effective_radius = ConstantRadiusParticles(30e-6))

# ## ecCKD radiative transfer models
#
# [`EcCKDOptics`](@ref) solves the same three full-spectrum cases with the ecCKD
# gas optics (32 longwave and 32 shortwave g-points) through NumericalRadiation.jl.
# One difference from the RRTMGP models above is built in: an ecCKD model extends
# the radiation column above the top of the grid with a [`ColumnExtension`](@ref)
# (by default 40 layers up to 65 km, following the U.S. Standard Atmosphere), so
# that the fluxes at the top of the domain include the stratosphere above it,
# whereas the RRTMGP models treat the 20 km grid top as the top of the atmosphere.
# All-sky ecCKD radiation uses the Mie droplet and Baum ice scattering tables
# selected by [`CloudScatteringTables`](@ref).

ecckd_clear_sky_radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                   surface_temperature,
                                                   surface_emissivity = 0.98,
                                                   surface_albedo = 0.1,
                                                   solar_constant = 1361)

ecckd_high_co2_radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                  background_atmosphere = high_co2_atmosphere,
                                                  surface_temperature,
                                                  surface_emissivity = 0.98,
                                                  surface_albedo = 0.1,
                                                  solar_constant = 1361)

ecckd_all_sky_radiation = RadiativeTransferModel(grid, EcCKDOptics(clouds = CloudScatteringTables()), constants;
                                                 surface_temperature,
                                                 surface_emissivity = 0.98,
                                                 surface_albedo = 0.1,
                                                 solar_constant = 1361,
                                                 liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                                 ice_effective_radius = ConstantRadiusParticles(30e-6))

# We collect the seven radiation models in a `NamedTuple` so that the rest of the
# example can treat them uniformly.

radiation = (gray            = gray_radiation,
             clear_sky       = clear_sky_radiation,
             high_co2        = high_co2_radiation,
             all_sky         = all_sky_radiation,
             ecckd_clear_sky = ecckd_clear_sky_radiation,
             ecckd_high_co2  = ecckd_high_co2_radiation,
             ecckd_all_sky   = ecckd_all_sky_radiation)

# ## Atmosphere models
#
# Build one atmosphere model per radiation model, with saturation adjustment microphysics.

clock = Clock(time=DateTime(1950, 11, 1, 17, 0, 0))  # local noon at λ = -76°
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

models = map(radiation) do radiation
    AtmosphereModel(grid; clock, dynamics, microphysics, radiation)
end

# ## Initial condition: idealized tropical profile with a cloud
#
# We prescribe a tropical-like temperature profile: 300 K at the surface, cooling at
# 6.5 K/km to a 196 K tropopause at 16 km, and isothermal above. The relative humidity
# is 80 % throughout, except for a layer between 1 and 2 km that we supersaturate
# slightly so that saturation adjustment produces a cloud for the all-sky comparison.

Tᵢ(z) = max(300 - 6.5e-3 * z, 196)
ℋᵢ(z) = ifelse(1kilometer < z < 2kilometers, 1.05, 0.8)

foreach(model -> set!(model; T=Tᵢ, ℋ=ℋᵢ), models)

# ## Visualization
#
# After `set!`, the radiation has been computed. We build Fields and
# AbstractOperations to visualize the atmospheric state and radiative fluxes.

T = models.gray.temperature
pᵣ = reference_state.pressure
qᵛ = specific_humidity(models.gray)
ℋ = RelativeHumidityField(models.gray)

# The net flux is the sum of all four components. Because downwelling fluxes are
# stored with a negative sign, the sum is already "up minus down". The upwelling
# shortwave carries the radiation scattered back to space by air and clouds and
# reflected by the surface, so leaving it out would misstate the net flux, and with
# it the heating rate, wherever the shortwave scatters or reflects. Gray optics uses a
# non-scattering shortwave solver, so its upwelling shortwave is identically zero.

net_flux(radiation) = radiation.upwelling_longwave_flux + radiation.downwelling_longwave_flux +
                      radiation.upwelling_shortwave_flux + radiation.downwelling_shortwave_flux

# Get cloud liquid for visualization
qˡ = models.all_sky.microphysical_fields.qˡ

set_theme!(fontsize=14, linewidth=2.5)

# Format altitude ticks in km (but keep internal units in meters).
z_ticks_km = 0:5:20
z_ticks_m = ((z_ticks_km .* 1000), string.(z_ticks_km))

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

# Colors distinguish the radiation scenarios; line styles the gas optics
# (solid for RRTMGP, dashed for ecCKD).
c_gray = :black
c_clear = :dodgerblue
c_2xco2 = :orangered
c_allsky = :lime

styles = (gray            = (color=c_gray,   linestyle=:solid),
          clear_sky       = (color=c_clear,  linestyle=:solid),
          high_co2        = (color=c_2xco2,  linestyle=:solid),
          all_sky         = (color=c_allsky, linestyle=:solid),
          ecckd_clear_sky = (color=c_clear,  linestyle=:dash),
          ecckd_high_co2  = (color=c_2xco2,  linestyle=:dash),
          ecckd_all_sky   = (color=c_allsky, linestyle=:dash))

# Downwelling fluxes are negative, so we negate them for display.
function plot_fluxes!(name)
    rtm = radiation[name]
    style = styles[name]
    lines!(ax_lw_up,  rtm.upwelling_longwave_flux;    style...)
    lines!(ax_lw_dn, -rtm.downwelling_longwave_flux;  style...)
    lines!(ax_sw_dn, -rtm.downwelling_shortwave_flux; style...)
    lines!(ax_sw_up,  rtm.upwelling_shortwave_flux;   style...)
    lines!(ax_net,    net_flux(rtm);                  style...)
    return nothing
end

foreach(plot_fluxes!, keys(radiation))

# Legend
scenario_handles = [LineElement(color=c, linewidth=3) for c in (c_gray, c_clear, c_2xco2, c_allsky)]
scenario_labels = ["Gray", "Clear-sky (420 ppm)", "2×CO₂ (840 ppm)", "All-sky (cloudy)"]
optics_handles = [LineElement(color=:gray50, linewidth=3, linestyle=s) for s in (:solid, :dash)]
optics_labels = ["RRTMGP", "ecCKD"]
Legend(fig[1, 5], [scenario_handles, optics_handles], [scenario_labels, optics_labels], ["Scenario", "Gas optics"];
       framevisible=false, tellwidth=false)

fig

# ## Fluxes at the surface and at the top of the domain
#
# Four numbers per model summarize the figure: the upwelling longwave flux at the
# top of the domain (the outgoing longwave radiation, OLR, for the RRTMGP models),
# the downwelling longwave and shortwave fluxes at the surface, and the shortwave
# reflected out of the top of the domain.

function print_boundary_fluxes(name)
    rtm = radiation[name]
    @printf("%-16s  LW ↑ top: %6.1f  LW ↓ top: %5.1f  LW ↓ surface: %6.1f  SW ↓ surface: %6.1f  SW ↑ top: %6.1f  W/m²\n",
            name,
            rtm.upwelling_longwave_flux[1, 1, Nz+1],
            -rtm.downwelling_longwave_flux[1, 1, Nz+1],
            -rtm.downwelling_longwave_flux[1, 1, 1],
            -rtm.downwelling_shortwave_flux[1, 1, 1],
            rtm.upwelling_shortwave_flux[1, 1, Nz+1])
    return nothing
end

foreach(print_boundary_fluxes, keys(radiation))

# The two gas optics agree closely where they solve the same problem. In clear sky
# the upwelling longwave flux at 20 km differs by 1.5 W/m² (RRTMGP 256.2, ecCKD 257.7),
# the downwelling longwave at the surface by 2.4 W/m² (410.4 versus 412.8), the
# downwelling shortwave at the surface by 4 W/m² (570.2 versus 566.3) and the reflected
# shortwave at 20 km by 1.8 W/m² (97.9 versus 96.1) -- the spread expected between two
# independent correlated-k models. Doubling CO₂ reduces the upwelling longwave flux
# at 20 km by 4.1 W/m² in both. With the cloud, both models reflect about 490 W/m² back
# out of the domain and pass 93 W/m² to the surface, and the surface receives 3 W/m²
# more longwave from the ecCKD model's cloud than from RRTMGP's.
#
# The one systematic difference is the column extension. The ecCKD models receive
# 6 W/m² of longwave radiation on the top face from the stratosphere above 20 km, which
# the RRTMGP models (whose atmosphere ends at 20 km) do not, and which grows to 7.1 W/m²
# with doubled CO₂; that back-radiation is why the ecCKD 2×CO₂ forcing at the top of the
# domain (4.8 W/m²) exceeds RRTMGP's (4.2 W/m²) even though the two outgoing-longwave
# reductions agree. The shortwave side of the same effect appears in the heating rates
# below.

# ## Heating rates
#
# The `RadiativeTransferModel` automatically computes the heating tendency
# `Q = -dF_net/dz` (W/m³) from the radiative flux divergence. We convert to K/day
# using `dT/dt = Q / (ρᵣ cₚ)`. For this compact comparison we use the reference
# density and dry-air heat capacity; the model tendencies use the local mixture
# heat capacity and are therefore slightly different in moist and cloudy layers.

# Convert W/m³ → K/day: Q / (ρᵣ cᵖᵈ) × 86400
ρᵣ = reference_state.density
cᵖᵈ = constants.dry_air.heat_capacity  # J/(kg·K)
to_K_per_day = 86400 / cᵖᵈ

heating_rate(radiation) = to_K_per_day * radiation.flux_divergence / ρᵣ

fig2 = Figure(size=(800, 500), fontsize=14)

ax_Q = Axis(fig2[1, 1]; xlabel="Heating rate (K/day)", ylabel="Altitude (km)",
            yticks=z_ticks_m, title="Radiative heating rates")

heating_labels = (gray            = "Gray",
                  clear_sky       = "Clear-sky (420 ppm), RRTMGP",
                  high_co2        = "2×CO₂ (840 ppm), RRTMGP",
                  all_sky         = "All-sky (cloudy), RRTMGP",
                  ecckd_clear_sky = "Clear-sky (420 ppm), ecCKD",
                  ecckd_high_co2  = "2×CO₂ (840 ppm), ecCKD",
                  ecckd_all_sky   = "All-sky (cloudy), ecCKD")

plot_heating!(name) = lines!(ax_Q, heating_rate(radiation[name]); label=heating_labels[name], styles[name]...)
foreach(plot_heating!, keys(radiation))

vlines!(ax_Q, 0; color=:gray50, linestyle=:dash, linewidth=1)
axislegend(ax_Q, position=:lt)

fig2

# Below 17 km the clear-sky heating rates of the two gas optics differ by 0.19 K/day
# (root mean square), most in the lowest cell where the ecCKD model cools 1 K/day
# faster, and the cloud-top longwave cooling reaches -10.6 K/day with RRTMGP and
# -13.1 K/day with ecCKD. The profiles part in the top two kilometers: the RRTMGP
# models absorb the ultraviolet sunlight that ozone would have absorbed higher up in
# their top cells (6.6 K/day at 19.8 km), whereas the ecCKD models absorb it in the
# column extension above the grid and heat the top cell by 1.5 K/day.
