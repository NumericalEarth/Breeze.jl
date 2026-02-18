# # Single column radiation (gray, clear-sky, and all-sky)
#
# This example sets up a single-column atmospheric model with an idealized
# temperature and moisture profile. We compute radiative fluxes using RRTMGP's
# gray atmosphere solver with the optical thickness parameterization
# by [OGormanSchneider2008](@citet), and compare against clear-sky full-spectrum
# gas optics, doubled CO₂, and all-sky (cloudy) radiation.

using Breeze
using Oceananigans.Units
using CairoMakie

using NCDatasets  # For RRTMGP lookup tables
using RRTMGP

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
                                 surface_pressure = 101325,
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
                                           liquid_effective_radius = ConstantRadiusParticles(10.0),  # μm
                                           ice_effective_radius = ConstantRadiusParticles(30.0))     # μm

# ## Atmosphere models
#
# Build the atmosphere models with saturation adjustment microphysics.

clock = Clock(time=DateTime(1950, 11, 1, 12, 0, 0))
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

gray_model = AtmosphereModel(grid; clock, dynamics, microphysics, radiation=gray_radiation)
clear_sky_model = AtmosphereModel(grid; clock, dynamics, microphysics, radiation=clear_sky_radiation)
high_co2_model = AtmosphereModel(grid; clock, dynamics, microphysics, radiation=high_co2_radiation)
all_sky_model = AtmosphereModel(grid; clock, dynamics, microphysics, radiation=all_sky_radiation)

# ## Initial condition: idealized tropical profile with a cloud
#
# We prescribe a simple tropical-like temperature profile with a moist boundary
# layer. To produce clouds for the all-sky comparison, we use high moisture
# that will saturate in the lower troposphere via saturation adjustment.

θ₀ = reference_state.potential_temperature
q₀ = 0.020    # surface specific humidity (kg/kg) - high enough to saturate
Hᵗ = 3000     # moisture scale height (m)
qᵗᵢ(z) = q₀ * exp(-z / Hᵗ)

set!(gray_model; θ=θ₀, qᵗ=qᵗᵢ)
set!(clear_sky_model; θ=θ₀, qᵗ=qᵗᵢ)
set!(high_co2_model; θ=θ₀, qᵗ=qᵗᵢ)
set!(all_sky_model; θ=θ₀, qᵗ=qᵗᵢ)

# ## Visualization
#
# After `set!`, the radiation has been computed. We build Fields and
# AbstractOperations to visualize the atmospheric state and radiative fluxes.

T = gray_model.temperature
pᵣ = reference_state.pressure
qᵗ = specific_prognostic_moisture(gray_model)
ℋ = RelativeHumidityField(gray_model)

ℐ_lw_up_gray = gray_radiation.upwelling_longwave_flux
ℐ_lw_dn_gray = gray_radiation.downwelling_longwave_flux
ℐ_sw_gray = gray_radiation.downwelling_shortwave_flux
ℐ_net_gray = ℐ_lw_up_gray + ℐ_lw_dn_gray + ℐ_sw_gray

ℐ_lw_up_clear = clear_sky_radiation.upwelling_longwave_flux
ℐ_lw_dn_clear = clear_sky_radiation.downwelling_longwave_flux
ℐ_sw_clear = clear_sky_radiation.downwelling_shortwave_flux
ℐ_net_clear = ℐ_lw_up_clear + ℐ_lw_dn_clear + ℐ_sw_clear

ℐ_lw_up_2xco2 = high_co2_radiation.upwelling_longwave_flux
ℐ_lw_dn_2xco2 = high_co2_radiation.downwelling_longwave_flux
ℐ_sw_2xco2 = high_co2_radiation.downwelling_shortwave_flux
ℐ_net_2xco2 = ℐ_lw_up_2xco2 + ℐ_lw_dn_2xco2 + ℐ_sw_2xco2

ℐ_lw_up_allsky = all_sky_radiation.upwelling_longwave_flux
ℐ_lw_dn_allsky = all_sky_radiation.downwelling_longwave_flux
ℐ_sw_allsky = all_sky_radiation.downwelling_shortwave_flux
ℐ_net_allsky = ℐ_lw_up_allsky + ℐ_lw_dn_allsky + ℐ_sw_allsky

# Get cloud liquid for visualization
qˡ = all_sky_model.microphysical_fields.qˡ

set_theme!(fontsize=14, linewidth=2.5)

# Format altitude ticks in km (but keep internal units in meters).
z_ticks_km = 0:5:20
z_ticks_m = ((z_ticks_km .* 1000), string.(z_ticks_km))

fig = Figure(size=(1600, 800), fontsize=14)
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
ax_net = Axis(fig[2, 4]; xlabel="Net flux (W/m²)", yticks=z_ticks_m)

# Hide y-axis decorations on inner panels
[hideydecorations!(ax, grid=false) for ax in (ax_q, ax_H, ax_ql, ax_lw_dn, ax_sw_dn, ax_net)]

# Atmospheric state
lines!(ax_T, T; color=:gray30)
lines!(ax_q, qᵗ; color=:gray30)
lines!(ax_H, 100ℋ; color=:gray30)
lines!(ax_ql, 1000qˡ; color=:lime)  # Convert to g/kg

# Colors for radiation schemes
c_gray = :black
c_clear = :dodgerblue
c_2xco2 = :orangered
c_allsky = :lime

# LW upwelling (positive)
lines!(ax_lw_up, ℐ_lw_up_gray;   color=c_gray)
lines!(ax_lw_up, ℐ_lw_up_clear;  color=c_clear)
lines!(ax_lw_up, ℐ_lw_up_2xco2;  color=c_2xco2)
lines!(ax_lw_up, ℐ_lw_up_allsky; color=c_allsky)

# LW downwelling (negative, so we negate for display)
lines!(ax_lw_dn, -ℐ_lw_dn_gray;   color=c_gray)
lines!(ax_lw_dn, -ℐ_lw_dn_clear;  color=c_clear)
lines!(ax_lw_dn, -ℐ_lw_dn_2xco2;  color=c_2xco2)
lines!(ax_lw_dn, -ℐ_lw_dn_allsky; color=c_allsky)

# SW downwelling (negative, so we negate for display)
lines!(ax_sw_dn, -ℐ_sw_gray;   color=c_gray)
lines!(ax_sw_dn, -ℐ_sw_clear;  color=c_clear)
lines!(ax_sw_dn, -ℐ_sw_2xco2;  color=c_2xco2)
lines!(ax_sw_dn, -ℐ_sw_allsky; color=c_allsky)

# Net flux
lines!(ax_net, ℐ_net_gray;   color=c_gray)
lines!(ax_net, ℐ_net_clear;  color=c_clear)
lines!(ax_net, ℐ_net_2xco2;  color=c_2xco2)
lines!(ax_net, ℐ_net_allsky; color=c_allsky)

# Legend
scheme_handles = [
    LineElement(color=c_gray, linewidth=3),
    LineElement(color=c_clear, linewidth=3),
    LineElement(color=c_2xco2, linewidth=3),
    LineElement(color=c_allsky, linewidth=3),
]
scheme_labels = ["Gray", "Clear-sky (420 ppm)", "2×CO₂ (840 ppm)", "All-sky (cloudy)"]
Legend(fig[0, :], scheme_handles, scheme_labels; orientation=:horizontal, framevisible=false, tellwidth=false)

fig
