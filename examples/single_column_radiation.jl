# # Single column radiation (gray vs clear-sky full-spectrum)
#
# This example sets up a single-column atmospheric model with an idealized
# temperature and moisture profile. We compute radiative fluxes using RRTMGP's
# gray atmosphere solver with the optical thickness parameterization
# by [OGormanSchneider2008](@citet), and compare against clear-sky full-spectrum
# gas optics.

using Breeze
using Oceananigans.Units
using CairoMakie

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

formulation = AnelasticFormulation(reference_state)

# ## Radiative transfer models
#
# We create a gray radiative transfer model using the [OGormanSchneider2008](@citet)
# optical thickness parameterization. The solar zenith angle is computed from the
# model clock and grid location. We also create a clear-sky full-spectrum model.

using Dates

gray_radiation = RadiativeTransferModel(grid, :gray, constants;
                                        surface_temperature,
                                        surface_emissivity = 0.98,
                                        surface_albedo = 0.1,
                                        solar_constant = 1361)        # W/m²

clear_sky_radiation = RadiativeTransferModel(grid, :clear_sky, constants;
                                             surface_temperature,
                                             surface_emissivity = 0.98,
                                             surface_albedo = 0.1,
                                             solar_constant = 1361)    # W/m²

# ## Atmosphere model
#
# Build the atmosphere model with saturation adjustment microphysics.

clock = Clock(time=DateTime(1950, 11, 1, 12, 0, 0))
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

gray_model = AtmosphereModel(grid; clock, formulation, microphysics, radiation=gray_radiation)
clear_sky_model = AtmosphereModel(grid; clock, formulation, microphysics, radiation=clear_sky_radiation)

# ## Initial condition: idealized tropical profile with a cloud
#
# We prescribe a simple tropical-like temperature profile with a moist boundary
# layer and a cloud between 1-2 km altitude.

# Use a mildly stable profile so temperatures remain within RRTMGP's supported range.
θᵢ(z) = surface_temperature + 5e-3 * z
q₀ = 0.015    # surface specific humidity (kg/kg)
Hᵗ = 2500     # moisture scale height (m)
qᵗᵢ(z) = q₀ * exp(-z / Hᵗ)

set!(gray_model; θ=θᵢ, qᵗ=qᵗᵢ)
set!(clear_sky_model; θ=θᵢ, qᵗ=qᵗᵢ)

# ## Visualization
#
# After `set!`, the radiation has been computed. We build Fields and
# AbstractOperations to visualize the atmospheric state and radiative fluxes.

T = gray_model.temperature
pᵣ = reference_state.pressure
qᵗ = gray_model.specific_moisture
ℋ = RelativeHumidityField(gray_model)

ℐ_lw_up_gray = gray_radiation.upwelling_longwave_flux
ℐ_lw_dn_gray = gray_radiation.downwelling_longwave_flux
ℐ_sw_gray = gray_radiation.downwelling_shortwave_flux
ℐ_net_gray = ℐ_lw_up_gray + ℐ_lw_dn_gray + ℐ_sw_gray

ℐ_lw_up_clear = clear_sky_radiation.upwelling_longwave_flux
ℐ_lw_dn_clear = clear_sky_radiation.downwelling_longwave_flux
ℐ_sw_clear = clear_sky_radiation.downwelling_shortwave_flux
ℐ_net_clear = ℐ_lw_up_clear + ℐ_lw_dn_clear + ℐ_sw_clear

set_theme!(fontsize=14, linewidth=3)

# Format altitude ticks in km (but keep internal units in meters).
z_ticks_km = 0:5:20
z_ticks_m = ((z_ticks_km .* 1000), string.(z_ticks_km))

fig = Figure(size=(1200, 420), fontsize=14)

ax_T = Axis(fig[2, 1]; xlabel="Temperature, T (K)", ylabel="Altitude (km)",
            yticks=z_ticks_m, xticks=150:50:300)
ax_q = Axis(fig[2, 2]; xlabel="Total specific humidity, qᵗ (kg/kg)", yticks=z_ticks_m)
ax_H = Axis(fig[2, 3]; xlabel="Relative humidity, ℋ (%)", yticks=z_ticks_m)
ax_I = Axis(fig[2, 4:6]; xlabel="Radiative flux, ℐ (W/m²)",
            ylabel="Altitude (km)", yticks=z_ticks_m, yaxisposition=:right)

[hideydecorations!(ax, grid=false) for ax in (ax_q, ax_H)]
hidespines!(ax_T, :r, :t)
hidespines!(ax_q, :l, :r, :t)
hidespines!(ax_H, :l, :r, :t)
hidespines!(ax_I, :l, :t)

lines!(ax_T, T)
lines!(ax_q, qᵗ; label="qᵗ")
lines!(ax_H, 100ℋ)  # Convert to %

# Radiation comparison
c_gray = :black
c_clear = :dodgerblue

ls_lw_up = :solid
ls_lw_dn = :dash
ls_sw_dn = :dot

lines!(ax_I, ℐ_lw_up_gray;  color=c_gray, linestyle=ls_lw_up, alpha=0.85)
lines!(ax_I, ℐ_lw_dn_gray;  color=c_gray, linestyle=ls_lw_dn, alpha=0.85)
lines!(ax_I, ℐ_sw_gray;     color=c_gray, linestyle=ls_sw_dn, alpha=0.85)
lines!(ax_I, ℐ_net_gray;    color=c_gray, linewidth=4, alpha=0.35)

lines!(ax_I, ℐ_lw_up_clear; color=c_clear, linestyle=ls_lw_up, alpha=0.85)
lines!(ax_I, ℐ_lw_dn_clear; color=c_clear, linestyle=ls_lw_dn, alpha=0.85)
lines!(ax_I, ℐ_sw_clear;    color=c_clear, linestyle=ls_sw_dn, alpha=0.85)
lines!(ax_I, ℐ_net_clear;   color=c_clear, linewidth=4, alpha=0.35)

# Two compact legends: one for scheme colors, one for component line styles.
scheme_handles = [
    LineElement(color=c_gray, linewidth=4),
    LineElement(color=c_clear, linewidth=4),
]
scheme_labels = ["Gray", "Clear-sky full-spectrum"]
Legend(fig[1, 4:6], scheme_handles, scheme_labels; orientation=:horizontal, framevisible=false)

component_handles = [
    LineElement(color=:gray30, linestyle=ls_lw_up, linewidth=3),
    LineElement(color=:gray30, linestyle=ls_lw_dn, linewidth=3),
    LineElement(color=:gray30, linestyle=ls_sw_dn, linewidth=3),
    LineElement(color=:gray30, linestyle=:solid, linewidth=6),
]
component_labels = ["LW ↑", "LW ↓", "SW ↓", "Net"]
Legend(fig[1, 1:3], component_handles, component_labels; orientation=:horizontal, framevisible=false)

title = "Single Column Radiation: gray vs clear-sky full-spectrum"
fig[0, :] = Label(fig, title, fontsize=18, tellwidth=false)

fig
