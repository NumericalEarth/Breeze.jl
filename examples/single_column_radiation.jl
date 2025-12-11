# # Single column radiation
#
# This example sets up a single-column atmospheric model with an idealized
# temperature and moisture profile. We compute radiative fluxes using RRTMGP's
# gray atmosphere solver with the O'Gorman and Schneider (2008) optical thickness
# parameterization.
#
# The gray radiation parameterization follows:
#
# > O'Gorman, P. A., & Schneider, T. (2008). The hydrological cycle over a wide
# > range of climates simulated with an idealized GCM. *Journal of Climate*, 21(15),
# > 3815–3832. DOI: [10.1175/2007JCLI2065.1](https://doi.org/10.1175/2007JCLI2065.1)
#
# See also Schneider (2004) for background on idealized gray radiation:
#
# > Schneider, T. (2004). The tropopause and the thermal stratification in the
# > extratropics of a dry atmosphere. *Journal of the Atmospheric Sciences*, 61(12),
# > 1317–1340. DOI: [10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2](https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2)

using Breeze
using Oceananigans.Units
using RRTMGP
using CairoMakie

# ## Grid and thermodynamics
#
# We create a single column spanning 20 km with 64 layers, located
# at Beverly, Massachusetts, USA (42.5°N, 70.9°W).

Nz = 64
λ, φ = -70.9, 42.5

grid = RectilinearGrid(size=Nz, x=λ, y=φ, z=(0, 20kilometers),
                       topology=(Flat, Flat, Bounded))

# Set up the thermodynamic constants and reference state.

constants = ThermodynamicConstants()

surface_temperature = 300

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = surface_temperature)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

# ## Radiative transfer model
#
# We create a gray radiative transfer model using the O'Gorman and Schneider (2008)
# optical thickness parameterization. The solar zenith angle is computed from the
# model clock and grid location.

using Dates

radiation = GrayRadiation(grid;
                          surface_temperature,
                          surface_emissivity = 0.98,
                          surface_albedo = 0.1,
                          solar_constant = 1361)        # W/m²

# ## Atmosphere model
#
# Build the atmosphere model with saturation adjustment microphysics.

clock = Clock(time=DateTime(2024, 9, 27, 16, 0, 0))
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
model = AtmosphereModel(grid; clock, formulation, microphysics, radiation)

# ## Initial condition: idealized tropical profile with a cloud
#
# We prescribe a simple tropical-like temperature profile with a moist boundary
# layer and a cloud between 1-2 km altitude.

θ₀ = formulation.reference_state.potential_temperature
cᵖᵈ = constants.dry_air.heat_capacity
g = constants.gravitational_acceleration
Γ = g / cᵖᵈ
θ_profile(z) = θ₀ + Γ * z / 1000

q₀ = 0.015    # surface specific humidity (kg/kg)
Hᵗ = 2500     # moisture scale height (m)
qᵗ_profile(z) = q₀ * exp(-z / Hᵗ)

set!(model; θ=θ_profile, qᵗ=qᵗ_profile)

# ## Visualization
#
# After `set!`, the radiation has been computed. We build Fields and
# AbstractOperations to visualize the atmospheric state and radiative fluxes.

T = model.auxiliary_fields.T
pᵣ = reference_state.pressure
qᵗ = model.specific_moisture
ℋ = RelativeHumidityField(model)

F_lw_up = radiation.upwelling_longwave_flux
F_lw_dn = radiation.downwelling_longwave_flux
F_sw = radiation.downwelling_shortwave_flux
F_net = Field(F_lw_up - F_lw_dn - F_sw)

# Convert altitude to km for plotting
zc = znodes(grid, Center()) ./ 1e3
zf = znodes(grid, Face()) ./ 1e3

fig = Figure(size=(1400, 350), fontsize=14)

ax_T = Axis(fig[1, 1], xlabel="T (K)",      ylabel="Altitude (km)")
ax_p = Axis(fig[1, 2], xlabel="pᵣ (hPa)",   ylabel="Altitude (km)")
ax_q = Axis(fig[1, 3], xlabel="qᵗ (kg/kg)", ylabel="Altitude (km)")
ax_H = Axis(fig[1, 4], xlabel="ℋ",          ylabel="Altitude (km)")
ax_F = Axis(fig[1, 5], xlabel="F (W/m²)",   ylabel="Altitude (km)")

lines!(ax_T, zc, T)
lines!(ax_p, zc, pᵣ / 100)  # Convert Pa to hPa
lines!(ax_q, zc, qᵗ)
lines!(ax_H, zc, ℋ)

# All radiation fluxes in one panel
lines!(ax_F, zf, F_lw_up; label="LW ↑")
lines!(ax_F, zf, F_lw_dn; label="LW ↓")
lines!(ax_F, zf, F_sw; linestyle=:dash, label="SW ↓")
lines!(ax_F, zf, F_net; linewidth=4, alpha=0.6, color=:black, label="Net")
axislegend(ax_F, position=:lb)

fig[0, :] = Label(fig, "Single Column Gray Radiation (O'Gorman & Schneider, 2008)", fontsize=18, tellwidth=false)

save("single_column_radiation.png", fig)
fig

