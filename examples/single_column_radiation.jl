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

reference_state = ReferenceState(grid, constants,
                                 base_pressure = 101325,
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

θ = liquid_ice_potential_temperature(model)
T = model.auxiliary_fields.T
qᵗ = model.specific_moisture
qˡ = model.microphysical_fields.qˡ

F_lw_up = radiation.upwelling_longwave_flux
F_lw_dn = radiation.downwelling_longwave_flux
F_sw = radiation.downwelling_shortwave_flux  # Direct beam only (no-scattering solver)

# Net flux = upwelling LW - downwelling LW - downwelling SW (absorbed)
F_net = Field(F_lw_up - F_lw_dn - F_sw)

# Extract data and convert altitude to km
zc = znodes(grid, Center()) ./ 1e3   # cell centers in km
zf = znodes(grid, Face()) ./ 1e3     # cell faces in km

fig = Figure(size=(1100, 600), fontsize=14)

ax_T = Axis(fig[1, 1], xlabel="Temperature (K)", ylabel="Altitude (km)")
ax_q = Axis(fig[1, 2], xlabel="Specific humidity (g/kg)", ylabel="Altitude (km)")
ax_lw = Axis(fig[2, 1], xlabel="Longwave flux (W/m²)", ylabel="Altitude (km)")
ax_sw = Axis(fig[2, 2], xlabel="Shortwave flux (W/m²)", ylabel="Altitude (km)")
ax_net = Axis(fig[1:2, 3], xlabel="Net flux (W/m²)", ylabel="Altitude (km)")

lines!(ax_T, interior(T, 1, 1, :), zc; label="T")
lines!(ax_T, interior(θ, 1, 1, :), zc; linestyle=:dash, label="θ")
axislegend(ax_T, position=:lt)

lines!(ax_q, interior(qᵗ, 1, 1, :) .* 1000, zc; label="qᵗ")
lines!(ax_q, interior(qˡ, 1, 1, :) .* 1000, zc; label="qˡ")
axislegend(ax_q, position=:rt)

lines!(ax_lw, interior(F_lw_up, 1, 1, :), zf; label="↑ upwelling")
lines!(ax_lw, interior(F_lw_dn, 1, 1, :), zf; label="↓ downwelling")
axislegend(ax_lw, position=:rt)

# Non-scattering solver: only direct beam
lines!(ax_sw, interior(F_sw, 1, 1, :), zf; label="↓ direct beam")
axislegend(ax_sw, position=:lb)

lines!(ax_net, interior(F_net, 1, 1, :), zf)

fig[0, :] = Label(fig, "Single Column Gray Radiation (O'Gorman & Schneider, 2008)", fontsize=18, tellwidth=false)

save("single_column_radiation.png", fig)
fig

