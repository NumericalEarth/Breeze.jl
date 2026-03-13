# # Single column model with grid nudging
#
# This example sets up a single-column atmospheric model that nudges the
# horizontal wind and potential temperature profiles toward a reference
# [`FieldTimeSeries`](@ref) above 1500 m, while applying time-varying
# surface heat and moisture fluxes read from separate `FieldTimeSeries`.
#
# This configuration is suitable for driving a column LES with large-scale
# reanalysis data (e.g. ERA5), where the mesoscale environment is prescribed
# and only the sub-grid turbulence is explicitly simulated.

using Breeze
using Oceananigans
using Oceananigans.Units
using Dates
using Printf
using Statistics: mean

# ## Grid
#
# A single column spanning 15 km with 100 layers.
# `Flat` topology in x and y eliminates horizontal dimensions entirely.

Nz = 100
grid = RectilinearGrid(size = Nz,
                       x = 0, y = 0,
                       z = (0, 15kilometers),
                       topology = (Flat, Flat, Bounded))

FT = eltype(grid)

# ## Dynamics and reference state

constants = ThermodynamicConstants(FT)
reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = FT(300))
dynamics = AnelasticDynamics(reference_state)

# ## Nudging FTS
#
# Three one-dimensional `FieldTimeSeries` objects provide the reference
# profiles for u, v, and θ as a function of height and time.
# The times are in seconds; here we create 49-step hourly time series
# (as one might load from ERA5).
#
# In practice, load these from disk, e.g.:
#   u_ref = FieldTimeSeries("era5_column.jld2", "u")
#
# Requirements:
#   - Grid must be 1×1×Nz (or Flat×Flat×Nz) with the same vertical levels
#     as the simulation grid so that no vertical interpolation is needed.
#   - Values must be specific (u in m/s, v in m/s, θ in K) — not density-weighted.

nudging_times = range(0, 48hours, step=1hour) |> collect  # 49 hourly snapshots

u_ref = FieldTimeSeries{Face, Center, Center}(grid, nudging_times)
v_ref = FieldTimeSeries{Center, Face, Center}(grid, nudging_times)
θ_ref = FieldTimeSeries{Center, Center, Center}(grid, nudging_times)

# Fill with placeholder profiles (replace with actual data)
for n in 1:length(nudging_times)
    set!(u_ref[n], (x, y, z) -> FT(-5 + 0.001z))   # e.g. westerly shear
    set!(v_ref[n], (x, y, z) -> FT(0))
    set!(θ_ref[n], (x, y, z) -> FT(300 + 8e-3 * z)) # stable stratification
end

# ## Nudging forcings
#
# `RelaxationForcing` automatically enters profile mode when the reference FTS
# has no horizontal dimensions (Nx=Ny=1). The horizontal average of each field
# is compared to the reference column, and nudging is applied only above
# `z_bottom = 1500` m with a 6-hour relaxation time scale.

τ_nudging = 1hour

u_nudging = RelaxationForcing(u_ref; time_scale=τ_nudging, z_bottom=1500)
v_nudging = RelaxationForcing(v_ref; time_scale=τ_nudging, z_bottom=1500)
θ_nudging = RelaxationForcing(θ_ref; time_scale=τ_nudging, z_bottom=1500)

forcing = (; ρu = u_nudging, ρv = v_nudging, ρθ = θ_nudging)

# ## Surface flux FTS
#
# Two scalar `FieldTimeSeries` provide the surface fluxes.
# Units:
#   - heat_flux:     kinematic heat flux ρ w'θ'  [kg K m⁻² s⁻¹]
#                    (= sensible heat flux H [W m⁻²] divided by dry air heat capacity cₚ ≈ 1004 J kg⁻¹ K⁻¹)
#   - moisture_flux: kinematic moisture flux ρ w'q'  [kg m⁻² s⁻¹]
#                    (= latent heat flux LE [W m⁻²] divided by Lᵥ ≈ 2.5×10⁶ J kg⁻¹)
#
# Positive values denote upward flux from the surface into the atmosphere.

surface_times = nudging_times   # can differ from nudging times

# Single-point FTS for surface scalars
surface_grid = RectilinearGrid(size = 1,
                               x = 0, y = 0,
                               z = (0, 1),
                               topology = (Flat, Flat, Bounded))

heat_flux_fts     = FieldTimeSeries{Center, Center, Center}(surface_grid, surface_times)
moisture_flux_fts = FieldTimeSeries{Center, Center, Center}(surface_grid, surface_times)

# Fill with placeholder diurnal cycles (replace with actual data)
for n in 1:length(surface_times)
    t = surface_times[n]
    H_kinematic = FT(8e-3 * max(0, sin(2π * t / 86400)))  # diurnal sensible heat
    E_kinematic = FT(5.2e-5)                                # constant evaporation
    set!(heat_flux_fts[n],     (x, y, z) -> H_kinematic)
    set!(moisture_flux_fts[n], (x, y, z) -> E_kinematic)
end

# Continuous-form BC functions: called at every grid point with (x, y, t)
@inline surface_heat_flux(x, y, t)     = heat_flux_fts[1, 1, 1, Time(t)]
@inline surface_moisture_flux(x, y, t) = moisture_flux_fts[1, 1, 1, Time(t)]

ρθ_bcs  = FieldBoundaryConditions(bottom = FluxBoundaryCondition(surface_heat_flux))
ρqᵗ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(surface_moisture_flux))

boundary_conditions = (; ρθ = ρθ_bcs, ρqᵗ = ρqᵗ_bcs)

# ## Model

model = AtmosphereModel(grid;
                        dynamics,
                        formulation = :LiquidIcePotentialTemperature,
                        forcing,
                        boundary_conditions)

# ## Initial conditions

θ₀ = reference_state.potential_temperature
set!(model; θ = θ₀, qᵗ = FT(0.01))

# ## Simulation

simulation = Simulation(model; Δt = 60seconds, stop_time = 48hours)

# Progress reporting
function progress(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    θ_mean = mean(sim.model.temperature)
    @printf "t = %s, max|u| = %.2f m/s, mean T = %.2f K\n" prettytime(sim) u_max θ_mean
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(60))

# Output
simulation.output_writers[:fields] = JLD2Writer(model, merge(model.velocities, (; θ=model.temperature));
                                                filename = "single_column_nudging.jld2",
                                                schedule = TimeInterval(1hour),
                                                overwrite_existing = true)

run!(simulation)
