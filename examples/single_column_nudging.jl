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
using Oceananigans.TurbulenceClosures
using Oceananigans.Fields: fractional_x_index, fractional_y_index
using Oceananigans.OutputReaders: Time
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, download_dataset
using NumericalEarth.DataWrangling.ERA5
using CDSAPI # for ERA5 download
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
                       z = (0, 4kilometers),
                       topology = (Flat, Flat, Bounded))

ref_loc = (latitude=18.0, longitude=-61.5)

# ## Dynamics and reference state

FT = eltype(grid)
θ₀ = FT(300)
p₀ = FT(101325)
q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero

constants = ThermodynamicConstants(FT)

ref_state = ReferenceState(grid, constants;
                           surface_pressure = p₀,
                           potential_temperature = θ₀)
dynamics = AnelasticDynamics(ref_state)

# ## Pre-download ERA5 data (optional)

dates = DateTime(2004, 12, 16):Hour(1):DateTime(2005, 01, 09)

# bounding box should enclose ref_loc
bounding_box = BoundingBox(latitude=(17, 18.5), longitude=(-62.5, -61))

plev_vars = [:temperature,
             :eastward_velocity,
             :northward_velocity]

selected_levels = filter(≥(250hPa), ERA5_all_pressure_levels) # select all levels below 250 hPa
dataset = ERA5HourlyPressureLevels(pressure_levels=selected_levels)

download_dataset(plev_vars, dataset, dates; bounding_box)

# ## Nudging FTS
#
# Three one-dimensional `FieldTimeSeries` objects provide the reference
# profiles for u, v, and θ as a function of height and time. Note these are
# the primitive, not conserved, variables.

# We downloaded a subset of the ERA5, defined by bounding_box. Because this data grid is not
# coincident with the simulation grid, we need to perform a mapping to the reference lat, lon.

temp = Field(Metadatum(:temperature; dataset, date=first(dates), bounding_box))
Δλ = temp.grid.Δλᶜᵃᵃ
Δφ = temp.grid.Δφᵃᶜᵃ

xn = [ref_loc.longitude - Δλ/2, ref_loc.longitude + Δλ/2]
yn = [ref_loc.latitude  - Δφ/2, ref_loc.latitude  + Δφ/2]
zn = znodes(grid, Center(), Center(), Face(); with_halos=false)

column_grid = RectilinearGrid(size=(1, 1, Nz),
                              x=xn, y=yn, z=zn,
                              topology=(Bounded, Bounded, Bounded))

u_ref = FieldTimeSeries(Metadata(:eastward_velocity;  dataset, dates, bounding_box), column_grid)
v_ref = FieldTimeSeries(Metadata(:northward_velocity; dataset, dates, bounding_box), column_grid)
T_ref = FieldTimeSeries(Metadata(:temperature;        dataset, dates, bounding_box), column_grid)

# convert to potential temperature
θ_ref = FieldTimeSeries{Center, Center, Center}(column_grid, T_ref.times)
p_col = interior(ref_state.pressure, 1, 1, :)  # 1D pressure profile
R_cp = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
for n in eachindex(T_ref.times)
    interior(θ_ref[n]) .= interior(T_ref[n]) .* (ref_state.standard_pressure ./ reshape(p_col, 1, 1, :)).^R_cp
end

# ## Nudging forcings
#
# `RelaxationForcing` automatically enters profile mode when the reference FTS
# has no horizontal dimensions (Nx=Ny=1). The horizontal average of each field
# is compared to the reference column, and nudging is applied only above
# `z_bottom = 1500` m with a 1-hour relaxation time scale.

τ_nudging = 6hours

u_nudging = RelaxationForcing(u_ref; time_scale=τ_nudging, z_bottom=1500)
v_nudging = RelaxationForcing(v_ref; time_scale=τ_nudging, z_bottom=1500)
θ_nudging = RelaxationForcing(θ_ref; time_scale=τ_nudging, z_bottom=1500)

forcing = (; ρu = u_nudging, ρv = v_nudging, ρθ = θ_nudging)

# ## Diffusion
#
# For this simple demo, use constant diffusivity instead of a PBL scheme

closure = VerticalScalarDiffusivity(ν=10, κ=10)   # [m² s⁻¹]

# ## Model

model = AtmosphereModel(grid;
                        dynamics,
                        formulation = :LiquidIcePotentialTemperature,
                        forcing,
                        closure)

# ## Initial conditions

set!(model; θ = θ₀, qᵗ = FT(0.01))

# ## Simulation

simulation = Simulation(model; Δt = 60seconds, stop_time = T_ref.times[end])

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
                                                schedule = TimeInterval(600seconds),
                                                overwrite_existing = true)

run!(simulation)
