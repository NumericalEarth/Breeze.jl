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
using Oceananigans.Fields: interpolate!
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputReaders: Time
using NumericalEarth
using NumericalEarth.DataWrangling: download_dataset
using NumericalEarth.DataWrangling.ERA5
using CDSAPI # for ERA5 download
using Dates
using Printf
using Statistics: mean
using CairoMakie
using ColorSchemes

# ## HI-SCALE September 10 case
#
# A case day from the Holistic Interactions of Shallow Clouds, Aerosols and
# Land Ecosystems (HI-SCALE) campaign at the U.S. Department of Energy's
# Atmospheric Radiation Measurement (ARM) Climate Research Facility's Southern
# Great Plains (SGP) site. Features clear skies with periods of cirrus.

start_date = DateTime(2016, 09, 10, 12)
end_date   = DateTime(2016, 09, 11)

# ## Grid
#
# A single column spanning 15 km with 100 layers.
# `Flat` topology in x and y eliminates horizontal dimensions entirely.

Nz = 100
grid = RectilinearGrid(size = Nz,
                       x = 0, y = 0,
                       z = (0, 4kilometers),
                       topology = (Flat, Flat, Bounded))

ref_lat, ref_lon = 36.605, -97.485  # ARM SGP Central Facility at Lamont, OK

# ## Dynamics and reference state

θ₀ = 300
p₀ = 101325

constants = ThermodynamicConstants()

ref_state = ReferenceState(grid, constants;
                           surface_pressure = p₀,
                           potential_temperature = θ₀)
dynamics = AnelasticDynamics(ref_state)

# ## Pre-download ERA5 data (optional)
#
# Download all pressure-level variables at all dates to save time
vars_on_pressure_levels = [
    :temperature,
    :eastward_velocity,
    :northward_velocity,
    :geopotential, # to calculate geopotential height
]

selected_levels = filter(≥(250hPa), ERA5_all_pressure_levels) # select all levels below 250 hPa
dataset = ERA5HourlyPressureLevels(pressure_levels=selected_levels)

dates = start_date:Hour(1):end_date

region = Column(ref_lon, ref_lat)

download_dataset(vars_on_pressure_levels, dataset, dates; region)

# ## Nudging FTS
#
# Three one-dimensional `FieldTimeSeries` objects provide the reference
# profiles for u, v, and θ as a function of height and time. Note these are
# the primitive, not conserved, variables.

# We downloaded a subset of the ERA5, defined by bounding_box. Because this data grid is not
# coincident with the simulation grid, we need to perform a mapping to the reference lat, lon.

# reference fields to nudge toward
u_meta = Metadata(:eastward_velocity;  dataset, dates, region)
v_meta = Metadata(:northward_velocity; dataset, dates, region)
T_meta = Metadata(:temperature;        dataset, dates, region)
U_era5 = FieldTimeSeries(u_meta)
V_era5 = FieldTimeSeries(v_meta)
T_era5 = FieldTimeSeries(T_meta)

# for potential temperature conversion
pᵣ  = ref_state.pressure
pˢᵗ = ref_state.standard_pressure
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity

uᵣ  = FieldTimeSeries{Nothing, Nothing, Center}(grid, U_era5.times)
vᵣ  = FieldTimeSeries{Nothing, Nothing, Center}(grid, V_era5.times)
θᵣ  = FieldTimeSeries{Nothing, Nothing, Center}(grid, T_era5.times)

Tᵣn = Field{Nothing,Nothing,Center}(grid) # temporary Field
for n in eachindex(T_era5.times)
    # interpolate from ERA5 field to Breeze field (coarse → fine)
    interpolate!(uᵣ[n], U_era5[n])
    interpolate!(vᵣ[n], V_era5[n])
    interpolate!(Tᵣn,   T_era5[n])

    # convert to potential temperature
    set!(θᵣ[n], Tᵣn * (pˢᵗ / pᵣ)^(Rᵈ/cᵖᵈ))
end

# ## Nudging forcings
#
# `FieldTimeSeriesRelaxation` automatically enters profile mode when the reference FTS
# has no horizontal dimensions (Nx=Ny=1). The horizontal average of each field
# is compared to the reference column, and nudging is applied only above
# `z_bottom = 1500` m with a 1-hour relaxation time scale.

τ_nudging = 2hours
z_bottom  = 1500  # (m), height above which nudging occurs

u_nudging = FieldTimeSeriesRelaxation(uᵣ; time_scale=τ_nudging, z_bottom)
v_nudging = FieldTimeSeriesRelaxation(vᵣ; time_scale=τ_nudging, z_bottom)
θ_nudging = FieldTimeSeriesRelaxation(θᵣ; time_scale=τ_nudging, z_bottom)

forcing = (; ρu = u_nudging, ρv = v_nudging, ρθ = θ_nudging)

# ## Diffusion
#
# For this simple demo, use constant diffusivity instead of a turbulence closure (typically a
# planteary boundary layer scheme)

closure = VerticalScalarDiffusivity(ν=10, κ=10)   # [m² s⁻¹]

# ## Model

model = AtmosphereModel(grid; dynamics, forcing, closure)

# ## Initial conditions

set!(model; θ = θᵣ[1], u = uᵣ[1], v = vᵣ[1])

# ## Simulation

simulation = Simulation(model; Δt = 60seconds, stop_time = θᵣ.times[end])

# Progress reporting
function progress(sim)
    u_max = maximum(abs, sim.model.velocities.u)
    θ_mean = mean(sim.model.temperature)
    @printf "t = %s, max|u| = %.2f m/s, mean T = %.2f K\n" prettytime(sim) u_max θ_mean
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(60))

# Output
outfile = "single_column_nudging.jld2"
output_interval = 600seconds
θ_field = model.formulation.potential_temperature
simulation.output_writers[:fields] = JLD2Writer(model, merge(model.velocities, (; θ=θ_field));
                                                filename = outfile,
                                                schedule = TimeInterval(output_interval),
                                                overwrite_existing = true)

run!(simulation)

# ## Postprocess

u_ts = FieldTimeSeries(outfile, "u")
v_ts = FieldTimeSeries(outfile, "v")
θ_ts = FieldTimeSeries(outfile, "θ")

# Build Nz × Nt data matrices
u_data  = interior(u_ts,  1, 1, :, :)
v_data  = interior(v_ts,  1, 1, :, :)
θ_data  = interior(θ_ts,  1, 1, :, :)

# Temperature anomaly
θᵣ_on_ts = FieldTimeSeries{Nothing, Nothing, Center}(grid, θ_ts.times)
interpolate!(θᵣ_on_ts, θᵣ)
θᵣ_data = interior(θᵣ_on_ts, 1, 1, :, :)
θ′_data = θ_data - θᵣ_data

# Color limits
ulim = max(maximum(abs, u_data), 1e-6)
vlim = max(maximum(abs, v_data), 1e-6)
θlim = max(maximum(abs, θ′_data), 1e-6)

# Coordinates
z = znodes(u_ts.grid, Center()) ./ 1e3  # [km]
simtimes = start_date .+ Second.(round.(Int, u_ts.times))

Nt = length(simtimes)
Nz = length(z)

# Figure
fig1 = Figure(size = (1000, 900))

ax_θ = Axis(fig1[1, 1], ylabel = "z (km)", title = "Potential temperature anomaly")
ax_u = Axis(fig1[2, 1], ylabel = "z (km)", title = "Zonal wind")
ax_v = Axis(fig1[3, 1], ylabel = "z (km)", title = "Meridional wind", xlabel = "Starting from $(start_date)")

# Use numeric x-axis, then set ticks to dates
t_num = Float64.(1:Nt)  # use integer indices as x

# Need to transpose to get Nt x Nz
hm_θ = heatmap!(ax_θ, t_num, z, θ′_data'; colormap = :magma,         colorrange = (-θlim, θlim))
hm_u = heatmap!(ax_u, t_num, z, u_data';  colormap = Reverse(:RdBu), colorrange = (-ulim, ulim))
hm_v = heatmap!(ax_v, t_num, z, v_data';  colormap = Reverse(:RdBu), colorrange = (-vlim, vlim))

Colorbar(fig1[1, 2], hm_θ, label = "θ - θᵢ (K)")
Colorbar(fig1[2, 2], hm_u, label = "u (m s⁻¹)")
Colorbar(fig1[3, 2], hm_v, label = "v (m s⁻¹)")

# Date ticks
tick_interval = Int(2hours / output_interval)
tick_indices = 1:tick_interval:Nt
tick_labels  = Dates.format.(simtimes[tick_indices], "HH:MM")
tick_values  = Float64.(tick_indices)

for ax in (ax_θ, ax_u, ax_v)
    ax.xticks = (tick_values, tick_labels)
end
linkxaxes!(ax_θ, ax_u, ax_v)
hidexdecorations!(ax_θ)
hidexdecorations!(ax_u)

fig1

# ERA5 state vs Breeze SCM
#
fig2 = Figure(size = (800, 600))

ax_θ = Axis(fig2[1, 1], xlabel = "θ [K]",   ylabel = "z (km)", title = "Potential temperature")
ax_u = Axis(fig2[1, 2], xlabel = "u [m/s]", ylabel = "z (km)", title = "Zonal wind")
ax_v = Axis(fig2[1, 3], xlabel = "v [m/s]", ylabel = "z (km)", title = "Meridional wind")

linkyaxes!(ax_θ, ax_u, ax_v)
hideydecorations!(ax_u, grid=false)
hideydecorations!(ax_v, grid=false)

cmap = ColorSchemes.viridis
for n in eachindex(T_era5.times)
    frac = (n-1) / (length(T_era5.times)-1)
    color = get(cmap, frac)

    lines!(ax_θ, θᵣ[n]; color, linestyle=:dash)
    lines!(ax_u, uᵣ[n]; color, linestyle=:dash)
    lines!(ax_v, vᵣ[n]; color, linestyle=:dash)

    ti = T_era5.times[n]
    lines!(ax_θ, θ_ts[Time(ti)]; color)
    lines!(ax_u, u_ts[Time(ti)]; color)
    lines!(ax_v, v_ts[Time(ti)]; color)
end

fig2
