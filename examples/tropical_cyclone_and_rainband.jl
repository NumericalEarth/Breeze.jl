using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: FieldTimeSeries

using AtmosphericProfilesLibrary
using CairoMakie
using CloudMicrophysics
using Printf
using Random
using CUDA
using CSV
using DataFrames
using Interpolations: linear_interpolation

Random.seed!(42)

Random.TaskLocalRNG()

## Domain and grid
Oceananigans.defaults.FloatType = Float32

Nx = Ny = 200
Nz = 100

x = y = (0, Nx*1000)
z = (0, 16000)

grid = RectilinearGrid(GPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()

## reference state

# Load moist tropical sounding from Dunion 2011
println("\n=== Loading Dunion 2011 moist tropical sounding ===")
sounding = CSV.read("dunion2011_moist_tropical_MT.csv", DataFrame)
T_C = sounding[:, :Temperature_C]
T_dew = sounding[:, :Dewpoint_C]
p_hPa = Float64.(sounding[:, :Pressure_hPa])
z_sounding = Float64.(sounding[:, :GPH_m])
θ_sounding = Float64.(sounding[:, :Theta_K])
mixing_ratio_g_kg = Float64.(sounding[:, :Mixing_ratio_g_kg])

## set reference state

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = p_hPa[end]*100,
                                 potential_temperature = θ_sounding[end])

dynamics = AnelasticDynamics(reference_state)



## surface fluxes
Cᴰ = 1.229e-3 # Drag coefficient for momentum
Cᵀ = 1.094e-3 # Sensible heat transfer coefficient
Cᵛ = 1.133e-3 # Moisture flux transfer coefficient
T₀ = 300     # Sea surface temperature (K)

ρe_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T₀)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T₀)

ρe_bcs = FieldBoundaryConditions(bottom=ρe_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

## damping
sponge_rate = 1/8  # s⁻¹ - relaxation rate (8 s timescale)
sponge_mask = GaussianMask{:z}(center=3500, width=500)
sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

## subsidence
FT = eltype(grid)
wˢ_profile = AtmosphericProfilesLibrary.Rico_subsidence(FT)
wˢ = Field{Nothing, Nothing, Face}(grid)
set!(wˢ, z -> wˢ_profile(z))
subsidence = SubsidenceForcing(wˢ)


## geostrophic forcings
println("\n=== Setting up geostrophic forcings ===")
coriolis = FPlane(f=4.5e-5)
println("  Coriolis parameter: f = $(coriolis.f) s⁻¹")
uᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_ug(FT)
vᵍ = AtmosphericProfilesLibrary.Rico_geostrophic_vg(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))
println("  Geostrophic wind profiles loaded from RICO")


## moisture tendency
ρᵣ = reference_state.density
∂t_ρqᵗ_large_scale = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Rico_dqtdt(FT)
set!(∂t_ρqᵗ_large_scale, z -> dqdt_profile(z))
set!(∂t_ρqᵗ_large_scale, ρᵣ * ∂t_ρqᵗ_large_scale)
∂t_ρqᵗ_large_scale_forcing = Forcing(∂t_ρqᵗ_large_scale)

## radiative forcing
∂t_ρθ_large_scale = Field{Nothing, Nothing, Center}(grid)
∂t_θ_large_scale = - 2.5 / day # K / day
set!(∂t_ρθ_large_scale, ρᵣ * ∂t_θ_large_scale)
ρθ_large_scale_forcing = Forcing(∂t_ρθ_large_scale)


## forcing and boundary conditions
Fρu = (subsidence, geostrophic.ρu)
Fρv = (subsidence, geostrophic.ρv)
Fρw = sponge
Fρqᵗ = (subsidence, ∂t_ρqᵗ_large_scale_forcing)
Fρθ = (subsidence, ρθ_large_scale_forcing)

forcing = (ρu=Fρu, ρv=Fρv, ρw=Fρw, ρqᵗ=Fρqᵗ, ρθ=Fρθ)
boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

## model
println("\n=== Creating AtmosphereModel ===")
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

weno = WENO(order=5)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))

momentum_advection = weno
scalar_advection = (ρθ = weno,
                    ρqᵗ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection, scalar_advection, forcing, boundary_conditions)


###########################
# Initial conditions
###########################

RMW = 31000
V_RMW = 43 # m/s
a = 0.5
Deltap = 4500 # Pa
pressure_top = 50000 # Pa (500 mb)
Δz = znodes(grid, Center())[2] - znodes(grid, Center())[1]

# Center of domain for perturbation
x_center = (x[1] + x[2]) / 2
y_center = (y[1] + y[2]) / 2
println("  Perturbation center: ($(x_center/1000), $(y_center/1000)) km")


# =============================================================================
# VORTEX PROFILES (Moon and Nolan 2010 / Emanuel 1986 / Stern and Nolan 2009 / Yu and Didlake 2019)
# =============================================================================


# Convert mixing ratio from g/kg to kg/kg, then to specific humidity qᵗ
# Specific humidity q = w / (1 + w) where w is mixing ratio in kg/kg
mixing_ratio_kg_kg = mixing_ratio_g_kg ./ 1000
qᵗ_sounding = mixing_ratio_kg_kg ./ (1 .+ mixing_ratio_kg_kg)

# Create interpolations for the sounding profiles
# Reverse to get ascending order (z increasing)
z_asc = reverse(z_sounding)
θ_asc = reverse(θ_sounding)
qᵗ_asc = reverse(qᵗ_sounding)
T_K_asc = reverse(T_C .+ 273.15)

θ_sounding_interp = linear_interpolation(z_asc, θ_asc)
qᵗ_sounding_interp = linear_interpolation(z_asc, qᵗ_asc)
T_sounding_interp = linear_interpolation(z_asc, T_K_asc)

println("  Sounding loaded: $(length(z_sounding)) levels from $(minimum(z_sounding)) to $(maximum(z_sounding)) m")
println("  Surface θ: $(θ_sounding[end]) K, qᵗ: $(qᵗ_sounding[end]) kg/kg")
println("  Tropopause θ: $(θ_sounding[1]) K, qᵗ: $(qᵗ_sounding[1]) kg/kg")

# The radius of the angular momentum surface changes with height,
# following Eq. 4.4 in Stern and Nolan 2009
T_out_K = T_sounding_interp(16000)
radius_of_angular_momentum_surface = Array{Float64}(undef, Nz)
radius_of_angular_momentum_surface[1] = RMW
for k in 2:Nz
    z_k = znodes(grid, Center())[k]
    dTdZ = (T_sounding_interp(z_k + Δz) - T_sounding_interp(z_k))/Δz
    denom = 2 * (T_sounding_interp(z_k) - T_out_K)
    drdZ = -radius_of_angular_momentum_surface[k-1] / denom * dTdZ
    radius_of_angular_momentum_surface[k] = radius_of_angular_momentum_surface[k-1] + drdZ * Δz
end

function radius_of_angular_momentum_surface_func(z)
    z_nodes = znodes(grid, Center())
    k_idx = searchsortedfirst(z_nodes, z)
    k_idx = clamp(k_idx, 1, Nz)
    return radius_of_angular_momentum_surface[k_idx] - RMW
end

# Alias for use in other functions
radius_of_eyewall_adjustment = radius_of_angular_momentum_surface_func




function tangential_wind(x, y, z)
    v_adjustment_factor = (RMW) ./ (RMW + radius_of_eyewall_adjustment(z))
    r = sqrt((x-x_center)^2 + (y-y_center)^2)
    if r <= RMW + radius_of_eyewall_adjustment(z)
        return V_RMW * v_adjustment_factor * r/(RMW + radius_of_eyewall_adjustment(z))
    else
        return V_RMW * v_adjustment_factor * ((RMW + radius_of_eyewall_adjustment(z))/r)^a
    end
end

# calculate the pressure gradient due to the vortex
# Integrate from far away (1000+ km) where pressure is at background value
# This ensures proper gradient wind balance from the undisturbed environment
max_radius = 1500kilometers  # Integrate from 1000 km outward
∂r = 1000 # m
rrange = collect(max_radius:-∂r:0)  # Descending from outer edge to center
rrange_asc = reverse(rrange)  # Ascending for searchsortedfirst

Nz_p = length(znodes(grid, Center()))
Nr = length(rrange)
p = zeros(Nz_p, Nr)

# Compute pressure profile integrating from outer edge inward
for k in 1:Nz_p
    z_k = znodes(grid, Center())[k]
    z_clamped = clamp(z_k, minimum(z_asc), maximum(z_asc))

    # Background pressure at this height (using sounding)
    T_k = T_sounding_interp(z_clamped)
    R = constants.molar_gas_constant/constants.dry_air.molar_mass
    p_background = reference_state.surface_pressure * exp(-constants.gravitational_acceleration * z_k / (R * T_k))

    # Start from outer edge (no perturbation)
    p[k, end] = p_background

    # Integrate inward using gradient wind balance
    # Gradient wind balance: (1/ρ) * dp/dr = f*v + v²/r
    # So: dp/dr = ρ * (f*v + v²/r)
    # For a cyclonic vortex, this is positive (pressure increases with radius)
    # When integrating inward (decreasing r), pressure decreases
    for r_idx in (Nr-1):-1:1
        r = rrange[r_idx]
        # Density at this radius (approximate using background)
        ρ = p_background / (R * T_k)

        # Compute pressure gradient from gradient wind balance
        # Use radius from center, not absolute position
        v_tang = tangential_wind(x_center + r, y_center, z_k)
        # Gradient wind: dp/dr = ρ * (f*v + v²/r)
        dp_dr = ρ * (v_tang * coriolis.f + v_tang^2 / max(r, 100))  # Avoid division by zero

        # When moving inward (r decreases by ∂r), pressure change is dp_dr * (-∂r)
        # Since we're going from larger r to smaller r, the change is negative
        dp = -dp_dr * ∂r

        # Pressure decreases as we move inward
        p[k, r_idx] = p[k, r_idx + 1] + dp
    end
end
p = p[:,1:Nx]
# Compute pressure deficit relative to outer edge
p_outer = p[:, end]
δp = p .- reshape(repeat(p_outer, Nx), Nz_p, Nx)

# Create interpolation function for pressure deficit
function δp_func(x, y, z)
    radius = sqrt((x - x_center)^2 + (y - y_center)^2)

    # Clamp to valid ranges
    radius_clamped = clamp(radius, 0, max_radius)
    z_clamped = clamp(z, minimum(znodes(grid, Center())), maximum(znodes(grid, Center())))

    # Find indices
    z_idx = searchsortedfirst(znodes(grid, Center()), z_clamped)
    z_idx = clamp(z_idx, 1, Nz_p)

    r_idx = searchsortedfirst(rrange_asc, radius_clamped)
    r_idx = clamp(r_idx, 1, Nr)

    # Return pressure deficit in Pa (not hPa)
    return δp[z_idx, r_idx]
end

# plot the pressure profile
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, 1:1:Nx, znodes(grid, Center())/1000, δp', levels=-50:1:0, colormap=Reverse(:tol_YlOrBr))
Colorbar(fig[1, 2], limits=(-50, 0), label="pressure deficit (hPa)", colormap=Reverse(:tol_YlOrBr))
ax.ylabel = "Height (km)"
ax.xlabel = "Radius (km)"
Makie.save("pressure_profile.png", fig)



function u_init(x, y, z)
    return -sin(atan(y-y_center, x-x_center))*tangential_wind(x, y, z)
end

function v_init(x, y, z)
    return cos(atan(y-y_center, x-x_center))*tangential_wind(x, y, z)
end




function θ_init(x, y, z)
    # Get background potential temperature from Dunion 2011 sounding
    # Clamp z to sounding range for interpolation
    z_clamped = clamp(z, minimum(z_asc), maximum(z_asc))
    θ_background = θ_sounding_interp(z_clamped)

    # Get reference pressure at height z using hydrostatic relation
    # Use the sounding temperature profile to compute pressure
    p_surface = reference_state.surface_pressure
    g = constants.gravitational_acceleration
    R = constants.molar_gas_constant/constants.dry_air.molar_mass

    # Use temperature from sounding to compute reference pressure
    T_background = T_sounding_interp(z_clamped)
    p_ref = p_surface * exp(-g * z / (R * T_background))

    # Get pressure deficit from vortex (in Pa, negative value at center)
    δp = δp_func(x, y, z)

    # Potential temperature perturbation (negative sign gives warm core for pressure deficit)
    δθ = -θ_background * (δp * 100 / p_ref)

    # Return background potential temperature from sounding plus vortex perturbation
    return θ_background + δθ
end

function qᵗ_init(x, y, z)
    # Get background total moisture from Dunion 2011 sounding
    # Clamp z to sounding range for interpolation
    z_clamped = clamp(z, minimum(z_asc), maximum(z_asc))
    qᵗ_background = qᵗ_sounding_interp(z_clamped)

    # For now, use the sounding profile without radial variation
    # (The vortex perturbation primarily affects temperature/pressure, not moisture)
    # In future, could add moisture enhancement in the eyewall region
    return qᵗ_background
end

# set the initial conditions
set!(model,
    θ=θ_init,
    qᵗ=qᵗ_init,
    u=u_init,
    v=v_init)



tangential_wind_data = (Array(interior(model.velocities.u, :, :, 1)).^2 + Array(interior(model.velocities.v, :, :, 1)).^2).^0.5
tangential_wind_slice = (Array(interior(model.velocities.u, :, Nx÷2, :)).^2 + Array(interior(model.velocities.v, :, Nx÷2, :)).^2).^0.5
temperature_profile = Array(interior(liquid_ice_potential_temperature(model), 1, 1, :))
############## PLOT INITIAL CONDITIONS ##############

# Extract coordinate arrays
x_coords = xnodes(grid, Center())
y_coords = ynodes(grid, Center())
z_coords = znodes(grid, Center())

fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, x_coords, y_coords,
    Array(interior(liquid_ice_potential_temperature(model), :, :, 2)), levels=295:1:330)

Colorbar(fig[1, 2], limits=(295, 330), label="Potential Temperature (K)")
Makie.save("theta_init.png", fig)

## cross section at center of domain
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, y_coords, z_coords, Array(interior(liquid_ice_potential_temperature(model), Nx÷2, :, :)) .- temperature_profile', levels=0:1:20)
Colorbar(fig[1, 2], limits=(0, 20), label="Potential Temperature (K)")
Makie.save("theta_init_cross_section.png", fig)

## tangential wind profile
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, x_coords, y_coords, tangential_wind_data, levels=0:1:43)
Colorbar(fig[1, 2], limits=(0, 43), label="Tangential Wind (m/s)")
Makie.save("tangential_wind_profile.png", fig)

## tangential wind profile at cross section
fig = Figure()
ax = Axis(fig[1, 1])
contourf!(ax, y_coords/1000, z_coords/1000, tangential_wind_slice, levels=100)
Colorbar(fig[1, 2], limits=(0, 33), label="Tangential Wind (m/s)")
Makie.save("tangential_wind_profile_cross_section.png", fig)


## run model

simulation = Simulation(model; Δt=2, stop_time=12hours)

conjure_time_step_wizard!(simulation, cfl=0.7)


θˡⁱ = liquid_ice_potential_temperature(model)
qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ = model.microphysical_fields.qʳ
qᵛ = model.microphysical_fields.qᵛ
u, v, w = model.velocities

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s, max|u|: %.2f m/s, max w: %.2f m/s, min w: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(w), minimum(w))

    msg *= @sprintf(", max(qᵛ): %.2e, max(qᶜˡ): %.2e, max(qʳ): %.2e",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

max_w_ts = []
max_w_times = []

function collect_max_w(sim)
    push!(max_w_times, time(sim))
    push!(max_w_ts, maximum(w))
    return nothing
end

add_callback!(simulation, collect_max_w, TimeInterval(1minutes))
println("  Callbacks added:")
println("    - Progress report every 100 iterations")
println("    - Max vertical velocity collection every 1 minute")

println("\n=== Setting up output ===")
z = znodes(grid, Center())
k_5km = searchsortedfirst(z, 5000)
j_mid = Ny ÷ 2  # Middle of domain in y-direction
println("  Saving xy slices at z = $(z[k_5km]) m (k = $k_5km)")
println("  Saving xz slices at y = $(ynodes(grid, Center())[j_mid]) m (j = $j_mid)")

slice_outputs = (
    wxy = view(w, :, :, k_5km),
    qʳxy = view(qʳ, :, :, k_5km),
    qᶜˡxy = view(qᶜˡ, :, :, k_5km),
    wxz = view(w, :, j_mid, :),
    qʳxz = view(qʳ, :, j_mid, :),
    qᶜˡxz = view(qᶜˡ, :, j_mid, :),
)

slices_filename = "tropical_cyclone_and_rainband_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename=slices_filename,
                                                including = [:grid],
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)


run!(simulation)

## plot the slices

# Load the saved slice data
wxy_ts = FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts = FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts = FieldTimeSeries(slices_filename, "qᶜˡxy")
wxz_ts = FieldTimeSeries(slices_filename, "wxz")
qʳxz_ts = FieldTimeSeries(slices_filename, "qʳxz")
qᶜˡxz_ts = FieldTimeSeries(slices_filename, "qᶜˡxz")

times = wxy_ts.times
Nt = length(times)

println("\n=== Plotting slices ===")
println("  Loaded $Nt time snapshots from $slices_filename")
println("  Time range: $(prettytime(times[1])) to $(prettytime(times[end]))")

# Compute color limits (use same limits for xy and xz slices)
wlim = max(maximum(abs, wxy_ts), maximum(abs, wxz_ts)) / 2
qʳlim = max(maximum(qʳxy_ts), maximum(qʳxz_ts)) / 4
qᶜˡlim = max(maximum(qᶜˡxy_ts), maximum(qᶜˡxz_ts)) / 4

# Extract coordinates
x_coords = xnodes(grid, Center())
y_coords = ynodes(grid, Center())
z_coords = znodes(grid, Center())

# Create figure with 6 panels (3 xy slices on top, 3 xz slices on bottom)
fig = Figure(size=(1200, 800), fontsize=12)

# Top row: xy slices at z ≈ 5 km
axw_xy = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="w at z ≈ 5 km (m/s)")
axqᶜˡ_xy = Axis(fig[1, 2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qᶜˡ at z ≈ 5 km (kg/kg)")
axqʳ_xy = Axis(fig[1, 3], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qʳ at z ≈ 5 km (kg/kg)")

# Bottom row: xz slices at middle of domain
axw_xz = Axis(fig[2, 1], aspect=DataAspect(), xlabel="x (km)", ylabel="z (km)", title="w at y = center (m/s)")
axqᶜˡ_xz = Axis(fig[2, 2], aspect=DataAspect(), xlabel="x (km)", ylabel="z (km)", title="qᶜˡ at y = center (kg/kg)")
axqʳ_xz = Axis(fig[2, 3], aspect=DataAspect(), xlabel="x (km)", ylabel="z (km)", title="qʳ at y = center (kg/kg)")

# Use Observable for interactive plotting (can be animated)
n = Observable(1)
wxy_n = @lift wxy_ts[$n]
qᶜˡxy_n = @lift qᶜˡxy_ts[$n]
qʳxy_n = @lift qʳxy_ts[$n]
wxz_n = @lift wxz_ts[$n]
qᶜˡxz_n = @lift qᶜˡxz_ts[$n]
qʳxz_n = @lift qʳxz_ts[$n]
title_text = @lift "Tropical cyclone, t = " * prettytime(times[$n])

# Create heatmaps for xy slices
hmw_xy = heatmap!(axw_xy, x_coords/1000, y_coords/1000, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ_xy = heatmap!(axqᶜˡ_xy, x_coords/1000, y_coords/1000, qᶜˡxy_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ_xy = heatmap!(axqʳ_xy, x_coords/1000, y_coords/1000, qʳxy_n, colormap=:amp, colorrange=(0, qʳlim))

# Create heatmaps for xz slices
hmw_xz = heatmap!(axw_xz, x_coords/1000, z_coords/1000, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ_xz = heatmap!(axqᶜˡ_xz, x_coords/1000, z_coords/1000, qᶜˡxz_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ_xz = heatmap!(axqʳ_xz, x_coords/1000, z_coords/1000, qʳxz_n, colormap=:amp, colorrange=(0, qʳlim))

# Add colorbars
Colorbar(fig[3, 1], hmw_xy, vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmqᶜˡ_xy, vertical=false, label="qᶜˡ (kg/kg)")
Colorbar(fig[3, 3], hmqʳ_xy, vertical=false, label="qʳ (kg/kg)")

# Add title
fig[0, :] = Label(fig, title_text, fontsize=14, tellwidth=false)

# Save final time snapshot
n[] = Nt
Makie.save("tropical_cyclone_slices_final.png", fig)
println("  Saved final snapshot: tropical_cyclone_slices_final.png")

# Also save a few key time snapshots
key_times = 1:10:Nt
for (idx, t_idx) in enumerate(key_times)
    n[] = t_idx
    filename = "tropical_cyclone_slices_t$(idx).png"
    Makie.save(filename, fig)
    println("  Saved snapshot $idx/$(length(key_times)): $filename (t = $(prettytime(times[t_idx])))")
end
