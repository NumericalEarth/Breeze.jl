using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Breeze: WENO
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

arch = GPU()
Nx = Ny = 100
Nz = 128

x = y = (0, 2000Nx)
z = (0, 20500)

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()

## reference state

# Load moist tropical sounding from Dunion 2011
println("\n=== Loading Dunion 2011 moist tropical sounding ===")
sounding = CSV.read("dunion2011_moist_tropical_MT.csv", DataFrame)
Tˢ_data = 273.15 .+ Float64.(sounding[:, :Temperature_C])
pˢ_data = 100 .* Float64.(sounding[:, :Pressure_hPa])
zˢ_data = Float64.(sounding[:, :GPH_m])
θˢ_data = Float64.(sounding[:, :Theta_K])
qᵗˢ_data = 1000 .* Float64.(sounding[:, :Mixing_ratio_g_kg])

pˢ_data = reverse(pˢ_data)
zˢ_data = reverse(zˢ_data)
θˢ_data = reverse(θˢ_data)
qᵗˢ_data = reverse(qᵗˢ_data)

# --- <--
#  o  zˢ[5] 
# ---
#  o  zˢ[4] 
# ---
#  o  zˢ[3] 
# ---
#  o  zˢ[2]  
# --- zˢ[1]

zˢᶠ = (zˢ_data[1:end-1] .+ zˢ_data[2:end]) ./ 2
Δz = zˢ_data[end] - zˢ_data[end-1]
zˢ_data[1] = 0
push!(zˢᶠ, zˢᶠ[end] + Δz)
Nzˢ = length(zˢ_data) - 1
sounding_grid = RectilinearGrid(size=Nzˢ, z=zˢᶠ, topology=(Flat, Flat, Bounded))

# Pass grid (and location) so BCs are only set for Bounded direction; Flat directions get no east/west/north/south BCs
sounding_loc = (Center(), Center(), Center())
Tˢ_bcs = FieldBoundaryConditions(sounding_grid, sounding_loc, bottom=ValueBoundaryCondition(Tˢ_data[1]))
θˢ_bcs = FieldBoundaryConditions(sounding_grid, sounding_loc, bottom=ValueBoundaryCondition(θˢ_data[1]))
Tˢ = CenterField(sounding_grid, boundary_conditions=Tˢ_bcs)
θˢ = CenterField(sounding_grid, boundary_conditions=θˢ_bcs)
qᵗˢ = CenterField(sounding_grid)
set!(Tˢ, Tˢ_data[2:end])
set!(θˢ, θˢ_data[2:end])
set!(qᵗˢ, qᵗˢ_data[2:end])

## make a plot
fig = Figure()
axθˢ = Axis(fig[1, 1], xlabel="Potential Temperature (K)", ylabel="Height (m)", title="Dunion 2011 Sounding: Potential Temperature")
axqᵗˢ = Axis(fig[1, 2], xlabel="Total Moisture (kg/kg)", ylabel="Height (m)", title="Dunion 2011 Sounding: Total Moisture")
lines!(axθˢ, θˢ)
lines!(axqᵗˢ, qᵗˢ)
Makie.save("dunion2011_sounding.png", fig)

## set reference state
reference_state = ReferenceState(grid, constants,
                                 surface_pressure = pˢ_data[1],
                                 potential_temperature = θˢ_data[1])
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
forcing = (; ρw=sponge)
boundary_conditions = (ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

## model
println("\n=== Creating AtmosphereModel ===")
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics
cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)
weno = WENO(order=9)
bounds_preserving_weno = WENO(order=9, bounds=(0, 1))
momentum_advection = weno
scalar_advection = (ρθ = weno,
                    ρqᵗ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno)
coriolis = FPlane(f=5e-5)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection, scalar_advection, forcing, boundary_conditions)


model = AtmosphereModel(grid; dynamics, coriolis, microphysics,
                        momentum_advection, scalar_advection, boundary_conditions)



###########################
# Initial conditions
###########################

RMW = 31000
V_RMW = 43 # m/s
a = 0.5
Δz = znodes(grid, Center())[2] - znodes(grid, Center())[1]

# Center of domain for perturbation
x_center = (x[1] + x[2]) / 2
y_center = (y[1] + y[2]) / 2
println("  Perturbation center: ($(x_center/1000), $(y_center/1000)) km")

# Convert mixing ratio from g/kg to kg/kg, then to specific humidity qᵗ
# Specific humidity q = w / (1 + w) where w is mixing ratio in kg/kg
qᵗ_sounding = qᵗˢ_data ./ (1 .+ qᵗˢ_data)

# Create interpolations for the sounding profiles
θ_sounding_interp = linear_interpolation((zˢ_data), (θˢ_data))
qᵗ_sounding_interp = linear_interpolation((zˢ_data), (qᵗˢ_data))
T_sounding_interp = linear_interpolation((zˢ_data), (Tˢ_data))

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
    # Linear adjustment factor for sloping eyewall (Stern and Nolan 2009): scale wind so that
    # when RMW increases with height, peak wind is reduced consistently with angular momentum.
    v_adjustment_factor = RMW / (RMW + radius_of_eyewall_adjustment(z))
    r = sqrt((x-x_center)^2 + (y-y_center)^2)
    if r <= RMW + radius_of_eyewall_adjustment(z) && z < 16000
        return V_RMW * v_adjustment_factor * r / (RMW + radius_of_eyewall_adjustment(z))
    elseif z < 16000
        return V_RMW * v_adjustment_factor * ((RMW + radius_of_eyewall_adjustment(z)) / r)^a
    else
        return 0
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
    for r_idx in (Nr-1):-1:1
        r = rrange[r_idx]
        # Compute pressure gradient from gradient wind balance
        v_tang = tangential_wind(x_center + r, y_center, z_k)
        # Gradient wind: dp/dr = ρ * (f*v + v²/r)
        if r_idx == Nr
            ρ = p_background / (R * T_k)
        else
            ρ = p[k, r_idx+1] / (R * T_k)
        end
        ρ = p_background / (R * T_k)
        
        dp_dr = ρ * (v_tang * coriolis.f + v_tang^2 / max(r, 100))
        dp = dp_dr * ∂r
        p[k, r_idx] = p[k, r_idx + 1] + dp
    end
end
p_outer = p[:, 1]
p=reverse(p, dims=2)



function p_func(x, y, z)
    radius = sqrt((x - x_center)^2 + (y - y_center)^2)
    z_clamped = clamp(z, minimum(znodes(grid, Center())), maximum(znodes(grid, Center())))
    z_idx = searchsortedfirst(znodes(grid, Center()), z_clamped)
    z_idx = clamp(z_idx, 1, Nz_p)
    r_idx = searchsortedfirst(rrange_asc, radius)
    r_idx = clamp(r_idx, 1, Nr)
    return p[z_idx, r_idx]
end

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
    z_idx = searchsortedfirst(znodes(grid, Center()), z_clamped)
    
    # Use temperature from sounding to compute reference pressure
    p_ref = p_outer[z_idx]
    

    # Return background potential temperature from sounding plus vortex perturbation
    return θ_background * ( p_ref/(p_func(x, y, z)))
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


###########################
# Figure generation
###########################

# Shared coordinates (km) for consistent axis labels
x_coords = xnodes(grid, Center())
y_coords = ynodes(grid, Center())
z_coords = znodes(grid, Center())
x_km = x_coords / 1000
y_km = y_coords / 1000
z_km = z_coords / 1000

## Initial conditions and diagnostics

# Pressure deficit (r–z cross section)
fig = Figure()
fig[0, :] = Label(fig, "Pressure deficit (hPa)", fontsize=14, tellwidth=false)
data = (p'/100 .- p_outer'/100)'  # transpose for (height, radius)
levels = minimum(data):5:maximum(data)
ax = Axis(fig[1, 1], xlabel="Radius (km)", ylabel="Height (km)")
contourf!(ax, rrange_asc/1000, z_km, data', levels=levels, colormap=:viridis)
Colorbar(fig[1, 2], limits=(minimum(levels), maximum(levels)), label="Pressure (hPa)", colormap=:viridis)
Makie.save("pressure_deficit_profile.png", fig)

# θ perturbation plan view at z ≈ surface
fig = Figure()
fig[0, :] = Label(fig, "Potential temperature perturbation at z ≈ surface (K)", fontsize=14, tellwidth=false)
data = Array(interior(liquid_ice_potential_temperature(model), :, :, 2)) .- θ_sounding_interp(z_coords[2])
limits = minimum(data):0.5:maximum(data)+0.5
ax = Axis(fig[1, 1], xlabel="x (km)", ylabel="y (km)")
contourf!(ax, x_km, y_km, data, levels=limits)
Colorbar(fig[1, 2], limits=(minimum(limits), maximum(limits)), label="θ (K)")
Makie.save("theta_init.png", fig)

# θ perturbation cross section (y–z)
θ_background = [θ_sounding_interp(z_coords[k]) for k in 1:Nz]
data = Array(interior(liquid_ice_potential_temperature(model), Nx÷2, :, :)) .- θ_background'
limits = minimum(data):0.5:maximum(data)+0.5
fig = Figure()
fig[0, :] = Label(fig, "Potential temperature perturbation cross section (K)", fontsize=14, tellwidth=false)
ax = Axis(fig[1, 1], xlabel="y (km)", ylabel="z (km)")
contourf!(ax, y_km, z_km, data, levels=limits)
Colorbar(fig[1, 2], limits=(minimum(limits), maximum(limits)), label="θ (K)")
Makie.save("theta_init_cross_section.png", fig)

# Tangential wind plan view at z ≈ surface
data = (Array(interior(model.velocities.u, :, :, 1)).^2 + Array(interior(model.velocities.v, :, :, 1)).^2).^0.5
limits = minimum(data):0.5:maximum(data)+0.5
fig = Figure()
fig[0, :] = Label(fig, "Tangential wind at z ≈ surface (m/s)", fontsize=14, tellwidth=false)
ax = Axis(fig[1, 1], xlabel="x (km)", ylabel="y (km)")
contourf!(ax, x_km, y_km, data, levels=limits)
Colorbar(fig[1, 2], limits=(minimum(limits), maximum(limits)), label="|v| (m/s)")
Makie.save("tangential_wind_profile.png", fig)

# Tangential wind cross section (y–z)
data = (Array(interior(model.velocities.u, Nx÷2, :, :)).^2 + Array(interior(model.velocities.v, Nx÷2, :, :)).^2).^0.5
limits = minimum(data):0.5:maximum(data)+0.5
fig = Figure()
fig[0, :] = Label(fig, "Tangential wind cross section (m/s)", fontsize=14, tellwidth=false)
ax = Axis(fig[1, 1], xlabel="y (km)", ylabel="z (km)")
contourf!(ax, y_km, z_km, data, levels=limits)
Colorbar(fig[1, 2], limits=(0, 33), label="|v| (m/s)")
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

## Simulation output (slices)

# Load the saved slice data
wxy_ts = Oceananigans.OutputReaders.FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts = Oceananigans.OutputReaders.FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts = Oceananigans.OutputReaders.FieldTimeSeries(slices_filename, "qᶜˡxy")
wxz_ts = Oceananigans.OutputReaders.FieldTimeSeries(slices_filename, "wxz")
qʳxz_ts = Oceananigans.OutputReaders.FieldTimeSeries(slices_filename, "qʳxz")
qᶜˡxz_ts = Oceananigans.OutputReaders.FieldTimeSeries(slices_filename, "qᶜˡxz")

times = wxy_ts.times
Nt = length(times)

println("\n=== Plotting slices ===")
println("  Loaded $Nt time snapshots from $slices_filename")
println("  Time range: $(prettytime(times[1])) to $(prettytime(times[end]))")

# Compute color limits (use same limits for xy and xz slices)
wlim = 5
qʳlim = max(maximum(qʳxy_ts), maximum(qʳxz_ts)) / 4
qᶜˡlim = max(maximum(qᶜˡxy_ts), maximum(qᶜˡxz_ts)) / 4

# Create figure with 6 panels (3 xy slices on top, 3 xz slices on bottom)
fig = Figure(size=(1200, 800), fontsize=12)
n = Observable(1)
fig[0, :] = Label(fig, @lift("Tropical cyclone and rainband: w, qᶜˡ, qʳ — t = " * prettytime(times[$n])),
                  fontsize=14, tellwidth=false)

# Top row: xy slices at z ≈ 5 km
axw_xy = Axis(fig[1, 1], aspect=1, xlabel="x (km)", ylabel="y (km)", title="w at z ≈ 5 km (m/s)")
axqᶜˡ_xy = Axis(fig[1, 2], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qᶜˡ at z ≈ 5 km (kg/kg)")
axqʳ_xy = Axis(fig[1, 3], aspect=1, xlabel="x (km)", ylabel="y (km)", title="qʳ at z ≈ 5 km (kg/kg)")

# Bottom row: xz slices at middle of domain
axw_xz = Axis(fig[2, 1], aspect=DataAspect(), xlabel="x (km)", ylabel="z (km)", title="w at y = center (m/s)")
axqᶜˡ_xz = Axis(fig[2, 2], aspect=DataAspect(), xlabel="x (km)", ylabel="z (km)", title="qᶜˡ at y = center (kg/kg)")
axqʳ_xz = Axis(fig[2, 3], aspect=DataAspect(), xlabel="x (km)", ylabel="z (km)", title="qʳ at y = center (kg/kg)")

# Use Observable for interactive plotting (can be animated)
wxy_n = @lift wxy_ts[$n]
qᶜˡxy_n = @lift qᶜˡxy_ts[$n]
qʳxy_n = @lift qʳxy_ts[$n]
wxz_n = @lift wxz_ts[$n]
qᶜˡxz_n = @lift qᶜˡxz_ts[$n]
qʳxz_n = @lift qʳxz_ts[$n]

# Create heatmaps for xy slices
hmw_xy = heatmap!(axw_xy, x_km, y_km, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ_xy = heatmap!(axqᶜˡ_xy, x_km, y_km, qᶜˡxy_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ_xy = heatmap!(axqʳ_xy, x_km, y_km, qʳxy_n, colormap=:amp, colorrange=(0, qʳlim))

# Create heatmaps for xz slices
hmw_xz = heatmap!(axw_xz, x_km, z_km, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ_xz = heatmap!(axqᶜˡ_xz, x_km, z_km, qᶜˡxz_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ_xz = heatmap!(axqʳ_xz, x_km, z_km, qʳxz_n, colormap=:amp, colorrange=(0, qʳlim))

# Add colorbars
Colorbar(fig[3, 1], hmw_xy, vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmqᶜˡ_xy, vertical=false, label="qᶜˡ (kg/kg)")
Colorbar(fig[3, 3], hmqʳ_xy, vertical=false, label="qʳ (kg/kg)")

# Save final time snapshot
n[] = Nt
Makie.save("tropical_cyclone_slices_final.png", fig)
println("  Saved final snapshot: tropical_cyclone_slices_final.png")

# Also save a few key time snapshots
key_times = 1:1:Nt
for (idx, t_idx) in enumerate(key_times)
    n[] = t_idx
    filename = "tropical_cyclone_slices_t$(idx).png"
    Makie.save(filename, fig)
    println("  Saved snapshot $idx/$(length(key_times)): $filename (t = $(prettytime(times[t_idx])))")
end
