# # Splitting supercell
#
# This example simulates the development of a splitting supercell thunderstorm, following the
# idealized test case described by [Klemp et al. (2015)](@cite KlempEtAl2015) and the DCMIP2016
# supercell intercomparison by [Zarzycki et al. (2019)](@cite Zarzycki2019). This benchmark evaluates the model's
# ability to capture deep moist convection with warm-rain microphysics and strong updrafts.
#
# For microphysics we use the Kessler scheme, which includes prognostic cloud water
# and rain water with autoconversion, accretion, rain evaporation, and sedimentation processes.
# This is the same scheme used in the DCMIP2016 supercell intercomparison [Zarzycki et al. (2019)](@cite Zarzycki2019).
#
# ## Physical setup
#
# The simulation initializes a conditionally unstable atmosphere with a warm bubble perturbation
# that triggers deep convection. The environment includes:
# - A realistic tropospheric potential temperature profile with a tropopause at 12 km
# - Moisture that decreases with height, with relative humidity dropping above the tropopause
# - Wind shear in the lower 5 km to promote storm rotation and supercell development
#
# ### Potential temperature profile
#
# The background potential temperature follows a piecewise profile
# (Equation 14 in [Klemp et al. (2015)](@cite KlempEtAl2015)):
#
# ```math
# θ(z) = \begin{cases}
#     θ_0 + (θ_{\rm tr} - θ_0) \left(\frac{z}{z_{\rm tr}}\right)^{5/4} & z \leq z_{\rm tr} \\
#     θ_{\rm tr} \exp\left(\frac{g}{c_p^d T_{\rm tr}} (z - z_{\rm tr})\right) & z > z_{\rm tr}
# \end{cases}
# ```
#
# where ``θ_0 = 300 \, {\rm K}`` is the surface potential temperature,
# ``θ_{\rm tr} = 343 \, {\rm K}`` is the tropopause potential temperature,
# ``z_{\rm tr} = 12 \, {\rm km}`` is the tropopause height, and
# ``T_{\rm tr} = 213 \, {\rm K}`` is the tropopause temperature.
#
# ### Warm bubble perturbation
#
# A localized warm bubble triggers convection (Equations 17–18 in [Klemp et al. (2015)](@cite KlempEtAl2015)):
#
# ```math
# θ'(x, y, z) = \begin{cases}
#     Δθ \cos^2\left(\frac{\pi}{2} R\right) & R < 1 \\
#     0 & R \geq 1
# \end{cases}
# ```
#
# where ``R = \sqrt{(r/r_h)^2 + ((z-z_c)/r_z)^2}`` is the normalized radius,
# ``r = \sqrt{(x-x_c)^2 + (y-y_c)^2}`` is the horizontal distance from the bubble center,
# ``Δθ = 3 \, {\rm K}`` is the perturbation amplitude, ``r_h = 10 \, {\rm km}`` is the
# horizontal radius, and ``r_z = 1.5 \, {\rm km}`` is the vertical radius.
#
# ### Wind shear profile
#
# The zonal wind increases linearly with height up to the shear layer ``z_s = 5 \, {\rm km}``,
# with a smooth transition zone, providing the environmental shear necessary for supercell
# development and mesocyclone formation (Equations 15-16 in [Klemp et al. (2015)](@cite KlempEtAl2015)).

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, znodes

using CairoMakie
using CUDA
using Printf

# ## Domain and grid
#
# The domain is 168 km × 168 km × 20 km with 168 × 168 × 40 grid points, giving
# 1 km horizontal resolution and 500 m vertical resolution. The grid uses periodic
# lateral boundary conditions and bounded top/bottom boundaries.

Oceananigans.defaults.FloatType = Float32

Nx, Ny, Nz = 168, 168, 40
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and dynamics
#
# We define the anelastic reference state with surface pressure ``p_0 = 1000 \, {\rm hPa}``
# and reference potential temperature ``θ_0 = 300 \, {\rm K}``.

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 100000,
                                 potential_temperature = 300)

dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere profiles
#
# The atmospheric stratification parameters define the troposphere-stratosphere transition.

FT = eltype(grid)

θ₀ = 300        # K - surface potential temperature
θₜᵣ = 343       # K - tropopause potential temperature
zₜᵣ = 12000     # m - tropopause height
Tₜᵣ = 213       # K - tropopause temperature

# Wind shear parameters control the low-level environmental wind profile:

zₛ = 5kilometers  # m - shear layer height
uₛ = 30           # m/s - maximum shear wind speed
uᶜ = 15           # m/s - storm motion (Galilean translation speed)

# Extract thermodynamic constants for profile calculations:

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

# Background potential temperature profile (Equation 14 in [Klemp et al. (2015)](@cite KlempEtAl2015)):

function θ_background(z)
    θ_troposphere = θ₀ + (θₜᵣ - θ₀) * (z / zₜᵣ)^(5/4)
    θ_stratosphere = θₜᵣ * exp(g / (cᵖᵈ * Tₜᵣ) * (z - zₜᵣ))
    return (z <= zₜᵣ) * θ_troposphere + (z > zₜᵣ) * θ_stratosphere
end

# Relative humidity profile (decreases with height, 25% above tropopause):

ℋ_background(z) = (1 - 3/4 * (z / zₜᵣ)^(5/4)) * (z <= zₜᵣ) + 1/4 * (z > zₜᵣ)

# Zonal wind profile with linear shear below ``z_s`` and smooth transition (Equations 15-16):

function u_background(z)
    u_lower = uₛ * (z / zₛ) - uᶜ
    u_transition = (-4/5 + 3 * (z / zₛ) - 5/4 * (z / zₛ)^2) * uₛ - uᶜ
    u_upper = uₛ - uᶜ
    return (z < (zₛ - 1000)) * u_lower +
           (abs(z - zₛ) <= 1000) * u_transition +
           (z > (zₛ + 1000)) * u_upper
end

# ## Initial thermodynamic profiles
#
# Visualize the background potential temperature and relative humidity profiles
# that define the environmental stratification:

z_plot = range(0, Lz, length=200)
θ_profile = [θ_background(z) for z in z_plot]
ℋ_profile = [ℋ_background(z) * 100 for z in z_plot]

fig = Figure(size=(900, 500), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (km)", title="Background potential temperature")
lines!(axθ, θ_profile, z_plot ./ 1000, linewidth=2, color=:magenta)
hlines!(axθ, [zₜᵣ / 1000], color=:gray, linestyle=:dash, label="Tropopause")

axℋ = Axis(fig[1, 2], xlabel="ℋ (%)", ylabel="z (km)", title="Relative humidity profile")
lines!(axℋ, ℋ_profile, z_plot ./ 1000, linewidth=2, color=:dodgerblue)
hlines!(axℋ, [zₜᵣ / 1000], color=:gray, linestyle=:dash)

save("supercell_thermodynamic_profiles.png", fig) #src
fig

# ## Background wind profile
#
# The environmental wind shear promotes supercell rotation and mesocyclone formation.
# The zonal wind increases from -15 m/s at the surface to +15 m/s at 5 km (in a frame
# moving with the storm at 15 m/s):

u_profile = [u_background(z) for z in z_plot]

fig = Figure(size=(500, 500), fontsize=14)
axu = Axis(fig[1, 1], xlabel="u (m/s)", ylabel="z (km)", title="Background wind profile")
lines!(axu, u_profile, z_plot ./ 1000, linewidth=2, color=:orangered)
hlines!(axu, [zₛ / 1000], color=:gray, linestyle=:dash, label="Shear layer top")
vlines!(axu, [0], color=:black, linestyle=:dot)

save("supercell_wind_profile.png", fig) #src
fig

# ## Warm bubble perturbation
#
# The warm bubble parameters following Equations 17–18 in [Klemp et al. (2015)](@cite KlempEtAl2015):

Δθ = 3              # K - perturbation amplitude
rₕ = 10kilometers   # m - bubble horizontal radius
rᵥ = 1500           # m - bubble vertical radius
zᵦ = 1500           # m - bubble center height
xᵦ = Lx / 2         # m - bubble center x-coordinate
yᵦ = Ly / 2         # m - bubble center y-coordinate

# The total initial potential temperature combines the background profile with the
# cosine-squared warm bubble perturbation:

function θᵢ(x, y, z)
    θ_base = θ_background(z)
    r = sqrt((x - xᵦ)^2 + (y - yᵦ)^2)
    R = sqrt((r / rₕ)^2 + ((z - zᵦ) / rᵥ)^2)
    θ_pert = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ_base + θ_pert
end

uᵢ(x, y, z) = u_background(z)

# Visualize the warm bubble perturbation on a vertical slice through the domain center:

x_slice = range(0, Lx, length=200)
z_slice = range(0, Lz, length=200)

θ′_slice = [θᵢ(x, yᵦ, z) - θ_background(z) for x in x_slice, z in z_slice]

fig = Figure(size=(800, 300), fontsize=14)
ax = Axis(fig[1, 1], xlabel="x (km)", ylabel="z (km)",
          title="Warm bubble perturbation θ′ (K) at y = Ly/2", aspect=DataAspect())

hm = heatmap!(ax, x_slice ./ 1000, z_slice ./ 1000, θ′_slice,
              colormap=:thermal, colorrange=(0, Δθ))
Colorbar(fig[1, 2], hm, label="θ′ (K)")

save("supercell_warm_bubble.png", fig) #src
fig

# ## Model setup
#
# We use the DCMIP2016 Kessler microphysics scheme with high-order WENO advection
# and anisotropic minimum dissipation turbulence closure. The Kessler scheme includes
# prognostic cloud water and rain water with autoconversion, accretion, rain evaporation,
# and sedimentation processes.

microphysics = DCMIP2016KesslerMicrophysics()
advection = WENO(order=9, minimum_buffer_upwind_order=3)
closure = AnisotropicMinimumDissipation()

model = AtmosphereModel(grid; dynamics, closure, microphysics, advection, thermodynamic_constants=constants)

# ## Water vapor initialization
#
# We initialize the model with the background potential temperature to compute
# the hydrostatic pressure, then compute initial water vapor from relative humidity
# and saturation specific humidity using the Tetens formula:
#
# ```math
# q_v^* = \frac{380}{p} \exp\left(17.27 \frac{T - 273}{T - 36}\right)
# ```

set!(model, θ = (x, y, z) -> θ_background(z))

pₕ = Breeze.AtmosphereModels.compute_hydrostatic_pressure!(CenterField(grid), model)
T = model.temperature

# Transfer to CPU for scalar indexing (required when using GPU arrays):

pₕ_host = Array(parent(pₕ))
T_host = Array(parent(T))
qᵛ_host = similar(pₕ_host)

for k in axes(qᵛ_host, 3), j in axes(qᵛ_host, 2), i in axes(qᵛ_host, 1)
    zᵢⱼₖ = znode(i, j, k, grid, Center(), Center(), Center())
    Tᵢⱼₖ = @inbounds T_host[i, j, k]
    pᵢⱼₖ = @inbounds pₕ_host[i, j, k]
    qᵛ⁺ = 380 / pᵢⱼₖ * exp(17.27 * ((Tᵢⱼₖ - 273) / (Tᵢⱼₖ - 36)))
    @inbounds qᵛ_host[i, j, k] = ℋ_background(zᵢⱼₖ) * qᵛ⁺
end

qᵛ_init = CenterField(grid)
copyto!(parent(qᵛ_init), qᵛ_host)

# Set the full initial conditions (water vapor, potential temperature with bubble, and wind):

set!(model, qᵗ=qᵛ_init, θ=θᵢ, u=uᵢ)

# ## Simulation
#
# Run for 2 hours with adaptive time stepping (CFL = 0.7):

simulation = Simulation(model; Δt=2, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress
#
# We set up callbacks to monitor simulation health and collect diagnostics.
# The maximum vertical velocity is tracked during the simulation to avoid
# saving large 3D datasets.

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

# Collect maximum vertical velocity time series during simulation:

max_w_timeseries = Float64[]
max_w_times = Float64[]

function collect_max_w(sim)
    push!(max_w_times, time(sim))
    push!(max_w_timeseries, maximum(w))
    return nothing
end

add_callback!(simulation, collect_max_w, TimeInterval(1minutes))

# Save horizontal slices at z ≈ 5 km for animation:

z = znodes(grid, Center())
k_5km = searchsortedfirst(z, 5000)
@info "Saving xy slices at z = $(z[k_5km]) m (k = $k_5km)"

slice_outputs = (
    wxy = view(w, :, :, k_5km),
    qʳxy = view(qʳ, :, :, k_5km),
    qᶜˡxy = view(qᶜˡ, :, :, k_5km),
)

slices_filename = "splitting_supercell_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename=slices_filename,
                                                schedule = TimeInterval(1minutes),
                                                overwrite_existing = true)

run!(simulation)

# ## Results: maximum vertical velocity time series
#
# The maximum updraft velocity is a key diagnostic for supercell intensity.
# Strong supercells typically develop updrafts exceeding 30–50 m/s.
#
# Our simulated storm intensity is notably stronger than the DCMIP2016 intercomparison
# results reported by [Zarzycki et al. (2019)](@cite Zarzycki2019). One explanation is that
# no explicit numerical diffusion is applied in this simulation. As noted by
# [Klemp et al. (2015)](@cite KlempEtAl2015), the simulated storm intensity and structure
# are highly sensitive to numerical diffusion.

fig = Figure(size=(700, 400), fontsize=14)
ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Maximum w (m/s)", title="Maximum Vertical Velocity")
lines!(ax, max_w_times, max_w_timeseries, linewidth=2)

save("supercell_max_w.png", fig) #src
fig

# ## Animation: horizontal slices at z ≈ 5 km
#
# We create a 3-panel animation showing the storm structure at mid-levels:
# - Vertical velocity ``w``: reveals the updraft/downdraft structure
# - Cloud liquid ``qᶜˡ``: shows the cloud boundaries
# - Rain ``qʳ``: indicates precipitation regions
#
# The simulated supercell exhibits splitting behavior, with the initial storm
# dividing into right-moving and left-moving cells, consistent with the
# DCMIP2016 intercomparison results [Zarzycki et al. (2019)](@cite Zarzycki2019).

wxy_ts = FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts = FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts = FieldTimeSeries(slices_filename, "qᶜˡxy")

times = wxy_ts.times
Nt = length(times)

wlim = maximum(abs, wxy_ts) / 2
qʳlim = maximum(qʳxy_ts) / 4
qᶜˡlim = maximum(qᶜˡxy_ts) / 4

fig = Figure(size=(1200, 500), fontsize=14)

axw = Axis(fig[1, 1], aspect=1, xlabel="x (m)", ylabel="y (m)", title="Vertical velocity w (m/s)")
axqᶜˡ = Axis(fig[1, 2], aspect=1, xlabel="x (m)", ylabel="y (m)", title="Cloud water qᶜˡ (kg/kg)")
axqʳ = Axis(fig[1, 3], aspect=1, xlabel="x (m)", ylabel="y (m)", title="Rain qʳ (kg/kg)")

n = Observable(1)
wxy_n = @lift wxy_ts[$n]
qᶜˡxy_n = @lift qᶜˡxy_ts[$n]
qʳxy_n = @lift qʳxy_ts[$n]
title = @lift "Splitting supercell at z ≈ 5 km, t = " * prettytime(times[$n])

hmw = heatmap!(axw, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ = heatmap!(axqᶜˡ, qᶜˡxy_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ = heatmap!(axqʳ, qʳxy_n, colormap=:amp, colorrange=(0, qʳlim))

Colorbar(fig[2, 1], hmw, vertical=false)
Colorbar(fig[2, 2], hmqᶜˡ, vertical=false)
Colorbar(fig[2, 3], hmqʳ, vertical=false)

fig[0, :] = Label(fig, title, fontsize=18, tellwidth=false)

CairoMakie.record(fig, "splitting_supercell_slices.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end

@info "Animation saved to splitting_supercell_slices.mp4"

# ![](splitting_supercell_slices.mp4)
