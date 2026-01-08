# # Supercell thunderstorm
#
# This example simulates the development of a splitting supercell thunderstorm, following the idealized
# test case described by [KlempEtAl2015](@citet) ("Idealized global nonhydrostatic atmospheric
# test cases on a reduced-radius sphere"). This benchmark evaluates the model's ability to
# capture deep moist convection with cloud microphysics, and strong updrafts.
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
# The background potential temperature follows a piecewise profile:
#
# ```math
# θ^{\rm bg}(z) = \begin{cases}
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
# A localized warm bubble triggers convection (Equations 17–18 in [KlempEtAl2015](@cite)):
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
# development and mesocyclone formation.

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

using Oceananigans.Grids: znode
using Oceananigans: Center, Face
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Breeze.Thermodynamics: dry_air_gas_constant
using CUDA

using CloudMicrophysics
import Breeze: Breeze

# Access extension module and define aliases to avoid namespace conflicts:

const BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
const BreezeOneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics

# ## Grid configuration
#
# The domain is 168 km × 168 km × 20 km with 168 × 168 × 40 grid points, giving
# 1 km horizontal resolution and 500 m vertical resolution. The grid uses periodic
# lateral boundary conditions and bounded top/bottom boundaries.

Nx, Ny, Nz = 168, 168, 40
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Thermodynamic parameters
#
# We define the reference state with surface pressure ``p_0 = 1000 \, {\rm hPa}`` and
# reference potential temperature ``θ_0 = 300 \, {\rm K}``:

p₀ = 100000  # Pa - surface pressure
θ₀ = 300     # K - reference potential temperature
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere profiles
#
# The atmospheric stratification parameters define the troposphere-stratosphere transition:

θₜᵣ = 343           # K - tropopause potential temperature
zₜᵣ = 12000         # m - tropopause height
Tₜᵣ = 213           # K - tropopause temperature

# Wind shear parameters control the low-level environmental wind profile:

zₛ = 5kilometers    # m - shear layer height
uₛ = 30             # m/s - maximum shear wind speed
u_c = 15            # m/s - storm motion (Galilean translation speed)

# Extract thermodynamic constants for profile calculations:

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(constants)

# Background potential temperature profile (Equation 14 in [KlempEtAl2015](@cite)):

function θᵇᵍ(x, y, z)
    θ_troposphere = θ₀ + (θₜᵣ - θ₀) * (z / zₜᵣ)^(5/4)
    θ_stratosphere = θₜᵣ * exp(g / (cᵖᵈ * Tₜᵣ) * (z - zₜᵣ))
    return (z <= zₜᵣ) * θ_troposphere + (z > zₜᵣ) * θ_stratosphere
end

# Relative humidity profile (decreases with height, 25% above tropopause):

RHᵇᵍ(z) = (1 - 3/4 * (z / zₜᵣ)^(5/4)) * (z <= zₜᵣ) + 1/4 * (z > zₜᵣ)

# ## Plot: Initial thermodynamic profiles
#
# Visualize the background potential temperature and relative humidity profiles:

z_plot = range(0, Lz, length=200)
θ_profile = [θᵇᵍ(0, 0, z) for z in z_plot]
RH_profile = [RHᵇᵍ(z) * 100 for z in z_plot]  # Convert to percentage

fig_thermo = Figure(size=(900, 600))

ax_theta = Axis(fig_thermo[1, 1],
                xlabel = "Potential temperature θ (K)",
                ylabel = "Height (m)",
                title = "Background Potential Temperature")
lines!(ax_theta, θ_profile, collect(z_plot), linewidth = 2, color = :red)

ax_rh = Axis(fig_thermo[1, 2],
             xlabel = "Relative humidity (%)",
             ylabel = "Height (m)",
             title = "Relative Humidity Profile")
lines!(ax_rh, RH_profile, collect(z_plot), linewidth = 2, color = :blue)

save("supercell_thermo_profiles.png", fig_thermo) #src
fig_thermo

# Zonal wind profile with linear shear below ``z_s`` and smooth transition (Equation 15-16):

function uᵢ(x, y, z)
    u_lower = uₛ * (z / zₛ) - u_c
    u_transition = (-4/5 + 3 * (z / zₛ) - 5/4 * (z / zₛ)^2) * uₛ - u_c
    u_upper = uₛ - u_c
    return (z < (zₛ - 1000)) * u_lower +
           (abs(z - zₛ) <= 1000) * u_transition +
           (z > (zₛ + 1000)) * u_upper
end

# Meridional wind is zero (no cross-flow):

vᵢ(x, y, z) = 0.0

# ## Plot: Background wind profile
#
# Visualize the environmental wind shear that promotes supercell rotation:

u_profile = [uᵢ(0, 0, z) for z in z_plot]
v_profile = [vᵢ(0, 0, z) for z in z_plot]

fig_wind = Figure(size=(500, 600))
ax_wind = Axis(fig_wind[1, 1],
               xlabel = "Wind speed (m/s)",
               ylabel = "Height (m)",
               title = "Background Wind Profile")

lines!(ax_wind, u_profile, collect(z_plot), label = "u (zonal)", linewidth = 2)
lines!(ax_wind, v_profile, collect(z_plot), label = "v (meridional)", linewidth = 2, linestyle = :dash)
axislegend(ax_wind, position = :rb)

save("supercell_wind_profile.png", fig_wind) #src
fig_wind

# ## Warm bubble initial perturbation
#
# The warm bubble parameters following Equations 17–18 in [KlempEtAl2015](@cite):

Δθ = 3              # K - perturbation amplitude
rₕ = 10kilometers   # m - bubble horizontal radius
rᵥ = 1500           # m - bubble vertical radius
zᵦ = 1500           # m - bubble center height
xᵦ = Lx / 2         # m - bubble center x-coordinate
yᵦ = Ly / 2         # m - bubble center y-coordinate

# The total initial potential temperature combines the background profile with the
# cosine-squared warm bubble perturbation:

function θᵢ(x, y, z)
    θ_base = θᵇᵍ(x, y, z)
    r = sqrt((x - xᵦ)^2 + (y - yᵦ)^2)
    R = sqrt((r / rₕ)^2 + ((z - zᵦ) / rᵥ)^2)
    θ_pert = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ_base + θ_pert
end

# ## Plot: Warm bubble perturbation
#
# Visualize the warm bubble perturbation on a vertical slice through the domain center:

x_slice = range(0, Lx, length=200)
z_slice = range(0, Lz, length=200)  # Focus on lower atmosphere where bubble is located

θ_pert_slice = [θᵢ(x, yᵦ, z) - θᵇᵍ(x, yᵦ, z) for x in x_slice, z in z_slice]

fig_bubble = Figure(size=(800, 400))
ax_bubble = Axis(fig_bubble[1, 1],
                 xlabel = "x (km)",
                 ylabel = "Height (m)",
                 title = "Warm Bubble Perturbation θ' (K) at y = Ly/2")

hm = heatmap!(ax_bubble, collect(x_slice) ./ 1000, collect(z_slice), θ_pert_slice,
              colormap = :thermal, colorrange = (0, Δθ))
Colorbar(fig_bubble[1, 2], hm, label = "θ' (K)")

save("supercell_warm_bubble.png", fig_bubble) #src
fig_bubble

# ## Model initialization
#
# Create the atmosphere model with one-moment cloud microphysics from CloudMicrophysics.jl,
# high-order WENO advection, and anisotropic minimum dissipation turbulence closure:

microphysics = BreezeOneMomentCloudMicrophysics()
advection = WENO(order=9, minimum_buffer_upwind_order=3)
closure = AnisotropicMinimumDissipation()
model = AtmosphereModel(grid; dynamics, closure, microphysics, advection)

# Initialize with background potential temperature to compute hydrostatic pressure:

set!(model, θ = θᵇᵍ)

# ## Water vapor initialization
#
# Compute the initial water vapor mixing ratio from the saturation mixing ratio
# and relative humidity profile. The saturation mixing ratio uses the Tetens formula:
#
# ```math
# q_v^* = \frac{380}{p} \exp\left(17.27 \frac{T - 273}{T - 36}\right)
# ```

ph = Breeze.AtmosphereModels.compute_hydrostatic_pressure!(CenterField(grid), model)
T = model.temperature

qᵛᵢ = Field{Center, Center, Center}(grid)

# Transfer to CPU for scalar indexing (required when using GPU arrays):

ph_host = Array(parent(ph))
T_host = Array(parent(T))
qᵛᵢ_host = similar(ph_host)

# Compute initial water vapor from relative humidity and saturation mixing ratio:

for k in axes(qᵛᵢ_host, 3), j in axes(qᵛᵢ_host, 2), i in axes(qᵛᵢ_host, 1)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    T_eq = @inbounds T_host[i, j, k]
    p_eq = @inbounds ph_host[i, j, k]
    qᵛ_sat = 380 / p_eq * exp(17.27 * ((T_eq - 273) / (T_eq - 36)))
    @inbounds qᵛᵢ_host[i, j, k] = RHᵇᵍ(z) * qᵛ_sat
end

copyto!(parent(qᵛᵢ), qᵛᵢ_host)

# Set the full initial conditions (water vapor, potential temperature with bubble, and wind):

set!(model, qᵗ = qᵛᵢ, θ = θᵢ, u = uᵢ)

# Compute potential temperature perturbation for diagnostics:

θ = Breeze.AtmosphereModels.liquid_ice_potential_temperature(model)
θᵇᵍf = CenterField(grid)
set!(θᵇᵍf, (x, y, z) -> θᵇᵍ(x, y, z))
θ′ = θ - θᵇᵍf

# Extract microphysical fields for output:

qᶜˡ = model.microphysical_fields.qᶜˡ
qᶜⁱ = model.microphysical_fields.qᶜⁱ
qᵛ = model.microphysical_fields.qᵛ

# ## Simulation
#
# Run for 2 hours with adaptive time stepping (CFL = 0.7) starting from Δt = 2 s:

simulation = Simulation(model; Δt=2, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Progress callback to monitor simulation health:

function progress(sim)
    u, v, w = sim.model.velocities
    qᵛ = model.microphysical_fields.qᵛ
    qᶜˡ = model.microphysical_fields.qᶜˡ
    qᶜⁱ = model.microphysical_fields.qᶜⁱ

    ρe = Breeze.AtmosphereModels.static_energy_density(sim.model)
    ρemean = mean(ρe)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, mean(ρe): %.6e J/kg, max|u|: %.5f m/s, max w: %.5f m/s, min w: %.5f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), ρemean,
                   maximum(abs, u), maximum(w), minimum(w))
    @info msg

    msg *= @sprintf(", max(qᵛ): %.5e, max(qᶜˡ): %.5e, max(qᶜⁱ): %.5e",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qᶜⁱ))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# Save full 3D fields for post-processing analysis:

outputs = merge(model.velocities, model.tracers, (; θ, θ′, qᶜˡ, qᶜⁱ, qᵛ))

filename = "supercell.jld2"
ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)
simulation.output_writers[:jld2] = ow

run!(simulation)

# ## Results: maximum vertical velocity time series
#
# The maximum updraft velocity is a key diagnostic for supercell intensity.
# Strong supercells typically develop updrafts exceeding 30–50 m/s:

wt = FieldTimeSeries(filename, "w")
times = wt.times
max_w = [maximum(wt[n]) for n in 1:length(times)]

fig = Figure()
ax = Axis(fig[1, 1],
          xlabel = "Time [s]",
          ylabel = "Max w [m/s]",
          title = "Maximum Vertical Velocity",
          xticks = 0:900:maximum(times))
lines!(ax, times, max_w)

save("max_w_timeseries.png", fig) #src
fig

# ## Animation: horizontal slices at 5 km
#
# We create a 3-panel animation showing the storm structure at mid-levels (z ≈ 5 km):
# - Vertical velocity ``w``: reveals the updraft/downdraft structure
# - Cloud water ``q^{cl}``: shows the cloud boundaries
# - Rain water ``q^r``: indicates precipitation regions

wxy_ts = FieldTimeSeries("supercell_slices.jld2", "wxy")
qʳxy_ts = FieldTimeSeries("supercell_slices.jld2", "qʳxy")
qᶜˡxy_ts = FieldTimeSeries("supercell_slices.jld2", "qᶜˡxy")

times = wxy_ts.times
Nt = length(times)

# Set color limits for visualization:

wlim = 25       # m/s - vertical velocity range
qʳlim = 0.01    # kg/kg - rain water range
qᶜˡlim = 0.001  # kg/kg - cloud water range

# Create the figure with 3 panels:

slices_fig = Figure(size=(900, 1000), fontsize=14)

axw = Axis(slices_fig[1, 1], xlabel="x (m)", ylabel="y (m)", title="Vertical velocity w")
axqᶜˡ = Axis(slices_fig[1, 2], xlabel="x (m)", ylabel="y (m)", title="Cloud water qᶜˡ")
axqʳ = Axis(slices_fig[3, 1], xlabel="x (m)", ylabel="y (m)", title="Rain water qʳ")

# Set up observables for animation:

n = Observable(1)
wxy_n = @lift wxy_ts[$n]
qᶜˡxy_n = @lift qᶜˡxy_ts[$n]
qʳxy_n = @lift qʳxy_ts[$n]
title_text = @lift "Supercell: Horizontal slices at z ≈ 5 km, t = " * prettytime(times[$n])

# Create heatmaps and colorbars:

hmw = heatmap!(axw, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ = heatmap!(axqᶜˡ, qᶜˡxy_n, colormap=:viridis, colorrange=(0, qᶜˡlim))
hmqʳ = heatmap!(axqʳ, qʳxy_n, colormap=:viridis, colorrange=(0, qʳlim))

Colorbar(slices_fig[2, 1], hmw, label="w (m/s)", vertical=false, flipaxis=false)
Colorbar(slices_fig[2, 2], hmqᶜˡ, label="qᶜˡ (kg/kg)", vertical=false, flipaxis=false)
Colorbar(slices_fig[4, 1], hmqʳ, label="qʳ (kg/kg)", vertical=false, flipaxis=false)

slices_fig[0, :] = Label(slices_fig, title_text, fontsize=18, tellwidth=false)

# Record the animation:

CairoMakie.record(slices_fig, "supercell_horizontal_5km.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end

@info "Animation saved to supercell_horizontal_5km.mp4"

# ![](supercell_horizontal_5km.mp4)
