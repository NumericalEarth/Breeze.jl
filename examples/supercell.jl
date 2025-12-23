# # Idealized splitting supercell
#
# This example simulates an idealized splitting supercell thunderstorm, following the
# test case described by [KlempEtAl2015](@citet). This highly idealized benchmark
# evaluates the model's ability to capture deep moist convection with cloud microphysics
# and strong updrafts.
#
# Splitting supercells are a classic phenomenon in severe convective storms: when a
# single updraft interacts with environmental wind shear, it divides into two
# counter-rotating cells—a right-mover and a left-mover—that propagate in opposite
# directions relative to the mean wind. This idealized configuration, with unidirectional
# shear and symmetric initial conditions, produces nearly symmetric splitting, making it
# an excellent test case for numerical models. Real-world supercells exhibit more complex
# behavior due to asymmetric shear profiles and environmental heterogeneity.
#
# ## Physical setup
#
# The simulation initializes a conditionally unstable atmosphere with a warm bubble perturbation
# that triggers deep convection. The idealized environment includes:
# - An realistic tropospheric potential temperature profile with a tropopause at 12 km
# - Moisture that decreases with height, with relative humidity of 25% above the tropopause
# - Unidirectional (purely zonal) wind shear in the lowest 5 km to promote symmetric storm splitting
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
#     \Delta\theta \cos^2\left(\frac{\pi}{2} R\right) & R < 1 \\
#     0 & R \geq 1
# \end{cases}
# ```
#
# where the normalized radius is
#
# ```math
# R = \sqrt{\left(\frac{r}{r_h}\right)^2 + \left(\frac{z - z_b}{r_v}\right)^2}
# ```
#
# with ``r = \sqrt{(x - x_b)^2 + (y - y_b)^2}`` the horizontal distance from the bubble center,
# ``\Delta\theta = 3 \, {\rm K}`` the perturbation amplitude, ``r_h = 10 \, {\rm km}`` the
# horizontal radius, ``r_v = 1.5 \, {\rm km}`` the vertical radius, and ``z_b = 1.5 \, {\rm km}``
# the bubble center height.
#
# ### Wind shear profile
#
# The zonal wind increases linearly with height up to the shear layer depth ``z_s = 5 \, {\rm km}``,
# with a smooth transition zone (Equations 15–16 in [KlempEtAl2015](@cite)). This unidirectional
# shear provides the environmental vorticity necessary for supercell development and mesocyclone
# formation.

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

using Oceananigans.Grids: znode, znodes
using Oceananigans: Center, Face
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Breeze.Thermodynamics: dry_air_gas_constant
using CUDA

using CloudMicrophysics

# Access extension module and import the Breeze-wired microphysics constructor

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

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

# ## Reference state and formulation
#
# We use the anelastic formulation with liquid-ice potential temperature thermodynamics.
# The reference state has surface pressure ``p_0 = 1000 \, {\rm hPa}`` and reference
# potential temperature ``θ_0 = 300 \, {\rm K}``:

p₀ = 100000  # Pa - surface pressure
θ₀ = 300     # K - reference potential temperature

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)

# ## Background atmosphere profiles
#
# The atmospheric stratification parameters define the troposphere-stratosphere transition
# following [KlempEtAl2015](@citet):

θₜᵣ = 343           # K - tropopause potential temperature
zₜᵣ = 12000         # m - tropopause height
Tₜᵣ = 213           # K - tropopause temperature

# Wind shear parameters control the low-level environmental wind profile:

zₛ = 5kilometers    # m - shear layer depth
uₛ = 30             # m/s - wind speed at top of shear layer
u_c = 15            # m/s - storm motion (Galilean translation speed)

# Extract thermodynamic constants for profile calculations:

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(constants)

# Background potential temperature profile (Equation 14 in [KlempEtAl2015](@cite)):

function θᵇᵍ(x, y, z)
    θ_troposphere = θ₀ + (θₜᵣ - θ₀) * (z / zₜᵣ)^(5/4)
    θ_stratosphere = θₜᵣ * exp(g / (cᵖᵈ * Tₜᵣ) * (z - zₜᵣ))
    return ifelse(z <= zₜᵣ, θ_troposphere, θ_stratosphere)
end

# Relative humidity profile (decreases with height, 25% above tropopause):

ℋᵇᵍ(z) = ifelse(z <= zₜᵣ, 1 - 3/4 * (z / zₜᵣ)^(5/4), 1/4)

# ## Visualization: initial thermodynamic profiles
#
# Visualize the background potential temperature and relative humidity profiles:

z_plot = range(0, Lz, length=200)
θ_profile = [θᵇᵍ(0, 0, z) for z in z_plot]
ℋ_profile = [ℋᵇᵍ(z) * 100 for z in z_plot]  # Convert to percentage

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
lines!(ax_rh, ℋ_profile, collect(z_plot), linewidth = 2, color = :blue)

save("supercell_thermo_profiles.png", fig_thermo)
fig_thermo

# Zonal wind profile with linear shear below ``z_s`` and smooth transition
# (Equations 15–16 in [KlempEtAl2015](@cite)):

function uᵢ(x, y, z)
    u_lower = uₛ * (z / zₛ) - u_c
    u_transition = (-4/5 + 3 * (z / zₛ) - 5/4 * (z / zₛ)^2) * uₛ - u_c
    u_upper = uₛ - u_c
    return ifelse(z < zₛ - 1000, u_lower,
           ifelse(abs(z - zₛ) <= 1000, u_transition, u_upper))
end

# Meridional wind is zero (no cross-flow):

vᵢ(x, y, z) = 0.0

# ## Visualization: background wind profile
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

save("supercell_wind_profile.png", fig_wind)
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

# ## Visualization: warm bubble perturbation
#
# Visualize the warm bubble perturbation on a vertical slice through the domain center:

x_slice = range(0, Lx, length=200)
z_slice = range(0, Lz, length=200)

θ_pert_slice = [θᵢ(x, yᵦ, z) - θᵇᵍ(x, yᵦ, z) for x in x_slice, z in z_slice]

fig_bubble = Figure(size=(800, 400))
ax_bubble = Axis(fig_bubble[1, 1],
                 xlabel = "x (km)",
                 ylabel = "Height (m)",
                 title = "Warm Bubble Perturbation θ′ (K) at y = Ly/2")

hm = heatmap!(ax_bubble, collect(x_slice) ./ 1000, collect(z_slice), θ_pert_slice,
              colormap = :thermal, colorrange = (0, Δθ))
Colorbar(fig_bubble[1, 2], hm, label = "θ′ (K)")

save("supercell_warm_bubble.png", fig_bubble)
fig_bubble

# ## Atmosphere model
#
# Create the atmosphere model with one-moment cloud microphysics from
# [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl),
# high-order WENO advection, and anisotropic minimum dissipation turbulence closure:

FT = eltype(grid)

cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)

weno = WENO(order=5)
bounds_preserving_weno = WENO(order=5, bounds=(0, 1))
momentum_advection = weno
scalar_advection = (ρθ = weno,
                    ρqᵗ = bounds_preserving_weno,
                    ρqᶜˡ = bounds_preserving_weno,
                    ρqʳ = bounds_preserving_weno)

closure = AnisotropicMinimumDissipation()

model = AtmosphereModel(grid; formulation, closure, microphysics, momentum_advection, scalar_advection)

# Initialize with background potential temperature to compute hydrostatic pressure:

set!(model, θ = θᵇᵍ)

# ## Water vapor initialization
#
# Compute the initial water vapor mass fraction from the saturation mass fraction
# and relative humidity profile. The saturation mixing ratio uses the Tetens formula:
#
# ```math
# q^{v*} = \frac{380}{p} \exp\left(17.27 \frac{T - 273}{T - 36}\right)
# ```

ph = Breeze.AtmosphereModels.compute_hydrostatic_pressure!(CenterField(grid), model)
T = model.temperature

qᵛᵢ = Field{Center, Center, Center}(grid)

# Transfer to CPU for scalar indexing (required when using GPU arrays):

ph_host = Array(parent(ph))
T_host = Array(parent(T))
qᵛᵢ_host = similar(ph_host)

# Compute initial water vapor from relative humidity and saturation mass fraction:

for k in axes(qᵛᵢ_host, 3), j in axes(qᵛᵢ_host, 2), i in axes(qᵛᵢ_host, 1)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    T_eq = @inbounds T_host[i, j, k]
    p_eq = @inbounds ph_host[i, j, k]
    qᵛ_sat = 380 / p_eq * exp(17.27 * ((T_eq - 273) / (T_eq - 36)))
    @inbounds qᵛᵢ_host[i, j, k] = ℋᵇᵍ(z) * qᵛ_sat
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
qᵛ = model.microphysical_fields.qᵛ
qʳ = model.microphysical_fields.qʳ

# ## Simulation
#
# Run for 2 hours with adaptive time stepping (CFL = 0.7) starting from ``\Delta t = 2`` s:

simulation = Simulation(model; Δt=2, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Progress callback to monitor simulation health:

function progress(sim)
    u, v, w = sim.model.velocities
    qᵛ = sim.model.microphysical_fields.qᵛ
    qᶜˡ = sim.model.microphysical_fields.qᶜˡ
    qʳ = sim.model.microphysical_fields.qʳ

    ρe = Breeze.AtmosphereModels.static_energy_density(sim.model)
    ρemean = mean(ρe)
    
    msg = @sprintf("Iter: %d, t: %s, Δt: %s, mean(ρe): %.6e J/kg, max|u|: %.2f m/s, max w: %.2f m/s, min w: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), ρemean,
                   maximum(abs, u), maximum(w), minimum(w))
    @info msg
    
    msg *= @sprintf(", max(qᵛ): %.5e, max(qᶜˡ): %.5e, max(qʳ): %.5e",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# Save horizontal slices at ``z \approx 5 \, {\rm km}`` for efficient visualization.
# We also record the maximum vertical velocity over time to track storm intensity:

times = Float64[]
max_w = Float64[]

function record_max_w(sim)
    push!(times, time(sim))
    push!(max_w, maximum(sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, record_max_w, TimeInterval(1minutes))

z = znodes(grid, Center())
k_5km = searchsortedfirst(z, 5000)
@info "Saving xy slices at z = $(z[k_5km]) m (k = $k_5km)"

w = model.velocities.w
slice_outputs = (
    wxy = view(w, :, :, k_5km),
    qʳxy = view(qʳ, :, :, k_5km),
    qᶜˡxy = view(qᶜˡ, :, :, k_5km),
)

slices_filename = "supercell_slices.jld2"
isfile(slices_filename) && rm(slices_filename; force=true)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename=slices_filename,
                                                 schedule = TimeInterval(1minutes),
                                                 overwrite_existing = true)

run!(simulation)

# ## Results: maximum vertical velocity
#
# The maximum updraft velocity is a key diagnostic for supercell intensity.
# Strong supercells typically develop updrafts exceeding 30–50 m/s [Emanuel1994](@cite):

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
          xlabel = "Time (s)",
          ylabel = "Maximum w (m/s)",
          title = "Maximum Vertical Velocity")
lines!(ax, times, max_w, linewidth = 2)

save("supercell_max_w.png", fig)
fig

# ## Animation: horizontal slices at 5 km
#
# We create a 3-panel animation showing the storm structure at mid-levels (``z \approx 5`` km):
# - Vertical velocity ``w``: reveals the updraft/downdraft structure
# - Cloud water ``q^{cl}``: shows the cloud boundaries
# - Rain water ``q^r``: indicates precipitation regions

wxy_ts = FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts = FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts = FieldTimeSeries(slices_filename, "qᶜˡxy")

times_ts = wxy_ts.times
Nt = length(times_ts)

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
title_text = @lift "Supercell at z ≈ 5 km, t = " * prettytime(times_ts[$n])

# Create heatmaps and colorbars:

hmw = heatmap!(axw, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ = heatmap!(axqᶜˡ, qᶜˡxy_n, colormap=:viridis, colorrange=(0, qᶜˡlim))
hmqʳ = heatmap!(axqʳ, qʳxy_n, colormap=:viridis, colorrange=(0, qʳlim))

Colorbar(slices_fig[2, 1], hmw, label="w (m/s)", vertical=false, flipaxis=false)
Colorbar(slices_fig[2, 2], hmqᶜˡ, label="qᶜˡ (kg/kg)", vertical=false, flipaxis=false)
Colorbar(slices_fig[4, 1], hmqʳ, label="qʳ (kg/kg)", vertical=false, flipaxis=false)

slices_fig[0, :] = Label(slices_fig, title_text, fontsize=18, tellwidth=false)

# Record the animation:

record(slices_fig, "supercell.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end

@info "Animation saved to supercell.mp4"

# ![](supercell.mp4)
