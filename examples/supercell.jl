# # Supercell thunderstorm
#
# This example simulates the development of a supercell thunderstorm, following the idealized
# test case described by [KlempEtAl2015](@citet) ("Idealized global nonhydrostatic atmospheric
# test cases on a reduced-radius sphere"). This benchmark evaluates the model's ability to
# capture deep moist convection with rotation, cloud microphysics, and strong updrafts.
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
# The domain is 168 km × 168 km × 20 km with 336 × 336 × 40 grid points, giving
# 500 m horizontal resolution and 500 m vertical resolution. The grid uses periodic
# lateral boundary conditions and bounded top/bottom boundaries.

Nx, Ny, Nz = 336, 336, 40
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

pᵣ = 100000  # Pa - surface pressure
θᵣ = 300     # K - reference potential temperature
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo, surface_pressure=pᵣ, potential_temperature=θᵣ)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)

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

g = thermo.gravitational_acceleration
cᵖᵈ = thermo.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(thermo)

# Background potential temperature profile (Equation 14 in [KlempEtAl2015](@cite)):

function θᵢ₀(x, y, z)
    θ_troposphere = θᵣ + (θₜᵣ - θᵣ) * (z / zₜᵣ)^(5/4)
    θ_stratosphere = θₜᵣ * exp(g / (cᵖᵈ * Tₜᵣ) * (z - zₜᵣ))
    return (z <= zₜᵣ) * θ_troposphere + (z > zₜᵣ) * θ_stratosphere
end

# Relative humidity profile (decreases with height, 25% above tropopause):

RHᵢ(z) = (1 - 3/4 * (z / zₜᵣ)^(5/4)) * (z <= zₜᵣ) + 1/4 * (z > zₜᵣ)

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

z_plot = range(0, Lz, length=200)
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

# ## Plot: Skew-T Log-P diagram
#
# The Skew-T Log-P diagram is a standard meteorological tool for visualizing the
# atmospheric thermodynamic profile. Temperature lines are skewed 45° to the right,
# and pressure uses a logarithmic scale.
#
# We compute temperature and dewpoint from the potential temperature and relative
# humidity profiles:

# Compute pressure profile using hydrostatic balance:
# p(z) = p₀ * (θ₀/θ(z))^(cₚ/R) for dry adiabatic reference

function pressure_profile(z)
    θ = θᵢ₀(0, 0, z)
    # Use Poisson's equation: T = θ * (p/p₀)^(R/cₚ)
    # For hydrostatic atmosphere with varying θ, integrate numerically
    # Simplified: use reference pressure profile
    return pᵣ * exp(-g * z / (Rᵈ * 270))  # Approximate scale height
end

# More accurate pressure calculation using hydrostatic integration:

function compute_pressure_and_temperature(z_levels)
    nz = length(z_levels)
    p = zeros(nz)
    T = zeros(nz)
    
    # Surface values
    p[1] = pᵣ
    θ_surf = θᵢ₀(0, 0, z_levels[1])
    T[1] = θ_surf * (p[1] / pᵣ)^(Rᵈ / cᵖᵈ)
    
    # Integrate upward using hydrostatic equation
    for k in 2:nz
        dz = z_levels[k] - z_levels[k-1]
        θ_k = θᵢ₀(0, 0, z_levels[k])
        
        # Estimate temperature at midpoint for integration
        T_mid = (T[k-1] + θ_k * (p[k-1] / pᵣ)^(Rᵈ / cᵖᵈ)) / 2
        
        # Hydrostatic equation: dp/dz = -ρg = -pg/(RT)
        p[k] = p[k-1] * exp(-g * dz / (Rᵈ * T_mid))
        
        # Temperature from potential temperature
        T[k] = θ_k * (p[k] / pᵣ)^(Rᵈ / cᵖᵈ)
    end
    
    return p, T
end

# Compute saturation vapor pressure (Tetens formula):

eₛ(T) = 610.78 * exp(17.27 * (T - 273.15) / (T - 35.85))

# Compute dewpoint temperature from relative humidity:

function dewpoint(T, RH)
    e = RH * eₛ(T)
    # Inverse Tetens formula
    return 35.85 + 243.04 * log(e / 610.78) / (17.27 - log(e / 610.78)) + 273.15
end

# Generate profile data for Skew-T:

z_skewt = range(0, 15000, length=100)
p_profile, T_profile = compute_pressure_and_temperature(collect(z_skewt))
RH_profile = [RHᵢ(z) for z in z_skewt]
Td_profile = [dewpoint(T_profile[k], RH_profile[k]) for k in eachindex(T_profile)]

# Convert to Celsius for plotting:

T_celsius = T_profile .- 273.15
Td_celsius = Td_profile .- 273.15
p_hPa = p_profile ./ 100  # Convert to hPa

# Skew-T transformation: x_skew = T + skew_factor * log(p₀/p)

skew_factor = 40  # Skew angle parameter

function skew_transform(T_celsius, p_hPa)
    return T_celsius .+ skew_factor * log10.(1000 ./ p_hPa)
end

T_skewed = skew_transform(T_celsius, p_hPa)
Td_skewed = skew_transform(Td_celsius, p_hPa)

# Create Skew-T diagram:

fig_skewt = Figure(size=(700, 800))
ax_skewt = Axis(fig_skewt[1, 1],
                xlabel = "Temperature (°C) [skewed]",
                ylabel = "Pressure (hPa)",
                title = "Skew-T Log-P Diagram: Initial Sounding",
                yreversed = true,
                yscale = log10,
                yticks = [1000, 850, 700, 500, 300, 200, 100],
                yminorticks = IntervalsBetween(5))

ylims!(ax_skewt, 1050, 100)
xlims!(ax_skewt, -40, 60)

# Add isotherms (skewed):

for T_iso in -80:10:40
    T_iso_skewed = skew_transform(fill(T_iso, length(p_hPa)), p_hPa)
    lines!(ax_skewt, T_iso_skewed, p_hPa, color = (:gray, 0.3), linewidth = 0.5)
end

# Add isobars:

for p_iso in [1000, 850, 700, 500, 300, 200, 100]
    hlines!(ax_skewt, p_iso, color = (:gray, 0.3), linewidth = 0.5)
end

# Plot temperature and dewpoint profiles:

lines!(ax_skewt, T_skewed, p_hPa, color = :red, linewidth = 2.5, label = "Temperature")
lines!(ax_skewt, Td_skewed, p_hPa, color = :green, linewidth = 2.5, label = "Dewpoint")

# Add dry adiabats (lines of constant θ):

for θ_adiabat in 280:10:400
    T_adiabat = [θ_adiabat * (p / 1000)^(Rᵈ / cᵖᵈ) - 273.15 for p in p_hPa]
    T_adiabat_skewed = skew_transform(T_adiabat, p_hPa)
    lines!(ax_skewt, T_adiabat_skewed, p_hPa, color = (:orange, 0.3), linewidth = 0.5)
end

axislegend(ax_skewt, position = :rt)

save("supercell_skewt.png", fig_skewt)
fig_skewt

# ## Warm bubble initial perturbation
#
# The warm bubble parameters following Equations 17–18 in [KlempEtAl2015](@cite):

Δθ = 3              # K - perturbation amplitude
r_h = 10kilometers  # m - bubble horizontal radius
r_z = 1500          # m - bubble vertical radius
z_c = 1500          # m - bubble center height
x_c = Lx / 2        # m - bubble center x-coordinate
y_c = Ly / 2        # m - bubble center y-coordinate

# The total initial potential temperature combines the background profile with the
# cosine-squared warm bubble perturbation:

function θᵢ(x, y, z)
    θ_base = θᵢ₀(x, y, z)
    r = sqrt((x - x_c)^2 + (y - y_c)^2)
    R = sqrt((r / r_h)^2 + ((z - z_c) / r_z)^2)
    θ_pert = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ_base + θ_pert
end

# ## Model initialization
#
# Create the atmosphere model with one-moment cloud microphysics from CloudMicrophysics.jl,
# high-order WENO advection, and anisotropic minimum dissipation turbulence closure:

microphysics = BreezeOneMomentCloudMicrophysics()
advection = WENO(order=9, minimum_buffer_upwind_order=3)
closure = AnisotropicMinimumDissipation()
model = AtmosphereModel(grid; formulation, closure, microphysics, advection)

# Initialize with background potential temperature to compute hydrostatic pressure:

set!(model, θ = θᵢ₀)

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
    @inbounds qᵛᵢ_host[i, j, k] = RHᵢ(z) * qᵛ_sat
end

copyto!(parent(qᵛᵢ), qᵛᵢ_host)

# Set the full initial conditions (water vapor, potential temperature with bubble, and wind):

set!(model, qᵗ = qᵛᵢ, θ = θᵢ, u = uᵢ)

# Compute potential temperature perturbation for diagnostics:

θ = Breeze.AtmosphereModels.liquid_ice_potential_temperature(model)
θᵇᵍf = CenterField(grid)
set!(θᵇᵍf, (x, y, z) -> θᵢ₀(x, y, z))
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

save("max_w_timeseries.png", fig)
