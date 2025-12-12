# # Moist baroclinic instability
#
# This example simulates the development of moist baroclinic instability in a
# 3D channel on a beta plane. The setup follows the classic "baroclinic life cycle"
# configuration [Thorncroft1993](@cite), extended to include moisture and latent heating
# [Mak1994, Tippett1999](@cite).
#
# The key idea is that the initial state is **dry-stable** (stable to dry baroclinic
# instability) but becomes **moist-unstable** when condensation and latent heat release
# are activated during the nonlinear growth of the wave. This demonstrates how moisture
# can fundamentally alter baroclinic development.
#
# ## Physical setup
#
# We use an Eady-like configuration with:
# - Constant vertical wind shear (thermal wind balance)
# - Constant dry Brunt–Väisälä frequency
# - A beta-plane approximation for planetary vorticity variation
# - Near-saturated initial humidity so clouds form during ascent
#
# The fastest-growing baroclinic mode has wavelength approximately
# ``λ ≈ 3.9 N H / f₀`` and e-folding time ``τ ≈ N / (0.31 f₀ Λ)`` where
# ``Λ = ∂u/∂z`` is the vertical shear.

using Breeze
using Oceananigans.Units
using Oceananigans: Oceananigans
using CairoMakie
using Printf
using Random
using CUDA

Random.seed!(1234)

# ## Domain and grid
#
# We set up a 3D channel that is periodic in ``x``, bounded by walls in ``y``,
# and bounded vertically. The domain is sized to capture about one wavelength
# of the fastest-growing baroclinic mode.

Oceananigans.defaults.FloatType = Float32
arch = GPU()              # architecture for the simulation
Nx, Ny, Nz = 128, 128, 32   # resolution 
Lx = 6000kilometers       # zonal extent
Ly = 3000kilometers       # meridional extent (channel width)
Lz = 10kilometers         # vertical extent (troposphere depth)

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (0, Lz),
                       topology = (Periodic, Bounded, Bounded))

# ## Physical parameters
#
# We use typical midlatitude values for the Coriolis parameter and its
# meridional gradient (beta-plane), stratification, and vertical shear.

f₀ = 1e-4       # Coriolis parameter at reference latitude [s⁻¹]
β = 1.6e-11     # Meridional gradient of Coriolis parameter [m⁻¹ s⁻¹]
N = 0.01        # Brunt–Väisälä frequency [s⁻¹]
Λ = 3e-3        # Vertical wind shear ∂u/∂z [s⁻¹]

# The Eady model predicts the fastest-growing wavelength and e-folding time
# [Vallis2017, Chapter 9, Section 9.4](@cite). The constants 3.9 and 0.31 arise from
# the linear stability analysis: maximum growth occurs at non-dimensional wavenumber
# kH ≈ 1.61 (giving λ ≈ 2π/1.61 × NH/f ≈ 3.9 NH/f) with non-dimensional growth rate σ* ≈ 0.31.

λ_eady = 3.9 * N * Lz / f₀
τ_eady = N / (0.31 * f₀ * Λ)

@info "Eady model predictions" λ_eady/1e3 τ_eady/day

# ## Model setup
#
# We use the anelastic `AtmosphereModel` with warm-phase saturation adjustment
# microphysics. The beta-plane Coriolis is provided by Oceananigans.

coriolis = BetaPlane(; f₀, β)
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
advection = WENO(order = 5)

model = AtmosphereModel(grid; coriolis, microphysics, advection)

# ## Background state and initial conditions
#
# ### Stratification
#
# The background potential temperature profile has constant Brunt–Väisälä frequency:
# ```math
# θ^{\rm bg}(z) = θ_0 \exp\left( \frac{N^2 z}{g} \right)
# ```

constants = model.thermodynamic_constants
g = constants.gravitational_acceleration
θ₀ = model.formulation.reference_state.potential_temperature

θ_bg(z) = θ₀ * exp(N^2 * z / g)

# ### Thermal wind balance
#
# For a zonal wind with constant vertical shear ``u(z) = Λ z``, thermal wind
# balance requires a meridional temperature gradient:
# ```math
# f_0 \frac{∂u}{∂z} = -\frac{g}{θ_0} \frac{∂θ}{∂y}
# ```
# which gives ``∂θ/∂y = -f_0 θ_0 Λ / g``.

∂θ∂y = -f₀ * θ₀ * Λ / g

@info "Thermal wind parameters" Λ ∂θ∂y

# The background zonal wind and the baroclinic contribution to potential temperature:

u_bg(z) = Λ * z
θ_baroclinic(y) = ∂θ∂y * y

# ### Perturbation
#
# We add a small-amplitude temperature perturbation to seed the instability.
# The perturbation has a sinusoidal structure in ``x`` with wavelength equal
# to the domain length (one full wave), and is localized near the center of
# the channel and mid-troposphere.

k = 2π / Lx                 # wavenumber for one wavelength in domain
δθ = 0.5                    # perturbation amplitude [K]
y_width = Ly / 6            # meridional localization width
z_center = Lz / 2           # vertical center of perturbation
z_width = Lz / 4            # vertical localization width

function θ_perturbation(x, y, z)
    return δθ * cos(k * x) * exp(-y^2 / (2 * y_width^2)) * exp(-(z - z_center)^2 / (2 * z_width^2))
end

# ### Combined initial potential temperature and velocity

θ_init(x, y, z) = θ_bg(z) + θ_baroclinic(y) + θ_perturbation(x, y, z)
u_init(x, y, z) = u_bg(z)

# ### Initial moisture: near-saturated
#
# We initialize with a relative humidity profile that decreases with height,
# keeping the atmosphere sub-saturated but close enough to saturation that
# ascending air will condense.

RH₀ = 0.85   # surface relative humidity
RH_top = 0.3 # relative humidity at model top

RH(z) = RH₀ + (RH_top - RH₀) * z / Lz

# We use a two-pass initialization:
# 1. First set θ and u with a placeholder humidity to get saturation
# 2. Compute saturation specific humidity and reset moisture to RH × saturation

# First pass: set temperature and velocity with a placeholder moisture
set!(model; θ = θ_init, u = u_init, qᵗ = 0.01)

# Compute saturation specific humidity field from the model state
qᵛ⁺ = SaturationSpecificHumidityField(model, :equilibrium)

# Define moisture as RH × saturation
RH_field = CenterField(grid)
set!(RH_field, (x, y, z) -> RH(z))
qᵗ_new = Field(RH_field * qᵛ⁺)

# Second pass: reset model with the computed humidity profile
set!(model; θ = θ_init, u = u_init, qᵗ = qᵗ_new)

# ## Visualize initial conditions
#
# Let's plot the horizontally-averaged initial profiles to verify the setup.

θ = liquid_ice_potential_temperature(model)
u, v, w = model.velocities
qᵗ = model.specific_moisture
qᵛ⁺ = SaturationSpecificHumidity(model, :equilibrium)

# Compute horizontally-averaged profiles
θ_avg = Field(Average(θ, dims=(1, 2)))
u_avg = Field(Average(u, dims=(1, 2)))
qᵗ_avg = Field(Average(qᵗ, dims=(1, 2)))
qᵛ⁺_avg = Field(Average(qᵛ⁺, dims=(1, 2)))

z_km = znodes(grid, Center()) ./ 1e3

fig = Figure(size = (1000, 400), fontsize = 14)

ax1 = Axis(fig[1, 1], xlabel = "θ (K)", ylabel = "z (km)", title = "Potential temperature")
ax2 = Axis(fig[1, 2], xlabel = "u (m/s)", ylabel = "z (km)", title = "Zonal velocity")
ax3 = Axis(fig[1, 3], xlabel = "q (g/kg)", ylabel = "z (km)", title = "Specific humidity")

lines!(ax1, θ_avg)
lines!(ax2, u_avg)
lines!(ax3, qᵗ_avg * 1000, label = "qᵗ")
lines!(ax3, qᵛ⁺_avg * 1000, label = "qᵛ⁺", linestyle = :dash)
axislegend(ax3, position = :rt)

save("moist_baroclinic_instability_ic.png", fig)
fig

# ## Simulation
#
# We run the simulation for 10 days to observe the development of the
# baroclinic wave and the formation of clouds in the ascending regions.

stop_time = 10days
simulation = Simulation(model; Δt = 60, stop_time)

# Adaptive time stepping:
conjure_time_step_wizard!(simulation; cfl = 0.5)

# Progress callback:
qˡ = model.microphysical_fields.qˡ

function progress(sim)
    u, v, w = sim.model.velocities
    max_u = maximum(abs, u)
    max_v = maximum(abs, v)
    max_w = maximum(abs, w)
    max_qˡ = maximum(qˡ)

    @info @sprintf("Iter % 5d, time: % 10s, Δt: % 12s, max|u|: (%.1f, %.1f, %.2f) m/s, max qˡ: %.2e kg/kg",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   max_u, max_v, max_w, max_qˡ)
    return nothing
end

add_callback!(simulation, progress, TimeInterval(6hours))

# ## Output
#
# We save the key fields for later visualization.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵗ = model.specific_moisture
ζ = ∂x(v) - ∂y(u)  # vertical vorticity

outputs = (; u, v, w, θ, qˡ, qᵗ, ζ)

filename = "moist_baroclinic_instability.jld2"
output_interval = 3hours

simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                filename,
                                                schedule = TimeInterval(output_interval),
                                                overwrite_existing = true)

# ## Run the simulation

run!(simulation)

# ## Visualization
#
# Load the saved output and create visualizations.

θt = FieldTimeSeries(filename, "θ")
wt = FieldTimeSeries(filename, "w")
qˡt = FieldTimeSeries(filename, "qˡ")
ζt = FieldTimeSeries(filename, "ζ")

times = θt.times
Nt = length(times)

# ### Surface vorticity and midlevel vertical velocity

n = Observable(Nt)

# Surface level
k_sfc = 1
# Midlevel
k_mid = Nz ÷ 2

ζn_sfc = @lift view(ζt[$n], :, :, k_sfc)
wn_mid = @lift view(wt[$n], :, :, k_mid)
qˡn_mid = @lift view(qˡt[$n], :, :, k_mid)

fig = Figure(size = (1400, 500), fontsize = 14)

ax1 = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "Surface vorticity ζ")
ax2 = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "y (km)", title = "Midlevel vertical velocity w")
ax3 = Axis(fig[1, 3], xlabel = "x (km)", ylabel = "y (km)", title = "Midlevel liquid water qˡ")

ζ_lim = maximum(abs, ζt) / 2
w_lim = maximum(abs, wt) / 2
qˡ_max = maximum(qˡt)

hm1 = heatmap!(ax1, ζn_sfc, colormap = :balance, colorrange = (-ζ_lim, ζ_lim))
hm2 = heatmap!(ax2, wn_mid, colormap = :balance, colorrange = (-w_lim, w_lim))
hm3 = heatmap!(ax3, qˡn_mid, colormap = Reverse(:Blues), colorrange = (0, qˡ_max))

Colorbar(fig[2, 1], hm1, label = "s⁻¹", vertical = false)
Colorbar(fig[2, 2], hm2, label = "m/s", vertical = false)
Colorbar(fig[2, 3], hm3, label = "kg/kg", vertical = false)

title = @lift "Moist baroclinic instability — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize = 18, tellwidth = false)

fig

# ### Animation

record(fig, "moist_baroclinic_instability.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](moist_baroclinic_instability.mp4)

# ### Vertical cross-section through channel center

fig_xz = Figure(size = (1200, 800), fontsize = 14)

j_center = Ny ÷ 2 + 1

θn_xz = @lift view(θt[$n], :, j_center, :)
wn_xz = @lift view(wt[$n], :, j_center, :)
qˡn_xz = @lift view(qˡt[$n], :, j_center, :)

ax1 = Axis(fig_xz[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Potential temperature θ")
ax2 = Axis(fig_xz[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Vertical velocity w")
ax3 = Axis(fig_xz[3, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Liquid water qˡ")

θ_range = extrema(θt)
w_lim = maximum(abs, wt) / 2

hm1 = heatmap!(ax1, θn_xz, colormap = :thermal, colorrange = θ_range)
hm2 = heatmap!(ax2, wn_xz, colormap = :balance, colorrange = (-w_lim, w_lim))
hm3 = heatmap!(ax3, qˡn_xz, colormap = Reverse(:Blues), colorrange = (0, maximum(qˡt)))

Colorbar(fig_xz[1, 2], hm1, label = "K")
Colorbar(fig_xz[2, 2], hm2, label = "m/s")
Colorbar(fig_xz[3, 2], hm3, label = "kg/kg")

title_xz = @lift "Vertical cross-section (y = 0) — t = $(prettytime(times[$n]))"
fig_xz[0, :] = Label(fig_xz, title_xz, fontsize = 18, tellwidth = false)

# Animate the cross-section
record(fig_xz, "moist_baroclinic_instability_xz.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](moist_baroclinic_instability_xz.mp4)

