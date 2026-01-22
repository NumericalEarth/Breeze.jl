# # Precipitating thermal bubble
#
# This example simulates a precipitating moist thermal bubble, closely following
# the benchmark configuration from CM1 (Cloud Model 1) with `iinit=4` and `isnd=4`.
# The simulation demonstrates the evolution of a warm, moist air parcel rising through
# a saturated, neutrally-stable atmosphere, producing both cloud condensate and rain.
#
# This extends the cloudy thermal bubble example by using `OneMomentCloudMicrophysics`
# instead of simple saturation adjustment. The one-moment scheme includes autoconversion
# (cloud → rain) and accretion processes, allowing precipitation to develop as the
# thermal bubble rises and cloud liquid accumulates.
#
# The CM1 benchmark uses a saturated background with constant equivalent potential
# temperature θᵉ = 320 K and total water mixing ratio qᵗ = 0.020.

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

# ## Grid and thermodynamic setup
#
# We use the same grid as the dry thermal bubble case: a 20 km × 10 km domain
# with 128 × 128 grid points, giving 156 m horizontal and 78 m vertical resolution.

grid = RectilinearGrid(CPU();
                       size = (128, 128), halo = (5, 5),
                       x = (-10e3, 10e3),
                       z = (0, 10e3),
                       topology = (Bounded, Flat, Bounded))

thermodynamic_constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermodynamic_constants, surface_pressure=1e5, potential_temperature=300)
dynamics = AnelasticDynamics(reference_state)
advection = WENO(order=9)

# ## One-moment cloud microphysics
#
# The key difference from the cloudy thermal bubble is using `OneMomentCloudMicrophysics`
# which wraps saturation adjustment for cloud formation but adds prognostic rain.
# This enables autoconversion (cloud liquid converting to rain) and accretion
# (rain collecting cloud droplets), producing precipitation that falls.

using CloudMicrophysics  # Required to load the BreezeCloudMicrophysicsExt extension
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

model = AtmosphereModel(grid; dynamics, thermodynamic_constants, advection, microphysics)

# ## Initial conditions: CM1-style moist thermal bubble
#
# We initialize with total water qᵗ = 0.020 following CM1's `qt_mb` parameter
# for the saturated, neutrally-stable sounding. The thermal perturbation uses
# a cosine-squared profile centered at 2 km height with 2 km radius.

r₀ = 2e3        # Bubble radius (m)
z₀ = 2e3        # Bubble center height (m)
Δθ = 2          # Potential temperature perturbation (K)
qᵗ₀ = 0.020     # Total water specific humidity (CM1 qt_mb value)

θ₀ = model.dynamics.reference_state.potential_temperature
g = model.thermodynamic_constants.gravitational_acceleration

function θᵢ(x, z)
    r = sqrt((x / r₀)^2 + ((z - z₀) / r₀)^2)
    return θ₀ + Δθ * cos(π * min(1, r) / 2)^2
end

# Set potential temperature perturbation and uniform total water
set!(model, θ=θᵢ, qᵗ=qᵗ₀)

# ## Initial condition visualization

θ = liquid_ice_potential_temperature(model)
qᵗ = model.specific_moisture
qˡ = model.microphysical_fields.qˡ

fig = Figure(size=(900, 400))
ax1 = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial θ (K)")
ax2 = Axis(fig[1, 2], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Initial qˡ (kg/kg)")
hm1 = heatmap!(ax1, θ, colormap=:thermal)
hm2 = heatmap!(ax2, qˡ, colormap=:dense)
Colorbar(fig[1, 3], hm1)
Colorbar(fig[1, 4], hm2)
fig

# ## Simulation
#
# We run for 60 minutes to allow precipitation to develop. The one-moment scheme
# requires time for cloud liquid to accumulate and autoconversion to produce rain.

simulation = Simulation(model; Δt=2, stop_time=60minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Diagnostics and outputs
#
# Track cloud liquid, rain, and precipitation rate to observe the microphysical
# processes at work.

θ = liquid_ice_potential_temperature(model)
u, v, w = model.velocities
qˡ = model.microphysical_fields.qˡ     # Total liquid (cloud + rain)
qᶜˡ = model.microphysical_fields.qᶜˡ   # Cloud liquid only
qʳ = model.microphysical_fields.qʳ     # Rain mixing ratio

function progress(sim)
    qᶜˡmax = maximum(qᶜˡ)
    qʳmax = maximum(qʳ)
    wmax = maximum(abs, w)

    msg = @sprintf("Iter: %4d, t: %14s, Δt: %14s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), wmax)
    msg *= @sprintf(", max(qᶜˡ): %.2e, max(qʳ): %.2e", qᶜˡmax, qʳmax)

    @info msg
    return nothing
end

add_callback!(simulation, progress, TimeInterval(5minutes))

# Output fields for animation
outputs = (; θ, w, qᶜˡ, qʳ)

filename = "precipitating_thermal_bubble.jld2"
writer = JLD2Writer(model, outputs; filename,
                    including = [:grid],
                    schedule = TimeInterval(30seconds),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

run!(simulation)

# ## Visualization: 4-panel animation
#
# We create a 4-panel animation showing the evolution of:
# - Potential temperature θ (thermal structure)
# - Vertical velocity w (updraft dynamics)
# - Cloud liquid qᶜˡ (cloud formation)
# - Rain qʳ (precipitation development)

θts = FieldTimeSeries(filename, "θ")
wts = FieldTimeSeries(filename, "w")
qᶜˡts = FieldTimeSeries(filename, "qᶜˡ")
qʳts = FieldTimeSeries(filename, "qʳ")

times = θts.times
Nt = length(times)

# Compute color ranges
θ_range = (minimum(θts), maximum(θts))
w_range = maximum(abs, wts)
qᶜˡ_range = (0, max(1e-6, maximum(qᶜˡts)))
qʳ_range = (0, max(1e-6, maximum(qʳts)))

fig = Figure(size=(1000, 900), fontsize=12)

axθ = Axis(fig[1, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="θ (K)")
axw = Axis(fig[1, 2], aspect=2, xlabel="x (m)", ylabel="z (m)", title="w (m/s)")
axqᶜˡ = Axis(fig[2, 1], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Cloud liquid qᶜˡ (kg/kg)")
axqʳ = Axis(fig[2, 2], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Rain qʳ (kg/kg)")

n = Observable(1)
θn = @lift θts[$n]
wn = @lift wts[$n]
qᶜˡn = @lift qᶜˡts[$n]
qʳn = @lift qʳts[$n]

title = @lift "Precipitating thermal bubble — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

# Colorblind-friendly colormaps
hmθ = heatmap!(axθ, θn, colorrange=θ_range, colormap=:viridis)
hmw = heatmap!(axw, wn, colorrange=(-w_range, w_range), colormap=:balance)
hmqᶜˡ = heatmap!(axqᶜˡ, qᶜˡn, colorrange=qᶜˡ_range, colormap=:dense)
hmqʳ = heatmap!(axqʳ, qʳn, colorrange=qʳ_range, colormap=:amp)

Colorbar(fig[1, 3], hmθ, vertical=true)
Colorbar(fig[1, 4], hmw, vertical=true)
Colorbar(fig[2, 3], hmqᶜˡ, vertical=true)
Colorbar(fig[2, 4], hmqʳ, vertical=true)

CairoMakie.record(fig, "precipitating_thermal_bubble.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

nothing #hide

# ![](precipitating_thermal_bubble.mp4)
