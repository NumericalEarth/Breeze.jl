# # Splitting supercell with scheduled microphysics
#
# This example mirrors the [splitting supercell](@ref) case but exercises Breeze's
# `microphysics_schedule` keyword on [`AtmosphereModel`](@ref). Microphysics fires
# every 20 simulated seconds (= 5 dycore steps at the fixed ``Δt = 4`` s) rather
# than every step. The operator-split state update and the cached tendency fields
# are both gated by the same schedule, which lets the dynamics super-step a
# potentially expensive microphysics package without breaking thermodynamic
# consistency.
#
# On firing, the schedule passes ``Δt_{\rm eff} = t - t_{\rm last\,fire}`` to
# [`microphysics_model_update!`](@ref Breeze.AtmosphereModels.microphysics_model_update!),
# so the operator-split Kessler scheme integrates over the actual elapsed window
# rather than a single dycore step. In between firings, the dycore tendency kernels
# read precomputed `CenterField` tendencies that were filled at the last firing.
#
# !!! warning
#     Holding microphysical rates constant across 20 s windows is physically
#     reasonable only for the slowly-varying part of the scheme. For Kessler-style
#     sedimentation and rain evaporation in active convection, results may differ
#     from the every-step reference. Use this example to evaluate the trade-off,
#     not as a recommended production setting for deep convection.
#
# Other than the microphysics schedule, the physical setup, initial conditions,
# and diagnostics are identical to the unscheduled `splitting_supercell` example.

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.Thermodynamics: hydrostatic_density, hydrostatic_temperature
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znodes

using CairoMakie
using CUDA
using Printf

# ## Domain and grid

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

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 100000,
                                 potential_temperature = 300)

dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere profiles
#
# Same Klemp et al. (2015) profile used in the unscheduled example.

θ₀ = 300       # K - surface potential temperature
θᵖ = 343       # K - tropopause potential temperature
zᵖ = 12000     # m - tropopause height
Tᵖ = 213       # K - tropopause temperature
qᵛ_max = 0.014 # kg/kg - cap on water vapor mixing ratio
nothing #hide

zˢ = 5kilometers  # m - shear layer height
uˢ = 30           # m/s - maximum shear wind speed
uᶜ = 15           # m/s - storm motion (Galilean translation speed)
nothing #hide

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
nothing #hide

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z ≤ zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

function qᵛ_bg(z)
    ℋ = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z ≤ zᵖ) + 1/4 * (z > zᵖ)
    p₀ = reference_state.surface_pressure
    pˢᵗ = reference_state.standard_pressure
    T = hydrostatic_temperature(z, p₀, θ_background, pˢᵗ, constants)
    ρ = hydrostatic_density(z, p₀, θ_background, pˢᵗ, constants)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return min(ℋ * qᵛ⁺, qᵛ_max)
end

qᵛ_column = Field{Nothing, Nothing, Center}(grid)
set!(qᵛ_column, qᵛ_bg)

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) ≤ 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

# ## Warm bubble perturbation

Δθ = 3              # K - perturbation amplitude
rᵇʰ = 10kilometers  # m - bubble horizontal radius
rᵇᵛ = 1500          # m - bubble vertical radius
zᵇ = 1500           # m - bubble center height
xᵇ = Lx / 2         # m - bubble center x-coordinate
yᵇ = Ly / 2         # m - bubble center y-coordinate
nothing #hide

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos(π * R / 2)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

# ## Model setup with scheduled microphysics
#
# We use the DCMIP2016 Kessler microphysics scheme with high-order WENO advection,
# and a 20-second `TimeInterval` schedule on microphysics.

microphysics = DCMIP2016KesslerMicrophysics()
advection = WENO(order=9)

# `microphysics_schedule = TimeInterval(20)` makes microphysics fire every 20 simulated
# seconds. The model allocates cached tendency fields at construction; in between
# firings, the dycore tendency kernels read those caches via the cache-aware
# `grid_microphysical_tendency` overload.

microphysics_schedule = TimeInterval(20)

model = AtmosphereModel(grid; dynamics, microphysics, microphysics_schedule, advection,
                        thermodynamic_constants = constants)

# ## Initialize and run

set!(model, θ=θᵢ, qᵛ=qᵛ_column, u=uᵢ)

## Fixed Δt = 4 s (no adaptive time-stepping wizard) so the schedule fires deterministically
## every 5 dycore steps.
simulation = Simulation(model; Δt=4, stop_time=2hours)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Diagnostics and progress

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

    msg *= @sprintf(", max(qᵛ): %.2e, max(qᶜˡ): %.2e, max(qʳ): %.2e, μp last fire: %d",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ),
                    sim.model.microphysics_state.last_fire_iteration)
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

z = znodes(grid, Center())
k_5km = searchsortedfirst(z, 5000)
@info "Saving xy slices at z = $(z[k_5km]) m (k = $k_5km)"

slice_outputs = (
    wxy = view(w, :, :, k_5km),
    qʳxy = view(qʳ, :, :, k_5km),
    qᶜˡxy = view(qᶜˡ, :, :, k_5km),
)

slices_filename = "splitting_supercell_scheduled_microphysics_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs; filename=slices_filename,
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)

run!(simulation)

# ## Animation: horizontal slices at z ≈ 5 km

wxy_ts = FieldTimeSeries(slices_filename, "wxy")
qʳxy_ts = FieldTimeSeries(slices_filename, "qʳxy")
qᶜˡxy_ts = FieldTimeSeries(slices_filename, "qᶜˡxy")

times = wxy_ts.times
Nt = length(times)

wlim = maximum(abs, wxy_ts) / 2
qʳlim = maximum(qʳxy_ts) / 4
qᶜˡlim = maximum(qᶜˡxy_ts) / 4

fig = Figure(size=(900, 400), fontsize=12)

axw = Axis(fig[1, 1], aspect=1, xlabel="x (m)", ylabel="y (m)", title="w (m/s)")
axqᶜˡ = Axis(fig[1, 2], aspect=1, xlabel="x (m)", ylabel="y (m)", title="qᶜˡ (kg/kg)")
axqʳ = Axis(fig[1, 3], aspect=1, xlabel="x (m)", ylabel="y (m)", title="qʳ (kg/kg)")

n = Observable(1)
wxy_n = @lift wxy_ts[$n]
qᶜˡxy_n = @lift qᶜˡxy_ts[$n]
qʳxy_n = @lift qʳxy_ts[$n]
title = @lift "Splitting supercell (μp every 20 s) at z ≈ 5 km, t = " * prettytime(times[$n])

hmw = heatmap!(axw, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ = heatmap!(axqᶜˡ, qᶜˡxy_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ = heatmap!(axqʳ, qʳxy_n, colormap=:amp, colorrange=(0, qʳlim))

Colorbar(fig[2, 1], hmw, vertical=false)
Colorbar(fig[2, 2], hmqᶜˡ, vertical=false)
Colorbar(fig[2, 3], hmqʳ, vertical=false)

fig[0, :] = Label(fig, title, fontsize=14, tellwidth=false)

CairoMakie.record(fig, "splitting_supercell_scheduled_microphysics_slices.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
nothing #hide

# ![](splitting_supercell_scheduled_microphysics_slices.mp4)

# ## Maximum vertical velocity time series

fig = Figure(size=(700, 400), fontsize=14)
ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Maximum w (m/s)",
          title="Maximum Vertical Velocity (microphysics every 20 s)",
          xticks=0:1800:7200)
lines!(ax, max_w_times, max_w_ts, linewidth=2)

save("supercell_scheduled_microphysics_max_w.png", fig) #src
fig
