# # Saturated internal wave packet
#
# This literate example adapts [Oceananigans' internal wave example](https://clima.github.io/OceananigansDocumentation/stable/literated/internal_wave/)
# to AtmosphereModel to explore how slight supersaturation modifies the buoyancy frequency
# following [durran1982effects](@citet).
# The idea is simple:
# 1. Build a 2-D `x-z` anelastic AtmosphereModel with a realistic dry stratification.
# 2. Initialize total specific humidity `qᵗ` just **above** saturation everywhere.
# 3. Launch a Gaussian inertia-gravity wave packet.
# 4. Output fields at high cadence to resolve oscillations.
# 5. Compare the oscillation frequency we observe with
#    (a) the dry/naive Brunt–Väisälä frequency and
#    (b) the saturated (moist) Brunt–Väisälä frequency.
#
# The same script also produces a quick-look animation of the propagating packet.

using Breeze
using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant
using Oceananigans.Units
using CairoMakie
using Printf

# ## Grid and AtmosphereModel

Nx, Nz = 128, 128
Lx, Lz = 30_000, 12_000 # 30 km

grid = RectilinearGrid(size = (Nx, Nz),
                       x = (-Lx/2, Lx/2),
                       z = (-Lz/2, Lz/2),
                       topology = (Periodic, Flat, Bounded))

microphysics = SaturationAdjustment()  # instant warm-phase adjustment
advection = WENO(order=5)
model = AtmosphereModel(grid; advection, microphysics)

thermo = model.thermodynamics
reference_state = model.formulation.reference_state

# ## Background stratification and supersaturation

g = thermo.gravitational_acceleration
p₀ = reference_state.base_pressure
θ₀ = reference_state.potential_temperature
Rᵈ = dry_air_gas_constant(thermo)
Rᵛ = vapor_gas_constant(thermo)
cᵖᵈ = thermo.dry_air.heat_capacity
ρ₀ = p₀ / (Rᵈ * θ₀)

N = 0.012
θᵇ(x, z) = θ₀ * exp(N² * z / g)
set!(model, θ=θᵇ)

# Let's plot

fig = Figure()
axe = Axis(fig[1, 1])
axθ = Axis(fig[1, 2])
E = Average(model.specific_energy, dims=1) |> Field
θ = Average(PotentialTemperature(model), dims=1) |> Field
lines!(axe, E)
lines!(axθ, θ)

qᵗ = SaturationSpecificHumidity(model, :total_moisture) |> Field
cᵖᵈ = thermo.dry_air.heat_capacity
cᵖᵛ = thermo.vapor.heat_capacity
cᵖᵐ = (1 - qᵗ) * cᵖᵈ + qᵗ * cᵖᵛ
e₀ = model.specific_energy
eᵢ = cᵖᵐ / cᵖᵈ * e₀
set!(model, e=eᵢ, qᵗ=qᵗ)

lines!(axe, E)
lines!(axθ, θ)
fig

# ## Internal wave polarization relations

λx = 2_000.0
λz = 2_000.0  # 4 km vertical wavelength
k = 2π / λx
m = 2π / λz
ω = sqrt(N^2 * k^2 / (k^2 + m^2))

Δw = Lx / 8
δb = 1e-3 * N^2 * Δw
ϕ(x, z) = k * x + m * z
a(x, z) = δb * exp(-(x^2 + z^2) / 2Δw^2)

uᵢ(x, z) = - ω / N^2 * a(x, z) * sin(ϕ(x, z))
wᵢ(x, z) = - ω / N^2 * k / m * a(x, z) * sin(ϕ(x, z)) # ux + wz = 0
bᵢ(x, z) = a(x, z) * sin(ϕ(x, z))

# Compute energy perturbation
b = CenterField(grid)
set!(b, bᵢ)
T_avg = Average(model.temperature, dims=1) |> Field
g = thermo.gravitational_acceleration
e′ = cᵖᵐ * T_avg * b / g
e₀ = model.specific_energy
eᵢ = e₀ + e′

set!(model; u=uᵢ, w=wᵢ, e=eᵢ)

# and plot the initial condition

fig = Figure()
axθ = Axis(fig[1, 2])
axw = Axis(fig[1, 2])
θ = PotentialTemperature(model) |> Field
u, v, w = model.velocities
heatmap!(axθ, θ)
heatmap!(axw, w)
fig

# Now let's set up a simulation
Δt = 0.001 * 2π / ω
stop_time = 8 * 2π / ω
simulation = Simulation(model; Δt, stop_time)

function progress(sim)
    w = sim.model.velocities.w
    msg = @sprintf("iter: %d, t: %s, max|w| = %.3f m/s",
                   iteration(sim), prettytime(sim), maximum(abs, w))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

outputs = (w = model.velocities.w,
           θ = model.temperature,
           qᵗ = model.specific_moisture)

filename = "moist_internal_wave.jld2"

writer = JLD2Writer(model, outputs;
                    filename,
                    schedule = IterationInterval(10),
                    overwrite_existing = true)

simulation.output_writers[:fields] = writer

run!(simulation)

# Let's plot w

heatmap(model.velocities.w)

#=
@info "Simulation complete"

# ## Diagnostics: observed vs predicted buoyancy frequency

w_series = FieldTimeSeries(filename, "w")
θ_series = FieldTimeSeries(filename, "θ")
q_series = FieldTimeSeries(filename, "qᵗ")

times = w_series.times
Nt = length(times)
i_mid = Nx ÷ 2
k_mid = Nz ÷ 2

@allowscalar begin
    w_signal = [w_series[i_mid, 1, k_mid, n] for n in 1:Nt]
end

function dominant_frequency(times, signal)
    demeaned = signal .- mean(signal)
    dt = mean(diff(times))
    spectrum = abs.(rfft(demeaned))
    freqs = (0:length(spectrum)-1) ./ (dt * length(signal))
    idx = argmax(spectrum[2:end]) + 1       # skip zero frequency
    return 2π * freqs[idx]
end

ω_observed = dominant_frequency(times, w_signal)
period_observed = 2π / ω_observed

@info "Wave frequencies" ω_dry ω_moist ω_observed

# ## Visualization

x = range(grid.x[1], grid.x[2]; length = Nx)
z = range(grid.z[1], grid.z[2]; length = Nz)
time_minutes = times ./ 60

w_snapshot = Array(permutedims(w_series[:, 1, :, end]))
w_amplitude = maximum(abs, w_signal)

dry_fit = w_amplitude * sin.(ω_dry .* (times .- times[1]))
moist_fit = w_amplitude * sin.(ω_moist .* (times .- times[1]))

fig = Figure(size = (900, 950), fontsize = 14)

ax_field = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "z (km)",
                title = "Vertical velocity w at t = $(prettytime(times[end]))")
hm = heatmap!(ax_field, x ./ 1_000, z ./ 1_000, w_snapshot;
              colormap = :balance, colorrange = (-w_amplitude, w_amplitude))
Colorbar(fig[1, 2], hm, label = "w (m s⁻¹)")

ax_ts = Axis(fig[2, 1]; xlabel = "time (min)", ylabel = "w (m s⁻¹)",
             title = "Oscillation at domain center")
lines!(ax_ts, time_minutes, w_signal; label = "simulation", color = :black)
lines!(ax_ts, time_minutes, dry_fit; label = "dry N", color = :royalblue, linestyle = :dash)
lines!(ax_ts, time_minutes, moist_fit; label = "saturated N", color = :firebrick, linestyle = :dot)
axislegend(ax_ts; position = :rt)

ax_freq = Axis(fig[3, 1]; xlabel = "", ylabel = "ω (s⁻¹)",
               title = "Dominant frequency comparison",
               xticks = (1:3, ["observed", "dry", "saturated"]))
barplot!(ax_freq, 1:3, [ω_observed, ω_dry, ω_moist];
         color = (:gray50, :royalblue, :firebrick))

fig
=#

# ## Optional animation
#
# Uncomment the block below to record an MP4 movie of the packet.
# using Printf
# n = Observable(1)
# w_lim = maximum(abs, w_series)
# fig_anim = Figure(size = (700, 500))
# ax_anim = Axis(fig_anim[1, 1]; xlabel = "x (km)", ylabel = "z (km)",
#                title = "Saturated internal wave packet")
# hm_anim = heatmap!(ax_anim, x ./ 1_000, z ./ 1_000,
#                    permutedims(w_series[:, 1, :, 1]);
#                    colormap = :balance,
#                    colorrange = (-w_lim, w_lim))
# title_obs = @lift "t = $(prettytime(times[$n]))"
# fig_anim[0, :] = Label(fig_anim, title_obs, fontsize = 18, tellwidth = false)
#
# record(fig_anim, "moist_internal_wave.mp4", 1:Nt, framerate = 12) do nn
#     hm_anim[1] = permutedims(w_series[:, 1, :, nn])
#     n[] = nn
# end
# @info "Animation saved to moist_internal_wave.mp4"