# # Saturated internal wave packet
#
# This literate example adapts Oceananigans' `internal_wave.jl`
# to AtmosphereModel to explore how slight supersaturation
# modifies the buoyancy frequency following the study by [Durran1982](@citet). #Durran & Klemp (1982).
# The idea is simple:
# 1. Build a 2D ``x-z`` anelastic AtmosphereModel with a realistic dry stratification.
# 2. Initialize total specific humidity ``q`` just **above** saturation everywhere.
# 3. Launch a Gaussian inertia-gravity wave packet.
# 4. Output fields at high cadence to resolve oscillations.
# 5. Compare the oscillation frequency we observe with
#    (a) the dry/naive Brunt–Väisälä frequency and
#    (b) the saturated (moist) Brunt–Väisälä frequency.
#
# The same script also produces a quick-look animation of the propagating packet.

using Breeze
using Oceananigans.Units
using FFTW
using Statistics
using GLMakie
using Printf

using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant, saturation_specific_humidity

# ## Grid and AtmosphereModel

Nx = 192
Nz = 128
Lx = 30_000        # 30 km
Lz = 12_000        # 12 km

grid = RectilinearGrid(; size = (Nx, Nz), x = (-Lx/2, Lx/2), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

microphysics = SaturationAdjustment()  # instant warm-phase adjustment
advection = WENO(order = 5)
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

N_dry = 0.012        # 1.2×10⁻² s⁻¹, typical mid-troposphere stability
dθdz = θ₀ * N_dry^2 / g

θ_background(z) = θ₀ + dθdz * z
supersaturation = 1.01      # 1% above saturation everywhere

reference_pressure(z) = p₀ * (1 - g * z / (cᵖᵈ * θ₀))^(cᵖᵈ / Rᵈ)
reference_density(z) = ρ₀ * (reference_pressure(z) / p₀)^(1 - Rᵈ / cᵖᵈ)

exner(p) = (p / p₀)^(Rᵈ / cᵖᵈ)
temperature_background(z) = θ_background(z) * exner(reference_pressure(z))

function saturation_profile(z)
    T = temperature_background(z)
    ρ = reference_density(z)
    return saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
end

qᵗ_background(z) = supersaturation * saturation_profile(z)

# ### Moist Brunt–Väisälä frequency following Durran & Klemp (1982)

Lᵥ = thermo.liquid.reference_latent_heat
ϵ = Rᵈ / Rᵛ

function saturated_buoyancy_frequency(z)
    T = temperature_background(z)
    q_sat = saturation_profile(z)

    δ = 25.0                     # finite difference step (m)
    z₊ = clamp(z + δ, 0, Lz)
    z₋ = clamp(z - δ, 0, Lz)
    dTdz = (temperature_background(z₊) - temperature_background(z₋)) / (2δ)
    Γ_env = -dTdz                # environmental lapse rate (K / m)

    Γ_m = g * (1 + (Lᵥ * q_sat) / (Rᵈ * T)) /
          (cᵖᵈ + (Lᵥ^2 * q_sat * ϵ) / (Rᵈ * T^2))

    return sqrt(max(0, g / T * (Γ_m - Γ_env)))
end

# Let's plot the moist buoyancy frequency

using GLMakie

z = znodes(model.grid, Center())

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Moist Brunt–Väisälä frequency (s⁻¹)", ylabel="z (m)")
lines!(ax, saturated_buoyancy_frequency.(z), z)
fig

# Now let's continue

z₀ = Lz / 2
N_moist = saturated_buoyancy_frequency(z₀)

@info "Dry vs saturated buoyancy frequency" N_dry N_moist

# ## Internal wave polarization relations

λx = Lx / 4
λz = Lz / 5
k = 2π / λx
m = 2π / λz

ω_dry = sqrt(N_dry^2 * k^2 / (k^2 + m^2))
ω_moist = sqrt(N_moist^2 * k^2 / (k^2 + m^2))

gaussian_amplitude = 0.08         # scales velocities/pressure perturbations
gaussian_width = Lz / 8
x₀ = mean(xnodes(grid, Center()))

phase(x, z) = k * (x - x₀) + m * (z - z₀)
gaussian_envelope(x, z) = gaussian_amplitude * exp(-((x - x₀)^2 + (z - z₀)^2) / (2gaussian_width^2))

function buoyancy_perturbation(x, z)
    denominator = ω_dry^2 - N_dry^2
    return gaussian_envelope(x, z) * m * N_dry^2 / denominator * sin(phase(x, z))
end

θ_perturbation(x, z) = θ₀ / g * buoyancy_perturbation(x, z)
θ_initial(x, z) = θ_background(z) + θ_perturbation(x, z)

function u_initial(x, z)
    return gaussian_envelope(x, z) * k / ω_dry * cos(phase(x, z))
end

function w_initial(x, z)
    denominator = ω_dry^2 - N_dry^2
    return gaussian_envelope(x, z) * m * ω_dry / denominator * cos(phase(x, z))
end

qᵗ_initial(x, z) = qᵗ_background(z)

set!(model, u = u_initial,
            w = w_initial,
            θ = θ_initial,
            qᵗ = qᵗ_initial)

# ## Plot

u, v, w = model.velocities

w_max = maximum(abs, w)
levels = range(-w_max, stop=w_max, length=12)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
contourf!(ax, w; levels, colormap=:balance)
fig


# ## Time stepping and high-frequency output

period_dry = 2π / ω_dry
stop_time = 8period_dry
Δt_output = period_dry / 20

simulation = Simulation(model; Δt = 1, stop_time)
conjure_time_step_wizard!(simulation; cfl = 0.7)

function progress(sim)
    u, v, w = model.velocities
    max_w = maximum(abs, w)
    @info @sprintf("iteration: %d, time: %s, Δt: %s, max|w|: %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), max_w)
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

outputs = (; w = model.velocities.w,
             θ = model.temperature,
             qᵗ = model.specific_moisture)

filename = "moist_internal_wave.jld2"
writer = JLD2Writer(model, outputs;
                    filename,
                    schedule = TimeInterval(Δt_output),
                    overwrite_existing = true)
simulation.output_writers[:wave] = writer

@info "Running saturated internal wave packet..." period_dry stop_time
run!(simulation)
@info "Simulation complete"

# ## Diagnostics: observed vs predicted buoyancy frequency

wt = FieldTimeSeries(filename, "w")
θt = FieldTimeSeries(filename, "θ")
qt = FieldTimeSeries(filename, "qᵗ")

times = wt.times
Nt = length(times)
i_mid = Nx ÷ 2
k_mid = Nz ÷ 2

w_signal = [wt[i_mid, 1, k_mid, n] for n in 1:Nt]

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

time_minutes = times ./ 60

n = Observable(Nt)
wn = @lift wt[$n]
w_amplitude = maximum(abs, w_signal)

dry_fit = w_amplitude * sin.(ω_dry .* (times .- times[1]))
moist_fit = w_amplitude * sin.(ω_moist .* (times .- times[1]))

fig = Figure(fontsize = 14)

axw = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)",
                      title = "Vertical velocity w at t = $(prettytime(times[end]))")
hm = heatmap!(axw, wn;
              colormap = :balance, colorrange = (-w_amplitude, w_amplitude))
Colorbar(fig[1, 2], hm, label = "w (m s⁻¹)")

# axts = Axis(fig[2, 1]; xlabel = "time (min)", ylabel = "w (m s⁻¹)",
#             title = "Oscillation at domain center")
# lines!(axts, time_minutes, w_signal; label = "simulation", color = :black)
# lines!(axts, time_minutes, dry_fit; label = "dry N", color = :royalblue, linestyle = :dash)
# lines!(axts, time_minutes, moist_fit; label = "saturated N", color = :firebrick, linestyle = :dot)
# axislegend(axts; position = :rt)

# axf = Axis(fig[3, 1]; xlabel = "", ylabel = "ω (s⁻¹)",
#            title = "Dominant frequency comparison",
#            xticks = (1:3, ["observed", "dry", "saturated"]))
# barplot!(axf, 1:3, [ω_observed, ω_dry, ω_moist];
#          color = (:gray50, :royalblue, :firebrick))

fig

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

