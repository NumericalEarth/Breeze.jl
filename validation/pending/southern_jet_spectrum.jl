# # Power spectrum of the unperturbed southern jet
#
# How noisy is the balanced DCMIP2016 jet when left completely alone?
# This script initializes the Jablonowski–Williamson balanced state with
# **no perturbation** and integrates for 3 days, saving the meridional
# velocity `v` at the southern jet core (~40° S) every 6 hours.
# We then compute the **zonal power spectrum** of `v` at each snapshot
# to see which wavenumbers are present in the numerical noise and how
# they evolve.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA
using FFTW

# ## DCMIP2016 parameters

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616
Oceananigans.defaults.planet_radius = 6371220
Oceananigans.defaults.planet_rotation_rate = 7.29212e-5

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ
p₀  = 1e5
a   = Oceananigans.defaults.planet_radius
Ω   = Oceananigans.defaults.planet_rotation_rate

# ## Grid

Nλ = 360; Nφ = 150; Nz = 32

H = 30kilometers

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                             halo = (5, 5, 5),
                             longitude = (0, 360),
                             latitude = (-75, 75),
                             z = (0, H))

# ## Analytic initial conditions (no perturbation)

Tᴱ = 310
Tᴾ = 240
Tᴹ = (Tᴱ + Tᴾ) / 2
Γ  = 0.005
K  = 3
b  = 2

function τ_and_integrals(z)
    Hˢ = Rᵈ * Tᴹ / g
    η  = z / (b * Hˢ)
    e  = exp(-η^2)

    A = (Tᴹ - Tᴾ) / (Tᴹ * Tᴾ)
    C = (K + 2) * (Tᴱ - Tᴾ) / (2 * Tᴱ * Tᴾ)

    τ₁  = A * (1 - 2η^2) * e + exp(Γ * z / Tᴹ) / Tᴹ
    ∫τ₁ = A * z * e + (exp(Γ * z / Tᴹ) - 1) / Γ

    τ₂  = C * (1 - 2η^2) * e
    ∫τ₂ = C * z * e

    return τ₁, τ₂, ∫τ₁, ∫τ₂
end

F(φ)  = cosd(φ)^K - K / (K + 2) * cosd(φ)^(K + 2)
dF(φ) = cosd(φ)^(K - 1) - cosd(φ)^(K + 1)

function virtual_temperature(λ, φ, z)
    τ₁, τ₂, _, _ = τ_and_integrals(z)
    return 1 / (τ₁ - τ₂ * F(φ))
end

function pressure(λ, φ, z)
    _, _, ∫τ₁, ∫τ₂ = τ_and_integrals(z)
    return p₀ * exp(-g / Rᵈ * (∫τ₁ - ∫τ₂ * F(φ)))
end

density(λ, φ, z) = pressure(λ, φ, z) / (Rᵈ * virtual_temperature(λ, φ, z))

potential_temperature(λ, φ, z) = virtual_temperature(λ, φ, z) * (p₀ / pressure(λ, φ, z))^κ

function zonal_velocity(λ, φ, z)
    _, _, _, ∫τ₂ = τ_and_integrals(z)
    Tᵥ = virtual_temperature(λ, φ, z)
    U = g / a * K * ∫τ₂ * dF(φ) * Tᵥ
    rcosφ  = a * cosd(φ)
    Ωrcosφ = Ω * rcosφ
    return -Ωrcosφ + sqrt(Ωrcosφ^2 + rcosφ * U)
end

# ## Model — no perturbation at all

coriolis = SphericalCoriolis(rotation_rate=Ω)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO())

set!(model; θ=potential_temperature, u=zonal_velocity, ρ=density)

# ## Simulation — 3 days, saving v every 6 hours

simulation = Simulation(model; Δt=12minutes, stop_time=3days)
conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=12minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | Δt = %s | max|u| = %.1f | max|v| = %.4e | max|w| = %.4e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

## Save v at every grid level (we'll extract the southern jet latitude later)
outputs = (; v = model.velocities.v)

simulation.output_writers[:spectra] = JLD2Writer(model, outputs;
                                                 filename = "southern_jet_v",
                                                 schedule = TimeInterval(6hours),
                                                 overwrite_existing = true)

# ## Run

run!(simulation)

# ## Power spectrum analysis — southern jet
#
# The DCMIP2016 jet is symmetric about the equator, so the southern jet
# core sits near 40° S. We find the closest grid latitude index and
# compute the zonal FFT of `v(λ)` at that latitude and the lowest model
# level (k = 1) for each saved snapshot.

v_ts = FieldTimeSeries("southern_jet_v.jld2", "v")
times = v_ts.times
Nt = length(times)

## Find the latitude index closest to -40°
## v lives on (Center, Face, Center); face latitudes are φᵃᶠᵃ
φ_nodes = v_ts.grid.φᵃᶠᵃ[1:Nφ+1]
j_south = argmin(abs.(φ_nodes .- (-40)))
@info "Southern jet latitude index j = $j_south, φ = $(φ_nodes[j_south])°"

k_sfc = 1  ## lowest model level

## Compute power spectra
wavenumbers = 0:(Nλ÷2)
spectra = zeros(length(wavenumbers), Nt)

for n in 1:Nt
    v_snap = v_ts[n]
    v_slice = vec(Array(interior(v_snap, :, j_south, k_sfc)))
    V̂ = rfft(v_slice)
    spectra[:, n] = abs.(V̂) .^ 2 / Nλ^2
end

# ## Plot — power spectrum evolution
#
# Each line is one snapshot (every 6 hours over 3 days). Color encodes time.

fig = Figure(size=(900, 500))
ax = Axis(fig[1, 1];
          xlabel = "Zonal wavenumber",
          ylabel = "Power spectral density (m²/s²)",
          title = "Zonal power spectrum of v at 40°S (surface), no perturbation",
          yscale = log10,
          xscale = log10)

colormap = cgrad(:viridis, Nt, categorical=true)

for n in 1:Nt
    t_hours = times[n] / 3600
    label = n == 1 || n == Nt ? @sprintf("t = %.0f h", t_hours) : nothing
    lines!(ax, wavenumbers[2:end], spectra[2:end, n];
           color = colormap[n], linewidth = 1.5, label)
end

## Mark wavenumber 9 for reference
vlines!(ax, [9]; linestyle = :dash, color = :gray60, label = "k = 9")

axislegend(ax; position = :rt)
Colorbar(fig[1, 2]; colormap = :viridis,
         limits = (0, times[end] / 3600),
         label = "Time (hours)")

save("southern_jet_spectrum.png", fig)
@info "Saved southern_jet_spectrum.png"
nothing

# ## Plot — spectrum at selected times (cleaner view)

fig2 = Figure(size=(900, 500))
ax2 = Axis(fig2[1, 1];
           xlabel = "Zonal wavenumber",
           ylabel = "Power spectral density (m²/s²)",
           title = "Zonal power spectrum of v at 40°S (surface)",
           yscale = log10,
           xscale = log10)

selected = [1, Nt÷4, Nt÷2, 3Nt÷4, Nt]
colors = [:black, :dodgerblue, :green, :orange, :red]

for (i, n) in enumerate(selected)
    t_hours = times[n] / 3600
    lines!(ax2, wavenumbers[2:end], spectra[2:end, n];
           color = colors[i], linewidth = 2,
           label = @sprintf("t = %.0f h", t_hours))
end

vlines!(ax2, [9]; linestyle = :dash, color = :gray60, label = "k = 9")
axislegend(ax2; position = :rt)

save("southern_jet_spectrum_selected.png", fig2)
@info "Saved southern_jet_spectrum_selected.png"
nothing
