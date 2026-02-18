# # Diurnal cycle of radiative convection
#
# This example simulates moist convection over a warm tropical ocean, driven by
# surface fluxes and modulated by interactive all-sky RRTMGP radiation with a
# full diurnal cycle of solar forcing. Saturation-adjustment microphysics diagnoses
# cloud liquid water, which feeds back on the radiation through longwave emission
# and shortwave absorption.
#
# The initial sounding has a dry-adiabatic sub-cloud layer (0–1 km) capped by a
# conditionally unstable troposphere (6.5 K/km lapse rate). The dry-adiabatic
# sub-cloud layer eliminates convective inhibition, allowing surface perturbations
# to trigger moist convection that releases the available potential energy. A
# stretched vertical grid resolves the cloud layer (100 m spacing below 3 km)
# while extending to 25 km so that RRTMGP sees a realistic atmospheric column
# for both longwave and shortwave radiative transfer.
#
# The warm sea surface (303 K) drives strong sensible and latent heat fluxes
# that sustain the convection. Moist updrafts produce clouds from about 1 km
# up to 3–5 km, and the radiative transfer responds to the evolving cloud field.
# A stratospheric sponge layer above 8 km prevents spurious temperature drift
# in the coarse upper cells.

using Breeze
using Oceananigans
using Oceananigans.Units
using Dates: DateTime
using Printf, Random, Statistics
using CairoMakie

using NCDatasets  # Required for RRTMGP lookup tables
using RRTMGP
using CUDA

Random.seed!(2025)

# ## Grid
#
# We use a 2D vertical slice (x-z) that is periodic in x and bounded in z.
# The vertical grid is stretched: fine 100 m cells resolve the cloud layer below
# 3 km, then a smooth transition to 1 km cells carries the column up to 25 km.
# This gives RRTMGP a realistic atmospheric column (including the stratosphere)
# while keeping the total cell count modest.

Nx = 128
Lx = 12800   # 12.8 km

arch = GPU()
Oceananigans.defaults.FloatType = Float32

z = PiecewiseStretchedDiscretization(
    z  = [0, 3000, 8000, 25000],
    Δz = [100,  100, 1000,  1000])

Nz = length(z) - 1

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z,
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

# ## Reference state

p₀ = 101325  # Surface pressure [Pa]
θ₀ = 300     # Reference potential temperature [K]

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀,
                                 vapor_mass_fraction = 0)

dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere
#
# RRTMGP requires trace gas concentrations to compute spectral absorption and
# emission. We specify well-mixed greenhouse gas concentrations and a tropical
# ozone profile that transitions from low tropospheric values to a stratospheric
# peak near 25 km.

@inline function tropical_ozone(z)
    troposphere_O₃ = 30e-9 * (1 + 0.5 * z / 10_000)
    zˢᵗ = 25e3
    Hˢᵗ = 5e3
    stratosphere_O₃ = 8e-6 * exp(-((z - zˢᵗ) / Hˢᵗ)^2)
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

background_atmosphere = BackgroundAtmosphere(
    CO₂ = 348e-6,
    CH₄ = 1650e-9,
    N₂O = 306e-9,
    O₃ = tropical_ozone
)

# ## Radiation with a diurnal cycle
#
# We place the domain at 15°N latitude on the prime meridian and start at
# sunrise on the spring equinox (March 20). The `epoch` keyword converts the
# model's floating-point clock to an absolute `DateTime` for computing the solar
# zenith angle. With the full solar constant (1361 W/m²), daytime shortwave
# fluxes are realistic rather than reduced to a diurnal mean. The sun rises at
# t ≈ 0, reaches noon at t ≈ 6 hours, and sets at t ≈ 12 hours.

SST = 303  # Sea surface temperature [K]

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_temperature = SST,
                                   surface_albedo = 0.07,
                                   surface_emissivity = 0.98,
                                   solar_constant = 1361,
                                   background_atmosphere,
                                   coordinate = (0.0, 15.0),
                                   epoch = DateTime(2020, 3, 20, 6, 0, 0),
                                   schedule = TimeInterval(5minutes),
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius = ConstantRadiusParticles(30e-6))

# ## Surface fluxes
#
# Bulk aerodynamic formulae provide surface sensible heat, moisture, and momentum
# fluxes. The 3 K air–sea temperature disequilibrium drives strong surface fluxes
# that warm and moisten the boundary layer, feeding the convection.

Cᴰ = 1.0e-3
Cᵀ = 1.0e-3
Cᵛ = 1.2e-3
Uᵍ = 1.0  # Gustiness [m/s] — ensures surface fluxes even in calm conditions

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, gustiness=Uᵍ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, gustiness=Uᵍ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ))

# ## Microphysics
#
# Warm-phase saturation adjustment instantly converts supersaturated moisture to
# cloud liquid water.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

# ## Stratospheric sponge
#
# The domain extends to 25 km, but the initial stratosphere isn't in radiative
# equilibrium: ozone absorbs shortwave radiation and the coarse upper cells
# respond strongly. A Newtonian relaxation of temperature toward the initial
# profile above 8 km keeps the stratosphere anchored without affecting the
# tropospheric dynamics. We apply this as an energy forcing on `ρe`, which
# Breeze automatically converts to a `ρθ` tendency.

Tᵣ = reference_state.temperature
ρᵣ = reference_state.density
cᵖᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass  # J/(kg·K)
τ_sponge = 6hours

@inline function stratospheric_relaxation(i, j, k, grid, clock, model_fields, p)
    @inbounds T = model_fields.T[i, j, k]
    @inbounds Tᵣ = p.Tᵣ[i, j, k]
    @inbounds ρ = p.ρᵣ[i, j, k]
    z = znode(i, j, k, grid, Center(), Center(), Center())
    α = clamp((z - 8000) / 4000, 0, 1)
    ∂T∂t = -α * (T - Tᵣ) / p.τ
    return ρ * p.cᵖᵈ * ∂T∂t
end

sponge = Forcing(stratospheric_relaxation; discrete_form=true,
                 parameters=(; Tᵣ, ρᵣ, cᵖᵈ, τ=τ_sponge))

forcing = (; ρe=sponge)

# ## Model assembly

boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs)

weno_order = 5
momentum_advection = WENO(order=weno_order)

scalar_advection = (ρθ  = WENO(order=weno_order),
                    ρqᵗ = WENO(order=weno_order, bounds=(0, 1)))

model = AtmosphereModel(grid; dynamics, microphysics, radiation, forcing,
                        momentum_advection, scalar_advection,
                        boundary_conditions)

# ## Initial conditions
#
# The sounding has a dry-adiabatic sub-cloud layer (0–1 km, 9.8 K/km)
# capped by a conditionally unstable troposphere (6.5 K/km lapse rate)
# that transitions to an isothermal stratosphere at 210 K. Moisture is
# 20 g/kg at the surface with a 2.5 km scale height, typical of the
# tropical maritime boundary layer.

function Tᵇᵍ(z)
    T₀ = 300.0
    T_strat = 210.0
    if z ≤ 1000
        T = T₀ - 9.8e-3 * z
    else
        T = T₀ - 9.8e-3 * 1000 - 6.5e-3 * (z - 1000)
    end
    return max(T, T_strat)
end

qᵗᵇᵍ(z) = 0.020 * exp(-z / 2500)

uᵢ(x, z) = -5 * max(1 - z / 3000, 0)

# Random perturbations in the lowest 1 km trigger convection.

δT = 2.0
δq = 2e-3
zδ = 1000

ϵ() = rand() - 0.5
Tᵢ(x, z) = Tᵇᵍ(z) + δT * ϵ() * (z < zδ)
qᵢ(x, z) = qᵗᵇᵍ(z) + δq * ϵ() * (z < zδ)

# We recompute the reference state from the initial profiles for Float32 accuracy,
# then set initial conditions using temperature (not potential temperature).

compute_reference_state!(reference_state, Tᵇᵍ, qᵗᵇᵍ, constants)
set!(model; T=Tᵢ, qᵗ=qᵢ, u=uᵢ)

T = model.temperature
qᵗ = model.specific_moisture
u, w = model.velocities.u, model.velocities.w
qˡ = model.microphysical_fields.qˡ

@info "Diurnal Radiative Convection (2D)"
@info "Grid: $(Nx) × $(Nz) (stretched), domain: $(Lx/1000) km × 25 km"
@info "Initial T range: $(minimum(T)) – $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) – $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# We run for a full 24-hour diurnal cycle, starting at sunrise.

simulation = Simulation(model; Δt=1, stop_time=24hours)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=10)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)

    OLR = mean(view(radiation.upwelling_longwave_flux, :, 1, Nz+1))

    msg = @sprintf("Iter: %5d, t: %8s, Δt: %5.1fs, wall: %8s",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed))
    msg *= @sprintf(", max|w|: %5.2f m/s, T: [%5.1f, %5.1f] K, max(qˡ): %.2e",
                   wmax, Tmin, Tmax, qˡmax)
    msg *= @sprintf(", OLR: %.1f W/m²", OLR)
    @info msg

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(500))

# ## Output
#
# Horizontally-averaged profiles are saved every hour (time-averaged) and 2D
# slices every 10 minutes for animation. We restrict the slice output to the
# lowest 3 km where the interesting dynamics live.

qᵛ = model.microphysical_fields.qᵛ
Q = radiation.flux_divergence

outputs = (; u, w, T, qˡ, qᵛ, Q)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=1) for name in keys(outputs))

filename = "radiative_convection"
averages_filename = filename * "_averages.jld2"
slices_filename = filename * "_slices.jld2"

simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = averages_filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

slice_outputs = (; w, qˡ, T)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(10minutes),
                                                overwrite_existing = true)

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed!"

# ## Mean profile evolution
#
# Hourly-averaged profiles reveal the diurnal modulation of the boundary layer.
# We plot profiles at four representative times: sunrise, noon, sunset, and
# midnight. The radiative flux divergence (converted to K/day) shows longwave
# cooling that peaks at cloud top — weakened during the day when shortwave
# absorption partially compensates, and strongest at night.

Tts  = FieldTimeSeries(averages_filename, "T")
qˡts = FieldTimeSeries(averages_filename, "qˡ")
Qts  = FieldTimeSeries(averages_filename, "Q")

times = Tts.times
Nt = length(times)

ρᵣ_data = Array(interior(reference_state.density, 1, 1, :))
cᵖᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass  # J/(kg·K)
to_K_per_day = 86400 / cᵖᵈ

zc = znodes(grid, Center())
zc_km = Array(zc) ./ 1000

# We plot profiles at four times of day and zoom in on the lowest 6 km
# where the clouds and convective dynamics are.

snapshot_hours = [0, 6, 12, 18]
snapshot_labels = ["Sunrise (t = 0)", "Noon (t = 6 h)", "Sunset (t = 12 h)", "Midnight (t = 18 h)"]
snapshot_colors = [:goldenrod, :orangered, :purple, :midnightblue]

fig = Figure(size=(1400, 450), fontsize=14)

axT  = Axis(fig[1, 1]; xlabel="T (K)", ylabel="z (km)", limits=(nothing, (0, 6)))
axqˡ = Axis(fig[1, 2]; xlabel="qˡ (g/kg)", limits=(nothing, (0, 6)))
axQ  = Axis(fig[1, 3]; xlabel="Flux divergence (K/day)", limits=(nothing, (0, 6)))

for (ih, hour) in enumerate(snapshot_hours)
    n = findfirst(t -> t ≥ hour * 3600, times)
    isnothing(n) && continue

    T_data  = Array(interior(Tts[n],  1, 1, :))
    qˡ_data = Array(interior(qˡts[n], 1, 1, :)) .* 1000
    Q_data  = to_K_per_day .* Array(interior(Qts[n], 1, 1, :)) ./ ρᵣ_data

    lines!(axT,  T_data,  zc_km; color=snapshot_colors[ih], label=snapshot_labels[ih])
    lines!(axqˡ, qˡ_data, zc_km; color=snapshot_colors[ih])
    lines!(axQ,  Q_data,  zc_km; color=snapshot_colors[ih])
end

vlines!(axQ, 0; color=:gray50, linestyle=:dash, linewidth=1)
hideydecorations!(axqˡ; grid=false)
hideydecorations!(axQ; grid=false)

fig[0, 1:3] = Label(fig, "Diurnal Cycle — Mean Profiles", fontsize=16, tellwidth=false)
Legend(fig[2, :], axT; orientation=:horizontal, framevisible=false, tellwidth=false)

save("radiative_convection_profiles.png", fig) #src
fig

# ## Animation of cloud structure
#
# We animate xz slices of vertical velocity and cloud liquid water, zoomed
# to the lowest 5 km where the convective dynamics and clouds live.

wts  = FieldTimeSeries(slices_filename, "w")
qˡts_slice = FieldTimeSeries(slices_filename, "qˡ")

slice_times = wts.times
Nt_slices = length(slice_times)

wlim  = max(maximum(abs, wts) / 2, 1f-6)
qˡlim = max(maximum(qˡts_slice) / 2, 1f-6)

fig = Figure(size=(1000, 600), fontsize=14)

n = Observable(Nt_slices)
title = @lift "Diurnal Radiative Convection at t = " * prettytime(slice_times[$n])
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

axw  = Axis(fig[1, 1]; xlabel="x (km)", ylabel="z (km)", title="w (m/s)", limits=(nothing, (0, 5)))
axqˡ = Axis(fig[1, 2]; xlabel="x (km)", ylabel="z (km)", title="qˡ (g/kg)", limits=(nothing, (0, 5)))

w_n  = @lift wts[$n]
qˡ_n = @lift qˡts_slice[$n]

hmw  = heatmap!(axw,  w_n;  colormap=:balance, colorrange=(-wlim, wlim))
hmqˡ = heatmap!(axqˡ, qˡ_n; colormap=:dense,   colorrange=(0, qˡlim))

Colorbar(fig[2, 1], hmw;  vertical=false, label="w (m/s)")
Colorbar(fig[2, 2], hmqˡ; vertical=false, label="qˡ (g/kg)")

hideydecorations!(axqˡ; grid=false)

CairoMakie.record(fig, "radiative_convection.mp4", 1:Nt_slices; framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](radiative_convection.mp4)
