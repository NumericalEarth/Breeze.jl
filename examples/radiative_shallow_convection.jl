# # Diurnal cycle of radiative convection
#
# This example simulates shallow convection driven by interactive all-sky RRTMGP
# radiation over a warm ocean, featuring a full diurnal cycle of solar forcing.
# Saturation-adjustment microphysics diagnoses cloud liquid water, which feeds back
# on the radiation through cloud-top longwave cooling and shortwave absorption.
#
# The initial sounding is a RICO-like tropical trade-cumulus profile with a
# well-mixed sub-cloud layer, a moist layer that transitions to cloud below
# a strong inversion at ~2 km, and a dry, stable free troposphere above.
# The 5 km domain gives ample headroom above the capping inversion, so clouds
# form in the interior of the domain rather than near the lid.
#
# The key insight from this example is the diurnal modulation of cloud-radiation
# interaction. During the day, shortwave absorption partially offsets longwave
# cooling at cloud top, weakening the radiative destabilization and thinning
# the cloud layer. At night, unopposed longwave cooling strengthens convection
# and deepens the clouds.

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
# The domain extends 5 km vertically — deep enough for RRTMGP to compute
# reasonable longwave fluxes through the column, with ample headroom above
# the cloud layer at ~1–2 km.

Nx = 128
Nz = 64
Lx = 12800   # 12.8 km
zᵗ = 5000    # 5 km domain top

arch = GPU()
Oceananigans.defaults.FloatType = Float32

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, zᵗ),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

# ## Reference state
#
# We use the anelastic formulation with a reference state defined by the surface
# pressure and a potential temperature. The reference state is later overwritten
# by `compute_reference_state!` using the actual initial temperature and moisture
# profiles — this ensures that the reference density closely matches the initial
# density, which improves Float32 accuracy.

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
# emission. We specify well-mixed greenhouse gas concentrations and a simple
# tropical ozone profile.

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
# Instead of perpetual insolation, we drive the simulation with a full diurnal
# cycle of solar radiation. We place the domain at 15°N on the prime meridian
# and start at sunrise on the spring equinox (March 20). The `epoch` keyword
# tells Breeze how to convert the model's floating-point clock (in seconds)
# to an absolute `DateTime` for computing the solar zenith angle.
#
# With the full solar constant (1361 W/m²), daytime shortwave fluxes are
# realistic rather than reduced to a diurnal mean. The sun rises at t ≈ 0,
# reaches noon at t ≈ 6 hours, sets at t ≈ 12 hours, and rises again at
# t ≈ 24 hours.

SST = 300  # Sea surface temperature [K]

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
# fluxes. The SST of 300 K drives a modest air-sea temperature disequilibrium
# that sustains the boundary layer against radiative cooling.

Cᴰ = 1.0e-3
Cᵀ = 1.0e-3
Cᵛ = 1.2e-3

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ))

# ## Microphysics
#
# Warm-phase saturation adjustment instantly converts supersaturated moisture to
# cloud liquid water. This is the simplest cloud scheme, but it captures the
# essential cloud-radiation feedback: condensation produces cloud, cloud modifies
# radiative fluxes, and the resulting differential heating drives circulations
# that redistribute moisture.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

# ## Model assembly

boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs)

weno_order = 5
momentum_advection = WENO(order=weno_order)

scalar_advection = (ρθ  = WENO(order=weno_order),
                    ρqᵗ = WENO(order=weno_order, bounds=(0, 1)))

model = AtmosphereModel(grid; dynamics, microphysics, radiation,
                        momentum_advection, scalar_advection,
                        boundary_conditions)

# ## Initial conditions
#
# The sounding has three layers: a well-mixed sub-cloud layer (0–700 m), a moist
# cloud layer (700–2000 m), and a stable, dry free troposphere (above 2000 m).
# A 4 K temperature inversion at 2 km caps the boundary layer, keeping clouds
# well below the domain top. Moisture drops sharply at the inversion from
# ~8 g/kg to 4 g/kg, which prevents cloud formation in the free troposphere.

function Tᵇᵍ(z)
    T₀ = 299.2  # Surface air temperature [K]
    if z ≤ 700
        return T₀ - 9.8e-3 * z                                      # Dry adiabat
    elseif z ≤ 2000
        return T₀ - 9.8e-3 * 700 - 5e-3 * (z - 700)                # Moist adiabat
    else
        return T₀ - 9.8e-3 * 700 - 5e-3 * 1300 + 4 - 3e-3 * (z - 2000)  # +4 K inversion, stable
    end
end

# Moisture is high in the boundary layer (supporting cloud formation around 800–1500 m)
# and drops sharply above the inversion.

function qᵗᵇᵍ(z)
    if z ≤ 2000
        q₀ = 0.018   # 18 g/kg at the surface
        return q₀ * exp(-z / 2500)
    else
        return 0.004 * exp(-(z - 2000) / 5000)
    end
end

uᵢ(x, z) = -5 * max(1 - z / 3000, 0)

# Random perturbations in the lowest 500 m trigger convection.

δT = 0.5
δq = 5e-4
zδ = 500

ϵ() = rand() - 0.5
Tᵢ(x, z) = Tᵇᵍ(z) + δT * ϵ() * (z < zδ)
qᵢ(x, z) = qᵗᵇᵍ(z) + δq * ϵ() * (z < zδ)

# We recompute the reference state from the initial profiles for Float32 accuracy,
# then set initial conditions using temperature (not potential temperature) to avoid
# errors from the Exner function.

compute_reference_state!(reference_state, Tᵇᵍ, qᵗᵇᵍ, constants)
set!(model; T=Tᵢ, qᵗ=qᵢ, u=uᵢ)

T = model.temperature
qᵗ = model.specific_moisture
u, w = model.velocities.u, model.velocities.w
qˡ = model.microphysical_fields.qˡ

@info "Diurnal Radiative Convection (2D)"
@info "Grid: $(Nx) × $(Nz), domain: $(Lx/1000) km × $(zᵗ/1000) km"
@info "Initial T range: $(minimum(T)) – $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) – $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# We run for a full 24-hour diurnal cycle, starting at sunrise. Adaptive
# time-stepping keeps the simulation stable as convection intensifies.

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

add_callback!(simulation, progress, IterationInterval(200))

# ## Output
#
# Horizontally-averaged profiles are saved every hour (time-averaged over
# each interval) and 2D slices are saved every 10 minutes for animation.
# The hourly profiles show how the diurnal cycle modulates the boundary
# layer structure.

qᵛ = model.microphysical_fields.qᵛ
Q = radiation.flux_divergence

outputs = (; u, w, T, qˡ, qᵛ, Q)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=1) for name in keys(outputs))

filename = "radiative_shallow_convection"
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
# The radiative flux divergence (converted to K/day) shows longwave cooling that
# peaks at cloud top — weakened during the day when shortwave absorption partially
# compensates, and strongest at night when the cloud cools unopposed.

Tts  = FieldTimeSeries(averages_filename, "T")
qᵛts = FieldTimeSeries(averages_filename, "qᵛ")
qˡts = FieldTimeSeries(averages_filename, "qˡ")
Qts  = FieldTimeSeries(averages_filename, "Q")

times = Tts.times
Nt = length(times)

# Convert radiative flux divergence from W/m³ to K/day for easier interpretation.
# The conversion factor is ``86400 / (ρᵣ cᵖᵈ)`` where the factor of 86400 converts
# seconds to days.

ρᵣ_data = Array(interior(reference_state.density, 1, 1, :))
cᵖᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass  # J/(kg·K)
to_K_per_day = 86400 / cᵖᵈ

Δz = zᵗ / Nz
zc_km = [(k - 0.5) * Δz / 1000 for k in 1:Nz]

# We plot profiles at four representative times of day: sunrise (t = 0),
# noon (t = 6 h), sunset (t = 12 h), and midnight (t = 18 h).

snapshot_hours = [0, 6, 12, 18]
snapshot_labels = ["Sunrise (t = 0)", "Noon (t = 6 h)", "Sunset (t = 12 h)", "Midnight (t = 18 h)"]
snapshot_colors = [:goldenrod, :orangered, :purple, :midnightblue]

fig = Figure(size=(1400, 450), fontsize=14)

axT  = Axis(fig[1, 1]; xlabel="T (K)", ylabel="z (km)")
axqˡ = Axis(fig[1, 2]; xlabel="qˡ (g/kg)")
axQ  = Axis(fig[1, 3]; xlabel="Flux divergence (K/day)")

for (ih, hour) in enumerate(snapshot_hours)
    n = findfirst(t -> t ≥ hour * 3600, times)
    isnothing(n) && continue

    T_data  = interior(Tts[n],  1, 1, :)
    qˡ_data = interior(qˡts[n], 1, 1, :) .* 1000
    Q_data  = to_K_per_day .* interior(Qts[n], 1, 1, :) ./ ρᵣ_data

    lines!(axT,  T_data,  zc_km; color=snapshot_colors[ih], label=snapshot_labels[ih])
    lines!(axqˡ, qˡ_data, zc_km; color=snapshot_colors[ih])
    lines!(axQ,  Q_data,  zc_km; color=snapshot_colors[ih])
end

vlines!(axQ, 0; color=:gray50, linestyle=:dash, linewidth=1)
hideydecorations!(axqˡ; grid=false)
hideydecorations!(axQ; grid=false)

fig[0, 1:3] = Label(fig, "Diurnal Cycle — Mean Profiles", fontsize=16, tellwidth=false)
Legend(fig[2, :], axT; orientation=:horizontal, framevisible=false, tellwidth=false)

save("radiative_shallow_convection_profiles.png", fig) #src
fig

# ## Animation of cloud structure
#
# We animate xz slices of vertical velocity and cloud liquid water to watch
# convection develop through the diurnal cycle. Clouds form in the updrafts
# and are modulated by the alternating shortwave and longwave forcing.

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

axw  = Axis(fig[1, 1]; xlabel="x (km)", ylabel="z (km)", title="w (m/s)")
axqˡ = Axis(fig[1, 2]; xlabel="x (km)", ylabel="z (km)", title="qˡ (g/kg)")

w_n  = @lift wts[$n]
qˡ_n = @lift qˡts_slice[$n]

hmw  = heatmap!(axw,  w_n;  colormap=:balance, colorrange=(-wlim, wlim))
hmqˡ = heatmap!(axqˡ, qˡ_n; colormap=:dense,   colorrange=(0, qˡlim))

Colorbar(fig[2, 1], hmw;  vertical=false, label="w (m/s)")
Colorbar(fig[2, 2], hmqˡ; vertical=false, label="qˡ (g/kg)")

hideydecorations!(axqˡ; grid=false)

CairoMakie.record(fig, "radiative_shallow_convection.mp4", 1:Nt_slices; framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](radiative_shallow_convection.mp4)
