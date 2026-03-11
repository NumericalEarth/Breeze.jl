# # Diurnal cycle of radiative convection
#
# This example simulates a dramatic diurnal cycle of moist convection over a
# tropical land surface. During the day, the sun heats the ground, driving
# vigorous boundary-layer convection that lofts moisture into towering cumulus
# clouds. At night, the surface cools rapidly by longwave emission, stabilizing
# the boundary layer and shutting off convection.
#
# The key ingredient is a **time-varying surface temperature** that follows the
# sun: peaking in the early afternoon and dipping well below the air temperature
# at night. This creates a strong diurnal contrast — afternoon thunderstorms
# that die at sunset and don't return until the next morning.
#
# Interactive all-sky RRTMGP radiation computes spectrally-resolved shortwave
# and longwave fluxes. Saturation-adjustment microphysics diagnoses cloud liquid
# water that feeds back on the radiation. A stretched vertical grid resolves the
# cloud layer (100 m spacing below 3 km) while extending to 25 km for a
# realistic atmospheric column. A stratospheric sponge layer above 8 km prevents
# spurious temperature drift in the coarse upper cells.

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
    z  = [0, 3000, 8000, 15000],
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

# ## Diurnal surface temperature
#
# Over land, the surface temperature swings dramatically with the sun.
# We model this as a sinusoidal cycle peaking 2 hours after solar noon:
# Tₛ(t) = T̄ₛ + ΔTₛ cos(2π(t - t_peak) / 24h), with T̄ₛ = 300 K (mean)
# and ΔTₛ = 10 K (amplitude). This gives 310 K in early afternoon and
# 290 K at night — a 20 K diurnal range, typical of tropical semi-arid land.
#
# We start at midnight (t = 0), so the surface starts cold (290 K), warms
# through the morning, peaks at t = 14 h (2 pm local), and cools at night.
# A `Field` stores the surface temperature and a callback updates it each
# time step, keeping both the bulk fluxes and RRTMGP in sync.

T̄ₛ = 300   # Mean surface temperature [K]
ΔTₛ = 20   # Diurnal amplitude [K]

Tₛ = Field{Center, Center, Nothing}(grid)
set!(Tₛ, T̄ₛ - ΔTₛ)  # Start at midnight minimum

# ## Radiation with a diurnal cycle
#
# We place the domain at 15°N latitude on the prime meridian and start at
# midnight on the spring equinox (March 20). The sun rises at t ≈ 6 h,
# reaches noon at t ≈ 12 h, and sets at t ≈ 18 h.

latitude = 15 # 15°N

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_temperature = Tₛ,
                                   surface_albedo = 0.20,
                                   surface_emissivity = 0.95,
                                   solar_constant = 1361,
                                   background_atmosphere,
                                   coordinate = (0, latitude),
                                   epoch = DateTime(2020, 3, 20, 0, 0, 0),
                                   schedule = TimeInterval(5minutes),
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius = ConstantRadiusParticles(30e-6))

# ## Surface fluxes
#
# Bulk aerodynamic formulae provide surface sensible heat, moisture, and momentum
# fluxes driven by the time-varying surface temperature. During the day Tₛ > Tair
# drives strong upward fluxes; at night Tₛ < Tair can produce downward fluxes
# that cool the boundary layer.

Cᴰ = Cᵀ = 1e-3
Cᵛ = 1.2e-3
Uᵍ = 1  # Gustiness [m/s]

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, gustiness=Uᵍ, surface_temperature=Tₛ)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, gustiness=Uᵍ, surface_temperature=Tₛ)

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

coriolis = FPlane(; latitude)
boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs)

weno_order = 5
momentum_advection = WENO(order=weno_order)

scalar_advection = (ρθ  = WENO(order=weno_order),
                    ρqᵗ = WENO(order=weno_order, bounds=(0, 1)))

model = AtmosphereModel(grid; dynamics, microphysics, radiation, forcing,
                        momentum_advection, scalar_advection,
                        boundary_conditions, coriolis)

# ## Initial conditions
#
# The sounding has a dry-adiabatic sub-cloud layer (0–1 km) capped by a
# conditionally unstable troposphere (5 K/km lapse rate) that transitions
# to an isothermal stratosphere at 210 K. Moisture is 20 g/kg at the surface
# with a 2.5 km scale height, typical of the tropical maritime boundary layer.

function Tᵇᵍ(z)
    T₀ = 300
    Tˢᵗ = 210 # stratosphere temperature
    T = T₀ - 1e-3 * max(z, 1000) - 5e-3 * max(0, z - 1000)
    return max(T, Tˢᵗ)
end

uᵢ(x, z) = -5 * max(1 - z / 3000, 0)

# Random perturbations in the lowest 1 km trigger convection.

δT = 2
δℋ = 1e-2
zδ = 1000

ϵ() = rand() - 0.5
Tᵢ(x, z) = Tᵇᵍ(z) + δT * ϵ() * (z < zδ)
ℋᵢ(x, z) = (0.5 + δℋ * ϵ()) * (z < zδ)

# After setting initial conditions, we recompute the reference state from the
# horizontally-averaged model state. This is important for Float32 accuracy in
# tall domains: the default dry-adiabat reference state diverges from the actual
# stratospheric profile, causing large density errors that overwhelm Float32
# precision. `set_to_mean!` adjusts `ρᵣ` to match the current state while
# rescaling density-weighted prognostic fields to preserve specific quantities.

set!(model; T=Tᵢ, ℋ=ℋᵢ, u=uᵢ)

reference_state = model.dynamics.reference_state
set_to_mean!(reference_state, model, rescale_densities=true)

T = model.temperature
qᵗ = specific_prognostic_moisture(model)
u, w = model.velocities.u, model.velocities.w
qˡ = model.microphysical_fields.qˡ

@info "Diurnal Radiative Convection (2D)"
@info "Grid: $(Nx) × $(Nz) (stretched), domain: $(Lx/1000) km × 25 km"
@info "Initial T range: $(minimum(T)) – $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) – $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# We run for two full diurnal cycles (48 hours) starting at midnight,
# so the on/off pattern of convection repeats convincingly.

simulation = Simulation(model; Δt=1, stop_time=3days)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Surface temperature callback
#
# At each time step we update the surface temperature field following
# a cosine curve that peaks 2 hours after solar noon (t = 14 h local).
# The period is 24 hours and the simulation starts at midnight.

function update_surface_temperature!(sim)
    t = time(sim)
    t_peak = 14hour  # peak at 14:00 local (2 pm)
    Tₛ_now = T̄ₛ + ΔTₛ * cos(2π * (t - t_peak) / day)
    set!(Tₛ, Tₛ_now)
    return nothing
end

add_callback!(simulation, update_surface_temperature!, TimeInterval(1minute))

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)
    Tₛ_now = T̄ₛ + ΔTₛ * cos(2π * (time(sim) - 14hours) / 24hours)

    OLR = mean(view(radiation.upwelling_longwave_flux, :, 1, Nz+1))

    msg = @sprintf("Iter: %5d, t: %8s, Δt: %5.1fs, wall: %8s",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed))
    msg *= @sprintf(", max|w|: %5.2f m/s, T: [%5.1f, %5.1f] K, max(qˡ): %.2e",
                   wmax, Tmin, Tmax, qˡmax)
    msg *= @sprintf(", Tₛ: %.1f K, OLR: %.1f W/m²", Tₛ_now, OLR)
    @info msg

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output
#
# Horizontally-averaged profiles are saved every hour (time-averaged) and 2D
# slices every 10 minutes for animation.


qᵛ = model.microphysical_fields.qᵛ
Fᴿ = radiation.flux_divergence

outputs = (; u, w, T, qˡ, qᵛ, Fᴿ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=1) for name in keys(outputs))

filename = "radiative_convection"
averages_filename = filename * "_averages.jld2"
slices_filename = filename * "_slices.jld2"

simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = averages_filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

slice_outputs = (; w, qᵛ, T)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(10minutes),
                                                overwrite_existing = true)

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed!"

# ## Hovmöller diagrams of mean profile evolution
#
# Time–height Hovmöller diagrams show how the horizontally-averaged temperature,
# cloud liquid water, and radiative flux divergence evolve over two diurnal cycles.
# The vertical axis is zoomed to the lowest 6 km where the convective dynamics live.

Tts  = FieldTimeSeries(averages_filename, "T")
qᵛts = FieldTimeSeries(averages_filename, "qᵛ")
Fᴿts = FieldTimeSeries(averages_filename, "Fᴿ")

times = Tts.times
Nt = length(times)

z = Oceananigans.Grids.znodes(Tts.grid, Center())
t_hours = times ./ 3600

# Build time–height matrices from the averaged profiles.

T_zt  = permutedims(interior(Tts,  1, 1, :, :))
T_zt .-= mean(T_zt, dims=1)
qᵛ_zt = permutedims(interior(qᵛts, 1, 1, :, :))
Fᴿ_zt = permutedims(interior(Fᴿts, 1, 1, :, :))

fig = Figure(size=(900, 800), fontsize=14)

zmax = 12000

axT  = Axis(fig[1, 1]; ylabel="z (m)", title="Temperature anomaly (K)",
            limits=((t_hours[1], t_hours[end]), (0, zmax)))
axqᵛ = Axis(fig[2, 1]; ylabel="z (m)", title="Specific humidity (kg/kg)",
            limits=((t_hours[1], t_hours[end]), (0, zmax)))
axF  = Axis(fig[3, 1]; ylabel="z (m)", xlabel="time (hours)", title="Radiative flux divergence (W/m³)",
            limits=((t_hours[1], t_hours[end]), (0, zmax)))

hidexdecorations!(axT;  grid=false)
hidexdecorations!(axqᵛ; grid=false)

hmT  = heatmap!(axT,  t_hours, z, T_zt;  colormap=:balance, colorrange=(-2, 2))
hmqᵛ = heatmap!(axqᵛ, t_hours, z, qᵛ_zt; colormap=:dense, colorrange=(0, 1e-2))
hmF  = heatmap!(axF,  t_hours, z, Fᴿ_zt; colormap=:balance, colorrange=(-0.1, 0.1))

Colorbar(fig[1, 2], hmT;  label="T′ (K)")
Colorbar(fig[2, 2], hmqᵛ; label="qᵛ (kg/kg)")
Colorbar(fig[3, 2], hmF;  label="Fᴿ (W/m³)")

save("radiative_convection_profiles.png", fig) #src
fig

# ## Animation of cloud structure
#
# We animate xz slices of vertical velocity and cloud liquid water, zoomed
# to the lowest 5 km where the convective dynamics and clouds live.

wts  = FieldTimeSeries(slices_filename, "w")
qᵛts = FieldTimeSeries(slices_filename, "qᵛ")

times = wts.times
Nt = length(times)

wlim  = maximum(abs, wts) / 4
qᵛlim = maximum(qᵛts) / 2

fig = Figure(size=(1000, 600), fontsize=14)

n = Observable(Nt)
axw  = Axis(fig[1, 1]; xlabel="x (km)", ylabel="z (km)", title="w (m/s)", limits=(nothing, (0, 5e3)))
axqᵛ = Axis(fig[1, 2]; xlabel="x (km)", ylabel="z (km)", title="qᵛ (kg/kg)", limits=(nothing, (0, 5e3)))

title = @lift "Diurnal Radiative Convection at t = " * prettytime(times[$n])
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

wn  = @lift wts[$n]
qᵛn = @lift qᵛts[$n]

hmw  = heatmap!(axw,  wn;  colormap=:balance, colorrange=(-wlim, wlim))
hmqᵛ = heatmap!(axqᵛ, qᵛn; colormap=:dense,   colorrange=(0, qᵛlim))

Colorbar(fig[2, 1], hmw;  vertical=false, label="w (m/s)")
Colorbar(fig[2, 2], hmqᵛ; vertical=false, label="qᵛ (g/kg)")

hideydecorations!(axqᵛ; grid=false)

CairoMakie.record(fig, "radiative_convection.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](radiative_convection.mp4)
