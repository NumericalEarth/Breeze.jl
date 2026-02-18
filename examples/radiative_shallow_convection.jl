# # Convection driven by interactive radiation
#
# This example simulates shallow convection driven by interactive, all-sky RRTMGP
# radiation in a 2D (x-z) domain. Saturation adjustment microphysics provides
# cloud-radiation feedback: clouds form, modify the radiative fluxes, and the
# resulting differential heating or cooling drives further circulations.
#
# The setup resembles a trade-cumulus regime with a warm, moist boundary layer beneath
# a capping inversion at ~2 km. Perpetual-insolation radiation parameters follow the
# Radiative-Convective Equilibrium Model Intercomparison Project (RCEMIP) protocol
# (Wing et al., Geosci. Model Dev., 11, 793–813, 2018), which specifies a reduced
# solar constant and fixed zenith angle to represent the diurnal-mean insolation.
#
# The 2D configuration keeps costs low while still capturing the essential coupling
# between radiation, clouds, and convective dynamics.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf, Random, Statistics
using CairoMakie

using NCDatasets  # Required for RRTMGP lookup tables
using RRTMGP
using CUDA

Random.seed!(2025)

# ## Grid
#
# We use a 2D vertical slice (x-z) that is periodic in x and bounded in z.
# The domain extends 4 km vertically, which is enough to capture the shallow
# trade-cumulus layer and the lower free troposphere above the inversion.
# The horizontal extent is 12.8 km with 100 m resolution.

Nx = 128
Nz = 80
Lx = 12800   # 12.8 km
zᵗ = 4000    # 4 km domain top

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
# tropical ozone profile that transitions from low tropospheric values to a
# stratospheric peak near 25 km (irrelevant for our shallow 4 km domain, but
# provided for completeness).

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

# ## Radiation
#
# We use all-sky RRTMGP, which accounts for the radiative effects of cloud liquid
# and ice water via effective radius parameterizations. Radiation is updated every
# 5 minutes to tightly couple with the rapidly evolving shallow cloud layer.
#
# The radiation parameters follow the RCEMIP protocol:
# - Sea surface temperature: 300 K
# - Reduced solar constant: 551.58 W/m² (mimicking a diurnal mean)
# - Fixed zenith angle: 42.05° (giving cos θ ≈ 0.743)
# - Ocean surface albedo: 0.07

SST = 300                    # Sea surface temperature [K]
solar_constant = 551.58      # RCEMIP reduced solar constant [W/m²]
cos_zenith = cosd(42.05)     # Fixed zenith angle (RCEMIP perpetual insolation)
surface_albedo = 0.07        # Ocean surface albedo

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_albedo,
                                   solar_constant,
                                   background_atmosphere,
                                   surface_temperature = SST,
                                   surface_emissivity = 0.98,
                                   schedule = TimeInterval(5minutes),
                                   coordinate = cos_zenith,
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius = ConstantRadiusParticles(30e-6))

# ## Surface fluxes
#
# Surface sensible heat, moisture, and momentum fluxes are computed with bulk
# aerodynamic formulae using constant transfer coefficients. The SST serves as
# the surface boundary condition for both the sensible heat and moisture fluxes.

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
# Warm-phase saturation adjustment diagnoses cloud liquid water from temperature
# and total moisture. This is the simplest cloud scheme: excess moisture beyond
# saturation is instantly converted to liquid water.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

# ## Model assembly
#
# We assemble the model with 5th-order WENO advection and the surface boundary
# conditions.

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
# We prescribe a RICO-like tropical trade-cumulus sounding with three layers:
# 1. A well-mixed boundary layer (surface to 740 m) with a lapse rate of 4 K/km
# 2. A cloud layer (740 m to 2000 m) with a lapse rate of 3 K/km
# 3. A free troposphere (above 2000 m) with a lapse rate of 2 K/km
#
# Moisture decreases exponentially with a 3 km scale height. Trade winds blow
# at 5 m/s near the surface and decay to zero by 3 km.

function Tᵇᵍ(z)
    T₀ = 299.2  # Surface air temperature [K]
    if z ≤ 740
        return T₀ - 0.004 * z                              # Well-mixed boundary layer
    elseif z ≤ 2000
        return T₀ - 0.004 * 740 - 0.003 * (z - 740)       # Cloud layer
    else
        return T₀ - 0.004 * 740 - 0.003 * 1260 - 0.002 * (z - 2000)  # Free troposphere
    end
end

function qᵗᵇᵍ(z)
    q₀ = 0.020    # Surface specific humidity [kg/kg] (~90% RH at SST)
    Hq = 3000     # Moisture scale height [m]
    q_min = 1e-6  # Minimum humidity
    return max(q₀ * exp(-z / Hq), q_min)
end

uᵢ(x, z) = -5 * max(1 - z / 3000, 0)

# Random temperature and moisture perturbations are added in the lowest 500 m
# to trigger convection. Without these perturbations the atmosphere would remain
# horizontally uniform and no convective cells would develop.

δT = 0.5
δq = 5e-4
zδ = 500

ϵ() = rand() - 0.5
Tᵢ(x, z) = Tᵇᵍ(z) + δT * ϵ() * (z < zδ)
qᵢ(x, z) = qᵗᵇᵍ(z) + δq * ϵ() * (z < zδ)

# We recompute the reference state from the initial temperature and moisture
# profiles. This ensures that the reference density closely matches the actual
# density, which is important for Float32 accuracy. After recomputing the
# reference state we set the initial conditions using temperature (not potential
# temperature) to avoid errors from the Exner function.

compute_reference_state!(reference_state, Tᵇᵍ, qᵗᵇᵍ, constants)
set!(model; T=Tᵢ, qᵗ=qᵢ, u=uᵢ)

T = model.temperature
qᵗ = model.specific_moisture
u, w = model.velocities.u, model.velocities.w
qˡ = model.microphysical_fields.qˡ

@info "Radiative Shallow Convection (2D)"
@info "Grid: $(Nx) × $(Nz), domain: $(Lx/1000) km × $(zᵗ/1000) km"
@info "Initial T range: $(minimum(T)) - $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) - $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# We run for 2 hours with adaptive time-stepping. This is long enough for
# convective cells to develop, clouds to form, and radiative-convective
# interactions to produce an interesting boundary layer evolution.

simulation = Simulation(model; Δt=1, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=5)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)

    OLR = mean(view(radiation.upwelling_longwave_flux, :, 1, Nz+1))
    SW_dn = mean(view(radiation.downwelling_shortwave_flux, :, 1, Nz+1))

    msg = @sprintf("Iter: %5d, t: %8s, Δt: %5.1fs, wall: %8s",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed))
    msg *= @sprintf(", max|w|: %5.2f m/s, T: [%5.1f, %5.1f] K, max(qˡ): %.2e",
                   wmax, Tmin, Tmax, qˡmax)
    msg *= @sprintf(", OLR: %.1f W/m², SW_dn: %.1f W/m²", OLR, SW_dn)
    @info msg

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# We save horizontally-averaged profiles every 30 minutes (time-averaged over
# each interval) and 2D xz slices every 2 minutes for animation. The profiles
# capture the boundary layer evolution while the slices show the structure
# of individual convective cells and clouds.

qᵛ = model.microphysical_fields.qᵛ
Q = radiation.flux_divergence

outputs = (; u, w, T, qˡ, qᵛ, Q)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=1) for name in keys(outputs))

filename = "radiative_shallow_convection"
averages_filename = filename * "_averages.jld2"
slices_filename = filename * "_slices.jld2"

simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = averages_filename,
                                                  schedule = AveragedTimeInterval(30minutes),
                                                  overwrite_existing = true)

slice_outputs = (; w, qˡ, T)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed!"

# ## Mean profile evolution
#
# We plot horizontally-averaged profiles of temperature, moisture, and radiative
# flux divergence at several times to visualize the boundary layer evolution.
# The atmosphere is largely transparent to shortwave radiation in this shallow
# domain, so the net radiative effect is dominated by longwave cooling. This
# cooling destabilizes the boundary layer and drives convection, which in turn
# produces clouds that modify the longwave fluxes.

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

colormap = cgrad(:viridis, Nt, categorical=true)

fig = Figure(size=(1200, 400), fontsize=14)

axT = Axis(fig[1, 1]; xlabel="T (K)", ylabel="z (km)")
axq = Axis(fig[1, 2]; xlabel="qᵛ (g/kg)")
axQ = Axis(fig[1, 3]; xlabel="Flux divergence (K/day)")

for n in 1:Nt
    label = @sprintf("t = %s", prettytime(times[n]))

    T_data  = interior(Tts[n],  1, 1, :)
    qᵛ_data = interior(qᵛts[n], 1, 1, :) .* 1000  # kg/kg → g/kg
    Q_data  = to_K_per_day .* interior(Qts[n], 1, 1, :) ./ ρᵣ_data

    lines!(axT, T_data,  zc_km; color=colormap[n], label)
    lines!(axq, qᵛ_data, zc_km; color=colormap[n])
    lines!(axQ, Q_data,  zc_km; color=colormap[n])
end

vlines!(axQ, 0; color=:gray50, linestyle=:dash, linewidth=1)
hideydecorations!(axq; grid=false)
hideydecorations!(axQ; grid=false)

fig[0, 1:3] = Label(fig, "Radiative Shallow Convection — Mean Profiles", fontsize=16, tellwidth=false)
Legend(fig[2, :], axT; orientation=:horizontal, framevisible=false, tellwidth=false)

save("radiative_shallow_convection_profiles.png", fig) #src
fig

# ## Animation of cloud structure
#
# We animate xz slices of vertical velocity and cloud liquid water content
# to visualize the convective dynamics and cloud formation. Updrafts appear
# as red plumes in the vertical velocity field, while clouds (liquid water)
# form near the tops of the strongest updrafts where moist air is lifted
# above the saturation level.

wts  = FieldTimeSeries(slices_filename, "w")
qˡts_slice = FieldTimeSeries(slices_filename, "qˡ")

slice_times = wts.times
Nt_slices = length(slice_times)

wlim  = max(maximum(abs, wts) / 2, 1f-6)
qˡlim = max(maximum(qˡts_slice) / 2, 1f-6)

fig = Figure(size=(1000, 600), fontsize=14)

n = Observable(Nt_slices)
title = @lift "Radiative Shallow Convection at t = " * prettytime(slice_times[$n])
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

CairoMakie.record(fig, "radiative_shallow_convection.mp4", 1:Nt_slices; framerate=8) do nn
    n[] = nn
end
nothing #hide

# ![](radiative_shallow_convection.mp4)
