# # Radiative Shallow Convection (2D)
#
# A 2D (x-z) shallow convection case with all-sky RRTMGP radiation and
# saturation adjustment cloud microphysics. This example validates the coupling
# between radiation and dynamics in a computationally cheap configuration.
#
# The setup resembles a trade-cumulus regime: a warm, moist boundary layer beneath
# a capping inversion at ~2 km, with shallow clouds forming and precipitating.
# All-sky radiation provides cloud-radiation feedback via RRTMGP spectral transfer.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf, Random, Statistics
using CairoMakie

using NCDatasets  # Required for RRTMGP lookup tables
using RRTMGP
using CUDA

Random.seed!(2025)

# ## Parameters
#
# Radiation parameters follow the RCEMIP protocol
# (Wing et al., Geosci. Model Dev., 11, 793–813, 2018).

SST = 300                    # Sea surface temperature [K]
solar_constant = 551.58      # RCEMIP reduced solar constant [W/m²]
cos_zenith = cosd(42.05)     # Fixed zenith angle (RCEMIP perpetual insolation)
surface_albedo = 0.07        # Ocean surface albedo

# ## Grid
#
# 2D vertical slice: periodic in x, flat in y, bounded in z.
# Shallow 4 km domain captures the trade-cumulus layer.

Nx = 128
Nz = 80
Lx = 12800   # 12.8 km (100 m horizontal spacing)
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
# Trace gas concentrations for RRTMGP. Ozone profile is a simple tropical approximation.

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
# All-sky RRTMGP with cloud-radiation interaction. Scheduled every 5 minutes
# for tight coupling with the shallow cloud layer.

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

Cᴰ = 1.0e-3
Cᵀ = 1.0e-3
Cᵛ = 1.2e-3

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=Breeze.BulkDrag(coefficient=Cᴰ))

# ## Microphysics

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
# RICO-like tropical trade-cumulus sounding: moist boundary layer with a
# weak inversion near 1.5 km and drier free troposphere above.

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

# Trade wind profile: ~5 m/s at surface, decreasing with height
uᵢ(x, z) = -5 * max(1 - z / 3000, 0)

# Random perturbations in the lowest 500 m to trigger convection
δT = 0.5
δq = 5e-4
zδ = 500

ϵ() = rand() - 0.5
Tᵢ(x, z) = Tᵇᵍ(z) + δT * ϵ() * (z < zδ)
qᵢ(x, z) = qᵗᵇᵍ(z) + δq * ϵ() * (z < zδ)

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

simulation = Simulation(model; Δt=1, stop_time=30minutes)
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

qᵛ = model.microphysical_fields.qᵛ
Q = radiation.flux_divergence

outputs = (; u, w, T, qˡ, qᵛ, Q)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=1) for name in keys(outputs))

filename = "radiative_shallow_convection"
averages_filename = filename * "_averages.jld2"
slices_filename = filename * "_slices.jld2"

simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = averages_filename,
                                                  schedule = AveragedTimeInterval(10minutes),
                                                  overwrite_existing = true)

slice_outputs = (; w, qˡ, T)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(5minutes),
                                                overwrite_existing = true)

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed!"

# ## Mean profile evolution
#
# Plot horizontally-averaged profiles of temperature, moisture, and radiative
# heating rate at several times to show the evolution of the boundary layer.

Tts  = FieldTimeSeries(averages_filename, "T")
qᵛts = FieldTimeSeries(averages_filename, "qᵛ")
qˡts = FieldTimeSeries(averages_filename, "qˡ")
Qts  = FieldTimeSeries(averages_filename, "Q")

times = Tts.times
Nt = length(times)

# Convert radiative flux divergence from W/m³ to K/day
ρᵣ_data = Array(interior(reference_state.density, 1, 1, :))
cᵖᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass  # J/(kg·K)
to_K_per_day = 86400 / cᵖᵈ

Δz = zᵗ / Nz
zc_km = [(k - 0.5) * Δz / 1000 for k in 1:Nz]

colormap = cgrad(:viridis, Nt, categorical=true)

fig = Figure(size=(1200, 400), fontsize=14)

axT = Axis(fig[1, 1]; xlabel="T (K)", ylabel="z (km)")
axq = Axis(fig[1, 2]; xlabel="qᵛ (g/kg)")
axQ = Axis(fig[1, 3]; xlabel="Heating rate (K/day)")

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

save("radiative_shallow_convection_profiles.png", fig)

fig

# ## Animation of cloud structure
#
# Animate xz slices of vertical velocity and cloud liquid water content.

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

save("radiative_shallow_convection.png", fig)

CairoMakie.record(fig, "radiative_shallow_convection.mp4", 1:Nt_slices; framerate=4) do nn
    n[] = nn
end

fig
