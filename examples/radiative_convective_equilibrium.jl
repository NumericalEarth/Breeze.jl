# # Radiative-Convective Equilibrium (RCEMIP)
#
# Radiative-convective equilibrium (RCE) following the RCEMIP-I protocol ([Wing2018](@citet)).
# This is a scaled-down version of the "RCE_small" configuration (Table 2).

using Breeze
using Breeze.AtmosphereModels: set_to_mean!
using Oceananigans: Oceananigans
using Oceananigans.Units
using CUDA

using CairoMakie
using Printf
using Random
using Statistics

using NCDatasets  # Required for RRTMGP lookup tables
using RRTMGP

Random.seed!(2024)

# ## RCEMIP parameters (Wing et al. 2018, Table 1)

SST = 300                    # Sea surface temperature [K]
solar_constant = 551.58      # Reduced solar constant [W/m²] (corrigendum)
cos_zenith = cosd(42.05)     # cos(42.05°) ≈ 0.7434 (corrigendum)
surface_albedo = 0.07        # Ocean surface albedo

# f-plane at 20°N (RCEMIP baseline is f = 0)
latitude = 20
coriolis = FPlane(latitude=latitude)

# ## Domain and grid

Nx = Ny = 2 * 128
Lx = Ly = 2 * 128000   # 96 km horizontal domain (1 km grid spacing)
zᵗ = 20000        # 20 km model top
Nz = 128

z_faces = range(0, zᵗ, length=Nz+1)

@info "RCE grid: $(Nx)×$(Ny)×$(Nz), Δz = $(zᵗ/Nz) m"

arch = GPU()
Oceananigans.defaults.FloatType = Float32

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state
#
# The initial dry-adiabat reference is reset to match the actual initial
# profiles via `compute_reference_state!` before setting initial conditions.

p₀ = 101325  # Surface pressure [Pa]
θ₀ = 350     # Reference potential temperature [K]

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀,
                                 vapor_mass_fraction = 0)

dynamics = AnelasticDynamics(reference_state)

# ## Trace gas concentrations (Wing et al. 2018, Table 1)
#
# Analytical approximation of the tropical ozone profile:
# low in the troposphere (~30 ppbv), peak in the stratosphere (~8 ppmv at 25 km).

@inline function tropical_ozone(z)
    ## Tropospheric ozone: ~30 ppbv near surface, increasing slowly
    troposphere_O₃ = 30e-9 * (1 + 0.5 * z / 10_000)
    ## Stratospheric ozone: peaks around 25 km at ~8 ppmv
    zˢᵗ = 25e3
    Hˢᵗ = 5e3
    stratosphere_O₃ = 8e-6 * exp(-((z - zˢᵗ) / Hˢᵗ)^2)
    ## Smooth transition using a sigmoid
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

background_atmosphere = BackgroundAtmosphere(
    CO₂ = 348e-6,       # 348 ppmv
    CH₄ = 1650e-9,      # 1650 ppbv
    N₂O = 306e-9,       # 306 ppbv
    O₃ = tropical_ozone
)

# ## Radiation
#
# Full RRTMGP radiation with all-sky optics (cloud-radiation interaction).
# The radiation is recomputed every 5 minutes to maintain tight cloud-radiation coupling.
# A fixed zenith angle gives perpetual diurnal-mean insolation (no day-night cycle).

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_albedo,
                                   solar_constant,
                                   background_atmosphere,
                                   surface_temperature = SST,
                                   surface_emissivity = 0.98,
                                   schedule = TimeInterval(30minutes),
                                   coordinate = cos_zenith,
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),  # 10 μm
                                   ice_effective_radius = ConstantRadiusParticles(30e-6))     # 30 μm

# ## Surface fluxes
#
# Bulk aerodynamic surface fluxes for momentum (drag), sensible heat, and moisture.
# The fixed SST acts as an infinite heat reservoir, driving the system toward equilibrium.

Cᴰ = 1.0e-3  # Drag coefficient
Cᵀ = 1.0e-3  # Sensible heat transfer coefficient
Cᵛ = 1.2e-3  # Moisture transfer coefficient

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# ## Sponge layer
#
# Rayleigh damping in the upper 6 km (14-20 km) to absorb gravity waves,
# prevent reflection from the model top, and damp convective overshoots
# at the tropopause (~15 km). Applies to momentum (relaxing ρu, ρv, ρw toward zero)
# and potential temperature (relaxing ρθ toward the reference state) to damp both
# velocity and thermal perturbations from convective overshoots.
# The damping strength increases linearly from zero at 14 km to λ at 20 km.

zˢ = 14000  # Sponge starts at 14 km
λ = 1/20    # Maximum damping rate [1/s] (1-minute e-folding timescale)

@inline function sponge_mask(i, j, k, grid, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    return clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
end

@inline function w_sponge(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end

@inline function u_sponge(i, j, k, grid, clock, fields, p)
    mask = sponge_mask(i, j, k, grid, p)
    @inbounds ρu = fields.ρu[i, j, k]
    return -p.λ * mask * ρu
end

# Theta sponge: relax ρθ toward the reference state to damp thermal perturbations
# near the tropopause. Without this, convective overshoots produce extreme T values
# (>2000 K) that eventually cause NaN.
# Note: θᵣ is computed on-the-fly from Tᵣ and pᵣ (the reference state fields)
# because reference_state.potential_temperature is just a scalar (θ₀ from the constructor),
# not the actual θ profile. This also adapts automatically when set_to_mean! updates
# the reference state temperature and pressure.
@inline function θ_sponge(i, j, k, grid, clock, fields, p)
    mask = sponge_mask(i, j, k, grid, p)
    @inbounds begin
        ρθ = fields.ρθ[i, j, k]
        Tᵣ = p.Tᵣ[i, j, k]
        pᵣ = p.pᵣ[i, j, k]
        ρᵣ = p.ρᵣ[i, j, k]
    end
    Π = (pᵣ / p.pˢᵗ) ^ p.κ
    θᵣ = Tᵣ / Π
    ρθᵣ = ρᵣ * θᵣ
    return -p.λ * mask * (ρθ - ρθᵣ)
end

cᵖᵈ = constants.dry_air.heat_capacity
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
κ = Rᵈ / cᵖᵈ  # ≈ 0.286 for dry air
pˢᵗ = reference_state.standard_pressure

sponge_params = (; λ, zˢ, zᵗ)
ρw_sponge = Forcing(w_sponge, discrete_form=true, parameters=sponge_params)
ρu_sponge = Forcing(u_sponge, discrete_form=true, parameters=sponge_params)

θ_sponge_params = (; λ, zˢ, zᵗ, κ, pˢᵗ,
                     Tᵣ=reference_state.temperature,
                     pᵣ=reference_state.pressure,
                     ρᵣ=reference_state.density)
ρθ_sponge = Forcing(θ_sponge, discrete_form=true, parameters=θ_sponge_params)

# ## Model assembly
#
# Uses the potential temperature formulation (prognostic variable ρθ = ρ₀θ)
# with one-moment cloud microphysics for autoconversion, accretion, and rain evaporation.
# Smagorinsky-Lilly closure provides consistent subgrid-scale diffusion.

boundary_conditions = (; ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

using CloudMicrophysics
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

weno_order = 5
momentum_advection = WENO(order=weno_order)

scalar_advection = (ρθ = WENO(order=weno_order),
                    ρqᵗ = WENO(order=weno_order, bounds=(0, 1)),
                    ρqᶜˡ = WENO(order=weno_order, bounds=(0, 1)),
                    ρqʳ = WENO(order=weno_order, bounds=(0, 1)),
                    ρqᶜⁱ = WENO(order=weno_order, bounds=(0, 1)),
                    ρqˢ = WENO(order=weno_order, bounds=(0, 1)))

closure = nothing #DynamicSmagorinsky(schedule=IterationInterval(5))

forcing = (; ρw=ρw_sponge, ρu=ρu_sponge, ρv=ρu_sponge, ρθ=ρθ_sponge)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, closure,
                        momentum_advection, scalar_advection, radiation,
                        boundary_conditions, forcing)

# ## Initial conditions
#
# Approximate tropical sounding: ~6.5 K/km lapse rate with 200 K isothermal stratosphere,
# exponentially decreasing moisture with 2.5 km scale height.

function Tᵢ(z)
    T_surface = SST - 1  # Slight air-sea temperature difference
    Γ = 6.5e-3           # Lapse rate [K/m]
    T_trop = 200         # Stratosphere temperature [K]
    T_adiabat = T_surface - Γ * z
    ## Smooth transition to isothermal stratosphere (softplus avoids kink in dT/dz)
    δ = 3  # Smoothing scale [K] (~500 m transition zone)
    return T_trop + δ * log(1 + exp((T_adiabat - T_trop) / δ))
end

function qᵗᵢ(z)
    q₀ = 0.018    # Surface specific humidity (~80% RH at 300 K)
    Hq = 2500     # Moisture scale height [m]
    q_min = 1e-6  # Minimum humidity in stratosphere
    return max(q₀ * exp(-z / Hq), q_min)
end

# Random perturbations in the lowest 2 km to trigger convection
δT = 0.5      # Temperature perturbation amplitude [K]
δq = 1e-4     # Moisture perturbation amplitude [kg/kg]
zδ = 2000     # Perturbation depth [m]

ϵ() = rand() - 0.5
Tᵢ_pert(x, y, z) = Tᵢ(z) + δT * ϵ() * (z < zδ)
qᵢ_pert(x, y, z) = qᵗᵢ(z) + δq * ϵ() * (z < zδ)

# Recompute the hydrostatic reference state from the initial profiles.
# This must happen before `set!` so the reference density matches the initial conditions.
compute_reference_state!(reference_state, Tᵢ, qᵗᵢ, constants)
set!(model; T=Tᵢ_pert, qᵗ=qᵢ_pert)

T = model.temperature
qᵗ = model.specific_moisture
u, v, w = model.velocities
qˡ = model.microphysical_fields.qˡ

@info "RCEMIP RCE"
@info "Domain: $(Lx/1000) km × $(Ly/1000) km × $(zᵗ/1000) km"
@info "Grid: $(Nx) × $(Ny) × $(Nz)"
@info "SST = $(SST) K"
@info "Initial T range: $(minimum(T)) - $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) - $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# 2.5-day spinup demonstrating convection onset and intensification.

simulation = Simulation(model; Δt=1, stop_time=14days) #2.5days)
conjure_time_step_wizard!(simulation, cfl=0.7) #, max_Δt=2)

update_reference_state!(sim) = set_to_mean!(sim.model.dynamics.reference_state, sim.model)
add_callback!(simulation, update_reference_state!, IterationInterval(10))

wall_clock = Ref(time_ns())
progress_time = Ref(time(simulation))

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)

    OLR = mean(view(radiation.upwelling_longwave_flux, :, :, Nz+1))
    SW_in = -mean(view(radiation.downwelling_shortwave_flux, :, :, Nz+1))

    msg = @sprintf("Iter: %5d, t: %8s, Δt: %6s, wall: %8s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed))
    msg *= @sprintf(", max|w|: %5.2f m/s, T: [%5.1f, %5.1f] K, max(qˡ): %.2e",
                   wmax, Tmin, Tmax, qˡmax)
    msg *= @sprintf(", OLR: %.1f W/m², SW_in: %.1f W/m²", OLR, SW_in)

    elapsed_simulation_time = time(sim) - progress_time[]
    SYPD = elapsed_simulation_time / 365days / (elapsed / 86400)

    msg *= @sprintf(", SYPD: %.2f", SYPD)
    @info msg

    progress_time[] = time(sim)
    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# Save horizontally-averaged profiles every 12 hours (time-averaged over the interval)
# and 2D slices every 2 hours for visualization of convective structures.

qᵛ = model.microphysical_fields.qᵛ

outputs = (; u, v, w, T, qˡ, qᵛ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "rce_$(Nx).jld2"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename,
                                                  schedule = AveragedTimeInterval(12hours),
                                                  overwrite_existing = true)

k_slice = 10  # Vertical index for horizontal slices (~3 km, cloud layer)

slice_outputs = (
    wxz = view(w, :, 1, :),     # Vertical velocity in the xz plane (y = 0)
    qˡxz = view(qˡ, :, 1, :),  # Cloud liquid in the xz plane
    wxy = view(w, :, :, k_slice),    # Vertical velocity at ~3 km
    qˡxy = view(qˡ, :, :, k_slice), # Cloud liquid at ~3 km
)

slices_filename = "rce_slices_$(Nx).jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(2hours),
                                                overwrite_existing = true)

@info "Starting RCE simulation..."
run!(simulation)

@info "Simulation completed!"
@info "Final T range: $(minimum(T)) - $(maximum(T)) K"
@info "max|w|: $(maximum(abs, w)) m/s"
@info "max(qˡ): $(maximum(qˡ)) kg/kg"

# ## Visualization
#
# Load the time-averaged profiles and plot the evolution of temperature,
# vapor mass fraction, and mean vertical velocity over the simulation.

Tts = FieldTimeSeries(filename, "T")
qˡts = FieldTimeSeries(filename, "qˡ")
qᵛts = FieldTimeSeries(filename, "qᵛ")
wts = FieldTimeSeries(filename, "w")

times = Tts.times
Nt = length(times)

# ## Mean profile evolution
#
# Plot every few days to show clear progression without overcrowding.

fig = Figure(size=(1000, 400), fontsize=14)

axT = Axis(fig[1, 1], xlabel="T (K)", ylabel="z (km)")
axq = Axis(fig[1, 2], xlabel="qᵛ (kg/kg)")
axw = Axis(fig[1, 3], xlabel="w (m/s)")

# Select profiles at 12-hour intervals
stop_days = 2.5
plot_hours = collect(0:12:stop_days*24)
plot_indices = Int[]
for h in plot_hours
    idx = findfirst(t -> t ≥ h * 3600, times)
    idx !== nothing && push!(plot_indices, idx)
end
Np = length(plot_indices)

colormap = cgrad(:viridis, Np, categorical=true)
Δz = zᵗ / Nz
zc_km = [(k - 0.5) * Δz / 1000 for k in 1:Nz]
zf_km = [(k - 1) * Δz / 1000 for k in 1:Nz+1]

for (i, n) in enumerate(plot_indices)
    t_hrs = times[n] / 3600
    label = t_hrs == 0 ? "initial" : @sprintf("t = %g hr", t_hrs)

    T_data = interior(Tts[n], 1, 1, :)
    q_data = interior(qᵛts[n], 1, 1, :)
    w_data = interior(wts[n], 1, 1, :)

    lines!(axT, T_data, zc_km, color=colormap[i], label=label)
    lines!(axq, q_data, zc_km, color=colormap[i])
    lines!(axw, w_data, zf_km, color=colormap[i])
end

hideydecorations!(axq, grid=false)
hideydecorations!(axw, grid=false)

fig[0, 1:3] = Label(fig, "RCE Mean Profile Evolution (SST = $(SST) K)", fontsize=16, tellwidth=false)
Legend(fig[2, :], axT, orientation=:horizontal, framevisible=false, tellwidth=false)

save("rce_profiles.png", fig)
fig

# ## Animation of cloud structure
#
# Top: stacked xz slices of w and qˡ showing vertical cross-sections.
# Bottom: xy slices at cloud level (~3 km) showing horizontal structure.

wxz_ts = FieldTimeSeries(slices_filename, "wxz")
qˡxz_ts = FieldTimeSeries(slices_filename, "qˡxz")
wxy_ts = FieldTimeSeries(slices_filename, "wxy")
qˡxy_ts = FieldTimeSeries(slices_filename, "qˡxy")

slice_times = wxz_ts.times
Nt_slices = length(slice_times)

wlim = max(maximum(abs, wxz_ts) / 2, 1f-6)
qˡlim = max(maximum(qˡxz_ts) / 2, 1f-6)

fig = Figure(size=(1000, 1200), fontsize=14)

# xz slices (vertical cross-sections)
axwxz = Axis(fig[2, 1], ylabel="z (km)", title="w (xz slice, y = 0)")
axqxz = Axis(fig[2, 2], ylabel="z (km)", title="qˡ (xz slice, y = 0)")

# xy slices (horizontal cross-sections at ~3 km)
axwxy = Axis(fig[4, 1], ylabel="y (km)", xlabel="x (km)",
             title="w (xy slice, z ≈ 3 km)", aspect=1)
axqxy = Axis(fig[4, 2], ylabel="y (km)", xlabel="x (km)",
             title="qˡ (xy slice, z ≈ 3 km)", aspect=1)

n = Observable(Nt_slices)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
title = @lift "Radiative-Convective Equilibrium at t = " * prettytime(slice_times[$n])

hmwxz = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqxz = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡlim))
heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axqxy, qˡxy_n, colormap=:dense, colorrange=(0, qˡlim))

hideydecorations!(axqxz, grid=false)
hideydecorations!(axqxy, grid=false)

Colorbar(fig[3, 1], hmwxz, vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmqxz, vertical=false, label="qˡ (kg/kg)")

fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

save("rce.png", fig)
fig

# Create animation
CairoMakie.record(fig, "rce.mp4", 1:Nt_slices, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](rce.mp4)
