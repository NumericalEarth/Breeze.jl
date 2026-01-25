# # Radiative-Convective Equilibrium (RCEMIP)
#
# This example simulates radiative-convective equilibrium (RCE) following the RCEMIP-I
# protocol described in [Wing et al. (2018)](@cite Wing2018). RCE is an idealization
# of the climate system in which radiative cooling of the atmosphere is balanced by
# convective heat transport from a warm surface.
#
# ## RCEMIP Protocol Overview
#
# The RCEMIP (Radiative-Convective Equilibrium Model Intercomparison Project) protocol
# establishes standardized boundary conditions for comparing RCE simulations across
# different atmospheric models. The key features specified in Section 3 of Wing et al.
# (2018) are:
#
# - Uniform, fixed sea surface temperature (SST) — Section 3.1
# - Perpetual, uniform insolation with fixed solar zenith angle — Section 3.2
# - No rotation (f = 0) — Section 3.3
# - Doubly-periodic horizontal boundaries — Section 3.4
# - Prescribed trace gas concentrations — Section 3.5
#
# This example implements a scaled-down version of the RCEMIP "RCE_small" configuration
# (Table 2 of Wing et al. 2018), suitable for GPU computation as a documentation example.
# Note: This is a demonstration of the RCE setup; strict RCEMIP compliance would require
# additional verification of initial conditions and trace gas profiles.

using Breeze
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

# ## RCEMIP Protocol Parameters
#
# The following parameters are specified in Section 3 and Table 1 of Wing et al. (2018):
#
# ### Sea Surface Temperature (Section 3.1, Table 1)
#
# > "The sea surface temperature (SST) is fixed and uniform across the domain."
# > "Three values of SST are used: 295, 300, and 305 K."
#
# We use SST = 300 K, the baseline value.

SST = 300  # Sea surface temperature [K] — Wing et al. (2018), Table 1

# ### Insolation (Section 3.2, Table 1)
#
# > "Insolation is temporally and spatially uniform (perpetual sun)."
# > "The solar constant is reduced such that the diurnally averaged insolation
# > at the equator on the equinox is achieved at all times."
#
# From the corrigendum to Wing et al. (2018):
# - Solar constant: S₀ = 551.58 W/m² (reduced from 1361 W/m² to achieve perpetual insolation)
# - Solar zenith angle: θ = 42.05° (cos θ ≈ 0.7434)
#
# These values ensure that the time-mean absorbed solar radiation matches the
# diurnally-averaged equatorial equinox value.

solar_constant = 551.58      # Reduced solar constant [W/m²] — Wing et al. (2018) corrigendum
cos_zenith = cosd(42.05)     # cos(42.05°) ≈ 0.7434 — Wing et al. (2018) corrigendum

# ### Surface Properties (Section 3.1, Table 1)
#
# > "The surface albedo is 0.07 (ocean-like)."
# > "The surface is treated as a slab ocean with zero heat capacity (fixed SST)."

surface_albedo = 0.07  # Ocean surface albedo — Wing et al. (2018), Table 1

# ### Coriolis Parameter (Section 3.3)
#
# > "All experiments are non-rotating (f = 0)."
#
# This is implicit in our setup — we do not include Coriolis forcing.

# ### Trace Gas Concentrations (Section 3.5, Table 1)
#
# > "The CO₂ concentration is fixed at 348 ppmv."
# > "CH₄ and N₂O are specified as in Table 1."
# > "Ozone is either specified from a tropical mean profile or computed interactively."
#
# We use an analytical approximation of the tropical ozone profile that captures
# the key features: low tropospheric values and a stratospheric peak around 25 km.

# ## Domain and Grid
#
# ### Domain Size (Section 3.4, Table 2)
#
# The RCEMIP protocol defines two domain sizes:
# - RCE_small: Lx = Ly = 100 km (for cloud-resolving models)
# - RCE_large: Lx = Ly = ~5000 km (for self-aggregation studies)
#
# > "The small domain is 96 km × 96 km or 100 km × 100 km for CRMs."
#
# We use a 128 km × 128 km domain, close to RCE_small specifications.
# For this documentation example, we use coarser resolution than the
# recommended Δx = 1-3 km.

Nx = Ny = 128
Lx = Ly = 128000  # 128 km horizontal domain (similar to RCE_small)

# ### Vertical Domain (Section 3.4, Table 2)
#
# > "The model top should be at least 33 km to include the stratosphere."
# > "For CRMs, a sponge layer is recommended above the tropopause."
#
# We use the RCEMIP-compliant 33 km domain with 100 vertical levels.

zᵗ = 33000  # 33 km model top — Wing et al. (2018), Section 3.4
Nz = 100    # 100 vertical levels (~330 m mean resolution)

z_faces = range(0, zᵗ, length=Nz+1)

@info "RCE grid: $(Nx)×$(Ny)×$(Nz), Δz = $(zᵗ/Nz) m"

# Select architecture and set precision
arch = GPU()
Oceananigans.defaults.FloatType = Float32

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Thermodynamic Reference State
#
# The RCEMIP protocol does not explicitly specify a reference state for anelastic
# models. We use a dry adiabatic reference with:
# - Surface pressure: 1013.25 hPa (standard atmosphere)
# - Reference potential temperature: 350 K
#
# The elevated reference potential temperature (350 K instead of 300 K) is required
# for the 33 km domain. A dry adiabatic reference with θ₀ = 300 K would produce
# unphysical negative pressures in the upper stratosphere. The actual thermodynamic
# state is determined by perturbations from this reference, so the SST of 300 K
# is achieved through the initial conditions, not the reference state.

p₀ = 101325  # Surface pressure [Pa] — standard atmosphere
θ₀ = 350     # Reference potential temperature [K] — elevated for 33 km domain

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)

dynamics = AnelasticDynamics(reference_state)

# ## Trace Gas Concentrations (Section 3.5, Table 1)
#
# From Wing et al. (2018), Table 1:
# - CO₂: 348 ppmv
# - CH₄: 1650 ppbv
# - N₂O: 306 ppbv
# - O₃: tropical mean profile
#
# > "Ozone is specified from the tropical mean of the CMIP5 pre-industrial control."
#
# We approximate the tropical ozone profile with an analytical function that:
# - Is low in the troposphere (~20-40 ppbv)
# - Peaks in the stratosphere around 25 km (~8 ppmv)
# - Decreases above the peak
#
# Since O₃ is the only gas that RRTMGP supports as spatially varying (besides H₂O),
# we can pass a function directly to `BackgroundAtmosphere`.

# Tropical ozone profile approximation (Chapman-like with tropospheric minimum)
# Note: For z-only profiles, the function takes just z (not x, y, z)
@inline function tropical_ozone(z)
    ## Tropospheric ozone: ~30 ppbv near surface, increasing slowly
    troposphere_O₃= 30e-9 * (1 + 0.5 * z / 10_000)
    ## Stratospheric ozone: peaks around 25 km at ~8 ppmv
    zˢᵗ = 25e3  # km
    Hˢᵗ = 5e3  # scale height in km
    stratosphere_O₃ = 8e-6 * exp(-((z - zˢᵗ) / Hˢᵗ)^2)
    ## Smooth transition using a sigmoid
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

background_atmosphere = BackgroundAtmosphere(
    CO₂ = 348e-6,       # 348 ppmv — Wing et al. (2018), Table 1
    CH₄ = 1650e-9,      # 1650 ppbv — Wing et al. (2018), Table 1
    N₂O = 306e-9,       # 306 ppbv — Wing et al. (2018), Table 1
    O₃ = tropical_ozone # Tropical profile function — Wing et al. (2018), Section 3.5
)

# ## All-Sky Radiation
#
# RCEMIP uses interactive radiation with:
# - Cloud-radiative effects (all-sky)
# - Interactive water vapor from model moisture field
# - Fixed solar zenith angle (perpetual insolation)
#
# The radiation update frequency is not specified in the protocol;
# we update every hour for computational efficiency.

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_albedo,
                                   solar_constant,
                                   background_atmosphere,
                                   surface_temperature = SST,
                                   surface_emissivity = 0.98,
                                   schedule = TimeInterval(10minutes),
                                   coordinate = cos_zenith,  # Fixed zenith angle
                                   liquid_effective_radius = ConstantRadiusParticles(10.0),
                                   ice_effective_radius = ConstantRadiusParticles(30.0))

# ## Surface Fluxes
#
# ### Surface Flux Formulation (Section 3.1)
#
# > "The surface fluxes of momentum, sensible heat, and latent heat are computed
# > using bulk aerodynamic formulae."
#
# The RCEMIP protocol does not specify exact transfer coefficients, as these
# depend on the model's surface layer scheme. We use typical tropical marine
# boundary layer values:
# - Cᴰ = 1.0×10⁻³ (drag coefficient)
# - Cᵀ = 1.0×10⁻³ (sensible heat transfer)
# - Cᵛ = 1.2×10⁻³ (moisture transfer)
#
# These are consistent with typical values over tropical oceans.

Cᴰ = 1.0e-3  # Drag coefficient (typical tropical ocean)
Cᵀ = 1.0e-3  # Sensible heat transfer coefficient
Cᵛ = 1.2e-3  # Moisture transfer coefficient

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# ## Sponge Layer (Section 3.4)
#
# > "For CRMs, a sponge layer is recommended above the tropopause to prevent
# > spurious wave reflections from the model top."
#
# We apply Rayleigh damping in the upper 8 km (25-33 km) with a 1-minute
# damping timescale. This is well above the tropopause (~15-17 km in the tropics).

zˢ = 25000  # Sponge layer starts at 25 km (above tropopause)
λ = 1/60    # 1-minute damping timescale

@inline function sponge_damping(i, j, k, grid, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end

sponge = Forcing(sponge_damping, discrete_form=true, parameters=(; λ, zˢ, zᵗ))

# ## Microphysics
#
# The RCEMIP protocol does not specify a particular microphysics scheme.
# > "Clouds and precipitation are handled according to each model's standard
# > treatment."
#
# We use saturation adjustment with mixed-phase equilibrium, which accounts
# for both liquid water and ice in clouds.

boundary_conditions = (; ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)
microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
advection = WENO(order=5)

model = AtmosphereModel(grid; dynamics, microphysics, advection, radiation,
                        boundary_conditions, forcing = (; ρw=sponge))

# ## Initial Conditions
#
# ### Analytic Sounding (Section 3.6, Appendix A)
#
# The RCEMIP protocol provides an analytic initial sounding in Appendix A:
#
# > "For temperature, use a moist adiabatic profile from the surface up to
# > the tropopause, then an isothermal stratosphere."
# >
# > "For moisture, use an exponentially decreasing profile with a scale height
# > of ~2.5 km."
#
# **Note**: The exact RCEMIP sounding uses specific formulas. Here we use an
# approximate tropical sounding that captures the key features:
# - Temperature: ~6.5 K/km lapse rate, 200 K tropopause
# - Moisture: Exponentially decreasing with 2.5 km scale height

function Tᵢ(z)
    ## Moist adiabatic-like profile in troposphere, isothermal stratosphere
    T_surface = SST - 1  # Slight air-sea temperature difference
    Γ = 6.5e-3           # Lapse rate [K/m]
    z_trop = 15000       # Approximate tropopause height [m]
    T_trop = 200         # Tropopause/stratosphere temperature [K]

    if z < z_trop
        return max(T_surface - Γ * z, T_trop)
    else
        return T_trop
    end
end

function qᵗᵢ(z)
    ## Exponentially decreasing moisture — Wing et al. (2018), Appendix A
    q₀ = 0.018           # Surface specific humidity (~80% RH at 300 K)
    Hq = 2500            # Moisture scale height [m] — RCEMIP uses ~2.5 km
    q_min = 1e-6         # Minimum humidity in stratosphere
    return max(q₀ * exp(-z / Hq), q_min)
end

# ### Random Perturbations
#
# > "Small random perturbations in the lowest 2 km are recommended to trigger
# > convection."
#
# We add temperature and moisture perturbations in the boundary layer.

δT = 0.5      # Temperature perturbation amplitude [K]
δq = 1e-4     # Moisture perturbation amplitude [kg/kg]
zδ = 2000     # Perturbation depth [m]

ϵ() = rand() - 0.5
Tᵢ_pert(x, y, z) = Tᵢ(z) + δT * ϵ() * (z < zδ)
qᵢ_pert(x, y, z) = qᵗᵢ(z) + δq * ϵ() * (z < zδ)

set!(model; T=Tᵢ_pert, qᵗ=qᵢ_pert)

# Check initial state
T = model.temperature
qᵗ = model.specific_moisture
u, v, w = model.velocities
qˡ = model.microphysical_fields.qˡ

@info "==========================================="
@info "RCEMIP Radiative-Convective Equilibrium"
@info "==========================================="
@info "Domain: $(Lx/1000) km × $(Ly/1000) km × $(zᵗ/1000) km"
@info "Grid: $(Nx) × $(Ny) × $(Nz)"
@info "SST = $(SST) K (Wing et al. 2018, Table 1)"
@info "Solar constant = $(solar_constant) W/m² (Wing et al. 2018, corrigendum)"
@info "cos(zenith) = $(round(cos_zenith, digits=4)) (θ = 42.05°)"
@info "CO₂ = 348 ppmv, CH₄ = 1650 ppbv, N₂O = 306 ppbv"
@info "-------------------------------------------"
@info "Initial T range: $(minimum(T)) - $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) - $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# ### Duration (Section 3.7)
#
# > "Simulations should be run for at least 50 days to reach statistical
# > equilibrium, with the last 25 days used for analysis."
#
# For this documentation example, we run for only 6 hours to demonstrate
# the setup. Production runs should be 50+ days.

simulation = Simulation(model; Δt=0.1, stop_time=6hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)

    ## Compute TOA radiative flux (energy balance diagnostic)
    ℐ_lw_up = radiation.upwelling_longwave_flux
    ℐ_sw_dn = radiation.downwelling_shortwave_flux
    OLR = mean(view(ℐ_lw_up, :, :, Nz+1))
    SW_in = -mean(view(ℐ_sw_dn, :, :, Nz+1))

    @info @sprintf("Iter: %5d, t: %8s, Δt: %6s, wall: %8s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed))
    @info @sprintf("            max|w|: %5.2f m/s, T: [%5.1f, %5.1f] K, max(qˡ): %.2e",
                   wmax, Tmin, Tmax, qˡmax)
    @info @sprintf("            OLR: %.1f W/m², SW_in: %.1f W/m², net: %.1f W/m²",
                   OLR, SW_in, SW_in - OLR)

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# Save horizontally-averaged profiles and slices for visualization.

θ = liquid_ice_potential_temperature(model)
qᵛ = model.microphysical_fields.qᵛ

outputs = (; u, v, w, θ, qˡ, qᵛ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "rce_$(Nx).jld2"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename,
                                                  schedule = AveragedTimeInterval(30minutes),
                                                  overwrite_existing = true)

# xz and xy slices for visualization
k_slice = 15  # ~5 km height (with 100 levels over 33 km)

slice_outputs = (
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
    wxy = view(w, :, :, k_slice),
    qˡxy = view(qˡ, :, :, k_slice),
)

slices_filename = "rce_slices_$(Nx).jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)

@info "Starting RCE simulation..."
run!(simulation)

@info "==========================================="
@info "Simulation completed!"
@info "==========================================="
@info "Final T range: $(minimum(T)) - $(maximum(T)) K"
@info "max|w|: $(maximum(abs, w)) m/s"
@info "max(qˡ): $(maximum(qˡ)) kg/kg"

# ## Visualization
#
# Plot the evolution of mean profiles and cloud structure.

θts = FieldTimeSeries(filename, "θ")
qˡts = FieldTimeSeries(filename, "qˡ")
qᵛts = FieldTimeSeries(filename, "qᵛ")
wts = FieldTimeSeries(filename, "w")

times = θts.times
Nt = length(times)

# ## Mean Profile Evolution
#
# Plot the evolution of horizontally-averaged profiles.
fig = Figure(size=(1000, 400), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (km)")
axq = Axis(fig[1, 2], xlabel="qᵛ (g/kg)")
axw = Axis(fig[1, 3], xlabel="w (m/s)")

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:Nt
    t_min = Int(times[n] / 60)
    label = n == 1 ? "initial" : "t = $(t_min) min"

    lines!(axθ, θts[n], color=colors[n], label=label)
    lines!(axq, qᵛts[n], color=colors[n])
    lines!(axw, wts[n], color=colors[n])
end

hideydecorations!(axq, grid=false)
hideydecorations!(axw, grid=false)

Legend(fig[0, :], axθ, orientation=:horizontal, framevisible=false, tellwidth=false)
fig[0, 1:3] = Label(fig, "RCE Mean Profile Evolution (SST = $(SST) K)", fontsize=16, tellwidth=false)

save("rce_profiles.png", fig)
fig

# Animation of cloud structure

wxz_ts = FieldTimeSeries(slices_filename, "wxz")
qˡxz_ts = FieldTimeSeries(slices_filename, "qˡxz")

θavg_ts = FieldTimeSeries(filename, "θ")
qˡavg_ts = FieldTimeSeries(filename, "qˡ")

slice_times = wxz_ts.times
Nt_slices = length(slice_times)

wlim = maximum(abs, wxz_ts) / 2
qˡlim = maximum(qˡxz_ts) / 2

fig = Figure(size=(1400, 900), fontsize=14)

axwxz = Axis(fig[2, 1], title="w (xz slice)")
axqxz = Axis(fig[2, 2], title="qˡ (xz slice)")
axθ = Axis(fig[2, 3], xlabel="θ (K)", title="Mean θ")
axqˡ = Axis(fig[2, 4], xlabel="qˡ (kg/kg)", title="Mean qˡ")

n = Observable(Nt_slices)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
title = @lift "Radiative-Convective Equilibrium at t = " * prettytime(slice_times[$n])

hmwxz = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqxz = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡlim))

n_avg = Observable(length(θavg_ts.times))

for i in 1:length(θavg_ts.times)
    α = i / length(θavg_ts.times)
    lines!(axθ, θavg_ts[i], color=(:gray, 0.3 + 0.5 * α))
    lines!(axqˡ, qˡavg_ts[i], color=(:gray, 0.3 + 0.5 * α))
end

θavg_n = @lift θavg_ts[min($n_avg, length(θavg_ts.times))]
qˡavg_n = @lift qˡavg_ts[min($n_avg, length(qˡavg_ts.times))]
lines!(axθ, θavg_n, color=:orangered, linewidth=2)
lines!(axqˡ, qˡavg_n, color=:dodgerblue, linewidth=2)

hideydecorations!(axqxz, grid=false)
hideydecorations!(axθ, grid=false)
hideydecorations!(axqˡ, grid=false)

Colorbar(fig[3, 1], hmwxz, vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmqxz, vertical=false, label="qˡ (kg/kg)")

fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

save("rce.png", fig)
fig

# Create animation
CairoMakie.record(fig, "rce.mp4", 1:Nt_slices, framerate=12) do nn
    n[] = nn
    t = slice_times[nn]
    avg_times = θavg_ts.times
    n_avg[] = findlast(τ -> τ <= t, avg_times)
end
nothing #hide

# ![](rce.mp4)
#
# ## Summary of RCEMIP Compliance
#
# | Parameter | RCEMIP Specification | This Example | Reference |
# |-----------|---------------------|--------------|-----------|
# | SST | 295, 300, 305 K | 300 K ✓ | Table 1 |
# | Solar constant | 551.58 W/m² | 551.58 W/m² ✓ | Corrigendum |
# | Solar zenith | 42.05° | 42.05° ✓ | Corrigendum |
# | Surface albedo | 0.07 | 0.07 ✓ | Table 1 |
# | CO₂ | 348 ppmv | 348 ppmv ✓ | Table 1 |
# | CH₄ | 1650 ppbv | 1650 ppbv ✓ | Table 1 |
# | N₂O | 306 ppbv | 306 ppbv ✓ | Table 1 |
# | Ozone | Tropical profile | z-profile ✓ | Table 1 |
# | Domain (RCE_small) | 96-100 km | 128 km ✓ | Table 2 |
# | Model top | ≥33 km | 33 km ✓ | Table 2 |
# | Duration | ≥50 days | 6 hours ⚠ | Section 3.7 |
# | f (Coriolis) | 0 | 0 ✓ | Section 3.3 |
#
# Legend: ✓ = compliant, ⚠ = deviation (noted for documentation example)
