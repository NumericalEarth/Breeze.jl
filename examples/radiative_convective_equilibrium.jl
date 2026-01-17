# # Radiative-Convective Equilibrium (RCEMIP)
#
# This example simulates radiative-convective equilibrium (RCE) following the RCEMIP-I
# protocol by [Wing2018](@cite). RCE is a fundamental idealized climate state where
# radiative cooling of the atmosphere is balanced by convective heat transport from
# a warm surface.
#
# The RCEMIP (Radiative-Convective Equilibrium Model Intercomparison Project) protocol
# establishes standardized boundary conditions for comparing RCE simulations across
# different atmospheric models. Key features include:
#
# - Uniform, fixed sea surface temperature (SST)
# - Perpetual insolation (no diurnal or seasonal cycle)
# - No rotation (f = 0)
# - Doubly-periodic horizontal boundaries
#
# This example implements a scaled-down version of the RCEMIP "RCE_small" configuration,
# suitable for GPU computation as a documentation example.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode

using CairoMakie
using Printf
using Random
using Statistics

using NCDatasets  # Required for RRTMGP lookup tables
using RRTMGP

Random.seed!(2024)

# ## RCEMIP Protocol Parameters
#
# The RCEMIP-I protocol specifies the following parameters for RCE experiments:
#
# | Parameter | Value | Notes |
# |-----------|-------|-------|
# | SST | 300 K | Also 295 K, 305 K for sensitivity |
# | Solar constant | 551.58 W/m² | Reduced for perpetual insolation |
# | Solar zenith angle | 42.04° | Fixed (cos θ ≈ 0.743) |
# | Surface albedo | 0.07 | Ocean-like |
# | CO₂ | 348 ppm | Pre-industrial |
# | Coriolis | f = 0 | No rotation |
#
# We scale the domain to 64 km × 64 km × 20 km to run efficiently on GPU.
# The 20 km domain captures the troposphere where deep convection occurs.

SST = 300                    # Sea surface temperature (K)
solar_constant = 551.58      # Reduced solar constant for perpetual insolation (W/m²)
cos_zenith = cosd(42.04)     # Fixed cosine of solar zenith angle (≈ 0.743)
surface_albedo = 0.07        # Ocean surface albedo

# ## Domain and Grid
#
# We use a doubly-periodic domain with a uniform vertical grid.
# For production runs, a stretched grid with higher resolution near the surface
# would be preferable.

Nx = Ny = 64
Lx = Ly = 64000  # 64 km horizontal domain (1 km grid spacing)
zᵗ = 20000       # 20 km model top (captures troposphere)
Nz = 80          # 80 vertical levels (250 m resolution)

z_faces = range(0, zᵗ, length=Nz+1)

@info "RCE grid: $(Nx)×$(Ny)×$(Nz), Δz = $(zᵗ/Nz) m"

# Select architecture and set precision
arch = Oceananigans.GPU()
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
# We use an anelastic formulation with a dry adiabatic reference state.
# The surface pressure is set to 1013.25 hPa and the reference potential
# temperature is set to match the SST.

p₀ = 101325  # Surface pressure (Pa)
θ₀ = 300     # Reference potential temperature (K)

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)

dynamics = AnelasticDynamics(reference_state)

# ## All-Sky Radiation (RCEMIP Configuration)
#
# We use all-sky radiation with cloud-radiative effects. The RCEMIP protocol specifies:
# - Fixed trace gas concentrations (CO₂ = 348 ppm)
# - No aerosols
# - Interactive water vapor from model moisture field
# - Fixed solar zenith angle (perpetual insolation)

FT = eltype(grid)

background = BackgroundAtmosphere{FT}(
    CO₂ = 348e-6,      # Pre-industrial CO₂ (~348 ppm)
    CH₄ = 1650e-9,     # Methane
    N₂O = 306e-9,      # Nitrous oxide
    O₃ = 30e-9         # Approximate tropical mean ozone
)

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_temperature = SST,
                                   surface_emissivity = 0.98,
                                   surface_albedo,
                                   solar_constant,
                                   background_atmosphere = background,
                                   coordinate = cos_zenith,  # Fixed zenith angle!
                                   liquid_effective_radius = ConstantRadiusParticles(10.0),
                                   ice_effective_radius = ConstantRadiusParticles(30.0))

# ## Surface Fluxes
#
# RCEMIP uses bulk aerodynamic formulas with fixed SST. We use typical tropical
# marine boundary layer transfer coefficients.

Cᴰ = 1.0e-3  # Drag coefficient
Cᵀ = 1.0e-3  # Sensible heat transfer coefficient
Cᵛ = 1.2e-3  # Moisture transfer coefficient

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# ## Sponge Layer
#
# Rayleigh damping in the upper atmosphere (above 16 km) prevents spurious
# wave reflections from the model top.

zˢ = 16000  # Sponge layer starts at 16 km
λ = 1/60    # 1-minute damping timescale at sponge center

@inline function sponge_damping(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end

sponge = Forcing(sponge_damping, discrete_form=true, parameters=(; λ, zˢ, zᵗ))

# ## Model Assembly
#
# We use warm-phase saturation adjustment microphysics. For deep tropical convection,
# mixed-phase microphysics would be more realistic, but warm-phase is sufficient
# to demonstrate the RCE state.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=5)

model = AtmosphereModel(grid;
                        dynamics,
                        microphysics,
                        advection,
                        radiation,
                        forcing = (; ρw=sponge),
                        boundary_conditions = (; ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs,
                                                 ρu=ρu_bcs, ρv=ρv_bcs))

# ## Initial Conditions
#
# We initialize with a tropical-like temperature and moisture profile.
# Small random perturbations in the boundary layer trigger convection.

function Tᵢ(z)
    # Moist adiabatic-like profile in troposphere, isothermal stratosphere
    T_surface = SST - 1
    Γ = 6.5e-3           # Lapse rate (K/m)
    z_trop = 15000       # Tropopause height
    T_trop = 200         # Tropopause temperature

    if z < z_trop
        return max(T_surface - Γ * z, T_trop)
    else
        return T_trop
    end
end

function qᵗᵢ(z)
    # Exponentially decreasing moisture with scale height
    q₀ = 0.018           # Surface specific humidity (~80% RH at 300 K)
    Hq = 2500            # Moisture scale height (m)
    q_min = 1e-6         # Minimum humidity in stratosphere
    return max(q₀ * exp(-z / Hq), q_min)
end

# Add random perturbations to trigger convection
δT = 0.5      # Temperature perturbation amplitude (K)
δq = 1e-4     # Moisture perturbation amplitude (kg/kg)
zδ = 2000     # Perturbation depth (m)

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
@info "SST = $(SST) K"
@info "Solar constant = $(solar_constant) W/m²"
@info "cos(zenith) = $(round(cos_zenith, digits=4))"
@info "-------------------------------------------"
@info "Initial T range: $(minimum(T)) - $(maximum(T)) K"
@info "Initial qᵗ range: $(minimum(qᵗ)*1000) - $(maximum(qᵗ)*1000) g/kg"

# ## Simulation
#
# We run for a few days to see the development of convection and approach
# to radiative-convective equilibrium. For full equilibrium, 50+ days would be needed.

simulation = Simulation(model; Δt=1, stop_time=6hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)

    # Compute TOA radiative flux (energy balance diagnostic)
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

add_callback!(simulation, progress, IterationInterval(100))

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
# xy slices at mid-troposphere (~5 km, index k=20 for 80 levels over 20 km)
k_slice = 20  # ~5 km height

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

# Mean profile evolution
fig = Figure(size=(1000, 400), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (km)")
axq = Axis(fig[1, 2], xlabel="qᵛ (g/kg)")
axw = Axis(fig[1, 3], xlabel="w (m/s)")

z_km = znodes(grid, Center()) ./ 1000

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:Nt
    t_min = Int(times[n] / 60)
    label = n == 1 ? "initial" : "t = $(t_min) min"
    
    θ_profile = interior(θts[n], 1, 1, :)
    qᵛ_profile = interior(qᵛts[n], 1, 1, :) .* 1000  # Convert to g/kg
    w_profile = interior(wts[n], 1, 1, :)
    
    lines!(axθ, θ_profile, z_km, color=colors[n], label=label)
    lines!(axq, qᵛ_profile, z_km, color=colors[n])
    lines!(axw, w_profile, z_km, color=colors[n])
end

for ax in (axθ, axq, axw)
    ylims!(ax, 0, 20)
end

hideydecorations!(axq, grid=false)
hideydecorations!(axw, grid=false)

Legend(fig[0, :], axθ, orientation=:horizontal, framevisible=false, tellwidth=false)
fig[0, 1:3] = Label(fig, "RCE Mean Profile Evolution (SST = $(SST) K)", fontsize=16, tellwidth=false)

save("rce_profiles.png", fig)
fig

# Animation of cloud structure (xz and xy slices)

wxz_ts = FieldTimeSeries(slices_filename, "wxz")
qˡxz_ts = FieldTimeSeries(slices_filename, "qˡxz")
wxy_ts = FieldTimeSeries(slices_filename, "wxy")
qˡxy_ts = FieldTimeSeries(slices_filename, "qˡxy")

slice_times = wxz_ts.times
Nt_slices = length(slice_times)

# Set color limits based on maximum values
wlim = max(maximum(abs, wxz_ts), maximum(abs, wxy_ts)) / 2
qˡlim = max(maximum(qˡxz_ts), maximum(qˡxy_ts)) / 2

# Create 2x2 figure: top row = xz slices, bottom row = xy slices
fig = Figure(size=(1200, 1000), fontsize=14)

# xz slices (vertical cross-sections)
axwxz = Axis(fig[2, 1], ylabel="z (km)", title="w (xz slice, y=0)")
axqxz = Axis(fig[2, 2], ylabel="z (km)", title="qˡ (xz slice, y=0)")

# xy slices (horizontal at ~5 km)
z_slice_km = round(k_slice * zᵗ / Nz / 1000, digits=1)
axwxy = Axis(fig[3, 1], xlabel="x (km)", ylabel="y (km)", title="w (xy slice, z=$(z_slice_km) km)")
axqxy = Axis(fig[3, 2], xlabel="x (km)", ylabel="y (km)", title="qˡ (xy slice, z=$(z_slice_km) km)")

n = Observable(Nt_slices)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
title = @lift "Radiative-Convective Equilibrium at t = " * prettytime(slice_times[$n])

# xz heatmaps
hmwxz = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqxz = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡlim))

# xy heatmaps  
hmwxy = heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqxy = heatmap!(axqxy, qˡxy_n, colormap=:dense, colorrange=(0, qˡlim))

# Colorbars
Colorbar(fig[4, 1], hmwxy, vertical=false, label="w (m/s)")
Colorbar(fig[4, 2], hmqxy, vertical=false, label="qˡ (kg/kg)")

# Hide x-axis labels on top row
hidexdecorations!(axwxz, grid=false)
hidexdecorations!(axqxz, grid=false)

# Title
fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

save("rce.png", fig)
fig

# Create animation
CairoMakie.record(fig, "rce.mp4", 1:Nt_slices, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](rce.mp4)
