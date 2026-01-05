# # GATE III deep tropical convection (idealized)
#
# This example simulates deep tropical convection following the idealized GATE
# (Global Atmospheric Research Program Atlantic Tropical Experiment) Phase III case,
# in the "GigaLES" benchmark configuration [Khairoutdinov2009](@cite).
#
# The GATE III GigaLES is a canonical test case for large eddy simulations of deep
# tropical convection with organized mesoscale convective systems. The benchmark
# was established circa 2009 using very large domain LES (~200 km) to capture
# multi-scale convective organization.
#
# This implementation uses the **idealized** GATE case (`GATE_IDEAL` in SAM), which has:
# - Steady-state initial conditions and forcings (not time-varying)
# - Easterly near-surface winds
# - Prescribed radiative cooling + large-scale temperature/moisture tendencies
#
# This is the standard intercomparison benchmark used by SAM, DP-SCREAMv1, and other
# cloud-resolving models. For the time-varying observational forcing case, see the
# `GATE` directory in SAM.
#
# ## Key specifications
#
# - **Domain**: 204.8 km × 204.8 km horizontally (GigaLES standard)
# - **Vertical extent**: 27 km (model top) — *not* 19 km (which is the sponge layer start)
# - **Resolution**: "Quarter resolution" starter run: dx = dy = 400 m, Nx = Ny = 512
# - **Vertical grid**: Stretched (~50 m near surface, 100 m in troposphere)
# - **Forcings**: Prescribed radiative cooling + large-scale tendencies (steady)
# - **Surface**: Prescribed SST (299.88 K) with bulk formulas
# - **Sponge layer**: Rayleigh damping from 19 km to 27 km
# - **Winds**: Easterly shear profile, no meridional wind (v = 0)

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode

using CairoMakie
using Printf
using Random
using CUDA
using Interpolations: linear_interpolation, Flat

Random.seed!(743)

# ## Configuration flags
#
# Set to `true` to run at full GigaLES resolution (2048×2048×256),
# or `false` (default) for the quarter-resolution starter run.

full_resolution = false

# ## Domain geometry
#
# The GigaLES GATE III domain is 204.8 km × 204.8 km × 27 km.
# This is **non-negotiable**: the model top is 27 km, not 19 km.
# The sponge/damping layer begins near 19 km.

Lx = Ly = 204800  # m (204.8 km)
z_top = 27000     # m (27 km) — model top, NOT sponge layer start
z_sponge_start = 19000  # m — where damping begins

# Quarter resolution uses 512×512 (dx=dy=400 m) instead of 2048×2048 (dx=dy=100 m)
if full_resolution
    Nx = Ny = 2048
else
    Nx = Ny = 512
end

# ## Stretched vertical grid
#
# We construct a stretched vertical grid similar to SAM's GATE_IDEAL configuration:
# - 50 m uniform spacing near the surface (z < 1275 m)
# - Geometric stretching through the lower-mid troposphere (1275 m to ~5000 m)
# - 100 m uniform spacing through the bulk troposphere (5000 m to 18000 m)
# - Geometric stretching through the upper atmosphere (18000 m to z_top)
#
# This provides fine resolution near the surface and in the boundary layer
# where gradients are strongest, while using coarser resolution aloft.

function gate_vertical_grid(z_top; dz_surface=50, dz_tropo=100, dz_top=300)
    # Build z_faces array from 0 to z_top
    z_faces = [0.0]
    
    # Parameters for stretching
    z_stretch_start_1 = 1275    # Start of first stretch zone
    z_stretch_end_1 = 5100      # End of first stretch zone
    z_uniform_end = 18000       # End of uniform 100 m zone
    z_stretch_end_2 = z_top     # End of second stretch zone (model top)
    
    z = 0
    while z < z_top
        if z < z_stretch_start_1
            # Uniform 50 m near surface
            dz = dz_surface
        elseif z < z_stretch_end_1
            # Stretch from 50 m to 100 m
            fraction = (z - z_stretch_start_1) / (z_stretch_end_1 - z_stretch_start_1)
            dz = dz_surface + fraction * (dz_tropo - dz_surface)
        elseif z < z_uniform_end
            # Uniform 100 m through troposphere
            dz = dz_tropo
        elseif z < z_stretch_end_2
            # Stretch from 100 m to 300 m for damping layer
            fraction = (z - z_uniform_end) / (z_stretch_end_2 - z_uniform_end)
            dz = dz_tropo + fraction * (dz_top - dz_tropo)
        else
            dz = dz_top
        end
        
        z_new = z + dz
        if z_new > z_top
            z_new = z_top
        end
        push!(z_faces, z_new)
        z = z_new
    end
    
    return z_faces
end

z_faces = gate_vertical_grid(z_top)
Nz = length(z_faces) - 1

@info "Vertical grid: Nz = $Nz levels, z_top = $(z_faces[end]) m"
@info "Δz range: $(minimum(diff(z_faces))) m to $(maximum(diff(z_faces))) m"

# Runtime assertion: z_top must be exactly 27 km
@assert abs(z_faces[end] - 27000) < 1 "Model top must be 27 km! Got $(z_faces[end]) m"

# ## Grid construction

Oceananigans.defaults.FloatType = Float32

grid = RectilinearGrid(GPU();
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Thermodynamic reference state
#
# The idealized GATE case has surface pressure around 1012 hPa.
# We use a reference potential temperature consistent with the
# tropical marine boundary layer (~298 K).

p₀ = 101200  # Pa (1012 hPa)
θ₀ = 298     # K (approximate surface θ)

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants,
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)

dynamics = AnelasticDynamics(reference_state)

# ## Idealized GATE profiles (from SAM GATE_IDEAL)
#
# These are the steady-state profiles used in the GigaLES benchmark.
# Data from SAM's GATE_IDEAL/snd file (33 levels from surface to ~17 km).
# Units:
# - z: meters
# - θ: Kelvin (potential temperature)
# - qᵛ: kg/kg (converted from g/kg)
# - u: m/s (v = 0 in idealized case)

# Height levels (m) — 33 points from ~47 m to 17.4 km
z_data = [
    46.6, 160.3, 301.7, 470.7, 667.2, 891.4, 1143.1, 1422.4, 1729.3, 2063.8,
    2425.9, 2815.5, 3232.8, 3677.6, 4150.0, 4650.0, 5177.6, 5732.8, 6315.5,
    6925.9, 7563.8, 8229.3, 8922.4, 9643.1, 10391.4, 11167.2, 11970.7, 12801.7,
    13660.3, 14546.6, 15460.3, 16401.7, 17370.7
]
z_data_top = z_data[end]

# Potential temperature θ (K)
θ_data = [
    297.876, 297.905, 297.911, 297.911, 299.235, 300.136, 301.200, 302.536,
    304.035, 305.737, 307.545, 309.369, 311.280, 313.213, 315.411, 317.924,
    320.610, 323.480, 326.258, 329.099, 331.582, 334.065, 336.200, 338.102,
    339.802, 341.230, 342.889, 345.010, 348.292, 354.478, 364.703, 380.539, 400.375
]

# Water vapor mixing ratio (kg/kg, converted from g/kg)
qᵛ_data = [
    16.500, 16.500, 16.500, 16.500, 15.004, 14.054, 13.020, 11.982, 10.891,
    9.860, 8.829, 7.929, 7.036, 6.212, 5.392, 4.593, 3.763, 2.899, 2.175,
    1.497, 1.120, 0.787, 0.544, 0.347, 0.190, 0.114, 0.060, 0.027, 0.011,
    0.010, 0.010, 0.010, 0.010
] ./ 1000  # Convert g/kg to kg/kg

# Zonal velocity u (m/s) — easterly shear profile
u_data = [
    -1.00, -1.00, -1.40, -1.70, -1.80, -2.10, -2.60, -3.60, -4.90, -6.70,
    -8.70, -10.60, -12.20, -13.20, -13.10, -12.00, -9.70, -6.60, -3.20, -0.30,
    1.30, 1.30, 0.40, -1.00, -2.90, -4.30, -4.50, -2.80, -0.60, -0.50,
    -2.20, -2.50, -3.00
]

# Meridional velocity v = 0 throughout (idealized case)
# No v_data needed

# ## Large-scale forcing tendencies (from SAM GATE_IDEAL/lsf)
#
# These are the steady-state large-scale tendencies (combined advection effects).
# Units: K/s for temperature, kg/kg/s for moisture

# Large-scale θ tendency (K/s) — from GATE_IDEAL/lsf tls column
dθdt_ls_data = [
    0.1620e-05, -0.1472e-05, -0.6236e-05, -0.1208e-04, -0.1825e-04, -0.2415e-04,
    -0.3005e-04, -0.3451e-04, -0.3866e-04, -0.4113e-04, -0.4335e-04, -0.4478e-04,
    -0.4602e-04, -0.4686e-04, -0.4728e-04, -0.4716e-04, -0.4621e-04, -0.4458e-04,
    -0.4232e-04, -0.3982e-04, -0.3651e-04, -0.3279e-04, -0.2739e-04, -0.2128e-04,
    -0.1468e-04, -0.9586e-05, -0.5854e-05, -0.3348e-05, -0.1626e-05, -0.6005e-06,
    0.0, 0.0, 0.0
]

# Large-scale qᵛ tendency (kg/kg/s) — from GATE_IDEAL/lsf qls column
dqᵛdt_ls_data = [
    0.3708e-08, 0.5934e-08, 0.9795e-08, 0.1461e-07, 0.1908e-07, 0.2208e-07,
    0.2468e-07, 0.2516e-07, 0.2534e-07, 0.2446e-07, 0.2342e-07, 0.2203e-07,
    0.2050e-07, 0.1876e-07, 0.1704e-07, 0.1536e-07, 0.1375e-07, 0.1219e-07,
    0.1059e-07, 0.8993e-08, 0.7456e-08, 0.5973e-08, 0.4573e-08, 0.3282e-08,
    0.2091e-08, 0.1213e-08, 0.5837e-09, 0.2026e-09, 0.5677e-10, 0.1018e-10,
    0.0, 0.0, 0.0
]

# ## Radiative forcing (from SAM GATE_IDEAL/rad)
#
# Prescribed radiative cooling tendency for temperature.
# Note: This is applied as dT/dt, which approximately equals dθ/dt for
# the lower atmosphere.

# Pressure levels for radiative forcing (hPa)
p_rad = [
    1006.68, 993.76, 977.87, 959.12, 937.67, 913.70, 887.38, 858.91, 828.51,
    796.40, 762.83, 728.03, 692.23, 655.67, 618.59, 581.27, 543.94, 506.84,
    470.17, 434.13, 398.88, 364.59, 331.38, 299.38, 268.72, 239.49, 211.81,
    185.81, 161.60, 139.36, 119.24, 101.39, 85.80
]

# Radiative θ tendency (K/s)
dθdt_rad_data = [
    -0.3200e-04, -0.2633e-04, -0.1935e-04, -0.1112e-04, -0.7770e-05, -0.9033e-05,
    -0.1042e-04, -0.1192e-04, -0.1300e-04, -0.1390e-04, -0.1485e-04, -0.1530e-04,
    -0.1544e-04, -0.1559e-04, -0.1640e-04, -0.1735e-04, -0.1808e-04, -0.1772e-04,
    -0.1736e-04, -0.1683e-04, -0.1607e-04, -0.1534e-04, -0.1424e-04, -0.1290e-04,
    -0.1161e-04, -0.1020e-04, -0.8555e-05, -0.7008e-05, -0.5568e-05, -0.4244e-05,
    0.0, 0.0, 0.0
]

# Total θ tendency (large-scale + radiation)
dθdt_total_data = dθdt_ls_data .+ dθdt_rad_data

# ## Interpolation functions
#
# We create interpolants for the profiles and extend them
# appropriately above the data top (~17.4 km) to the model top (27 km).

θ_interp = linear_interpolation(z_data, θ_data, extrapolation_bc=Flat())
qᵛ_interp = linear_interpolation(z_data, qᵛ_data, extrapolation_bc=Flat())
u_interp = linear_interpolation(z_data, u_data, extrapolation_bc=Flat())
dθdt_total_interp = linear_interpolation(z_data, dθdt_total_data, extrapolation_bc=Flat())
dqᵛdt_ls_interp = linear_interpolation(z_data, dqᵛdt_ls_data, extrapolation_bc=Flat())

# Extension functions for z > z_data_top
# Above the data top:
# - θ: extend with stable gradient
# - qᵛ: hold at near-zero value
# - u: hold constant at last value
# - tendencies: set to zero

θ_gradient_top = (θ_data[end] - θ_data[end-1]) / (z_data[end] - z_data[end-1])
θ_top = θ_data[end]
u_top = u_data[end]
qᵛ_top = qᵛ_data[end]

function θ_profile(z)
    if z ≤ z_data_top
        return θ_interp(z)
    else
        # Linear extrapolation with stable gradient
        return θ_top + θ_gradient_top * (z - z_data_top)
    end
end

function qᵛ_profile(z)
    if z ≤ z_data_top
        return max(0, qᵛ_interp(z))
    else
        return qᵛ_top  # Hold at stratospheric value (~0.01 g/kg)
    end
end

function u_profile(z)
    if z ≤ z_data_top
        return u_interp(z)
    else
        return u_top  # Hold constant
    end
end

# v = 0 everywhere in idealized case
v_profile(z) = 0

function dθdt_profile(z)
    if z ≤ z_data_top
        return dθdt_total_interp(z)
    else
        return 0  # No forcing above data
    end
end

function dqᵛdt_profile(z)
    if z ≤ z_data_top
        return dqᵛdt_ls_interp(z)
    else
        return 0  # No forcing above data
    end
end

# ## Surface conditions
#
# The idealized GATE case uses prescribed SST with bulk formulas.
# From SAM GATE_IDEAL/sfc: SST = 299.88 K

T_surface = 299.88  # K (idealized SST)

# Bulk transfer coefficients
Cᴰ = 1.2e-3  # Drag coefficient for momentum
Cᵀ = 1.1e-3  # Sensible heat transfer coefficient
Cᵛ = 1.2e-3  # Moisture flux transfer coefficient

ρθ_flux = BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=T_surface)
ρqᵗ_flux = BulkVaporFlux(coefficient=Cᵛ, surface_temperature=T_surface)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_flux)
ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_flux)
ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# ## Sponge layer
#
# Rayleigh damping layer from z_sponge_start (19 km) to z_top (27 km).
# Damps vertical velocity toward zero to prevent spurious reflections.

sponge_width = z_top - z_sponge_start
sponge_center = (z_top + z_sponge_start) / 2
sponge_rate = 1/60  # s⁻¹ (~1 minute timescale at center)
sponge_mask = GaussianMask{:z}(center=sponge_center, width=sponge_width/2)
sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

# ## Wind nudging
#
# Nudge horizontal winds toward the initial profiles.
# Using a 6-hour timescale following SAM convention.

τ_uv = 21600  # s (6 hours)

FT = eltype(grid)
ρᵣ = reference_state.density

# Create target wind profile fields
uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
set!(uᵍ, z -> u_profile(z))
set!(vᵍ, z -> v_profile(z))  # v = 0

# Wind nudging toward target profiles
u_nudge = Relaxation(rate=1/τ_uv, target=uᵍ)
v_nudge = Relaxation(rate=1/τ_uv, target=vᵍ)

# ## Large-scale tendencies
#
# Apply the prescribed large-scale θ tendency (including radiation)
# and moisture tendency as forcing terms.

∂t_ρθ_ls = Field{Nothing, Nothing, Center}(grid)
∂t_ρqᵗ_ls = Field{Nothing, Nothing, Center}(grid)

set!(∂t_ρθ_ls, z -> dθdt_profile(z))
set!(∂t_ρθ_ls, ρᵣ * ∂t_ρθ_ls)

set!(∂t_ρqᵗ_ls, z -> dqᵛdt_profile(z))
set!(∂t_ρqᵗ_ls, ρᵣ * ∂t_ρqᵗ_ls)

Fρθ_ls = Forcing(∂t_ρθ_ls)
Fρqᵗ_ls = Forcing(∂t_ρqᵗ_ls)

# ## Coriolis forcing
#
# GATE observations are at latitude ~8.5°N.
# f = 2Ω sin(φ) where Ω = 7.292e-5 s⁻¹

latitude = 8.5  # degrees N
f = 2 * 7.292e-5 * sind(latitude)
coriolis = FPlane(f=f)

# ## Assemble forcing and boundary conditions

forcing = (ρu = u_nudge,
           ρv = v_nudge,
           ρw = sponge,
           ρθ = Fρθ_ls,
           ρqᵗ = Fρqᵗ_ls)

boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

# ## Model setup
#
# We use saturation adjustment microphysics (warm phase for simplicity,
# though ice processes are important for deep convection).

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=5)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, advection,
                        forcing, boundary_conditions)

# ## Initial conditions
#
# Initialize with the idealized profiles plus small random perturbations
# in the boundary layer to trigger convection.

δθ = 0.5      # K - perturbation amplitude
δqᵛ = 1e-4    # kg/kg - moisture perturbation
zδ = 2000     # m - perturbation depth

ϵ() = rand() - 0.5
θᵢ(x, y, z) = θ_profile(z) + δθ * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵛ_profile(z) + δqᵛ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u_profile(z)
vᵢ(x, y, z) = v_profile(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ, v=vᵢ)

# ## Diagnostics output

@info "GATE III idealized GigaLES configuration summary:"
@info "  Domain: $(Lx/1000) km × $(Ly/1000) km × $(z_top/1000) km"
@info "  Grid: $(Nx) × $(Ny) × $(Nz)"
@info "  Δx = Δy = $(Lx/Nx) m"
@info "  Δz range: $(minimum(diff(z_faces))) m to $(maximum(diff(z_faces))) m"
@info "  z_top = $(z_faces[end]) m (model top)"
@info "  z_sponge_start = $(z_sponge_start) m"
@info "  SST = $(T_surface) K"
@info "  Wind nudging τ = $(τ_uv/3600) hours"
@info "  Coriolis f = $(f) s⁻¹ (latitude $(latitude)°N)"

# ## Simulation

simulation = Simulation(model; Δt=5, stop_time=24hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Progress callback
qˡ = model.microphysical_fields.qˡ
wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    qˡmax = maximum(qˡ)
    wmax = maximum(abs, model.velocities.w)
    
    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall: %s, max|w|: %.2f m/s, max(qˡ): %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax, qˡmax)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(500))

# ## Output

θ = liquid_ice_potential_temperature(model)
qᵛ = model.microphysical_fields.qᵛ
u, v, w = model.velocities

outputs = (; u, v, w, θ, qˡ, qᵛ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename = "gate.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs; filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# xz slices for visualization
slice_outputs = (
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "gate_slices.jld2",
                                                schedule = TimeInterval(5minutes),
                                                overwrite_existing = true)

@info "Running GATE III idealized simulation..."
run!(simulation)

# ## Visualization

wxz_ts = FieldTimeSeries("gate_slices.jld2", "wxz")
qˡxz_ts = FieldTimeSeries("gate_slices.jld2", "qˡxz")

times = wxz_ts.times
Nt = length(times)

wlim = maximum(abs, wxz_ts) / 3
qˡlim = maximum(qˡxz_ts) / 3

fig = Figure(size=(1000, 600), fontsize=14)

axw = Axis(fig[2, 1], xlabel="x (km)", ylabel="z (km)", title="Vertical velocity w")
axq = Axis(fig[2, 2], xlabel="x (km)", ylabel="z (km)", title="Cloud liquid qˡ")

n = Observable(Nt)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
title = @lift "GATE III deep convection at t = " * prettytime(times[$n])

hmw = heatmap!(axw, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmq = heatmap!(axq, qˡxz_n, colormap=Reverse(:Blues_4), colorrange=(0, qˡlim))

Colorbar(fig[3, 1], hmw, vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hmq, vertical=false, label="qˡ (kg/kg)")

fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

save("gate.png", fig)
fig

# Animation
CairoMakie.record(fig, "gate.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](gate.mp4)
