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

using AtmosphericProfilesLibrary
using CairoMakie
using Printf
using Random
using CUDA

Random.seed!(743)

# Select architecture
arch = GPU()
@info "Using architecture: $arch"

# ## Configuration flags
#
# Set to `true` to run at full GigaLES resolution (2048×2048×256),
# or `false` (default) for the quarter-resolution starter run.

full_resolution = false

# ## Domain geometry
#
# The GigaLES GATE III domain is 204.8 km × 204.8 km × 27 km.
# The sponge/damping layer begins near 19 km.

Nx = Ny = 2048
Lx = Ly = 204800  # m (204.8 km)
zᵗ = 27000        # m (27 km) — model top, NOT sponge layer start
zˢ = 19000        # m — where damping begins

# ## Stretched vertical grid
#
# We construct a stretched vertical grid similar to SAM's GATE_IDEAL configuration:
# - 50 m uniform spacing near the surface (z < 1275 m)
# - Linear stretching through the lower-mid troposphere (1275 m to ~5000 m)
# - 100 m uniform spacing through the bulk troposphere (5000 m to 18000 m)
# - Linear stretching through the upper atmosphere (18000 m to z_top)
#
# This provides fine resolution near the surface and in the boundary layer
# where gradients are strongest, while using coarser resolution aloft.

function gate_vertical_grid(zᵗ; Δz⁰=50, Δzᵖ=100, Δzᵗ=300)
    z₁, z₂, z₃ = 1275, 5100, 18000 # transition heights
    z_faces = [0.0]
    z = 0.0

    while z < zᵗ
        α = clamp((z - z₁) / (z₂ - z₁), 0, 1)
        β = clamp((z - z₂) / (z₃ - z₂), 0, 1)
        Δz = Δz⁰ + α * (Δzᵖ - Δz⁰) + β * (Δzᵗ - Δzᵖ)
        z = min(z + Δz, zᵗ)
        push!(z_faces, z)
    end

    return z_faces
end

z_faces = gate_vertical_grid(zᵗ)
Nz = length(z_faces) - 1

@info "Vertical grid: Nz = $Nz levels, z_top = $(z_faces[end]) m"
@info "Δz range: $(minimum(diff(z_faces))) m to $(maximum(diff(z_faces))) m"

# Runtime assertion: z_top must be exactly 27 km
@assert abs(z_faces[end] - 27000) < 1 "Model top must be 27 km! Got $(z_faces[end]) m"

# ## Grid construction

Oceananigans.defaults.FloatType = Float32

grid = RectilinearGrid(arch;
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

# ## Idealized GATE profiles from AtmosphericProfilesLibrary
#
# We use the GATE III profiles from AtmosphericProfilesLibrary, which provides
# the standard GigaLES benchmark profiles including temperature, moisture,
# winds, and large-scale forcing tendencies.

FT = eltype(grid)
T₀ = AtmosphericProfilesLibrary.GATE_III_T(FT)
qᵗ₀ = AtmosphericProfilesLibrary.GATE_III_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.GATE_III_u(FT)
∂t_T = AtmosphericProfilesLibrary.GATE_III_dTdt(FT)
∂t_qᵗ = AtmosphericProfilesLibrary.GATE_III_dqtdt(FT)

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
# Rayleigh damping layer from zˢ (19 km) to zᵗ (27 km).
# Damps vertical velocity toward zero to prevent spurious reflections.

@inline function sponge_damping(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ)/ (p.zᶜ - p.zˢ), 0, 1) 
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end

zᶜ = (zᵗ + zˢ) / 2
λ = 1/10  # s⁻¹ (~1 minute timescale at center)
sponge = Forcing(sponge_damping, discrete_form=true, parameters=(; λ, zˢ, zᶜ))

# ## Wind nudging
#
# Nudge horizontal winds toward the initial profiles.
# Using a 6-hour timescale following SAM convention.

τ_uv = 21600  # s (6 hours)
ρᵣ = reference_state.density

uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
set!(uᵍ, z -> u₀(z))
set!(vᵍ, 0)

@inline function u_nudging(i, j, k, grid, clock, fields, p)
    @inbounds u = fields.u[i, j, k]
    @inbounds uᵍ = p.uᵍ[1, 1, k]
    @inbounds ρᵣ = p.ρᵣ[1, 1, k]
    return -ρᵣ * (u - uᵍ) / p.τ
end

@inline function v_nudging(i, j, k, grid, clock, fields, p)
    @inbounds v = fields.v[i, j, k]
    @inbounds vᵍ = p.vᵍ[1, 1, k]
    @inbounds ρᵣ = p.ρᵣ[1, 1, k]
    return -ρᵣ * (v - vᵍ) / p.τ
end

u_nudge = Forcing(u_nudging, discrete_form=true, parameters=(; uᵍ, ρᵣ, τ=τ_uv))
v_nudge = Forcing(v_nudging, discrete_form=true, parameters=(; vᵍ, ρᵣ, τ=τ_uv))

# ## Large-scale tendencies
#
# Apply the prescribed large-scale temperature tendency (including radiation)
# and moisture tendency as forcing terms. The temperature tendency ∂T/∂t
# must be converted to potential temperature tendency ∂θ/∂t = (∂T/∂t) / Π,
# where Π = (p/pˢᵗ)^(R/cₚ) is the Exner function.

∂t_ρe_ls = Field{Nothing, Nothing, Center}(grid)
∂t_ρqᵗ_ls = Field{Nothing, Nothing, Center}(grid)

# Compute reference Exner function Πᵣ = (pᵣ/pˢᵗ)^(R/cₚ)
cᵖᵈ = constants.dry_air.heat_capacity

# Convert ∂T/∂t → ∂(ρθ)/∂t = ρᵣ × (∂T/∂t) / Πᵣ
set!(∂t_ρe_ls, z -> ∂t_T(z))
set!(∂t_ρe_ls, ρᵣ * cᵖᵈ * ∂t_ρe_ls)

set!(∂t_ρqᵗ_ls, z -> ∂t_qᵗ(z))
set!(∂t_ρqᵗ_ls, ρᵣ * ∂t_ρqᵗ_ls)

Fρe_ls = Forcing(∂t_ρe_ls)
Fρqᵗ_ls = Forcing(∂t_ρqᵗ_ls)

# ## Coriolis forcing
#
# GATE observations are at latitude ~8.5°N.

coriolis = FPlane(latitude=8.5)

# ## Assemble forcing and boundary conditions

forcing = (ρu = u_nudge,
           ρv = v_nudge,
           ρw = sponge,
           ρe = Fρe_ls,
           ρqᵗ = Fρqᵗ_ls)

boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

# ## Model setup
#
# We use saturation adjustment microphysics (warm phase for simplicity,
# though ice processes are important for deep convection).

microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())
advection = WENO(order=5)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, advection,
                        forcing, boundary_conditions)

# ## Initial conditions
#
# Initialize with the idealized profiles plus small random perturbations
# in the boundary layer to trigger convection.

δT = 0.5      # K - temperature perturbation amplitude
δqᵗ = 1e-4    # kg/kg - moisture perturbation
zδ = 2000     # m - perturbation depth

ϵ() = rand() - 0.5
Tᵢ(x, y, z) = T₀(z) + δT * ϵ() * (z < zδ)
qᵗᵢ(x, y, z) = qᵗ₀(z) + δqᵗ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u₀(z)

set!(model, T=Tᵢ, qᵗ=qᵗᵢ, u=uᵢ)

# Check initial state
T = model.temperature
qᵗ = model.specific_moisture
u, v, w = model.velocities
qˡ = model.microphysical_fields.qˡ
qⁱ = model.microphysical_fields.qⁱ

@info "=========================================="
@info "GATE III Idealized GigaLES Simulation"
@info "=========================================="
@info "Domain: $(Lx/1000) km × $(Ly/1000) km × $(zᵗ/1000) km"
@info "Grid: $(Nx) × $(Ny) × $(Nz)"
@info "Δx = Δy = $(Lx/Nx) m"
@info "Δz range: $(minimum(diff(z_faces))) m to $(maximum(diff(z_faces))) m"
@info "Model top: $(z_faces[end]) m"
@info "Sponge start: $(zˢ) m"
@info "SST = $(T_surface) K"
@info "Wind nudging τ = $(τ_uv/3600) hours"
@info "------------------------------------------"
@info "Initial conditions:"
@info "  T range: $(minimum(T)) - $(maximum(T)) K"
@info "  qᵗ range: $(minimum(qᵗ)*1000) - $(maximum(qᵗ)*1000) g/kg"
@info "  u range: $(minimum(u)) - $(maximum(u)) m/s"

# ## Simulation

simulation = Simulation(model; Δt=1, stop_time=24hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

wall_clock = Ref(time_ns())
sim_clock = Ref(time(simulation))

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    simulated = time(simulation) - sim_clock[]
    SDPD = simulated / elapsed

    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)
    qˡmax = maximum(qˡ)
    qⁱmax = maximum(qⁱ)

    @info @sprintf("Iter: %6d, t: %10s, Δt: %8s, wall: %10s, SDPD: %5.1f",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), SDPD)
    @info @sprintf("            max|w|: %6.2f m/s, T: [%6.1f, %6.1f] K, max(qˡ): %.2e, max(qⁱ): %.2e",
                   wmax, Tmin, Tmax, qˡmax, qⁱmax)

    wall_clock[] = time_ns()
    sim_clock[] = time(simulation)

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Output

θ = liquid_ice_potential_temperature(model)
qᵛ = model.microphysical_fields.qᵛ

outputs = (; u, v, w, θ, qˡ, qᵛ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

averages_filename = "gate_$Nx.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = averages_filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# xz slices for visualization
slice_outputs = (
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
    qⁱxz = view(qⁱ, :, 1, :),
)

slices_filename = "gate_slices_$Nx.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(5minutes),
                                                overwrite_existing = true)

@info "Starting 24-hour simulation..."
run!(simulation)

@info "=========================================="
@info "Simulation completed!"
@info "=========================================="
@info "Final state:"
@info "  T range: $(minimum(T)) - $(maximum(T)) K"
@info "  max|w|: $(maximum(abs, w)) m/s"
@info "  max(qˡ): $(maximum(qˡ)) kg/kg"
@info "  max(qⁱ): $(maximum(qⁱ)) kg/kg"

# ## Visualization

wxz_ts = FieldTimeSeries(slices_filename, "wxz")
qˡxz_ts = FieldTimeSeries(slices_filename, "qˡxz")
qⁱxz_ts = FieldTimeSeries(slices_filename, "qⁱxz")

times = wxz_ts.times
Nt = length(times)

wlim = maximum(abs, wxz_ts) / 3
qˡlim = maximum(qˡxz_ts) / 3
qⁱlim = maximum(qⁱxz_ts) / 3

fig = Figure(size=(1400, 500), fontsize=14)

axw = Axis(fig[2, 1], xlabel="x (km)", ylabel="z (km)", title="Vertical velocity w")
axl = Axis(fig[2, 2], xlabel="x (km)", ylabel="z (km)", title="Cloud liquid qˡ")
axi = Axis(fig[2, 3], xlabel="x (km)", ylabel="z (km)", title="Cloud ice qⁱ")

n = Observable(Nt)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
qⁱxz_n = @lift qⁱxz_ts[$n]
title = @lift "GATE III deep convection at t = " * prettytime(times[$n])

hmw = heatmap!(axw, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hml = heatmap!(axl, qˡxz_n, colormap=:dense, colorrange=(0, qˡlim))
hmi = heatmap!(axi, qⁱxz_n, colormap=:ice, colorrange=(0, qⁱlim))

Colorbar(fig[3, 1], hmw, vertical=false, label="w (m/s)")
Colorbar(fig[3, 2], hml, vertical=false, label="qˡ (kg/kg)")
Colorbar(fig[3, 3], hmi, vertical=false, label="qⁱ (kg/kg)")

fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

save("gate.png", fig)
fig

# Animation
CairoMakie.record(fig, "gate.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](gate.mp4)
