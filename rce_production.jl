# RCEMIP Radiative-Convective Equilibrium - Production Run
# 1024² × 80 grid, 102.4 km domain, 100m resolution
# Target: 50 days to equilibrium

using Breeze, Oceananigans, Oceananigans.Units, CUDA, NCDatasets, RRTMGP, Printf, Random, Statistics
using Oceananigans.Grids: znode, znodes

Random.seed!(2024)

# ============================================================
# RCEMIP PARAMETERS (Wing et al. 2018, 2024)
# ============================================================
SST = 300                      # Sea surface temperature (K)
solar_constant = 551.58        # Reduced for perpetual insolation (W/m²)
cos_zenith = cosd(42.04)       # Fixed solar zenith angle (42.04°)

# Grid configuration
Nx = Ny = 1024                 # Horizontal resolution
Δx = Δy = 100                  # 100m horizontal spacing
Lx = Ly = Nx * Δx              # 102.4 km domain
zᵗ = 20000                     # 20 km model top
Nz = 80                        # 80 vertical levels
z_faces = range(0, zᵗ, length=Nz+1)

# Simulation timing
stop_time = 50days             # Run to equilibrium
checkpoint_interval = 1day    # Save checkpoints daily
profile_interval = 15minutes   # Profile output interval
slice_interval = 30minutes     # Slice output interval

arch = Oceananigans.GPU()
Oceananigans.defaults.FloatType = Float32

println("="^70)
println("RCEMIP RCE PRODUCTION RUN")
println("="^70)
println("Grid: $Nx × $Ny × $Nz = $(round(Nx*Ny*Nz/1e6, digits=1))M cells")
println("Domain: $(Lx/1000) km × $(Ly/1000) km × $(zᵗ/1000) km")
println("Resolution: Δx = Δy = $Δx m")
println("Target: $(stop_time / day) days")
println("="^70)

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=z_faces,
                       halo=(5,5,5), topology=(Periodic,Periodic,Bounded))

# Thermodynamics and dynamics
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
dynamics = AnelasticDynamics(reference_state)

# Background atmosphere (RCEMIP values)
background = BackgroundAtmosphere(
    CO₂ = 348e-6,
    CH₄ = 1650e-9,
    N₂O = 306e-9,
    O₃ = 30e-9
)

# Radiation - update every 15 minutes
radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
    surface_temperature = SST,
    surface_emissivity = 0.98,
    surface_albedo = 0.07,
    solar_constant = solar_constant,
    background_atmosphere = background,
    coordinate = cos_zenith,
    schedule = TimeInterval(15minutes),
    liquid_effective_radius = ConstantRadiusParticles(10.0),
    ice_effective_radius = ConstantRadiusParticles(30.0))

# Surface fluxes (bulk formulae)
Cᴰ, Cᵀ, Cᵛ = 1e-3, 1e-3, 1.2e-3
ρθ_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=Cᵀ, surface_temperature=SST))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=Cᵛ, surface_temperature=SST))
ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

# Sponge layer at model top (16-20 km)
zˢ, λ = 16000, 1/60
@inline function sponge_damping(i, j, k, grid, clock, fields, p)
    z = znode(i, j, k, grid, Center(), Center(), Face())
    mask = clamp((z - p.zˢ) / (p.zᵗ - p.zˢ), 0, 1)
    @inbounds ρw = fields.ρw[i, j, k]
    return -p.λ * mask * ρw
end
sponge = Forcing(sponge_damping, discrete_form=true, parameters=(; λ, zˢ, zᵗ))

# Mixed-phase microphysics with saturation adjustment
microphysics = SaturationAdjustment(equilibrium=MixedPhaseEquilibrium())

model = AtmosphereModel(grid;
    dynamics,
    microphysics,
    advection = WENO(order=5),
    radiation,
    forcing = (; ρw=sponge),
    boundary_conditions = (; ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

# Initial conditions - standard tropical sounding with perturbation
Tᵢ(z) = max(299 - 6.5e-3*z, 200)
qᵗᵢ(z) = max(0.018 * exp(-z/2500), 1e-6)
ϵ() = rand() - 0.5
Tᵢ_pert(x, y, z) = Tᵢ(z) + 2.0 * ϵ() * (z < 2000)
qᵢ_pert(x, y, z) = qᵗᵢ(z) + 2e-3 * ϵ() * (z < 2000)

set!(model; T=Tᵢ_pert, qᵗ=qᵢ_pert)

# Extract fields for diagnostics
w = model.velocities.w
u = model.velocities.u
v = model.velocities.v
T = model.temperature
qˡ = model.microphysical_fields.qˡ
qⁱ = model.microphysical_fields.qⁱ
qᵛ = model.microphysical_fields.qᵛ
θ = liquid_ice_potential_temperature(model)

# Set up simulation with adaptive time stepping
simulation = Simulation(model; Δt=1.0, stop_time=stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7)

# Progress tracking
wall_start = Ref(time())
function progress(sim)
    wall_elapsed = time() - wall_start[]
    sim_days = time(sim) / 86400
    sdpd = wall_elapsed > 0 ? sim_days / (wall_elapsed / 86400) : 0.0
    wall_days = wall_elapsed / 86400
    eta_days = (stop_time / day - sim_days) / sdpd

    OLR = mean(view(radiation.upwelling_longwave_flux, :, :, Nz+1))
    SW_dn = mean(view(radiation.downwelling_shortwave_flux, :, :, Nz+1))

    @printf("[Day %.2f] Δt=%.1fs | SDPD=%.1f | ETA=%.1f days | max|w|=%.2f | qˡ=%.1e | OLR=%.1f\n",
            sim_days, simulation.Δt, sdpd, eta_days, maximum(abs, w), maximum(qˡ), OLR)
end
add_callback!(simulation, progress, TimeInterval(1hour))

# ============================================================
# OUTPUT WRITERS
# ============================================================

# 1. Mean profiles (horizontal averages) - for equilibrium analysis
avg_outputs = (
    T = Average(T, dims=(1, 2)),
    θ = Average(θ, dims=(1, 2)),
    qᵛ = Average(qᵛ, dims=(1, 2)),
    qˡ = Average(qˡ, dims=(1, 2)),
    qⁱ = Average(qⁱ, dims=(1, 2)),
    u = Average(u, dims=(1, 2)),
    v = Average(v, dims=(1, 2)),
    w² = Average(w * w, dims=(1, 2)),
    u² = Average(u * u, dims=(1, 2)),
    v² = Average(v * v, dims=(1, 2)),
)

simulation.output_writers[:profiles] = JLD2Writer(model, avg_outputs;
    filename = "rce_production_profiles",
    schedule = AveragedTimeInterval(profile_interval, window=profile_interval),
    overwrite_existing = true)

# 2. Slices for visualization
k_slice = 20  # ~5 km altitude
slice_outputs = (
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
    wxy = view(w, :, :, k_slice),
    qˡxy = view(qˡ, :, :, k_slice),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
    filename = "rce_production_slices",
    schedule = TimeInterval(slice_interval),
    overwrite_existing = true)

# 3. Checkpoints for restart
simulation.output_writers[:checkpointer] = Checkpointer(model;
    schedule = TimeInterval(checkpoint_interval),
    prefix = "rce_checkpoint",
    overwrite_existing = true)

# ============================================================
# RUN SIMULATION
# ============================================================

println("\nStarting 50-day RCE production run...")
println("Expected wall-clock time: ~2 days")
println("Checkpoints saved every $(checkpoint_interval / day) day")
println()

wall_start[] = time()
run!(simulation)

wall_total = time() - wall_start[]
SDPD = (time(simulation) / 86400) / (wall_total / 86400)

println("\n" * "="^70)
println("PRODUCTION RUN COMPLETE")
println("="^70)
println(@sprintf("Simulated time: %.1f days", time(simulation) / 86400))
println(@sprintf("Wall-clock time: %.1f days (%.1f hours)", wall_total / 86400, wall_total / 3600))
println(@sprintf("SDPD: %.2f", SDPD))
println(@sprintf("Iterations: %d", model.clock.iteration))
println(@sprintf("Mean Δt: %.2f s", time(simulation) / model.clock.iteration))

# Final state
OLR = mean(view(radiation.upwelling_longwave_flux, :, :, Nz+1))
SW_dn = mean(view(radiation.downwelling_shortwave_flux, :, :, Nz+1))
println(@sprintf("\nFinal radiation: OLR=%.1f W/m², SW_dn=%.1f W/m²", OLR, SW_dn))
println(@sprintf("Final state: max|w|=%.2f m/s, max(qˡ)=%.2e, max(qⁱ)=%.2e",
                 maximum(abs, w), maximum(qˡ), maximum(qⁱ)))
println("="^70)
