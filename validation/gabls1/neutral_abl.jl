# # Neutral ABL: Log-law mismatch evaluation
#
# A neutral atmospheric boundary layer based on the Mirocha et al. (2018) SWiFT
# benchmark (see also Lattanzi et al. 2025, ERF paper). This case cleanly isolates
# the near-surface log-law mismatch because MOST predicts φ_m = 1 under neutral
# stratification — any deviation is unambiguously an LES artifact.
#
# The script runs a single simulation at a specified grid aspect ratio Δx/Δz.
# Set `aspect_ratio` below to control horizontal resolution; vertical resolution
# is fixed at Δz = 10 m.
#
# References:
#   Mirocha et al. (2018) Wind Energ. Sci. 3, 589-613
#   Lattanzi et al. (2025) JAMES (ERF model description)
#   Brasseur & Wei (2010) Phys. Fluids 22, 021303

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using Random

Random.seed!(42)

# ## Resolution control
#
# Fixed vertical resolution Δz = 10 m. The grid aspect ratio Δx/Δz determines
# horizontal resolution. Runs from the evaluation plan:
#
#   aspect_ratio = 6   → Δx = 60 m, Nx = 40   (0.3M points)
#   aspect_ratio = 3   → Δx = 30 m, Nx = 80   (1.3M points)
#   aspect_ratio = 2   → Δx = 20 m, Nx = 120  (2.9M points)
#   aspect_ratio = 1.5 → Δx = 15 m, Nx = 160  (5.1M points)
#   aspect_ratio = 1   → Δx = 10 m, Nx = 240  (11.5M points)

aspect_ratio = 2  # ← Change this to run different resolutions

# ## Domain and grid
#
# Domain: 2400 × 2400 × 2000 m. Vertical spacing Δz = 10 m (Nz = 200).
# Horizontal spacing determined by aspect ratio.

Oceananigans.defaults.FloatType = Float32

Lx = Ly = 2400.0 # m
Lz = 2000.0       # m
Δz = 10.0         # m (fixed)
Δx = Δy = aspect_ratio * Δz

Nz = Int(Lz / Δz)
Nx = Ny = Int(Lx / Δx)

@info @sprintf("Grid: %d × %d × %d (Δx = Δy = %.1f m, Δz = %.1f m, Δx/Δz = %.1f, %.1fM points)",
               Nx, Ny, Nz, Δx, Δz, aspect_ratio, Nx * Ny * Nz / 1e6)

grid = RectilinearGrid(GPU();
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state
#
# Dry anelastic dynamics with θ₀ = 300 K and surface pressure 1000 hPa.

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 100000,
                                 potential_temperature = 300.0)

dynamics = AnelasticDynamics(reference_state)

# ## Coriolis
#
# SWiFT facility at 33.5°N: f = 2Ω sin(33.5°) ≈ 8.05 × 10⁻⁵ s⁻¹.

coriolis = FPlane(f = 2 * 7.2921e-5 * sind(33.5))

# ## Surface boundary conditions
#
# Neutral log-law drag coefficient:
#   C_D = (κ / ln(z₁ / z₀))²
# where z₁ = Δz/2 = 5 m and z₀ = 0.05 m (Mirocha et al. 2018).
# No surface heat flux — this is a neutral case.

z₀ = 0.05  # Surface roughness length (m)
z₁ = Δz / 2  # First cell center height (m)
κ  = 0.4   # von Kármán constant
Cd = (κ / log(z₁ / z₀))^2

@info @sprintf("Surface drag coefficient: Cd = %.6f (z₀ = %.3f m, z₁ = %.2f m)", Cd, z₀, z₁)

ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cd))
ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cd))

# ## Geostrophic forcing
#
# Constant geostrophic wind (Ug, Vg) = (6.5, 0) m/s.

geostrophic = geostrophic_forcings(z -> 6.5, z -> 0.0)

# ## Sponge layer
#
# Rayleigh damping in the upper 400 m to prevent wave reflections.
# Damping strength 0.003 s⁻¹ following the ERF setup (Lattanzi et al. 2025).

sponge_rate = 0.003 # s⁻¹
sponge_mask = GaussianMask{:z}(center=Lz, width=200)
sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

# ## Assemble forcing and boundary conditions

forcing = (ρu = geostrophic.ρu,
           ρv = geostrophic.ρv,
           ρw = sponge)

boundary_conditions = (ρu = ρu_bcs,
                       ρv = ρv_bcs)

# ## Model setup
#
# 9th-order WENO with minimum buffer upwind order 1 for low numerical dissipation.
# No explicit SGS closure (ILES).

model = AtmosphereModel(grid;
                        dynamics,
                        coriolis,
                        advection = WENO(order=9, minimum_buffer_upwind_order=1),
                        forcing,
                        boundary_conditions)

# ## Initial conditions
#
# Mirocha et al. (2018) SWiFT benchmark:
#   θ(z) = 300 K for z ≤ 500 m
#   θ(z) = 300 + 0.01(z - 500) for z > 500 m  (10 K/km capping inversion)
#   u = 6.5 m/s, v = 0 everywhere
#   Random θ perturbations ±0.25 K below 500 m to trigger turbulence

zᵢ = 500.0 # inversion base (m)
θᵢ(x, y, z) = (z ≤ zᵢ ? 300.0 : 300.0 + 0.01 * (z - zᵢ)) + 0.25 * (2rand() - 1) * (z < zᵢ)
uᵢ(x, y, z) = 6.5

set!(model, θ=θᵢ, u=uᵢ, v=0)

# ## Simulation
#
# Run for 15 hours to capture the first inertial oscillation wind speed maximum.
# The analysis window is a 2-hour average around that maximum (~hours 11-15
# depending on resolution).

simulation = Simulation(model; Δt=0.5, stop_time=15hours)
conjure_time_step_wizard!(simulation, cfl=0.5)

# ### Progress reporting

wall_clock = Ref(time_ns())

function progress(sim)
    u, v, w = model.velocities
    wmax = maximum(abs, w)
    umax = maximum(abs, u)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    @info @sprintf("Iter: %d, t: %s, Δt: %s, wall: %s, max|w|: %.2e m/s, max|u|: %.1f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax, umax)
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output
#
# Horizontally-averaged profiles averaged over 1-hour windows.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)

outputs = (; u, v, w, θ,
             w² = w^2,
             uw = u * w,
             vw = v * w,
             wθ = w * θ)

averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2))
                              for name in keys(outputs))

ar_str = replace(@sprintf("%.1f", aspect_ratio), "." => "p")
output_filename = "neutral_abl_ar$(ar_str)_averages.jld2"

simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs;
    filename = output_filename,
    schedule = AveragedTimeInterval(1hour),
    overwrite_existing = true)

# ## Run

@info @sprintf("Running neutral ABL (Δx/Δz = %.1f) for 15 hours...", aspect_ratio)
@info @sprintf("Output file: %s", output_filename)
run!(simulation)
@info "Done!"
