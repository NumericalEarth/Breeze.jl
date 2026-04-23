# # GABLS1: Stable boundary layer intercomparison
#
# The GABLS1 case [Beare2006](@cite) is a canonical test case for large eddy simulation
# of the stably stratified atmospheric boundary layer, based on the GEWEX Atmospheric
# Boundary Layer Study. The case prescribes a surface cooling rate that drives the
# development of a stably stratified boundary layer capped by an inversion.
#
# This validation case demonstrates the near-surface log-law mismatch problem in LES,
# where resolved wind speeds near the surface overshoot the expected Monin-Obukhov
# similarity profile. This overshoot motivates improved subgrid-scale closures such as
# the Sullivan et al. (1994) two-part eddy viscosity model and the Kosovic (1997)
# nonlinear backscatter and anisotropy (NBA) closure.
#
# We compare Breeze output against intercomparison data from 8 LES groups
# (Beare et al. 2006, Table 1) at 3.125 m resolution.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using Random

Random.seed!(42)

# ## Domain and grid
#
# GABLS1 specifies a 400 × 400 × 400 m domain with uniform 3.125 m grid spacing
# (128 × 128 × 128 grid). This resolution matches the intercomparison data from
# Beare et al. (2006).

Oceananigans.defaults.FloatType = Float32

Nx = Ny = Nz = 128
Lx = Ly = Lz = 400.0 # m

grid = RectilinearGrid(GPU();
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state
#
# The initial potential temperature is 265 K (uniform below 100 m).
# We use dry anelastic dynamics with a surface pressure of 1000 hPa.

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 100000,
                                 potential_temperature = 265.0)

dynamics = AnelasticDynamics(reference_state)

# ## Coriolis
#
# GABLS1 uses f = 1.39 × 10⁻⁴ s⁻¹, corresponding to 73°N latitude.

coriolis = FPlane(f = 1.39e-4)

# ## Surface boundary conditions
#
# Surface fluxes are computed using bulk aerodynamic formulae with a drag coefficient
# derived from the neutral log-law:
#
# ```math
# C_D = \left(\frac{κ}{\ln(z_1 / z_0)}\right)^2
# ```
#
# where κ = 0.4, z₁ = Δz/2 = 1.5625 m (first cell center), and z₀ = 0.1 m (GABLS1 roughness).

z₀ = 0.1   # Surface roughness length (m)
z₁ = Lz / Nz / 2  # First cell center height (m)
κ  = 0.4   # von Kármán constant
Cd = (κ / log(z₁ / z₀))^2

@info @sprintf("Surface drag coefficient: Cd = %.4f (z₀ = %.2f m, z₁ = %.4f m)", Cd, z₀, z₁)

# ### Time-dependent surface temperature
#
# GABLS1 prescribes surface cooling at 0.25 K/hr from an initial temperature of 265 K.
# We store the surface temperature in a 2D Field and update it each timestep via a callback.

T_sfc = Field{Center, Center, Nothing}(grid)
set!(T_sfc, 265.0)

ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cd, surface_temperature=T_sfc))
ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cd, surface_temperature=T_sfc))

ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cd,
                                                                surface_temperature=T_sfc))

# ## Geostrophic forcing
#
# The geostrophic wind is (Ug, Vg) = (8, 0) m/s, constant with height.

geostrophic = geostrophic_forcings(z -> 8.0, z -> 0.0)

# ## Sponge layer
#
# A Rayleigh damping sponge layer in the upper portion of the domain prevents
# spurious wave reflections from the upper boundary. We damp vertical velocity
# toward zero with a Gaussian mask centered at the domain top.

sponge_rate = 1/60 # s⁻¹ (60 s relaxation timescale — gentle for stable ABL)
sponge_mask = GaussianMask{:z}(center=Lz, width=50)
sponge = Relaxation(rate=sponge_rate, mask=sponge_mask)

# ## Assemble forcing and boundary conditions

forcing = (ρu = geostrophic.ρu,
           ρv = geostrophic.ρv,
           ρw = sponge)

boundary_conditions = (ρu = ρu_bcs,
                       ρv = ρv_bcs,
                       ρe = ρe_bcs)

# ## Model setup
#
# We use 5th-order WENO advection. No explicit SGS closure (ILES) — this is
# intentional to demonstrate the log-law overshoot problem that motivates
# improved closures like Sullivan et al. (1994) and Kosovic (1997).

model = AtmosphereModel(grid;
                        dynamics,
                        coriolis,
                        advection = WENO(order=5),
                        forcing,
                        boundary_conditions)

# ## Initial conditions
#
# GABLS1 initial profiles:
# - θ(z) = 265 K for z ≤ 100 m; θ(z) = 265 + 0.01(z - 100) for z > 100 m
# - u = 8 m/s, v = 0 everywhere
# - Random θ perturbations of ±0.1 K below 50 m to trigger turbulence

zₚ = 50.0 # perturbation height (m)
θᵢ(x, y, z) = (z ≤ 100 ? 265.0 : 265.0 + 0.01 * (z - 100)) + 0.1 * (2rand() - 1) * (z < zₚ)
uᵢ(x, y, z) = 8.0

set!(model, θ=θᵢ, u=uᵢ, v=0)

# ## Simulation

simulation = Simulation(model; Δt=0.5, stop_time=9hours)
conjure_time_step_wizard!(simulation, cfl=0.5)

# ### Callbacks
#
# Update the surface temperature each timestep to simulate the prescribed cooling rate.

function update_surface_temperature!(sim)
    t = time(sim)
    T_value = Float32(265 - 0.25 * t / 3600)
    set!(T_sfc, T_value)
end

add_callback!(simulation, update_surface_temperature!, IterationInterval(1))

# Progress reporting

wall_clock = Ref(time_ns())

function progress(sim)
    u, v, w = model.velocities
    wmax = maximum(abs, w)
    umax = maximum(abs, u)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    t = time(sim)
    T_sfc_now = 265 - 0.25 * t / 3600

    @info @sprintf("Iter: %d, t: %s, Δt: %s, wall: %s, max|w|: %.2e m/s, max|u|: %.1f m/s, T_sfc: %.2f K",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax, umax, T_sfc_now)
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output
#
# We output horizontally-averaged profiles of velocity, potential temperature,
# and turbulent fluxes, averaged over 1-hour windows. The final window (8-9 hr)
# is what we compare against the intercomparison data.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)

outputs = (; u, v, w, θ,
             w² = w^2,
             uw = u * w,
             vw = v * w,
             wθ = w * θ)

averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2))
                              for name in keys(outputs))

simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs;
    filename = "gabls1_averages.jld2",
    schedule = AveragedTimeInterval(1hour),
    overwrite_existing = true)

# ## Run

@info "Running GABLS1 stable boundary layer simulation (9 hours)..."
run!(simulation)
@info "Done!"
