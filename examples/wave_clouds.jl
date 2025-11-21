# # Kelvin–Helmholtz "wave clouds" with Breeze.jl
#
# This script sets up a two-dimensional (x–z) Kelvin–Helmholtz instability
# in a moist, stably stratified atmosphere using Breeze.jl (for thermodynamics
# and moist buoyancy) and Oceananigans.jl (for the nonhydrostatic Boussinesq
# dynamics).
#
# The configuration is intentionally simple but reasonably "meteorological":
#
# - We impose a **tanh shear layer** in the horizontal wind u(z).
# - We impose a **stably stratified** potential temperature profile θ(z) with
#   a specified dry Brunt–Väisälä frequency N.
# - We embed a **Gaussian moisture layer** q(z) centered on the shear layer.
# - We add a **small vertical velocity perturbation** localized at the shear
#   layer to seed the KH instability.
#
# As the shear layer rolls up, the moist layer is advected and deformed,
# producing billow-like patterns reminiscent of observed "wave clouds".
#
# Physically, this reflects the classic picture of billow clouds / KH wave
# clouds:
#
# - A statically stable inversion or stable layer.
# - Strong vertical shear concentrated in a layer.
# - Air in that layer near saturation so that rising branches of the KH
#   waves cross the lifting condensation level (LCL), producing clouds in
#   the wave crests.
#
# For background reading on billow clouds and KH wave clouds in the atmosphere,
# see for example:
#
# - R. Stull, "Billow clouds", UBC ATSC 113 course notes.
# - WW2010 "Kelvin-Helmholtz Billow Clouds" (UIUC).
# - Ludlam (1967), "Characteristics of billow clouds and their relation to
#   clear-air turbulence", J. Atmos. Sci.
# - Case studies of cloud-top KH waves (e.g. mid-level altocumulus billows).
#
# For the effects of moisture on static stability and the moist Brunt–Väisälä
# frequency, see:
#
# - Durran & Klemp (1982), "On the Effects of Moisture on the Brunt–Väisälä
#   Frequency", J. Atmos. Sci.
# - Marquet & Geleyn (2014), "On a general definition of the squared
#   Brunt–Väisälä Frequency associated with the specific moist entropy
#   potential temperature".
#
# Breeze encapsulates much of this thermodynamics for us via
# `MoistAirBuoyancy` and saturation adjustment.


using Breeze
using Oceananigans.Units
# using CairoMakie
using GLMakie
using Printf

# ## Domain and grid
#
# We use a 2-D x–z slice with periodic boundaries in x and rigid, impermeable
# boundaries at the top and bottom.
#
# Grid resolution is modest but enough to clearly resolve the KH billows and
# rolled-up moisture filament.

Nx = 256     # horizontal resolution
Nz = 128     # vertical resolution

Lx = 10_000  # domain length in x [m]  (10 km)
Lz =  2_000  # domain depth in z [m]  (2 km)

grid = RectilinearGrid(CPU();
                       size = (Nx, Nz),
                       x = (0.0, Lx),
                       z = (0.0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## Construct the model
microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
model = AtmosphereModel(grid, advection=WENO(order=5), microphysics=microphysics)

# ## Background thermodynamic state
#
# We set a reference potential temperature θ₀ and a linear θ gradient that
# corresponds to a desired dry Brunt–Väisälä frequency N.
#
# For a dry atmosphere,
#
#     N² ≈ (g / θ₀) dθ/dz,
#
# so for given θ₀ and N² we choose
#
#     dθ/dz = N² θ₀ / g.
#
# Here we pick N ≈ 0.01 s⁻¹, representative of mid-tropospheric stability.

thermo = ThermodynamicConstants()
g = thermo.gravitational_acceleration
θ₀ = model.formulation.reference_state.potential_temperature
N = 0.01                  # target dry Brunt–Väisälä frequency [s⁻¹]
dθdz =  θ₀ * N^2 / g      # dθ/dz [K m⁻¹] ~ 0.003 K/m = 3 K/km
θᵇ(z) = θ₀ + dθdz * z

# ## Shear and moisture profiles
#
# We want:
#
# - A shear layer centered at height z₀ where u(z) transitions from a lower
#   speed U_bot to an upper speed U_top.
# - A moist layer centered at the same height with a Gaussian profile.
#
# This mimics a moist, stably stratified layer embedded in stronger flow
# above and weaker flow below.

z₀    = 1_000.0   # center of shear & moist layer [m]
Δzᶸ   = 150.0     # shear layer half-thickness [m]
U_bot =  5.0      # lower-layer wind [m/s]
U_top = 25.0      # upper-layer wind [m/s]

# Smooth shear layer:
#
#   u(z) ≈ U_bot for z << z₀
#   u(z) ≈ U_top for z >> z₀
#
uᵇ(z) = U_bot + 0.5 * (U_top - U_bot) * (1 + tanh((z - z₀) / Δzᶸ))

# Moisture layer: Gaussian in z around z₀.
#
# q_max ~ 0.012 corresponds to ~12 g/kg, a reasonable mid-level specific humidity.
#
q_max     = 0.012     # peak specific humidity [kg/kg]
Δz_q = 200.0     # moist layer half-width [m]
qᵇ(z) = q_max * exp(-((z - z₀)^2) / 2Δz_q^2))

# ## Initial perturbation: seed the KH instability
#
# The Miles–Howard criterion tells us that Kelvin–Helmholtz instability
# occurs where Ri = N² / (dU/dz)² < 1/4. With the parameters chosen above,
# the shear layer easily satisfies this.
#
# To actually *trigger* the instability in a numerical model, we add a small
# vertical velocity perturbation localized at the shear layer.
#
# We choose:
#
#   w'(x, z) = w₀ * sin(kx) * exp(-((z - z₀)/Δz_pert)^2)
#
# with a few wavelengths across the domain.

# ## Define initial conditions
#
# Oceananigans `set!` can take functions of (x, z) for 2-D grids.
#
# We define:
#
#   u(x, z) = u_profile(z)
#   w(x, z) = w_perturbation(x, z)
#   θ(x, z) = θ_background(z)
#   q(x, z) = q_background(z)
#
# plus a tiny bit of random noise in θ and q so that secondary structures
# have something to grow from.

δθ = 0.01           # ~1% of a typical θ range
δu = 1e-3           # ~1% of a typical θ range
δq = 0.05 * q_max   # 5% of peak humidity

θᵢ(x, z) = θᵇ(z) * (1.0 + δθ * rand() / θ₀)
qᵗᵢ(x, z) = qᵇ(z) + δq * rand()
uᵢ(x, z) = uᵇ(z) + δu * rand()

# Apply initial conditions.
set!(model; u=uᵢ, qᵗ=qᵗᵢ, θ=θᵢ)

# ## Set up and run the simulation
#
# We run for ~30 minutes of physical time, which should be enough for the
# KH billows to roll up significantly for these parameters.
#
# Use the time-step wizard to keep the CFL number under control.

stop_time = 30minutes   # total simulation time

simulation = Simulation(model; Δt=1, stop_iteration=1000) #stop_time)
conjure_time_step_wizard!(simulation; cfl = 0.7)

function progress(sim)
    u, v, w = model.velocities
    max_w = maximum(abs, w)
    @info @sprintf("Iter: %d, t: %s, max|w|: %.2e m/s", iteration(sim), prettytime(sim), max_w)
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

u, v, w = model.velocities
ξ = ∂x(w) - ∂z(u)
θ = PotentialTemperatureField(model)
outputs = merge(model.velocities, model.microphysical_fields, (; ξ, θ))

output_writer = JLD2Writer(model, outputs;
                           filename = "wave_clouds.jld2",
                           schedule = IterationInterval(100))

simulation.output_writers[:fields] = output_writer

run!(simulation)

fig = Figure(size=(1200, 800), fontsize=12)

axu = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axξ = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axl = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")

heatmap!(axu, model.velocities.u)
heatmap!(axξ, ξ)
heatmap!(axl, model.microphysical_fields.qˡ)
display(fig)
