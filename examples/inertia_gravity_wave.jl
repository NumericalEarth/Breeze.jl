# # Inertia-gravity waves
#
# This example simulates the propagation of inertia-gravity waves in a stably stratified
# atmosphere, following the classical benchmark test case described by [SkamarockKlemp1994](@citet).
# This test evaluates the accuracy of numerical pressure solvers by introducing a small-amplitude
# temperature perturbation into a stratified environment with constant Brunt-Väisälä frequency,
# triggering propagating inertia-gravity waves.
#
# The test case is particularly useful for validating anelastic and compressible solvers,
# as discussed at the [CM1 inertia-gravity wave test page](https://www2.mmm.ucar.edu/people/bryan/cm1/test_inertia_gravity_waves/).
#
# ## Physical setup
#
# The background state is a stably stratified atmosphere with constant Brunt-Väisälä frequency ``N``,
# which gives a potential temperature profile
#
# ```math
# θ^{\rm bg}(z) = θ_0 \exp\left( \frac{N^2 z}{g} \right)
# ```
#
# where ``θ_0 = 300 \, {\rm K}`` is the surface potential temperature and ``g`` is
# the gravitational acceleration.
#
# The initial perturbation is a localized temperature anomaly centered at ``x = x_0``:
#
# ```math
# θ'(x, z) = Δθ \frac{\sin(π z / L_z)}{1 + (x - x_0)^2 / a^2}
# ```
#
# with amplitude ``Δθ = 0.01 \, {\rm K}``, half-width parameter ``a = 5000 \, {\rm m}``,
# and perturbation center ``x_0 = L_x / 3``. A uniform mean wind ``U = 20 \, {\rm m \, s^{-1}}``
# advects the waves.
#
# ## Comparison of dynamical formulations
#
# Following the CM1 test case comparisons, we compare four different formulations:
#
# 1. **Anelastic**: Filters acoustic waves via the anelastic approximation
# 2. **Anelastic (Boussinesq-like)**: Anelastic with constant reference density
# 3. **Compressible (explicit)**: Fully compressible with explicit time stepping
# 4. **Compressible (acoustic substepping)**: Fully compressible with acoustic substepping
#
# The key differences are:
# - Anelastic formulations filter acoustic waves and use a pressure solver
# - Compressible formulations retain acoustic waves and require smaller time steps
# - Acoustic substepping allows larger advective time steps by substepping the fast acoustic modes

using Breeze
using Breeze.Thermodynamics: adiabatic_hydrostatic_density, adiabatic_hydrostatic_pressure
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

# ## Problem parameters
#
# We define the thermodynamic base state and mean wind following [SkamarockKlemp1994](@citet):

p₀ = 100000  # Pa - surface pressure
θ₀ = 300     # K - reference potential temperature
U  = 20      # m s⁻¹ - mean wind
N  = 0.01    # s⁻¹ - Brunt-Väisälä frequency
N² = N^2

# ## Grid configuration
#
# The domain is 300 km × 10 km with 300 × 10 grid points, matching the nonhydrostatic case
# configuration in the paper by [SkamarockKlemp1994](@citet).

Nx, Nz = 300, 10
Lx, Lz = 300kilometers, 10kilometers

grid = RectilinearGrid(CPU(), size = (Nx, Nz), halo = (5, 5),
                       x = (0, Lx), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## Initial condition functions
#
# The perturbation parameters from the paper by [SkamarockKlemp1994](@citet):

Δθ = 0.01               # K - perturbation amplitude
a  = 5000               # m - perturbation half-width parameter
x₀ = Lx / 3             # m - perturbation center in x

constants = ThermodynamicConstants()
g = constants.gravitational_acceleration

# The background potential temperature profile with a constant Brunt-Väisälä frequency:

θᵇᵍ(z) = θ₀ * exp(N² * z / g)

# The initial condition combines the background profile with the localized perturbation:

θᵢ(x, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

# For compressible dynamics, we also need the background density:

ρᵢ(x, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, constants)

# ## Case 1: Anelastic dynamics
#
# We use the anelastic formulation with liquid-ice potential temperature thermodynamics:

reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
dynamics_anelastic = AnelasticDynamics(reference_state)
advection = WENO(minimum_buffer_upwind_order=3)
model_anelastic = AtmosphereModel(grid; dynamics=dynamics_anelastic, advection)

set!(model_anelastic, θ=θᵢ, u=U)

# ## Case 2: Anelastic (Boussinesq-like with constant reference density)
#
# For the Boussinesq-like case, we set the reference density to a constant value.
# This tests how density variation in the reference state affects wave propagation.

reference_state_const = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)

# Get the surface density and set the entire reference density field to this constant value:
ρ_surface = adiabatic_hydrostatic_density(0, p₀, θ₀, constants)
set!(reference_state_const.density, ρ_surface)

dynamics_boussinesq = AnelasticDynamics(reference_state_const)
model_boussinesq = AtmosphereModel(grid; dynamics=dynamics_boussinesq, advection)

set!(model_boussinesq, θ=θᵢ, u=U)

# ## Case 3: Compressible dynamics (fully explicit)
#
# Fully compressible dynamics without acoustic substepping.
# This requires smaller time steps due to the acoustic CFL constraint.
# We explicitly request `SSPRungeKutta3` to override the default acoustic substepping.

dynamics_compressible = CompressibleDynamics(surface_pressure=p₀)
model_compressible = AtmosphereModel(grid;
                                     dynamics = dynamics_compressible,
                                     advection,
                                     timestepper = :SSPRungeKutta3)  # Override default

set!(model_compressible; θ=θᵢ, u=U, qᵗ=0, ρ=ρᵢ)

# ## Case 4: Compressible dynamics with acoustic substepping (default)
#
# Fully compressible dynamics with acoustic substepping following the Wicker-Skamarock scheme.
# This is the default for `CompressibleDynamics` and allows larger advective time steps
# by substepping the fast acoustic modes.

dynamics_acoustic = CompressibleDynamics(surface_pressure=p₀)
model_acoustic = AtmosphereModel(grid;
                                 dynamics = dynamics_acoustic,
                                 advection)  # Uses default AcousticSSPRungeKutta3

set!(model_acoustic; θ=θᵢ, u=U, qᵗ=0, ρ=ρᵢ)

# ## Time stepping constraints
#
# The anelastic models can use larger time steps since acoustic waves are filtered.
# The fully explicit compressible model requires time steps limited by the sound speed.

Δx = Lx / Nx
Δz = Lz / Nz

# Sound speed (approximately)
Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
cₛ = sqrt(γ * Rᵈ * θ₀)  # ~347 m/s

# CFL-based time steps
cfl = 0.5
Δt_anelastic = cfl * min(Δx, Δz) / U  # Based on advective velocity
Δt_compressible = cfl * min(Δx, Δz) / (cₛ + U)  # Based on sound speed + advection

# For acoustic substepping, we use the advective time step
# The number of acoustic substeps is determined by nsound parameter (default 6)
Δt_acoustic = Δt_anelastic

@info "Time steps:" Δt_anelastic Δt_compressible Δt_acoustic

# ## Simulations
#
# We run for 3000 seconds, matching the simulation time in [SkamarockKlemp1994](@cite):

stop_time = 3000  # seconds

simulation_anelastic = Simulation(model_anelastic; Δt=Δt_anelastic, stop_time)
simulation_boussinesq = Simulation(model_boussinesq; Δt=Δt_anelastic, stop_time)
simulation_compressible = Simulation(model_compressible; Δt=Δt_compressible, stop_time)
simulation_acoustic = Simulation(model_acoustic; Δt=Δt_acoustic, stop_time)

# Progress callbacks:

function make_progress(name, model)
    θ = PotentialTemperature(model)
    θᵇᵍf = CenterField(grid)
    set!(θᵇᵍf, (x, z) -> θᵇᵍ(z))
    θ′ = θ - θᵇᵍf
    
    function progress(sim)
        u, v, w = sim.model.velocities
        msg = @sprintf("%s - Iter: % 4d, t: % 14s, max(θ′): %.4e, max|w|: %.4f",
                       name, iteration(sim), prettytime(sim), maximum(θ′), maximum(abs, w))
        @info msg
        return nothing
    end
    return progress
end

add_callback!(simulation_anelastic, make_progress("Anelastic", model_anelastic), IterationInterval(50))
add_callback!(simulation_boussinesq, make_progress("Boussinesq", model_boussinesq), IterationInterval(50))
add_callback!(simulation_compressible, make_progress("Compressible", model_compressible), IterationInterval(500))
add_callback!(simulation_acoustic, make_progress("Acoustic SS", model_acoustic), IterationInterval(50))

# ## Output
#
# We save the potential temperature perturbation for each case:

function setup_output(simulation, model, filename)
    θ = PotentialTemperature(model)
    θᵇᵍf = CenterField(grid)
    set!(θᵇᵍf, (x, z) -> θᵇᵍ(z))
    θ′ = θ - θᵇᵍf
    
    outputs = merge(model.velocities, (; θ′))
    simulation.output_writers[:jld2] = JLD2Writer(model, outputs; filename,
                                                  schedule = TimeInterval(100),
                                                  overwrite_existing = true)
    return nothing
end

setup_output(simulation_anelastic, model_anelastic, "igw_anelastic.jld2")
setup_output(simulation_boussinesq, model_boussinesq, "igw_boussinesq.jld2")
setup_output(simulation_compressible, model_compressible, "igw_compressible.jld2")
setup_output(simulation_acoustic, model_acoustic, "igw_acoustic.jld2")

# Run all simulations:

@info "Running anelastic simulation..."
run!(simulation_anelastic)

@info "Running Boussinesq-like simulation..."
run!(simulation_boussinesq)

@info "Running fully explicit compressible simulation..."
run!(simulation_compressible)

@info "Running compressible with acoustic substepping..."
run!(simulation_acoustic)

# ## Results: Comparison of dynamical formulations
#
# Following the CM1 test case, we compare the potential temperature perturbation
# at the final time for all four formulations. This comparison reveals:
# - How well each formulation captures inertia-gravity wave propagation
# - The effect of the anelastic approximation vs full compressibility
# - The effect of constant vs variable reference density

θ′_anelastic = FieldTimeSeries("igw_anelastic.jld2", "θ′")
θ′_boussinesq = FieldTimeSeries("igw_boussinesq.jld2", "θ′")
θ′_compressible = FieldTimeSeries("igw_compressible.jld2", "θ′")
θ′_acoustic = FieldTimeSeries("igw_acoustic.jld2", "θ′")

times = θ′_anelastic.times
Nt = length(times)

# Create a 2×2 panel comparison plot (similar to CM1 comparison figures):

fig = Figure(size=(1200, 800))

# Convert x to km for plotting
x_km = range(0, Lx/1000, length=Nx)
z_km = range(0, Lz/1000, length=Nz)

levels = range(-Δθ/2, stop=Δθ/2, length=21)

# Final snapshots
θ′_an = interior(θ′_anelastic[Nt], :, 1, :)
θ′_bo = interior(θ′_boussinesq[Nt], :, 1, :)
θ′_co = interior(θ′_compressible[Nt], :, 1, :)
θ′_ac = interior(θ′_acoustic[Nt], :, 1, :)

ax1 = Axis(fig[1, 1], ylabel = "z (km)", title = "Anelastic")
ax2 = Axis(fig[1, 2], title = "Anelastic (constant ρ)")
ax3 = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Compressible (explicit)")
ax4 = Axis(fig[2, 2], xlabel = "x (km)", title = "Compressible (acoustic SS)")

hidexdecorations!(ax1, grid=false)
hidexdecorations!(ax2, grid=false)
hideydecorations!(ax2, grid=false)
hideydecorations!(ax4, grid=false)

hm1 = contourf!(ax1, x_km, z_km, θ′_an; colormap=:balance, levels)
hm2 = contourf!(ax2, x_km, z_km, θ′_bo; colormap=:balance, levels)
hm3 = contourf!(ax3, x_km, z_km, θ′_co; colormap=:balance, levels)
hm4 = contourf!(ax4, x_km, z_km, θ′_ac; colormap=:balance, levels)

Colorbar(fig[1:2, 3], hm1; label = "θ′ (K)")

fig[0, :] = Label(fig, "Inertia-gravity waves: θ′ at t = $(prettytime(times[Nt]))", fontsize=20)

save("inertia_gravity_wave_comparison.png", fig)

fig

# ## Animation of wave propagation
#
# We create an animation showing the evolution of all four cases side by side:

n = Observable(1)

θ′_an_n = @lift interior(θ′_anelastic[$n], :, 1, :)
θ′_bo_n = @lift interior(θ′_boussinesq[$n], :, 1, :)
θ′_co_n = @lift interior(θ′_compressible[$n], :, 1, :)
θ′_ac_n = @lift interior(θ′_acoustic[$n], :, 1, :)

fig_anim = Figure(size=(1200, 800))

ax1 = Axis(fig_anim[1, 1], ylabel = "z (km)", title = "Anelastic")
ax2 = Axis(fig_anim[1, 2], title = "Anelastic (constant ρ)")
ax3 = Axis(fig_anim[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Compressible (explicit)")
ax4 = Axis(fig_anim[2, 2], xlabel = "x (km)", title = "Compressible (acoustic SS)")

hidexdecorations!(ax1, grid=false)
hidexdecorations!(ax2, grid=false)
hideydecorations!(ax2, grid=false)
hideydecorations!(ax4, grid=false)

hm1 = contourf!(ax1, x_km, z_km, θ′_an_n; colormap=:balance, levels, extendhigh=:auto, extendlow=:auto)
hm2 = contourf!(ax2, x_km, z_km, θ′_bo_n; colormap=:balance, levels, extendhigh=:auto, extendlow=:auto)
hm3 = contourf!(ax3, x_km, z_km, θ′_co_n; colormap=:balance, levels, extendhigh=:auto, extendlow=:auto)
hm4 = contourf!(ax4, x_km, z_km, θ′_ac_n; colormap=:balance, levels, extendhigh=:auto, extendlow=:auto)

Colorbar(fig_anim[1:2, 3], hm1; label = "θ′ (K)")

title = @lift "Inertia-gravity waves: θ′ at t = $(prettytime(times[$n]))"
fig_anim[0, :] = Label(fig_anim, title, fontsize=20, tellwidth=false)

record(fig_anim, "inertia_gravity_wave.mp4", 1:Nt, framerate=8) do nn
    n[] = nn
end
nothing #hide

# ![](inertia_gravity_wave.mp4)

# ## Cross-section comparison
#
# A vertical cross-section at mid-height shows the wave phase and amplitude differences:

z_mid = Nz ÷ 2

fig_cross = Figure(size=(900, 400))
ax = Axis(fig_cross[1, 1], xlabel = "x (km)", ylabel = "θ′ (K)",
          title = "Potential temperature perturbation at z = $(round(z_km[z_mid], digits=1)) km, t = $(prettytime(times[Nt]))")

lines!(ax, x_km, θ′_an[:, z_mid], label = "Anelastic", linewidth=2)
lines!(ax, x_km, θ′_bo[:, z_mid], label = "Anelastic (const ρ)", linewidth=2, linestyle=:dash)
lines!(ax, x_km, θ′_co[:, z_mid], label = "Compressible (explicit)", linewidth=2, linestyle=:dot)
lines!(ax, x_km, θ′_ac[:, z_mid], label = "Compressible (acoustic SS)", linewidth=2, linestyle=:dashdot)

axislegend(ax, position=:rt)

save("inertia_gravity_wave_cross_section.png", fig_cross)

fig_cross
