# # Inertia-gravity waves
#
# This example simulates the propagation of inertia-gravity waves in a stably stratified
# atmosphere, following the classical benchmark test case described by [SkamarockKlemp1994](@cite).
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

using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

# ## Problem parameters
#
# We define the thermodynamic base state and mean wind following [SkamarockKlemp1994](@cite):

p₀ = 100000  # Pa - surface pressure
θ₀ = 300     # K - reference potential temperature
U  = 20      # m s⁻¹ - mean wind
N  = 0.01    # s⁻¹ - Brunt-Väisälä frequency
N² = N^2

# ## Grid configuration
#
# The domain is 300 km × 10 km with 300 × 10 grid points, matching the nonhydrostatic case
# configuration in [SkamarockKlemp1994](@cite).

Nx, Nz = 300, 10
Lx, Lz = 300kilometers, 10kilometers

grid = RectilinearGrid(CPU(), size = (Nx, Nz), halo = (5, 5),
                       x = (0, Lx), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## Atmosphere model setup
#
# We use the anelastic formulation with liquid-ice potential temperature thermodynamics:

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state, thermodynamics=:LiquidIcePotentialTemperature)
advection = WENO(minimum_buffer_upwind_order=3)
model = AtmosphereModel(grid; formulation, advection)

# ## Initial conditions
#
# The perturbation parameters from [SkamarockKlemp1994](@cite):

Δθ = 0.01               # K - perturbation amplitude
a  = 5000               # m - perturbation half-width parameter
x₀ = Lx / 3             # m - perturbation center in x

# The background potential temperature profile with a constant Brunt-Väisälä frequency:

g = model.thermodynamic_constants.gravitational_acceleration
θᵇᵍ(z) = θ₀ * exp(N² * z / g)

# The initial condition combines the background profile with the localized perturbation:

θᵢ(x, z) = θᵇᵍ(z) + Δθ * sin(π * z / Lz) / (1 + (x - x₀)^2 / a^2)

set!(model, θ=θᵢ, u=U)

# ## Simulation
#
# We run for 3000 seconds with a fixed time step equal 24 seconds, matching the simulation time in
# [SkamarockKlemp1994](@cite):

simulation = Simulation(model; Δt=24, stop_time=3000)

# Progress callback:

θ = PotentialTemperature(model)
θᵇᵍf = CenterField(grid)
set!(θᵇᵍf, (x, z) -> θᵇᵍ(z))
θ′ = θ - θᵇᵍf

function progress(sim)
    u, v, w = sim.model.velocities
    msg = @sprintf("Iter: % 4d, t: % 14s, max(θ′): %.4e, max|w|: %.4f",
                   iteration(sim), prettytime(sim), maximum(θ′), maximum(abs, w))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(20))

# ## Output
#
# We save the potential temperature for visualization, including an animation of the
# wave propagation:


outputs = merge(model.velocities, (; θ′))

filename = "inertia_gravity_wave.jld2"
simulation.output_writers[:jld2] = JLD2Writer(model, outputs; filename,
                                              schedule = TimeInterval(100),
                                              overwrite_existing = true)

run!(simulation)

# ## Results: potential temperature perturbation
#
# Following [SkamarockKlemp1994](@cite), we visualize the potential temperature perturbation
# ``θ' = θ - θ^{\rm bg}``. The final state at ``t = 3000 \, {\rm s}`` can be compared
# directly to Figure 3b in [SkamarockKlemp1994](@cite), which shows the analytic solution
# for incompressible flow.
#
# The [CM1 model test page](https://www2.mmm.ucar.edu/people/bryan/cm1/test_inertia_gravity_waves/)
# provides additional comparisons between compressible, anelastic, and incompressible solvers.

θ′t = FieldTimeSeries(filename, "θ′")
times = θ′t.times
Nt = length(times)

# Plot the final potential temperature perturbation (compare to Figure 3b in
# [SkamarockKlemp1994](@cite)):

θ′N = θ′t[Nt]

fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)",
          title = "Potential temperature perturbation θ′ at t = $(Int(times[end])) s")

levels = range(-Δθ/2, stop=Δθ/2, length=20)
hm = contourf!(ax, θ′N, colormap=:balance; levels)
fig

save("inertia_gravity_wave.png", fig)
# ![](inertia_gravity_wave.png)

# ## Animation of wave propagation
#
# The animation shows the evolution of the potential temperature perturbation as the
# inertia-gravity waves propagate away from the initial disturbance:

fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)")
n = Observable(1)

θ′n = @lift θ′t[$n]
title = @lift "Potential temperature perturbation θ′ at t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

hm = heatmap!(ax, θ′n, colormap = :balance, colorrange = (-Δθ/2, Δθ/2))
fig

record(fig, "inertia_gravity_wave.mp4", 1:Nt, framerate=8) do nn
    n[] = nn
end
nothing #hide

# ![](inertia_gravity_wave.mp4)
