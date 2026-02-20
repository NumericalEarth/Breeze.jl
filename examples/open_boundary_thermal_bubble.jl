# # Open boundaries: radiating gravity waves
#
# A rising thermal bubble in a stably stratified atmosphere generates internal
# gravity waves that propagate outward.  In a periodic domain these waves
# re-enter from the opposite side and contaminate the solution.  Open
# lateral boundaries let the waves leave the domain cleanly, which is
# essential for regional and nested simulations.
#
# At each open boundary a Sommerfeld-type radiation condition
# [Klemp (1978)](@cite Klemp1978) advects perturbations outward:
# ```math
# \frac{\partial \psi}{\partial t} + c_b \frac{\partial \psi}{\partial n} = 0
# ```
# where ``c_b`` is a boundary-normal phase speed estimated from the interior
# flow.  Breeze discretizes this equation implicitly (backward Euler) so
# the scheme is unconditionally stable regardless of time-step size.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid
#
# A 2-D vertical slice with `Bounded` lateral topology.
# The domain is 20 km wide and 10 km tall.

Nx, Nz = 200, 100

grid = RectilinearGrid(CPU(); size = (Nx, Nz), halo = (5, 5),
                       x = (0, 20e3), z = (0, 10e3),
                       topology = (Bounded, Flat, Bounded))

# ## Open boundary conditions
#
# [`OpenBoundaryCondition`](@ref) wraps an external reference value
# together with a radiation scheme.  With no mean wind
# the external momentum is zero.  `outflow_timescale = Inf`
# makes the boundary purely radiative (perturbations are extrapolated outward)
# while `inflow_timescale = 0` imposes the external state
# instantly whenever flow reverses.

scheme = PerturbationAdvection(outflow_timescale = Inf, inflow_timescale = 0.0)
open_bc = OpenBoundaryCondition(0.0; scheme)
boundary_conditions = (; ρu = FieldBoundaryConditions(west = open_bc, east = open_bc))

# ## Model
#
# The anelastic model filters acoustic waves while retaining gravity waves.
# Before each pressure solve, the net mass flux through open boundaries is
# corrected to satisfy the anelastic solvability condition
# ``∮ ρ_r \boldsymbol{u} \cdot \mathrm{d}\boldsymbol{A} = 0``.

model = AtmosphereModel(grid; boundary_conditions, advection = WENO(order = 5))

# ## Initial condition
#
# A warm circular bubble (``Δθ = 2`` K, radius 2 km) is placed in a stably
# stratified background (``N² = 10^{-4} \; \mathrm{s}^{-2}``).
# As the bubble rises it excites gravity waves whose group velocity
# carries energy laterally toward the open boundaries.

N² = 1e-4  # Brunt-Väisälä frequency squared (1/s²)
Δθ = 2     # Bubble perturbation (K)
r₀ = 2e3   # Bubble radius (m)

θ₀ = model.dynamics.reference_state.potential_temperature
g = model.thermodynamic_constants.gravitational_acceleration

xc = grid.Lx / 2
zc = 2e3

function θᵢ(x, z)
    θ̄ = θ₀ * exp(N² * z / g)
    r = sqrt((x - xc)^2 + (z - zc)^2)
    θ′ = Δθ * max(0, 1 - r / r₀)
    return θ̄ + θ′
end

set!(model, θ = θᵢ)

# ## Run the simulation

simulation = Simulation(model; Δt = 1.0, stop_time = 15minutes)
conjure_time_step_wizard!(simulation, cfl = 0.5)

function progress(sim)
    w = sim.model.velocities.w
    @info @sprintf("Iter %d  t = %s  Δt = %s  max|w| = %.2f m/s",
                   iteration(sim), prettytime(sim),
                   prettytime(sim.Δt), maximum(abs, w))
end

add_callback!(simulation, progress, TimeInterval(2minutes))

# Output potential temperature perturbation and vertical velocity.

θ = liquid_ice_potential_temperature(model)
Θ = Field(Average(θ, dims = 1))
θ′ = Field(θ - Θ)

w = model.velocities.w

filename = "open_boundary_thermal_bubble.jld2"

simulation.output_writers[:fields] = JLD2Writer(model, (; θ′, w);
                                                filename,
                                                schedule = TimeInterval(15seconds),
                                                overwrite_existing = true)

run!(simulation)

# ## Visualization
#
# An animation of ``θ'`` and ``w`` reveals gravity waves propagating laterally
# and leaving the domain through the open boundaries without visible
# reflection.

θ′ts = FieldTimeSeries(filename, "θ′")
wts = FieldTimeSeries(filename, "w")

times = θ′ts.times
Nt = length(times)

fig = Figure(size = (800, 700), fontsize = 12)

axθ = Axis(fig[1, 1]; aspect = 2, ylabel = "z (m)")
axw = Axis(fig[2, 1]; aspect = 2, xlabel = "x (m)", ylabel = "z (m)")

hidexdecorations!(axθ, grid = false)

n = Observable(Nt)

θ′n = @lift θ′ts[$n]
wn = @lift wts[$n]

θ′lim = maximum(abs, θ′ts) / 2
wlim = maximum(abs, wts) / 2

hmθ = heatmap!(axθ, θ′n; colormap = :balance, colorrange = (-θ′lim, θ′lim))
hmw = heatmap!(axw, wn; colormap = :balance, colorrange = (-wlim, wlim))

Colorbar(fig[1, 2], hmθ; label = "θ′ (K)")
Colorbar(fig[2, 2], hmw; label = "w (m/s)")

title = @lift "Open-boundary thermal bubble — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title; fontsize = 16, tellwidth = false)

CairoMakie.record(fig, "open_boundary_thermal_bubble.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](open_boundary_thermal_bubble.mp4)