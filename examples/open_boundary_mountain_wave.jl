# # Mountain wave with open boundaries
#
# This example demonstrates flow over a mountain with open lateral boundaries
# using [`PerturbationAdvection`](@ref Oceananigans.BoundaryConditions.PerturbationAdvection)
# boundary conditions. A uniform wind of 10 m/s
# flows over a witch-of-Agnesi ridge in a stably stratified atmosphere,
# generating vertically propagating gravity waves (lee waves).
#
# The open boundaries radiate disturbances out of the domain,
# avoiding the artificial periodicity of a doubly-periodic setup.

using Breeze
using Oceananigans.Grids: ImmersedBoundaryGrid, PartialCellBottom
using Oceananigans.Units
using Printf
using CairoMakie

# ## Grid and topography
#
# We set up a 2D (x–z) domain with `Bounded` lateral topology so that
# open boundary conditions can be imposed on the west and east faces.

Nx, Nz = 256, 128
H = 20kilometers
L = 100kilometers

underlying_grid = RectilinearGrid(CPU();
                                  size = (Nx, Nz),
                                  halo = (5, 5),
                                  x = (-L/2, L/2),
                                  z = (0, H),
                                  topology = (Bounded, Flat, Bounded))

# The witch-of-Agnesi profile is a classic test case for mountain-wave codes:
#
# ``h(x) = h_0 \, a^2 / (x^2 + a^2)``

h₀ = 100meters
a = 5kilometers
witch_of_agnesi(x) = h₀ * a^2 / (x^2 + a^2)
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(witch_of_agnesi))

# ## Open boundary conditions
#
# We use [`PerturbationAdvection`](@ref Oceananigans.BoundaryConditions.PerturbationAdvection)
# open boundaries on the x-momentum `ρu`.
# The scheme radiates perturbations through the boundary while relaxing
# to the prescribed mean momentum on inflow faces.
#
# Because the prognostic variable is momentum density `ρu` (not velocity `u`),
# the exterior value must be set to `ρᵣ U` using the reference density.
# Setting `outflow_timescale = 1` adds relaxation at outflow boundaries,
# which prevents wave energy accumulation at the boundary.

U = 10 # m/s mean flow

## Build a temporary model to access the reference density
tmp_model = AtmosphereModel(grid; advection = WENO())
ρᵣ = tmp_model.dynamics.reference_state.density
ρu_mean = Field{Face, Nothing, Center}(grid)
set!(ρu_mean, ρᵣ * U)

scheme = PerturbationAdvection(; outflow_timescale = 1)

ρu_bcs = FieldBoundaryConditions(
    west = OpenBoundaryCondition(ρu_mean; scheme),
    east = OpenBoundaryCondition(ρu_mean; scheme))

# ## Model construction
#
# The [`AtmosphereModel`](@ref) combines anelastic dynamics with the open boundaries.

model = AtmosphereModel(grid;
                         advection = WENO(),
                         boundary_conditions = (; ρu = ρu_bcs))

# ## Initial conditions
#
# A uniform horizontal wind with stable stratification:
# ``\theta(z) = \theta_0 \exp(N^2 z / g)`` where ``N^2 = 10^{-4} \, \mathrm{s}^{-2}``.

N² = 1e-4
θ₀ = model.dynamics.reference_state.potential_temperature
g = model.thermodynamic_constants.gravitational_acceleration

θᵢ(x, z) = θ₀ * exp(N² * z / g)
set!(model; u = U, θ = θᵢ)

# ## Run the simulation

simulation = Simulation(model; Δt = 1, stop_time = 2hours)
conjure_time_step_wizard!(simulation, cfl = 0.5)

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))

wall_clock = Ref(time_ns())

function progress(sim)
    compute!(δ)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, Δt: %s, max|w|: %.3e m/s, max|δ|: %.3e s⁻¹, wall: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w), maximum(abs, δ),
                   prettytime(elapsed))

    wall_clock[] = time_ns()
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

# ## Output

filename = "open_boundary_mountain_wave.jld2"
outputs = merge(model.velocities, (; δ))

simulation.output_writers[:fields] = JLD2Writer(model, outputs; filename,
                                                schedule = TimeInterval(2minutes),
                                                overwrite_existing = true)

run!(simulation)

# ## Visualization
#
# Lee waves appear as tilted phase lines in vertical velocity,
# propagating energy upward and downstream of the mountain.

@info "Creating visualization..."

wt = FieldTimeSeries(filename, "w")
times = wt.times
Nt = length(wt)

fig = Figure(size = (900, 500))

n = Observable(Nt)
wn = @lift wt[$n]

title = @lift "Mountain wave — t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

ax = Axis(fig[1, 1]; aspect = 4,
          xlabel = "x (km)", ylabel = "z (km)",
          xtickformat = x -> string.(x ./ 1e3),
          ytickformat = z -> string.(z ./ 1e3))

w_lim = maximum(abs, wt)
hm = heatmap!(ax, wn; colorrange = (-w_lim, w_lim), colormap = :balance)
Colorbar(fig[1, 2], hm, label = "w (m/s)")

CairoMakie.record(fig, "open_boundary_mountain_wave.mp4", 1:Nt, framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](open_boundary_mountain_wave.mp4)
