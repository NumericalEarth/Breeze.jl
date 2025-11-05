using Breeze
using Oceananigans.Units

Nx, Nz = 256, 128
H, L = 2kilometers, 1000kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = (0, H),
                                  topology = (Periodic, Flat, Bounded))

h₀ = 250meters
a = 5kilometers
λ = 4kilometers
hill(x) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

model = AtmosphereModel(grid, advection = WENO())

# Initial conditions
θ₀ = 288
g = 9.81
N² = 1e-6           # Brunt-Väisälä frequency squared (s⁻²)
dθdz = N² * θ₀ / g  # Background potential temperature gradient
θᵢ(x, z) = θ₀ + dθdz * z # background stratification
Uᵢ = 1
set!(model, θ=θᵢ, u=Uᵢ)

Δt = 5minutes
stop_time = 4days
simulation = Simulation(model; Δt, stop_time)

# We add a callback to print a message about how the simulation is going,

using Printf

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(200))

filename = "mountain_waves"
simulation.output_writers[:fields] = JLD2Writer(model, model.velocities; filename,
                                                schedule = TimeInterval(save_fields_interval),
                                                overwrite_existing = true)

run!(simulation)

using GLMakie

heatmap(model.velocities.w)

