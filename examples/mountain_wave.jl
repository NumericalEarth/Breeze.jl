using Breeze
using Oceananigans.Grids: ImmersedBoundaryGrid, PartialCellBottom
using Oceananigans.Units
using Printf

Nx, Nz = 512, 512
H, L = 20kilometers, 200kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = (0, H),
                                  topology = (Periodic, Flat, Bounded))

h₀ = 250meters
a = 5kilometers
λ = 4kilometers
hill(x) = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(hill))

model = AtmosphereModel(grid, advection = WENO())

# Initial conditions
θ₀ = model.formulation.constants.potential_temperature
g = model.thermodynamics.gravitational_acceleration
N² = 1e-4           # Brunt-Väisälä frequency squared (s⁻²)
θᵢ(x, z) = θ₀ * exp(N² * z / g)
Uᵢ = 10
set!(model, θ=θᵢ, u=Uᵢ)

Δt = 1 # seconds
stop_iteration = 1000
simulation = Simulation(model; Δt, stop_iteration)
conjure_time_step_wizard!(simulation, cfl=0.7)

wall_clock = Ref(time_ns())

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))

function progress(sim)
    compute!(δ)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹, max|δ|: %6.3e s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w), maximum(abs, δ))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(200))

filename = "mountain_waves"
outputs = merge(model.velocities, (; δ))
simulation.output_writers[:fields] = JLD2Writer(model, outputs; filename,
                                                schedule = IterationInterval(10),
                                                overwrite_existing = true)

run!(simulation)

using GLMakie

fig = Figure()
axw = Axis(fig[1, 1], title="Vertical Velocity w (m/s)")
axδ = Axis(fig[1, 2], title="Divergence δ (s⁻¹)")

heatmap!(axw, model.velocities.w)
heatmap!(axδ, δ)
fig
