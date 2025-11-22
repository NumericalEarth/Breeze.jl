using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie


# inertia Gravity Wave
# Reference:
# Skamarock and Klemp (1994): "Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique"

# Problem parameters: isothermal base state and mean wind
P₀ = 100000                 # Pa
θ₀ = 300                    # K - reference potential temperature
U  = 20                     # m s^-1 (mean wind)
N  = 0.01
N² = N^2                    # Brunt–Väisälä frequency squared

Δθ₀ = 0.01                  # K - perturbation amplitude
a = 5000                    # m   (perturbation half-width parameter)
f₀ = 1e-4                   # Coriolis parameter


#  grid configuration
Nx = 300
Nz = 10
Lx, Lz = 300kilometers, 10kilometers


grid = RectilinearGrid(CPU(),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

# Atmosphere model setup
model = AtmosphereModel(grid; coriolis = FPlane(f=f₀), advection = WENO(order=5))


# Initial conditions and initialization
g = model.thermodynamics.gravitational_acceleration
function θᵢ(x, z; x₀=mean(xnodes(grid, Center())), z₀=0.3*grid.Lz)
    θ̄ = θ₀ * exp(N² * z / g)
    θ′ = Δθ₀ * sin(π * z / Lz)  / (1 + ((x - Lx/3)/a)^2)
    return θ̄ + θ′
end

# Save initial potential temperature without perturbation for later comparison
θ̄_0(z) = θ₀ * exp(N² * z / g)
θᵢ₀ = Field{Center, Nothing, Center}(grid)
set!(θᵢ₀, (x, z) -> θ̄_0(z) )

# Boundary conditions
free_slip_bcs = FieldBoundaryConditions(
    top = FluxBoundaryCondition(nothing),
    bottom = FluxBoundaryCondition(nothing)
)

set!(model, θ = θᵢ, u=U, boundary_conditions = (u = free_slip_bcs, w = free_slip_bcs))

Δt = 1 # seconds
stop_time = 3000
simulation = Simulation(model; Δt, stop_time)


function progress(sim)
    ρe = sim.model.energy_density
    u, w = sim.model.velocities

    ρemean = mean(ρe)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, mean(ρe): %.6e J/kg, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   ρemean,
                   maximum(abs, u), maximum(abs, w))

    @info msg
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1minute))

# Output setup
u, w = model.velocities
θ = Breeze.AtmosphereModels.PotentialTemperatureField(model)

outputs = merge(model.velocities, (; θ))

filename = "inertia_gravity_wave.jld2"
writer = JLD2Writer(model, outputs; filename,
                    schedule = TimeInterval(Δt),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

# Model execution
run!(simulation)


# Plotting
fig = Figure(size=(900, 300))

xs = LinRange(0, Lx, Nx)
zs = LinRange(0, Lz, Nz+1)
pdata = Array(interior(θ, :,1,:)) - Array(interior(θᵢ₀, :,1,:))
ax, hm = heatmap(fig[1, 1][1, 1], xs, zs, pdata, colormap = :balance, colorrange = (-0.01, 0.01))
ax.xlabel = "x [m]"
ax.ylabel = "z [m]"
ax.title = "Potential Temperature Anomaly at t = $(stop_time)s"
Colorbar(fig[1, 1][1, 2], hm; label = "Potential Temperature Anomaly [K]")

save("inertia_gravity_wave.png", fig)