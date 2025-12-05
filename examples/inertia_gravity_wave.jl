using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie


# inertia Gravity Wave
# Reference:
# Skamarock and Klemp (1994): "Efficiency and Accuracy of the Klemp-Wilhelmson Time-Splitting Technique"

# Problem parameters: isothermal base state and mean wind
p₀ = 100000                 # Pa
θ₀ = 300                    # K - reference potential temperature
U  = 20                     # m s^-1 (mean wind)
N  = 0.01
N² = N^2                    # Brunt–Väisälä frequency squared


#  grid configuration
Nx, Nz = 300, 10
Lx, Lz = 300kilometers, 10kilometers

grid = RectilinearGrid(CPU(),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))


                       
# Atmosphere model setup
constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state)

model = AtmosphereModel(grid; formulation, advection = Centered(order=2))


# Initial conditions and initialization
Δθ₀ = 0.01                  # K - perturbation amplitude
a = 5000                    # m   (perturbation half-width parameter)
x_c = Lx / 3                # m   (perturbation center in x)

# Background potential temperature profile (isothermal)
g = model.thermodynamic_constants.gravitational_acceleration
θ̄ᵦ(z) = θ₀ * exp(N² * z / g)
# Save initial potential temperature without perturbation to compute anomaly later
θᵢ₀ = Field{Center, Nothing, Center}(grid)
set!(θᵢ₀, (x, z) -> θ̄ᵦ(z))

# Perturbation
function θᵢ(x, z)
    θ′ = Δθ₀ * sin(π * z / Lz)  / (1 + (x - x_c)^2/a^2)
    return θ̄ᵦ(z) + θ′
end

set!(model, θ = θᵢ, u = U)

Δt = 6 # seconds
stop_time = 3000
simulation = Simulation(model; Δt, stop_time)


function progress(sim)
    ρe = sim.model.formulation.thermodynamics.energy_density
    u, v, w = sim.model.velocities

    ρemean = mean(ρe)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, mean(ρe): %.6e J/kg, max|u|: %.5f m/s, max w: %.5f m/s, min w: %.5f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), ρemean, maximum(abs, u), maximum(w), minimum(w))

    @info msg
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1minute))

# Output setup
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
gb = fig[1, 1]

xs = LinRange(0, Lx, Nx)
zs = LinRange(0, Lz, Nz+1)
pdata = Array(interior(θ, :,1,:)) - Array(interior(θᵢ₀, :,1,:))
ax, hm = heatmap(gb[1,1], xs, zs, pdata, colormap = :balance, colorrange = (-0.01, 0.01))
ax.xlabel = "x [m]"
ax.ylabel = "z [m]"
ax.title = "θ Anomaly at t = $(stop_time)s"

Colorbar(gb[1:1, 2], hm; label = "θ Anomaly [K]")

save("inertia_gravity_wave.png", fig)

