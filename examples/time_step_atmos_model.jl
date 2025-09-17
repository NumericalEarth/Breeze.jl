using Breeze.AtmosphereModels: AtmosphereModel
using Oceananigans
using Printf

Lx = 30e3
Ly = 30e3
Lz = 30e3

grid = RectilinearGrid(size=(64, 64, 64), x=(0, Lx), y=(0, Ly), z=(0, 25e3))

Q₀ = 1000 # heat flux in W / m²
e_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q₀))
model = AtmosphereModel(grid, advection=WENO(), boundary_conditions=(; e=e_bcs))

Lz = grid.Lz
Δθ = 5 # K
Tₛ = model.formulation.constants.reference_potential_temperature
θᵢ(x, y, z) = Tₛ + Δθ * z / Lz
qᵢ(x, y, z) = 0
Ξᵢ(x, y, z) = 1e-6 * randn()
set!(model, θ=θᵢ, q=qᵢ, u=Ξᵢ, v=Ξᵢ)

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))
compute!(δ)

simulation = Simulation(model, Δt=1e-6, stop_iteration=100)
# conjure_time_step_wizard!(simulation, cfl=0.7)

using Printf

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))

function progress(sim)
    T = sim.model.temperature
    u, v, w = sim.model.velocities
    T_max = maximum(T)
    T_min = minimum(T)
    u_max = maximum(abs, u)
    v_max = maximum(abs, v)
    w_max = maximum(abs, w)
    compute!(δ)
    δ_max = maximum(abs, δ)
    msg = @sprintf("Iter: %d, time: %s, extrema(T): (%.2f, %.2f) K, max|δ|: %.2e",
                   iteration(sim), prettytime(sim), T_min, T_max, δ_max)
    msg *= @sprintf(", max|u|: (%.2e, %.2e, %.2e) m s⁻¹", u_max, v_max, w_max)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

run!(simulation)
