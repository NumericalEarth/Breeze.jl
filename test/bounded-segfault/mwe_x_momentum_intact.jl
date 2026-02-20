using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.Utils: launch!
using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_momentum_tendencies!
using Breeze.AtmosphereModels: dynamics_density, compute_x_momentum_tendency!
using Breeze.AtmosphereModels: update_state!
using Breeze.AtmosphereModels: compute_tendencies!
using Reactant
using Statistics: mean
using CUDA
using Pkg

Reactant.set_default_backend("cpu")
arch = ReactantState()
grid = RectilinearGrid(arch;
    size = (10, 10),
    extent = (1e3, 1e3),
    halo = (3, 3),
    topology = (Bounded, Bounded, Flat))
model = AtmosphereModel(grid; dynamics = CompressibleDynamics())

θ = CenterField(grid)
set!(θ, (x, y) -> 300 + 0.01x)

nsteps = 2000

# function loss(model, θ, nsteps)
#     set!(model, θ = θ, ρ = 1.0)

#     @trace track_numbers = false for _ in 1:nsteps
#         for _ in 1:3
#             update_state!(model, compute_tendencies=false)
#             model_fields = Oceananigans.fields(model)

#             Gρu = model.timestepper.Gⁿ.ρu
        
#             momentum_args = (
#                 dynamics_density(model.dynamics),
#                 model.advection.momentum,
#                 model.velocities,
#                 model.closure,
#                 model.closure_fields,
#                 model.momentum,
#                 model.coriolis,
#                 model.clock,
#                 model_fields)
        
#             u_args = tuple(momentum_args..., model.forcing.ρu, model.dynamics)
#             launch!(arch, grid, :xyz, compute_x_momentum_tendency!, Gρu, grid, u_args)
#         end
#     end

#     return mean(interior(model.temperature) .^ 2)
# end

function loss(model, θ, nsteps)
    set!(model, θ = 300, ρ = 1.0)
    @trace track_numbers = false for _ in 1:nsteps
        time_step!(model, 0.1)
    end
    return mean(interior(model.temperature) .^ 2)
end

Reactant.@compile raise_first = true raise = true sync = true loss(
    model, θ, nsteps)

result = compiled(model, θ, nsteps)

println("\nResult: $result")
