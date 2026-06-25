##### Compute the full model tendencies via compute_tendencies!, compiled via Reactant.
##### Run with: julia --project=test benchmark_model_tendency.jl

using Breeze
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_tendencies!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(); size=(64, 64, 32), extent=(1e3, 1e3, 1e3),
                       halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=WENO(order=5))

# set! runs update_state! (compute_tendencies=false), so the diagnostic state
# (velocities, temperature) is fresh before we compute tendencies.
θ = CenterField(grid)
set!(θ, (x, y, z) -> 300 + 0.01 * (x + y + z))
set!(model; θ=θ, ρ=1.0)

compiled! = Reactant.@compile raise=true raise_first=true sync=true compute_tendencies!(model)
compiled!(model)

# @show extrema(Array(interior(model.timestepper.Gⁿ)))
@info model.timestepper.Gⁿ.ρ
