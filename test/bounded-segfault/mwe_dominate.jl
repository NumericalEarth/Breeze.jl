# MWE: compute_velocities! × 6 steps → "operand #0 does not dominate this use"
#
# Passes at nsteps=4, fails at nsteps=6.

using Oceananigans, Breeze, Reactant, Enzyme
using Oceananigans.Architectures: ReactantState
using Breeze.AtmosphereModels: compute_velocities!
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(4,4), extent=(1e3,1e3), halo=(3,3),
    topology=(Bounded, Bounded, Flat))

model  = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)

function loss(model, n)
    @trace track_numbers=false for _ in 1:n
        compute_velocities!(model)
    end
    return mean(interior(model.velocities.u) .^ 2)
end

function grad(model, dmodel, n)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Const(n))
end

nsteps = 100
compiled_loss = Reactant.@compile raise_first=true raise=true sync=true loss(model, nsteps)
compiled_grad = Reactant.@compile raise_first=true raise=true sync=true grad(model, dmodel, nsteps)
compiled_loss(model, nsteps)
compiled_grad(model, dmodel, nsteps)
