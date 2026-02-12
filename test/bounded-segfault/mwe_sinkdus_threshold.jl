# MWE: SinkDUS segfault — 2x update_state! passes, 3x crashes.
# Reactant v0.2.217, Enzyme v0.13.118, Breeze @ main, Julia 1.11.8
# Topology must be Bounded. Periodic does not crash.

using Oceananigans, Breeze, Reactant, Enzyme
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Statistics: mean
using CUDA

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState();
    size=(4,4), extent=(1e3,1e3), halo=(3,3),
    topology=(Bounded, Bounded, Flat))

model  = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)
θ  = CenterField(grid); set!(θ, (x,y) -> 300 + 0.01x)
dθ = CenterField(grid); set!(dθ, 0)

function loss(model, θ, n)
    set!(model, θ=θ, ρ=1.0)
    @trace track_numbers=false for _ in 1:n
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
        # ↓ uncomment for segfault ↓
        update_state!(model)
        parent(model.momentum.ρu) .= parent(model.momentum.ρu) .* 0.99
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad(model, dmodel, θ, dθ, n)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(θ, dθ),
        Enzyme.Const(n))
end

c = Reactant.@compile raise_first=true raise=true sync=true grad(model, dmodel, θ, dθ, 2)
c(model, dmodel, θ, dθ, 2)
