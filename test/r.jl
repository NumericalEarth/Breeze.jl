using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState, GPU
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA
using Test

function loss(model, θ_init, Δt, nsteps)
    set!(model; θ=θ_init, ρ=1)
    @trace track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature).^2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_value
end

topo = (Bounded, Bounded, Flat)
sz = (6, 6)
ext = (1, 1)
@time "grid" grid = RectilinearGrid(ReactantState(); size=sz, extent=ext, topology=topo)
@time "model" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)
θ_init = CenterField(grid)
set!(θ_init, (x, y) -> 300 + 0.01 * x + 0.01 * y)
dθ_init = CenterField(grid)
set!(dθ_init, 0)
Δt = 0.0001
nsteps = 6000
grad_loss_compiled = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

@time "grad, loss_val" grad, loss_val = grad_loss_compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)
println("loss_val: ", loss_val)
println("grad: ", Array(grad))
