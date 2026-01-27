#####
##### MWE: Enzyme cannot compute adjoints for kernel operations
#####
#
# Error: "could not compute the adjoint for this operation"
#        %N = "enzymexla.kernel_call"(...)
#

using Reactant, Enzyme, Oceananigans, Breeze
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior
using Statistics: mean

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1, 1),
                       topology=(Periodic, Periodic, Flat))

model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
dmodel = Enzyme.make_zero(model)

function loss(model, Δt)
    time_step!(model, Δt)
    return mean(interior(model.temperature).^2)
end

function grad_loss(model, dmodel, Δt)
    Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active,
                    Enzyme.Duplicated(model, dmodel), Enzyme.Const(Δt))
    return nothing
end

@compile grad_loss(model, dmodel, 0.01)
