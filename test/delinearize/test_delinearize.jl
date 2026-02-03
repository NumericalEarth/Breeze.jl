using Reactant
using Enzyme
using Statistics: mean
using CUDA

mutable struct StateWrapper
    state
end

myset!(statewrapper::StateWrapper, num::Float64) = statewrapper.state = num

function loss(statewrapper, num)
    myset!(statewrapper, num)
    return num^2
end

function grad_loss(statewrapper, dstatewrapper, num, dnum)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(statewrapper, dstatewrapper),
        Enzyme.Duplicated(num, dnum)
    )
    return dnum, loss_value
end

statewrapper = StateWrapper(0.0)
num = 1.0
dstatewrapper = Enzyme.make_zero(statewrapper)
dnum = Enzyme.make_zero(num)

Reactant.@compile grad_loss(statewrapper, dstatewrapper, num, dnum)