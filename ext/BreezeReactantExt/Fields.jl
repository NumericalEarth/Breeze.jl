using Reactant: Reactant
using OffsetArrays: OffsetArray
using Oceananigans.AbstractOperations: AbstractOperations, Average
using Oceananigans.Fields: Field

const ConcreteReactantAveragedField =
    Field{<:Any, <:Any, <:Any, <:Average, <:Any, <:Any, <:OffsetArray{<:Any, <:Any, <:ConcreteReactantArray}}

# TODO Oceananigans: move this method into OceananigansReactantExt. `average!` divides the
# summed field by the averaging volume with an in-place broadcast over `interior(avg)`, a view
# of a concrete Reactant array. Broadcasting over such a view falls back to element-wise
# indexing, which device buffers forbid. Forcings such as `SubsidenceForcing` compute horizontal
# averages eagerly when the model is constructed or `set!`, so compile the eager average with `@jit`.
function AbstractOperations.average!(avg::ConcreteReactantAveragedField, operand)
    Reactant.@jit AbstractOperations.average!(avg, operand)
    return avg
end
