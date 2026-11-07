using Oceananigans: ReactantState
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Fields: Field, compute!
using Breeze.AtmosphereModels: AtmosphereModel, AbstractSubgridCorrelation, NoSubgridCorrelation
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties

const ReactantModel = AtmosphereModel{<:Any, <:Any, <:ReactantState}

# Reactant cannot trace a lazy operation such as a `KernelFunctionOperation` as a kernel
# argument on every grid (it fails converting the operation over a traced column-ensemble
# grid), so the conserved variables handed to P3's subgrid statistics, qᵗ and (when lazy) θˡⁱ,
# are materialized into Fields first. The values are the same; the generic method keeps them
# lazy so that it does not allocate on every tendency evaluation.
materialize_operand(operand::AbstractOperation) = compute!(Field(operand))
materialize_operand(operand) = operand

PredictedParticleProperties.p3_subgrid_state(::NoSubgridCorrelation, ::ReactantModel) = nothing

function PredictedParticleProperties.p3_subgrid_state(subgrid::AbstractSubgridCorrelation, model::ReactantModel)
    state = invoke(PredictedParticleProperties.p3_subgrid_state, Tuple{Any, Any}, subgrid, model)
    return map(materialize_operand, state)
end
