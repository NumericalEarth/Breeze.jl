using Oceananigans: AbstractModel, ReactantState, initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, update_state!, time_step!
using Breeze.TimeSteppers: SSPRungeKutta3

# Reactant handles initialization via first_time_step!, so this is a no-op.
Breeze.TimeSteppers.maybe_initialize_state!(::AbstractModel{<:Any, <:ReactantState}, callbacks) = nothing

function OceananigansTimeSteppers.first_time_step!(model::AbstractModel{<:SSPRungeKutta3, <:ReactantState}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end
