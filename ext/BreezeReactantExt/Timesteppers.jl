using Oceananigans: AbstractModel, ReactantState, initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_step!, update_state!
using Breeze.TimeSteppers: SSPRungeKutta3

# TODO: move maybe_initialize_state! to Oceananigans
# (see https://github.com/CliMA/Oceananigans.jl/issues/5300)
Breeze.TimeSteppers.maybe_initialize_state!(::AbstractModel{<:Any, <:ReactantState}, callbacks) = nothing

function OceananigansTimeSteppers.first_time_step!(model::AbstractModel{<:SSPRungeKutta3, <:ReactantState}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end
