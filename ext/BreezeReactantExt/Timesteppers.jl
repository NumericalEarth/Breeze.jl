using Oceananigans: AbstractModel, ReactantState, initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_step!, update_state!
using Breeze.CompressibleEquations: CompressibleDynamics
using Breeze.TimeSteppers: SSPRungeKutta3

function OceananigansTimeSteppers.first_time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:ReactantState, <:SSPRungeKutta3}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end
