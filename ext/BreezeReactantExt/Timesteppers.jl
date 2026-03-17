using Oceananigans: ReactantState, initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, time_step!, update_state!
using Breeze.TimeSteppers: SSPRungeKutta3

function OceananigansTimeSteppers.first_time_step!(model::AtmosphereModel{<:Any, <:Any, <:ReactantState, <:SSPRungeKutta3}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end
