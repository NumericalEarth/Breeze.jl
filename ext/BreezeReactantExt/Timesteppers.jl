using Oceananigans: ReactantState, initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, update_state!
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.TimeSteppers: SSPRungeKutta3, CompressibleAcousticModel

function OceananigansTimeSteppers.first_time_step!(model::AtmosphereModel{<:Any, <:Any, <:ReactantState, <:SSPRungeKutta3}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

# Reactant tracing breaks if Δt is materialized via `convert` outside the kernel;
# pass it through unchanged and let in-kernel arithmetic see the traced value.
@inline AtmosphereModels.kernel_time_step(::ReactantState, grid, Δt) = Δt

# Reactant initializes through `first_time_step!`, and the eager guard branches on traced
# clock fields; mirrors OceananigansReactantExt's no-ops for the upstream steppers.
OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::CompressibleAcousticModel{<:ReactantState}, Δt, callbacks) = nothing
