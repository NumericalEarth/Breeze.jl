using Oceananigans: ReactantState, initialize!
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, update_state!
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.TimeSteppers: SSPRungeKutta3, AcousticRungeKutta3
using Breeze.CompressibleEquations: CompressibleDynamics

function OceananigansTimeSteppers.first_time_step!(model::AtmosphereModel{<:Any, <:Any, <:ReactantState, <:SSPRungeKutta3}, Δt)
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

# Reactant tracing breaks if Δt is materialized via `convert` outside the kernel;
# pass it through unchanged and let in-kernel arithmetic see the traced value.
@inline AtmosphereModels.kernel_time_step(::ReactantState, grid, Δt) = Δt

# Mirrors OceananigansReactantExt: Reactant handles initialization through
# `first_time_step!`, and the eager guard branches on clock fields, which are traced
# values a boolean context cannot accept. Constrains dynamics AND architecture so this
# method is strictly more specific than the eager companion in acoustic_runge_kutta_3.jl.
OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::AtmosphereModel{<:CompressibleDynamics, <:Any, <:ReactantState, <:AcousticRungeKutta3},
                                                        Δt, callbacks) = nothing
