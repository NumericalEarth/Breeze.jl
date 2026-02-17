"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including:
- `SSPRungeKutta3`: Standard SSP RK3 scheme for explicit time stepping
- `AcousticSSPRungeKutta3`: SSP RK3 with acoustic substepping for compressible dynamics
- `AcousticRungeKutta3`: Wicker-Skamarock RK3 with acoustic substepping for compressible dynamics
"""
module TimeSteppers

export SSPRungeKutta3, AcousticSSPRungeKutta3, AcousticRungeKutta3,
       store_initial_state!,
       ssp_rk3_substep!,
       maybe_initialize_state!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: AbstractModel
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers,
                                 update_state!

# TODO: move maybe_initialize_state! to Oceananigans
# (see https://github.com/CliMA/Oceananigans.jl/issues/5300)

"""
    $(TYPEDSIGNATURES)

Update the model state at iteration 0, in case `run!` is not used.
"""
function maybe_initialize_state!(model, callbacks)
    if model.clock.iteration == 0
        update_state!(model, callbacks; compute_tendencies=true)
    end
    return nothing
end

include("ssp_runge_kutta_3.jl")
include("acoustic_ssp_runge_kutta_3.jl")
include("acoustic_runge_kutta_3.jl")

# Extend TimeStepper to support time steppers via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticSSPRungeKutta3}, args...; kwargs...) =
    AcousticSSPRungeKutta3(args...; kwargs...)

OceananigansTimeSteppers.TimeStepper(::Val{:AcousticRungeKutta3}, args...; kwargs...) =
    AcousticRungeKutta3(args...; kwargs...)

end # module
