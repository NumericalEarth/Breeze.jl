"""
TimeSteppers module for Breeze.jl

Provides time stepping schemes for AtmosphereModel, including the SSP RK3 scheme
which is TVD (total variation diminishing) with CFL coefficient = 1.
"""
module TimeSteppers

export SSPRungeKutta3

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers

include("ssp_runge_kutta_3.jl")

# Extend TimeStepper to support SSPRK3 via Symbol
OceananigansTimeSteppers.TimeStepper(::Val{:SSPRungeKutta3}, args...; kwargs...) =
    SSPRungeKutta3(args...; kwargs...)

#####
##### `AtmosphereModel` extensions for the Oceananigans TimeSteppers
#####

OceananigansTimeSteppers.ab2_step!(model::AtmosphereModel, Δt, callbacks) = NonhydrostaticModel.pressure_correction_ab2_step!(model, Δt, callbacks)
OceananigansTimeSteppers.rk3_substep!(model::AtmosphereModel, Δt, γⁿ, ζⁿ, callbacks) = NonhydrostaticModel.pressure_correction_rk3_substep!(model, Δt, γⁿ, ζⁿ, callbacks)

function OceananigansTimeSteppers.cache_previous_tendencies!(model::AtmosphereModel)
    model_fields = prognostic_fields(model)

    for field_name in keys(model_fields)
        parent(model.timestepper.G⁻[field_name]) .= parent(model.timestepper.Gⁿ[field_name])
    end

    return nothing
end

end # module
