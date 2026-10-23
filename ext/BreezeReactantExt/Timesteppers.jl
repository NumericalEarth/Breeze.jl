using Oceananigans: ReactantState, initialize!
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Utils: IterationInterval
using Reactant: TracedRNumber, @trace
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, update_state!
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Microphysics: Microphysics, DCMIP2016KesslerMicrophysics
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

# A traced Δt is a time step: traced stepping sets `last_Δt` before the first step, and the
# eager guard cannot branch on a traced value.
@inline AtmosphereModels.invalid_time_step(::TracedRNumber) = false

# Reactant initializes through `first_time_step!`, and the eager guard branches on traced
# clock fields; mirrors OceananigansReactantExt's no-ops for the upstream steppers.
OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::CompressibleAcousticModel{<:ReactantState}, Δt, callbacks) = nothing

#####
##### Radiation schedules (see `update_radiation!`)
#####

# The clock's time and iteration are traced but its stage is not, so only the first-stage
# `update_state!` of a step carries a traced conditional. Initialization (iteration 0, between
# steps) is not traced: the traced caller prepares the initial fluxes explicitly.
const TracedClock = Clock{<:TracedRNumber}

function AtmosphereModels.radiation_update_due(schedule::IterationInterval, clock::TracedClock, model)
    clock.stage == 2 || return false
    return (clock.iteration > 0) & schedule(model)
end

function AtmosphereModels.conditionally_update_radiation!(due::TracedRNumber{Bool}, rtm, model)
    @trace track_numbers=false if due
        AtmosphereModels._update_radiation!(rtm, model)
    end
    return nothing
end

AtmosphereModels.validate_schedule(schedule::IterationInterval, ::ReactantState) = schedule

AtmosphereModels.validate_schedule(schedule, ::ReactantState) =
    throw(ArgumentError("The radiation schedule $(summary(schedule)) cannot be used on ReactantState. " *
                        "Under Reactant the clock is traced, and only a stateless IterationInterval " *
                        "can decide inside a traced time step when radiation is due. Use " *
                        "`schedule = IterationInterval(N)`, for example IterationInterval(80) for " *
                        "updates every 600 s with Δt = 7.5 s."))

# Reverse mode cannot differentiate the data-dependent number of Kessler sedimentation substeps:
# loop a static `maximum_sedimentation_substeps` (see `sedimentation_substep_limit`)
Microphysics.sedimentation_substep_limit(::ReactantState, microphysics::DCMIP2016KesslerMicrophysics) =
    Val(microphysics.maximum_sedimentation_substeps)

function AtmosphereModels.validate_microphysics_architecture(microphysics::DCMIP2016KesslerMicrophysics, ::ReactantState)
    microphysics.maximum_sedimentation_substeps ≥ 1 && return nothing
    throw(ArgumentError("DCMIP2016KesslerMicrophysics on ReactantState needs `maximum_sedimentation_substeps ≥ 1`: " *
                        "Reactant cannot differentiate a data-dependent number of sedimentation substeps, so " *
                        "it loops a static number, and a step that needs more is rejected (its update is NaN). " *
                        "Choose it from eager control runs of the same case, where " *
                        "`model.microphysical_fields.sedimentation_substeps` records the most substeps each " *
                        "column needed."))
end
