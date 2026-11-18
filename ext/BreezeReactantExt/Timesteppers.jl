using Oceananigans: Oceananigans, ReactantState, initialize!
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Utils: IterationInterval
using Reactant: TracedRNumber, @trace
using Oceananigans.TimeSteppers: TimeSteppers as OceananigansTimeSteppers, update_state!, reconcile_state!
using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.Microphysics: Microphysics, DCMIP2016KesslerMicrophysics
using Breeze.TimeSteppers: TimeSteppers, SSPRungeKutta3, CompressibleAcousticModel,
                           stage_increment, cache_transport_velocity!
using Breeze.CompressibleEquations: seed_time_averaged_velocities!

const ReactantSSPRungeKutta3Model = AtmosphereModel{<:Any, <:Any, <:ReactantState, <:SSPRungeKutta3}

# The eager preparation of `maybe_prepare_first_time_step!`, which Reactant skips because its guard
# branches on the traced iteration. The traced `update_state!` cannot tell that it runs at
# initialization either (see `radiation_update_due`), so the initial radiative fluxes are computed
# here explicitly, and the tendencies refreshed with them.
function TimeSteppers.prepare_first_time_step!(model::ReactantSSPRungeKutta3Model, Δt)
    model.clock.last_Δt = Δt
    reconcile_state!(model)
    update_state!(model; compute_tendencies = isnothing(model.radiation))
    initialize_radiation!(model.radiation, model)
    return nothing
end

initialize_radiation!(::Nothing, model) = nothing

function initialize_radiation!(radiation, model)
    AtmosphereModels._update_radiation!(radiation, model)
    update_state!(model)
    return nothing
end

function OceananigansTimeSteppers.first_time_step!(model::ReactantSSPRungeKutta3Model, Δt)
    initialize!(model)
    TimeSteppers.prepare_first_time_step!(model, Δt)
    time_step!(model, Δt)
    return nothing
end

# Reactant tracing breaks if Δt is materialized via `convert` outside the kernel;
# pass it through unchanged and let in-kernel arithmetic see the traced value.
@inline AtmosphereModels.kernel_time_step(::ReactantState, grid, Δt) = Δt

# A traced Δt is a time step: traced stepping sets `last_Δt` before the first step, and the
# eager guard cannot branch on a traced value.
@inline AtmosphereModels.invalid_time_step(::TracedRNumber) = false

#####
##### The acoustic stepper's first step
#####

const ReactantAcousticModel = CompressibleAcousticModel{<:ReactantState}

# Eagerly, the acoustic stepper's `maybe_prepare_first_time_step!` runs at iteration 0, before
# the first tendencies: it seeds the clock's last step sizes, reconciles the state, seeds the
# substepper's time-averaged transport velocity with the initial velocities, recomputes the
# tendencies with it and freezes the vertical component for the first stage's implicit remainder
# and sedimentation. Its guard branches on the iteration, which is traced, so under Reactant the
# work is split between the places a compiled program does reach:
#
# - `initialize!(model)`, which `initialize!(sim)` calls right before its `update_state!`, seeds
#   the transport velocity and the frozen copy, so those tendencies are built with the initial
#   velocities rather than the constructor's zeros (the frozen copy equals what the eager path
#   freezes after its own `update_state!`, which does not touch the transport velocity);
# - `maybe_prepare_first_time_step!`, at the top of every step, seeds the clock's last step sizes
#   through a traced select at iteration 0, as `adaptive_advection_timestep` and the perturbation
#   advection open boundaries read them on the first step;
# - `prepare_first_time_step!`, which a traced loop of `time_step!` calls before the loop, does
#   the whole eager sequence unconditionally, with the initial radiative fluxes as for SSP-RK3.
function Oceananigans.initialize!(model::ReactantAcousticModel)
    invoke(Oceananigans.initialize!, Tuple{AtmosphereModel}, model)
    reconcile_state!(model)
    seed_time_averaged_velocities!(model.timestepper.substepper, model)
    cache_transport_velocity!(model)
    return nothing
end

function OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::ReactantAcousticModel, Δt, callbacks)
    clock = model.clock
    first_step = (clock.iteration == 0) | !isfinite(clock.last_stage_Δt)
    clock.last_Δt = ifelse(first_step, Δt, clock.last_Δt)
    clock.last_stage_Δt = ifelse(first_step, stage_increment(model.timestepper, 3) * Δt, clock.last_stage_Δt)
    return nothing
end

function TimeSteppers.prepare_first_time_step!(model::ReactantAcousticModel, Δt)
    clock = model.clock
    clock.last_Δt = Δt
    clock.last_stage_Δt = stage_increment(model.timestepper, 3) * Δt
    reconcile_state!(model)
    seed_time_averaged_velocities!(model.timestepper.substepper, model)
    update_state!(model; compute_tendencies = isnothing(model.radiation))
    initialize_radiation!(model.radiation, model)
    cache_transport_velocity!(model)
    return nothing
end

function OceananigansTimeSteppers.first_time_step!(model::ReactantAcousticModel, Δt)
    initialize!(model)
    TimeSteppers.prepare_first_time_step!(model, Δt)
    time_step!(model, Δt)
    return nothing
end

#####
##### Radiation schedules (see `update_radiation!`)
#####

# The clock's time and iteration are traced but its stage is not, so only the first-stage
# `update_state!` of a step carries a traced conditional. Initialization (iteration 0, between
# steps) is not traced: `prepare_first_time_step!` computes the initial fluxes explicitly.
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
