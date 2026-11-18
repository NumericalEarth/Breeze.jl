using Oceananigans: Oceananigans, prognostic_fields, prognostic_state, restore_prognostic_state!

function Oceananigans.prognostic_state(model::AtmosphereModel)
    return (clock          = prognostic_state(model.clock),
            particles      = prognostic_state(model.particles),
            fields         = prognostic_state(prognostic_fields(model)),
            closure_fields = prognostic_state(model.closure_fields),
            radiation      = prognostic_state(model.radiation),
            dynamics       = dynamics_restart_state(model.dynamics),
            timestepper    = timestepper_prognostic_state(model.timestepper, model))
end

# What a restart needs of the time stepper. A time stepper whose restart state depends on the
# model it steps extends this.
timestepper_prognostic_state(timestepper, model) = prognostic_state(timestepper)

# State of the dynamics that is not a prognostic field but must survive a restart, such as
# accumulated budgets. Nothing by default.
dynamics_restart_state(dynamics) = nothing
restore_dynamics_restart_state!(dynamics, from) = dynamics
restore_dynamics_restart_state!(dynamics, ::Nothing) = dynamics

function Oceananigans.restore_prognostic_state!(restored::AtmosphereModel, from)
    restore_prognostic_state!(restored.clock, from.clock)
    restore_prognostic_state!(restored.particles, from.particles)
    restore_prognostic_state!(prognostic_fields(restored), from.fields)
    restore_prognostic_state!(restored.closure_fields, from.closure_fields)
    # Older checkpoints have no `radiation` entry
    hasproperty(from, :radiation) && restore_prognostic_state!(restored.radiation, from.radiation)
    # Older checkpoints have no `dynamics` entry
    hasproperty(from, :dynamics) && restore_dynamics_restart_state!(restored.dynamics, from.dynamics)
    restore_prognostic_state!(restored.timestepper, from.timestepper)
    return restored
end

Oceananigans.restore_prognostic_state!(::AtmosphereModel, ::Nothing) = nothing

# The model reads the fluxes between solves and the schedule decides when the next solve fires.
# Everything else (solvers, workspaces, per-column state, optics tables) is rebuilt by that solve.
radiation_restart_state(rtm) = (; rtm.schedule,
                                  rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
                                  rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
                                  rtm.flux_divergence)

Oceananigans.prognostic_state(rtm::RadiativeTransferModel) = prognostic_state(radiation_restart_state(rtm))
Oceananigans.restore_prognostic_state!(rtm::RadiativeTransferModel, from) = restore_prognostic_state!(radiation_restart_state(rtm), from)
Oceananigans.restore_prognostic_state!(rtm::RadiativeTransferModel, ::Nothing) = rtm
