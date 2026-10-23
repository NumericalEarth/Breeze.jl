using Oceananigans: Oceananigans, prognostic_fields, prognostic_state, restore_prognostic_state!

function Oceananigans.prognostic_state(model::AtmosphereModel)
    return (clock           = prognostic_state(model.clock),
            particles       = prognostic_state(model.particles),
            fields          = prognostic_state(prognostic_fields(model)),
            closure_fields  = prognostic_state(model.closure_fields),
            timestepper     = prognostic_state(model.timestepper),
            reference_frame = prognostic_state(reference_frame(model.dynamics)))
end

function Oceananigans.restore_prognostic_state!(restored::AtmosphereModel, from)
    restore_prognostic_state!(restored.clock, from.clock)
    restore_prognostic_state!(restored.particles, from.particles)
    restore_prognostic_state!(prognostic_fields(restored), from.fields)
    restore_prognostic_state!(restored.closure_fields, from.closure_fields)
    restore_prognostic_state!(restored.timestepper, from.timestepper)
    restore_prognostic_state!(reference_frame(restored.dynamics), get(from, :reference_frame, nothing))
    return restored
end

Oceananigans.restore_prognostic_state!(::AtmosphereModel, ::Nothing) = nothing
