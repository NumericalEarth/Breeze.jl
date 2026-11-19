using Oceananigans: Oceananigans, prognostic_fields, prognostic_state, restore_prognostic_state!

function Oceananigans.prognostic_state(model::AtmosphereModel)
    return (clock          = prognostic_state(model.clock),
            particles      = prognostic_state(model.particles),
            fields         = prognostic_state(prognostic_fields(model)),
            closure_fields = prognostic_state(model.closure_fields),
            radiation      = prognostic_state(model.radiation),
            timestepper    = timestepper_prognostic_state(model.timestepper, model))
end

# What a restart needs of the time stepper. A time stepper whose restart state depends on the
# model it steps extends this.
timestepper_prognostic_state(timestepper, model) = prognostic_state(timestepper)

function Oceananigans.restore_prognostic_state!(restored::AtmosphereModel, from)
    restore_prognostic_state!(restored.clock, from.clock)
    restore_prognostic_state!(restored.particles, from.particles)
    restore_prognostic_state!(prognostic_fields(restored), from.fields)
    restore_prognostic_state!(restored.closure_fields, from.closure_fields)
    # Older checkpoints can lack radiation
    restore_prognostic_state!(restored.radiation, get(from, :radiation, nothing))
    restore_prognostic_state!(restored.timestepper, from.timestepper)
    return restored
end

Oceananigans.restore_prognostic_state!(::AtmosphereModel, ::Nothing) = nothing

# What the next solve does not rebuild: the fluxes, their updates, the schedule and the validity
radiation_restart_state(rtm) = (; rtm.schedule,
                                  rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
                                  rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
                                  rtm.flux_divergence, rtm.cache_valid,
                                  approximate_updates = approximate_updates_restart_state(rtm.approximate_updates))

approximate_updates_restart_state(::Nothing) = nothing

# The current solar cosine and the workspace are rewritten before they are read
approximate_updates_restart_state(updates) = (; updates.upwelling_longwave_response, updates.downwelling_longwave_response,
                                                updates.solved_upwelling_longwave_flux, updates.solved_downwelling_longwave_flux,
                                                updates.solved_upwelling_shortwave_flux, updates.solved_downwelling_shortwave_flux,
                                                updates.solved_cos_zenith, updates.surface_temperature)

Oceananigans.prognostic_state(rtm::RadiativeTransferModel) = prognostic_state(radiation_restart_state(rtm))

# An incomplete checkpoint invalidates the cache, so the next `update_state!` solves once the
# diagnostics are restored
function Oceananigans.restore_prognostic_state!(rtm::RadiativeTransferModel, from)
    state = radiation_restart_state(rtm)
    valid = radiation_cache_is_valid(get(from, :cache_valid, true))
    for (name, destination) in pairs(state)
        name == :cache_valid && continue
        saved = get(from, name, nothing)
        if name == :approximate_updates
            isnothing(destination) && continue
            complete = !isnothing(saved) && all(key -> haskey(saved, key), keys(destination))
            valid &= complete
            if complete
                for key in keys(destination)
                    restore_prognostic_state!(destination[key], saved[key])
                end
            end
        elseif haskey(from, name)
            restore_prognostic_state!(destination, saved)
        else
            valid = false
        end
    end
    set_radiation_cache_validity!(rtm.cache_valid, valid)
    return rtm
end

function Oceananigans.restore_prognostic_state!(rtm::RadiativeTransferModel, ::Nothing)
    set_radiation_cache_validity!(rtm.cache_valid, false)
    return rtm
end
