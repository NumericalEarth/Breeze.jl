using Oceananigans: Oceananigans, prognostic_fields, prognostic_state, restore_prognostic_state!

function Oceananigans.prognostic_state(model::AtmosphereModel)
    return (clock          = prognostic_state(model.clock),
            particles      = prognostic_state(model.particles),
            fields         = prognostic_state(prognostic_fields(model)),
            closure_fields = prognostic_state(model.closure_fields),
            radiation      = prognostic_state(model.radiation),
            timestepper    = prognostic_state(model.timestepper))
end

function Oceananigans.restore_prognostic_state!(restored::AtmosphereModel, from)
    restore_prognostic_state!(restored.clock, from.clock)
    restore_prognostic_state!(restored.particles, from.particles)
    restore_prognostic_state!(prognostic_fields(restored), from.fields)
    restore_prognostic_state!(restored.closure_fields, from.closure_fields)
    # Older checkpoints can lack radiation or its approximate-update cache.
    restore_prognostic_state!(restored.radiation, get(from, :radiation, nothing))
    restore_prognostic_state!(restored.timestepper, from.timestepper)
    return restored
end

Oceananigans.restore_prognostic_state!(::AtmosphereModel, ::Nothing) = nothing

# The model reads the fluxes between solves, the approximate updates change them from what the last
# solve left, and the schedule decides when the next solve fires. Everything else (solvers,
# workspaces, per-column state, optics tables) is rebuilt by that solve.
radiation_restart_state(rtm) = (; rtm.schedule,
                                  rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
                                  rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
                                  rtm.flux_divergence, rtm.cache_valid,
                                  approximate_updates = approximate_updates_restart_state(rtm.approximate_updates))

approximate_updates_restart_state(::Nothing) = nothing

# The current solar cosine and daylight workspace are rewritten before they are read.
approximate_updates_restart_state(updates) = (; updates.upwelling_longwave_response, updates.downwelling_longwave_response,
                                                updates.solved_upwelling_longwave_flux, updates.solved_downwelling_longwave_flux,
                                                updates.solved_upwelling_shortwave_flux, updates.solved_downwelling_shortwave_flux,
                                                updates.solved_cos_zenith, updates.surface_temperature)

Oceananigans.prognostic_state(rtm::RadiativeTransferModel) = prognostic_state(radiation_restart_state(rtm))

# A checkpoint whose radiation fluxes cannot be used invalidates the cache, which defers a fresh
# solve until `update_state!` has restored the atmosphere's diagnostics and the coupled surface is
# ready. Validity is checkpointed too, in case another restart intervenes before that solve.
function Oceananigans.restore_prognostic_state!(rtm::RadiativeTransferModel, from)
    # Restore validity separately: native checkpoints store a Bool, Reactant checkpoints an array.
    restore_prognostic_state!(merge(radiation_restart_state(rtm), (; cache_valid=nothing)), from)
    # Earlier checkpoints have no flag. Their fluxes are usable if the required cache exists.
    valid = radiation_cache_is_valid(get(from, :cache_valid, true)) &&
            (isnothing(rtm.approximate_updates) || has_reference_fluxes(get(from, :approximate_updates, nothing)))
    set_radiation_cache_validity!(rtm.cache_valid, valid)
    return rtm
end

function Oceananigans.restore_prognostic_state!(rtm::RadiativeTransferModel, ::Nothing)
    set_radiation_cache_validity!(rtm.cache_valid, false)
    return rtm
end

# Incremental-update checkpoints lack the reference fluxes, and their saved temperature belongs to
# the last update rather than the solve. Refresh them together with a new solve.
has_reference_fluxes(::Nothing) = false
has_reference_fluxes(saved_updates) = hasproperty(saved_updates, :solved_upwelling_longwave_flux)
