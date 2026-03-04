#####
##### update_boundary_condition! dispatch for bulk BC types
#####
##### These methods extend Oceananigans.BoundaryConditions.update_boundary_condition!
##### to update filtered surface state for bulk flux boundary conditions.
##### The deduplication tracker (last_update Ref) prevents double-updating
##### when the same FilteredSurfaceVelocities is shared across multiple BCs.
#####

function Oceananigans.BoundaryConditions.update_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkDragFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    update_filtered_surface_state!(fv, model)
    return nothing
end

function Oceananigans.BoundaryConditions.update_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    update_filtered_surface_state!(fv, model)
    fs = bc.condition.filtered_scalar
    source = sensible_heat_source_field(bc.condition.formulation, model)
    update_filtered_surface_state!(fs, source, model)
    return nothing
end

function Oceananigans.BoundaryConditions.update_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    update_filtered_surface_state!(fv, model)
    fs = bc.condition.filtered_scalar
    source = vapor_source_field(model)
    update_filtered_surface_state!(fs, source, model)
    return nothing
end

# Source field helpers for FilteredSurfaceScalar
sensible_heat_source_field(::PotentialTemperatureFlux, model) = model.formulation.potential_temperature
sensible_heat_source_field(::StaticEnergyFlux, model) = model.formulation.specific_energy
sensible_heat_source_field(::Nothing, model) = model.formulation.potential_temperature

vapor_source_field(model) = AtmosphereModels.specific_prognostic_moisture(model)
