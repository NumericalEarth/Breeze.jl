#####
##### update_boundary_condition! dispatch for bulk BC types
#####
##### These methods extend Oceananigans.BoundaryConditions.update_boundary_condition!
##### to update filtered surface state for bulk flux boundary conditions.
##### The deduplication tracker (last_update Ref) prevents double-updating
##### when the same FilteredSurfaceVelocities is shared across multiple BCs.
#####

using Oceananigans: boundary_conditions

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

#####
##### initialize_boundary_conditions! — called from initialize!(model)
#####

initialize_boundary_condition!(bc, side, field, model) = nothing

function initialize_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkDragFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    initialize_filtered_surface_state!(fv, model)
    return nothing
end

function initialize_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    initialize_filtered_surface_state!(fv, model)
    fs = bc.condition.filtered_scalar
    source = sensible_heat_source_field(bc.condition.formulation, model)
    initialize_filtered_surface_state!(fs, source, model)
    return nothing
end

function initialize_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    initialize_filtered_surface_state!(fv, model)
    fs = bc.condition.filtered_scalar
    source = vapor_source_field(model)
    initialize_filtered_surface_state!(fs, source, model)
    return nothing
end

function initialize_boundary_conditions!(bcs::FieldBoundaryConditions, field, model)
    initialize_boundary_condition!(bcs.west, Val(:west), field, model)
    initialize_boundary_condition!(bcs.east, Val(:east), field, model)
    initialize_boundary_condition!(bcs.south, Val(:south), field, model)
    initialize_boundary_condition!(bcs.north, Val(:north), field, model)
    initialize_boundary_condition!(bcs.bottom, Val(:bottom), field, model)
    initialize_boundary_condition!(bcs.top, Val(:top), field, model)
    initialize_boundary_condition!(bcs.immersed, Val(:immersed), field, model)
    return nothing
end

initialize_boundary_conditions!(bcs, field, model) = nothing

initialize_boundary_conditions!(fields::NamedTuple, model) =
    initialize_boundary_conditions!(values(fields), model)

function initialize_boundary_conditions!(fields::Tuple, model)
    for field in fields
        bcs = boundary_conditions(field)
        initialize_boundary_conditions!(bcs, field, model)
    end
    return nothing
end

#####
##### Oceananigans.initialize! extension for AtmosphereModel
#####

function Oceananigans.initialize!(model::AtmosphereModels.AtmosphereModel)
    initialize_boundary_conditions!(AtmosphereModels.prognostic_fields(model), model)
    return nothing
end
