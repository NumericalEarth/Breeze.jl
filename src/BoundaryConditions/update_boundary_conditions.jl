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
    update_filtered_Δθᵥ!(fv, bc.condition.coefficient, bc.condition.surface_temperature, model)
    return nothing
end

function Oceananigans.BoundaryConditions.update_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    update_filtered_surface_state!(fv, model)
    update_filtered_Δθᵥ!(fv, bc.condition.coefficient, bc.condition.surface_temperature, model)
    fs = bc.condition.filtered_scalar
    source = sensible_heat_source_field(bc.condition, model)
    update_filtered_surface_state!(fs, source, model)
    return nothing
end

function Oceananigans.BoundaryConditions.update_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    update_filtered_surface_state!(fv, model)
    update_filtered_Δθᵥ!(fv, bc.condition.coefficient, bc.condition.surface_temperature, model)
    fs = bc.condition.filtered_scalar
    source = vapor_source_field(bc.condition, model)
    update_filtered_surface_state!(fs, source, model)
    return nothing
end

# Source field helpers for FilteredSurfaceScalar
function sensible_heat_source_field(bf, model)
    return KernelFunctionOperation{Center, Center, Center}(sensible_heat_difference, model.grid,
                                                         bf, model.clock, surface_layer_state(model))
end

@inline function sensible_heat_difference(i, j, k, grid, bf, clock, fields)
    Tˢ = wall_value(i, j, grid, Bottom(), bf.surface_temperature, clock)
    pˢ = wall_air_pressure(i, j, 1, grid, Bottom(), nothing, fields, bf.thermodynamic_constants)
    return bulk_sensible_heat_difference(i, j, k, grid, Bottom(), bf.formulation, bf, Tˢ, fields, pˢ, nothing)
end

function vapor_source_field(bf, model)
    return KernelFunctionOperation{Center, Center, Center}(vapor_difference, model.grid,
                                                         bf, model.clock, surface_layer_state(model))
end

@inline function vapor_difference(i, j, k, grid, bf, clock, fields)
    Tˢ = wall_value(i, j, grid, Bottom(), bf.surface_temperature, clock)
    pˢ = wall_air_pressure(i, j, 1, grid, Bottom(), nothing, fields, bf.thermodynamic_constants)
    qᵛˢ = wall_specific_humidity(i, j, grid, Bottom(), bf, Tˢ, clock, pˢ)
    return bulk_vapor_difference(i, j, k, fields, nothing, qᵛˢ)
end

# Δθᵥ filter — dedup-aware variants. Only a stability-corrected `PolynomialCoefficient`
# consumes the filtered surface-layer difference, so the update is a no-op for a constant
# coefficient, for a coefficient without a stability function, and when there is no
# `FilteredSurfaceVelocities` at all.
const StabilityCorrectedCoefficient = PolynomialCoefficient{<:Any, <:Any, <:FittedStabilityFunction}

initialize_filtered_Δθᵥ!(::Nothing, coef, Tˢ, model) = nothing
initialize_filtered_Δθᵥ!(fv::FilteredSurfaceVelocities, coef, Tˢ, model) = nothing

function initialize_filtered_Δθᵥ!(fv::FilteredSurfaceVelocities, coef::StabilityCorrectedCoefficient, Tˢ, model)
    initialize_Δθᵥ!(fv, coef, Tˢ, model.grid, model.clock, surface_layer_state(model))
    return nothing
end

update_filtered_Δθᵥ!(::Nothing, coef, Tˢ, model) = nothing
update_filtered_Δθᵥ!(fv::FilteredSurfaceVelocities, coef, Tˢ, model) = nothing

function update_filtered_Δθᵥ!(fv::FilteredSurfaceVelocities, coef::StabilityCorrectedCoefficient, Tˢ, model)
    key = (model.clock.iteration, model.clock.stage)
    fv.last_Δθᵥ_update[] == key && return nothing
    Δt = model.clock.last_Δt
    isinf(Δt) && return nothing # no valid Δt yet (before first time step)
    update_Δθᵥ!(fv, coef, Tˢ, model.grid, model.clock, Δt, surface_layer_state(model))
    fv.last_Δθᵥ_update[] = key
    return nothing
end

#####
##### initialize_boundary_conditions! — called from initialize!(model)
#####

initialize_boundary_condition!(bc, side, field, model) = nothing

function initialize_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkDragFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    initialize_filtered_surface_state!(fv, model)
    initialize_filtered_Δθᵥ!(fv, bc.condition.coefficient, bc.condition.surface_temperature, model)
    return nothing
end

function initialize_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}, side, field, model)
    validate_wall_density(bc.condition.moisture, model)
    fv = bc.condition.filtered_velocities
    initialize_filtered_surface_state!(fv, model)
    initialize_filtered_Δθᵥ!(fv, bc.condition.coefficient, bc.condition.surface_temperature, model)
    fs = bc.condition.filtered_scalar
    source = sensible_heat_source_field(bc.condition, model)
    initialize_filtered_surface_state!(fs, source, model)
    return nothing
end

function initialize_boundary_condition!(
        bc::BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}, side, field, model)
    fv = bc.condition.filtered_velocities
    initialize_filtered_surface_state!(fv, model)
    initialize_filtered_Δθᵥ!(fv, bc.condition.coefficient, bc.condition.surface_temperature, model)
    fs = bc.condition.filtered_scalar
    source = vapor_source_field(bc.condition, model)
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
