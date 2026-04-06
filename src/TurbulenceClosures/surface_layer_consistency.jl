using Oceananigans.Grids: Center, znode

using ..BoundaryConditions: BulkDragFunction,
                            BulkDragBoundaryCondition,
                            wind_speed²ᶜᶜᶜ,
                            surface_value,
                            bulk_coefficient,
                            κ

#####
##### Enforce Monin-Obukhov surface layer consistency for closure fields
#####
##### After `compute_closure_fields!` fills νₑ throughout the domain,
##### overwrite the first cell center with the MOST-consistent value:
#####
#####     νₑ(z₁) = κ u★ z₁
#####
##### where u★ is the friction velocity from the bulk surface drag.
##### This ensures the closure's eddy viscosity near the surface is
##### consistent with the prescribed surface flux, so that the shear
##### stress ρ νₑ ∂u/∂z recovers the surface stress ρ u★².
#####

"""
    enforce_surface_layer_consistency!(model)

Overwrite the near-surface eddy viscosity `νₑ[:, :, 1]` with a value
consistent with Monin-Obukhov similarity theory,

    `νₑ(z₁) = κ u★ z₁`

where `κ ≈ 0.4` is the von Kármán constant, `u★` is the friction velocity
derived from the surface `BulkDrag` boundary condition, and `z₁` is the
height of the first cell center.

This is a no-op when the closure has no eddy viscosity field (e.g.
`ScalarDiffusivity`) or the momentum fields have no `BulkDrag` bottom
boundary condition.
"""
function AtmosphereModels.enforce_surface_layer_consistency!(model)
    _enforce_surface_layer_consistency!(model.closure_fields, model)
end

# Fallback: closures without νₑ (e.g. ScalarDiffusivity, nothing)
_enforce_surface_layer_consistency!(closure_fields, model) = nothing

# Dispatch for closures that store νₑ as a named field
function _enforce_surface_layer_consistency!(closure_fields::NamedTuple{names}, model) where names
    :νₑ ∉ names && return nothing

    # Check if ρu has a BulkDrag bottom BC
    ρu_bc = model.momentum.ρu.boundary_conditions.bottom
    ρu_bc isa BulkDragBoundaryCondition || return nothing

    grid = model.grid
    arch = grid.architecture
    drag = ρu_bc.condition
    νₑ = closure_fields.νₑ
    model_fields = Oceananigans.fields(model)

    launch!(arch, grid, :xy,
            _set_surface_eddy_viscosity!, νₑ, grid, drag, model_fields)

    return nothing
end

@kernel function _set_surface_eddy_viscosity!(νₑ, grid, drag, model_fields)
    i, j = @index(Global, NTuple)

    # Surface wind speed including gustiness
    U² = wind_speed²ᶜᶜᶜ(i, j, grid, model_fields)
    Ũ² = U² + drag.gustiness^2

    # Drag coefficient at (i, j)
    T₀ = surface_value(i, j, drag.surface_temperature)
    Cᴰ = bulk_coefficient(i, j, grid, drag.coefficient, model_fields, T₀)

    # Friction velocity: u★² = Cᴰ Ũ²
    u_star = sqrt(Cᴰ * Ũ²)

    # Height of first cell center
    z₁ = znode(i, j, 1, grid, Center(), Center(), Center())

    # MOST eddy viscosity
    @inbounds νₑ[i, j, 1] = κ * u_star * z₁
end
