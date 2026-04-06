using Oceananigans.BoundaryConditions: getbc, BoundaryCondition, Flux
using Oceananigans.Grids: Center, znode

using ..AtmosphereModels: dynamics_density
using ..BoundaryConditions: κ

#####
##### Enforce Monin-Obukhov surface layer consistency for closure fields
#####
##### After `compute_closure_fields!` fills νₑ throughout the domain,
##### overwrite the first cell center with the MOST-consistent value:
#####
#####     νₑ(z₁) = κ u★ z₁
#####
##### where u★ is the friction velocity derived from the bottom momentum
##### flux boundary conditions. This ensures the closure's eddy viscosity
##### near the surface is consistent with the prescribed surface flux,
##### so that the shear stress ρ νₑ ∂u/∂z recovers the surface stress ρ u★².
#####

# Detect non-trivial flux boundary conditions
_is_flux_bc(::BoundaryCondition{<:Flux, Nothing}) = false
_is_flux_bc(::BoundaryCondition{<:Flux}) = true
_is_flux_bc(::Any) = false

"""
    enforce_surface_layer_consistency!(model)

Overwrite the near-surface eddy viscosity `νₑ[:, :, 1]` with a value
consistent with Monin-Obukhov similarity theory,

    `νₑ(z₁) = κ u★ z₁`

where `κ ≈ 0.4` is the von Kármán constant, `u★` is the friction velocity
derived from the bottom momentum flux boundary conditions, and `z₁` is the
height of the first cell center.

This is a no-op when the closure has no eddy viscosity field (e.g.
`ScalarDiffusivity`) or the momentum fields have no bottom flux boundary
conditions.
"""
function AtmosphereModels.enforce_surface_layer_consistency!(model)
    _enforce_surface_layer_consistency!(model.closure_fields, model)
end

# Fallback: closures without νₑ (e.g. ScalarDiffusivity, nothing)
_enforce_surface_layer_consistency!(closure_fields, model) = nothing

# Dispatch for closures that store νₑ as a named field
function _enforce_surface_layer_consistency!(closure_fields::NamedTuple{names}, model) where names
    :νₑ ∉ names && return nothing

    ρu_bc = model.momentum.ρu.boundary_conditions.bottom
    ρv_bc = model.momentum.ρv.boundary_conditions.bottom

    # At least one momentum component must have a non-trivial flux BC
    _is_flux_bc(ρu_bc) || _is_flux_bc(ρv_bc) || return nothing

    grid = model.grid
    arch = grid.architecture
    νₑ = closure_fields.νₑ
    ρ = dynamics_density(model.dynamics)
    model_fields = Oceananigans.fields(model)

    launch!(arch, grid, :xy,
            _set_surface_eddy_viscosity!, νₑ, grid,
            ρu_bc.condition, ρv_bc.condition, ρ,
            model.clock, model_fields)

    return nothing
end

@kernel function _set_surface_eddy_viscosity!(νₑ, grid, ρu_flux, ρv_flux, ρ, clock, model_fields)
    i, j = @index(Global, NTuple)

    # Evaluate surface stress from boundary conditions
    τ_x = getbc(ρu_flux, i, j, grid, clock, model_fields)
    τ_y = getbc(ρv_flux, i, j, grid, clock, model_fields)

    # Surface reference density
    @inbounds ρ₀ = ρ[i, j, 1]

    # Friction velocity: u★² = |τ| / ρ₀
    u★ = sqrt(sqrt(τ_x^2 + τ_y^2) / ρ₀)

    # Height of first cell center
    z₁ = znode(i, j, 1, grid, Center(), Center(), Center())

    # MOST eddy viscosity
    @inbounds νₑ[i, j, 1] = κ * u★ * z₁
end
