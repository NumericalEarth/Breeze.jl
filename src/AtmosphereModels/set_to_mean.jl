using ..Thermodynamics: ReferenceState, compute_hydrostatic_reference!
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, ZeroField
using Statistics: mean!

"""
    rescale_density_weighted_fields!(model, ρᵣ_old)

Rescale all density-weighted prognostic fields so that specific quantities
(velocity, potential temperature, moisture, etc.) are preserved after a change
in the reference density `ρᵣ`. Each field is multiplied by `ρᵣ_new / ρᵣ_old`.

`ρᵣ_old` is a `Field` containing the reference density before the update.
Oceananigans handles interpolation to staggered locations (e.g. face-located
momentum fields) automatically.
"""
function rescale_density_weighted_fields!(model, ρᵣ_old)
    ρᵣ = dynamics_density(model.dynamics)
    for field in prognostic_fields(model)
        field .= field * ρᵣ / ρᵣ_old
    end
    return nothing
end

"""
    set_to_mean!(reference_state, model; rescale_densities=false)

Recompute the reference pressure and density profiles from horizontally-averaged
temperature and moisture mass fractions of the current model state.

When `rescale_densities=true`, density-weighted prognostic fields (ρe, ρqᵗ, ρu,
etc.) are rescaled by `ρᵣ_new / ρᵣ_old` so that the specific quantities
(e, qᵗ, u, etc.) are unchanged. When `false` (default), the density-weighted
fields are left as-is and only diagnostics are recomputed.
"""
function set_to_mean!(ref::ReferenceState, model; rescale_densities=false)
    constants = model.thermodynamic_constants

    if rescale_densities
        ρᵣ_old = similar(dynamics_density(model.dynamics))
        parent(ρᵣ_old) .= parent(dynamics_density(model.dynamics))
    end

    # Update reference temperature and moisture from horizontal means
    mean!(ref.temperature, model.temperature)
    fill_halo_regions!(ref.temperature)

    mean_mass_fraction!(ref.vapor_mass_fraction, vapor_mass_fraction(model))
    mean_mass_fraction!(ref.liquid_mass_fraction, liquid_mass_fraction(model))
    mean_mass_fraction!(ref.ice_mass_fraction, ice_mass_fraction(model))

    # Recompute hydrostatic pressure and density
    compute_hydrostatic_reference!(ref, constants)

    if rescale_densities
        rescale_density_weighted_fields!(model, ρᵣ_old)
    end

    # Recompute all diagnostic variables (T, qᵗ, u, v, w, diffusivities, etc.)
    TimeSteppers.update_state!(model; compute_tendencies=false)

    return nothing
end

function mean_mass_fraction!(ref_field, field)
    mean!(ref_field, field)
    fill_halo_regions!(ref_field)
    return nothing
end

function mean_mass_fraction!(ref_field, ::Nothing)
    interior(ref_field) .= 0
    fill_halo_regions!(ref_field)
    return nothing
end

# ZeroField reference moisture: nothing to update
mean_mass_fraction!(::ZeroField, field) = nothing
mean_mass_fraction!(::ZeroField, ::Nothing) = nothing
