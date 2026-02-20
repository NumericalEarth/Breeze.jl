using ..Thermodynamics: ReferenceState, compute_hydrostatic_reference!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, ZeroField
using Statistics: mean!

"""
    set_to_mean!(reference_state, model)

Recompute the reference pressure and density profiles from horizontally-averaged
temperature and moisture mass fractions of the current model state.

Density-weighted prognostic fields (ρe, ρqᵗ, ρu, etc.) are left unchanged;
diagnostic fields are recomputed from the new reference state via `update_state!`.
"""
function set_to_mean!(ref::ReferenceState, model)
    constants = model.thermodynamic_constants

    # Update reference temperature and moisture from horizontal means
    mean!(ref.temperature, model.temperature)
    fill_halo_regions!(ref.temperature)

    mean_mass_fraction!(ref.vapor_mass_fraction, vapor_mass_fraction(model))
    mean_mass_fraction!(ref.liquid_mass_fraction, liquid_mass_fraction(model))
    mean_mass_fraction!(ref.ice_mass_fraction, ice_mass_fraction(model))

    # Recompute hydrostatic pressure and density
    compute_hydrostatic_reference!(ref, constants)

    # Recompute all diagnostic variables (T, qᵗ, u, v, w, diffusivities, etc.)
    # from the rescaled prognostics + new ρᵣ
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
