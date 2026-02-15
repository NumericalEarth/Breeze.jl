using ..Thermodynamics: ReferenceState, compute_hydrostatic_reference!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, ZeroField
using Statistics: mean!

"""
    set_to_mean!(reference_state, model)

Recompute the reference pressure and density profiles from horizontally-averaged
temperature and moisture mass fractions of the current model state.

This is useful as a simulation callback to keep the reference state close to the
evolving mean state, minimizing buoyancy perturbations `ρ - ρᵣ`.
"""
function set_to_mean!(ref::ReferenceState, model)
    constants = model.thermodynamic_constants

    mean!(ref.temperature, model.temperature)
    fill_halo_regions!(ref.temperature)

    mean_mass_fraction!(ref.vapor_mass_fraction, vapor_mass_fraction(model))
    mean_mass_fraction!(ref.liquid_mass_fraction, liquid_mass_fraction(model))
    mean_mass_fraction!(ref.ice_mass_fraction, ice_mass_fraction(model))

    compute_hydrostatic_reference!(ref, constants)

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
