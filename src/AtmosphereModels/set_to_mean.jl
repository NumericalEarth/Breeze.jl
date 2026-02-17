using ..Thermodynamics: ReferenceState, compute_hydrostatic_reference!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, ZeroField
using Statistics: mean!

"""
    set_to_mean!(reference_state, model)

Recompute the reference pressure and density profiles from horizontally-averaged
temperature and moisture mass fractions of the current model state.

Because prognostic variables are density-weighted (ρe = ρᵣ·e, ρqᵗ = ρᵣ·qᵗ,
ρu = ρᵣ·u, etc.), the momentum is reconstructed from the diagnostic velocity
fields (which are unaffected by the reference state change) using the new ρᵣ,
and scalar fields are rescaled by ρᵣ_new / ρᵣ_old.
"""
function set_to_mean!(ref::ReferenceState, model)
    constants = model.thermodynamic_constants
    ρᵣ = ref.density

    # Save old ρᵣ interior before recomputing (column field, size 1×1×Nz)
    ρᵣ_old = similar(interior(ρᵣ))
    ρᵣ_old .= interior(ρᵣ)

    # Update reference temperature and moisture from horizontal means
    mean!(ref.temperature, model.temperature)
    fill_halo_regions!(ref.temperature)

    mean_mass_fraction!(ref.vapor_mass_fraction, vapor_mass_fraction(model))
    mean_mass_fraction!(ref.liquid_mass_fraction, liquid_mass_fraction(model))
    mean_mass_fraction!(ref.ice_mass_fraction, ice_mass_fraction(model))

    # Recompute hydrostatic pressure and density
    compute_hydrostatic_reference!(ref, constants)

    # Reconstruct momentum from diagnostic velocities (handles face interpolation)
    # and rescale scalar prognostic fields by ρᵣ_new / ρᵣ_old
    rescale_prognostic_fields!(model, ρᵣ, ρᵣ_old)

    # Recompute all diagnostic variables (T, qᵗ, u, v, w, diffusivities, etc.)
    # from the rescaled prognostics + new ρᵣ
    TimeSteppers.update_state!(model; compute_tendencies=false)

    return nothing
end

function rescale_prognostic_fields!(model, ρᵣ_new, ρᵣ_old)
    ρ = dynamics_density(model.dynamics) # same as ρᵣ_new, but the Field object

    # Reconstruct momentum from diagnostic velocities using new ρᵣ.
    # This correctly handles face interpolation (ρu = ℑx(ρᵣ)·u, ρw = ℑz(ρᵣ)·w).
    for name in (:u, :v, :w)
        u = model.velocities[name]
        ρu = model.momentum[Symbol(:ρ, name)]
        set!(ρu, ρ * u)
    end

    # Rescale center-located scalar fields by ρᵣ_new / ρᵣ_old (broadcast: 1×1×Nz)
    ratio = interior(ρᵣ_new) ./ ρᵣ_old

    ρe = thermodynamic_density(model.formulation)
    interior(ρe) .*= ratio

    interior(model.moisture_density) .*= ratio

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
