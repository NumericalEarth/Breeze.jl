#####
##### Time stepping for parcel evolution
#####

"""
$(TYPEDSIGNATURES)

Advance the parcel state by one time step `Δt`.

The parcel is advected by the environmental velocity field, and the
thermodynamic/microphysical state evolves according to:

1. **Position update**: The parcel position is updated using the environmental
   velocity at the current location.

2. **Adiabatic adjustment**: The parcel thermodynamic state is adjusted for
   the pressure change at the new height (adiabatic expansion/compression).

3. **Microphysics tendencies**: Cloud condensate and precipitation evolve
   according to the microphysics scheme.

# Arguments
- `state`: Current [`ParcelState`](@ref)
- `model`: [`ParcelModel`](@ref) containing environmental profile and microphysics
- `Δt`: Time step [s]

# Returns
A new `ParcelState` representing the parcel at time `t + Δt`.

# Notes

This implements Forward Euler time stepping. For more accurate integration,
multiple sub-steps can be used or higher-order schemes implemented.

The parcel conserves its potential temperature (dry) or equivalent potential
temperature (moist) during adiabatic ascent, while microphysics processes
modify the moisture partition.
"""
function step_parcel!(state::ParcelState, model::ParcelModel, Δt)
    profile = model.profile
    microphysics = model.microphysics
    constants = model.constants

    # Current position and state
    x, y, z = position(state)
    ρ = density(state)
    qᵗ = total_moisture(state)
    𝒰 = state.thermodynamic_state
    ℳ = state.microphysical_state

    # 1. Get environmental velocity at current position
    u, v, w = environmental_velocity(profile, z)

    # 2. Update position (Forward Euler)
    x_new = x + u * Δt
    y_new = y + v * Δt
    z_new = z + w * Δt

    # 3. Get environmental pressure at new height for adiabatic adjustment
    p_new = environmental_pressure(profile, z_new)
    ρ_new = environmental_density(profile, z_new)

    # 4. Compute microphysics tendencies for all prognostic microphysical variables
    # The tendency functions operate on the scalar state ℳ (no grid indexing)
    ℳ_new = step_microphysics(microphysics, ℳ, ρ, 𝒰, constants, Δt)

    # 5. Adiabatic adjustment of thermodynamic state
    # TODO: Implement adiabatic expansion for different thermodynamic formulations
    # For now, we keep the same thermodynamic state (isothermal approximation)
    𝒰_new = 𝒰  # Placeholder: proper adiabatic adjustment needed

    # 6. Update moisture from microphysical evolution
    # Total water is conserved (no precipitation fallout in simple case)
    qᵗ_new = qᵗ

    return ParcelState(x_new, y_new, z_new, ρ_new, qᵗ_new, 𝒰_new, ℳ_new)
end

#####
##### Microphysics stepping for parcel
#####

"""
$(TYPEDSIGNATURES)

Advance the microphysical state by one time step.

This function applies Forward Euler integration to the microphysics tendencies.
For more robust evolution, sub-stepping or implicit methods may be needed.
"""
function step_microphysics(microphysics, ℳ::AbstractMicrophysicalState, ρ, 𝒰, constants, Δt)
    # Default: no microphysical evolution for abstract state
    return ℳ
end

# Trivial state: no evolution
step_microphysics(microphysics, ℳ::Nothing, ρ, 𝒰, constants, Δt) = nothing
