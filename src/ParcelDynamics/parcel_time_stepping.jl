#####
##### Time stepping for parcel evolution
#####

using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions,
    LiquidIcePotentialTemperatureState, StaticEnergyState,
    temperature, with_moisture, mixture_heat_capacity

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

    # 3. Get environmental conditions at new height
    p_new = environmental_pressure(profile, z_new)
    ρ_new = environmental_density(profile, z_new)

    # 4. Adiabatic adjustment of thermodynamic state
    𝒰_new = adiabatic_adjustment(𝒰, z_new, p_new, constants)

    # 5. Compute microphysics tendencies and update state
    ℳ_new = step_microphysics(microphysics, ℳ, ρ_new, 𝒰_new, constants, Δt)

    # 6. Update moisture fractions in thermodynamic state based on new microphysics
    q_new = compute_moisture_fractions(ℳ_new, qᵗ)
    𝒰_new = with_moisture(𝒰_new, q_new)

    return ParcelState(x_new, y_new, z_new, ρ_new, qᵗ, 𝒰_new, ℳ_new)
end

#####
##### Adiabatic adjustment for different thermodynamic formulations
#####

"""
$(TYPEDSIGNATURES)

Adjust the thermodynamic state for adiabatic ascent/descent to a new height.

For `StaticEnergyState`: The moist static energy is conserved, so we update
the height and reference pressure while keeping `e` constant.

For `LiquidIcePotentialTemperatureState`: The liquid-ice potential temperature
is conserved, so we update the reference pressure while keeping `θˡⁱ` constant.
"""
function adiabatic_adjustment end

# StaticEnergyState: conserve static energy, update height and pressure
@inline function adiabatic_adjustment(𝒰::StaticEnergyState{FT}, z_new, p_new, constants) where FT
    # Static energy is conserved during adiabatic processes
    return StaticEnergyState{FT}(𝒰.static_energy, 𝒰.moisture_mass_fractions, z_new, p_new)
end

# LiquidIcePotentialTemperatureState: conserve θˡⁱ, update pressure
@inline function adiabatic_adjustment(𝒰::LiquidIcePotentialTemperatureState{FT}, z_new, p_new, constants) where FT
    # Liquid-ice potential temperature is conserved during moist adiabatic processes
    return LiquidIcePotentialTemperatureState{FT}(
        𝒰.potential_temperature,
        𝒰.moisture_mass_fractions,
        𝒰.standard_pressure,
        p_new
    )
end

#####
##### Microphysics stepping for parcel
#####

"""
$(TYPEDSIGNATURES)

Advance the microphysical state by one time step using Forward Euler.

This function computes tendencies for all prognostic microphysical variables
and integrates them forward in time.
"""
function step_microphysics end

# Default: no microphysical evolution for abstract or trivial state
step_microphysics(microphysics, ℳ::Nothing, ρ, 𝒰, constants, Δt) = nothing
step_microphysics(microphysics::Nothing, ℳ, ρ, 𝒰, constants, Δt) = ℳ

#####
##### Compute moisture fractions from microphysical state
#####

"""
$(TYPEDSIGNATURES)

Compute moisture mass fractions from the microphysical state.
"""
function compute_moisture_fractions end

# Trivial state: all moisture is vapor
@inline function compute_moisture_fractions(ℳ::Nothing, qᵗ)
    return MoistureMassFractions(qᵗ)
end

# TrivialMicrophysicalState: all moisture is vapor
@inline function compute_moisture_fractions(ℳ::TrivialMicrophysicalState, qᵗ)
    return MoistureMassFractions(qᵗ)
end
