using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions,
    LiquidIcePotentialTemperatureState, StaticEnergyState,
    temperature, with_moisture, mixture_heat_capacity

#####
##### ParcelState: state of a rising parcel
#####

"""
$(TYPEDEF)

The complete state of a Lagrangian air parcel.

The parcel state contains all variables needed to evolve the parcel through
an atmospheric profile. Position `(x, y, z)` tracks the parcel location,
while thermodynamic variables describe the parcel's internal state.

The thermodynamic state `𝒰` and microphysical state `ℳ` use the same scalar
struct abstractions as the grid-based `AtmosphereModel`, enabling code reuse
for tendency calculations.

# Fields
$(TYPEDFIELDS)

# Notes

The parcel evolves adiabatically (conserving entropy/potential temperature)
as it moves through the environmental profile. Microphysics tendencies modify
the moisture partition while conserving total water.

For warm-phase microphysics, the prognostic variables are typically:
- `qᶜˡ`: cloud liquid mixing ratio
- `qʳ`: rain mixing ratio

The vapor mixing ratio `qᵛ = qᵗ - qˡ - qⁱ` is diagnostic.
"""
struct ParcelState{FT, TH, MI}
    x :: FT
    y :: FT
    z :: FT
    ρ :: FT
    qᵗ :: FT
    𝒰 :: TH
    ℳ :: MI
end

# Accessors
@inline position(state::ParcelState) = (state.x, state.y, state.z)
@inline height(state::ParcelState) = state.z
@inline density(state::ParcelState) = state.ρ
@inline total_moisture(state::ParcelState) = state.qᵗ

Base.eltype(::ParcelState{FT}) where FT = FT

function Base.show(io::IO, state::ParcelState{FT}) where FT
    print(io, "ParcelState{$FT}(")
    print(io, "x=", state.x, ", ")
    print(io, "y=", state.y, ", ")
    print(io, "z=", state.z, ", ")
    print(io, "ρ=", round(state.ρ, digits=4), ", ")
    print(io, "qᵗ=", round(state.qᵗ * 1000, digits=2), " g/kg)")
end

"""
$(TYPEDSIGNATURES)

Create a new `ParcelState` with updated fields.

This is the primary way to evolve the parcel state, returning a new
immutable state with modified values while preserving unspecified fields.
"""
function with_state(state::ParcelState;
                    x = state.x,
                    y = state.y,
                    z = state.z,
                    ρ = state.ρ,
                    qᵗ = state.qᵗ,
                    𝒰 = state.𝒰,
                    ℳ = state.ℳ)
    return ParcelState(x, y, z, ρ, qᵗ, 𝒰, ℳ)
end

#####
##### ParcelDynamics: rules for evolving the parcel state
#####

struct ParcelDynamics{S}
    state :: S
end

ParcelDynamics(state::ParcelState) = ParcelDynamics(state)

Adapt.adapt_structure(to, dynamics::ParcelDynamics) =
    ParcelDynamics(adapt(to, dynamics.state))

AtmosphereModels.default_dynamics(grid, constants) =
    ParcelDynamics(ParcelState(grid, constants))

AtmosphereModels.materialize_dynamics(dynamics::ParcelDynamics, grid, boundary_conditions, thermodynamic_constants) =
    ParcelDynamics(dynamics.state)

const ParcelModel = AtmosphereModel{<:ParcelDynamics}

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
function time_step!(model::ParcelModel, Δt)
    state = model.dynamics.state
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

# NothingMicrophysicalState: all moisture is vapor
@inline function compute_moisture_fractions(ℳ::NothingMicrophysicalState, qᵗ)
    return MoistureMassFractions(qᵗ)
end