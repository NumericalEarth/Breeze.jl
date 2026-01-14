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
struct ParcelState{FT, 𝒰, ℳ}
    "Parcel x-position [m]"
    x :: FT

    "Parcel y-position [m]"
    y :: FT

    "Parcel z-position (height) [m]"
    z :: FT

    "Parcel density [kg/m³]"
    ρ :: FT

    "Total specific humidity (water mixing ratio) [kg/kg]"
    qᵗ :: FT

    "Thermodynamic state (e.g., StaticEnergyState or LiquidIcePotentialTemperatureState)"
    thermodynamic_state :: 𝒰

    "Microphysical state (e.g., WarmPhaseOneMomentState)"
    microphysical_state :: ℳ
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
                    thermodynamic_state = state.thermodynamic_state,
                    microphysical_state = state.microphysical_state)
    return ParcelState(x, y, z, ρ, qᵗ, thermodynamic_state, microphysical_state)
end
