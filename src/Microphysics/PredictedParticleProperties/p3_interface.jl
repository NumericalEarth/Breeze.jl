#####
##### Microphysics interface implementation for P3
#####
##### These functions integrate the P3 scheme with AtmosphereModel,
##### allowing it to be used as a drop-in microphysics scheme.
#####
##### This file follows the MicrophysicalState abstraction pattern:
##### - P3MicrophysicalState encapsulates local microphysical variables
##### - Gridless microphysical_state(p3, ρ, μ, 𝒰) builds the state
##### - State-based microphysical_tendency(p3, name, ρ, ℳ, 𝒰, constants) computes tendencies
#####

using Oceananigans: CenterField
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze.Thermodynamics: MoistureMassFractions

const P3 = PredictedParticlePropertiesMicrophysics

#####
##### P3MicrophysicalState
#####

"""
    P3MicrophysicalState{FT} <: AbstractMicrophysicalState{FT}

Microphysical state for P3 (Predicted Particle Properties) microphysics.

Contains the local mixing ratios and number concentrations needed to compute
tendencies for cloud liquid, rain, ice, rime, and predicted liquid fraction.

# Fields
$(TYPEDFIELDS)
"""
struct P3MicrophysicalState{FT} <: AbstractMicrophysicalState{FT}
    "Cloud liquid mixing ratio [kg/kg]"
    qᶜˡ :: FT
    "Rain mixing ratio [kg/kg]"
    qʳ  :: FT
    "Rain number concentration [1/kg]"
    nʳ  :: FT
    "Ice mixing ratio [kg/kg]"
    qⁱ  :: FT
    "Ice number concentration [1/kg]"
    nⁱ  :: FT
    "Rime mass mixing ratio [kg/kg]"
    qᶠ  :: FT
    "Rime volume [m³/kg]"
    bᶠ  :: FT
    "Ice sixth moment [m⁶/kg]"
    zⁱ  :: FT
    "Liquid water on ice mixing ratio [kg/kg]"
    qʷⁱ :: FT
end

#####
##### Prognostic field names
#####

"""
$(TYPEDSIGNATURES)

Return prognostic field names for the P3 scheme.

P3 v5.5 with 3-moment ice and predicted liquid fraction has 9 prognostic fields:
- Cloud: ρqᶜˡ (number is prescribed, not prognostic)
- Rain: ρqʳ, ρnʳ
- Ice: ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ
"""
function AM.prognostic_field_names(::P3)
    # Cloud number is prescribed (not prognostic) in this implementation
    cloud_names = (:ρqᶜˡ,)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρzⁱ, :ρqʷⁱ)

    return tuple(cloud_names..., rain_names..., ice_names...)
end

#####
##### Specific humidity
#####

"""
$(TYPEDSIGNATURES)

Return the vapor specific humidity field for P3 microphysics.

For P3, vapor is diagnosed from total moisture minus all condensates:
qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ
"""
function AM.specific_humidity(::P3, model)
    # P3 stores vapor diagnostically
    return model.microphysical_fields.qᵛ
end

#####
##### Materialize microphysical fields
#####

"""
$(TYPEDSIGNATURES)

Create prognostic and diagnostic fields for P3 microphysics.

The P3 scheme requires the following fields on `grid`:

**Prognostic (density-weighted):**
- `ρqᶜˡ`: Cloud liquid mass density
- `ρqʳ`, `ρnʳ`: Rain mass and number densities
- `ρqⁱ`, `ρnⁱ`: Ice mass and number densities
- `ρqᶠ`, `ρbᶠ`: Rime mass and volume densities
- `ρzⁱ`: Ice sixth moment (reflectivity) density
- `ρqʷⁱ`: Liquid water on ice mass density

**Diagnostic:**
- `qᵛ`: Vapor specific humidity (computed from total moisture)
"""
function AM.materialize_microphysical_fields(::P3, grid, bcs)
    # Create all prognostic fields
    ρqᶜˡ = CenterField(grid)  # Cloud liquid
    ρqʳ  = CenterField(grid)  # Rain mass
    ρnʳ  = CenterField(grid)  # Rain number
    ρqⁱ  = CenterField(grid)  # Ice mass
    ρnⁱ  = CenterField(grid)  # Ice number
    ρqᶠ  = CenterField(grid)  # Rime mass
    ρbᶠ  = CenterField(grid)  # Rime volume
    ρzⁱ  = CenterField(grid)  # Ice 6th moment
    ρqʷⁱ = CenterField(grid)  # Liquid on ice

    # Diagnostic field for vapor
    qᵛ = CenterField(grid)

    return (; ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ, qᵛ)
end

#####
##### Gridless MicrophysicalState construction
#####
#
# P3 is a non-equilibrium scheme: all condensate comes from prognostic fields μ.

"""
$(TYPEDSIGNATURES)

Build a [`P3MicrophysicalState`](@ref) from density-weighted prognostic variables.

P3 is a non-equilibrium scheme, so all cloud and precipitation variables come
from the prognostic fields `μ`, not from the thermodynamic state `𝒰`.
"""
@inline function AM.microphysical_state(::P3, ρ, μ, 𝒰)
    qᶜˡ = μ.ρqᶜˡ / ρ
    qʳ  = μ.ρqʳ / ρ
    nʳ  = μ.ρnʳ / ρ
    qⁱ  = μ.ρqⁱ / ρ
    nⁱ  = μ.ρnⁱ / ρ
    qᶠ  = μ.ρqᶠ / ρ
    bᶠ  = μ.ρbᶠ / ρ
    zⁱ  = μ.ρzⁱ / ρ
    qʷⁱ = μ.ρqʷⁱ / ρ
    return P3MicrophysicalState(qᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ)
end

#####
##### Update microphysical auxiliary fields
#####

"""
$(TYPEDSIGNATURES)

Update diagnostic microphysical fields after state update.

For P3, we compute vapor as the residual: qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ
"""
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, ::P3, ℳ::P3MicrophysicalState, ρ, 𝒰, constants)
    # Get total moisture from thermodynamic state
    q = 𝒰.moisture_mass_fractions
    qᵗ = q.vapor + q.liquid + q.ice

    # Vapor is residual (total - all condensates)
    qᵛ = max(0, qᵗ - ℳ.qᶜˡ - ℳ.qʳ - ℳ.qⁱ - ℳ.qʷⁱ)

    @inbounds μ.qᵛ[i, j, k] = qᵛ
    return nothing
end

#####
##### Moisture fractions (state-based)
#####

"""
$(TYPEDSIGNATURES)

Compute moisture mass fractions from P3 microphysical state.

Returns `MoistureMassFractions` with vapor, liquid (cloud + rain + liquid on ice),
and ice components.
"""
@inline function AM.moisture_fractions(::P3, ℳ::P3MicrophysicalState, qᵗ)
    # Total liquid = cloud + rain + liquid on ice
    qˡ = ℳ.qᶜˡ + ℳ.qʳ + ℳ.qʷⁱ

    # Ice (frozen fraction)
    qⁱ = ℳ.qⁱ

    # Vapor is residual (ensuring non-negative)
    qᵛ = max(0, qᵗ - qˡ - qⁱ)

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Microphysical velocities (sedimentation)
#####

"""
$(TYPEDSIGNATURES)

Return terminal velocity for precipitating species.

P3 has separate fall speeds for rain and ice particles.
Returns a NamedTuple with `(u=0, v=0, w=-vₜ)` where `vₜ` is the terminal velocity.

For mass fields (ρqʳ, ρqⁱ, ρqᶠ, ρqʷⁱ), uses mass-weighted velocity.
For number fields (ρnʳ, ρnⁱ), uses number-weighted velocity.
For reflectivity (ρzⁱ), uses reflectivity-weighted velocity.
"""
@inline AM.microphysical_velocities(p3::P3, μ, name) = nothing  # Default: no sedimentation

# Rain mass: mass-weighted fall speed
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρqʳ})
    return RainMassSedimentationVelocity(p3, μ)
end

# Rain number: number-weighted fall speed
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρnʳ})
    return RainNumberSedimentationVelocity(p3, μ)
end

# Ice mass: mass-weighted fall speed
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρqⁱ})
    return IceMassSedimentationVelocity(p3, μ)
end

# Ice number: number-weighted fall speed
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρnⁱ})
    return IceNumberSedimentationVelocity(p3, μ)
end

# Rime mass: same as ice mass (rime falls with ice)
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρqᶠ})
    return IceMassSedimentationVelocity(p3, μ)
end

# Rime volume: same as ice mass
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρbᶠ})
    return IceMassSedimentationVelocity(p3, μ)
end

# Ice reflectivity: reflectivity-weighted fall speed
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρzⁱ})
    return IceReflectivitySedimentationVelocity(p3, μ)
end

# Liquid on ice: same as ice mass
@inline function AM.microphysical_velocities(p3::P3, μ, ::Val{:ρqʷⁱ})
    return IceMassSedimentationVelocity(p3, μ)
end

#####
##### Sedimentation velocity types
#####
##### These are callable structs that compute terminal velocities at (i, j, k).
#####

"""
Callable struct for rain mass sedimentation velocity.
"""
struct RainMassSedimentationVelocity{P, M}
    p3 :: P
    microphysical_fields :: M
end

@inline function (v::RainMassSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields
    p3 = v.p3

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        nʳ = μ.ρnʳ[i, j, k] / ρ
    end

    vₜ = rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for rain number sedimentation velocity.
"""
struct RainNumberSedimentationVelocity{P, M}
    p3 :: P
    microphysical_fields :: M
end

@inline function (v::RainNumberSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields
    p3 = v.p3

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        nʳ = μ.ρnʳ[i, j, k] / ρ
    end

    vₜ = rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for ice mass sedimentation velocity.
"""
struct IceMassSedimentationVelocity{P, M}
    p3 :: P
    microphysical_fields :: M
end

@inline function (v::IceMassSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields
    p3 = v.p3

    @inbounds begin
        qⁱ = μ.ρqⁱ[i, j, k] / ρ
        nⁱ = μ.ρnⁱ[i, j, k] / ρ
        qᶠ = μ.ρqᶠ[i, j, k] / ρ
        bᶠ = μ.ρbᶠ[i, j, k] / ρ
    end

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    vₜ = ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for ice number sedimentation velocity.
"""
struct IceNumberSedimentationVelocity{P, M}
    p3 :: P
    microphysical_fields :: M
end

@inline function (v::IceNumberSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields
    p3 = v.p3

    @inbounds begin
        qⁱ = μ.ρqⁱ[i, j, k] / ρ
        nⁱ = μ.ρnⁱ[i, j, k] / ρ
        qᶠ = μ.ρqᶠ[i, j, k] / ρ
        bᶠ = μ.ρbᶠ[i, j, k] / ρ
    end

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    vₜ = ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for ice reflectivity sedimentation velocity.
"""
struct IceReflectivitySedimentationVelocity{P, M}
    p3 :: P
    microphysical_fields :: M
end

@inline function (v::IceReflectivitySedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields
    p3 = v.p3

    @inbounds begin
        qⁱ = μ.ρqⁱ[i, j, k] / ρ
        nⁱ = μ.ρnⁱ[i, j, k] / ρ
        zⁱ = μ.ρzⁱ[i, j, k] / ρ
        qᶠ = μ.ρqᶠ[i, j, k] / ρ
        bᶠ = μ.ρbᶠ[i, j, k] / ρ
    end

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    vₜ = ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

#####
##### Microphysical tendencies (state-based)
#####
#
# The new interface uses state-based tendencies: microphysical_tendency(p3, name, ρ, ℳ, 𝒰, constants)
# where ℳ is the P3MicrophysicalState.

# Helper to compute P3 rates and extract ice properties from ℳ
@inline function p3_rates_and_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    FT = typeof(ρ)

    # TODO: Compute all process rates from ℳ and 𝒰
    # For now, return placeholder rates structure
    # rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    rates = nothing  # Placeholder until process rates are fully implemented

    Fᶠ = safe_divide(ℳ.qᶠ, ℳ.qⁱ, zero(FT))
    ρᶠ = safe_divide(ℳ.qᶠ, ℳ.bᶠ, FT(400))

    return rates, ℳ.qⁱ, ℳ.nⁱ, ℳ.zⁱ, Fᶠ, ρᶠ
end

"""
Cloud liquid tendency: loses mass to autoconversion, accretion, and riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶜˡ(rates, ρ)
end

"""
Rain mass tendency: gains from autoconversion, accretion, melting, shedding; loses to evaporation, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʳ(rates, ρ)
end

"""
Rain number tendency: gains from autoconversion, melting, shedding; loses to self-collection, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, qⁱ, nⁱ, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnʳ(rates, ρ, nⁱ, qⁱ)
end

"""
Ice mass tendency: gains from deposition, riming, refreezing; loses to melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqⁱ(rates, ρ)
end

"""
Ice number tendency: loses from melting and aggregation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnⁱ(rates, ρ)
end

"""
Rime mass tendency: gains from cloud/rain riming, refreezing; loses proportionally with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, Fᶠ, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶠ(rates, ρ, Fᶠ)
end

"""
Rime volume tendency: gains from new rime; loses with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρbᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, Fᶠ, ρᶠ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ)
end

"""
Ice sixth moment tendency: changes with deposition, melting, riming, and nucleation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρzⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, qⁱ, nⁱ, zⁱ, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ)
end

"""
Liquid on ice tendency: loses from shedding and refreezing.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʷⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʷⁱ(rates, ρ)
end

# Fallback for any unhandled field names - return zero tendency
@inline AM.microphysical_tendency(::P3, name, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) = zero(ρ)

#####
##### Thermodynamic state adjustment
#####

"""
$(TYPEDSIGNATURES)

Apply saturation adjustment for P3.

P3 is a non-equilibrium scheme - cloud formation and dissipation are handled
by explicit process rates, not instantaneous saturation adjustment.
Therefore, this function returns the state unchanged.
"""
@inline AM.maybe_adjust_thermodynamic_state(𝒰, ::P3, qᵗ, constants) = 𝒰

#####
##### Model update
#####

"""
$(TYPEDSIGNATURES)

Apply P3 model update during state update phase.

Currently does nothing - this is where substepping or implicit updates would go.
"""
function AM.microphysics_model_update!(::P3, model)
    return nothing
end
