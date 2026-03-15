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
using Oceananigans.Fields: ZeroField
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
##### Moisture prognostic name
#####

"""
$(TYPEDSIGNATURES)

P3 is a non-equilibrium scheme: vapor (`qᵛ`) is the prognostic moisture variable.
"""
AM.moisture_prognostic_name(::P3) = :ρqᵛ

"""
$(TYPEDSIGNATURES)

Convert total moisture to the prognostic moisture variable for P3.

For P3, the prognostic moisture is vapor: `qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ`.
"""
@inline function AM.specific_prognostic_moisture_from_total(::P3, qᵗ, ℳ::P3MicrophysicalState)
    return max(0, qᵗ - ℳ.qᶜˡ - ℳ.qʳ - ℳ.qⁱ - ℳ.qʷⁱ)
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

    # Sedimentation velocity fields (pre-computed during update_state!)
    wʳ  = CenterField(grid)  # Rain mass-weighted terminal velocity
    wʳₙ = CenterField(grid)  # Rain number-weighted terminal velocity
    wⁱ  = CenterField(grid)  # Ice mass-weighted terminal velocity
    wⁱₙ = CenterField(grid)  # Ice number-weighted terminal velocity
    wⁱ_z = CenterField(grid) # Ice reflectivity-weighted terminal velocity

    # Microphysical tendency cache (written in update_microphysical_auxiliaries!, read by
    # grid_microphysical_tendency). Storing the microphysics-only contribution avoids 10×
    # redundant compute_p3_process_rates calls — one per prognostic field per grid point.
    cache_ρqᶜˡ = CenterField(grid)
    cache_ρqʳ  = CenterField(grid)
    cache_ρnʳ  = CenterField(grid)
    cache_ρqⁱ  = CenterField(grid)
    cache_ρnⁱ  = CenterField(grid)
    cache_ρqᶠ  = CenterField(grid)
    cache_ρbᶠ  = CenterField(grid)
    cache_ρzⁱ  = CenterField(grid)
    cache_ρqʷⁱ = CenterField(grid)
    cache_ρqᵛ  = CenterField(grid)

    return (; ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ, qᵛ,
              wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
              cache_ρqᶜˡ, cache_ρqʳ, cache_ρnʳ, cache_ρqⁱ, cache_ρnⁱ,
              cache_ρqᶠ, cache_ρbᶠ, cache_ρzⁱ, cache_ρqʷⁱ, cache_ρqᵛ)
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
@inline function AM.microphysical_state(::P3, ρ, μ, 𝒰, velocities)
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

# Disambiguation for P3 with Nothing or empty microphysical fields
@inline AM.microphysical_state(::P3, ρ, ::Nothing, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))
@inline AM.microphysical_state(::P3, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))

#####
##### Update microphysical auxiliary fields
#####

"""
$(TYPEDSIGNATURES)

Update diagnostic microphysical fields after state update.

After the moisture refactor, vapor is the prognostic moisture variable.
The diagnostic `qᵛ` field is updated from the thermodynamic state.
"""
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, p3::P3, ℳ::P3MicrophysicalState, ρ, 𝒰, constants)
    FT = typeof(ρ)

    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor

    # Compute ice properties for terminal velocity
    Fᶠ = safe_divide(ℳ.qᶠ, ℳ.qⁱ, zero(FT))
    ρᶠ = safe_divide(ℳ.qᶠ, ℳ.bᶠ, FT(400))

    # Pre-compute terminal velocities for sedimentation (stored as negative w)
    @inbounds μ.wʳ[i, j, k]   = -rain_terminal_velocity_mass_weighted(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    @inbounds μ.wʳₙ[i, j, k]  = -rain_terminal_velocity_number_weighted(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    vⁱ = ice_terminal_velocities(p3, ℳ.qⁱ, ℳ.nⁱ, Fᶠ, ρᶠ, ρ)
    @inbounds μ.wⁱ[i, j, k]   = -vⁱ.mass_weighted
    @inbounds μ.wⁱₙ[i, j, k]  = -vⁱ.number_weighted
    @inbounds μ.wⁱ_z[i, j, k] = -vⁱ.reflectivity_weighted

    # Compute all process rates once and cache every microphysical tendency contribution.
    # grid_microphysical_tendency overrides below read from these cache fields, eliminating
    # the 10× redundant compute_p3_process_rates calls (one per P3 prognostic field).
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    @inbounds μ.cache_ρqᶜˡ[i, j, k] = tendency_ρqᶜˡ(rates, ρ)
    @inbounds μ.cache_ρqʳ[i, j, k]  = tendency_ρqʳ(rates, ρ)
    @inbounds μ.cache_ρnʳ[i, j, k]  = tendency_ρnʳ(rates, ρ, ℳ.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
    @inbounds μ.cache_ρqⁱ[i, j, k]  = tendency_ρqⁱ(rates, ρ)
    @inbounds μ.cache_ρnⁱ[i, j, k]  = tendency_ρnⁱ(rates, ρ)
    @inbounds μ.cache_ρqᶠ[i, j, k]  = tendency_ρqᶠ(rates, ρ, Fᶠ)
    @inbounds μ.cache_ρbᶠ[i, j, k]  = tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, ℳ.qⁱ, p3.process_rates)
    @inbounds μ.cache_ρzⁱ[i, j, k]  = tendency_ρzⁱ(rates, ρ, ℳ.qⁱ, ℳ.nⁱ, ℳ.zⁱ)
    @inbounds μ.cache_ρqʷⁱ[i, j, k] = tendency_ρqʷⁱ(rates, ρ)
    @inbounds μ.cache_ρqᵛ[i, j, k]  = tendency_ρqᵛ(rates, ρ)

    return nothing
end

#####
##### Moisture fractions (state-based)
#####

"""
$(TYPEDSIGNATURES)

Compute moisture mass fractions from P3 microphysical state.

After the moisture refactor, the first argument `qᵛ` is the prognostic
vapor specific humidity (not total moisture). Returns `MoistureMassFractions`
with vapor, liquid (cloud + rain + liquid on ice), and ice components.
"""
@inline function AM.moisture_fractions(::P3, ℳ::P3MicrophysicalState, qᵛ)
    # Total liquid = cloud + rain + liquid on ice
    qˡ = ℳ.qᶜˡ + ℳ.qʳ + ℳ.qʷⁱ

    # Ice (frozen fraction)
    qⁱ = ℳ.qⁱ

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Microphysical velocities (sedimentation)
#####
#
# Terminal velocities are pre-computed in update_microphysical_auxiliaries!
# and stored in diagnostic fields. microphysical_velocities returns NamedTuples
# compatible with Oceananigans' sum_of_velocities.

@inline AM.microphysical_velocities(::P3, μ, name) = nothing  # Default: no sedimentation

# Rain mass: mass-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqʳ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wʳ)

# Rain number: number-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρnʳ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wʳₙ)

# Ice mass: mass-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

# Ice number: number-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρnⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱₙ)

# Rime mass: same as ice mass (rime falls with ice)
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqᶠ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

# Rime volume: same as ice mass
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρbᶠ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

# Ice reflectivity: reflectivity-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρzⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ_z)

# Liquid on ice: same as ice mass
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqʷⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

#####
##### Microphysical tendencies
#####
#
# Two paths:
#   1. Grid-based (AtmosphereModel): grid_microphysical_tendency reads from the cache
#      fields populated by update_microphysical_auxiliaries! — one compute_p3_process_rates
#      call per grid point serves all 10 P3 fields.
#   2. Gridless (ParcelModel): microphysical_tendency builds state and computes rates directly.

# Helper to compute P3 rates and extract ice properties from ℳ
@inline function p3_rates_and_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    FT = typeof(ρ)

    # Compute all process rates from microphysical state ℳ and thermodynamic state 𝒰
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

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
    return tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
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
    return tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, ℳ.qⁱ, p3.process_rates)
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

"""
Vapor tendency: loses from condensation, deposition, nucleation; gains from evaporation, sublimation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᵛ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᵛ(rates, ρ)
end

# Fallback for any unhandled field names - return zero tendency
@inline AM.microphysical_tendency(::P3, name, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) = zero(ρ)

#####
##### Grid-indexed tendency overrides (fast path for AtmosphereModel)
#####
#
# These overrides read from the tendency cache populated by update_microphysical_auxiliaries!,
# bypassing recomputation of compute_p3_process_rates for each P3 prognostic field.
# The microphysical_tendency methods above remain the gridless fallback for ParcelModels.

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqᶜˡ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqᶜˡ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqʳ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqʳ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρnʳ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρnʳ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρnⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρnⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqᶠ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqᶠ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρbᶠ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρbᶠ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρzⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρzⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqʷⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqʷⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqᵛ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqᵛ[i, j, k]

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
@inline AM.maybe_adjust_thermodynamic_state(𝒰, ::P3, qᵛ, constants) = 𝒰

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
