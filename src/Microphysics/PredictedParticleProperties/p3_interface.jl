#####
##### Microphysics interface implementation for P3
#####
##### These functions integrate the P3 scheme with AtmosphereModel,
##### allowing it to be used as a drop-in microphysics scheme.
#####

using Oceananigans: CenterField
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.AtmosphereModels: AtmosphereModels

using Breeze.Thermodynamics:
    MoistureMassFractions

const P3 = PredictedParticlePropertiesMicrophysics

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
function AtmosphereModels.prognostic_field_names(::P3)
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
function AtmosphereModels.specific_humidity(::P3, model)
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
function AtmosphereModels.materialize_microphysical_fields(::P3, grid, bcs)
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
##### Update microphysical fields
#####

"""
$(TYPEDSIGNATURES)

Update diagnostic microphysical fields after state update.

For P3, we compute vapor as the residual: qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ
"""
@inline function AtmosphereModels.update_microphysical_fields!(μ, ::P3, i, j, k, grid, ρ, 𝒰, constants)
    # Get total moisture from thermodynamic state
    qᵗ = 𝒰.moisture_mass_fractions.vapor + 𝒰.moisture_mass_fractions.liquid + 𝒰.moisture_mass_fractions.ice

    # Get condensate mass fractions from prognostic fields
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ  = @inbounds μ.ρqʳ[i, j, k] / ρ
    qⁱ  = @inbounds μ.ρqⁱ[i, j, k] / ρ
    qʷⁱ = @inbounds μ.ρqʷⁱ[i, j, k] / ρ

    # Vapor is residual
    qᵛ = max(0, qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ)

    @inbounds μ.qᵛ[i, j, k] = qᵛ
    return nothing
end

#####
##### Compute moisture fractions
#####

"""
$(TYPEDSIGNATURES)

Compute moisture mass fractions from P3 prognostic fields.

Returns `MoistureMassFractions` with vapor, liquid (cloud + rain), and ice components.
"""
@inline function AtmosphereModels.compute_moisture_fractions(i, j, k, grid, ::P3, ρ, qᵗ, μ)
    # Get condensate mass fractions
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ  = @inbounds μ.ρqʳ[i, j, k] / ρ
    qⁱ  = @inbounds μ.ρqⁱ[i, j, k] / ρ
    qʷⁱ = @inbounds μ.ρqʷⁱ[i, j, k] / ρ

    # Total liquid = cloud + rain + liquid on ice
    qˡ = qᶜˡ + qʳ + qʷⁱ

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
@inline AtmosphereModels.microphysical_velocities(p3::P3, μ, name) = nothing  # Default: no sedimentation

# Rain mass: mass-weighted fall speed
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρqʳ})
    return RainMassSedimentationVelocity(μ)
end

# Rain number: number-weighted fall speed
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρnʳ})
    return RainNumberSedimentationVelocity(μ)
end

# Ice mass: mass-weighted fall speed
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρqⁱ})
    return IceMassSedimentationVelocity(μ)
end

# Ice number: number-weighted fall speed
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρnⁱ})
    return IceNumberSedimentationVelocity(μ)
end

# Rime mass: same as ice mass (rime falls with ice)
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρqᶠ})
    return IceMassSedimentationVelocity(μ)
end

# Rime volume: same as ice mass
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρbᶠ})
    return IceMassSedimentationVelocity(μ)
end

# Ice reflectivity: reflectivity-weighted fall speed
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρzⁱ})
    return IceReflectivitySedimentationVelocity(μ)
end

# Liquid on ice: same as ice mass
@inline function AtmosphereModels.microphysical_velocities(p3::P3, μ, ::Val{:ρqʷⁱ})
    return IceMassSedimentationVelocity(μ)
end

#####
##### Sedimentation velocity types
#####
##### These are callable structs that compute terminal velocities at (i, j, k).
#####

"""
Callable struct for rain mass sedimentation velocity.
"""
struct RainMassSedimentationVelocity{M}
    microphysical_fields :: M
end

@inline function (v::RainMassSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        nʳ = μ.ρnʳ[i, j, k] / ρ
    end

    vₜ = rain_terminal_velocity_mass_weighted(qʳ, nʳ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for rain number sedimentation velocity.
"""
struct RainNumberSedimentationVelocity{M}
    microphysical_fields :: M
end

@inline function (v::RainNumberSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        nʳ = μ.ρnʳ[i, j, k] / ρ
    end

    vₜ = rain_terminal_velocity_number_weighted(qʳ, nʳ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for ice mass sedimentation velocity.
"""
struct IceMassSedimentationVelocity{M}
    microphysical_fields :: M
end

@inline function (v::IceMassSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields

    @inbounds begin
        qⁱ = μ.ρqⁱ[i, j, k] / ρ
        nⁱ = μ.ρnⁱ[i, j, k] / ρ
        qᶠ = μ.ρqᶠ[i, j, k] / ρ
        bᶠ = μ.ρbᶠ[i, j, k] / ρ
    end

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    vₜ = ice_terminal_velocity_mass_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for ice number sedimentation velocity.
"""
struct IceNumberSedimentationVelocity{M}
    microphysical_fields :: M
end

@inline function (v::IceNumberSedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields

    @inbounds begin
        qⁱ = μ.ρqⁱ[i, j, k] / ρ
        nⁱ = μ.ρnⁱ[i, j, k] / ρ
        qᶠ = μ.ρqᶠ[i, j, k] / ρ
        bᶠ = μ.ρbᶠ[i, j, k] / ρ
    end

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    vₜ = ice_terminal_velocity_number_weighted(qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

"""
Callable struct for ice reflectivity sedimentation velocity.
"""
struct IceReflectivitySedimentationVelocity{M}
    microphysical_fields :: M
end

@inline function (v::IceReflectivitySedimentationVelocity)(i, j, k, grid, ρ)
    FT = eltype(grid)
    μ = v.microphysical_fields

    @inbounds begin
        qⁱ = μ.ρqⁱ[i, j, k] / ρ
        nⁱ = μ.ρnⁱ[i, j, k] / ρ
        zⁱ = μ.ρzⁱ[i, j, k] / ρ
        qᶠ = μ.ρqᶠ[i, j, k] / ρ
        bᶠ = μ.ρbᶠ[i, j, k] / ρ
    end

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ, bᶠ, FT(400))

    vₜ = ice_terminal_velocity_reflectivity_weighted(qⁱ, nⁱ, zⁱ, Fᶠ, ρᶠ, ρ)

    return (u = zero(FT), v = zero(FT), w = -vₜ)
end

#####
##### Microphysical tendencies
#####

# Helper to compute P3 rates and extract ice properties
@inline function p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    FT = eltype(grid)

    # Compute all process rates
    rates = compute_p3_process_rates(i, j, k, grid, p3, μ, ρ, 𝒰, constants)

    # Extract fields for ratio calculations
    qⁱ = @inbounds μ.ρqⁱ[i, j, k] / ρ
    nⁱ = @inbounds μ.ρnⁱ[i, j, k] / ρ
    qᶠ = @inbounds μ.ρqᶠ[i, j, k] / ρ
    bᶠ = @inbounds μ.ρbᶠ[i, j, k] / ρ
    zⁱ = @inbounds μ.ρzⁱ[i, j, k] / ρ

    Fᶠ = safe_divide(qᶠ, qⁱ, zero(FT))
    ρᶠ = safe_divide(qᶠ * ρ, bᶠ * ρ, FT(400))

    return rates, qⁱ, nⁱ, zⁱ, Fᶠ, ρᶠ
end

"""
Cloud liquid tendency: loses mass to autoconversion, accretion, and riming.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρqᶜˡ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρqᶜˡ(rates, ρ)
end

"""
Rain mass tendency: gains from autoconversion, accretion, melting, shedding; loses to evaporation, riming.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρqʳ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρqʳ(rates, ρ)
end

"""
Rain number tendency: gains from autoconversion, melting, shedding; loses to self-collection, riming.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρnʳ}, ρ, μ, 𝒰, constants)
    rates, qⁱ, nⁱ, _, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρnʳ(rates, ρ, nⁱ, qⁱ)
end

"""
Ice mass tendency: gains from deposition, riming, refreezing; loses to melting.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρqⁱ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρqⁱ(rates, ρ)
end

"""
Ice number tendency: loses from melting and aggregation.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρnⁱ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρnⁱ(rates, ρ)
end

"""
Rime mass tendency: gains from cloud/rain riming, refreezing; loses proportionally with melting.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρqᶠ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, Fᶠ, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρqᶠ(rates, ρ, Fᶠ)
end

"""
Rime volume tendency: gains from new rime; loses with melting.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρbᶠ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, Fᶠ, ρᶠ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ)
end

"""
Ice sixth moment tendency: changes with deposition, melting, and riming.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρzⁱ}, ρ, μ, 𝒰, constants)
    rates, qⁱ, _, zⁱ, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρzⁱ(rates, ρ, qⁱ, zⁱ)
end

"""
Liquid on ice tendency: loses from shedding and refreezing.
"""
@inline function AtmosphereModels.microphysical_tendency(i, j, k, grid, p3::P3, ::Val{:ρqʷⁱ}, ρ, μ, 𝒰, constants)
    rates, _, _, _, _, _ = p3_rates_and_properties(i, j, k, grid, p3, μ, ρ, 𝒰, constants)
    return tendency_ρqʷⁱ(rates, ρ)
end

# Fallback for any unhandled field names - return zero tendency
@inline AtmosphereModels.microphysical_tendency(i, j, k, grid, ::P3, name, ρ, μ, 𝒰, constants) = zero(grid)

#####
##### Saturation adjustment
#####

"""
$(TYPEDSIGNATURES)

Apply saturation adjustment for P3.

P3 is a non-equilibrium scheme - cloud formation and dissipation are handled
by explicit process rates, not instantaneous saturation adjustment.
Therefore, this function returns the state unchanged.
"""
@inline function AtmosphereModels.maybe_adjust_thermodynamic_state(i, j, k, state, ::P3, ρᵣ, μ, qᵗ, thermo)
    # P3 is non-equilibrium: no saturation adjustment
    return state
end

#####
##### Model update
#####

"""
$(TYPEDSIGNATURES)

Apply P3 model update during state update phase.

Currently does nothing - this is where substepping or implicit updates would go.
"""
function AtmosphereModels.microphysics_model_update!(::P3, model)
    return nothing
end
