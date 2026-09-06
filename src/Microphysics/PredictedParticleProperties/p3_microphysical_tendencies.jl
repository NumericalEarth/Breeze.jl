using Oceananigans: CenterField
using Oceananigans.Fields: ZeroField

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze: Microphysics

const P3 = PredictedParticlePropertiesMicrophysics

#####
##### Gridless tendencies
#####
#
# All twelve tendencies come out of one process-rate evaluation. The prognostic loop calls
# `microphysical_tendencies`, so each parcel RHS evaluation computes the bundle once; the per-name
# methods below evaluate it once per name asked for.

@inline function AM.microphysical_tendencies(p3::P3, names::Tuple, ρ,
                                             ℳ::P3MicrophysicalState, 𝒰, constants)
    result = p3_state_tendencies(p3, ρ, ℳ, 𝒰, constants)
    return ntuple(i -> p3_tendency_component(result, Val(names[i])), Val(length(names)))
end

@inline p3_single_tendency(p3::P3, name, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_tendency_component(p3_state_tendencies(p3, ρ, ℳ, 𝒰, constants), name)

"""
$(TYPEDSIGNATURES)

Cloud number tendency: gains from activation and loses proportionally with cloud sinks.

In the prescribed-Nᶜˡ path (`p3.aerosol === nothing`), the droplet number is a
scheme-level parameter, not a prognostic. `ρnᶜˡ` is neither allocated nor
transported there, and every rate takes the prescribed value from
[`effective_cloud_droplet_number`](@ref), so this tendency is zero and is never
reached through the prognostic loop.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρnᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Cloud liquid tendency: loses mass to autoconversion, accretion, and riming.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρqᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Rain mass tendency: gains from autoconversion, accretion, melting, shedding; loses to evaporation, riming.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρqʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Rain number tendency: gains from autoconversion, melting, shedding; loses to self-collection, riming.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρnʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Ice mass tendency: gains from deposition, riming, refreezing; loses to melting.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρqⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Ice number tendency: loses from melting and aggregation.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρnⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Rime mass tendency: gains from cloud/rain riming, refreezing; loses proportionally with melting.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρqᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Rime volume tendency: gains from new rime; loses with melting.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρbᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Liquid on ice tendency: loses from shedding and refreezing.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρqʷⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Supersaturation tendency: zero when `predict_supersaturation = false`.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρsᵛ⁺ˡ}, ρ,
                                  ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Vapor tendency: loses from condensation, deposition, nucleation; gains from evaporation, sublimation.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρqᵛ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

"""
$(TYPEDSIGNATURES)

Aerosol number tendency: depletion equal to the cloud-droplet activation rate.
Zero in the prescribed-Nᶜˡ path.
"""
@inline AM.microphysical_tendency(p3::P3, name::Val{:ρnᵃ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) =
    p3_single_tendency(p3, name, ρ, ℳ, 𝒰, constants)

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
@inline AM.maybe_adjust_thermodynamic_state(𝒰, ::P3, qᵛ, constants) = 𝒰
