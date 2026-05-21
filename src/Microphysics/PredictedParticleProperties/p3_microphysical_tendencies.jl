using Oceananigans: CenterField
using Oceananigans.Fields: ZeroField
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze.Thermodynamics: MoistureMassFractions

using Breeze: Microphysics

const P3 = PredictedParticlePropertiesMicrophysics

"""
$(TYPEDSIGNATURES)

Cloud number tendency: gains from activation and loses proportionally with cloud sinks.

In the prescribed-Nᶜ path (`p3.aerosol === nothing`), `nc` is a scheme-level
parameter (Fortran `nccnst_2`), not a prognostic. The `ρnᶜˡ` field is still
allocated but carries no physical meaning, so the microphysical tendency is
zero and the field remains at its initial value.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    isnothing(p3.aerosol) && return zero(ρ)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnᶜˡ(rates, ρ, props.Nᶜ, ℳ.qᶜˡ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Cloud liquid tendency: loses mass to autoconversion, accretion, and riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶜˡ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Rain mass tendency: gains from autoconversion, accretion, melting, shedding; loses to evaporation, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʳ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Rain number tendency: gains from autoconversion, melting, shedding; loses to self-collection, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnʳ(rates, ρ, props.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Ice mass tendency: gains from deposition, riming, refreezing; loses to melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqⁱ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Ice number tendency: loses from melting and aggregation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnⁱ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Rime mass tendency: gains from cloud/rain riming, refreezing; loses proportionally with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶠ(rates, ρ, props.Fᶠ)
end

"""
$(TYPEDSIGNATURES)

Rime volume tendency: gains from new rime; loses with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρbᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρbᶠ(rates, ρ, props.Fᶠ, props.ρᶠ, ℳ.qⁱ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Ice sixth moment tendency: changes with deposition, melting, riming, and nucleation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρz̃ⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    # Convert physical z tendency to advected z̃ = √(z·N) tendency
    tendency_ρz_phys = p3_ice_sixth_moment_tendency(ice_integrals_table(p3), p3, rates, ρ, ℳ, props)
    tendency_ρn = tendency_ρnⁱ(rates, ρ)
    ρz̃ⁱ = ρ * sqrt(max(0, ℳ.zⁱ * props.nⁱ))
    return z̃ⁱ_tendency(props.nⁱ, props.zⁱ_bounded, tendency_ρz_phys, tendency_ρn,
                        ρz̃ⁱ, p3.process_rates.sink_limiting_timescale)
end

"""
$(TYPEDSIGNATURES)

Liquid on ice tendency: loses from shedding and refreezing.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʷⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʷⁱ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Supersaturation tendency: zero when predict_supersaturation = false.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρsˢᵃᵗ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Vapor tendency: loses from condensation, deposition, nucleation; gains from evaporation, sublimation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᵛ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᵛ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Aerosol number tendency: depletion equal to the cloud-droplet activation rate.
Zero in the prescribed-Nᶜ path.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnᵃ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnᵃ(rates, ρ)
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
@inline AM.maybe_adjust_thermodynamic_state(𝒰, ::P3, qᵛ, constants) = 𝒰
