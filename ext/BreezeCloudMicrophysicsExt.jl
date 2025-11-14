"""
Extension module for integrating CloudMicrophysics.jl schemes with Breeze.jl.

This extension provides integration between CloudMicrophysics.jl microphysics schemes
and Breeze.jl's microphysics interface, allowing CloudMicrophysics schemes to be used
with AtmosphereModel.

The extension is automatically loaded when CloudMicrophysics is available in the environment.
"""
module BreezeCloudMicrophysicsExt

using CloudMicrophysics
using CloudMicrophysics.Parameters: Parameters0M, Parameters1M
using CloudMicrophysics.Microphysics0M: remove_precipitation
using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    conv_q_icl_to_q_sno_no_supersat,
    accretion,
    evaporation_sublimation,
    snow_melt

# Import Breeze modules needed for integration
using ..Breeze
using ..Breeze.AtmosphereModels
using ..Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions
using ..Breeze.Microphysics: BulkMicrophysics, center_field_tuple

import ..Breeze.AtmosphereModels:
    compute_thermodynamic_state,
    prognostic_field_names,
    materialize_microphysical_fields,
    update_microphysical_fields!,
    moisture_mass_fractions

import ..Breeze.Thermodynamics:
    total_moisture_mass_fraction,
    with_moisture,
    MoistureMassFractions

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDSIGNATURES

#####
##### Zero-moment bulk microphysics (CloudMicrophysics 0M)
#####

"""
    ZeroMomentBulkMicrophysics

Type alias for `BulkMicrophysics` with CloudMicrophysics 0M precipitation scheme.

The 0M scheme instantly removes precipitable condensate above a threshold.
Interface is identical to non-precipitating microphysics except that
`compute_thermodynamic_state` calls CloudMicrophysics `remove_precipitation` first.
"""
const ZeroMomentBulkMicrophysics = BulkMicrophysics{<:Any, <:Parameters0M}
const ZMBM = ZeroMomentBulkMicrophysics
const ATC = AbstractThermodynamicState

prognostic_field_names(::ZMBM) = tuple()
materialize_microphysical_fields(bμp::ZMBM, grid, bcs) = materialize_microphysical_fields(bμp.clouds, grid, bcs)
@inline update_microphysical_fields!(μ, bμp::ZMBM, i, j, k, grid, 𝒰, thermo) = update_microphysical_fields!(μ, bμp.clouds, i, j, k, grid, 𝒰, thermo)
@inline moisture_mass_fractions(i, j, k, grid, bμp::ZMBM, μ, qᵗ) = moisture_mass_fractions(i, j, k, grid, bμp.clouds, μ, qᵗ)
@inline compute_thermodynamic_state(𝒰₀::ATC, bμp::ZMBM, thermo) = compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)
    
#####
##### One-moment bulk microphysics (CloudMicrophysics 1M)
#####

const OneMomentBulkMicrophysics = BulkMicrophysics{<:Any, <:Parameters1M}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:Parameters1M}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:Parameters1M}

prognostic_field_names(::WP1M) = (:qᵛ, :qᶜˡ, :qʳ)
prognostic_field_names(::MP1M) = (:qᵛ, :qᶜˡ, :qᶜⁱ, :qʳ, :qˢ)

function materialize_microphysical_fields(bμp::OneMomentBulkMicrophysics, grid, bcs)
    names = prognostic_field_names(bμp)
    fields = center_field_tuple(grid, names...)
    return NamedTuple{names}(fields)
end

@inline @inbounds function update_microphysical_fields!(μ, bμp::WP1M, i, j, k, grid, 𝒰, thermo)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qᴸ = 𝒰.moisture_mass_fractions.liquid
    qʳ = μ.qʳ[i, j, k]

    μ.qᵛ[i, j, k] = qᵛ
    μ.qᶜˡ[i, j, k] = qᴸ - qʳ

    return nothing
end

@inline @inbounds function update_microphysical_fields!(μ, bμp::MP1M, i, j, k, grid, 𝒰, thermo)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid
    qⁱ = 𝒰.moisture_mass_fractions.ice
    qʳ = μ.qʳ[i, j, k]
    qˢ = μ.qˢ[i, j, k]

    μ.qᵛ[i, j, k] = qᵛ
    μ.qᶜˡ[i, j, k] = qˡ - qʳ
    μ.qᶜⁱ[i, j, k] = qⁱ - qˢ

    return nothing
end

"""
$(TYPEDSIGNATURES)

Extract moisture mass fractions from microphysical fields for 1M scheme.
"""
@inline @inbounds function moisture_mass_fractions(i, j, k, grid, bμp::OMBM, μ, qᵗ)
    qᵛ = μ.qᵛ[i, j, k]
    qˡ = μ.qᶜˡ[i, j, k] + μ.qʳ[i, j, k] 
    qⁱ = μ.qᶜⁱ[i, j, k] + μ.qˢ[i, j, k]
    return MoistureMassFractions(qᵛ, qˡ, qᶜ)
end

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for one-moment bulk microphysics.

Delegates to clouds scheme (saturation adjustment) for vapor↔cloud conversion.
CloudMicrophysics 1M handles cloud↔precipitation processes via tendencies
computed in `update_microphysical_fields!`.
"""
@inline compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, bμp::OMBM, thermo) =
    compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)

end # module BreezeCloudMicrophysicsExt

