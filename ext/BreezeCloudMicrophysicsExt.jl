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
@inline update_microphysical_fields!(μ, bμp::ZMBM, i, j, k, grid, density, 𝒰, thermo) = update_microphysical_fields!(μ, bμp.clouds, i, j, k, grid, density, 𝒰, thermo)
@inline moisture_mass_fractions(i, j, k, grid, bμp::ZMBM, density, qᵗ, μ) = moisture_mass_fractions(i, j, k, grid, bμp.clouds, density, qᵗ, μ)
@inline compute_thermodynamic_state(𝒰₀::ATC, bμp::ZMBM, thermo) = compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)
    
#####
##### One-moment bulk microphysics (CloudMicrophysics 1M)
#####

const OneMomentBulkMicrophysics = BulkMicrophysics{<:Any, <:Parameters1M}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:Parameters1M}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:Parameters1M}

prognostic_field_names(::WP1M) = tuple(:ρqʳ)
prognostic_field_names(::MP1M) = (:ρqʳ, :ρqˢ)

function materialize_microphysical_fields(bμp::WP1M, grid, bcs)
    names = (:qᵛ, :qᶜˡ, :ρqʳ)
    fields = center_field_tuple(grid, names...)
    return NamedTuple{names}(fields)
end

function materialize_microphysical_fields(bμp::MP1M, grid, bcs)
    names = (:qᵛ, :qᶜˡ, :qᶜⁱ, :ρqʳ, :ρqˢ)
    fields = center_field_tuple(grid, names...)
    return NamedTuple{names}(fields)
end

# Note: we perform saturation adjustment on vapor, total liquid, and total ice.
# This differs from the adjustment described in Yatunin et al 2025, wherein
# precipitating species are excluded from the adjustment.
# The reason we do this is because excluding precipiating species from adjustment requires
# a more complex algorithm in which precipitating species are passed into compute_thermodynamic_state!
# We can consider changing this in the future.
@inline @inbounds function update_microphysical_fields!(μ, bμp::WP1M, i, j, k, grid, density, 𝒰, thermo)
    ρ = density[i, j, k]
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid
    qʳ = μ.ρqʳ[i, j, k] / ρ

    μ.qᵛ[i, j, k] = qᵛ
    μ.qˡ[i, j, k] = qʳ + qˡ

    return nothing
end

@inline @inbounds function update_microphysical_fields!(μ, bμp::MP1M, i, j, k, grid, density, 𝒰, thermo)
    ρ = density[i, j, k]
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid
    qⁱ = 𝒰.moisture_mass_fractions.ice
    qʳ = μ.ρqʳ[i, j, k] / ρ
    qˢ = μ.ρqˢ[i, j, k] / ρ

    μ.qᵛ[i, j, k] = qᵛ
    μ.qᶜˡ[i, j, k] = qʳ + qˡ
    μ.qᶜⁱ[i, j, k] = qˢ + qⁱ

    return nothing
end

"""
$(TYPEDSIGNATURES)

Extract moisture mass fractions from microphysical fields for 1M scheme.
"""
@inline @inbounds function moisture_mass_fractions(i, j, k, grid, bμp::OneMomentBulkMicrophysics, density, qᵗ, μ)
    ρ = density[i, j, k]
    ρqʳ = μ.ρqʳ[i, j, k] / ρ
    ρqˢ = μ.ρqˢ[i, j, k] / ρ
    qᶜˡ = μ.qᶜˡ[i, j, k]
    qᶜⁱ = μ.qᶜⁱ[i, j, k]

    qᵛ = μ.qᵛ[i, j, k]
    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for one-moment bulk microphysics.

Delegates to clouds scheme (saturation adjustment) for vapor↔cloud conversion.
CloudMicrophysics 1M handles cloud↔precipitation processes via tendencies
computed in `update_microphysical_fields!`.
"""
@inline function compute_thermodynamic_state(𝒰₀::AbstractThermodynamicState, bμp::OneMomentBulkMicrophysics, thermo)
    return compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)
end

end # module BreezeCloudMicrophysicsExt

