"""
Extension module for integrating CloudMicrophysics.jl schemes with Breeze.jl.

This extension provides integration between CloudMicrophysics.jl microphysics schemes
and Breeze.jl's microphysics interface, allowing CloudMicrophysics schemes to be used
with AtmosphereModel.

The extension is automatically loaded when CloudMicrophysics is available in the environment.
"""
module BreezeCloudMicrophysicsExt

using CloudMicrophysics
using CloudMicrophysics.Parameters: Parameters0M, Rain, Snow, CloudIce, CloudLiquid, CollisionEff
using CloudMicrophysics.Microphysics0M: remove_precipitation

#=
using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    conv_q_icl_to_q_sno_no_supersat,
    accretion,
    evaporation_sublimation,
    snow_melt
=#

# Import Breeze modules needed for integration
using Breeze
using Breeze.AtmosphereModels
using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions
using Breeze.Microphysics: BulkMicrophysics, center_field_tuple
using Breeze

using Breeze.AtmosphereModels
using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions
using Breeze.Microphysics:
    center_field_tuple,
    BulkMicrophysics,
    WarmPhaseSaturationAdjustment,
    MixedPhaseSaturationAdjustment

import Breeze.AtmosphereModels:
    compute_thermodynamic_state,
    prognostic_field_names,
    materialize_microphysical_fields,
    update_microphysical_fields!,
    moisture_mass_fractions

import Breeze.Thermodynamics:
    total_moisture_mass_fraction,
    with_moisture,
    MoistureMassFractions

import Breeze.Microphysics: FourCategories

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
@inline moisture_mass_fractions(i, j, k, grid, bμp::ZMBM, ρ, qᵗ, μ) = moisture_mass_fractions(i, j, k, grid, bμp.nucleation, ρ, qᵗ, μ)
@inline compute_thermodynamic_state(𝒰₀::ATC, bμp::ZMBM, thermo) = compute_thermodynamic_state(𝒰₀, bμp.clouds, thermo)
    
#####
##### One-moment bulk microphysics (CloudMicrophysics 1M)
#####

function FourCategories(FT::DataType = Oceananigans.defaults.FloatType;
                        cloud_liquid = CloudLiquid(FT),
                        cloud_ice = CloudIce(FT),
                        rain = Rain(FT),
                        snow = Snow(FT),
                        collisions = CollisionEff(FT))

    return FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions)
end

const CM1MCategories = FourCategories{<:CloudLiquid, <:CloudIce, <:Rain, <:Snow, <:CollisionEff}
const OneMomentBulkMicrophysics = BulkMicrophysics{<:Any, <:CM1MCategories}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories}

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

@inline @inbounds function moisture_mass_fractions(i, j, k, grid, bμp::MP1M, density, qᵗ, μ)
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

#####
##### show methods
#####

import Oceananigans.Utils: prettysummary

function prettysummary(cl::CloudLiquid)
    return string("CloudLiquid(",
                  "ρw=", prettysummary(cl.ρw), ", ",
                  "r_eff=", prettysummary(cl.r_eff), ", ",
                  "τ_relax=", prettysummary(cl.τ_relax))
end

function prettysummary(ci::CloudIce)
    return string("CloudIce(",
                  "r0=", prettysummary(ci.r0), ", ",
                  "r_eff=", prettysummary(ci.r_eff), ", ",
                  "ρᵢ=", prettysummary(ci.ρᵢ), ", ",
                  "r_ice_snow=", prettysummary(ci.r_ice_snow), ", ",
                  "τ_relax=", prettysummary(ci.τ_relax), ", ",
                  "mass=", prettysummary(ci.mass), ", ",
                  "pdf=", prettysummary(ci.pdf), ")")
end

function prettysummary(mass::CloudMicrophysics.Parameters.ParticleMass)
    return string("ParticleMass(",
                  "r0=", prettysummary(mass.r0), ", ",
                  "m0=", prettysummary(mass.m0), ", ",
                  "me=", prettysummary(mass.me), ", ",
                  "Δm=", prettysummary(mass.Δm), ", ",
                  "χm=", prettysummary(mass.χm), ")")
end
    
function prettysummary(pdf::CloudMicrophysics.Parameters.ParticlePDFIceRain)
    return string("ParticlePDFIceRain(n0=", prettysummary(pdf.n0), ")")
end

function prettysummary(eff::CloudMicrophysics.Parameters.CollisionEff)
    return string("CollisionEff(",
                  "e_lcl_rai=", prettysummary(eff.e_lcl_rai), ", ",
                  "e_lcl_sno=", prettysummary(eff.e_lcl_sno), ", ",
                  "e_icl_rai=", prettysummary(eff.e_icl_rai), ", ",
                  "e_icl_sno=", prettysummary(eff.e_icl_sno), ", ",
                  "e_rai_sno=", prettysummary(eff.e_rai_sno), ")")
end

prettysummary(rain::CloudMicrophysics.Parameters.Rain) = "CloudMicrophysics.Parameters.Rain"
prettysummary(snow::CloudMicrophysics.Parameters.Snow) = "CloudMicrophysics.Parameters.Snow"

#=
function prettysummary(rain::CloudMicrophysics.Parameters.Rain)
    return string("Rain(",
                  "acnv1M=", prettysummary(rain.acnv1M), ", ",
                  "area=", prettysummary(rain.area), ", ",
                  "vent=", prettysummary(rain.vent), ", ",
                  "r0=", prettysummary(rain.r0), ", ",
                  "mass=", prettysummary(rain.mass), ", ",
                  "pdf=", prettysummary(rain.pdf), ")")
end
=#

function prettysummary(acnv::CloudMicrophysics.Parameters.Acnv1M)
    return string("Acnv1M(",
                  "τ=", prettysummary(acnv.τ), ", ",
                  "q_threshold=", prettysummary(acnv.q_threshold), ", ",
                  "k=", prettysummary(acnv.k), ")")
end

function prettysummary(area::CloudMicrophysics.Parameters.ParticleArea)
    return string("ParticleArea(",
                  "a0=", prettysummary(area.a0), ", ",
                  "ae=", prettysummary(area.ae), ", ",
                  "Δa=", prettysummary(area.Δa), ", ",
                  "χa=", prettysummary(area.χa), ")")
end

function prettysummary(vent::CloudMicrophysics.Parameters.Ventilation)
    return string("Ventilation(",
                  "a=", prettysummary(vent.a), ", ",
                  "b=", prettysummary(vent.b), ")")
end

function prettysummary(aspr::CloudMicrophysics.Parameters.SnowAspectRatio)
    return string("SnowAspectRatio(",
                  "ϕ=", prettysummary(aspr.ϕ), ", ",
                  "κ=", prettysummary(aspr.κ), ")")
end

function Base.show(io::IO, bμp::BulkMicrophysics{<:Any, <:CM1MCategories})
    print(io, summary(bμp), ":\n",
          "├── nucleation: ", prettysummary(bμp.nucleation), '\n',
          "├── collisions: ", prettysummary(bμp.categories.collisions), '\n',
          "├── cloud_liquid: ", prettysummary(bμp.categories.cloud_liquid), '\n',
          "├── cloud_ice: ", prettysummary(bμp.categories.cloud_ice), '\n',
          "├── rain: ", prettysummary(bμp.categories.rain), '\n',
          "│   ├── acnv1M: ", prettysummary(bμp.categories.rain.acnv1M), '\n',
          "│   ├── area:   ", prettysummary(bμp.categories.rain.area), '\n',
          "│   ├── vent:   ", prettysummary(bμp.categories.rain.vent), '\n',
          "│   └── pdf:    ", prettysummary(bμp.categories.rain.pdf), '\n',
          "└── snow: ", prettysummary(bμp.categories.snow), "\n",
          "    ├── acnv1M: ", prettysummary(bμp.categories.snow.acnv1M), '\n',
          "    ├── area:   ", prettysummary(bμp.categories.snow.area), '\n',
          "    ├── mass:   ", prettysummary(bμp.categories.snow.mass), '\n',
          "    ├── r0:     ", prettysummary(bμp.categories.snow.r0), '\n',
          "    ├── ρᵢ:     ", prettysummary(bμp.categories.snow.ρᵢ), '\n',
          "    └── aspr:   ", prettysummary(bμp.categories.snow.aspr))
end


end # module BreezeCloudMicrophysicsExt

