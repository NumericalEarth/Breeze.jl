#####
##### One-moment bulk microphysics (CloudMicrophysics 1M)
#####
# This file contains common code for all 1M schemes.
# Specific implementations are in:
#   - saturation_adjustment_1m.jl (WP1M, MP1M)
#   - nonequilibrium_1m.jl (WPNE1M)

function one_moment_cloud_microphysics_categories(
    FT::DataType = Oceananigans.defaults.FloatType;
    cloud_liquid = CloudLiquid(FT),
    cloud_ice = CloudIce(FT),
    rain = Rain(FT),
    snow = Snow(FT),
    collisions = CollisionEff(FT),
    hydrometeor_velocities = Blk1MVelType(FT),
    air_properties = AirProperties(FT))

    return FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities, air_properties)
end

const CM1MCategories = FourCategories{<:CloudLiquid, <:CloudIce, <:Rain, <:Snow, <:CollisionEff, <:Blk1MVelType, <:AirProperties}
const OneMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM1MCategories}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories}

"""
    OneMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                               cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing),
                               categories = one_moment_cloud_microphysics_categories(FT))

Return a `OneMomentCloudMicrophysics` microphysics scheme for warm-rain and mixed-phase precipitation.

The one-moment scheme uses CloudMicrophysics.jl 1M processes:
- Condensation/evaporation of cloud liquid (relaxation toward saturation)
- Autoconversion of cloud liquid to rain
- Accretion of cloud liquid by rain
- Terminal velocity for rain sedimentation

By default, non-equilibrium cloud formation is used, where cloud liquid is a prognostic
variable that evolves via condensation/evaporation tendencies following Morrison and
Milbrandt (2015). The prognostic variables are `ρqᶜˡ` (cloud liquid mass density) and
`ρqʳ` (rain mass density).

For equilibrium (saturation adjustment) cloud formation, pass:
```julia
cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
```

See the [CloudMicrophysics.jl documentation](https://clima.github.io/CloudMicrophysics.jl/dev/) for details.
"""
function OneMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing),
                                    categories = one_moment_cloud_microphysics_categories(FT))
    return BulkMicrophysics(cloud_formation, categories)
end

#####
##### Default fallbacks for OneMomentCloudMicrophysics
#####

# Default fallback for OneMomentCloudMicrophysics tendencies that are not explicitly implemented
@inline microphysical_tendency(i, j, k, grid, bμp::OneMomentCloudMicrophysics, args...) = zero(grid)

# Default fallback for OneMomentCloudMicrophysics velocities
@inline microphysical_velocities(bμp::OneMomentCloudMicrophysics, μ, name) = nothing

# Rain sedimentation: rain falls with terminal velocity (stored in microphysical fields)
@inline function microphysical_velocities(bμp::OneMomentCloudMicrophysics, μ, ::Val{:ρqʳ})
    wʳ = μ.wʳ
    return (; u = ZeroField(), v = ZeroField(), w = wʳ)
end

# Ice precipitation not yet implemented for one-moment scheme
precipitation_rate(model, ::OneMomentCloudMicrophysics, ::Val{:ice}) = nothing

#####
##### Precipitation rate kernel (shared by all 1M schemes)
#####

struct OneMomentPrecipitationRateKernel{C, QL, RR, RS}
    categories :: C
    cloud_liquid :: QL
    rain_density :: RR
    reference_density :: RS
end

Adapt.adapt_structure(to, k::OneMomentPrecipitationRateKernel) =
    OneMomentPrecipitationRateKernel(adapt(to, k.categories),
                                      adapt(to, k.cloud_liquid),
                                      adapt(to, k.rain_density),
                                      adapt(to, k.reference_density))

@inline function (k::OneMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    categories = k.categories
    @inbounds qᶜˡ = k.cloud_liquid[i, j, k_idx]
    @inbounds ρqʳ = k.rain_density[i, j, k_idx]
    @inbounds ρ = k.reference_density[i, j, k_idx]

    qʳ = ρqʳ / ρ

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Total precipitation production rate (kg/kg/s)
    return Sᵃᶜⁿᵛ + Sᵃᶜᶜ
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

prettysummary(vel::Blk1MVelType) = "Blk1MVelType(...)"
prettysummary(vel::Blk1MVelTypeRain) = "Blk1MVelTypeRain(...)"
prettysummary(vel::Blk1MVelTypeSnow) = "Blk1MVelTypeSnow(...)"

function prettysummary(ne::NonEquilibriumCloudFormation)
    liquid_str = isnothing(ne.liquid) ? "nothing" : "CloudLiquid(τ_relax=$(ne.liquid.τ_relax))"
    ice_str = isnothing(ne.ice) ? "nothing" : "CloudIce(τ_relax=$(ne.ice.τ_relax))"
    return "NonEquilibriumCloudFormation($liquid_str, $ice_str)"
end

function Base.show(io::IO, bμp::BulkMicrophysics{<:Any, <:CM1MCategories})
    print(io, summary(bμp), ":\n",
          "├── cloud_formation: ", prettysummary(bμp.cloud_formation), '\n',
          "├── collisions: ", prettysummary(bμp.categories.collisions), '\n',
          "├── cloud_liquid: ", prettysummary(bμp.categories.cloud_liquid), '\n',
          "├── cloud_ice: ", prettysummary(bμp.categories.cloud_ice), '\n',
          "├── rain: ", prettysummary(bμp.categories.rain), '\n',
          "│   ├── acnv1M: ", prettysummary(bμp.categories.rain.acnv1M), '\n',
          "│   ├── area:   ", prettysummary(bμp.categories.rain.area), '\n',
          "│   ├── vent:   ", prettysummary(bμp.categories.rain.vent), '\n',
          "│   └── pdf:    ", prettysummary(bμp.categories.rain.pdf), '\n',
          "├── snow: ", prettysummary(bμp.categories.snow), "\n",
          "│   ├── acnv1M: ", prettysummary(bμp.categories.snow.acnv1M), '\n',
          "│   ├── area:   ", prettysummary(bμp.categories.snow.area), '\n',
          "│   ├── mass:   ", prettysummary(bμp.categories.snow.mass), '\n',
          "│   ├── r0:     ", prettysummary(bμp.categories.snow.r0), '\n',
          "│   ├── ρᵢ:     ", prettysummary(bμp.categories.snow.ρᵢ), '\n',
          "│   └── aspr:   ", prettysummary(bμp.categories.snow.aspr), '\n',
          "└── velocities: ", prettysummary(bμp.categories.hydrometeor_velocities))
end
