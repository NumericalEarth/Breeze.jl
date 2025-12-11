module BreezeCloudMicrophysicsExt

using CloudMicrophysics: CloudMicrophysics
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
using Breeze.Thermodynamics: MoistureMassFractions
using Breeze.Microphysics: BulkMicrophysics, center_field_tuple
using Breeze

using Breeze.AtmosphereModels

using Breeze.Thermodynamics:
    MoistureMassFractions,
    saturation_specific_humidity,
    temperature,
    density

using Breeze.Microphysics:
    center_field_tuple,
    equilibrated_surface,
    BulkMicrophysics,
    FourCategories,
    WarmPhaseSaturationAdjustment,
    MixedPhaseSaturationAdjustment,
    adjust_thermodynamic_state

using Oceananigans: Oceananigans
using DocStringExtensions: TYPEDSIGNATURES

using Oceananigans: Center, Field
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Adapt: Adapt, adapt

import Breeze.AtmosphereModels:
    maybe_adjust_thermodynamic_state,
    prognostic_field_names,
    materialize_microphysical_fields,
    update_microphysical_fields!,
    compute_moisture_fractions,
    microphysical_tendency,
    microphysical_velocities,
    precipitation_rate

#####
##### Zero-moment bulk microphysics (CloudMicrophysics 0M)
#####

"""
    ZeroMomentBulkMicrophysics

Type alias for `BulkMicrophysics` with CloudMicrophysics 0M precipitation scheme.

The 0M scheme instantly removes precipitable condensate above a threshold.
Interface is identical to non-precipitating microphysics except that
`maybe_adjust_thermodynamic_state` calls CloudMicrophysics `remove_precipitation` first.
"""
const ZeroMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:Parameters0M}
const ZMCM = ZeroMomentCloudMicrophysics

prognostic_field_names(::ZMCM) = tuple()
materialize_microphysical_fields(bμp::ZMCM, grid, bcs) = materialize_microphysical_fields(bμp.nucleation, grid, bcs)
@inline update_microphysical_fields!(μ, bμp::ZMCM, i, j, k, grid, ρ, 𝒰, constants) = update_microphysical_fields!(μ, bμp.nucleation, i, j, k, grid, ρ, 𝒰, constants)
@inline compute_moisture_fractions(i, j, k, grid, bμp::ZMCM, ρ, qᵗ, μ) = compute_moisture_fractions(i, j, k, grid, bμp.nucleation, ρ, qᵗ, μ)
@inline microphysical_tendency(i, j, k, grid, bμp::ZMCM, args...) = zero(grid)
@inline microphysical_velocities(bμp::ZMCM, name) = nothing
@inline maybe_adjust_thermodynamic_state(𝒰₀, bμp::ZMCM, μ, qᵗ, constants) = adjust_thermodynamic_state(𝒰₀, bμp.nucleation, constants)

@inline function microphysical_tendency(i, j, k, grid, bμp::ZMCM, ::Val{:ρqᵗ}, μ, 𝒰, constants)
    # Get cloud liquid water from microphysical fields
    @inbounds qˡ = μ.qˡ[i, j, k]

    # Warm-phase only: no ice
    qⁱ = zero(qˡ)

    # remove_precipitation returns -dqᵗ/dt (rate of moisture removal)
    # Multiply by density to get the tendency for ρqᵗ
    ρ = density(𝒰, constants)
    return ρ * remove_precipitation(bμp.categories, qˡ, qⁱ)
end

"""
    ZeroMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                                τ_precip = 1000,
                                qc_0 = 5e-4,
                                S_0 = 0)

Return a `ZeroMomentCloudMicrophysics` microphysics scheme for warm-rain precipitation.

The zero-moment scheme removes cloud liquid water above a threshold at a specified rate:
- `τ_precip`: precipitation timescale in seconds (default: 1000 s)
- `qc_0`: cloud liquid water threshold for precipitation (default: 5×10⁻⁴ kg/kg)
- `S_0`: supersaturation threshold (default: 0)

The precipitation rate is computed as:
```math
\\frac{dq^t}{dt} = -\\frac{1}{τ} \\max(0, q^ˡ - q_c)
```
where `τ = τ_precip` and `q_c = qc_0`.
"""
function ZeroMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                     τ_precip = 1000,
                                     qc_0 = 5e-4,
                                     S_0 = 0)
    categories = Parameters0M{FT}(; τ_precip = FT(τ_precip),
                                    qc_0 = FT(qc_0),
                                    S_0 = FT(S_0))
    return BulkMicrophysics(SaturationAdjustment(FT), categories)
end

#####
##### Precipitation rate diagnostic for zero-moment microphysics
#####

struct ZeroMomentPrecipitationRateKernel{C, Q}
    categories :: C
    cloud_liquid :: Q
end

Adapt.adapt_structure(to, k::ZeroMomentPrecipitationRateKernel) =
    ZeroMomentPrecipitationRateKernel(adapt(to, k.categories),
                                       adapt(to, k.cloud_liquid))

@inline function (k::ZeroMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    @inbounds qˡ = k.cloud_liquid[i, j, k_idx]
    # Warm-phase only: no ice
    qⁱ = zero(qˡ)
    # remove_precipitation returns dqᵗ/dt (negative = moisture removal = precipitation)
    # We return positive precipitation rate (kg/kg/s)
    return -remove_precipitation(k.categories, qˡ, qⁱ)
end

"""
    precipitation_rate(model, microphysics::ZeroMomentCloudMicrophysics, ::Val{:liquid})

Return a `Field` representing the liquid precipitation rate (rain rate) in kg/kg/s.

For zero-moment microphysics, this is the rate at which cloud liquid water
is removed by precipitation: `-dqᵗ/dt` from the `remove_precipitation` function.
"""
function precipitation_rate(model, microphysics::ZMCM, ::Val{:liquid})
    grid = model.grid
    qˡ = model.microphysical_fields.qˡ
    kernel = ZeroMomentPrecipitationRateKernel(microphysics.categories, qˡ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

# Ice precipitation not supported for zero-moment warm-phase scheme
precipitation_rate(model, ::ZMCM, ::Val{:ice}) = nothing

#####
##### One-moment bulk microphysics (CloudMicrophysics 1M)
#####

function one_moment_cloud_microphysics_categories(
    FT::DataType = Oceananigans.defaults.FloatType;
    cloud_liquid = CloudLiquid(FT),
    cloud_ice = CloudIce(FT),
    rain = Rain(FT),
    snow = Snow(FT),
    collisions = CollisionEff(FT))

    return FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions)
end

const CM1MCategories = FourCategories{<:CloudLiquid, <:CloudIce, <:Rain, <:Snow, <:CollisionEff}
const OneMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM1MCategories}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories}

"""
$(TYPEDSIGNATURES)

Return a `OneMomentCloudMicrophysics` microphysics scheme with four `categories`.
"""
function OneMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType,
                                    categories = one_moment_cloud_microphysics_categories(FT))
    return BulkMicrophysics(SaturationAdjustment(FT), categories)
end

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
# a more complex algorithm in which precipitating species are passed into maybe_adjust_thermodynamic_state!
# We can consider changing this in the future.
@inline function update_microphysical_fields!(μ, bμp::WP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qˡ[i, j, k] = qʳ + qˡ
    end

    return nothing
end

@inline function update_microphysical_fields!(μ, bμp::MP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid
    qⁱ = 𝒰.moisture_mass_fractions.ice

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qˢ = μ.ρqˢ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qᶜˡ[i, j, k] = qʳ + qˡ
        μ.qᶜⁱ[i, j, k] = qˢ + qⁱ
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::MP1M, ρ, qᵗ, μ)
    @inbounds begin
        ρqʳ = μ.ρqʳ[i, j, k] / ρ
        ρqˢ = μ.ρqˢ[i, j, k] / ρ
        qᶜˡ = μ.qᶜˡ[i, j, k]
        qᶜⁱ = μ.qᶜⁱ[i, j, k]
        qᵛ = μ.qᵛ[i, j, k]
    end

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
@inline maybe_adjust_thermodynamic_state(𝒰₀, bμp::OneMomentCloudMicrophysics, microphysical_fields, qᵗ, constants) =
    maybe_adjust_thermodynamic_state(𝒰₀, bμp.nucleation, microphysical_fields, qᵗ, constants)

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
