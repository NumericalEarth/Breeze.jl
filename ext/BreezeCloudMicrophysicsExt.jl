module BreezeCloudMicrophysicsExt

using CloudMicrophysics: CloudMicrophysics
using CloudMicrophysics.Parameters: Parameters0M, Rain, Snow, CloudIce, CloudLiquid, CollisionEff
using CloudMicrophysics.Parameters: Blk1MVelType, Blk1MVelTypeRain, Blk1MVelTypeSnow
using CloudMicrophysics.Microphysics0M: remove_precipitation

using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    accretion

# Import Breeze modules needed for integration
using Breeze
using Breeze.AtmosphereModels
using Breeze.Thermodynamics: MoistureMassFractions
using Breeze.Microphysics: BulkMicrophysics, center_field_tuple
using Breeze

using Breeze.AtmosphereModels

using Breeze.Thermodynamics:
    MoistureMassFractions,
    density,
    with_moisture

using Breeze.Microphysics:
    center_field_tuple,
    BulkMicrophysics,
    FourCategories,
    WarmPhaseEquilibrium,
    SaturationAdjustment,
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

@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::ZMCM, μ, qᵗ, constants)
    # Initialize moisture state from total moisture qᵗ (not from stale microphysical fields)
    q₀ = MoistureMassFractions(qᵗ)
    𝒰₁ = with_moisture(𝒰₀, q₀)
    return adjust_thermodynamic_state(𝒰₁, bμp.nucleation, constants)
end

@inline function microphysical_tendency(i, j, k, grid, bμp::ZMCM, ::Val{:ρqᵗ}, μ, 𝒰, constants)
    # Get cloud liquid water from microphysical fields
    q = 𝒰.moisture_mass_fractions
    qˡ = q.liquid
    qⁱ = q.ice

    # remove_precipitation returns -dqᵗ/dt (rate of moisture removal)
    # Multiply by density to get the tendency for ρqᵗ
    # TODO: pass density into microphysical_tendency
    ρ = density(𝒰, constants)
    parameters_0M = bμp.categories

    return ρ * remove_precipitation(parameters_0M, qˡ, qⁱ)
end

"""
    ZeroMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                                τ_precip = 1000,
                                qc_0 = 5e-4,
                                S_0 = 0)

Return a `ZeroMomentCloudMicrophysics` microphysics scheme for warm-rain precipitation.

The zero-moment scheme removes cloud liquid water above a threshold at a specified rate:
- `τ_precip`: precipitation timescale in seconds (default: 1000 s)

and _either_

- `S_0`: supersaturation threshold (default: 0)
- `qc_0`: cloud liquid water threshold for precipitation (default: 5×10⁻⁴ kg/kg)

For more information see the
[`CloudMicrophysics.jl` documentation](https://clima.github.io/CloudMicrophysicsDocumentation.jl/dev/parameters/parameters0m/).
"""
function ZeroMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                     nucleation = SaturationAdjustment(FT),
                                     τ_precip = 1000,
                                     qc_0 = 5e-4,
                                     S_0 = 0)

    categories = Parameters0M{FT}(; τ_precip = FT(τ_precip),
                                    qc_0 = FT(qc_0),
                                    S_0 = FT(S_0))

    return BulkMicrophysics(nucleation, categories)
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
    collisions = CollisionEff(FT),
    hydrometeor_velocities = Blk1MVelType(FT))

    return FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities)
end

const CM1MCategories = FourCategories{<:CloudLiquid, <:CloudIce, <:Rain, <:Snow, <:CollisionEff, <:Blk1MVelType}
const OneMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM1MCategories}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories}

"""
    OneMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                               nucleation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium()),
                               categories = one_moment_cloud_microphysics_categories(FT))

Return a `OneMomentCloudMicrophysics` microphysics scheme for warm-rain and mixed-phase precipitation.

The one-moment scheme uses CloudMicrophysics.jl 1M processes:
- Autoconversion of cloud liquid to rain
- Accretion of cloud liquid by rain
- Terminal velocity for rain sedimentation

For warm-phase microphysics (the default), the prognostic variable is `ρqʳ` (rain mass density).
For mixed-phase microphysics, additional prognostic variable `ρqˢ` (snow mass density) is included.

See the [CloudMicrophysics.jl documentation](https://clima.github.io/CloudMicrophysics.jl/dev/) for details.
"""
function OneMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                    nucleation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium()),
                                    categories = one_moment_cloud_microphysics_categories(FT))
    return BulkMicrophysics(nucleation, categories)
end

prognostic_field_names(::WP1M) = tuple(:ρqʳ)
prognostic_field_names(::MP1M) = (:ρqʳ, :ρqˢ)

function materialize_microphysical_fields(bμp::WP1M, grid, bcs)
    names = (:qᵛ, :qˡ, :qᶜˡ, :ρqʳ)
    fields = center_field_tuple(grid, names...)
    return NamedTuple{names}(fields)
end

function materialize_microphysical_fields(bμp::MP1M, grid, bcs)
    names = (:qᵛ, :qˡ, :qᶜˡ, :qᶜⁱ, :ρqʳ, :ρqˢ)
    fields = center_field_tuple(grid, names...)
    return NamedTuple{names}(fields)
end

# Note: we perform saturation adjustment on vapor, total liquid, and total ice.
# This differs from the adjustment described in Yatunin et al 2025, wherein
# precipitating species are excluded from the adjustment.
# The reason we do this is because excluding precipitating species from adjustment requires
# a more complex algorithm in which precipitating species are passed into maybe_adjust_thermodynamic_state!
# We can consider changing this in the future.
@inline function update_microphysical_fields!(μ, bμp::WP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qˡ = 𝒰.moisture_mass_fractions.liquid  # cloud liquid from saturation adjustment

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qᶜˡ[i, j, k] = qˡ            # cloud liquid (non-precipitating)
        μ.qˡ[i, j, k] = qʳ + qˡ        # total liquid (cloud + rain)
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
        μ.qᶜˡ[i, j, k] = qˡ
        μ.qˡ[i, j, k] = qʳ + qˡ
        μ.qᶜⁱ[i, j, k] = qⁱ
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::WP1M, ρ, qᵗ, μ)
    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qᶜˡ = μ.qᶜˡ[i, j, k]
        qᵛ = μ.qᵛ[i, j, k]
    end

    qˡ = qᶜˡ + qʳ
    qⁱ = zero(qˡ)

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::MP1M, ρ, qᵗ, μ)
    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qˢ = μ.ρqˢ[i, j, k] / ρ
        qᶜˡ = μ.qᶜˡ[i, j, k]
        qᶜⁱ = μ.qᶜⁱ[i, j, k]
        qᵛ = μ.qᵛ[i, j, k]
    end

    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Microphysical tendencies for 1M warm-phase scheme
#####

# Default fallback for OneMomentCloudMicrophysics tendencies that are not explicitly implemented
@inline microphysical_tendency(i, j, k, grid, bμp::OneMomentCloudMicrophysics, args...) = zero(grid)

# Rain mass tendency (ρqʳ): autoconversion + accretion
# Note: ρqᵗ tendency is the negative of ρqʳ tendency (conservation of moisture)
@inline function microphysical_tendency(i, j, k, grid, bμp::WP1M, ::Val{:ρqʳ}, μ, 𝒰, constants)
    ρ = density(𝒰, constants)
    categories = bμp.categories

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]  # cloud liquid
    @inbounds qʳ = μ.ρqʳ[i, j, k] / ρ  # rain

    # Autoconversion: cloud liquid → rain
    acnv_rate = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    acc_rate = accretion(categories.cloud_liquid, categories.rain,
                         categories.hydrometeor_velocities.rain, categories.collisions,
                         qᶜˡ, qʳ, ρ)

    # Total tendency for ρqʳ (positive = rain increase)
    return ρ * (acnv_rate + acc_rate)
end

# Moisture tendency (ρqᵗ): loss to precipitation (currently zero since rain is tracked separately)
# TODO: add rain evaporation
@inline function microphysical_tendency(i, j, k, grid, bμp::WP1M, ::Val{:ρqᵗ}, μ, 𝒰, constants)
    return zero(grid)
end

# Default fallback for OneMomentCloudMicrophysics velocities
@inline microphysical_velocities(bμp::OneMomentCloudMicrophysics, name) = nothing

# TODO: Implement terminal velocity for rain sedimentation
# This requires building a velocity field from terminal_velocity(rain, vel.rain, ρ, qʳ)

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for one-moment bulk microphysics.

Saturation adjustment is performed on cloud moisture only, excluding precipitating
species (rain and snow). The precipitating moisture is then added back to the
final liquid/ice fractions.

This is required because:
1. Saturation adjustment represents fast vapor↔cloud condensate equilibration
2. Rain/snow represent slower precipitation processes that don't equilibrate instantly
3. Excluding rain/snow from adjustment prevents spurious evaporation of precipitation
"""
@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::WP1M, μ, qᵗ, constants)
    # Get rain mass fraction from prognostic microphysical field
    ρ = density(𝒰₀, constants)
    @inbounds ρqʳ = μ.ρqʳ[i, j, k]
    qʳ = ρqʳ / ρ
    
    # Compute cloud moisture (excluding rain)
    qᵗ_cloud = qᵗ - qʳ
    
    # Build moisture state for cloud-only adjustment
    q_cloud = MoistureMassFractions(qᵗ_cloud)
    𝒰_cloud = with_moisture(𝒰₀, q_cloud)
    
    # Perform saturation adjustment on cloud moisture only
    𝒰_adjusted = adjust_thermodynamic_state(𝒰_cloud, bμp.nucleation, constants)
    
    # Add rain back to the liquid fraction
    q_adj = 𝒰_adjusted.moisture_mass_fractions
    qᵛ = q_adj.vapor
    qˡ_total = q_adj.liquid + qʳ  # cloud liquid + rain
    q_final = MoistureMassFractions(qᵛ, qˡ_total)
    
    return with_moisture(𝒰_adjusted, q_final)
end

@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::MP1M, μ, qᵗ, constants)
    # Get rain and snow mass fractions from prognostic microphysical fields
    ρ = density(𝒰₀, constants)
    @inbounds ρqʳ = μ.ρqʳ[i, j, k]
    @inbounds ρqˢ = μ.ρqˢ[i, j, k]
    qʳ = ρqʳ / ρ
    qˢ = ρqˢ / ρ
    
    # Compute cloud moisture (excluding rain and snow)
    qᵗ_cloud = qᵗ - qʳ - qˢ
    
    # Build moisture state for cloud-only adjustment
    q_cloud = MoistureMassFractions(qᵗ_cloud)
    𝒰_cloud = with_moisture(𝒰₀, q_cloud)
    
    # Perform saturation adjustment on cloud moisture only
    𝒰_adjusted = adjust_thermodynamic_state(𝒰_cloud, bμp.nucleation, constants)
    
    # Add rain to liquid and snow to ice
    q_adj = 𝒰_adjusted.moisture_mass_fractions
    qᵛ = q_adj.vapor
    qˡ_total = q_adj.liquid + qʳ  # cloud liquid + rain
    qⁱ_total = q_adj.ice + qˢ     # cloud ice + snow
    q_final = MoistureMassFractions(qᵛ, qˡ_total, qⁱ_total)
    
    return with_moisture(𝒰_adjusted, q_final)
end

#####
##### Precipitation rate diagnostic for one-moment microphysics
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
    acnv_rate = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    acc_rate = accretion(categories.cloud_liquid, categories.rain,
                         categories.hydrometeor_velocities.rain, categories.collisions,
                         qᶜˡ, qʳ, ρ)

    # Total precipitation production rate (kg/kg/s)
    return acnv_rate + acc_rate
end

"""
    precipitation_rate(model, microphysics::OneMomentCloudMicrophysics, ::Val{:liquid})

Return a `Field` representing the liquid precipitation rate (rain production rate) in kg/kg/s.

For one-moment microphysics, this is the rate at which cloud liquid water
is converted to rain via autoconversion and accretion.
"""
function precipitation_rate(model, microphysics::WP1M, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = model.formulation.reference_state.density
    kernel = OneMomentPrecipitationRateKernel(microphysics.categories, qᶜˡ, ρqʳ, ρ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

# Ice precipitation not yet implemented for one-moment scheme
precipitation_rate(model, ::OneMomentCloudMicrophysics, ::Val{:ice}) = nothing

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
          "├── snow: ", prettysummary(bμp.categories.snow), "\n",
          "│   ├── acnv1M: ", prettysummary(bμp.categories.snow.acnv1M), '\n',
          "│   ├── area:   ", prettysummary(bμp.categories.snow.area), '\n',
          "│   ├── mass:   ", prettysummary(bμp.categories.snow.mass), '\n',
          "│   ├── r0:     ", prettysummary(bμp.categories.snow.r0), '\n',
          "│   ├── ρᵢ:     ", prettysummary(bμp.categories.snow.ρᵢ), '\n',
          "│   └── aspr:   ", prettysummary(bμp.categories.snow.aspr), '\n',
          "└── velocities: ", prettysummary(bμp.categories.hydrometeor_velocities))
end

end # module BreezeCloudMicrophysicsExt
