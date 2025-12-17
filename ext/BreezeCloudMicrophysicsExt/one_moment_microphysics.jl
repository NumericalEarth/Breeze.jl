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
    hydrometeor_velocities = Blk1MVelType(FT),
    air_properties = AirProperties(FT))

    return FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities, air_properties)
end

const CM1MCategories = FourCategories{<:CloudLiquid, <:CloudIce, <:Rain, <:Snow, <:CollisionEff, <:Blk1MVelType, <:AirProperties}
const OneMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM1MCategories}
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories}
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories}

# Non-equilibrium cloud formation with 1M precipitation (warm-phase only for now)
const WarmPhaseNonEquilibrium1M = BulkMicrophysics{<:NonEquilibriumCloudFormation{<:CloudLiquid, Nothing}, <:CM1MCategories}
const WPNE1M = WarmPhaseNonEquilibrium1M

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
##### Warm-phase saturation adjustment 1M (WP1M)
#####

prognostic_field_names(::WP1M) = tuple(:ρqʳ)

function materialize_microphysical_fields(bμp::WP1M, grid, bcs)
    center_names = (:qᵛ, :qˡ, :qᶜˡ, :qʳ, :ρqʳ)
    center_fields = center_field_tuple(grid, center_names...)
    wʳ = ZFaceField(grid)  # Rain terminal velocity (negative = downward)
    return (; zip(center_names, center_fields)..., wʳ)
end

# Note: we perform saturation adjustment on vapor, total liquid, and total ice.
# This differs from the adjustment described in Yatunin et al 2025, wherein
# precipitating species are excluded from the adjustment.
# The reason we do this is because excluding precipitating species from adjustment requires
# a more complex algorithm in which precipitating species are passed into maybe_adjust_thermodynamic_state!
# We can consider changing this in the future.
@inline function update_microphysical_fields!(μ, bμp::WP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qᶜˡ = 𝒰.moisture_mass_fractions.liquid  # cloud liquid from saturation adjustment
    categories = bμp.categories

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qʳ[i, j, k] = qʳ             # rain mass fraction (diagnostic)
        μ.qᶜˡ[i, j, k] = qᶜˡ           # cloud liquid (non-precipitating)
        μ.qˡ[i, j, k] = qʳ + qᶜˡ       # total liquid (cloud + rain)

        # Terminal velocity for rain (negative = downward)
        wᵗ = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
        μ.wʳ[i, j, k] = -wᵗ
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::WP1M, ρ, qᵗ, μ)
    @inbounds begin
        qʳ = μ.qʳ[i, j, k]
        qᶜˡ = μ.qᶜˡ[i, j, k]
        qᵛ = μ.qᵛ[i, j, k]
    end

    qˡ = qᶜˡ + qʳ
    qⁱ = zero(qˡ)

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

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
@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::WP1M, ρᵣ, μ, qᵗ, constants)
    # Get rain mass fraction from diagnostic microphysical field
    @inbounds qʳ = μ.ρqʳ[i, j, k] / ρᵣ
    
    # Compute cloud moisture (excluding rain)
    qᵗᶜ = qᵗ - qʳ
    
    # Build moisture state for cloud-only adjustment
    qᶜ = MoistureMassFractions(qᵗᶜ)
    𝒰ᶜ = with_moisture(𝒰₀, qᶜ)
    
    # Perform saturation adjustment on cloud moisture only
    𝒰′ = adjust_thermodynamic_state(𝒰ᶜ, bμp.cloud_formation, constants)
    
    # Add rain back to the liquid fraction
    q′ = 𝒰′.moisture_mass_fractions
    qᵛ = q′.vapor
    qˡ = q′.liquid + qʳ  # cloud liquid + rain
    q = MoistureMassFractions(qᵛ, qˡ)
    
    return with_moisture(𝒰′, q)
end

# Rain mass tendency (ρqʳ): autoconversion + accretion
# Note: ρqᵗ tendency is the negative of ρqʳ tendency (conservation of moisture)
@inline function microphysical_tendency(i, j, k, grid, bμp::WP1M, ::Val{:ρqʳ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    ρⁱʲᵏ = @inbounds ρ[i, j, k]

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]  # cloud liquid
    @inbounds qʳ = μ.qʳ[i, j, k]    # rain

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Total tendency for ρqʳ (positive = rain increase)
    return ρⁱʲᵏ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ)
end

# Moisture tendency (ρqᵗ): loss to precipitation (currently zero since rain is tracked separately)
# TODO: add rain evaporation
@inline function microphysical_tendency(i, j, k, grid, bμp::WP1M, ::Val{:ρqᵗ}, ρ, μ, 𝒰, constants)
    return zero(grid)
end

#####
##### Mixed-phase saturation adjustment 1M (MP1M)
#####

prognostic_field_names(::MP1M) = (:ρqʳ, :ρqˢ)

function materialize_microphysical_fields(bμp::MP1M, grid, bcs)
    center_names = (:qᵛ, :qˡ, :qᶜˡ, :qᶜⁱ, :qʳ, :qˢ, :ρqʳ, :ρqˢ)
    center_fields = center_field_tuple(grid, center_names...)
    wʳ = ZFaceField(grid)  # Rain terminal velocity (negative = downward)
    return (; zip(center_names, center_fields)..., wʳ)
end

@inline function update_microphysical_fields!(μ, bμp::MP1M, i, j, k, grid, ρ, 𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor
    qᶜˡ = 𝒰.moisture_mass_fractions.liquid
    qᶜⁱ = 𝒰.moisture_mass_fractions.ice
    categories = bμp.categories

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qˢ = μ.ρqˢ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = qᵛ
        μ.qʳ[i, j, k] = qʳ             # rain mass fraction (diagnostic)
        μ.qˢ[i, j, k] = qˢ             # snow mass fraction (diagnostic)
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qˡ[i, j, k] = qʳ + qᶜˡ
        μ.qᶜⁱ[i, j, k] = qᶜⁱ

        # Terminal velocity for rain (negative = downward)
        wᵗ = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
        μ.wʳ[i, j, k] = -wᵗ
    end

    return nothing
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

@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::MP1M, ρᵣ, μ, qᵗ, constants)
    # Get rain and snow mass fractions from diagnostic microphysical fields
    @inbounds qʳ = μ.ρqʳ[i, j, k] / ρᵣ   
    @inbounds qˢ = μ.ρqˢ[i, j, k] / ρᵣ
    
    # Compute cloud moisture (excluding rain and snow)
    qᵗᶜ = qᵗ - qʳ - qˢ
    
    # Build moisture state for cloud-only adjustment
    qᶜ = MoistureMassFractions(qᵗᶜ)
    𝒰ᶜ = with_moisture(𝒰₀, qᶜ)
    
    # Perform saturation adjustment on cloud moisture only
    𝒰′ = adjust_thermodynamic_state(𝒰ᶜ, bμp.cloud_formation, constants)
    
    # Add rain to liquid and snow to ice
    q′ = 𝒰′.moisture_mass_fractions
    qᵛ = q′.vapor
    qˡ = q′.liquid + qʳ  # cloud liquid + rain
    qⁱ = q′.ice + qˢ     # cloud ice + snow
    q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
    
    return with_moisture(𝒰′, q)
end

#####
##### Non-equilibrium 1M microphysics (warm-phase)
#####
# Cloud liquid is prognostic and evolves via condensation/evaporation tendencies
# following Morrison and Milbrandt (2015) relaxation formulation.

prognostic_field_names(::WPNE1M) = (:ρqᶜˡ, :ρqʳ)

function materialize_microphysical_fields(bμp::WPNE1M, grid, bcs)
    center_names = (:qᵛ, :qˡ, :qᶜˡ, :qʳ, :ρqᶜˡ, :ρqʳ)
    center_fields = center_field_tuple(grid, center_names...)
    wʳ = ZFaceField(grid)  # Rain terminal velocity (negative = downward)
    return (; zip(center_names, center_fields)..., wʳ)
end

@inline function update_microphysical_fields!(μ, bμp::WPNE1M, i, j, k, grid, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor
    qˡ = q.liquid  # total liquid from thermodynamic state
    categories = bμp.categories

    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ  # cloud liquid from prognostic field
        qʳ = μ.ρqʳ[i, j, k] / ρ    # rain from prognostic field
        μ.qᵛ[i, j, k] = qᵛ
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qʳ[i, j, k] = qʳ
        μ.qˡ[i, j, k] = qᶜˡ + qʳ  # total liquid (cloud + rain)

        # Terminal velocity for rain (negative = downward)
        wᵗ = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
        μ.wʳ[i, j, k] = -wᵗ
    end

    return nothing
end

@inline function compute_moisture_fractions(i, j, k, grid, bμp::WPNE1M, ρ, qᵗ, μ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ = μ.ρqʳ[i, j, k] / ρ
    end

    # Vapor is diagnosed from total moisture minus condensates
    qᵛ = qᵗ - qᶜˡ - qʳ
    qˡ = qᶜˡ + qʳ
    qⁱ = zero(qˡ)

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Compute thermodynamic state for non-equilibrium 1M microphysics.

Unlike saturation adjustment, cloud liquid is prognostic and temperature is computed
directly from the thermodynamic state without iteration. The moisture partition is
determined from the prognostic cloud liquid and rain fields.
"""
@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::WPNE1M, ρᵣ, μ, qᵗ, constants)
    # Get cloud liquid and rain from prognostic fields
    @inbounds qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρᵣ
    @inbounds qʳ = μ.ρqʳ[i, j, k] / ρᵣ

    # Vapor is diagnosed from total moisture minus condensates
    qᵛ = qᵗ - qᶜˡ - qʳ
    qˡ = qᶜˡ + qʳ

    # Build moisture state from prognostic fields
    q = MoistureMassFractions(qᵛ, qˡ)

    # Return thermodynamic state with prognostic moisture (no adjustment iteration)
    return with_moisture(𝒰₀, q)
end

#####
##### Microphysical tendencies for 1M schemes
#####

# Default fallback for OneMomentCloudMicrophysics tendencies that are not explicitly implemented
@inline microphysical_tendency(i, j, k, grid, bμp::OneMomentCloudMicrophysics, args...) = zero(grid)

# Rain tendency for non-equilibrium 1M: autoconversion + accretion - evaporation
@inline function microphysical_tendency(i, j, k, grid, bμp::WPNE1M, ::Val{:ρqʳ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    ρⁱʲᵏ = @inbounds ρ[i, j, k]

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]  # cloud liquid
    @inbounds qʳ = μ.qʳ[i, j, k]    # rain

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Get thermodynamic state for evaporation
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor
    qᵛ⁺ = saturation_specific_humidity(T, ρⁱʲᵏ, constants, PlanarLiquidSurface())

    # Rain evaporation (negative = rain decrease)
    τᵉᵛᵃᵖ = typeof(qᵛ)(DEFAULT_RAIN_EVAPORATION_TIMESCALE)
    Sᵉᵛᵃᵖ = rain_evaporation_rate(qᵛ, qᵛ⁺, qʳ, T, ρⁱʲᵏ, q, τᵉᵛᵃᵖ, constants)

    # Total tendency for ρqʳ (positive = rain increase)
    return ρⁱʲᵏ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)
end

"""
    rain_evaporation_rate(qᵛ, qᵛ⁺, qʳ, T, ρ, q, τᵉᵛᵃᵖ, constants)

Compute the rate of rain evaporation.

Rain evaporates when the air is subsaturated (qᵛ < qᵛ⁺). The evaporation rate
is proportional to the subsaturation and the rain content.

Returns a negative value (rain decrease) when subsaturated, zero otherwise.

The formula is a simplified version of the full ventilated evaporation formula,
using a relaxation approach similar to cloud condensation.

# Arguments
- `qᵛ`: vapor specific humidity
- `qᵛ⁺`: saturation specific humidity over liquid
- `qʳ`: rain specific humidity
- `T`: temperature
- `ρ`: air density
- `q`: MoistureMassFractions
- `τᵉᵛᵃᵖ`: evaporation timescale (typically ~100-1000 s for rain)
- `constants`: ThermodynamicConstants
"""
@inline function rain_evaporation_rate(qᵛ, qᵛ⁺, qʳ, T, ρ, q, τᵉᵛᵃᵖ, constants)
    FT = typeof(qᵛ)

    # No evaporation if rain is negligible or air is supersaturated
    no_evap = (qʳ ≤ eps(FT)) | (qᵛ ≥ qᵛ⁺)

    # Subsaturation (negative when subsaturated)
    S = (qᵛ - qᵛ⁺) / qᵛ⁺

    # Latent heat of vaporization at temperature T
    ℒˡ = liquid_latent_heat(T, constants)

    # Mixture heat capacity
    cᵖᵐ = mixture_heat_capacity(q, constants)

    # Vapor gas constant
    Rᵛ = vapor_gas_constant(constants)

    # Derivative of saturation specific humidity with respect to temperature
    dt_qᵛ⁺ = qᵛ⁺ * (ℒˡ / (Rᵛ * T^2) - 1 / T)

    # Thermodynamic adjustment factor
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dt_qᵛ⁺

    # Evaporation rate (negative = rain decrease)
    # This is proportional to subsaturation and rain content
    Sᵉᵛᵃᵖ = S * qʳ / (Γˡ * τᵉᵛᵃᵖ)

    # Only evaporate, clamp to zero when not subsaturated
    return ifelse(no_evap, zero(Sᵉᵛᵃᵖ), Sᵉᵛᵃᵖ)
end

# Default rain evaporation timescale (s) - can be overridden via parameters
const DEFAULT_RAIN_EVAPORATION_TIMESCALE = 500.0

"""
    condensation_rate(qᵛ, qᵛ⁺, T, τ_relax, constants)

Compute the condensation/evaporation rate following Morrison and Milbrandt (2015).

The rate is given by:
```math
\\frac{dq^{cℓ}}{dt} = \\frac{q^v - q^{v+}}{τ_{relax} Γ_ℓ}
```

where:
- `qᵛ` is the vapor specific humidity
- `qᵛ⁺` is the saturation specific humidity over liquid
- `τ_relax` is the relaxation timescale (typically ~10 s)
- `Γₗ = 1 + (Lᵥ/cₚ) * dqₛ/dT` is the thermodynamic adjustment factor

A positive rate indicates condensation (vapor → liquid), negative indicates evaporation.
"""
@inline function condensation_rate(qᵛ, qᵛ⁺, T, ρ, q, τᶜˡ, constants)
    # Latent heat of vaporization at temperature T
    ℒˡ = liquid_latent_heat(T, constants)

    # Mixture heat capacity
    cᵖᵐ = mixture_heat_capacity(q, constants)

    # Vapor gas constant
    Rᵛ = vapor_gas_constant(constants)

    # Derivative of saturation specific humidity with respect to temperature
    # dqₛ/dT = qᵛ⁺ * (Lᵥ / (Rᵛ * T²) - 1/T)
    dt_qᵛ⁺ = qᵛ⁺ * (ℒˡ / (Rᵛ * T^2) - 1 / T)

    # Thermodynamic adjustment factor (accounts for latent heat feedback)
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dt_qᵛ⁺

    # Condensation/evaporation rate (positive = condensation)
    return (qᵛ - qᵛ⁺) / (Γˡ * τᶜˡ)
end

# Cloud liquid tendency for non-equilibrium 1M: condensation/evaporation - (autoconversion + accretion)
@inline function microphysical_tendency(i, j, k, grid, bμp::WPNE1M, ::Val{:ρqᶜˡ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    cloud_formation = bμp.cloud_formation
    τᶜˡ = cloud_formation.liquid.τ_relax

    ρⁱʲᵏ = @inbounds ρ[i, j, k]

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]

    # Get thermodynamic state
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Saturation specific humidity over liquid
    qᵛ⁺ = saturation_specific_humidity(T, ρⁱʲᵏ, constants, PlanarLiquidSurface())

    # Condensation/evaporation rate (positive = condensation = cloud liquid increase)
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, T, ρⁱʲᵏ, q, τᶜˡ, constants)

    # Autoconversion: cloud liquid → rain (sink for cloud liquid)
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain (sink for cloud liquid)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Total tendency for ρqᶜˡ: condensation - autoconversion - accretion
    return ρⁱʲᵏ * (Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ)
end

# Default fallback for OneMomentCloudMicrophysics velocities
@inline microphysical_velocities(bμp::OneMomentCloudMicrophysics, μ, name) = nothing

# Rain sedimentation: rain falls with terminal velocity (stored in microphysical fields)
@inline function microphysical_velocities(bμp::OneMomentCloudMicrophysics, μ, ::Val{:ρqʳ})
    wʳ = μ.wʳ
    return (; u = ZeroField(), v = ZeroField(), w = wʳ)
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
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Total precipitation production rate (kg/kg/s)
    return Sᵃᶜⁿᵛ + Sᵃᶜᶜ
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

# Non-equilibrium 1M uses the same precipitation rate calculation (autoconversion + accretion)
function precipitation_rate(model, microphysics::WPNE1M, ::Val{:liquid})
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

