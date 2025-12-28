#####
##### One-moment microphysics (CloudMicrophysics 1M)
#####
#
# This file implements one-moment bulk microphysics for cloud liquid and rain,
# supporting both saturation adjustment (equilibrium) and non-equilibrium
# cloud formation.
#
# References:
#   - Morrison, H. and Grabowski, W.W. (2008). A novel approach for representing ice
#     microphysics in models: Description and tests using a kinematic framework.
#     J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1

# This file contains common infrastructure for all 1M schemes.
# Cloud liquid, rain, and tendency implementations are in one_moment_cloud_liquid_rain.jl

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
const OneMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM1MCategories, <:Any}

"""
    OneMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                               cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing),
                               categories = one_moment_cloud_microphysics_categories(FT),
                               precipitation_boundary_condition = nothing)

Return a `OneMomentCloudMicrophysics` microphysics scheme for warm-rain and mixed-phase precipitation.

The one-moment scheme uses CloudMicrophysics.jl 1M processes:
- Condensation/evaporation of cloud liquid (relaxation toward saturation)
- Autoconversion of cloud liquid to rain
- Accretion of cloud liquid by rain
- Terminal velocity for rain sedimentation

By default, non-equilibrium cloud formation is used, where cloud liquid is a prognostic
variable that evolves via condensation/evaporation tendencies following
[Morrison and Grabowski (2008)](@cite Morrison2008novel) (see Appendix A).
The prognostic variables are `ρqᶜˡ` (cloud liquid mass density) and `ρqʳ` (rain mass density).

For equilibrium (saturation adjustment) cloud formation, pass:

```jldoctest
using Breeze.Microphysics
cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

# output
SaturationAdjustment{WarmPhaseEquilibrium, Float64}(0.001, Inf, WarmPhaseEquilibrium())
```

# Keyword arguments
- `precipitation_boundary_condition`: Controls whether precipitation passes through the bottom boundary.
  - `nothing` (default): Rain exits through the bottom (open boundary)
  - `ImpenetrableBoundaryCondition()`: Rain collects at the bottom (zero terminal velocity at surface)

See the [CloudMicrophysics.jl documentation](https://clima.github.io/CloudMicrophysics.jl/dev/) for details.

# References
* Morrison, H. and Grabowski, W. W. (2008). A novel approach for representing ice
    microphysics in models: Description and tests using a kinematic framework.
    J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1
"""
function OneMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing),
                                    categories = one_moment_cloud_microphysics_categories(FT),
                                    precipitation_boundary_condition = nothing)

    return BulkMicrophysics(cloud_formation, categories, precipitation_boundary_condition)
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

# ImpenetrableBoundaryCondition alias
const IBC = BoundaryCondition{<:Open, Nothing}

# Helper for bottom terminal velocity based on precipitation_boundary_condition
# Used in update_microphysical_fields! to set wʳ[bottom] = 0 for ImpenetrableBoundaryCondition
@inline bottom_terminal_velocity(::Nothing, wʳ) = wʳ  # no boundary condition / open: keep computed value
@inline bottom_terminal_velocity(::IBC, wʳ) = zero(wʳ)  # impenetrable boundary condition

#####
##### Type aliases
#####

# Warm-phase saturation adjustment with 1M precipitation
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories, <:Any}

# Mixed-phase saturation adjustment with 1M precipitation
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories, <:Any}

# Warm-phase non-equilibrium with 1M precipitation
const WarmPhaseNonEquilibrium1M = BulkMicrophysics{<:NonEquilibriumCloudFormation{<:CloudLiquid, Nothing}, <:CM1MCategories, <:Any}
const WPNE1M = WarmPhaseNonEquilibrium1M

# Mixed-phase non-equilibrium with 1M precipitation
const MixedPhaseNonEquilibrium1M = BulkMicrophysics{<:NonEquilibriumCloudFormation{<:CloudLiquid, <:CloudIce}, <:CM1MCategories, <:Any}
const MPNE1M = MixedPhaseNonEquilibrium1M

# Union types for dispatch
const WarmPhase1M = Union{WP1M, WPNE1M}
const NonEquilibrium1M = Union{WPNE1M, MPNE1M}
const OneMomentLiquidRain = Union{WP1M, WPNE1M, MP1M, MPNE1M}

#####
##### Prognostic field names
#####

prognostic_field_names(::WP1M) = (:ρqʳ,)
prognostic_field_names(::WPNE1M) = (:ρqᶜˡ, :ρqʳ)
prognostic_field_names(::MP1M) = (:ρqʳ, :ρqˢ)
prognostic_field_names(::MPNE1M) = (:ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢ)

#####
##### Field materialization
#####

const warm_phase_field_names = (:ρqʳ, :qᵛ, :qˡ, :qᶜˡ, :qʳ)
const ice_phase_field_names = (:ρqˢ, :qⁱ, :qᶜⁱ, :qˢ)

function materialize_microphysical_fields(bμp::OneMomentLiquidRain, grid, bcs)
    if bμp isa WP1M
        center_names = warm_phase_field_names
    elseif bμp isa WPNE1M
        center_names = (:ρqᶜˡ, warm_phase_field_names...)
    elseif bμp isa MP1M
        center_names = (warm_phase_field_names..., ice_phase_field_names...)
    elseif bμp isa MPNE1M
        center_names = (:ρqᶜˡ, :ρqᶜⁱ, warm_phase_field_names..., ice_phase_field_names...)
    end

    center_fields = center_field_tuple(grid, center_names...)

    # Rain terminal velocity (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    wʳ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wʳ = ZFaceField(grid; boundary_conditions=wʳ_bcs)

    return (; zip(center_names, center_fields)..., wʳ)
end

#####
##### Update microphysical fields (diagnostics + terminal velocity)
#####

# Saturation adjustment: total liquid from thermodynamic state, cloud liquid = total - rain
@inline function update_microphysical_fields!(μ, bμp::Union{WP1M, MP1M}, i, j, k, grid, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    categories = bμp.categories

    @inbounds begin
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = q.vapor
        μ.qˡ[i, j, k] = q.liquid                 # total liquid from saturation adjustment
        μ.qᶜˡ[i, j, k] = max(0, q.liquid - qʳ)  # cloud liquid = total liquid - rain (clamped)
        μ.qʳ[i, j, k] = qʳ
    end

    maybe_update_ice_fields!(μ, bμp, i, j, k, grid, ρ, 𝒰, constants)
    update_rain_terminal_velocity!(μ, bμp, categories, i, j, k, ρ)

    return nothing
end

# Non-equilibrium warm-phase: cloud liquid from prognostic field
@inline function update_microphysical_fields!(μ, bμp::WPNE1M, i, j, k, grid, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    categories = bμp.categories

    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ  # cloud liquid from prognostic field
        qʳ = μ.ρqʳ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = q.vapor
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qʳ[i, j, k] = qʳ
        μ.qˡ[i, j, k] = qᶜˡ + qʳ  # total liquid = cloud + rain
    end

    update_rain_terminal_velocity!(μ, bμp, categories, i, j, k, ρ)

    return nothing
end

# Non-equilibrium mixed-phase: cloud liquid and ice from prognostic fields
@inline function update_microphysical_fields!(μ, bμp::MPNE1M, i, j, k, grid, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    categories = bμp.categories

    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ  # cloud liquid from prognostic field
        qᶜⁱ = μ.ρqᶜⁱ[i, j, k] / ρ  # cloud ice from prognostic field
        qʳ = μ.ρqʳ[i, j, k] / ρ
        qˢ = μ.ρqˢ[i, j, k] / ρ
        μ.qᵛ[i, j, k] = q.vapor
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qᶜⁱ[i, j, k] = qᶜⁱ
        μ.qʳ[i, j, k] = qʳ
        μ.qˢ[i, j, k] = qˢ
        μ.qˡ[i, j, k] = qᶜˡ + qʳ  # total liquid
        μ.qⁱ[i, j, k] = qᶜⁱ + qˢ  # total ice
    end

    update_rain_terminal_velocity!(μ, bμp, categories, i, j, k, ρ)

    return nothing
end

# Fallback for warm-phase schemes (no ice fields to update)
@inline maybe_update_ice_fields!(μ, bμp, i, j, k, grid, ρ, 𝒰, constants) = nothing

@inline function maybe_update_ice_fields!(μ, bμp::MP1M, i, j, k, grid, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    @inbounds begin
        μ.qᶜⁱ[i, j, k] = q.ice
        qˢ = μ.ρqˢ[i, j, k] / ρ
        μ.qˢ[i, j, k] = qˢ
    end
    return nothing
end

@inline function update_rain_terminal_velocity!(μ, bμp, categories, i, j, k, ρ)
    qʳ = @inbounds μ.qʳ[i, j, k]
    V = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, qʳ)
    wʳ = -V # negative = downward
    wʳ₀ = bottom_terminal_velocity(bμp.precipitation_boundary_condition, wʳ)
    @inbounds μ.wʳ[i, j, k] = ifelse(k == 1, wʳ₀, wʳ)
    return nothing
end

#####
##### Moisture fraction computation
#####

# Non-equilibrium warm-phase: cloud liquid is prognostic
@inline function compute_moisture_fractions(i, j, k, grid, bμp::WPNE1M, ρ, qᵗ, μ)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Non-equilibrium mixed-phase: cloud liquid and ice are prognostic
@inline function compute_moisture_fractions(i, j, k, grid, bμp::MPNE1M, ρ, qᵗ, μ)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qᶜⁱ = @inbounds μ.ρqᶜⁱ[i, j, k] / ρ
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˢ = @inbounds μ.ρqˢ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ
    qᵛ = qᵗ - qˡ - qⁱ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

# Saturation adjustment: read moisture partition from diagnostic fields (set in previous timestep).
# maybe_adjust_thermodynamic_state will then adjust to equilibrium for the current state.
@inline function compute_moisture_fractions(i, j, k, grid, bμp::WP1M, ρ, qᵗ, μ)
    qᶜˡ = @inbounds μ.qᶜˡ[i, j, k]
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Mixed-phase saturation adjustment: read moisture partition from diagnostic fields.
@inline function compute_moisture_fractions(i, j, k, grid, bμp::MP1M, ρ, qᵗ, μ)
    qᶜˡ = @inbounds μ.qᶜˡ[i, j, k]
    qᶜⁱ = @inbounds μ.qᶜⁱ[i, j, k]
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˢ = @inbounds μ.ρqˢ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ
    qᵛ = qᵗ - qˡ - qⁱ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Thermodynamic state adjustment
#####

# Non-equilibrium: no adjustment (cloud liquid and ice are prognostic)
@inline maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::NonEquilibrium1M, args...) = 𝒰₀

# Saturation adjustment (warm-phase and mixed-phase)
@inline function maybe_adjust_thermodynamic_state(i, j, k, 𝒰₀, bμp::Union{WP1M, MP1M}, ρᵣ, μ, qᵗ, constants)
    q₁ = MoistureMassFractions(qᵗ)
    𝒰₁ = with_moisture(𝒰₀, q₁)
    𝒰′ = adjust_thermodynamic_state(𝒰₁, bμp.cloud_formation, constants)
    return 𝒰′
end

#####
##### Condensation/evaporation for non-equilibrium cloud formation
#####
#
# The condensation rate follows Morrison and Grabowski (2008, JAS), Appendix Eq. (A3):
#
#   dqˡ/dt = (qᵛ - qᵛ⁺) / (Γˡ τˡ)
#
# where qᵛ⁺ is the saturation specific humidity, τˡ is the relaxation timescale,
# and Γˡ is a thermodynamic adjustment factor that accounts for latent heating:
#
#   Γˡ = 1 + (ℒˡ / cᵖᵐ) ⋅ dqᵛ⁺/dT
#
# This factor arises because condensation releases latent heat, which increases
# temperature and hence increases the saturation specific humidity, creating a
# negative feedback that slows the approach to equilibrium.
#
# The derivative dqᵛ*/dT follows from the Clausius-Clapeyron equation:
#
#   dqᵛ⁺/dT = qᵛ⁺ ⋅ (ℒˡ / (Rᵛ T²) - 1/T)
#
# See Morrison and Grabowski (2008, JAS), Appendix A, especially Eq. (A3).
#####
"""
    thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)

Compute the thermodynamic adjustment factor ``Γˡ`` for condensation/evaporation.

This factor accounts for the temperature dependence of saturation vapor pressure
during phase change, following [Morrison and Grabowski (2008)](@cite Morrison2008novel), Appendix Eq. (A3).

In cloud microphysics texts this is often described as the **psychrometric / thermal (latent-heat) correction**
that slows diffusional growth because latent heating warms air and increases ``qᵛ⁺``; see
[Pruppacher and Klett (2010)](@cite pruppacher2010microphysics) or [Rogers and Yau (1989)](@cite rogers1989short).

```math
Γˡ = 1 + \\frac{ℒˡ}{cᵖᵐ} \\frac{dqᵛ⁺}{dT}
```

where the temperature derivative of saturation specific humidity is:

```math
\\frac{dqᵛ⁺}{dT} = qᵛ⁺ \\left( \\frac{ℒˡ}{Rᵛ T²} - \\frac{1}{T} \\right)
```

# References
* Morrison, H. and Grabowski, W. W. (2008). A novel approach for representing ice
    microphysics in models: Description and tests using a kinematic framework.
    J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1
* Pruppacher, H. R. and Klett, J. D. (2010). Microphysics of Clouds and Precipitation (2nd ed.).
* Rogers, R. R. and Yau, M. K. (1989). A Short Course in Cloud Physics (3rd ed.).
"""
@inline function thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ * (ℒˡ / (Rᵛ * T^2) - 1 / T)
    return 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT
end

"""
    condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)

Compute the condensation/evaporation rate for cloud liquid water.

Returns the rate of change of cloud liquid mass fraction (kg/kg/s).
Positive values indicate condensation, negative values indicate evaporation.

The rate follows [Morrison and Grabowski (2008)](@cite Morrison2008novel), Appendix Eq. (A3):

```math
\\frac{dqᶜˡ}{dt} = \\frac{qᵛ - qᵛ⁺}{Γˡ τˡ}
```

Evaporation is limited to the available cloud liquid to prevent negative values.

# References
* Morrison, H. and Grabowski, W. W. (2008). A novel approach for representing ice
    microphysics in models: Description and tests using a kinematic framework.
    J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1
"""
@inline function condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Γˡ = thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺) / (Γˡ * τᶜˡ)

    # Limit evaporation to available cloud liquid
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / τᶜˡ
    Sᶜᵒⁿᵈ = max(Sᶜᵒⁿᵈ, Sᶜᵒⁿᵈ_min)

    return Sᶜᵒⁿᵈ
end

#####
##### Rain tendency (shared by all 1M schemes)
#####
#
# Rain mass evolves via:
#   - Autoconversion: cloud liquid → rain (source)
#   - Accretion: cloud liquid + rain → rain (source)
#   - Evaporation: rain → vapor in subsaturated air (sink)
#
# This tendency is the same for equilibrium and non-equilibrium cloud formation.
#####

# Numerical timescale for limiting negative-value relaxation
const τⁿᵘᵐ = 10  # seconds

@inline function microphysical_tendency(i, j, k, grid, bμp::OneMomentLiquidRain, ::Val{:ρqʳ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    ρⁱʲᵏ = ρ

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Rain evaporation in subsaturated air
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρⁱʲᵏ, T, constants)

    # Limit evaporation to available rain
    Sᵉᵛᵃᵖ_min = -max(0, qʳ) / τⁿᵘᵐ
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, Sᵉᵛᵃᵖ_min)

    # Total tendency for ρqʳ
    ΣρS = ρⁱʲᵏ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)

    # Numerical relaxation for negative values
    ρSⁿᵘᵐ = -ρⁱʲᵏ * qʳ / τⁿᵘᵐ

    return ifelse(qʳ >= 0, ΣρS, ρSⁿᵘᵐ)
end

#####
##### Cloud liquid tendency (non-equilibrium only)
#####

@inline function microphysical_tendency(i, j, k, grid, bμp::WPNE1M, ::Val{:ρqᶜˡ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    τᶜˡ = bμp.cloud_formation.liquid.τ_relax
    ρⁱʲᵏ = ρ

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]

    # Thermodynamic state
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρⁱʲᵏ, constants, PlanarLiquidSurface())

    # Condensation/evaporation rate
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρⁱʲᵏ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)

    # Autoconversion and accretion (sinks for cloud liquid)
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    # Total tendency
    ΣρS = ρⁱʲᵏ * (Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ)

    # Numerical relaxation for negative values
    ρSⁿᵘᵐ = -ρⁱʲᵏ * qᶜˡ / τᶜˡ

    return ifelse(qᶜˡ >= 0, ΣρS, ρSⁿᵘᵐ)
end

# Mixed-phase non-equilibrium: same as warm-phase for cloud liquid
@inline function microphysical_tendency(i, j, k, grid, bμp::MPNE1M, ::Val{:ρqᶜˡ}, ρ, μ, 𝒰, constants)
    categories = bμp.categories
    τᶜˡ = bμp.cloud_formation.liquid.τ_relax
    ρⁱʲᵏ = ρ

    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]

    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    qᵛ⁺ = saturation_specific_humidity(T, ρⁱʲᵏ, constants, PlanarLiquidSurface())
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρⁱʲᵏ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)

    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρⁱʲᵏ)

    ΣρS = ρⁱʲᵏ * (Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ)
    ρSⁿᵘᵐ = -ρⁱʲᵏ * qᶜˡ / τᶜˡ

    return ifelse(qᶜˡ >= 0, ΣρS, ρSⁿᵘᵐ)
end

#####
##### Cloud ice tendency (non-equilibrium mixed-phase only)
#####
#
# The deposition rate follows Morrison and Grabowski (2008, JAS), Appendix Eq. (A3), but for ice:
#
#   dqⁱ/dt = (qᵛ - qᵛ⁺ⁱ) / (Γⁱ τⁱ)
#
# where qᵛ⁺ⁱ is the saturation specific humidity over ice, τⁱ is the ice relaxation
# timescale, and Γⁱ is the thermodynamic adjustment factor using ice latent heat.
#####

"""
    ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, q, constants)

Compute the thermodynamic adjustment factor ``Γⁱ`` for deposition/sublimation.

Same as `thermodynamic_adjustment_factor` but uses ice latent heat and
saturation over ice surface.

See [Morrison and Grabowski (2008)](@cite Morrison2008novel), Appendix A, for the derivation of this
kind of latent-heat (psychrometric/thermal) correction in relaxation formulations.
"""
@inline function ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, q, constants)
    ℒⁱ = ice_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺ⁱ_dT = qᵛ⁺ⁱ * (ℒⁱ / (Rᵛ * T^2) - 1 / T)
    return 1 + (ℒⁱ / cᵖᵐ) * dqᵛ⁺ⁱ_dT
end

"""
    deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, ρ, q, τᶜⁱ, constants)

Compute the deposition/sublimation rate for cloud ice.

Returns the rate of change of cloud ice mass fraction (kg/kg/s).
Positive values indicate deposition, negative values indicate sublimation.

This is the ice analogue of `condensation_rate`, following
[Morrison and Grabowski (2008)](@cite Morrison2008novel), Appendix Eq. (A3), with ice latent heat and
saturation over ice.
"""
@inline function deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, ρ, q, τᶜⁱ, constants)
    Γⁱ = ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, q, constants)
    Sᵈᵉᵖ = (qᵛ - qᵛ⁺ⁱ) / (Γⁱ * τᶜⁱ)

    # Limit sublimation to available cloud ice
    Sᵈᵉᵖ_min = -max(0, qᶜⁱ) / τᶜⁱ
    Sᵈᵉᵖ = max(Sᵈᵉᵖ, Sᵈᵉᵖ_min)

    return Sᵈᵉᵖ
end

@inline function microphysical_tendency(i, j, k, grid, bμp::MPNE1M, ::Val{:ρqᶜⁱ}, ρ, μ, 𝒰, constants)
    τᶜⁱ = bμp.cloud_formation.ice.τ_relax
    ρⁱʲᵏ = ρ

    @inbounds qᶜⁱ = μ.qᶜⁱ[i, j, k]

    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Saturation specific humidity over ice
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρⁱʲᵏ, constants, PlanarIceSurface())

    # Deposition/sublimation rate
    Sᵈᵉᵖ = deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, ρⁱʲᵏ, q, τᶜⁱ, constants)
    Sᵈᵉᵖ = ifelse(isnan(Sᵈᵉᵖ), zero(Sᵈᵉᵖ), Sᵈᵉᵖ)

    # TODO: Add autoconversion cloud ice → snow when snow processes are implemented
    # For now, cloud ice only grows/shrinks via deposition/sublimation

    ΣρS = ρⁱʲᵏ * Sᵈᵉᵖ
    ρSⁿᵘᵐ = -ρⁱʲᵏ * qᶜⁱ / τᶜⁱ

    return ifelse(qᶜⁱ >= 0, ΣρS, ρSⁿᵘᵐ)
end
