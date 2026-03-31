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
#
# This file contains common infrastructure for all 1M schemes.
# Cloud liquid, rain, and tendency implementations are in one_moment_cloud_liquid_rain.jl
#
# ## MicrophysicalState pattern
#
# One-moment schemes use state structs (ℳ) to encapsulate local microphysical
# variables. This enables the same tendency functions to work for both grid-based
# LES and Lagrangian parcel models.
#
# For parcel models, the state is stored directly as `parcel.ℳ`.
# For grid models, the state is built via `grid_microphysical_state(i, j, k, grid, ...)`.
#####

using Breeze.AtmosphereModels: AbstractMicrophysicalState
using Breeze.AtmosphereModels: AtmosphereModels as AM

using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δzᵃᵃᶜ, ℑzᵃᵃᶜ
using Breeze.Thermodynamics: exner_function

#####
##### MicrophysicalState structs for one-moment schemes
#####

"""
    WarmPhaseOneMomentState{FT} <: AbstractMicrophysicalState{FT}

Microphysical state for warm-phase one-moment bulk microphysics.

Contains the local mixing ratios needed to compute tendencies for cloud liquid
and rain. This state is used for both saturation adjustment and non-equilibrium
cloud formation in warm-phase (liquid only) simulations.

# Fields
- `qᶜˡ`: Cloud liquid mixing ratio (kg/kg)
- `qʳ`: Rain mixing ratio (kg/kg)
"""
struct WarmPhaseOneMomentState{FT} <: AbstractMicrophysicalState{FT}
    qᶜˡ :: FT  # cloud liquid mixing ratio
    qʳ  :: FT  # rain mixing ratio
end

"""
    MixedPhaseOneMomentState{FT} <: AbstractMicrophysicalState{FT}

Microphysical state for mixed-phase one-moment bulk microphysics.

Contains the local mixing ratios for cloud liquid, cloud ice, rain, and snow.
This state is used for both saturation adjustment and non-equilibrium
cloud formation in mixed-phase simulations.

# Fields
- `qᶜˡ`: Cloud liquid mixing ratio (kg/kg)
- `qᶜⁱ`: Cloud ice mixing ratio (kg/kg)
- `qʳ`: Rain mixing ratio (kg/kg)
- `qˢ`: Snow mixing ratio (kg/kg)
"""
struct MixedPhaseOneMomentState{FT} <: AbstractMicrophysicalState{FT}
    qᶜˡ :: FT  # cloud liquid mixing ratio
    qᶜⁱ :: FT  # cloud ice mixing ratio
    qʳ  :: FT  # rain mixing ratio
    qˢ  :: FT  # snow mixing ratio
end

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

"""
    OneMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                               cloud_formation = NonEquilibriumCloudFormation(nothing, nothing),
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
                                    cloud_formation = NonEquilibriumCloudFormation(nothing, nothing),
                                    categories = one_moment_cloud_microphysics_categories(FT),
                                    precipitation_boundary_condition = nothing,
                                    negative_moisture_correction = nothing)

    # If `cloud_formation` is a `NonEquilibriumCloudFormation`, materialize `ConstantRateCondensateFormation`
    # models from the category parameters. The `rate` field stores `1/τ_relax`.
    # This allows users to pass:
    #   - `nothing` as a placeholder → replaced with rate from categories
    #   - `CloudLiquid` / `CloudIce` → replaced with rate from categories (ignoring the CM1M struct)
    #   - An `AbstractCondensateFormation` → used as-is
    if cloud_formation isa NonEquilibriumCloudFormation
        liquid = cloud_formation.liquid
        ice = cloud_formation.ice

        # Liquid: always materialize unless already an AbstractCondensateFormation
        liquid = materialize_condensate_formation(liquid, categories.cloud_liquid)

        # Ice: `nothing` → warm-phase (no ice), otherwise materialize
        ice = ifelse(ice === nothing,
                     nothing,
                     materialize_condensate_formation(ice, categories.cloud_ice))

        cloud_formation = NonEquilibriumCloudFormation(liquid, ice)
    end

    return BulkMicrophysics(cloud_formation, categories, precipitation_boundary_condition, negative_moisture_correction)
end

# Materialize a condensate-formation model from a placeholder or category parameter.
# If already an AbstractCondensateFormation, return as-is.
materialize_condensate_formation(cf::AbstractCondensateFormation, category) = cf
materialize_condensate_formation(::Nothing, category) = ConstantRateCondensateFormation(1 / category.τ_relax)
materialize_condensate_formation(::Any, category) = ConstantRateCondensateFormation(1 / category.τ_relax)

#####
##### Default fallbacks for OneMomentCloudMicrophysics
#####

const OMCM = OneMomentCloudMicrophysics

# Default fallback for OneMomentCloudMicrophysics tendencies (state-based)
@inline AM.microphysical_tendency(bμp::OMCM, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

# Default fallback for OneMomentCloudMicrophysics velocities
@inline AM.microphysical_velocities(bμp::OMCM, μ, name) = nothing

# Rain sedimentation: rain falls with terminal velocity (stored in microphysical fields)
const zf = ZeroField()
@inline AM.microphysical_velocities(bμp::OMCM, μ, ::Val{:ρqʳ}) = (u=zf, v=zf, w=μ.wʳ)

# ImpenetrableBoundaryCondition alias
const IBC = BoundaryCondition{<:Open, Nothing}

# Helper for bottom terminal velocity based on precipitation_boundary_condition
# Used in update_microphysical_fields! to set wʳ[bottom] = 0 for ImpenetrableBoundaryCondition
@inline bottom_terminal_velocity(::Nothing, wʳ) = wʳ  # no boundary condition / open: keep computed value
@inline bottom_terminal_velocity(::IBC, wʳ) = zero(wʳ)  # impenetrable boundary condition

#####
##### Type aliases
#####

# Shorthand for AbstractCondensateFormation (used in type constraints below)
const ACF = AbstractCondensateFormation

# Warm-phase saturation adjustment with 1M precipitation
const WP1M = BulkMicrophysics{<:WarmPhaseSaturationAdjustment, <:CM1MCategories}

# Mixed-phase saturation adjustment with 1M precipitation
const MP1M = BulkMicrophysics{<:MixedPhaseSaturationAdjustment, <:CM1MCategories}

# Non-equilibrium cloud formation type aliases (liquid only vs liquid + ice)
const WarmPhaseNE = NonEquilibriumCloudFormation{<:ACF, Nothing}
const MixedPhaseNE = NonEquilibriumCloudFormation{<:ACF, <:ACF}

# Warm-phase non-equilibrium with 1M precipitation
const WarmPhaseNonEquilibrium1M = BulkMicrophysics{<:WarmPhaseNE, <:CM1MCategories}
const WPNE1M = WarmPhaseNonEquilibrium1M

# Mixed-phase non-equilibrium with 1M precipitation
const MixedPhaseNonEquilibrium1M = BulkMicrophysics{<:MixedPhaseNE, <:CM1MCategories}
const MPNE1M = MixedPhaseNonEquilibrium1M

# Union types for dispatch
const WarmPhase1M = Union{WP1M, WPNE1M}
const MixedPhase1M = Union{MP1M, MPNE1M}
const NonEquilibrium1M = Union{WPNE1M, MPNE1M}
const OneMomentLiquidRain = Union{WP1M, WPNE1M, MP1M, MPNE1M}

#####
##### Gridless MicrophysicalState construction
#####
#
# Microphysics schemes implement the gridless microphysical_state(microphysics, ρ, μ, 𝒰, velocities)
# which takes density-weighted prognostic variables μ (NamedTuple of scalars) and
# thermodynamic state 𝒰. The grid-indexed version is a generic wrapper that extracts
# μ from fields and calls this.
#
# For saturation adjustment: cloud condensate comes from 𝒰.moisture_mass_fractions
# For non-equilibrium: cloud condensate comes from prognostic μ

# Warm-phase saturation adjustment: cloud liquid from thermodynamic state, rain from prognostic
# The velocities argument is required for interface compatibility but not used by one-moment schemes.
@inline function AM.microphysical_state(bμp::WP1M, ρ, μ, 𝒰, velocities)
    q = 𝒰.moisture_mass_fractions
    qʳ = μ.ρqʳ / ρ
    qᶜˡ = max(zero(qʳ), q.liquid - qʳ)  # cloud liquid = total liquid - rain
    return WarmPhaseOneMomentState(qᶜˡ, qʳ)
end

# Warm-phase non-equilibrium: all from prognostic μ
@inline function AM.microphysical_state(bμp::WPNE1M, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    qʳ = μ.ρqʳ / ρ
    return WarmPhaseOneMomentState(qᶜˡ, qʳ)
end

# Mixed-phase saturation adjustment: cloud condensate from thermodynamic state
@inline function AM.microphysical_state(bμp::MP1M, ρ, μ, 𝒰, velocities)
    q = 𝒰.moisture_mass_fractions
    qʳ = μ.ρqʳ / ρ
    qˢ = μ.ρqˢ / ρ
    qᶜˡ = max(zero(qʳ), q.liquid - qʳ)  # cloud liquid = total liquid - rain
    qᶜⁱ = max(zero(qˢ), q.ice - qˢ)     # cloud ice = total ice - snow
    return MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢ)
end

# Mixed-phase non-equilibrium: all from prognostic μ
@inline function AM.microphysical_state(bμp::MPNE1M, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    qᶜⁱ = μ.ρqᶜⁱ / ρ
    qʳ = μ.ρqʳ / ρ
    qˢ = μ.ρqˢ / ρ
    return MixedPhaseOneMomentState(qᶜˡ, qᶜⁱ, qʳ, qˢ)
end

#####
##### Relaxation timescales for non-equilibrium schemes
#####
#
# The `ConstantRateCondensateFormation.rate` field stores `1/τ_relax`, so we invert it.

@inline liquid_relaxation_timescale(cloud_formation, categories) = 1 / cloud_formation.liquid.rate
@inline ice_relaxation_timescale(cloud_formation::NonEquilibriumCloudFormation{<:Any, Nothing}, categories) = nothing
@inline ice_relaxation_timescale(cloud_formation, categories) = 1 / cloud_formation.ice.rate

#####
##### Prognostic field names
#####

AM.prognostic_field_names(::WP1M) = (:ρqʳ,)
AM.prognostic_field_names(::WPNE1M) = (:ρqᶜˡ, :ρqʳ)
AM.prognostic_field_names(::MP1M) = (:ρqʳ, :ρqˢ)
AM.prognostic_field_names(::MPNE1M) = (:ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢ)

# Negative moisture correction chains: heaviest → lightest → vapor
AM.correction_moisture_fields(::WP1M, μ) = (μ.ρqʳ,)
AM.correction_moisture_fields(::WPNE1M, μ) = (μ.ρqʳ, μ.ρqᶜˡ)
# Mixed-phase correction not yet implemented (requires energy adjustment for ice↔liquid)

#####
##### Field materialization
#####

const warm_phase_field_names = (:ρqʳ, :qᵛ, :qˡ, :qᶜˡ, :qʳ, :hˡ, :Θˡ)
const ice_phase_field_names = (:ρqˢ, :qⁱ, :qᶜⁱ, :qˢ, :hⁱ, :Θⁱ)

function AM.materialize_microphysical_fields(bμp::OneMomentLiquidRain, grid, bcs)
    if bμp isa WP1M
        center_names = (warm_phase_field_names..., :qᵉ)
    elseif bμp isa WPNE1M
        center_names = (:ρqᶜˡ, warm_phase_field_names...)
    elseif bμp isa MP1M
        center_names = (warm_phase_field_names..., ice_phase_field_names..., :qᵉ)
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
##### update_microphysical_auxiliaries! for one-moment schemes
#####
#
# This single function updates all auxiliary (non-prognostic) microphysical fields.
# Grid indices (i, j, k) are needed because:
# 1. Fields must be written at specific grid points
# 2. Terminal velocity needs k == 1 check for bottom boundary condition

# Warm-phase one-moment schemes
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, bμp::WarmPhase1M, ℳ::WarmPhaseOneMomentState, ρ, 𝒰, constants)
    # State fields
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qʳ[i, j, k] = ℳ.qʳ

    # Vapor from thermodynamic state
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor

    # Derived: total liquid
    @inbounds μ.qˡ[i, j, k] = ℳ.qᶜˡ + ℳ.qʳ

    # Terminal velocity with bottom boundary condition
    categories = bμp.categories
    𝕎 = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, ℳ.qʳ)
    wʳ = -𝕎 # negative = downward
    wʳ₀ = bottom_terminal_velocity(bμp.precipitation_boundary_condition, wʳ)
    @inbounds μ.wʳ[i, j, k] = ifelse(k == 1, wʳ₀, wʳ)

    # Sedimentation enthalpy per unit liquid mass: hˡ = cˡ T - ℒˡᵣ (for ρe flux)
    # and θˡⁱ contribution per unit liquid mass: Θˡ = hˡ / (cᵖᵐ Π) (for ρθ flux).
    q = 𝒰.moisture_mass_fractions
    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    cˡ = constants.liquid.heat_capacity
    θ = 𝒰.potential_temperature
    T = Π * θ + (ℒˡᵣ * q.liquid + constants.ice.reference_latent_heat * q.ice) / cᵖᵐ
    hˡ = cˡ * T - ℒˡᵣ
    @inbounds μ.hˡ[i, j, k] = hˡ
    @inbounds μ.Θˡ[i, j, k] = hˡ / (cᵖᵐ * Π)

    return nothing
end

# Mixed-phase one-moment schemes
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, bμp::MixedPhase1M, ℳ::MixedPhaseOneMomentState, ρ, 𝒰, constants)
    # State fields
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qᶜⁱ[i, j, k] = ℳ.qᶜⁱ
    @inbounds μ.qʳ[i, j, k] = ℳ.qʳ
    @inbounds μ.qˢ[i, j, k] = ℳ.qˢ

    # Vapor from thermodynamic state
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor

    # Derived: total liquid and ice
    @inbounds μ.qˡ[i, j, k] = ℳ.qᶜˡ + ℳ.qʳ
    @inbounds μ.qⁱ[i, j, k] = ℳ.qᶜⁱ + ℳ.qˢ

    # Terminal velocity with bottom boundary condition
    categories = bμp.categories
    𝕎 = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, ℳ.qʳ)
    wʳ = -𝕎 # negative = downward
    wʳ₀ = bottom_terminal_velocity(bμp.precipitation_boundary_condition, wʳ)
    @inbounds μ.wʳ[i, j, k] = ifelse(k == 1, wʳ₀, wʳ)

    # Sedimentation enthalpies per unit liquid/ice mass (for ρe flux)
    # and θˡⁱ contributions (for ρθ flux).
    q = 𝒰.moisture_mass_fractions
    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    cˡ = constants.liquid.heat_capacity
    cⁱ = constants.ice.heat_capacity
    θ = 𝒰.potential_temperature
    T = Π * θ + (ℒˡᵣ * q.liquid + ℒⁱᵣ * q.ice) / cᵖᵐ
    hˡ = cˡ * T - ℒˡᵣ
    hⁱ = cⁱ * T - ℒⁱᵣ
    @inbounds μ.hˡ[i, j, k] = hˡ
    @inbounds μ.hⁱ[i, j, k] = hⁱ
    @inbounds μ.Θˡ[i, j, k] = hˡ / (cᵖᵐ * Π)
    @inbounds μ.Θⁱ[i, j, k] = hⁱ / (cᵖᵐ * Π)

    return nothing
end

#####
##### specific_prognostic_moisture_from_total: convert qᵗ to qᵛᵉ
#####

# SA warm-phase: qᵉ = qᵗ - qʳ (subtract precipitation)
@inline AM.specific_prognostic_moisture_from_total(bμp::WP1M, qᵗ, ℳ::WarmPhaseOneMomentState) = qᵗ - ℳ.qʳ

# SA mixed-phase: qᵉ = qᵗ - qʳ - qˢ (subtract precipitation)
@inline AM.specific_prognostic_moisture_from_total(bμp::MP1M, qᵗ, ℳ::MixedPhaseOneMomentState) = qᵗ - ℳ.qʳ - ℳ.qˢ

# NE warm-phase: qᵛ = qᵗ - qᶜˡ - qʳ (subtract all condensate)
@inline AM.specific_prognostic_moisture_from_total(bμp::WPNE1M, qᵗ, ℳ::WarmPhaseOneMomentState) = max(0, qᵗ - ℳ.qᶜˡ - ℳ.qʳ)

# NE mixed-phase: qᵛ = qᵗ - qᶜˡ - qᶜⁱ - qʳ - qˢ (subtract all condensate)
@inline AM.specific_prognostic_moisture_from_total(bμp::MPNE1M, qᵗ, ℳ::MixedPhaseOneMomentState) = max(0, qᵗ - ℳ.qᶜˡ - ℳ.qᶜⁱ - ℳ.qʳ - ℳ.qˢ)

#####
##### Moisture fraction computation
#####

# State-based (gridless) moisture fraction computation for warm-phase 1M microphysics.
# Works with WarmPhaseOneMomentState which contains specific quantities (qᶜˡ, qʳ).
# Input qᵉ is total/equilibrium moisture; subtract condensate to get vapor.
# Used by parcel models. Grid models use grid_moisture_fractions instead.
@inline function AM.moisture_fractions(bμp::WarmPhase1M, ℳ::WarmPhaseOneMomentState, qᵉ)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qᵛ = qᵉ - ℳ.qᶜˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

# State-based moisture fraction computation for mixed-phase 1M microphysics.
# SA: qᵉ is equilibrium moisture, subtract condensate to get vapor
@inline function AM.moisture_fractions(bμp::MP1M, ℳ::MixedPhaseOneMomentState, qᵉ)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qⁱ = ℳ.qᶜⁱ + ℳ.qˢ
    qᵛ = qᵉ - ℳ.qᶜˡ - ℳ.qᶜⁱ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

# NE: input is vapor; subtract condensate to get vapor (for parcel models).
@inline function AM.moisture_fractions(bμp::MPNE1M, ℳ::MixedPhaseOneMomentState, qᵛ)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qⁱ = ℳ.qᶜⁱ + ℳ.qˢ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### grid_moisture_fractions for saturation adjustment schemes
#####
# Saturation adjustment schemes read cloud condensate from diagnostic fields (set in previous timestep).
# maybe_adjust_thermodynamic_state will then adjust to equilibrium for the current state.
@inline function AM.grid_moisture_fractions(i, j, k, grid, bμp::WP1M, ρ, qᵉ, μ)
    qᶜˡ = @inbounds μ.qᶜˡ[i, j, k]
    qʳ  = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qᵛ = qᵉ - qᶜˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Warm-phase non-equilibrium: prognostic stores true vapor; construct fractions directly.
@inline function AM.grid_moisture_fractions(i, j, k, grid, bμp::WPNE1M, ρ, qᵛ, μ)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ  = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Mixed-phase saturation adjustment: read moisture partition from diagnostic fields.
@inline function AM.grid_moisture_fractions(i, j, k, grid, bμp::MP1M, ρ, qᵉ, μ)
    qᶜˡ = @inbounds μ.qᶜˡ[i, j, k]
    qᶜⁱ = @inbounds μ.qᶜⁱ[i, j, k]
    qʳ  = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˢ  = @inbounds μ.ρqˢ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ
    qᵛ = qᵉ - qᶜˡ - qᶜⁱ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

# Mixed-phase non-equilibrium: prognostic stores true vapor; construct fractions directly.
@inline function AM.grid_moisture_fractions(i, j, k, grid, bμp::MPNE1M, ρ, qᵛ, μ)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ  = @inbounds μ.ρqʳ[i, j, k]  / ρ
    qᶜⁱ = @inbounds μ.ρqᶜⁱ[i, j, k] / ρ
    qˢ  = @inbounds μ.ρqˢ[i, j, k]  / ρ
    qˡ = qᶜˡ + qʳ
    qⁱ = qᶜⁱ + qˢ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Thermodynamic state adjustment
#####

# Non-equilibrium: no adjustment (cloud liquid and ice are prognostic)
@inline AM.maybe_adjust_thermodynamic_state(𝒰₀, bμp::NonEquilibrium1M, qᵛ, constants) = 𝒰₀

# Saturation adjustment (warm-phase and mixed-phase)
@inline function AM.maybe_adjust_thermodynamic_state(𝒰₀, bμp::Union{WP1M, MP1M}, qᵉ, constants)
    q₁ = MoistureMassFractions(qᵉ)
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
#
# `thermodynamic_adjustment_factor` and `condensation_rate` are defined in `Breeze.Microphysics`
# so they can be shared by multiple bulk microphysics schemes.

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

# State-based rain tendency for all warm-phase 1M schemes
@inline function AM.microphysical_tendency(bμp::WarmPhase1M, ::Val{:ρqʳ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants)
    categories = bμp.categories
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Rain evaporation in subsaturated air
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρ, T, constants)

    # Limit evaporation to available rain
    Sᵉᵛᵃᵖ_min = -max(0, qʳ) / τⁿᵘᵐ
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, Sᵉᵛᵃᵖ_min)

    # Total tendency for ρqʳ
    ΣρS = ρ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)

    # Numerical relaxation for negative values
    ρSⁿᵘᵐ = -ρ * qʳ / τⁿᵘᵐ

    return ifelse(qʳ >= 0, ΣρS, ρSⁿᵘᵐ)
end

# State-based rain tendency for mixed-phase 1M schemes
@inline function AM.microphysical_tendency(bμp::Union{MP1M, MPNE1M}, ::Val{:ρqʳ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    categories = bμp.categories
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Rain evaporation in subsaturated air
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρ, T, constants)

    # Limit evaporation to available rain
    Sᵉᵛᵃᵖ_min = -max(0, qʳ) / τⁿᵘᵐ
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, Sᵉᵛᵃᵖ_min)

    # Total tendency for ρqʳ
    ΣρS = ρ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)

    # Numerical relaxation for negative values
    ρSⁿᵘᵐ = -ρ * qʳ / τⁿᵘᵐ

    return ifelse(qʳ >= 0, ΣρS, ρSⁿᵘᵐ)
end

#####
##### Microphysical tendencies for warm-phase non-equilibrium 1M (WPNE1M)
#####
#
# Conservation: d(ρqᵛ)/dt + d(ρqᶜˡ)/dt + d(ρqʳ)/dt = 0 (from phase changes)
#
# The bundle function computes all phase-change rates once and returns every
# tendency derived from them. This guarantees discrete conservation: the same
# rate value appears in every tendency that references it.
#
#   ρqᵛ:  −Sᶜᵒⁿᵈ − Sᵉᵛᵃᵖ    (vapor loses to condensation; evaporation restores vapor)
#   ρqᶜˡ: +Sᶜᵒⁿᵈ − Sᵃᶜⁿᵛ − Sᵃᶜᶜ  (condensation source; autoconversion/accretion sinks)
#   ρqʳ:  +Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ  (autoconversion/accretion sources; evaporation sink)
#####

@inline function wpne1m_tendencies(bμp::WPNE1M, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants)
    categories = bμp.categories
    τᶜˡ = liquid_relaxation_timescale(bμp.cloud_formation, categories)
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ

    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Condensation: vapor ↔ cloud liquid
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)

    # Evaporation: rain → vapor (Sᵉᵛᵃᵖ < 0 when rain evaporates)
    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρ, T, constants)
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, -max(0, qʳ) / τⁿᵘᵐ)

    # Collection: cloud liquid → rain (does not involve vapor)
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Physics tendencies — conserved by construction: ρqᵛ_phys + ρqᶜˡ_phys + ρqʳ_phys = 0
    ρqᵛ_phys  = ρ * (-Sᶜᵒⁿᵈ - Sᵉᵛᵃᵖ)
    ρqᶜˡ_phys = ρ * ( Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ)
    ρqʳ_phys  = ρ * ( Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)

    # Numerical relaxation guards — conserved by routing each correction to its exchange partner.
    # When q < 0, replace with -ρq/τ and route the delta: v→cl, cl→r, r→v.
    # This preserves ρqᵛ + ρqᶜˡ + ρqʳ = 0 regardless of which guards fire.
    δᵛ  = ifelse(qᵛ  >= 0, zero(ρqᵛ_phys),  -ρ * qᵛ  / τⁿᵘᵐ      - ρqᵛ_phys)
    δᶜˡ = ifelse(qᶜˡ >= 0, zero(ρqᶜˡ_phys), -ρ * qᶜˡ / τᶜˡ        - ρqᶜˡ_phys)
    δʳ  = ifelse(qʳ  >= 0, zero(ρqʳ_phys),  -ρ * qʳ  / τⁿᵘᵐ      - ρqʳ_phys)

    ρqᵛ  = ρqᵛ_phys  + δᵛ  - δʳ
    ρqᶜˡ = ρqᶜˡ_phys + δᶜˡ - δᵛ
    ρqʳ  = ρqʳ_phys  + δʳ  - δᶜˡ

    return (; ρqᵛ, ρqᶜˡ, ρqʳ)
end

@inline function AM.microphysical_tendency(bμp::WPNE1M, ::Val{:ρqᵛ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants)
    return wpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᵛ
end

@inline function AM.microphysical_tendency(bμp::WPNE1M, ::Val{:ρqᶜˡ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants)
    return wpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᶜˡ
end

@inline function AM.microphysical_tendency(bμp::WPNE1M, ::Val{:ρqʳ}, ρ, ℳ::WarmPhaseOneMomentState, 𝒰, constants)
    return wpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqʳ
end

#####
##### Microphysical tendencies for mixed-phase non-equilibrium 1M (MPNE1M)
#####
#
# Conservation: d(ρqᵛ)/dt + d(ρqᶜˡ)/dt + d(ρqᶜⁱ)/dt + d(ρqʳ)/dt = 0 (from phase changes)
#
# The deposition rate follows Morrison and Grabowski (2008, JAS), Appendix Eq. (A3), but for ice:
#
#   dqⁱ/dt = (qᵛ - qᵛ⁺ⁱ) / (Γⁱ τⁱ)
#
# where qᵛ⁺ⁱ is the saturation specific humidity over ice, τⁱ is the ice relaxation
# timescale, and Γⁱ is the thermodynamic adjustment factor using ice latent heat.
#
# `ice_thermodynamic_adjustment_factor` and `deposition_rate` are defined in `Breeze.Microphysics`
# so they can be shared by multiple bulk microphysics schemes.
#
#   ρqᵛ:  −Sᶜᵒⁿᵈ − Sᵈᵉᵖ − Sᵉᵛᵃᵖ    (vapor loses to condensation and deposition)
#   ρqᶜˡ: +Sᶜᵒⁿᵈ − Sᵃᶜⁿᵛ − Sᵃᶜᶜ    (condensation source; collection sinks)
#   ρqᶜⁱ: +Sᵈᵉᵖ                     (deposition source only; ice→snow TODO)
#   ρqʳ:  +Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ    (collection sources; evaporation sink)
#####

@inline function mpne1m_tendencies(bμp::MPNE1M, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    categories = bμp.categories
    τᶜˡ = liquid_relaxation_timescale(bμp.cloud_formation, categories)
    τᶜⁱ = ice_relaxation_timescale(bμp.cloud_formation, categories)
    qᶜˡ = ℳ.qᶜˡ
    qᶜⁱ = ℳ.qᶜⁱ
    qʳ = ℳ.qʳ

    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Condensation: vapor ↔ cloud liquid
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)

    # Deposition: vapor ↔ cloud ice
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    Sᵈᵉᵖ = deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, ρ, q, τᶜⁱ, constants)
    Sᵈᵉᵖ = ifelse(isnan(Sᵈᵉᵖ), zero(Sᵈᵉᵖ), Sᵈᵉᵖ)

    # Evaporation: rain → vapor (Sᵉᵛᵃᵖ < 0 when rain evaporates)
    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρ, T, constants)
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, -max(0, qʳ) / τⁿᵘᵐ)

    # Collection: cloud liquid → rain (does not involve vapor)
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Physics tendencies — conserved by construction: ρqᵛ_phys + ρqᶜˡ_phys + ρqᶜⁱ_phys + ρqʳ_phys = 0
    ρqᵛ_phys  = ρ * (-Sᶜᵒⁿᵈ - Sᵈᵉᵖ - Sᵉᵛᵃᵖ)
    ρqᶜˡ_phys = ρ * ( Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ)
    ρqᶜⁱ_phys = ρ * Sᵈᵉᵖ
    ρqʳ_phys  = ρ * ( Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)

    # Numerical relaxation guards — conserved by routing each correction to its exchange partner.
    # When q < 0, replace with -ρq/τ and route the delta to the coupled tracer:
    #   v→cl (condensation), cl→r (collection), ci→v (deposition), r→v (evaporation).
    # This preserves ρqᵛ + ρqᶜˡ + ρqᶜⁱ + ρqʳ = 0 regardless of which guards fire.
    δᵛ  = ifelse(qᵛ  >= 0, zero(ρqᵛ_phys),  -ρ * qᵛ  / τⁿᵘᵐ - ρqᵛ_phys)
    δᶜˡ = ifelse(qᶜˡ >= 0, zero(ρqᶜˡ_phys), -ρ * qᶜˡ / τᶜˡ  - ρqᶜˡ_phys)
    δᶜⁱ = ifelse(qᶜⁱ >= 0, zero(ρqᶜⁱ_phys), -ρ * qᶜⁱ / τᶜⁱ  - ρqᶜⁱ_phys)
    δʳ  = ifelse(qʳ  >= 0, zero(ρqʳ_phys),  -ρ * qʳ  / τⁿᵘᵐ - ρqʳ_phys)

    ρqᵛ  = ρqᵛ_phys  + δᵛ  - δᶜⁱ - δʳ
    ρqᶜˡ = ρqᶜˡ_phys + δᶜˡ - δᵛ
    ρqᶜⁱ = ρqᶜⁱ_phys + δᶜⁱ
    ρqʳ  = ρqʳ_phys  + δʳ  - δᶜˡ

    return (; ρqᵛ, ρqᶜˡ, ρqᶜⁱ, ρqʳ)
end

@inline function AM.microphysical_tendency(bμp::MPNE1M, ::Val{:ρqᵛ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
   G = mpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants)
   return G.ρqᵛ
end

@inline function AM.microphysical_tendency(bμp::MPNE1M, ::Val{:ρqᶜˡ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    return mpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᶜˡ
end

@inline function AM.microphysical_tendency(bμp::MPNE1M, ::Val{:ρqᶜⁱ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    # TODO: Add autoconversion cloud ice → snow when snow processes are implemented
    return mpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᶜⁱ
end

@inline function AM.microphysical_tendency(bμp::MPNE1M, ::Val{:ρqʳ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    return mpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqʳ
end

#####
##### Precipitation sedimentation tendency for thermodynamic variables
#####
#
# Rain sediments at terminal velocity wʳ. Since θ_li and static energy e contain
# rain-water enthalpy terms, the missing sedimentation flux divergence is added
# here as a grid_microphysical_tendency.
#
# Face values of ρqʳ are reconstructed using 1st-order upwinding.
#
# For ρe the flux is  F = w · hˡ · ρq_upwind  and the tendency is  -V⁻¹ δz(F),
# where hˡ = cˡ T - ℒˡᵣ is the rain-water enthalpy per unit mass.
# Terminal-fall drag converts the lost hydrometeor geopotential energy into
# local heating, Qᵈ = -g Fᵐ where Fᵐ is the downward hydrometeor mass flux.
#
# For ρθ, rain carries both latent heat (-ℒˡᵣ) and sensible heat (cˡ T) per unit
# mass.  The θˡⁱ contribution per unit hydrometeor mass is
#   Θˡ = (cˡ T - ℒˡᵣ) / (cᵖᵐ Π)   [liquid]
#   Θⁱ = (cⁱ T - ℒⁱᵣ) / (cᵖᵐ Π)   [ice]
# These are precomputed at cell centers during update_microphysical_auxiliaries!
# and upwinded at faces.  The flux is  F_θ = w · ρq_upwind · Θ_upwind.
# The precipitation drag heating is added consistently as Qᵈ / (cᵖᵐ Π).
#
# Note: the mass sedimentation for ρqʳ uses Oceananigans' configured advection scheme,
# which may differ from the 1st-order upwinding used here. This introduces a small
# discrete inconsistency between the energy and mass sedimentation fluxes.
#####

# Vertical flux of hydrometeor enthalpy on z-faces.
# hˡ = cˡ T - ℒˡᵣ (and hⁱ = cⁱ T - ℒⁱᵣ) are precomputed and upwinded alongside ρq.
@inline function sedimentation_enthalpy_flux_z(i, j, k, grid, wʳ, ρqʳ, hˡ)
    @inbounds wʳ_face = wʳ[i, j, k]
    @inbounds ρqʳ_upwind = ifelse(wʳ_face <= 0, ρqʳ[i, j, k], ρqʳ[i, j, k-1])
    @inbounds hˡ_upwind  = ifelse(wʳ_face <= 0, hˡ[i, j, k], hˡ[i, j, k-1])
    return wʳ_face * ρqʳ_upwind * hˡ_upwind
end

@inline function sedimentation_enthalpy_flux_z(i, j, k, grid, wʳ, ρqʳ, hˡ, wˢ, ρqˢ, hⁱ)
    @inbounds wʳ_face = wʳ[i, j, k]
    @inbounds ρqʳ_upwind = ifelse(wʳ_face <= 0, ρqʳ[i, j, k], ρqʳ[i, j, k-1])
    @inbounds hˡ_r       = ifelse(wʳ_face <= 0, hˡ[i, j, k], hˡ[i, j, k-1])

    @inbounds wˢ_face = wˢ[i, j, k]
    @inbounds ρqˢ_upwind = ifelse(wˢ_face <= 0, ρqˢ[i, j, k], ρqˢ[i, j, k-1])
    @inbounds hⁱ_s       = ifelse(wˢ_face <= 0, hⁱ[i, j, k], hⁱ[i, j, k-1])

    return wʳ_face * ρqʳ_upwind * hˡ_r + wˢ_face * ρqˢ_upwind * hⁱ_s
end

# Vertical flux of hydrometeor mass on z-faces.
@inline function sedimentation_mass_flux_z(i, j, k, grid, wʳ, ρqʳ)
    @inbounds wʳ_face = wʳ[i, j, k]
    @inbounds ρqʳ_upwind = ifelse(wʳ_face <= 0, ρqʳ[i, j, k], ρqʳ[i, j, k-1])
    return wʳ_face * ρqʳ_upwind
end

@inline function sedimentation_mass_flux_z(i, j, k, grid, wʳ, ρqʳ, wˢ, ρqˢ)
    @inbounds wʳ_face = wʳ[i, j, k]
    @inbounds ρqʳ_upwind = ifelse(wʳ_face <= 0, ρqʳ[i, j, k], ρqʳ[i, j, k-1])

    @inbounds wˢ_face = wˢ[i, j, k]
    @inbounds ρqˢ_upwind = ifelse(wˢ_face <= 0, ρqˢ[i, j, k], ρqˢ[i, j, k-1])

    return wʳ_face * ρqʳ_upwind + wˢ_face * ρqˢ_upwind
end

# Sedimentation tendency for ρe: flux divergence of sensible + latent enthalpy,
# plus precipitation drag heating.
@inline function AM.grid_microphysical_tendency(i, j, k, grid, bμp::OMCM, ::Val{:ρe}, ρ, μ, 𝒰, constants, velocities)
    if hasproperty(μ, :wˢ) && hasproperty(μ, :ρqˢ)
        sedimentation = -V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, sedimentation_enthalpy_flux_z, μ.wʳ, μ.ρqʳ, μ.hˡ, μ.wˢ, μ.ρqˢ, μ.hⁱ)
        Fᵐ = ℑzᵃᵃᶜ(i, j, k, grid, sedimentation_mass_flux_z, μ.wʳ, μ.ρqʳ, μ.wˢ, μ.ρqˢ)
    else
        sedimentation = -V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, sedimentation_enthalpy_flux_z, μ.wʳ, μ.ρqʳ, μ.hˡ)
        Fᵐ = ℑzᵃᵃᶜ(i, j, k, grid, sedimentation_mass_flux_z, μ.wʳ, μ.ρqʳ)
    end
    precipitation_drag_heating = -constants.gravitational_acceleration * Fᵐ
    return sedimentation + precipitation_drag_heating
end

# Vertical flux of θˡⁱ contribution from rain sedimentation on z-faces.
# Θˡ = (cˡ T - ℒˡᵣ)/(cᵖᵐ Π) is precomputed and upwinded alongside ρqʳ.
@inline function sedimentation_theta_flux_z(i, j, k, grid, wʳ, ρqʳ, Θˡ)
    @inbounds wʳ_face = wʳ[i, j, k]
    @inbounds ρqʳ_upwind = ifelse(wʳ_face <= 0, ρqʳ[i, j, k], ρqʳ[i, j, k-1])
    @inbounds Θˡ_upwind  = ifelse(wʳ_face <= 0, Θˡ[i, j, k], Θˡ[i, j, k-1])
    return wʳ_face * ρqʳ_upwind * Θˡ_upwind
end

@inline function sedimentation_theta_flux_z(i, j, k, grid, wʳ, ρqʳ, Θˡ, wˢ, ρqˢ, Θⁱ)
    @inbounds wʳ_face = wʳ[i, j, k]
    @inbounds ρqʳ_upwind = ifelse(wʳ_face <= 0, ρqʳ[i, j, k], ρqʳ[i, j, k-1])
    @inbounds Θˡ_r       = ifelse(wʳ_face <= 0, Θˡ[i, j, k], Θˡ[i, j, k-1])

    @inbounds wˢ_face = wˢ[i, j, k]
    @inbounds ρqˢ_upwind = ifelse(wˢ_face <= 0, ρqˢ[i, j, k], ρqˢ[i, j, k-1])
    @inbounds Θⁱ_s       = ifelse(wˢ_face <= 0, Θⁱ[i, j, k], Θⁱ[i, j, k-1])

    return wʳ_face * ρqʳ_upwind * Θˡ_r + wˢ_face * ρqˢ_upwind * Θⁱ_s
end

# Sedimentation tendency for ρθ: upwinded θˡⁱ flux including sensible + latent heat,
# plus precipitation drag heating converted to θˡⁱ units.
@inline function AM.grid_microphysical_tendency(i, j, k, grid, bμp::OMCM, ::Val{:ρθ}, ρ, μ, 𝒰, constants, velocities)
    q = 𝒰.moisture_mass_fractions
    Π = exner_function(𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)

    if hasproperty(μ, :wˢ) && hasproperty(μ, :ρqˢ)
        sedimentation = -V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, sedimentation_theta_flux_z, μ.wʳ, μ.ρqʳ, μ.Θˡ, μ.wˢ, μ.ρqˢ, μ.Θⁱ)
        Fᵐ = ℑzᵃᵃᶜ(i, j, k, grid, sedimentation_mass_flux_z, μ.wʳ, μ.ρqʳ, μ.wˢ, μ.ρqˢ)
    else
        sedimentation = -V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, sedimentation_theta_flux_z, μ.wʳ, μ.ρqʳ, μ.Θˡ)
        Fᵐ = ℑzᵃᵃᶜ(i, j, k, grid, sedimentation_mass_flux_z, μ.wʳ, μ.ρqʳ)
    end
    precipitation_drag_heating = -constants.gravitational_acceleration * Fᵐ / (cᵖᵐ * Π)
    return sedimentation + precipitation_drag_heating
end
