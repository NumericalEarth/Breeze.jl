#####
##### Two-moment microphysics (CloudMicrophysics 2M - Seifert-Beheng 2006)
#####
#
# This file implements two-moment bulk microphysics for cloud liquid and rain,
# tracking both mass and number concentration. Cloud formation uses non-equilibrium
# relaxation toward saturation.
#
# References:
#   - Seifert, A. and Beheng, K.D. (2006). A two-moment cloud microphysics
#     parameterization for mixed-phase clouds. Part 1: Model description.
#     Meteorol. Atmos. Phys., 92, 45-66. https://doi.org/10.1007/s00703-005-0112-4
#   - Morrison, H. and Grabowski, W.W. (2008). A novel approach for representing ice
#     microphysics in models: Description and tests using a kinematic framework.
#     J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1
#
# ## MicrophysicalState pattern
#
# Two-moment schemes use state structs (ℳ) to encapsulate local microphysical
# variables. This enables the same tendency functions to work for both grid-based
# LES and Lagrangian parcel models.
#####

using Breeze.AtmosphereModels: AbstractMicrophysicalState

#####
##### MicrophysicalState struct for two-moment warm-phase microphysics
#####

"""
    WarmPhaseTwoMomentState{FT} <: AbstractMicrophysicalState{FT}

Microphysical state for warm-phase two-moment bulk microphysics.

Contains the local mixing ratios and number concentrations needed to compute
tendencies for cloud liquid and rain following the Seifert-Beheng 2006 scheme.

# Fields
- `qᶜˡ`: Cloud liquid mixing ratio (kg/kg)
- `nᶜˡ`: Cloud liquid number per unit mass (1/kg)
- `qʳ`: Rain mixing ratio (kg/kg)
- `nʳ`: Rain number per unit mass (1/kg)
"""
struct WarmPhaseTwoMomentState{FT} <: AbstractMicrophysicalState{FT}
    qᶜˡ :: FT  # cloud liquid mixing ratio
    nᶜˡ :: FT  # cloud liquid number per unit mass
    qʳ  :: FT  # rain mixing ratio
    nʳ  :: FT  # rain number per unit mass
end

using CloudMicrophysics.Parameters:
    SB2006,
    AirProperties,
    StokesRegimeVelType,
    SB2006VelType,
    Chen2022VelTypeRain

# Use qualified access to avoid conflicts with Microphysics1M
# CM2 is imported as a module alias in BreezeCloudMicrophysicsExt.jl

"""
    TwoMomentCategories{W, AP, LV, RV}

Parameters for two-moment ([Seifert and Beheng, 2006](@cite SeifertBeheng2006)) warm-rain microphysics.

# Fields
- `warm_processes`: [Seifert and Beheng (2006)](@cite SeifertBeheng2006) parameters bundling autoconversion, accretion, self-collection,
  breakup, evaporation, number adjustment, and size distribution parameters
- `air_properties`: `AirProperties` for thermodynamic calculations
- `cloud_liquid_fall_velocity`: `StokesRegimeVelType` for cloud droplet terminal velocity
- `rain_fall_velocity`: `SB2006VelType` or `Chen2022VelTypeRain` for raindrop terminal velocity

# References
* Seifert, A. and Beheng, K. D. (2006). A two-moment cloud microphysics
    parameterization for mixed-phase clouds. Part 1: Model description.
    Meteorol. Atmos. Phys., 92, 45-66. https://doi.org/10.1007/s00703-005-0112-4
"""
struct TwoMomentCategories{W, AP, LV, RV}
    warm_processes :: W
    air_properties :: AP
    cloud_liquid_fall_velocity :: LV
    rain_fall_velocity :: RV
end

Base.summary(::TwoMomentCategories) = "TwoMomentCategories"

"""
    two_moment_cloud_microphysics_categories(FT = Oceananigans.defaults.FloatType;
                                             warm_processes = SB2006(FT),
                                             air_properties = AirProperties(FT),
                                             cloud_liquid_fall_velocity = StokesRegimeVelType(FT),
                                             rain_fall_velocity = SB2006VelType(FT))

Construct `TwoMomentCategories` with default Seifert-Beheng 2006 parameters.

# Keyword arguments
- `warm_processes`: SB2006 parameters for warm-rain microphysics
- `air_properties`: Air properties for thermodynamic calculations
- `cloud_liquid_fall_velocity`: Terminal velocity parameters for cloud droplets (Stokes regime)
- `rain_fall_velocity`: Terminal velocity parameters for rain drops
"""
function two_moment_cloud_microphysics_categories(FT::DataType = Oceananigans.defaults.FloatType;
                                                  warm_processes = SB2006(FT),
                                                  air_properties = AirProperties(FT),
                                                  cloud_liquid_fall_velocity = StokesRegimeVelType(FT),
                                                  rain_fall_velocity = SB2006VelType(FT))

    return TwoMomentCategories(warm_processes, air_properties,
                               cloud_liquid_fall_velocity, rain_fall_velocity)
end

# Type aliases for two-moment microphysics
const CM2MCategories = TwoMomentCategories{<:SB2006, <:AirProperties, <:StokesRegimeVelType}
const TwoMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM2MCategories, <:Any}

# Warm-phase non-equilibrium with 2M precipitation
const WarmPhaseNonEquilibrium2M = BulkMicrophysics{<:WarmPhaseNE, <:CM2MCategories, <:Any}
const WPNE2M = WarmPhaseNonEquilibrium2M

#####
##### MicrophysicalState construction from fields
#####

# Gridless version: takes a NamedTuple of density-weighted scalars
@inline function AtmosphereModels.microphysical_state(bμp::WPNE2M, ρ, μ, 𝒰)
    qᶜˡ = μ.ρqᶜˡ / ρ
    nᶜˡ = μ.ρnᶜˡ / ρ
    qʳ = μ.ρqʳ / ρ
    nʳ = μ.ρnʳ / ρ
    return WarmPhaseTwoMomentState(qᶜˡ, nᶜˡ, qʳ, nʳ)
end

# Grid-indexed version: extracts from Fields
@inline function AtmosphereModels.grid_microphysical_state(i, j, k, grid, bμp::WPNE2M, μ, ρ, 𝒰)
    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds nᶜˡ = μ.nᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]
    @inbounds nʳ = μ.nʳ[i, j, k]
    return WarmPhaseTwoMomentState(qᶜˡ, nᶜˡ, qʳ, nʳ)
end

"""
    TwoMomentCloudMicrophysics(FT = Oceananigans.defaults.FloatType;
                               cloud_formation = NonEquilibriumCloudFormation(nothing, nothing),
                               categories = two_moment_cloud_microphysics_categories(FT),
                               precipitation_boundary_condition = nothing)

Return a `TwoMomentCloudMicrophysics` microphysics scheme for warm-rain precipitation
using the [Seifert and Beheng (2006)](@cite SeifertBeheng2006) two-moment parameterization.

The two-moment scheme tracks both mass and number concentration for cloud liquid and rain,
using CloudMicrophysics.jl 2M processes:
- Condensation/evaporation of cloud liquid (relaxation toward saturation)
- Autoconversion of cloud liquid to rain (mass and number)
- Accretion of cloud liquid by rain (mass and number)
- Cloud liquid self-collection (number only)
- Rain self-collection and breakup (number only)
- Rain evaporation (mass and number)
- Number adjustment to maintain physical mean particle mass bounds
- Terminal velocities (number-weighted and mass-weighted)

Non-equilibrium cloud formation is used, where cloud liquid mass and number are prognostic
variables that evolve via condensation/evaporation and microphysical tendencies.

The prognostic variables are:
- `ρqᶜˡ`: cloud liquid mass density [kg/m³]
- `ρnᶜˡ`: cloud liquid number density [1/m³]
- `ρqʳ`: rain mass density [kg/m³]
- `ρnʳ`: rain number density [1/m³]

# Keyword arguments
- `cloud_formation`: Cloud formation scheme (default: `NonEquilibriumCloudFormation`)
- `categories`: `TwoMomentCategories` containing SB2006 parameters
- `precipitation_boundary_condition`: Controls whether precipitation passes through the bottom boundary.
  - `nothing` (default): Rain exits through the bottom (open boundary)
  - `ImpenetrableBoundaryCondition()`: Rain collects at the bottom (zero terminal velocity at surface)

See the [CloudMicrophysics.jl 2M documentation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics2M/)
for details on the [Seifert and Beheng (2006)](@cite SeifertBeheng2006) scheme.

# References
* Seifert, A. and Beheng, K. D. (2006). A two-moment cloud microphysics
    parameterization for mixed-phase clouds. Part 1: Model description.
    Meteorol. Atmos. Phys., 92, 45-66. https://doi.org/10.1007/s00703-005-0112-4
"""
function TwoMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                    cloud_formation = NonEquilibriumCloudFormation(nothing, nothing),
                                    categories = two_moment_cloud_microphysics_categories(FT),
                                    precipitation_boundary_condition = nothing)

    # Two-moment scheme requires non-equilibrium cloud formation
    if !(cloud_formation isa NonEquilibriumCloudFormation)
        throw(ArgumentError("TwoMomentCloudMicrophysics requires NonEquilibriumCloudFormation. " *
                            "Saturation adjustment is not supported for two-moment schemes."))
    end

    # Materialize condensate formation models from category parameters if needed
    liquid = cloud_formation.liquid
    ice = cloud_formation.ice

    # For liquid, use SB2006 cloud parameters if not specified
    # Default relaxation timescale from CloudLiquid parameters
    liquid = materialize_2m_condensate_formation(liquid, categories)

    # Ice is not yet supported in warm-phase 2M
    if ice !== nothing
        @warn "Ice phase not yet implemented for TwoMomentCloudMicrophysics. " *
              "Cloud ice formation will be ignored."
    end

    cloud_formation = NonEquilibriumCloudFormation(liquid, nothing)

    return BulkMicrophysics(cloud_formation, categories, precipitation_boundary_condition)
end

# Default relaxation timescale for 2M cloud liquid (seconds)
const τ_relax_2m_default = 10.0

# Materialize condensate formation for 2M scheme
materialize_2m_condensate_formation(cf::AbstractCondensateFormation, categories) = cf
materialize_2m_condensate_formation(::Nothing, categories) = ConstantRateCondensateFormation(1 / τ_relax_2m_default)
materialize_2m_condensate_formation(::Any, categories) = ConstantRateCondensateFormation(1 / τ_relax_2m_default)

#####
##### Default fallbacks for TwoMomentCloudMicrophysics
#####

# Default fallback for tendencies (state-based)
@inline AtmosphereModels.microphysical_tendency(bμp::TwoMomentCloudMicrophysics, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

# Default fallback for velocities
@inline AtmosphereModels.microphysical_velocities(bμp::TwoMomentCloudMicrophysics, μ, name) = nothing

#####
##### Relaxation timescale for non-equilibrium cloud formation
#####

@inline liquid_relaxation_timescale(cloud_formation, categories::TwoMomentCategories) = 1 / cloud_formation.liquid.rate

#####
##### Prognostic field names
#####

AtmosphereModels.prognostic_field_names(::WPNE2M) = (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ)

#####
##### Field materialization
#####

const two_moment_center_field_names = (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :qᵛ, :qˡ, :qᶜˡ, :qʳ, :nᶜˡ, :nʳ)

function AtmosphereModels.materialize_microphysical_fields(bμp::WPNE2M, grid, bcs)
    center_fields = center_field_tuple(grid, two_moment_center_field_names...)

    # Terminal velocities (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    w_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)

    # Cloud liquid terminal velocity (mass-weighted)
    wᶜˡ = ZFaceField(grid; boundary_conditions=w_bcs)
    # Cloud liquid terminal velocity (number-weighted)
    wᶜˡₙ = ZFaceField(grid; boundary_conditions=w_bcs)
    # Rain terminal velocity (mass-weighted)
    wʳ = ZFaceField(grid; boundary_conditions=w_bcs)
    # Rain terminal velocity (number-weighted)
    wʳₙ = ZFaceField(grid; boundary_conditions=w_bcs)

    return (; zip(two_moment_center_field_names, center_fields)..., wᶜˡ, wᶜˡₙ, wʳ, wʳₙ)
end

#####
##### Update microphysical fields (diagnostics + terminal velocities)
#####

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, bμp::WPNE2M, ρ, 𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    categories = bμp.categories

    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ  # cloud liquid from prognostic field
        nᶜˡ = μ.ρnᶜˡ[i, j, k] / ρ  # cloud liquid number per unit mass
        qʳ = μ.ρqʳ[i, j, k] / ρ
        nʳ = μ.ρnʳ[i, j, k] / ρ

        # Update diagnostic fields
        μ.qᵛ[i, j, k] = q.vapor
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qʳ[i, j, k] = qʳ
        μ.qˡ[i, j, k] = qᶜˡ + qʳ  # total liquid
        μ.nᶜˡ[i, j, k] = nᶜˡ
        μ.nʳ[i, j, k] = nʳ
    end

    update_2m_terminal_velocities!(μ, i, j, k, bμp, categories, ρ)

    return nothing
end

@inline function update_2m_terminal_velocities!(μ, i, j, k, bμp, categories, ρ)
    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds nᶜˡ = μ.nᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]
    @inbounds nʳ = μ.nʳ[i, j, k]

    # Number density in [1/m³] for CloudMicrophysics functions
    Nᶜˡ = ρ * max(0, nᶜˡ)
    Nʳ = ρ * max(0, nʳ)

    sb = categories.warm_processes

    # Cloud liquid terminal velocities: (number-weighted, mass-weighted)
    vt_cloud = CM2.cloud_terminal_velocity(sb.pdf_c, categories.cloud_liquid_fall_velocity,
                                           max(0, qᶜˡ), ρ, Nᶜˡ)
    wᶜˡₙ = -vt_cloud[1]  # number-weighted, negative = downward
    wᶜˡ = -vt_cloud[2]   # mass-weighted

    # Rain terminal velocities: (number-weighted, mass-weighted)
    vt_rain = CM2.rain_terminal_velocity(sb, categories.rain_fall_velocity,
                                         max(0, qʳ), ρ, Nʳ)
    wʳₙ = -vt_rain[1]  # number-weighted
    wʳ = -vt_rain[2]   # mass-weighted

    # Apply bottom boundary condition
    bc = bμp.precipitation_boundary_condition
    wᶜˡ₀ = bottom_terminal_velocity(bc, wᶜˡ)
    wᶜˡₙ₀ = bottom_terminal_velocity(bc, wᶜˡₙ)
    wʳ₀ = bottom_terminal_velocity(bc, wʳ)
    wʳₙ₀ = bottom_terminal_velocity(bc, wʳₙ)

    @inbounds begin
        μ.wᶜˡ[i, j, k] = ifelse(k == 1, wᶜˡ₀, wᶜˡ)
        μ.wᶜˡₙ[i, j, k] = ifelse(k == 1, wᶜˡₙ₀, wᶜˡₙ)
        μ.wʳ[i, j, k] = ifelse(k == 1, wʳ₀, wʳ)
        μ.wʳₙ[i, j, k] = ifelse(k == 1, wʳₙ₀, wʳₙ)
    end

    return nothing
end

#####
##### Moisture fraction computation
#####

@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, bμp::WPNE2M, ρ, qᵗ, μ)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Gridless version for parcel models
@inline function AtmosphereModels.moisture_fractions(bμp::WPNE2M, ℳ::WarmPhaseTwoMomentState, qᵗ)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

#####
##### Thermodynamic state adjustment
#####

# Non-equilibrium: no adjustment (cloud liquid is prognostic)
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp::WPNE2M, qᵗ, constants) = 𝒰₀

#####
##### Microphysical velocities for advection
#####

# Cloud liquid mass: use mass-weighted terminal velocity
@inline function AtmosphereModels.microphysical_velocities(bμp::WPNE2M, μ, ::Val{:ρqᶜˡ})
    wᶜˡ = μ.wᶜˡ
    return (; u = ZeroField(), v = ZeroField(), w = wᶜˡ)
end

# Cloud liquid number: use number-weighted terminal velocity
@inline function AtmosphereModels.microphysical_velocities(bμp::WPNE2M, μ, ::Val{:ρnᶜˡ})
    wᶜˡₙ = μ.wᶜˡₙ
    return (; u = ZeroField(), v = ZeroField(), w = wᶜˡₙ)
end

# Rain mass: use mass-weighted terminal velocity
@inline function AtmosphereModels.microphysical_velocities(bμp::WPNE2M, μ, ::Val{:ρqʳ})
    wʳ = μ.wʳ
    return (; u = ZeroField(), v = ZeroField(), w = wʳ)
end

# Rain number: use number-weighted terminal velocity
@inline function AtmosphereModels.microphysical_velocities(bμp::WPNE2M, μ, ::Val{:ρnʳ})
    wʳₙ = μ.wʳₙ
    return (; u = ZeroField(), v = ZeroField(), w = wʳₙ)
end

#####
##### Microphysical tendencies
#####

# Numerical timescale for limiting negative-value relaxation
const τⁿᵘᵐ_2m = 10.0  # seconds

#####
##### Cloud liquid mass tendency (ρqᶜˡ) - state-based
#####

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρqᶜˡ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    categories = bμp.categories
    sb = categories.warm_processes
    τᶜˡ = liquid_relaxation_timescale(bμp.cloud_formation, categories)

    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nᶜˡ = ℳ.nᶜˡ

    # Number density [1/m³]
    Nᶜˡ = ρ * max(0, nᶜˡ)

    # Thermodynamic state
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Saturation specific humidity
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())

    # Condensation/evaporation rate (relaxation to saturation)
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)

    # Autoconversion: cloud liquid → rain
    au = CM2.autoconversion(sb.acnv, sb.pdf_c, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    Sᵃᶜⁿᵛ = au.dq_lcl_dt  # negative (sink for cloud)

    # Accretion: cloud liquid captured by falling rain
    ac = CM2.accretion(sb, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    Sᵃᶜᶜ = ac.dq_lcl_dt  # negative (sink for cloud)

    # Total tendency
    ΣρS = ρ * (Sᶜᵒⁿᵈ + Sᵃᶜⁿᵛ + Sᵃᶜᶜ)

    # Numerical relaxation for negative values
    ρSⁿᵘᵐ = -ρ * qᶜˡ / τⁿᵘᵐ_2m

    return ifelse(qᶜˡ >= 0, ΣρS, ρSⁿᵘᵐ)
end

#####
##### Cloud liquid number tendency (ρnᶜˡ) - state-based
#####

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρnᶜˡ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    categories = bμp.categories
    sb = categories.warm_processes

    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nᶜˡ = ℳ.nᶜˡ

    # Number density [1/m³]
    Nᶜˡ = ρ * max(0, nᶜˡ)

    # Autoconversion: reduces cloud droplet number
    au = CM2.autoconversion(sb.acnv, sb.pdf_c, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    dNᶜˡ_au = au.dN_lcl_dt  # [1/m³/s], negative

    # Cloud liquid self-collection: droplets collide to form larger droplets (number sink)
    dNᶜˡ_sc = CM2.cloud_liquid_self_collection(sb.acnv, sb.pdf_c, max(0, qᶜˡ), ρ, dNᶜˡ_au)

    # Accretion: cloud droplets collected by rain
    ac = CM2.accretion(sb, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    dNᶜˡ_ac = ac.dN_lcl_dt  # [1/m³/s], negative

    # Number adjustment to keep mean mass within physical bounds
    dNᶜˡ_adj_up = CM2.number_increase_for_mass_limit(sb.numadj, sb.pdf_c.xc_max, max(0, qᶜˡ), ρ, Nᶜˡ)
    dNᶜˡ_adj_dn = CM2.number_decrease_for_mass_limit(sb.numadj, sb.pdf_c.xc_min, max(0, qᶜˡ), ρ, Nᶜˡ)

    # Total tendency (convert from [1/m³/s] to [1/kg/s] by dividing by ρ, then multiply back)
    # Actually, we're computing ρnᶜˡ tendency, so we need [1/m³/s] which is already what we have
    Σ_dNᶜˡ = dNᶜˡ_au + dNᶜˡ_sc + dNᶜˡ_ac + dNᶜˡ_adj_up + dNᶜˡ_adj_dn

    # Numerical relaxation for negative values
    Sⁿᵘᵐ = -Nᶜˡ / τⁿᵘᵐ_2m

    return ifelse(nᶜˡ >= 0, Σ_dNᶜˡ, Sⁿᵘᵐ)
end

#####
##### Rain mass tendency (ρqʳ) - state-based
#####

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρqʳ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    categories = bμp.categories
    sb = categories.warm_processes

    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nᶜˡ = ℳ.nᶜˡ
    nʳ = ℳ.nʳ

    # Number densities [1/m³]
    Nᶜˡ = ρ * max(0, nᶜˡ)
    Nʳ = ρ * max(0, nʳ)

    # Autoconversion: cloud liquid → rain (source for rain)
    au = CM2.autoconversion(sb.acnv, sb.pdf_c, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    Sᵃᶜⁿᵛ = au.dq_rai_dt  # positive (source for rain)

    # Accretion: cloud liquid captured by falling rain (source for rain)
    ac = CM2.accretion(sb, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    Sᵃᶜᶜ = ac.dq_rai_dt  # positive (source for rain)

    # Rain evaporation (in subsaturated air)
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions

    evap = rain_evaporation_2m(sb, categories.air_properties, q, max(0, qʳ), ρ, Nʳ, T, constants)
    Sᵉᵛᵃᵖ = evap.evap_rate_1  # [kg/kg/s], negative (sink for rain)

    # Limit evaporation to available rain
    Sᵉᵛᵃᵖ_min = -max(0, qʳ) / τⁿᵘᵐ_2m
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, Sᵉᵛᵃᵖ_min)

    # Total tendency
    ΣρS = ρ * (Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ)

    # Numerical relaxation for negative values
    ρSⁿᵘᵐ = -ρ * qʳ / τⁿᵘᵐ_2m

    return ifelse(qʳ >= 0, ΣρS, ρSⁿᵘᵐ)
end

#####
##### Rain number tendency (ρnʳ) - state-based
#####

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρnʳ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    categories = bμp.categories
    sb = categories.warm_processes

    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nᶜˡ = ℳ.nᶜˡ
    nʳ = ℳ.nʳ

    # Number densities [1/m³]
    Nᶜˡ = ρ * max(0, nᶜˡ)
    Nʳ = ρ * max(0, nʳ)

    # Autoconversion: creates rain drops from cloud droplet pairs
    au = CM2.autoconversion(sb.acnv, sb.pdf_c, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    dNʳ_au = au.dN_rai_dt  # [1/m³/s], positive (source)

    # Rain self-collection: raindrops collide to form larger drops (number sink)
    dNʳ_sc = CM2.rain_self_collection(sb.pdf_r, sb.self, max(0, qʳ), ρ, Nʳ)  # negative

    # Rain breakup: large drops break into smaller drops (number source)
    dNʳ_br = CM2.rain_breakup(sb.pdf_r, sb.brek, max(0, qʳ), ρ, Nʳ, dNʳ_sc)  # positive

    # Rain evaporation (number change)
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions

    evap = rain_evaporation_2m(sb, categories.air_properties, q, max(0, qʳ), ρ, Nʳ, T, constants)
    dNʳ_evap = evap.evap_rate_0  # [1/m³/s], negative

    # Number adjustment to keep mean mass within physical bounds
    dNʳ_adj_up = CM2.number_increase_for_mass_limit(sb.numadj, sb.pdf_r.xr_max, max(0, qʳ), ρ, Nʳ)
    dNʳ_adj_dn = CM2.number_decrease_for_mass_limit(sb.numadj, sb.pdf_r.xr_min, max(0, qʳ), ρ, Nʳ)

    # Total tendency
    Σ_dNʳ = dNʳ_au + dNʳ_sc + dNʳ_br + dNʳ_evap + dNʳ_adj_up + dNʳ_adj_dn

    # Numerical relaxation for negative values
    Sⁿᵘᵐ = -Nʳ / τⁿᵘᵐ_2m

    return ifelse(nʳ >= 0, Σ_dNʳ, Sⁿᵘᵐ)
end
