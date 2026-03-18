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
#
# Note: WarmPhaseTwoMomentState is defined in cloud_microphysics_translations.jl
#####

using CloudMicrophysics.Parameters:
    SB2006,
    AirProperties,
    StokesRegimeVelType,
    SB2006VelType,
    Chen2022VelTypeRain,
    AerosolActivationParameters

# Use qualified access to avoid conflicts with Microphysics1M
# CM2 is imported as a module alias in BreezeCloudMicrophysicsExt.jl
# CMAM (AerosolModel) is imported in BreezeCloudMicrophysicsExt.jl
# erf from SpecialFunctions is imported for aerosol activation calculations

#####
##### Aerosol activation for two-moment microphysics
#####
#
# Aerosol activation provides the source term for cloud droplet number concentration.
# Without activation, there is no physical mechanism to create cloud droplets.
#
# References:
#   - Abdul-Razzak, H. and Ghan, S.J. (2000). A parameterization of aerosol activation:
#     2. Multiple aerosol types. J. Geophys. Res., 105(D5), 6837-6844.
#   - Petters, M.D. and Kreidenweis, S.M. (2007). A single parameter representation of
#     hygroscopic growth and cloud condensation nucleus activity. Atmos. Chem. Phys., 7, 1961-1971.
#####

"""
    default_aerosol_activation(FT = Float64; τⁿᵘᶜ = 1)

Create a default `AerosolActivation` representing a typical continental aerosol population.

The default distribution is a single mode with:
- Mean dry radius: 0.05 μm (50 nm)
- Geometric standard deviation: 2.0
- Number concentration: 100 cm⁻³ (100 × 10⁶ m⁻³)
- Hygroscopicity κ: 0.5 (typical for ammonium sulfate)

# Keyword arguments
- `τⁿᵘᶜ`: Nucleation timescale [s] for converting activation deficit to rate (default: 1s).
  Controls how quickly the cloud droplet number relaxes toward the target activated number.

This provides sensible out-of-the-box behavior for two-moment microphysics.
Users can customize the aerosol population by constructing their own `AerosolActivation`.

# Example

```julia
# Use default aerosol
microphysics = TwoMomentCloudMicrophysics()

# Custom aerosol: marine (fewer, larger particles)
marine_mode = CMAM.Mode_κ(0.08e-6, 1.8, 50e6, (1.0,), (1.0,), (0.058,), (1.0,))
marine_aerosol = AerosolActivation(
    AerosolActivationParameters(Float64),
    CMAM.AerosolDistribution((marine_mode,)),
    1  # τⁿᵘᶜ = 1s
)
microphysics = TwoMomentCloudMicrophysics(aerosol_activation = marine_aerosol)

# Disable aerosol activation (not recommended)
microphysics = TwoMomentCloudMicrophysics(aerosol_activation = nothing)
```
"""
function default_aerosol_activation(FT::DataType = Float64; τⁿᵘᶜ = 1)
    # Default continental aerosol mode using κ-Köhler theory
    # Mode_κ(r_dry, stdev, N, vol_mix_ratio, mass_mix_ratio, molar_mass, kappa)
    r_dry = 0.05e-6           # 50 nm dry radius
    stdev = 2.0               # geometric standard deviation
    Nᵃ₀ = 100e6               # 100 cm⁻³
    vol_mix_ratio = (1.0,)    # single component
    mass_mix_ratio = (1.0,)
    molar_mass = (0.132,)     # ammonium sulfate ~132 g/mol
    kappa = (0.5,)            # hygroscopicity

    mode = CMAM.Mode_κ(r_dry, stdev, Nᵃ₀, vol_mix_ratio, mass_mix_ratio, molar_mass, kappa)
    aerosol_distribution = CMAM.AerosolDistribution((mode,))

    activation_parameters = AerosolActivationParameters(FT)

    return AerosolActivation(activation_parameters, aerosol_distribution, FT(τⁿᵘᶜ))
end


"""
    TwoMomentCategories{W, AP, LV, RV, AA, TL}

Parameters for two-moment ([Seifert and Beheng, 2006](@cite SeifertBeheng2006)) warm-rain microphysics.

# Fields
- `warm_processes`: [Seifert and Beheng (2006)](@cite SeifertBeheng2006) parameters bundling autoconversion, accretion, self-collection,
  breakup, evaporation, number adjustment, and size distribution parameters
- `air_properties`: `AirProperties` for thermodynamic calculations
- `cloud_liquid_fall_velocity`: `StokesRegimeVelType` for cloud droplet terminal velocity
- `rain_fall_velocity`: `SB2006VelType` or `Chen2022VelTypeRain` for raindrop terminal velocity
- `aerosol_activation`: `AerosolActivation` parameters for cloud droplet nucleation (or `nothing` to disable)
- `τⁿᵘᵐ`: Timescale [s] for per-reservoir tendency limiting (default: 10)

# References

* Abdul-Razzak, H. and Ghan, S.J. (2000). A parameterization of aerosol activation:
  2. Multiple aerosol types. J. Geophys. Res., 105(D5), 6837-6844.
* Seifert, A. and Beheng, K. D. (2006). A two-moment cloud microphysics
    parameterization for mixed-phase clouds. Part 1: Model description.
    Meteorol. Atmos. Phys., 92, 45-66. https://doi.org/10.1007/s00703-005-0112-4
"""
struct TwoMomentCategories{W, AP, LV, RV, AA, TL}
    warm_processes :: W
    air_properties :: AP
    cloud_liquid_fall_velocity :: LV
    rain_fall_velocity :: RV
    aerosol_activation :: AA
    τⁿᵘᵐ :: TL
end

Base.summary(::TwoMomentCategories) = "TwoMomentCategories"

"""
    two_moment_cloud_microphysics_categories(FT = Oceananigans.defaults.FloatType;
                                             warm_processes = SB2006(FT),
                                             air_properties = AirProperties(FT),
                                             cloud_liquid_fall_velocity = StokesRegimeVelType(FT),
                                             rain_fall_velocity = SB2006VelType(FT),
                                             aerosol_activation = default_aerosol_activation(FT))

Construct `TwoMomentCategories` with default Seifert-Beheng 2006 parameters and aerosol activation.

# Keyword arguments
- `warm_processes`: SB2006 parameters for warm-rain microphysics
- `air_properties`: Air properties for thermodynamic calculations
- `cloud_liquid_fall_velocity`: Terminal velocity parameters for cloud droplets (Stokes regime)
- `rain_fall_velocity`: Terminal velocity parameters for rain drops
- `aerosol_activation`: Aerosol activation parameters (default: continental aerosol).
  Set to `nothing` to disable activation (not recommended for physical simulations).
- `τⁿᵘᵐ`: Timescale [s] for per-reservoir tendency limiting.
  Must satisfy `τⁿᵘᵐ ≥ Δt` to prevent reservoir overdraw.
  Default: 10 seconds.
"""
function two_moment_cloud_microphysics_categories(FT::DataType = Oceananigans.defaults.FloatType;
                                                  warm_processes = SB2006(FT),
                                                  air_properties = AirProperties(FT),
                                                  cloud_liquid_fall_velocity = StokesRegimeVelType(FT),
                                                  rain_fall_velocity = SB2006VelType(FT),
                                                  aerosol_activation = default_aerosol_activation(FT),
                                                  τⁿᵘᵐ = FT(10))

    return TwoMomentCategories(warm_processes, air_properties,
                               cloud_liquid_fall_velocity, rain_fall_velocity,
                               aerosol_activation, τⁿᵘᵐ)
end

# Type aliases for two-moment microphysics
const CM2MCategories = TwoMomentCategories{<:SB2006, <:AirProperties, <:StokesRegimeVelType, <:Any, <:Any, <:Any}
const TwoMomentCloudMicrophysics = BulkMicrophysics{<:Any, <:CM2MCategories, <:Any}

# Warm-phase non-equilibrium with 2M precipitation
const WarmPhaseNonEquilibrium2M = BulkMicrophysics{<:WarmPhaseNE, <:CM2MCategories, <:Any}
const WPNE2M = WarmPhaseNonEquilibrium2M


#####
##### Initial aerosol number from aerosol distribution
#####

function AtmosphereModels.initial_aerosol_number(microphysics::TwoMomentCloudMicrophysics)
    aa = microphysics.categories.aerosol_activation
    aa isa Nothing && return 0
    return sum(mode.N for mode in aa.aerosol_distribution.modes)
end

#####
##### MicrophysicalState construction from fields
#####

# Gridless version: takes density, prognostic NamedTuple, thermodynamic state, and velocities
@inline function AtmosphereModels.microphysical_state(bμp::WPNE2M, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    nᶜˡ = μ.ρnᶜˡ / ρ
    qʳ = μ.ρqʳ / ρ
    nʳ = μ.ρnʳ / ρ
    nᵃ = μ.ρnᵃ / ρ
    return WarmPhaseTwoMomentState(qᶜˡ, nᶜˡ, qʳ, nʳ, nᵃ, velocities)
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
- **Aerosol activation**: Creates cloud droplets when supersaturation develops (enabled by default)
- Condensation/evaporation of cloud liquid (relaxation toward saturation)
- Autoconversion of cloud liquid to rain (mass and number)
- Accretion of cloud liquid by rain (mass and number)
- Cloud liquid self-collection (number only)
- Rain self-collection and breakup (number only)
- Rain evaporation (mass and number)
- Number adjustment to maintain physical mean particle mass bounds
- Terminal velocities (number-weighted and mass-weighted)

Non-equilibrium cloud formation is used, where cloud liquid mass and number are prognostic
variables that evolve via condensation/evaporation, aerosol activation, and microphysical tendencies.

The prognostic variables are:
- `ρqᶜˡ`: cloud liquid mass density [kg/m³]
- `ρnᶜˡ`: cloud liquid number density [1/m³]
- `ρqʳ`: rain mass density [kg/m³]
- `ρnʳ`: rain number density [1/m³]

## Aerosol Activation

Aerosol activation is **enabled by default** and provides the physical source term for cloud
droplet number concentration. Without activation, cloud droplets cannot form. The default
aerosol population represents typical continental conditions (~100 cm⁻³).

To customize the aerosol population, pass a custom `categories` with different `aerosol_activation`:

```julia
# Marine aerosol (fewer, more hygroscopic particles)
marine_mode = CMAM.Mode_κ(0.08e-6, 1.8, 50e6, (1.0,), (1.0,), (0.058,), (1.0,))
marine_activation = AerosolActivation(
    AerosolActivationParameters(Float64),
    CMAM.AerosolDistribution((marine_mode,))
)
categories = two_moment_cloud_microphysics_categories(aerosol_activation = marine_activation)
microphysics = TwoMomentCloudMicrophysics(categories = categories)
```

# Keyword arguments
- `cloud_formation`: Cloud formation scheme (default: `NonEquilibriumCloudFormation`)
- `categories`: `TwoMomentCategories` containing SB2006 and aerosol activation parameters
- `precipitation_boundary_condition`: Controls whether precipitation passes through the bottom boundary.
  - `nothing` (default): Rain exits through the bottom (open boundary)
  - `ImpenetrableBoundaryCondition()`: Rain collects at the bottom (zero terminal velocity at surface)

See the [CloudMicrophysics.jl 2M documentation](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics2M/)
for details on the [Seifert and Beheng (2006)](@cite SeifertBeheng2006) scheme.

# References

* Seifert, A. and Beheng, K. D. (2006). A two-moment cloud microphysics
    parameterization for mixed-phase clouds. Part 1: Model description.
    Meteorol. Atmos. Phys., 92, 45-66. https://doi.org/10.1007/s00703-005-0112-4
* Abdul-Razzak, H. and Ghan, S.J. (2000). A parameterization of aerosol activation:
  2. Multiple aerosol types. J. Geophys. Res., 105(D5), 6837-6844.
"""
function TwoMomentCloudMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                                    cloud_formation = NonEquilibriumCloudFormation(nothing, nothing),
                                    categories = two_moment_cloud_microphysics_categories(FT),
                                    precipitation_boundary_condition = nothing,
                                    negative_moisture_correction = false)

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

    return BulkMicrophysics(cloud_formation, categories, precipitation_boundary_condition, negative_moisture_correction)
end

# Default relaxation timescale for 2M cloud liquid (seconds)
const τ_relax_2m_default = 10

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

AtmosphereModels.prognostic_field_names(::WPNE2M) = (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρnᵃ)

# Negative moisture correction chain: rain ← cloud ← vapor
AtmosphereModels.correction_moisture_fields(::WPNE2M, μ) = (μ.ρqʳ, μ.ρqᶜˡ)

# Zero number concentrations when corresponding mass is zeroed by borrowing
AtmosphereModels.correction_number_mass_pairs(::WPNE2M, μ) = ((μ.ρnʳ, μ.ρqʳ), (μ.ρnᶜˡ, μ.ρqᶜˡ))

# Clamp negative number concentrations to zero after advection
AtmosphereModels.correction_number_fields(::WPNE2M, μ) = (μ.ρnᶜˡ, μ.ρnʳ, μ.ρnᵃ)

#####
##### Field materialization
#####

const two_moment_center_field_names = (:ρqᶜˡ, :ρnᶜˡ, :ρqʳ, :ρnʳ, :ρnᵃ, :qᵛ, :qˡ, :qᶜˡ, :qʳ, :nᶜˡ, :nʳ, :nᵃ)

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
        nᵃ = μ.ρnᵃ[i, j, k] / ρ    # aerosol number per unit mass

        # Update diagnostic fields
        μ.qᵛ[i, j, k] = q.vapor
        μ.qᶜˡ[i, j, k] = qᶜˡ
        μ.qʳ[i, j, k] = qʳ
        μ.qˡ[i, j, k] = qᶜˡ + qʳ  # total liquid
        μ.nᶜˡ[i, j, k] = nᶜˡ
        μ.nʳ[i, j, k] = nʳ
        μ.nᵃ[i, j, k] = nᵃ
    end

    update_2m_terminal_velocities!(μ, i, j, k, bμp, categories, ρ)

    return nothing
end

@inline function update_2m_terminal_velocities!(μ, i, j, k, bμp, categories, ρ)
    @inbounds qᶜˡ = μ.qᶜˡ[i, j, k]
    @inbounds nᶜˡ = μ.nᶜˡ[i, j, k]
    @inbounds qʳ = μ.qʳ[i, j, k]
    @inbounds nʳ = μ.nʳ[i, j, k]

    # Number density in [1/m³] for CloudMicrophysics functions.
    # Enforce minimum Nᶜˡ so that the mean droplet mass qᶜˡ*ρ/Nᶜˡ ≤ xc_max.
    # Without this, advection can create grid points with qᶜˡ > 0 and nᶜˡ ≈ 0,
    # producing unphysically large terminal velocities that violate the CFL condition.
    sb = categories.warm_processes
    qᶜˡ⁺ = max(0, qᶜˡ)
    Nᶜˡ_min = ρ * qᶜˡ⁺ / sb.pdf_c.xc_max
    Nᶜˡ = max(ρ * max(0, nᶜˡ), Nᶜˡ_min)

    qʳ⁺ = max(0, qʳ)
    Nʳ_min = ρ * qʳ⁺ / sb.pdf_r.xr_max
    Nʳ = max(ρ * max(0, nʳ), Nʳ_min)

    # Cloud liquid terminal velocities: (number-weighted, mass-weighted)
    𝕎_cl = CM2.cloud_terminal_velocity(sb.pdf_c, categories.cloud_liquid_fall_velocity,
                                       qᶜˡ⁺, ρ, Nᶜˡ)

    wᶜˡₙ = -𝕎_cl[1]  # number-weighted, negative = downward
    wᶜˡ = -𝕎_cl[2]   # mass-weighted

    # Rain terminal velocities: (number-weighted, mass-weighted)
    𝕎  = CM2.rain_terminal_velocity(sb, categories.rain_fall_velocity, qʳ⁺, ρ, Nʳ)

    wʳₙ = -𝕎[1]  # number-weighted
    wʳ = -𝕎[2]   # mass-weighted

    # Apply bottom boundary condition
    bc = bμp.precipitation_boundary_condition
    wᶜˡ₀  = bottom_terminal_velocity(bc, wᶜˡ)
    wᶜˡₙ₀ = bottom_terminal_velocity(bc, wᶜˡₙ)
    wʳ₀   = bottom_terminal_velocity(bc, wʳ)
    wʳₙ₀  = bottom_terminal_velocity(bc, wʳₙ)

    @inbounds begin
        μ.wᶜˡ[i, j, k]  = ifelse(k == 1, wᶜˡ₀,  wᶜˡ)
        μ.wᶜˡₙ[i, j, k] = ifelse(k == 1, wᶜˡₙ₀, wᶜˡₙ)
        μ.wʳ[i, j, k]   = ifelse(k == 1, wʳ₀,   wʳ)
        μ.wʳₙ[i, j, k]  = ifelse(k == 1, wʳₙ₀,  wʳₙ)
    end

    return nothing
end

#####
##### specific_prognostic_moisture_from_total: convert qᵗ to qᵛᵉ
#####

# NE two-moment: qᵛ = qᵗ - qᶜˡ - qʳ (subtract all condensate)
@inline AtmosphereModels.specific_prognostic_moisture_from_total(bμp::WPNE2M, qᵗ, ℳ::WarmPhaseTwoMomentState) = max(0, qᵗ - ℳ.qᶜˡ - ℳ.qʳ)

#####
##### Moisture fraction computation
#####

@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, bμp::WPNE2M, ρ, qᵛ, μ)
    qᶜˡ = @inbounds μ.ρqᶜˡ[i, j, k] / ρ
    qʳ = @inbounds μ.ρqʳ[i, j, k] / ρ
    qˡ = qᶜˡ + qʳ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Gridless version for parcel models.
# Input qᵛᵉ is scheme-dependent specific moisture (vapor for non-equilibrium).
@inline function AtmosphereModels.moisture_fractions(bμp::WPNE2M, ℳ::WarmPhaseTwoMomentState, qᵛᵉ)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    return MoistureMassFractions(qᵛᵉ, qˡ)
end

#####
##### Thermodynamic state adjustment
#####

# Non-equilibrium: no adjustment (cloud liquid is prognostic)
@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp::WPNE2M, qᵛ, constants) = 𝒰₀

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

#####
##### Microphysical tendencies for warm-phase non-equilibrium 2M (WPNE2M)
#####
#
# Computes all mass and number tendencies in a single function so that
# coupled sink limiting can enforce the same scaling factor on both mass
# and number for each reservoir (cloud, rain). This prevents microphysics
# from depleting mass faster than number or vice versa.
#
# Conservation: d(ρqᵛ)/dt + d(ρqᶜˡ)/dt + d(ρqʳ)/dt = 0 (from phase changes)
#
# Activation and condensation are sequentially coupled: both consume vapor from
# the same supersaturation budget. Activation forms new droplets first; condensation
# then grows existing droplets with the remaining supersaturation.
#####

@inline function wpne2m_tendencies(bμp::WPNE2M, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    categories = bμp.categories
    sb = categories.warm_processes
    τᶜˡ = liquid_relaxation_timescale(bμp.cloud_formation, categories)
    τⁿᵘᵐ = categories.τⁿᵘᵐ

    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nᶜˡ = ℳ.nᶜˡ
    nʳ = ℳ.nʳ
    nᵃ = ℳ.nᵃ

    Nᶜˡ = ρ * max(0, nᶜˡ)
    Nʳ = ρ * max(0, nʳ)
    Nᵃ = ρ * max(0, nᵃ)

    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())

    # ===== Process rates =====

    # Aerosol activation (vapor → new cloud droplets)
    Sᵃᶜᵗ = aerosol_activation_mass_tendency(categories.aerosol_activation, categories.air_properties,
                                             ρ, ℳ, 𝒰, constants)
    dNᶜˡ_act = aerosol_activation_tendency(categories.aerosol_activation, categories.air_properties,
                                            ρ, ℳ, 𝒰, constants)

    # Condensation on existing droplets, budget reduced by activation
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / τᶜˡ
    Sᶜᵒⁿᵈ_eff = max(Sᶜᵒⁿᵈ - Sᵃᶜᵗ, Sᶜᵒⁿᵈ_min)

    # Rain evaporation (mass and number)
    evap = rain_evaporation_2m(sb, categories.air_properties, q, max(0, qʳ), ρ, Nʳ, T, constants)
    Sᵉᵛᵃᵖ = max(evap.evap_rate_1, -max(0, qʳ) / τⁿᵘᵐ)
    dNʳ_evap = evap.evap_rate_0

    # Collection: cloud liquid ↔ rain
    au = CM2.autoconversion(sb.acnv, sb.pdf_c, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)
    ac = CM2.accretion(sb, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)

    # Cloud self-collection (number only)
    dNᶜˡ_sc = CM2.cloud_liquid_self_collection(sb.acnv, sb.pdf_c, max(0, qᶜˡ), ρ, au.dN_lcl_dt)

    # Rain self-collection and breakup (number only)
    dNʳ_sc = CM2.rain_self_collection(sb.pdf_r, sb.self, max(0, qʳ), ρ, Nʳ)
    dNʳ_br = CM2.rain_breakup(sb.pdf_r, sb.brek, max(0, qʳ), ρ, Nʳ, dNʳ_sc)

    # Number adjustment to keep mean mass within physical bounds (Horn 2012)
    dNᶜˡ_adj_up = CM2.number_increase_for_mass_limit(sb.numadj, sb.pdf_c.xc_max, max(0, qᶜˡ), ρ, Nᶜˡ)
    dNᶜˡ_adj_dn = CM2.number_decrease_for_mass_limit(sb.numadj, sb.pdf_c.xc_min, max(0, qᶜˡ), ρ, Nᶜˡ)
    dNʳ_adj_up = CM2.number_increase_for_mass_limit(sb.numadj, sb.pdf_r.xr_max, max(0, qʳ), ρ, Nʳ)
    dNʳ_adj_dn = CM2.number_decrease_for_mass_limit(sb.numadj, sb.pdf_r.xr_min, max(0, qʳ), ρ, Nʳ)

    # ===== Coupled per-reservoir sink limiting =====
    #
    # For each reservoir we compute separate limiting factors for mass and number,
    # then use the MOST RESTRICTIVE (minimum) for both. This ensures microphysics
    # cannot deplete mass faster than number or vice versa.

    ε = eps(typeof(qᵛ))

    # Vapor (mass only — no number counterpart, no coupling needed)
    vapor_sink = max(0, Sᶜᵒⁿᵈ_eff) + max(0, Sᵃᶜᵗ)
    max_vapor_rate = max(0, qᵛ) / τⁿᵘᵐ
    α_vapor = ifelse(vapor_sink > max_vapor_rate, max_vapor_rate / max(vapor_sink, ε), one(qᵛ))

    # Cloud liquid: coupled mass + number
    cloud_mass_sink   = -au.dq_lcl_dt - ac.dq_lcl_dt + max(0, -Sᶜᵒⁿᵈ_eff)
    max_cloud_mass    = max(0, qᶜˡ) / τⁿᵘᵐ
    α_cloud_mass      = ifelse(cloud_mass_sink > max_cloud_mass,
                               max_cloud_mass / max(cloud_mass_sink, ε), one(qᶜˡ))

    cloud_number_sink = -au.dN_lcl_dt - dNᶜˡ_sc - ac.dN_lcl_dt - dNᶜˡ_adj_dn
    max_cloud_number  = max(0, Nᶜˡ) / τⁿᵘᵐ
    α_cloud_number    = ifelse(cloud_number_sink > max_cloud_number,
                               max_cloud_number / max(cloud_number_sink, ε), one(Nᶜˡ))

    α_cloud = min(α_cloud_mass, α_cloud_number)

    # Rain: coupled mass + number
    rain_mass_sink   = max(0, -Sᵉᵛᵃᵖ)
    max_rain_mass    = max(0, qʳ) / τⁿᵘᵐ
    α_rain_mass      = ifelse(rain_mass_sink > max_rain_mass,
                              max_rain_mass / max(rain_mass_sink, ε), one(qʳ))

    rain_number_sink = -dNʳ_sc - dNʳ_evap - dNʳ_adj_dn
    max_rain_number  = max(0, Nʳ) / τⁿᵘᵐ
    α_rain_number    = ifelse(rain_number_sink > max_rain_number,
                              max_rain_number / max(rain_number_sink, ε), one(Nʳ))

    α_rain = min(α_rain_mass, α_rain_number)

    # Aerosol (number only — no mass counterpart)
    aerosol_source = max(0, dNᶜˡ_act)
    max_aerosol_rate = max(0, Nᵃ) / τⁿᵘᵐ
    α_aerosol = ifelse(aerosol_source > max_aerosol_rate,
                       max_aerosol_rate / max(aerosol_source, ε), one(Nᵃ))

    # ===== Apply coupled limiting =====

    # Limited process rates
    Sᶜᵒⁿᵈ_lim    = α_vapor * max(0, Sᶜᵒⁿᵈ_eff) - α_cloud * max(0, -Sᶜᵒⁿᵈ_eff)
    Sᵃᶜᵗ_lim     = α_vapor * Sᵃᶜᵗ
    Sᵉᵛᵃᵖ_lim    = α_rain * Sᵉᵛᵃᵖ
    au_dq_lcl_lim = α_cloud * au.dq_lcl_dt
    ac_dq_lcl_lim = α_cloud * ac.dq_lcl_dt

    # Mass tendencies — conserved: ρqᵛ_phys + ρqᶜˡ_phys + ρqʳ_phys = 0
    ρqᵛ_phys  = ρ * (-(Sᶜᵒⁿᵈ_lim + Sᵃᶜᵗ_lim) - Sᵉᵛᵃᵖ_lim)
    ρqᶜˡ_phys = ρ * (  Sᶜᵒⁿᵈ_lim + Sᵃᶜᵗ_lim  + au_dq_lcl_lim + ac_dq_lcl_lim)
    ρqʳ_phys  = ρ * (                           -au_dq_lcl_lim - ac_dq_lcl_lim + Sᵉᵛᵃᵖ_lim)

    # Cloud number: sinks use SAME α_cloud as mass, sources limited by aerosol budget
    Σ_dNᶜˡ = (α_cloud * (au.dN_lcl_dt + dNᶜˡ_sc + ac.dN_lcl_dt + dNᶜˡ_adj_dn)
              + dNᶜˡ_adj_up
              + α_aerosol * dNᶜˡ_act)

    # Rain number: cloud→rain sources scaled by α_cloud, rain sinks by α_rain
    Σ_dNʳ = (α_cloud * au.dN_rai_dt + dNʳ_br + dNʳ_adj_up
             + α_rain * (dNʳ_sc + dNʳ_evap + dNʳ_adj_dn))

    # Aerosol number: activation sink
    dNᵃ_lim = -α_aerosol * dNᶜˡ_act

    # ===== Numerical relaxation guards =====

    # Mass: conserved routing v→cl, cl→r, r→v
    δᵛ  = ifelse(qᵛ  >= 0, zero(ρqᵛ_phys),  -ρ * qᵛ  / τⁿᵘᵐ - ρqᵛ_phys)
    δᶜˡ = ifelse(qᶜˡ >= 0, zero(ρqᶜˡ_phys), -ρ * qᶜˡ / τⁿᵘᵐ - ρqᶜˡ_phys)
    δʳ  = ifelse(qʳ  >= 0, zero(ρqʳ_phys),  -ρ * qʳ  / τⁿᵘᵐ - ρqʳ_phys)

    ρqᵛ  = ρqᵛ_phys  + δᵛ  - δʳ
    ρqᶜˡ = ρqᶜˡ_phys + δᶜˡ - δᵛ
    ρqʳ  = ρqʳ_phys  + δʳ  - δᶜˡ

    # Number: relaxation for negative values
    Sⁿᵘᵐ_cl   = -Nᶜˡ / τⁿᵘᵐ
    Sⁿᵘᵐ_rain = -Nʳ  / τⁿᵘᵐ
    Sⁿᵘᵐ_aer  = -Nᵃ  / τⁿᵘᵐ

    ρnᶜˡ = ifelse(nᶜˡ >= 0, Σ_dNᶜˡ, Sⁿᵘᵐ_cl)
    ρnʳ  = ifelse(nʳ  >= 0, Σ_dNʳ,  Sⁿᵘᵐ_rain)
    ρnᵃ  = ifelse(nᵃ  >= 0, dNᵃ_lim, Sⁿᵘᵐ_aer)

    return (; ρqᵛ, ρqᶜˡ, ρqʳ, ρnᶜˡ, ρnʳ, ρnᵃ)
end

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρqᵛ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    return wpne2m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᵛ
end

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρqᶜˡ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    return wpne2m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᶜˡ
end

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρnᶜˡ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    return wpne2m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρnᶜˡ
end

#####
##### Aerosol activation tendency
#####

# Nucleation radius [m] - fallback when supersaturation is negligible
# Matches CloudMicrophysics parcel model default: rⁿᵘᶜ = 0.5 * 1e-4 * 1e-6
const rⁿᵘᶜ = 5e-11  # 0.05 nm

# No activation when aerosol_activation is nothing
@inline aerosol_activation_tendency(::Nothing, aps, ρ, ℳ, 𝒰, constants) = zero(ρ)
@inline aerosol_activation_mass_tendency(::Nothing, aps, ρ, ℳ, 𝒰, constants) = zero(ρ)

# Compute activation tendency using Abdul-Razzak and Ghan (2000)
# The ARG2000 parameterization gives the fraction of the TOTAL aerosol population that should be activated.
# We compare this target to the current cloud droplet number and activate the deficit.
# The activation deficit is converted to a rate using the nucleation timescale τⁿᵘᶜ.
@inline function aerosol_activation_tendency(
    aerosol_activation::AerosolActivation,
    aps::AirProperties{FT},
    ρ::FT,
    ℳ::WarmPhaseTwoMomentState{FT},
    𝒰,
    constants,
) where {FT}

    # Extract and clamp values from microphysical state
    w = ℳ.velocities.w  # extract vertical velocity for aerosol activation
    w⁺ = max(0, w)
    Nᵃ⁺ = max(0, ℳ.nᵃ * ρ)
    Nᶜˡ⁺ = max(0, ℳ.nᶜˡ * ρ)

    # Construct clamped microphysical state for activation calculation
    velocities⁺ = (; u = ℳ.velocities.u, v = ℳ.velocities.v, w = w⁺)
    ℳ⁺ = WarmPhaseTwoMomentState(ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.qʳ, ℳ.nʳ, ℳ.nᵃ, velocities⁺)

    # Supersaturation - activation only occurs when air is supersaturated (S > 0)
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    S = supersaturation(T, ρ, q, constants, PlanarLiquidSurface())

    # Target: fraction of available aerosol that should activate
    N★ = aerosol_activated_fraction(aerosol_activation, aps, ρ, ℳ⁺, 𝒰, constants) * Nᵃ⁺

    # Disequilibrium: activate deficit, limited by available aerosol
    ΔNᵃᶜᵗ = clamp(N★ - Nᶜˡ⁺, zero(FT), Nᵃ⁺)

    # Nucleation timescale from aerosol activation parameters
    τⁿᵘᶜ = aerosol_activation.nucleation_timescale

    # Convert to rate [1/m³/s], zero if subsaturated
    dNᶜˡ_act = ifelse(S > 0, ΔNᵃᶜᵗ / τⁿᵘᶜ, zero(ρ))

    return dNᶜˡ_act
end

"""
    aerosol_activation_mass_tendency(aerosol_activation, aps, ρ, ℳ, 𝒰, constants)

Compute the cloud liquid mass tendency from aerosol activation.

When aerosol particles activate to form cloud droplets, the newly formed droplets
have a finite initial size given by the activation radius. This function computes
the corresponding mass source term for cloud liquid water.

The activation radius is derived from Köhler theory:
```math
r_{act} = \\frac{2A}{3 S}
```
where ``A = 2σ/(ρ_w R_v T)`` is the curvature parameter and ``S`` is the
instantaneous supersaturation. See eq. 19 in [Abdul-Razzak et al. (1998)](@cite AbdulRazzakGhan1998).

The mass tendency is then:
```math
\\frac{dq^{cl}}{dt}_{act} = \\frac{dN^{cl}}{dt}_{act} \\cdot \\frac{4π}{3} r_{act}^3 \\frac{ρ_w}{ρ}
```

The activation rate is controlled by the nucleation timescale `τⁿᵘᶜ` stored in
the [`AerosolActivation`](@ref) parameters (default: 1s).

# Returns
Mass tendency for cloud liquid [kg/kg/s]
"""
@inline function aerosol_activation_mass_tendency(
    aerosol_activation::AerosolActivation,
    aps::AirProperties{FT},
    ρ::FT,
    ℳ::WarmPhaseTwoMomentState{FT},
    𝒰,
    constants,
) where {FT}

    ap = aerosol_activation.activation_parameters

    # Compute number tendency using the disequilibrium approach
    dNᶜˡ_act = aerosol_activation_tendency(aerosol_activation, aps, ρ, ℳ, 𝒰, constants)

    # Get thermodynamic properties for activation radius calculation
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions

    # Compute activation radius from Köhler theory
    # A = 2σ / (ρw * Rv * T) is the curvature parameter
    # r_act = 2A / (3S) for the critical radius at supersaturation S
    Rᵛ = vapor_gas_constant(constants)
    ρᴸ = ap.ρ_w  # intrinsic density of liquid water [kg/m³]
    σ = ap.σ     # surface tension [N/m]

    A = 2 * σ / (ρᴸ * Rᵛ * T)

    # Use instantaneous supersaturation to compute activation radius
    # Following CloudMicrophysics parcel model: use r_nuc as fallback when no activation or no supersaturation
    S = supersaturation(T, ρ, q, constants, PlanarLiquidSurface())

    # Compute radius: rᵃᶜᵗ = 2A / (3S), capped at 1 μm (1e-6 m)
    # Use rⁿᵘᶜ as fallback when S is negligible (no supersaturation) or no activation
    is_activating = (dNᶜˡ_act > eps(FT)) & (S > eps(FT))
    rᵃᶜᵗ = ifelse(is_activating, min(FT(1e-6), 2 * A / (3 * max(S, eps(FT)))), rⁿᵘᶜ)

    # Mass of a single activated droplet [kg]
    # m = (4π/3) * r³ * ρᴸ
    mᵈʳᵒᵖ = FT(4π / 3) * rᵃᶜᵗ^3 * ρᴸ

    # Mass tendency [kg/kg/s] - zero if no activation
    # dq/dt = (dN/dt * mᵈʳᵒᵖ) / ρ
    dqᶜˡ_act = ifelse(dNᶜˡ_act > 0, dNᶜˡ_act * mᵈʳᵒᵖ / ρ, zero(ρ))

    return dqᶜˡ_act
end

"""
    aerosol_activated_fraction(aerosol_activation, aps, ρ, ℳ, 𝒰, constants)

Compute the fraction of aerosol that activates given current thermodynamic conditions.
Uses the maximum supersaturation to determine which aerosol modes activate.
"""
@inline function aerosol_activated_fraction(
    aerosol_activation::AerosolActivation,
    aps::AirProperties{FT},
    ρ::FT,
    ℳ::WarmPhaseTwoMomentState{FT},
    𝒰,
    constants,
) where {FT}

    ap = aerosol_activation.activation_parameters
    ad = aerosol_activation.aerosol_distribution

    # Compute maximum supersaturation
    Sᵐᵃˣ = max_supersaturation_breeze(aerosol_activation, aps, ρ, ℳ, 𝒰, constants)

    # Curvature coefficient
    T = temperature(𝒰, constants)
    Rᵛ = vapor_gas_constant(constants)
    A = 2 * ap.σ / (ap.ρ_w * Rᵛ * T)

    # Sum activated fraction from each mode
    Nᵗᵒᵗ = zero(FT)
    Nᵃᶜᵗ = zero(FT)
    for mode in ad.modes
        Nᵐᵒᵈᵉ = mode.N
        Nᵗᵒᵗ += Nᵐᵒᵈᵉ

        # Mean hygroscopicity for this mode
        κ̄ = max(eps(FT), mean_hygroscopicity(ap, mode))

        # Critical supersaturation for mode (Eq. 9 in ARG 2000)
        Sᶜʳⁱᵗ = max(eps(FT), 2 / sqrt(κ̄) * sqrt(max(0, A / (3 * mode.r_dry)))^3)

        # Activated fraction for this mode (Eq. 7 in ARG 2000)
        # Guard against log(0) or log(negative): when Sᵐᵃˣ ≈ 0, no activation occurs
        Sᵐᵃˣ_safe = max(eps(FT), Sᵐᵃˣ)
        Sᶜʳⁱᵗ_safe = max(eps(FT), Sᶜʳⁱᵗ)
        ϕ = 2 * log(Sᶜʳⁱᵗ_safe / Sᵐᵃˣ_safe) / 3 / sqrt(2) / log(mode.stdev)
        fᵃᶜᵗ = (1 - erf(ϕ)) / 2

        Nᵃᶜᵗ += fᵃᶜᵗ * Nᵐᵒᵈᵉ
    end

    # Return total activated fraction
    return ifelse(Nᵗᵒᵗ > 0, Nᵃᶜᵗ / Nᵗᵒᵗ, zero(T))
end

#####
##### Per-variable tendency dispatchers (all delegate to wpne2m_tendencies)
#####

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρqʳ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    return wpne2m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqʳ
end

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρnʳ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    return wpne2m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρnʳ
end

@inline function AtmosphereModels.microphysical_tendency(bμp::WPNE2M, ::Val{:ρnᵃ}, ρ, ℳ::WarmPhaseTwoMomentState, 𝒰, constants)
    return wpne2m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρnᵃ
end
