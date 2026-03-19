using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

using ..Thermodynamics:
    liquid_latent_heat,
    ice_latent_heat,
    vapor_gas_constant,
    mixture_heat_capacity

"""
$(TYPEDEF)

Correct negative moisture produced by advection via same-level borrowing
and vertical redistribution at each time step.

See [`correct_negative_moisture!`](@ref Breeze.AtmosphereModels.correct_negative_moisture!) for details.
"""
struct VerticalBorrowing end

"""
$(TYPEDEF)

Bulk microphysics scheme with cloud formation and precipitation categories.

# Fields
- `cloud_formation`: Cloud formation scheme (saturation adjustment or non-equilibrium)
- `categories`: Precipitation categories (e.g., rain, snow) or `nothing`
- `precipitation_boundary_condition`: Bottom boundary condition for precipitation sedimentation.
  - `nothing` (default): Precipitation passes through the bottom (open boundary)
  - `ImpenetrableBoundaryCondition()`: Precipitation collects at the bottom (zero terminal velocity at surface)
- `negative_moisture_correction`: Correction scheme for negative moisture produced by advection.
  - `nothing` (default): No correction
  - `VerticalBorrowing()`: Same-level borrowing and vertical redistribution
"""
struct BulkMicrophysics{N, C, B, NMC}
    cloud_formation :: N
    categories :: C
    precipitation_boundary_condition :: B
    negative_moisture_correction :: NMC
end

# Bulk microphysics schemes (including those from extensions like CloudMicrophysics)
# use the standard tendency interface, so the model-wide microphysics update is a no-op.
# We forward to the cloud_formation / saturation-adjustment component to allow specialized
# cloud formation schemes to hook into the update cycle.
AtmosphereModels.microphysics_model_update!(bμp::BulkMicrophysics, model) =
    AtmosphereModels.microphysics_model_update!(bμp.cloud_formation, model)

AtmosphereModels.negative_moisture_correction(bμp::BulkMicrophysics) = bμp.negative_moisture_correction !== nothing

Base.summary(::BulkMicrophysics) = "BulkMicrophysics"

struct NonEquilibriumCloudFormation{L, I}
    liquid :: L
    ice :: I

    @doc"""
        NonEquilibriumCloudFormation(liquid, ice=nothing)

    A cloud formation scheme where cloud liquid and ice are prognostic variables
    that evolve via condensation/evaporation and deposition/sublimation tendencies,
    rather than being diagnosed instantaneously via saturation adjustment.

    The condensation/evaporation and deposition/sublimation tendencies are commonly modeled as **relaxation toward
    saturation** with timescale `τ_relax`, including a latent-heat (psychrometric/thermal) correction factor; see
    [Morrison and Grabowski (2008)](@cite Morrison2008novel), Appendix Eq. (A3), and standard cloud microphysics
    texts such as [Pruppacher and Klett (2010)](@cite pruppacher2010microphysics) or
    [Rogers and Yau (1989)](@cite rogers1989short).

    For some bulk schemes (e.g. the CloudMicrophysics 1M extension), `liquid` and `ice` may be set to `nothing`
    and used purely as **phase indicators** (warm-phase vs mixed-phase), with any relaxation timescales sourced
    from the scheme's precipitation/category parameters instead.

    # Fields
    - `liquid`: Parameters for cloud liquid (contains relaxation timescale `τ_relax`)
    - `ice`: Parameters for cloud ice (contains relaxation timescale `τ_relax`), or `nothing` for warm-phase only

    # References

    * Morrison, H. and Grabowski, W. W. (2008). A novel approach for representing ice
        microphysics in models: Description and tests using a kinematic framework.
        J. Atmos. Sci., 65, 1528–1548. https://doi.org/10.1175/2007JAS2491.1
    * Pruppacher, H. R. and Klett, J. D. (2010). Microphysics of Clouds and Precipitation (2nd ed.).
    * Rogers, R. R. and Yau, M. K. (1989). A Short Course in Cloud Physics (3rd ed.).
    """
    function NonEquilibriumCloudFormation(liquid, ice=nothing)
        return new{typeof(liquid), typeof(ice)}(liquid, ice)
    end
end

Base.summary(::NonEquilibriumCloudFormation) = "NonEquilibriumCloudFormation"

# NonEquilibriumCloudFormation uses the standard tendency interface,
# so the model-wide microphysics update is a no-op.
AtmosphereModels.microphysics_model_update!(::NonEquilibriumCloudFormation, model) = nothing
#####
##### Condensate formation models (for non-equilibrium schemes)
#####

abstract type AbstractCondensateFormation end

"""
$(TYPEDSIGNATURES)

Return a condensate formation model that applies a **constant** phase-change rate.

This type is intended to be usable for both liquid (condensation/evaporation) and ice
(deposition/sublimation).
"""
struct ConstantRateCondensateFormation{FT} <: AbstractCondensateFormation
    rate :: FT
end

Base.summary(::ConstantRateCondensateFormation) = "ConstantRateCondensateFormation"

#####
##### Shared helpers for relaxation-to-saturation phase change (liquid + ice)
#####

"""
$(TYPEDSIGNATURES)

Compute the thermodynamic adjustment factor `Γ` used in relaxation-to-saturation
condensation/evaporation tendencies.
"""
@inline function thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ * (ℒˡ / (Rᵛ * T^2) - 1 / T)
    return 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT
end

"""
$(TYPEDSIGNATURES)

Compute the thermodynamic adjustment factor `Γ` used in relaxation-to-saturation
deposition/sublimation tendencies (ice analogue of `thermodynamic_adjustment_factor`).
"""
@inline function ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, q, constants)
    ℒⁱ = ice_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺ⁱ_dT = qᵛ⁺ⁱ * (ℒⁱ / (Rᵛ * T^2) - 1 / T)
    return 1 + (ℒⁱ / cᵖᵐ) * dqᵛ⁺ⁱ_dT
end

"""
$(TYPEDSIGNATURES)

Compute the condensation/evaporation rate for cloud liquid water in a relaxation-to-saturation model.

This returns the rate of change of cloud liquid mass fraction (kg/kg/s). Positive values indicate
condensation; negative values indicate evaporation. Evaporation is limited by the available cloud liquid.
"""
@inline function condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Γˡ = thermodynamic_adjustment_factor(qᵛ⁺, T, q, constants)
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺) / (Γˡ * τᶜˡ)

    # Limit evaporation to available cloud liquid
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / τᶜˡ
    return max(Sᶜᵒⁿᵈ, Sᶜᵒⁿᵈ_min)
end

"""
$(TYPEDSIGNATURES)

Compute the deposition/sublimation rate for cloud ice in a relaxation-to-saturation model.

This returns the rate of change of cloud ice mass fraction (kg/kg/s). Positive values indicate
deposition; negative values indicate sublimation. Sublimation is limited by the available cloud ice.
"""
@inline function deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, ρ, q, τᶜⁱ, constants)
    Γⁱ = ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, q, constants)
    Sᵈᵉᵖ = (qᵛ - qᵛ⁺ⁱ) / (Γⁱ * τᶜⁱ)

    # Limit sublimation to available cloud ice
    Sᵈᵉᵖ_min = -max(0, qᶜⁱ) / τᶜⁱ
    return max(Sᵈᵉᵖ, Sᵈᵉᵖ_min)
end

struct FourCategories{L, I, R, S, C, V, A}
    cloud_liquid :: L
    cloud_ice :: I
    rain :: R
    snow :: S
    collisions :: C
    hydrometeor_velocities :: V
    air_properties :: A
end

FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities) =
    FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities, nothing)

const FourCategoryBulkMicrophysics = BulkMicrophysics{<:Any, <:FourCategories}
Base.summary(::FourCategoryBulkMicrophysics) = "FourCategoryBulkMicrophysics"

"""
$(TYPEDSIGNATURES)

Return a `BulkMicrophysics` microphysics scheme.

# Keyword arguments
- `categories`: Microphysical categories (e.g., cloud liquid, cloud ice, rain, snow) or `nothing` for non-precipitating
- `cloud_formation`: Cloud formation scheme (default: `SaturationAdjustment`)
- `precipitation_boundary_condition`: Bottom boundary condition for precipitation sedimentation.
  - `nothing` (default): Precipitation passes through the bottom
  - `ImpenetrableBoundaryCondition()`: Precipitation collects at the bottom
- `negative_moisture_correction`: Correction scheme for negative moisture produced by advection.
  - `nothing` (default): No correction
  - `VerticalBorrowing()`: Same-level borrowing and vertical redistribution
"""
function BulkMicrophysics(FT::DataType = Oceananigans.defaults.FloatType;
                          categories = nothing,
                          cloud_formation = SaturationAdjustment(FT),
                          precipitation_boundary_condition = nothing,
                          negative_moisture_correction = nothing)

    return BulkMicrophysics(cloud_formation, categories, precipitation_boundary_condition, negative_moisture_correction)
end

# Forward moisture_prognostic_name to cloud_formation scheme
AtmosphereModels.moisture_prognostic_name(bμp::BulkMicrophysics) =
    AtmosphereModels.moisture_prognostic_name(bμp.cloud_formation)

AtmosphereModels.moisture_prognostic_name(::NonEquilibriumCloudFormation) = :ρqᵛ

# Non-categorical bulk microphysics
const NCBM = BulkMicrophysics{<:Any, Nothing}
const NPBM = NCBM  # Alias: Non-Precipitating Bulk Microphysics

maybe_adjust_thermodynamic_state(𝒰₀, bμp::NCBM, qᵛ, constants) =
    AtmosphereModels.adjust_thermodynamic_state(𝒰₀, bμp.cloud_formation, constants)

AtmosphereModels.prognostic_field_names(::NPBM) = tuple()
AtmosphereModels.materialize_microphysical_fields(bμp::NPBM, grid, bcs) = materialize_microphysical_fields(bμp.cloud_formation, grid, bcs)

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, bμp::NPBM, ρ, 𝒰, constants)
    return update_microphysical_fields!(μ, i, j, k, grid, bμp.cloud_formation, ρ, 𝒰, constants)
end

# Forward grid_moisture_fractions to cloud_formation scheme
@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, bμp::NPBM, ρ, qᵛ, μ)
    return grid_moisture_fractions(i, j, k, grid, bμp.cloud_formation, ρ, qᵛ, μ)
end

# Forward state-based moisture_fractions to cloud_formation scheme
@inline function AtmosphereModels.moisture_fractions(bμp::NPBM, ℳ, qᵛ)
    return moisture_fractions(bμp.cloud_formation, ℳ, qᵛ)
end

# Disambiguation for specific state types
@inline function AtmosphereModels.moisture_fractions(bμp::NPBM, ℳ::WarmRainState, qᵛ)
    return moisture_fractions(bμp.cloud_formation, ℳ, qᵛ)
end

@inline function AtmosphereModels.moisture_fractions(bμp::NPBM, ℳ::NothingMicrophysicalState, qᵛ)
    return moisture_fractions(bμp.cloud_formation, ℳ, qᵛ)
end

@inline function AtmosphereModels.moisture_fractions(bμp::NPBM, ℳ::NamedTuple, qᵛ)
    return moisture_fractions(bμp.cloud_formation, ℳ, qᵛ)
end

# Forward mass fraction diagnostics to cloud_formation scheme
AtmosphereModels.liquid_mass_fraction(bμp::NPBM, model) =
    AtmosphereModels.liquid_mass_fraction(bμp.cloud_formation, model)
AtmosphereModels.ice_mass_fraction(bμp::NPBM, model) =
    AtmosphereModels.ice_mass_fraction(bμp.cloud_formation, model)
