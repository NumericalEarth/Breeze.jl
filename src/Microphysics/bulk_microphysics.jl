using Adapt: Adapt, adapt
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF

using Breeze.Thermodynamics:
    AbstractReferencePressureState,
    LiquidIceDensityState,
    liquid_latent_heat,
    ice_latent_heat,
    vapor_gas_constant,
    mixture_gas_constant,
    mixture_heat_capacity,
    condensate_latent_energy,
    exner_exponent,
    specific_heat_difference

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
    - `VerticalBorrowing()`: Vertical redistribution of the moisture prognostic only
  - `SpeciesBorrowing()`: Same-level species borrowing only
  - `SpeciesBorrowing(vertical_borrowing=VerticalBorrowing())`: Species borrowing with vertical redistribution
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

AtmosphereModels.negative_moisture_correction(bμp::BulkMicrophysics) = bμp.negative_moisture_correction

Base.summary(::BulkMicrophysics) = "BulkMicrophysics"

"""
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
struct NonEquilibriumCloudFormation{L, I}
    liquid :: L
    ice :: I

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
$(TYPEDEF)

Return a condensate formation model that applies a **constant** phase-change rate.

This type is intended to be usable for both liquid (condensation/evaporation) and ice
(deposition/sublimation).
"""
struct ConstantRateCondensateFormation{FT} <: AbstractCondensateFormation
    rate :: FT
end

Base.summary(::ConstantRateCondensateFormation) = "ConstantRateCondensateFormation"

Adapt.adapt_structure(to, f::ConstantRateCondensateFormation) =
    ConstantRateCondensateFormation(adapt(to, f.rate))

"""
$(TYPEDEF)

Return a condensate formation model that applies a constant phase-change `rate = 1/τ` whose
explicit tendency, over a model time step `Δt`, produces the exact (exponential) relaxation
increment. The relaxation then uses the effective timescale

    τₑ = Δt / (1 - exp(-Δt / τ)),

so one step relaxes toward saturation by the factor `1 - exp(-Δt / τ)` rather than `Δt / τ`.
`τₑ → τ` as `Δt / τ → 0` and `τₑ → Δt` as `Δt / τ → ∞`, so the stage rate `Δt / (Γ τₑ) ≤ 1`
(with `Γ ≥ 1` the latent-heat factor) stays inside the stability region of the SSP-RK3 stepper
for any `Δt / τ`, whereas [`ConstantRateCondensateFormation`](@ref) is stable only for
`Δt / (Γ τ) ≲ 2.5`. `Δt` must match the model's time step.

```jldoctest
julia> using Breeze

julia> using Breeze.Microphysics: relaxation_timescale

julia> formation = ExponentialRateCondensateFormation(1 / 120, 900)
ExponentialRateCondensateFormation{Float64}(0.008333333333333333, 900.0)

julia> round(relaxation_timescale(formation), digits=1)
900.5
```
"""
struct ExponentialRateCondensateFormation{FT} <: AbstractCondensateFormation
    rate :: FT
    Δt :: FT
end

ExponentialRateCondensateFormation(rate, Δt) = ExponentialRateCondensateFormation(promote(float(rate), float(Δt))...)

Base.summary(::ExponentialRateCondensateFormation) = "ExponentialRateCondensateFormation"

Adapt.adapt_structure(to, f::ExponentialRateCondensateFormation) =
    ExponentialRateCondensateFormation(adapt(to, f.rate), adapt(to, f.Δt))

"""
$(TYPEDSIGNATURES)

Return the relaxation timescale of a condensate formation model: `1 / rate` for a constant-rate
formation and `Δt / (1 - exp(-Δt rate))` for an [`ExponentialRateCondensateFormation`](@ref).
"""
@inline relaxation_timescale(formation) = inv(formation.rate)
@inline relaxation_timescale(formation::ExponentialRateCondensateFormation) =
    formation.Δt / -expm1(-formation.Δt * formation.rate)

#####
##### Shared helpers for relaxation-to-saturation phase change (liquid + ice)
#####

# Response dqᵛ⁺/dqᶜ of the saturation specific humidity to forming condensate qᶜ from vapor at
# fixed total water and at the constraint the thermodynamic state is closed on, with
# qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) and d ln pᵛ⁺ / dT = ℒ / (Rᵛ T²). Forming condensate changes Rᵐ by
# −Rᵛ dqᶜ and cᵖᵐ by −Δcᵖ dqᶜ, where Δcᵖ = cᵖᵛ − cᶜ is the phase's `specific_heat_difference`.
#
# Fixed pressure: the enthalpy is conserved, so condensate heats by dT = ℒ dqᶜ / cᵖᵐ, and with
# ρ = p / (Rᵐ T) the saturation specific humidity is qᵛ⁺ = pᵛ⁺ Rᵐ / (p Rᵛ): the heating raises
# it at qᵛ⁺ ℒ / (Rᵛ T²) per kelvin, and the vapor lost lowers it at qᵛ⁺ Rᵛ / Rᵐ per unit qᶜ.
@inline function saturation_response(𝒰::AbstractReferencePressureState, qᵛ⁺, T, ℒ, Δcᵖ, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    return qᵛ⁺ * (ℒ^2 / (cᵖᵐ * Rᵛ * T^2) - Rᵛ / Rᵐ)
end

# Fixed θ and ρ: H = ĉᵖ Π θ with Π = (ρ Rᵐ T / pˢᵗ)^κ. Phase exchange holds total water,
# ĉᵖ, and κ fixed, with dH = cᵖᵐ dT - ℒ dqᶜ and dRᵐ = -Rᵛ dqᶜ. Differentiating gives
# (cᵖᵐ - κ H / T) dT = (ℒ - κ H Rᵛ / Rᵐ) dqᶜ. This agrees with the constant-internal-energy
# response only when κ H = Rᵐ T, as in condensate-free air with VaporEquivalentExponent.
# At fixed ρ, qᵛ⁺ does not depend on the composition. This is the response of the converged
# inversion, which `NewtonSolver` and `FixedIterations` approach.
@inline function saturation_response(𝒰::LiquidIceDensityState, qᵛ⁺, T, ℒ, Δcᵖ, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    κ = exner_exponent(𝒰.exner_exponent, q, constants)
    H = cᵖᵐ * T - condensate_latent_energy(q, constants)
    dT_dqᶜ = (ℒ - κ * H * Rᵛ / Rᵐ) / (cᵖᵐ - κ * H / T)
    return dT_dqᶜ * qᵛ⁺ * (ℒ / (Rᵛ * T^2) - 1 / T)
end

# Fixed θ and ρ with the non-iterated inversion (`temperature_solver = nothing`), whose
# temperature is the condensate-free root plus the latent shift at fixed density,
# T = T̃ + γ Hᶜ / cᵖᵐ with T̃ = θ̃^γ (ρ Rᵐ / pˢᵗ)^(γ - 1), θ̃ = ĉᵖ θ / cᵖᵐ, and γ = 1 / (1 - κ).
# Phase exchange holds θ̃ cᵖᵐ, ρ, and κ fixed while cᵖᵐ falls by Δcᵖ dqᶜ and Rᵐ by Rᵛ dqᶜ, so
# cᵖᵐ dT̃ = γ (Δcᵖ - κ cᵖᵐ Rᵛ / Rᵐ) T̃ dqᶜ and cᵖᵐ d(γ Hᶜ / cᵖᵐ) = γ (ℒ₀ + Δcᵖ Hᶜ / cᵖᵐ) dqᶜ. With
# ℒ = ℒ₀ + Δcᵖ T, (1 - κ) cᵖᵐ dT = [ℒ - κ (cᵖᵐ T̃ Rᵛ / Rᵐ + γ Δcᵖ Hᶜ / cᵖᵐ)] dqᶜ: the Exner
# feedback of the converged inversion, cᵖᵐ - κ H / T, is replaced by its condensate-free value.
@inline function saturation_response(𝒰::LiquidIceDensityState{<:Any, Nothing}, qᵛ⁺, T, ℒ, Δcᵖ, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵐ = mixture_gas_constant(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    κ = exner_exponent(𝒰.exner_exponent, q, constants)
    γ = 1 / (1 - κ)
    Hᶜ = condensate_latent_energy(q, constants)
    T̃ = T - γ * Hᶜ / cᵖᵐ
    dT_dqᶜ = γ * (ℒ - κ * (cᵖᵐ * T̃ * Rᵛ / Rᵐ + γ * Δcᵖ * Hᶜ / cᵖᵐ)) / cᵖᵐ
    return dT_dqᶜ * qᵛ⁺ * (ℒ / (Rᵛ * T^2) - 1 / T)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute the thermodynamic adjustment factor `Γ = 1 + dqᵛ⁺/dqᶜ` used in
relaxation-to-saturation condensation/evaporation tendencies, so that condensation alone relaxes
the supersaturation excess `qᵛ - qᵛ⁺` over the relaxation timescale.

The response `dqᵛ⁺/dqᶜ` to forming condensate from vapor follows the conserved variables of `𝒰` at
fixed total water: enthalpy at fixed pressure for reference-pressure states, where `qᵛ⁺` also
falls with the mixture gas constant, and `θ` at fixed density for `LiquidIceDensityState`, using
its chosen Exner exponent and the temperature its `temperature_solver` returns.
"""
@inline thermodynamic_adjustment_factor(qᵛ⁺, T, 𝒰, constants) =
    thermodynamic_adjustment_factor(qᵛ⁺, T, 𝒰, liquid_latent_heat(T, constants),
                                    specific_heat_difference(constants, constants.liquid), constants)

@inline thermodynamic_adjustment_factor(qᵛ⁺, T, 𝒰, ℒ, Δcᵖ, constants) =
    1 + saturation_response(𝒰, qᵛ⁺, T, ℒ, Δcᵖ, constants)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute the thermodynamic adjustment factor `Γ` used in relaxation-to-saturation
deposition/sublimation tendencies (ice analogue of `thermodynamic_adjustment_factor`).
"""
@inline ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, 𝒰, constants) =
    thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, 𝒰, ice_latent_heat(T, constants),
                                    specific_heat_difference(constants, constants.ice), constants)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute the condensation/evaporation rate for cloud liquid water in a relaxation-to-saturation model.

This returns the rate of change of cloud liquid mass fraction (kg/kg/s). Positive values indicate
condensation; negative values indicate evaporation. Evaporation is limited by the available cloud liquid.
"""
@inline function condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, 𝒰, τᶜˡ, constants)
    Γˡ = thermodynamic_adjustment_factor(qᵛ⁺, T, 𝒰, constants)
    timescale = Γˡ * τᶜˡ
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺) / timescale

    # Limit evaporation to available cloud liquid
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / timescale
    return max(Sᶜᵒⁿᵈ, Sᶜᵒⁿᵈ_min)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute the deposition/sublimation rate for cloud ice in a relaxation-to-saturation model.

This returns the rate of change of cloud ice mass fraction (kg/kg/s). Positive values indicate
deposition; negative values indicate sublimation. Sublimation is limited by the available cloud ice.
"""
@inline function deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, 𝒰, τᶜⁱ, constants)
    Γⁱ = ice_thermodynamic_adjustment_factor(qᵛ⁺ⁱ, T, 𝒰, constants)
    timescale = Γⁱ * τᶜⁱ
    Sᵈᵉᵖ = (qᵛ - qᵛ⁺ⁱ) / timescale

    # Limit sublimation to available cloud ice
    Sᵈᵉᵖ_min = -max(0, qᶜⁱ) / timescale
    return max(Sᵈᵉᵖ, Sᵈᵉᵖ_min)
end

struct FourCategories{L, I, R, S, C, V, A}
    cloud_liquid :: L
    cloud_ice :: I
    rain :: R
    snow :: S
    collisions :: C
    hydrometeor_velocities :: V
    air :: A
end

FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities) =
    FourCategories(cloud_liquid, cloud_ice, rain, snow, collisions, hydrometeor_velocities, nothing)

const FourCategoryBulkMicrophysics = BulkMicrophysics{<:Any, <:FourCategories}
Base.summary(::FourCategoryBulkMicrophysics) = "FourCategoryBulkMicrophysics"

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a `BulkMicrophysics` microphysics scheme.

# Keyword arguments
- `categories`: Microphysical categories (e.g., cloud liquid, cloud ice, rain, snow) or `nothing` for non-precipitating
- `cloud_formation`: Cloud formation scheme (default: `SaturationAdjustment`)
- `precipitation_boundary_condition`: Bottom boundary condition for precipitation sedimentation.
  - `nothing` (default): Precipitation passes through the bottom
  - `ImpenetrableBoundaryCondition()`: Precipitation collects at the bottom
- `negative_moisture_correction`: Correction scheme for negative moisture produced by advection.
  - `nothing` (default): No correction
    - `VerticalBorrowing()`: Vertical redistribution of the moisture prognostic only
  - `SpeciesBorrowing()`: Same-level species borrowing only
  - `SpeciesBorrowing(vertical_borrowing=VerticalBorrowing())`: Species borrowing with vertical redistribution
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

@inline AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp::NCBM, qᵛ, constants) =
    AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, bμp.cloud_formation, qᵛ, constants)

@inline AtmosphereModels.microphysical_velocities(bμp::NCBM, μ, name) =
    AtmosphereModels.microphysical_velocities(bμp.cloud_formation, μ, name)

@inline AtmosphereModels.microphysical_tendency(bμp::NCBM, name, ρ, ℳ, 𝒰, constants) =
    AtmosphereModels.microphysical_tendency(bμp.cloud_formation, name, ρ, ℳ, 𝒰, constants)

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
