using ..Thermodynamics:
    Thermodynamics,
    MoistureMassFractions,
    mixture_heat_capacity,
    saturation_specific_humidity,
    adjustment_saturation_specific_humidity,
    temperature,
    is_absolute_zero,
    with_moisture,
    total_specific_moisture,
    AbstractThermodynamicState,
    WarmPhaseEquilibrium,
    MixedPhaseEquilibrium,
    equilibrated_surface

using Oceananigans: Oceananigans, CenterField
using DocStringExtensions: TYPEDSIGNATURES

struct SaturationAdjustment{E, FT}
    tolerance :: FT
    maxiter :: FT
    equilibrium :: E
end

const SA = SaturationAdjustment

"""
$(TYPEDSIGNATURES)

Return `SaturationAdjustment` microphysics representing an instantaneous adjustment
to `equilibrium` between condensates and water vapor, computed by a solver with
`tolerance` and `maxiter`.

The options for `equilibrium` are:

* [`WarmPhaseEquilibrium()`](@ref WarmPhaseEquilibrium) representing an equilibrium between
  water vapor and liquid water.

* [`MixedPhaseEquilibrium()`](@ref MixedPhaseEquilibrium) representing a temperature-dependent
  equilibrium between water vapor, possibly supercooled liquid water, and ice. The equilibrium
  state is modeled as a linear variation of the equilibrium liquid fraction with temperature,
  between the freezing temperature (e.g. 273.15 K) below which liquid water is supercooled,
  and the temperature of homogeneous ice nucleation temperature (e.g. 233.15 K) at which
  the supercooled liquid fraction vanishes.
"""
function SaturationAdjustment(FT::DataType=Oceananigans.defaults.FloatType;
                              tolerance = 1e-3,
                              maxiter = Inf,
                              equilibrium = MixedPhaseEquilibrium(FT))
    tolerance = convert(FT, tolerance)
    maxiter = convert(FT, maxiter)
    return SaturationAdjustment(tolerance, maxiter, equilibrium)
end

@inline AtmosphereModels.microphysical_velocities(::SaturationAdjustment, μ, name) = nothing

# SaturationAdjustment operates through the thermodynamic state adjustment pathway,
# so no explicit model update is needed.
AtmosphereModels.microphysics_model_update!(::SaturationAdjustment, model) = nothing

#####
##### Warm-phase equilibrium moisture fractions
#####

@inline function equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, ::WarmPhaseEquilibrium)
    qˡ = max(0, qᵗ - qᵛ⁺)
    qᵛ = qᵗ - qˡ
    return MoistureMassFractions(qᵛ, qˡ)
end

#####
##### Mixed-phase equilibrium moisture fractions
#####

@inline function equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium::MixedPhaseEquilibrium)
    surface = equilibrated_surface(equilibrium, T)
    λ = surface.liquid_fraction
    qᶜ = max(0, qᵗ - qᵛ⁺)
    qᵛ = qᵗ - qᶜ
    qˡ = λ * qᶜ
    qⁱ = (1 - λ) * qᶜ
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

const WarmPhaseSaturationAdjustment{FT} = SaturationAdjustment{WarmPhaseEquilibrium, FT} where FT
const MixedPhaseSaturationAdjustment{FT} = SaturationAdjustment{MixedPhaseEquilibrium{FT}, FT} where FT

const WPSA = WarmPhaseSaturationAdjustment
const MPSA = MixedPhaseSaturationAdjustment

AtmosphereModels.moisture_prognostic_name(::SA) = :ρqᵉ

AtmosphereModels.prognostic_field_names(::WPSA) = tuple()
AtmosphereModels.prognostic_field_names(::MPSA) = tuple()

AtmosphereModels.liquid_mass_fraction(::SA, model) = model.microphysical_fields.qˡ
AtmosphereModels.ice_mass_fraction(::WPSA, model) = nothing
AtmosphereModels.ice_mass_fraction(::MPSA, model) = model.microphysical_fields.qⁱ

center_field_tuple(grid, names...) = NamedTuple{names}(CenterField(grid) for name in names)
AtmosphereModels.materialize_microphysical_fields(::WPSA, grid, bcs) = center_field_tuple(grid, :qᵛ, :qˡ, :qᵉ)
AtmosphereModels.materialize_microphysical_fields(::MPSA, grid, bcs) = center_field_tuple(grid, :qᵛ, :qˡ, :qⁱ, :qᵉ)

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, ::WPSA, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    @inbounds μ.qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
    # qᵉ is written in _compute_auxiliary_thermodynamic_variables!
    return nothing
end

@inline function AtmosphereModels.update_microphysical_fields!(μ, i, j, k, grid, ::MPSA, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    @inbounds μ.qˡ[i, j, k] = 𝒰.moisture_mass_fractions.liquid
    @inbounds μ.qⁱ[i, j, k] = 𝒰.moisture_mass_fractions.ice
    # qᵉ is written in _compute_auxiliary_thermodynamic_variables!
    return nothing
end

# Grid-indexed moisture fractions for saturation adjustment schemes.
# These read from diagnostic fields that are filled during update_microphysical_fields!.
@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::WPSA, ρ, qᵉ, μ)
    qᵛ = @inbounds μ.qᵛ[i, j, k]
    qˡ = @inbounds μ.qˡ[i, j, k]
    return MoistureMassFractions(qᵛ, qˡ)
end

@inline function AtmosphereModels.grid_moisture_fractions(i, j, k, grid, ::MPSA, ρ, qᵉ, μ)
    qᵛ = @inbounds μ.qᵛ[i, j, k]
    qˡ = @inbounds μ.qˡ[i, j, k]
    qⁱ = @inbounds μ.qⁱ[i, j, k]
    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

# State-based moisture fractions for saturation adjustment (used by parcel models).
# The moisture fractions come from the thermodynamic state after adjustment.
# Since NothingMicrophysicalState has no prognostic variables, we return all vapor.
# The parcel model's saturation adjustment updates the thermodynamic state directly.
@inline AtmosphereModels.moisture_fractions(::SA, ::NothingMicrophysicalState, qᵉ) = MoistureMassFractions(qᵉ)

# State-based tendency (used by parcel models)
# SaturationAdjustment operates through thermodynamic state adjustment, so explicit tendencies are zero
@inline AtmosphereModels.microphysical_tendency(::SA, name, ρ, ℳ, 𝒰, constants, clock) = zero(ρ)

#####
##### Saturation adjustment utilities
#####

@inline function adjust_state(𝒰₀, T, constants, equilibrium)
    pᵣ = 𝒰₀.reference_pressure
    qᵗ = total_specific_moisture(𝒰₀)
    qᵛ⁺ = adjustment_saturation_specific_humidity(T, pᵣ, qᵗ, constants, equilibrium)
    q₁ = equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium)
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰₀, constants, equilibrium)
    𝒰₁ = adjust_state(𝒰₀, T, constants, equilibrium)
    T₁ = temperature(𝒰₁, constants)
    return T - T₁
end

const ATS = AbstractThermodynamicState

# This function allows saturation adjustment to be used as a microphysics scheme directly
@inline function AtmosphereModels.maybe_adjust_thermodynamic_state(𝒰₀, saturation_adjustment::SA, qᵉ, constants)
    qᵃ = MoistureMassFractions(qᵉ) # compute moisture state to be adjusted
    𝒰ᵃ = with_moisture(𝒰₀, qᵃ)
    return adjust_thermodynamic_state(𝒰ᵃ, saturation_adjustment, constants)
end

"""
$(TYPEDSIGNATURES)

Return the saturation-adjusted thermodynamic state using a secant iteration.
"""
@inline function adjust_thermodynamic_state(𝒰₀::ATS, microphysics::SA, constants)
    FT = eltype(𝒰₀)
    is_absolute_zero(𝒰₀) && return 𝒰₀

    # Compute an initial guess assuming unsaturated conditions
    qᵗ = total_specific_moisture(𝒰₀)
    q₁ = MoistureMassFractions(qᵗ)
    𝒰₁ = with_moisture(𝒰₀, q₁)
    T₁ = temperature(𝒰₁, constants)

    equilibrium = microphysics.equilibrium
    qᵛ⁺₁ = saturation_specific_humidity(𝒰₁, constants, equilibrium)
    qᵗ <= qᵛ⁺₁ && return 𝒰₁

    # If we made it here, the state is saturated.
    # So, we re-initialize our first guess assuming saturation
    𝒰₁ = adjust_state(𝒰₀, T₁, constants, equilibrium)

    # Next, we generate a second guess that scaled by the supersaturation implied by T₁
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ₁ = q₁.liquid
    qⁱ₁ = q₁.ice
    cᵖᵐ = mixture_heat_capacity(q₁, constants)
    ΔT = (ℒˡᵣ * qˡ₁ + ℒⁱᵣ * qⁱ₁) / cᵖᵐ
    ϵT = convert(FT, 0.01) # minimum increment for second guess
    T₂ = T₁ + max(ϵT, ΔT / 2) # reduce the increment, recognizing it is an overshoot
    𝒰₂ = adjust_state(𝒰₁, T₂, constants, equilibrium)

    # Initialize secant iteration
    r₁ = saturation_adjustment_residual(T₁, 𝒰₁, constants, equilibrium)
    r₂ = saturation_adjustment_residual(T₂, 𝒰₂, constants, equilibrium)
    δ = microphysics.tolerance
    iter = 0

    while abs(r₂) > δ && iter < microphysics.maxiter
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂
        𝒰₁ = 𝒰₂

        # Update
        T₂ -= r₂ * ΔTΔr
        𝒰₂ = adjust_state(𝒰₂, T₂, constants, equilibrium)
        r₂ = saturation_adjustment_residual(T₂, 𝒰₂, constants, equilibrium)
        iter += 1
    end

    return 𝒰₂
end

"""
$(TYPEDSIGNATURES)

Perform saturation adjustment and return the temperature
associated with the adjusted state.
"""
function compute_temperature(𝒰₀, adjustment::SA, constants)
    𝒰₁ = adjust_thermodynamic_state(𝒰₀, adjustment, constants)
    return temperature(𝒰₁, constants)
end

# When no microphysics adjustment is needed
compute_temperature(𝒰₀, ::Nothing, constants) = temperature(𝒰₀, constants)
