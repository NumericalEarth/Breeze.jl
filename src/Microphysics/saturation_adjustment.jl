using ..Thermodynamics:
    Thermodynamics,
    MoistureMassFractions,
    mixture_gas_constant,
    mixture_heat_capacity,
    saturation_specific_humidity,
    adjustment_saturation_specific_humidity,
    temperature,
    is_absolute_zero,
    with_moisture,
    total_specific_moisture,
    AbstractThermodynamicState,
    LiquidIceDensityState,
    WarmPhaseEquilibrium,
    MixedPhaseEquilibrium,
    equilibrated_surface

using Breeze.Solvers: SecantSolver, secant_solve, materialize_solver

using Oceananigans: Oceananigans, CenterField
using DocStringExtensions: TYPEDSIGNATURES

struct SaturationAdjustment{E, S}
    equilibrium :: E
    solver :: S
end

const SA = SaturationAdjustment

"""
$(TYPEDSIGNATURES)

Return `SaturationAdjustment` microphysics representing an instantaneous adjustment
to `equilibrium` between condensates and water vapor, computed by a secant iteration
on the temperature residual controlled by `solver`.

The options for `equilibrium` are:

* [`WarmPhaseEquilibrium()`](@ref WarmPhaseEquilibrium) representing an equilibrium between
  water vapor and liquid water.

* [`MixedPhaseEquilibrium()`](@ref MixedPhaseEquilibrium) representing a temperature-dependent
  equilibrium between water vapor, possibly supercooled liquid water, and ice. The equilibrium
  state is modeled as a linear variation of the equilibrium liquid fraction with temperature,
  between the freezing temperature (e.g. 273.15 K) below which liquid water is supercooled,
  and the temperature of homogeneous ice nucleation temperature (e.g. 233.15 K) at which
  the supercooled liquid fraction vanishes.

The options for `solver` are [`SecantSolver`](@ref) (default:
`SecantSolver(abstol=1e-4, maxiter=20)`, an absolute tolerance on the temperature-like
residual in Kelvin) and [`FixedIterations`](@ref Breeze.Solvers.FixedIterations), which performs a fixed number of secant
steps with no convergence test (the form required for Reactant tracing and cheap
reverse-mode differentiation).
"""
function SaturationAdjustment(FT::DataType=Oceananigans.defaults.FloatType;
                              solver = SecantSolver(FT; abstol=1e-4, maxiter=20),
                              equilibrium = MixedPhaseEquilibrium(FT),
                              tolerance = nothing,
                              maxiter = nothing)

    if tolerance !== nothing || maxiter !== nothing
        throw(ArgumentError("The `tolerance` and `maxiter` keyword arguments have been replaced \
                             by `solver`. Use, for example, \
                             `SaturationAdjustment(solver = SecantSolver(abstol=1e-4, maxiter=20))` \
                             or `solver = FixedIterations(n)` for Reactant / differentiable runs."))
    end

    solver = materialize_solver(solver, FT)
    return SaturationAdjustment(equilibrium, solver)
end

@inline AtmosphereModels.sedimentation_velocity(::SaturationAdjustment, μ, name) = nothing

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

const WarmPhaseSaturationAdjustment{S} = SaturationAdjustment{WarmPhaseEquilibrium, S} where S
const MixedPhaseSaturationAdjustment{FT} = SaturationAdjustment{MixedPhaseEquilibrium{FT}} where FT

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
@inline AtmosphereModels.microphysical_tendency(::SA, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

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
    qᵗ ≤ qᵛ⁺₁ && return 𝒰₁

    # If we made it here, the state is saturated.
    # So, we re-initialize our first guess assuming saturation
    𝒰₁ = adjust_state(𝒰₀, T₁, constants, equilibrium)

    # Next, we generate a second guess scaled by the supersaturation implied by T₁.
    # Use the adjusted moisture fractions (not the all-vapor q₁) so ΔT reflects
    # the actual condensate released during adjustment.
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    q̃₁ = 𝒰₁.moisture_mass_fractions
    qˡ₁ = q̃₁.liquid
    qⁱ₁ = q̃₁.ice
    cᵖᵐ = mixture_heat_capacity(q̃₁, constants)
    ΔT = (ℒˡᵣ * qˡ₁ + ℒⁱᵣ * qⁱ₁) / cᵖᵐ
    ϵT = convert(FT, 0.01) # minimum increment for second guess
    T₂ = T₁ + max(ϵT, ΔT / 2) # reduce the increment, recognizing it is an overshoot

    # Secant iteration on the temperature residual. `adjust_state` depends only on the
    # invariants of 𝒰₀ (its reference pressure and total moisture), so the residual is a
    # pure function of T and the adjusted state is recovered from the converged root.
    @inline residual(T) = saturation_adjustment_residual(T, 𝒰₀, constants, equilibrium)
    T★ = secant_solve(residual, microphysics.solver, T₁, T₂, T₂)

    return adjust_state(𝒰₀, T★, constants, equilibrium)
end

#####
##### Density-consistent saturation adjustment for the density-based θˡⁱ state
#####

# Residual for a constant-density saturated state at temperature `T`: saturate at the actual
# density (density-based qsat), form the equilibrium partition, and return the density-based θˡⁱ
# minus the target θ₀ together with the partition `q`. A root in `T` is the state that is both
# saturated at ρ and conserves θˡⁱ. See NumericalEarth/Breeze.jl#765.
@inline function saturated_density_residual(T, θ₀, ρ, qᵗ, pˢᵗ, constants, equilibrium)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, equilibrium)
    q   = equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium)
    Rᵐ  = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    κ   = Rᵐ / cᵖᵐ
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    L   = (ℒˡᵣ * q.liquid + ℒⁱᵣ * q.ice) / cᵖᵐ
    p   = ρ * Rᵐ * T
    θ   = (T - L) * (pˢᵗ / p)^κ
    return θ - θ₀, q
end

"""
$(TYPEDSIGNATURES)

Saturation adjustment for the `LiquidIceDensityState`: a secant on the
constant-density θˡⁱ-conservation residual, so `qsat` and the θˡⁱ inversion are evaluated at the
state's actual density `ρ` (with true pressure `p = ρRᵐT`) rather than a fixed reference pressure.
This is the density-consistent analogue of the generic (reference-pressure) `adjust_state` secant;
like that one it holds θˡⁱ fixed (conserves it). See NumericalEarth/Breeze.jl#765.
"""
@inline function adjust_thermodynamic_state(𝒰₀::LiquidIceDensityState, microphysics::SA, constants)
    FT = eltype(𝒰₀)
    is_absolute_zero(𝒰₀) && return 𝒰₀

    θ₀  = 𝒰₀.potential_temperature
    ρ   = 𝒰₀.density
    pˢᵗ = 𝒰₀.standard_pressure
    qᵗ  = total_specific_moisture(𝒰₀)
    equilibrium = microphysics.equilibrium

    # Unsaturated? No condensation — return the all-vapor state.
    𝒰₁ = with_moisture(𝒰₀, MoistureMassFractions(qᵗ))
    T₁ = temperature(𝒰₁, constants)
    qᵗ ≤ saturation_specific_humidity(T₁, ρ, constants, equilibrium) && return 𝒰₁

    # Saturated: secant on the constant-density residual r(T) = θˡⁱ(T) − θ₀.
    _, q₁ = saturated_density_residual(T₁, θ₀, ρ, qᵗ, pˢᵗ, constants, equilibrium)

    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    cᵖ₁ = mixture_heat_capacity(q₁, constants)
    ΔT  = (ℒˡᵣ * q₁.liquid + ℒⁱᵣ * q₁.ice) / cᵖ₁   # latent warming implied at T₁
    T₂  = T₁ + max(convert(FT, 0.01), ΔT / 2)

    @inline residual(T) = first(saturated_density_residual(T, θ₀, ρ, qᵗ, pˢᵗ, constants, equilibrium))
    T★ = secant_solve(residual, microphysics.solver, T₁, T₂, T₂)

    _, q = saturated_density_residual(T★, θ₀, ρ, qᵗ, pˢᵗ, constants, equilibrium)
    return with_moisture(𝒰₀, q)
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
