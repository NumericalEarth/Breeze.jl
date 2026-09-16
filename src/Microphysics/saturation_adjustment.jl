using Breeze.Thermodynamics:
    Thermodynamics,
    MoistureMassFractions,
    mixture_heat_capacity,
    saturation_specific_humidity,
    adjustment_saturation_specific_humidity,
    density,
    temperature,
    is_absolute_zero,
    with_moisture,
    with_temperature,
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

@inline AtmosphereModels.microphysical_velocities(::SaturationAdjustment, μ, name) = nothing

# SaturationAdjustment operates through the thermodynamic state adjustment pathway,
# so no explicit model update is needed.
AtmosphereModels.microphysics_model_update!(::SaturationAdjustment, model) = nothing

#####
##### Warm-phase equilibrium moisture fractions
#####

@inline function equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, ::WarmPhaseEquilibrium, precipitation=(0, 0))
    qʳ, qˢⁿ = precipitation
    qᵉ = qᵗ - qʳ - qˢⁿ
    qᶜˡ = max(0, qᵉ - qᵛ⁺)
    qᵛ = qᵉ - qᶜˡ
    return MoistureMassFractions(qᵛ, qᶜˡ + qʳ, oftype(qᵛ, qˢⁿ))
end

#####
##### Mixed-phase equilibrium moisture fractions
#####

@inline function equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium::MixedPhaseEquilibrium, precipitation=(0, 0))
    surface = equilibrated_surface(equilibrium, T)
    λ = surface.liquid_fraction
    qʳ, qˢⁿ = precipitation
    qᵉ = qᵗ - qʳ - qˢⁿ
    qᶜ = max(0, qᵉ - qᵛ⁺)
    qᵛ = qᵉ - qᶜ
    qˡ = λ * qᶜ + qʳ
    qⁱ = (1 - λ) * qᶜ + qˢⁿ
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

# Pressure-based states saturate at fixed pressure and total water. Density-based
# states saturate at their own density, with pressure diagnosed by the equation of state.
@inline saturation_adjustment_specific_humidity(T, 𝒰, constants, equilibrium) =
    adjustment_saturation_specific_humidity(T, 𝒰.reference_pressure, total_specific_moisture(𝒰), constants, equilibrium)

@inline saturation_adjustment_specific_humidity(T, 𝒰::LiquidIceDensityState, constants, equilibrium) =
    saturation_specific_humidity(T, 𝒰.density, constants, equilibrium)

@inline function adjust_state(𝒰₀, T, constants, equilibrium, precipitation=(0, 0))
    qᵗ = total_specific_moisture(𝒰₀)
    qᵛ⁺ = saturation_adjustment_specific_humidity(T, 𝒰₀, constants, equilibrium)
    q₁ = equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium, precipitation)
    return with_moisture(𝒰₀, q₁)
end

@inline function saturation_adjustment_residual(T, 𝒰₀, constants, equilibrium, precipitation=(0, 0))
    𝒰₁ = adjust_state(𝒰₀, T, constants, equilibrium, precipitation)
    return saturation_adjustment_residual(T, 𝒰₁, constants)
end

@inline saturation_adjustment_residual(T, 𝒰, constants) = T - temperature(𝒰, constants)

# At fixed density, solve θˡⁱ(T, ρ, q) - θ₀ = 0 directly. Reusing with_temperature
# preserves the equation of state without nesting the θˡⁱ-to-T inversion in the secant solve.
@inline function saturation_adjustment_residual(T, 𝒰::LiquidIceDensityState, constants)
    return with_temperature(𝒰, T, constants).potential_temperature - 𝒰.potential_temperature
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

The state selects the constraint: fixed pressure for pressure-based states, or fixed
density with pressure `p = ρ Rᵐ T` for `LiquidIceDensityState`. The conserved
thermodynamic variable, initial temperature guesses, and solver are retained in either case.

`precipitation = (qʳ, qˢⁿ)` specifies fixed rain and snow mass fractions, included in
the state's total water, heat capacity, gas constant, and latent energy.
"""
@inline function adjust_thermodynamic_state(𝒰₀::ATS, microphysics::SA, constants, precipitation=(0, 0))
    FT = eltype(𝒰₀)
    is_absolute_zero(𝒰₀) && return 𝒰₀

    # Initial temperature with no cloud condensate.
    qᵗ = total_specific_moisture(𝒰₀)
    qʳ, qˢⁿ = precipitation
    qᵉ = qᵗ - qʳ - qˢⁿ
    q₁ = MoistureMassFractions(qᵉ, oftype(qᵗ, qʳ), oftype(qᵗ, qˢⁿ))
    𝒰₁ = with_moisture(𝒰₀, q₁)
    T₁ = temperature(𝒰₁, constants)

    # Unsaturated: keep everything but precipitation as vapor. For pressure-based states,
    # this all-vapor density-based `qᵛ⁺₁` and the pressure-based `adjustment_saturation_specific_humidity`
    # that `adjust_state` iterates with coincide exactly when qᵉ = qᵛ⁺ (both reduce to
    # ϵᵈᵛ (1 - qᵗ) pᵛ⁺ / (p - pᵛ⁺) there), and qᵉ - qᵛ⁺₁(qᵉ) increases with qᵉ, so this test and
    # the secant branch below share a single saturation threshold.
    equilibrium = microphysics.equilibrium
    qᵛ⁺₁ = saturation_specific_humidity(T₁, density(𝒰₁, constants), constants, equilibrium)
    qᵉ ≤ qᵛ⁺₁ && return 𝒰₁

    # First saturated estimate.
    𝒰₁ = adjust_state(𝒰₀, T₁, constants, equilibrium, precipitation)

    # Latent heating from cloud formation sets the second temperature estimate.
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    q̃₁ = 𝒰₁.moisture_mass_fractions
    qˡ₁ = q̃₁.liquid - qʳ
    qⁱ₁ = q̃₁.ice - qˢⁿ
    cᵖᵐ = mixture_heat_capacity(q̃₁, constants)
    ΔT = (ℒˡᵣ * qˡ₁ + ℒⁱᵣ * qⁱ₁) / cᵖᵐ
    ϵT = convert(FT, 0.01) # minimum increment for second guess
    T₂ = T₁ + max(ϵT, ΔT / 2) # reduce the increment, recognizing it is an overshoot

    # Keep total water and precipitation fixed during equilibration.
    @inline residual(T) = saturation_adjustment_residual(T, 𝒰₀, constants, equilibrium, precipitation)
    T★ = secant_solve(residual, microphysics.solver, T₁, T₂, T₂)

    return adjust_state(𝒰₀, T★, constants, equilibrium, precipitation)
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
