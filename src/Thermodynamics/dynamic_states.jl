abstract type AbstractThermodynamicState{FT} end

@inline Base.eltype(::AbstractThermodynamicState{FT}) where FT = FT

@inline function density(𝒰::AbstractThermodynamicState, constants)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    return density(T, pᵣ, q, constants)
end

@inline function saturation_specific_humidity(𝒰::AbstractThermodynamicState, constants, equil)
    T = temperature(𝒰, constants)
    ρ = density(𝒰, constants)
    return saturation_specific_humidity(T, ρ, constants, equil)
end

#####
##### Liquid-ice potential temperature state
#####

struct LiquidIcePotentialTemperatureState{FT} <: AbstractThermodynamicState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    standard_pressure :: FT # pˢᵗ: reference pressure for potential temperature
    reference_pressure :: FT
end

@inline is_absolute_zero(𝒰::LiquidIcePotentialTemperatureState) = 𝒰.potential_temperature == 0

@inline function exner_function(𝒰::LiquidIcePotentialTemperatureState, constants::ThermodynamicConstants)
    q = 𝒰.moisture_mass_fractions
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    pᵣ = 𝒰.reference_pressure
    pˢᵗ = 𝒰.standard_pressure
    return (pᵣ / pˢᵗ)^(Rᵐ / cᵖᵐ)
end

@inline total_specific_moisture(state::LiquidIcePotentialTemperatureState) =
    total_specific_moisture(state.moisture_mass_fractions)

@inline with_moisture(𝒰::LiquidIcePotentialTemperatureState{FT}, q::MoistureMassFractions{FT}) where FT =
    LiquidIcePotentialTemperatureState{FT}(𝒰.potential_temperature, q, 𝒰.standard_pressure, 𝒰.reference_pressure)

@inline function temperature(𝒰::LiquidIcePotentialTemperatureState, constants::ThermodynamicConstants)
    θ = 𝒰.potential_temperature
    Π = exner_function(𝒰, constants)

    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    return Π * θ + (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ
end

"""
$(TYPEDSIGNATURES)

Compute temperature from potential temperature and pressure.

This is a convenience function that constructs a `LiquidIcePotentialTemperatureState`
with no condensate and computes temperature using the standard thermodynamic relations.

# Arguments
- `θ`: Potential temperature [K]
- `p`: Pressure [Pa]
- `constants`: Thermodynamic constants

# Additional Arguments
- `pˢᵗ`: Standard pressure for potential temperature definition [Pa]
- `qᵛ`: Specific humidity [kg/kg]
"""
@inline function temperature_from_potential_temperature(θ, p, pˢᵗ, constants, qᵛ)
    FT = promote_type(typeof(θ), typeof(p), typeof(qᵛ))
    θ = convert(FT, θ)
    p = convert(FT, p)
    pˢᵗ = convert(FT, pˢᵗ)
    qᵛ = convert(FT, qᵛ)
    q = MoistureMassFractions(qᵛ)  # vapor only, no condensate
    𝒰 = LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, p)
    return temperature(𝒰, constants)
end

@inline temperature_from_potential_temperature(θ, p, pˢᵗ, constants) =
    temperature_from_potential_temperature(θ, p, pˢᵗ, constants, zero(θ))

@inline temperature_from_potential_temperature(θ, p, constants) =
    temperature_from_potential_temperature(θ, p, 1e5, constants)

"""
$(TYPEDSIGNATURES)

Compute potential temperature from temperature and pressure.

This is a convenience function that constructs a `LiquidIcePotentialTemperatureState`
with no condensate and computes potential temperature using the standard thermodynamic relations.

# Arguments
- `T`: Temperature [K]
- `p`: Pressure [Pa]
- `constants`: Thermodynamic constants

# Additional Arguments
- `pˢᵗ`: Standard pressure for potential temperature definition [Pa]
- `qᵛ`: Specific humidity [kg/kg]
"""
@inline function potential_temperature_from_temperature(T, p, pˢᵗ, constants, qᵛ)
    FT = promote_type(typeof(T), typeof(p), typeof(qᵛ))
    T = convert(FT, T)
    p = convert(FT, p)
    pˢᵗ = convert(FT, pˢᵗ)
    qᵛ = convert(FT, qᵛ)
    q = MoistureMassFractions(qᵛ)  # vapor only, no condensate
    𝒰₀ = LiquidIcePotentialTemperatureState(zero(T), q, pˢᵗ, p)
    𝒰₁ = with_temperature(𝒰₀, T, constants)
    return 𝒰₁.potential_temperature
end

@inline potential_temperature_from_temperature(T, p, pˢᵗ, constants) =
    potential_temperature_from_temperature(T, p, pˢᵗ, constants, zero(T))

@inline potential_temperature_from_temperature(T, p, constants) =
    potential_temperature_from_temperature(T, p, 1e5, constants)

@inline function with_temperature(𝒰::LiquidIcePotentialTemperatureState, T, constants)
    Π = exner_function(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    θ = (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Π

    return LiquidIcePotentialTemperatureState(θ, q, 𝒰.standard_pressure, 𝒰.reference_pressure)
end

@inline function density(𝒰::LiquidIcePotentialTemperatureState, constants)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    return density(T, pᵣ, q, constants)
end

#####
##### Liquid-ice potential temperature state (density-based)
#####

# The density-based counterpart of `LiquidIcePotentialTemperatureState`: it carries the density
# ρ rather than a reference pressure, so `temperature` inverts θˡⁱ at constant density (with the
# pressure diagnosed as p = ρ Rᵐ T) and `saturation_specific_humidity` (via the generic
# AbstractThermodynamicState method) is evaluated at the actual ρ. This is the natural closure when ρ
# is prognostic — e.g. on the compressible core, where using the pressure-based
# `LiquidIcePotentialTemperatureState` causes the temperature-inversion inconsistency and the
# saturation-against-a-stale-reference-pressure error. See NumericalEarth/Breeze.jl#765.
struct LiquidIceDensityState{FT, S} <: AbstractThermodynamicState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    standard_pressure :: FT          # pˢᵗ: reference pressure for potential temperature
    density :: FT                    # ρ: prognostic density (the variable the state is closed on)
    temperature_solver :: S          # solver for the θˡⁱ→T inversion (NewtonSolver, FixedIterations, or Nothing)
end

# Convenience constructor with the default Newton solver — used by tests and any construction
# site without a formulation to source `temperature_solver` from.
@inline LiquidIceDensityState(θ::FT, q::MoistureMassFractions{FT}, pˢᵗ::FT, ρ::FT) where FT =
    LiquidIceDensityState(θ, q, pˢᵗ, ρ, NewtonSolver(FT))

@inline is_absolute_zero(𝒰::LiquidIceDensityState) = 𝒰.potential_temperature == 0

@inline total_specific_moisture(state::LiquidIceDensityState) =
    total_specific_moisture(state.moisture_mass_fractions)

@inline with_moisture(𝒰::LiquidIceDensityState{FT}, q::MoistureMassFractions{FT}) where FT =
    LiquidIceDensityState(𝒰.potential_temperature, q, 𝒰.standard_pressure, 𝒰.density,
                          𝒰.temperature_solver)

# Density is carried directly (not derived from a reference pressure).
@inline density(𝒰::LiquidIceDensityState, constants) = 𝒰.density

# Invert θˡⁱ at constant density: solve  r(T) = T − (ρ Rᵐ T / pˢᵗ)^κ θ − (ℒˡ qˡ + ℒⁱ qⁱ)/cᵖᵐ = 0.
# Newton converges quadratically (r′ = 1 − κ Φ/T ≈ 1 − κ ≈ 0.72 is well away from zero). With no
# condensate (L = 0) the dry closed form is already the root, so a Newton step is exactly zero.
#
# The iteration form is selected by the solver type (see Breeze.Solvers):
#   • NewtonSolver     : tolerance-based `while` early-exit — fewest Newton steps on vanilla CPU/GPU.
#   • FixedIterations  : fixed trip count that unrolls to straight-line code. The `while` form traces
#     to an XLA `while` op whose reverse-mode is pathological under Reactant/Enzyme (it hangs the
#     differentiable acoustic_wave docs example — NumericalEarth/Breeze.jl#767); the unrolled form
#     differentiates cheaply. Use `FixedIterations` for differentiable / Reactant runs.
#   • nothing          : the non-iterated closed form below (dry inversion + latent shift).
@inline function temperature(𝒰::LiquidIceDensityState, constants::ThermodynamicConstants)
    θ   = 𝒰.potential_temperature
    ρ   = 𝒰.density
    pˢᵗ = 𝒰.standard_pressure
    q   = 𝒰.moisture_mass_fractions
    Rᵐ  = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    κ   = Rᵐ / cᵖᵐ
    γ   = cᵖᵐ / (cᵖᵐ - Rᵐ)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    L   = (ℒˡᵣ * q.liquid + ℒⁱᵣ * q.ice) / cᵖᵐ

    T₁ = θ^γ * (ρ * Rᵐ / pˢᵗ)^(γ - 1) + L     # dry closed form + latent shift (the non-iterated guess)

    @inline function residual_and_derivative(T)
        Φ = (ρ * Rᵐ * T / pˢᵗ)^κ * θ           # = T − L at the root
        return (T - Φ - L, 1 - κ * Φ / T)
    end

    return newton_solve(residual_and_derivative, 𝒰.temperature_solver, T₁)
end

# Exner function at the diagnosed (actual) pressure p = ρRᵐT: Π = (p/pˢᵗ)^κ.
@inline function exner_function(𝒰::LiquidIceDensityState, constants::ThermodynamicConstants)
    q   = 𝒰.moisture_mass_fractions
    Rᵐ  = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    p   = 𝒰.density * Rᵐ * temperature(𝒰, constants)
    return (p / 𝒰.standard_pressure)^(Rᵐ / cᵖᵐ)
end

@inline function with_temperature(𝒰::LiquidIceDensityState, T, constants)
    q   = 𝒰.moisture_mass_fractions
    ρ   = 𝒰.density
    pˢᵗ = 𝒰.standard_pressure
    Rᵐ  = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    κ   = Rᵐ / cᵖᵐ
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    L   = (ℒˡᵣ * q.liquid + ℒⁱᵣ * q.ice) / cᵖᵐ
    p   = ρ * Rᵐ * T
    θ   = (T - L) * (pˢᵗ / p)^κ
    return LiquidIceDensityState(θ, q, pˢᵗ, ρ, 𝒰.temperature_solver)
end

#####
##### Moist static energy state (for microphysics interfaces)
#####

struct StaticEnergyState{FT} <: AbstractThermodynamicState{FT}
    static_energy :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    height :: FT
    reference_pressure :: FT
end

@inline total_specific_moisture(state::StaticEnergyState) = total_specific_moisture(state.moisture_mass_fractions)
@inline is_absolute_zero(𝒰::StaticEnergyState) = 𝒰.static_energy == 0

@inline with_moisture(𝒰::StaticEnergyState{FT}, q::MoistureMassFractions{FT}) where FT =
    StaticEnergyState{FT}(𝒰.static_energy, q, 𝒰.height, 𝒰.reference_pressure)

@inline function temperature(𝒰::StaticEnergyState, constants::ThermodynamicConstants)
    e = 𝒰.static_energy
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)

    g = constants.gravitational_acceleration
    z = 𝒰.height

    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    # e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ
    return (e - g * z + ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ
end

@inline function with_temperature(𝒰::StaticEnergyState, T, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    g = constants.gravitational_acceleration
    z = 𝒰.height
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

    return StaticEnergyState(e, q, z, 𝒰.reference_pressure)
end
