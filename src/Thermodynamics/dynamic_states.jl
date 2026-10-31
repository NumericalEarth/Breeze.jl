abstract type AbstractThermodynamicState{FT} end

# States closed on a reference pressure: they carry a `reference_pressure` field, and their density
# is diagnosed from it. Code that reads `𝒰.reference_pressure` dispatches on this type, not on
# `AbstractThermodynamicState` — the density-closed `LiquidIceDensityState` (below) carries ρ
# instead. See NumericalEarth/Breeze.jl#859.
abstract type AbstractReferencePressureState{FT} <: AbstractThermodynamicState{FT} end

@inline Base.eltype(::AbstractThermodynamicState{FT}) where FT = FT

@inline function density(𝒰::AbstractReferencePressureState, constants)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    return density(T, pᵣ, q, constants)
end

"""
$(TYPEDSIGNATURES)

Return the air pressure of the thermodynamic state `𝒰`, in Pa.

States closed on a reference pressure — `LiquidIcePotentialTemperatureState` and
`StaticEnergyState` — carry it directly. `LiquidIceDensityState` is closed on the
density instead, so its pressure is diagnosed from the ideal gas law, `p = ρ Rᵐ T`.
"""
@inline air_pressure(𝒰::AbstractReferencePressureState, constants) = 𝒰.reference_pressure

@inline function saturation_specific_humidity(𝒰::AbstractThermodynamicState, constants, equil)
    T = temperature(𝒰, constants)
    ρ = density(𝒰, constants)
    return saturation_specific_humidity(T, ρ, constants, equil)
end

#####
##### Exner exponents of the liquid-ice potential temperature
#####

"""
$(TYPEDEF)

Abstract supertype for the exponent ``κ`` of the Exner function ``Π = (p / pˢᵗ)^κ`` that
defines the liquid-ice potential temperature ``θˡⁱ``. Subtypes are
[`VaporEquivalentExponent`](@ref) and [`DryExponent`](@ref); both make ``κ`` independent of how
water is split between vapor and condensate, so a phase change at fixed ``θˡⁱ``, pressure, and
total water conserves enthalpy exactly.
"""
abstract type AbstractExnerExponent end

"""
$(TYPEDEF)

The vapor-equivalent Exner exponent ``κ̂ = R̂ / ĉᵖ``. Here ``R̂ = qᵈ Rᵈ + qᵗ Rᵛ`` and
``ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ`` are the gas constant and heat capacity the parcel would have if all of its
water were vapor (see `vapor_equivalent_gas_constant` and
`vapor_equivalent_heat_capacity`). With this exponent ``θˡⁱ = T / Π̂`` in unsaturated air,
and dry or moist unsaturated adiabatic motion conserves ``θˡⁱ`` exactly. This is the default.

# Example

```jldoctest
using Breeze

VaporEquivalentExponent()

# output
VaporEquivalentExponent()
```
"""
struct VaporEquivalentExponent <: AbstractExnerExponent end

"""
$(TYPEDEF)

The dry-air Exner exponent ``κᵈ = Rᵈ / cᵖᵈ``, the convention of most cloud models. The Exner
function then depends on pressure alone. Unsaturated moist
adiabatic motion changes ``θˡⁱ`` slightly, which Breeze's dynamics correct with the same
expansion-work source that corrects saturated motion.

# Example

```jldoctest
using Breeze

DryExponent()

# output
DryExponent()
```
"""
struct DryExponent <: AbstractExnerExponent end

Base.summary(::VaporEquivalentExponent) = "VaporEquivalentExponent()"
Base.summary(::DryExponent) = "DryExponent()"
Base.show(io::IO, κ̂::AbstractExnerExponent) = print(io, summary(κ̂))

"""
$(TYPEDSIGNATURES)

Return the value ``κ`` of the Exner exponent for moisture mass fractions `q`:
``κ̂ = R̂ / ĉᵖ`` for [`VaporEquivalentExponent`](@ref) and ``κᵈ = Rᵈ / cᵖᵈ`` for [`DryExponent`](@ref).

# Example

```jldoctest
using Breeze
using Breeze.Thermodynamics: exner_exponent, MoistureMassFractions

constants = ThermodynamicConstants()
q = MoistureMassFractions(0.012, 0.0, 0.0)
exner_exponent(VaporEquivalentExponent(), q, constants), exner_exponent(DryExponent(), q, constants)

# output
(0.2847851886054834, 0.2855746334945912)
```
"""
@inline exner_exponent(::VaporEquivalentExponent, q, constants) =
    vapor_equivalent_gas_constant(q, constants) / vapor_equivalent_heat_capacity(q, constants)

@inline exner_exponent(::DryExponent, q, constants) =
    dry_air_gas_constant(constants) / constants.dry_air.heat_capacity

# `exner_exponent` for the vapor-only hydrostatic column solve, from the bare scalar constants the
# column kernels receive, paired with the heat capacity ĉᵖ = Rᵐ / κ in the continuous hydrostatic
# balance ∂Π/∂z = -g / (ĉᵖ θ). Without condensate the vapor-equivalent κ̂ is the moist κᵐ = Rᵐ / cᵖᵐ.
@inline column_exner_exponent(::VaporEquivalentExponent, Rᵐ, cᵖᵐ, κᵐ, Rᵈ, cᵖᵈ) = (κᵐ, cᵖᵐ)

@inline function column_exner_exponent(::DryExponent, Rᵐ, cᵖᵐ, κᵐ, Rᵈ, cᵖᵈ)
    κᵈ = Rᵈ / cᵖᵈ
    return κᵈ, Rᵐ / κᵈ
end

#####
##### Liquid-ice potential temperature state
#####

# The liquid-ice potential temperature is defined exactly from the enthalpy,
#
#   θˡⁱ = H / (ĉᵖ Π) ,   H = cᵖᵐ T − ℒˡ₀ qˡ − ℒⁱ₀ qⁱ ,   Π = (p / pˢᵗ)^κ ,
#
# with the vapor-equivalent heat capacity ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ, the latent heats ℒˣ₀ extrapolated
# to absolute zero, and an exponent κ = κ̂ = R̂ / ĉᵖ (`VaporEquivalentExponent`) or κᵈ = Rᵈ / cᵖᵈ
# (`DryExponent`). With Kirchhoff's law ℒˣ(T) = ℒˣ₀ + (cᵖᵛ − cˣ) T this is
#
#   θˡⁱ = [T − (ℒˡ(T) qˡ + ℒⁱ(T) qⁱ) / ĉᵖ] / Π .
#
# H = h − ℒˡ₀ qᵗ differs from the specific enthalpy h by a multiple of the total water, and ĉᵖ Π
# depends only on qᵗ and p, so any phase change (condensation, deposition, freezing, equilibrium
# or not) at fixed θˡⁱ, p and qᵗ conserves h exactly and releases ℒ(T). The inversion is closed:
# T = (ĉᵖ Π θˡⁱ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ) / cᵖᵐ. Adiabatic expansion at fixed θˡⁱ is made consistent with
# the first law by an expansion-work source that the dynamics supply (NumericalEarth/Breeze.jl#1119).
struct LiquidIcePotentialTemperatureState{FT, E} <: AbstractReferencePressureState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    standard_pressure :: FT # pˢᵗ: reference pressure for potential temperature
    reference_pressure :: FT
    exner_exponent :: E     # AbstractExnerExponent: selects κ in Π = (p / pˢᵗ)^κ
end

@inline LiquidIcePotentialTemperatureState(θ::FT, q::MoistureMassFractions{FT}, pˢᵗ::FT, pᵣ::FT) where FT =
    LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, pᵣ, VaporEquivalentExponent())

@inline is_absolute_zero(𝒰::LiquidIcePotentialTemperatureState) = 𝒰.potential_temperature == 0

@inline function exner_function(𝒰::LiquidIcePotentialTemperatureState, constants::ThermodynamicConstants)
    κ = exner_exponent(𝒰.exner_exponent, 𝒰.moisture_mass_fractions, constants)
    pᵣ = 𝒰.reference_pressure
    pˢᵗ = 𝒰.standard_pressure
    return (pᵣ / pˢᵗ)^κ
end

@inline total_specific_moisture(state::LiquidIcePotentialTemperatureState) =
    total_specific_moisture(state.moisture_mass_fractions)

@inline with_moisture(𝒰::LiquidIcePotentialTemperatureState{FT}, q::MoistureMassFractions{FT}) where FT =
    LiquidIcePotentialTemperatureState(𝒰.potential_temperature, q, 𝒰.standard_pressure, 𝒰.reference_pressure,
                                       𝒰.exner_exponent)

@inline function temperature(𝒰::LiquidIcePotentialTemperatureState, constants::ThermodynamicConstants)
    θ = 𝒰.potential_temperature
    Π = exner_function(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    Hᶜ = condensate_latent_energy(q, constants)

    # ĉᵖ Π θˡⁱ = cᵖᵐ T − ℒˡ₀ qˡ − ℒⁱ₀ qⁱ, arranged so that unsaturated air (ĉᵖ = cᵖᵐ, Hᶜ = 0)
    # gives exactly T = Π θˡⁱ
    return Π * θ * (ĉᵖ / cᵖᵐ) + Hᶜ / cᵖᵐ
end

"""
$(TYPEDSIGNATURES)

Compute temperature from liquid-ice potential temperature and pressure,
``T = (ĉᵖ Π θˡⁱ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ) / cᵖᵐ``.

# Arguments
- `θ`: Potential temperature [K]
- `p`: Pressure [Pa]
- `constants`: Thermodynamic constants

# Additional Arguments
- `pˢᵗ`: Standard pressure for potential temperature definition [Pa]
- `q`: Vapor specific humidity [kg/kg] or `MoistureMassFractions` containing the
  vapor, total liquid (cloud + rain), and total ice (cloud + snow) mass fractions.
  Defaults to dry air.
- `κ̂`: The Exner exponent, [`VaporEquivalentExponent`](@ref) (default) or [`DryExponent`](@ref).
"""
@inline function temperature_from_potential_temperature(θ, p, pˢᵗ, constants, q::MoistureMassFractions,
                                                        κ̂=VaporEquivalentExponent())
    FT = promote_type(typeof(θ), typeof(p), typeof(q.vapor))
    𝒰 = LiquidIcePotentialTemperatureState(convert(FT, θ), MoistureMassFractions{FT}(q),
                                           convert(FT, pˢᵗ), convert(FT, p), κ̂)
    return temperature(𝒰, constants)
end

@inline temperature_from_potential_temperature(θ, p, pˢᵗ, constants, qᵛ) =
    temperature_from_potential_temperature(θ, p, pˢᵗ, constants, MoistureMassFractions(qᵛ))

@inline temperature_from_potential_temperature(θ, p, pˢᵗ, constants) =
    temperature_from_potential_temperature(θ, p, pˢᵗ, constants, zero(θ))

@inline temperature_from_potential_temperature(θ, p, constants) =
    temperature_from_potential_temperature(θ, p, 1e5, constants)

"""
$(TYPEDSIGNATURES)

Compute liquid-ice potential temperature from temperature and pressure,
``θˡⁱ = (cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ) / (ĉᵖ Π)``.

# Arguments
- `T`: Temperature [K]
- `p`: Pressure [Pa]
- `constants`: Thermodynamic constants

# Additional Arguments
- `pˢᵗ`: Standard pressure for potential temperature definition [Pa]
- `q`: Vapor specific humidity [kg/kg] or `MoistureMassFractions` containing the
  vapor, total liquid (cloud + rain), and total ice (cloud + snow) mass fractions.
  Defaults to dry air.
- `κ̂`: The Exner exponent, [`VaporEquivalentExponent`](@ref) (default) or [`DryExponent`](@ref).
"""
@inline function potential_temperature_from_temperature(T, p, pˢᵗ, constants, q::MoistureMassFractions,
                                                        κ̂=VaporEquivalentExponent())
    FT = promote_type(typeof(T), typeof(p), typeof(q.vapor))
    T = convert(FT, T)
    𝒰₀ = LiquidIcePotentialTemperatureState(zero(T), MoistureMassFractions{FT}(q),
                                            convert(FT, pˢᵗ), convert(FT, p), κ̂)
    𝒰₁ = with_temperature(𝒰₀, T, constants)
    return 𝒰₁.potential_temperature
end

@inline potential_temperature_from_temperature(T, p, pˢᵗ, constants, qᵛ) =
    potential_temperature_from_temperature(T, p, pˢᵗ, constants, MoistureMassFractions(qᵛ))

@inline potential_temperature_from_temperature(T, p, pˢᵗ, constants) =
    potential_temperature_from_temperature(T, p, pˢᵗ, constants, zero(T))

@inline potential_temperature_from_temperature(T, p, constants) =
    potential_temperature_from_temperature(T, p, 1e5, constants)

@inline function with_temperature(𝒰::LiquidIcePotentialTemperatureState, T, constants)
    Π = exner_function(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    Hᶜ = condensate_latent_energy(q, constants)

    θ = (T - Hᶜ / cᵖᵐ) * (cᵖᵐ / ĉᵖ) / Π # = (cᵖᵐ T − Hᶜ) / (ĉᵖ Π), exactly T / Π in unsaturated air

    return LiquidIcePotentialTemperatureState(θ, q, 𝒰.standard_pressure, 𝒰.reference_pressure, 𝒰.exner_exponent)
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
struct LiquidIceDensityState{FT, S, E} <: AbstractThermodynamicState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    standard_pressure :: FT          # pˢᵗ: reference pressure for potential temperature
    density :: FT                    # ρ: prognostic density (the variable the state is closed on)
    temperature_solver :: S          # solver for the θˡⁱ→T inversion (NewtonSolver, FixedIterations, or Nothing)
    exner_exponent :: E              # AbstractExnerExponent: selects κ in Π = (p / pˢᵗ)^κ
end

# Convenience constructors with the default Newton solver and the default exponent — used by tests
# and any construction site without a formulation to source them from.
@inline LiquidIceDensityState(θ::FT, q::MoistureMassFractions{FT}, pˢᵗ::FT, ρ::FT) where FT =
    LiquidIceDensityState(θ, q, pˢᵗ, ρ, NewtonSolver(FT))

@inline LiquidIceDensityState(θ::FT, q::MoistureMassFractions{FT}, pˢᵗ::FT, ρ::FT, solver) where FT =
    LiquidIceDensityState(θ, q, pˢᵗ, ρ, solver, VaporEquivalentExponent())

@inline is_absolute_zero(𝒰::LiquidIceDensityState) = 𝒰.potential_temperature == 0

@inline total_specific_moisture(state::LiquidIceDensityState) =
    total_specific_moisture(state.moisture_mass_fractions)

@inline with_moisture(𝒰::LiquidIceDensityState{FT}, q::MoistureMassFractions{FT}) where FT =
    LiquidIceDensityState(𝒰.potential_temperature, q, 𝒰.standard_pressure, 𝒰.density,
                          𝒰.temperature_solver, 𝒰.exner_exponent)

# Density is carried directly (not derived from a reference pressure), so the pressure
# is the one quantity that has to be diagnosed rather than read off the state.
@inline density(𝒰::LiquidIceDensityState, constants) = 𝒰.density

@inline function air_pressure(𝒰::LiquidIceDensityState, constants)
    q = 𝒰.moisture_mass_fractions
    Rᵐ = mixture_gas_constant(q, constants)
    return 𝒰.density * Rᵐ * temperature(𝒰, constants)
end

# Invert θˡⁱ at constant density: cᵖᵐ T = ĉᵖ (ρ Rᵐ T / pˢᵗ)^κ θˡⁱ + ℒˡ₀ qˡ + ℒⁱ₀ qⁱ, that is,
# solve  r(T) = T − (ρ Rᵐ T / pˢᵗ)^κ θ̃ − L = 0  with θ̃ = ĉᵖ θˡⁱ / cᵖᵐ and L = (ℒˡ₀ qˡ + ℒⁱ₀ qⁱ) / cᵖᵐ.
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
    κ   = exner_exponent(𝒰.exner_exponent, q, constants)
    γ   = 1 / (1 - κ)
    θ̃   = θ * (vapor_equivalent_heat_capacity(q, constants) / cᵖᵐ)
    L   = condensate_latent_energy(q, constants) / cᵖᵐ

    T₁ = θ̃^γ * (ρ * Rᵐ / pˢᵗ)^(γ - 1) + L     # dry closed form + latent shift (the non-iterated guess)

    return solve_temperature(𝒰.temperature_solver, T₁, θ̃, ρ, Rᵐ, pˢᵗ, κ, L)
end

# The Newton iteration below is written with the loop body inline and plain scalar
# arguments — NOT through `Solvers.newton_solve` with a residual closure — because the
# Reactant GPU kernel-raising pipeline fails on the closure form ("failed to run pass
# manager on module") when Enzyme reverse-differentiates the compressible time step
# (NumericalEarth/Breeze.jl#780). The solver abstraction is preserved through dispatch.
@inline solve_temperature(::Nothing, T, θ, ρ, Rᵐ, pˢᵗ, κ, L) = T

@inline function solve_temperature(solver::NewtonSolver, T, θ, ρ, Rᵐ, pˢᵗ, κ, L)
    ΔT = T                                     # guarantees the convergence test fails before the first step
    iter = 0
    while abs(ΔT) > max(solver.abstol, solver.reltol * T) && iter < solver.maxiter
        Φ  = (ρ * Rᵐ * T / pˢᵗ)^κ * θ          # = T − L at the root
        ΔT = -(T - Φ - L) / (1 - κ * Φ / T)
        T += ΔT
        iter += 1
    end
    return T
end

@inline function solve_temperature(solver::FixedIterations, T, θ, ρ, Rᵐ, pˢᵗ, κ, L)
    for _ in 1:solver.iterations               # fixed trip count → unrolls (Reactant/Enzyme-safe)
        Φ = (ρ * Rᵐ * T / pˢᵗ)^κ * θ
        T += -(T - Φ - L) / (1 - κ * Φ / T)
    end
    return T
end

# Exner function at the diagnosed (actual) pressure p = ρRᵐT: Π = (p/pˢᵗ)^κ.
@inline function exner_function(𝒰::LiquidIceDensityState, constants::ThermodynamicConstants)
    q   = 𝒰.moisture_mass_fractions
    Rᵐ  = mixture_gas_constant(q, constants)
    κ   = exner_exponent(𝒰.exner_exponent, q, constants)
    p   = 𝒰.density * Rᵐ * temperature(𝒰, constants)
    return (p / 𝒰.standard_pressure)^κ
end

@inline function with_temperature(𝒰::LiquidIceDensityState, T, constants)
    q   = 𝒰.moisture_mass_fractions
    ρ   = 𝒰.density
    pˢᵗ = 𝒰.standard_pressure
    Rᵐ  = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ĉᵖ  = vapor_equivalent_heat_capacity(q, constants)
    Hᶜ  = condensate_latent_energy(q, constants)
    κ   = exner_exponent(𝒰.exner_exponent, q, constants)
    p   = ρ * Rᵐ * T
    θ   = (T - Hᶜ / cᵖᵐ) * (cᵖᵐ / ĉᵖ) * (pˢᵗ / p)^κ
    return LiquidIceDensityState(θ, q, pˢᵗ, ρ, 𝒰.temperature_solver, 𝒰.exner_exponent)
end

#####
##### Moist static energy state (for microphysics interfaces)
#####

struct StaticEnergyState{FT} <: AbstractReferencePressureState{FT}
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
    s = 𝒰.static_energy
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)

    g = constants.gravitational_acceleration
    z = 𝒰.height
    Hᶜ = condensate_latent_energy(q, constants)

    # With composition-dependent heat capacity, the latent offsets are at T = 0.
    # This makes phase exchange release ℒ(T), including the heat-capacity correction.
    # s = cᵖᵐ * T + g * z - ℒˡ₀ * qˡ - ℒⁱ₀ * qⁱ
    return (s - g * z + Hᶜ) / cᵖᵐ
end

@inline function with_temperature(𝒰::StaticEnergyState, T, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    g = constants.gravitational_acceleration
    z = 𝒰.height
    Hᶜ = condensate_latent_energy(q, constants)

    s = cᵖᵐ * T + g * z - Hᶜ

    return StaticEnergyState(s, q, z, 𝒰.reference_pressure)
end

#####
##### Internal energy at fixed density (for the total-energy formulation)
#####

"""
$(TYPEDSIGNATURES)

Return the mixture heat capacity at constant volume, ``cᵛᵐ = cᵖᵐ - Rᵐ``, that is
``cᵛᵐ = qᵈ cᵛᵈ + qᵛ cᵛᵛ + qˡ cˡ + qⁱ cⁱ`` with ``cᵛᵈ = cᵖᵈ - Rᵈ`` and ``cᵛᵛ = cᵖᵛ - Rᵛ``
(the condensed phases are incompressible).

```jldoctest
using Breeze.Thermodynamics: mixture_isochoric_heat_capacity, MoistureMassFractions

constants = ThermodynamicConstants()
mixture_isochoric_heat_capacity(MoistureMassFractions(0.0), constants)

# output
717.9974933379358
```
"""
@inline mixture_isochoric_heat_capacity(q::MMF, constants) =
    mixture_heat_capacity(q, constants) - mixture_gas_constant(q, constants)

"""
$(TYPEDSIGNATURES)

Return the latent energy of moist air relative to liquid water at zero kelvin,
``ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ``, with the latent heats ``ℒˡ₀`` and ``ℒⁱ₀`` extrapolated to absolute
zero ([`absolute_zero_latent_heat`](@ref)). Vapor carries ``ℒˡ₀`` above liquid water and ice
``ℒⁱ₀ - ℒˡ₀`` below it, so that the specific internal energy is
``𝓊 = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ`` and the specific enthalpy is
``h = cᵖᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ``.

```jldoctest
using Breeze.Thermodynamics: liquid_referenced_latent_energy, MoistureMassFractions

constants = ThermodynamicConstants()
liquid_referenced_latent_energy(MoistureMassFractions(0.01, 0.0, 0.0), constants)

# output
31375.1265
```
"""
@inline function liquid_referenced_latent_energy(q::MMF, constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    return ℒˡ₀ * q.vapor + (ℒˡ₀ - ℒⁱ₀) * q.ice
end

"""
$(TYPEDEF)

Thermodynamic state closed on the specific internal energy ``𝓊`` and the density ``ρ``,

```math
𝓊 = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ - ℒⁱ₀) qⁱ ,
```

with liquid water at zero kelvin as the origin and constant heat capacities
(see [`liquid_referenced_latent_energy`](@ref) and [`mixture_isochoric_heat_capacity`](@ref)).
The temperature is the closed form ``T = (𝓊 - ℒˡ₀ qᵛ - (ℒˡ₀ - ℒⁱ₀) qⁱ) / cᵛᵐ`` and the pressure
``p = ρ Rᵐ T``. A phase change at fixed ``𝓊`` and ``ρ`` conserves internal energy, so saturation
adjustment of this state is adjustment at fixed internal energy and density. This is the state
the total-energy formulation diagnoses, with ``𝓊 = E - K - g z`` the total energy per unit mass
less its kinetic and potential parts.

```jldoctest
using Breeze.Thermodynamics: InternalEnergyDensityState, MoistureMassFractions, with_temperature, temperature

constants = ThermodynamicConstants()
q = MoistureMassFractions(0.012, 0.001, 0.0)
𝒰 = InternalEnergyDensityState(0.0, q, 1.1)
𝒰 = with_temperature(𝒰, 290.0, constants)
temperature(𝒰, constants)

# output
290.0
```
"""
struct InternalEnergyDensityState{FT} <: AbstractThermodynamicState{FT}
    internal_energy :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    density :: FT
end

@inline is_absolute_zero(𝒰::InternalEnergyDensityState) = 𝒰.internal_energy == 0

@inline total_specific_moisture(𝒰::InternalEnergyDensityState) =
    total_specific_moisture(𝒰.moisture_mass_fractions)

@inline with_moisture(𝒰::InternalEnergyDensityState{FT}, q::MoistureMassFractions{FT}) where FT =
    InternalEnergyDensityState{FT}(𝒰.internal_energy, q, 𝒰.density)

@inline density(𝒰::InternalEnergyDensityState, constants) = 𝒰.density

@inline function temperature(𝒰::InternalEnergyDensityState, constants::ThermodynamicConstants)
    q = 𝒰.moisture_mass_fractions
    cᵛᵐ = mixture_isochoric_heat_capacity(q, constants)
    return (𝒰.internal_energy - liquid_referenced_latent_energy(q, constants)) / cᵛᵐ
end

@inline function with_temperature(𝒰::InternalEnergyDensityState, T, constants)
    q = 𝒰.moisture_mass_fractions
    𝓊 = mixture_isochoric_heat_capacity(q, constants) * T + liquid_referenced_latent_energy(q, constants)
    return InternalEnergyDensityState(𝓊, q, 𝒰.density)
end

@inline function air_pressure(𝒰::InternalEnergyDensityState, constants)
    q = 𝒰.moisture_mass_fractions
    return 𝒰.density * mixture_gas_constant(q, constants) * temperature(𝒰, constants)
end
