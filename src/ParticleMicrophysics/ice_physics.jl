#####
##### Per-particle physics of ice for Lagrangian (particle-based) cloud microphysics
#####
##### Ice particles grow by vapor deposition and shrink by sublimation. The growth law is the
##### electrostatic analogy of the droplet growth law: the radius r of the droplet is replaced by the
##### capacitance C of the crystal, the latent heat of condensation ℒˡ by that of sublimation ℒⁱ, and the
##### saturation vapor pressure over liquid by that over ice. Crystals keep a fixed shape, so that
##### C = χ r, where r is the radius of the sphere of equal volume and χ the capacitance ratio of the shape
##### (χ = 1 for spheres). Ice has no solute and no curvature term.
#####
##### Notation (in addition to that of droplet_physics.jl):
#####   r   equivalent-volume radius      χ   capacitance ratio C / r    ρᴵ  ice density
#####   𝒮ⁱ  supersaturation over ice      ℒⁱ  latent heat of sublimation αᵈ  deposition coefficient
#####

using Breeze.Thermodynamics: PlanarIceSurface, ice_latent_heat

"""
$(TYPEDSIGNATURES)

Return the capacitance ratio ``χ = C / r`` of a spheroid with aspect ratio `φ` (polar over equatorial
semi-axis; ``φ < 1`` for oblate plates, ``φ > 1`` for prolate columns), where ``r`` is the radius of
the sphere of equal volume ([Pruppacher and Klett, 2010](@cite pruppacher2010microphysics)). With equatorial
semi-axis ``a = r φ^{-1/3}`` and polar semi-axis ``c = r φ^{2/3}``,

```math
C = \\frac{a ε}{\\arcsin ε} , \\quad ε = \\sqrt{1 - φ^2} \\quad (φ < 1) , \\qquad
C = \\frac{c ε}{\\log[(1 + ε) / φ]} , \\quad ε = \\sqrt{1 - φ^{-2}} \\quad (φ > 1) .
```

```jldoctest
using Breeze

round.((spheroid_capacitance_ratio(1.0), spheroid_capacitance_ratio(0.1), spheroid_capacitance_ratio(4.0)), digits=4)

# output
(1.0, 1.3256, 1.1574)
```
"""
function spheroid_capacitance_ratio(φ)
    φ == 1 && return one(φ)
    if φ < 1
        ε = sqrt(1 - φ^2)
        return φ^(-1 / 3) * ε / asin(ε)
    else
        ε = sqrt(1 - 1 / φ^2)
        return φ^(2 / 3) * ε / log((1 + ε) * φ)
    end
end

"""
$(TYPEDEF)

Parameters of the per-particle physics of ice. See [`IcePhysicsParameters`](@ref).
"""
struct IcePhysicsParameters{FT, D, K, M, C, V, S}
    density :: FT
    capacitance_ratio :: FT
    minimum_radius :: FT
    vapor_diffusivity :: D
    thermal_conductivity :: K
    dynamic_viscosity :: M
    kinetic_correction :: C
    ventilation :: V
    solver :: S
end

"""
$(TYPEDSIGNATURES)

Return `IcePhysicsParameters` with float type `FT`.

Keyword arguments
=================

- `density`: ice density ``ρᴵ`` (kg m⁻³), or an effective density for porous crystals. Default: 917.
- `aspect_ratio`: aspect ratio ``φ`` of the spheroidal crystals, which sets the capacitance ratio
  ``χ = C / r`` ([`spheroid_capacitance_ratio`](@ref)). Default: 1 (spheres).
- `deposition_coefficient`: deposition coefficient ``αᵈ`` of the gas-kinetic correction. Default: 0.5.
- `minimum_radius`: crystals that sublimate below this radius within a step disappear (their radius is
  set to zero and their mass returned to the vapor). Default: 10 nm.
- `vapor_diffusivity`, `thermal_conductivity`, `dynamic_viscosity`, `ventilation`, `solver`:
  as for [`DropletPhysicsParameters`](@ref).
- `kinetic_correction`: Default: [`KineticCorrection`](@ref) with the `deposition_coefficient`.

```jldoctest
using Breeze

IcePhysicsParameters()

# output
IcePhysicsParameters{Float64}:
├── density: 917.0
├── capacitance_ratio: 1.0
├── minimum_radius: 1.0e-8
├── vapor_diffusivity: PowerLawVaporDiffusivity{Float64}(D₀=2.11e-5, T₀=273.15, p₀=101325.0, n=1.94)
├── thermal_conductivity: LinearThermalConductivity{Float64}(K₀=0.0238, K₁=7.11e-5, T₀=273.15)
├── dynamic_viscosity: SutherlandViscosity{Float64}(μ₀=1.716e-5, T₀=273.15, Tˢ=110.4)
├── kinetic_correction: KineticCorrection{Float64}(αᶜ=0.5, αᵀ=1.0, Δᵛ=1.04e-7, Δᵀ=2.16e-7)
├── ventilation: BeardPruppacherVentilation()
└── solver: NewtonSolver{Float64}(reltol=1.49012e-8, abstol=2.22045e-15, maxiter=50)
```
"""
function IcePhysicsParameters(FT = Oceananigans.defaults.FloatType;
                              density = 917,
                              aspect_ratio = 1,
                              deposition_coefficient = 0.5,
                              minimum_radius = 1e-8,
                              vapor_diffusivity = PowerLawVaporDiffusivity(FT),
                              thermal_conductivity = LinearThermalConductivity(FT),
                              dynamic_viscosity = SutherlandViscosity(FT),
                              kinetic_correction = KineticCorrection(FT; condensation_coefficient=deposition_coefficient),
                              ventilation = BeardPruppacherVentilation(),
                              solver = NewtonSolver(FT; reltol=sqrt(eps(FT)), abstol=10eps(FT), maxiter=50))

    χ = spheroid_capacitance_ratio(Float64(aspect_ratio))
    return IcePhysicsParameters(convert(FT, density), convert(FT, χ), convert(FT, minimum_radius),
                                vapor_diffusivity, thermal_conductivity, dynamic_viscosity,
                                kinetic_correction, ventilation, solver)
end

Base.summary(::IcePhysicsParameters{FT}) where FT = "IcePhysicsParameters{$FT}"

function Base.show(io::IO, p::IcePhysicsParameters)
    print(io, summary(p), ":", '\n',
          "├── density: ", prettysummary(p.density), '\n',
          "├── capacitance_ratio: ", prettysummary(p.capacitance_ratio), '\n',
          "├── minimum_radius: ", prettysummary(p.minimum_radius), '\n',
          "├── vapor_diffusivity: ", summary(p.vapor_diffusivity), '\n',
          "├── thermal_conductivity: ", summary(p.thermal_conductivity), '\n',
          "├── dynamic_viscosity: ", summary(p.dynamic_viscosity), '\n',
          "├── kinetic_correction: ", summary(p.kinetic_correction), '\n',
          "├── ventilation: ", summary(p.ventilation), '\n',
          "└── solver: ", summary(p.solver))
end

"""
$(TYPEDSIGNATURES)

Return the mass (kg) of an ice particle with equivalent-volume radius `r`, ``(4π/3) ρᴵ r^3``.
"""
@inline ice_particle_mass(r, parameters) = π * parameters.density * r^3 * 4 / 3

# Radius-independent coefficients of the depositional growth law at fixed T and p:
# Cᵛ = ρᴵ Rᵛ T / (Dᵛ pᵛ⁺ⁱ), Cᵀ = (ℒⁱ / (Rᵛ T) - 1) ℒⁱ ρᴵ / (Kᵀ T), and the self-interaction coefficient Cᴳ.
@inline function depositional_growth_coefficients(T, p, constants, parameters, self_interaction=nothing)
    Rᵛ = vapor_gas_constant(constants)
    Rᵈ = dry_air_gas_constant(constants)
    ρᴵ = parameters.density
    cᵖᵈ = constants.dry_air.heat_capacity
    ℒⁱ = ice_latent_heat(T, constants)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
    Dᵛ = vapor_diffusivity(parameters.vapor_diffusivity, T, p)
    Kᵀ = thermal_conductivity(parameters.thermal_conductivity, T, p)
    ρ = p / (Rᵈ * T)

    Cᵛ = ρᴵ * Rᵛ * T / (Dᵛ * pᵛ⁺)
    Cᵀ = (ℒⁱ / (Rᵛ * T) - 1) * ℒⁱ * ρᴵ / (Kᵀ * T)
    ℓᵛ, ℓᵀ = kinetic_lengths(parameters.kinetic_correction, Dᵛ, Kᵀ, ρ, cᵖᵈ, Rᵛ, Rᵈ, T)
    # The point-sink perturbation of the cell has the same form as for droplets, with ice properties
    Cᴳ = self_interaction_coefficient(self_interaction, T, ρ, ℒⁱ, pᵛ⁺, ρᴵ, Rᵛ, cᵖᵈ)

    return (; Cᵛ, Cᵀ, ℓᵛ, ℓᵀ, Cᴳ)
end

# Φ = r dr/dt = χ fᵛ 𝒮ⁱ G(r), G = 1 / (Cᵛ ψᵛ + Cᵀ ψᵀ - χ fᵛ Cᴳ r), and ∂Φ/∂r. The gas-kinetic factors are
# evaluated at the capacitance C = χ r, the length scale of the vapor and temperature fields of the crystal.
# The crystal's own sink in its cell is 4π ρᴵ r Φ, as for a droplet, so that the self-interaction term is
# χ fᵛ Cᴳ r: the effective radius of the cell bounds the field of the crystal from C, not from r.
@inline function deposition_rate_and_derivative(r, 𝒮ⁱ, fᵛ, coefficients, parameters)
    (; Cᵛ, Cᵀ, ℓᵛ, ℓᵀ, Cᴳ) = coefficients
    χ = parameters.capacitance_ratio
    correction = parameters.kinetic_correction
    ψᵛ, ∂ψᵛ, _ = vapor_kinetic_factor(correction, χ * r, ℓᵛ)
    ψᵀ, ∂ψᵀ, _ = thermal_kinetic_factor(correction, χ * r, ℓᵀ)
    R = Cᵛ * ψᵛ + Cᵀ * ψᵀ
    ∂R = χ * (Cᵛ * ∂ψᵛ + Cᵀ * ∂ψᵀ)
    a = χ * fᵛ * Cᴳ
    unlimited = R - a * r > R / 2
    G = 1 / ifelse(unlimited, R - a * r, R / 2)
    ∂G = - G^2 * ifelse(unlimited, ∂R - a, ∂R / 2)
    Φ = χ * fᵛ * G * 𝒮ⁱ
    ∂Φ = χ * fᵛ * ∂G * 𝒮ⁱ
    return Φ, ∂Φ
end

"""
$(TYPEDSIGNATURES)

Return the depositional growth rate ``r \\, dr/dt`` (m² s⁻¹) of an ice particle with equivalent-volume
radius `r` in air with supersaturation over ice `𝒮ⁱ`, temperature `T` and pressure `p`, with ventilation
factor `fᵛ` ([Pruppacher and Klett, 2010](@cite pruppacher2010microphysics)):

```math
\\frac{dm}{dt} = 4π C fᵛ \\frac{𝒮ⁱ}{Fᵛ + Fᵀ} , \\qquad
Fᵀ = \\left(\\frac{ℒⁱ}{Rᵛ T} - 1\\right) \\frac{ℒⁱ}{K̃ᵀ T} , \\qquad
Fᵛ = \\frac{Rᵛ T}{D̃ᵛ pᵛ⁺ⁱ(T)} ,
```

with ``m = (4π/3) ρᴵ r^3``, capacitance ``C = χ r``, latent heat of sublimation ``ℒⁱ(T)``, saturation
vapor pressure over ice ``pᵛ⁺ⁱ(T)``, and ``D̃ᵛ`` and ``K̃ᵀ`` corrected by the gas-kinetic factors
of `parameters.kinetic_correction` evaluated at ``C``, with deposition coefficient ``αᵈ``.
The optional `self_interaction` is as in [`diffusional_growth_rate`](@ref), with ``C`` in place of ``r``.

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = IcePhysicsParameters()
Φ = depositional_growth_rate(1e-5, 0.1, 1.0, 263.0, 8e4, constants, parameters)
round(Φ, sigdigits=4)

# output
5.739e-12
```
"""
@inline function depositional_growth_rate(r, 𝒮ⁱ, fᵛ, T, p, constants, parameters, self_interaction=nothing)
    coefficients = depositional_growth_coefficients(T, p, constants, parameters, self_interaction)
    Φ, _ = deposition_rate_and_derivative(r, 𝒮ⁱ, fᵛ, coefficients, parameters)
    return Φ
end

"""
$(TYPEDSIGNATURES)

Advance the equivalent-volume radius `r` of an ice particle over the time step `Δt` by deposition
(``𝒮ⁱ > 0``) or sublimation (``𝒮ⁱ < 0``) at fixed ambient conditions, with the backward-Euler
discretization ``r_{n+1}^2 = r_n^2 + 2 Δt \\, Φ(r_{n+1})`` of the growth law
(see [`depositional_growth_rate`](@ref)), solved for ``y = \\log(r_{n+1}^2 / r_n^2)`` as in
[`condensational_growth_step`](@ref).

A particle that would sublimate below `parameters.minimum_radius` within the step disappears: its radius
is set to zero, and particles with zero radius stay inactive.

Returns the named tuple `(radius, deposited_mass)`, where `deposited_mass` is the mass of vapor deposited
during the step (negative for sublimation).

```jldoctest
using Breeze

constants = ThermodynamicConstants()
parameters = IcePhysicsParameters()
step = depositional_growth_step(1e-5, 0.1, 1.0, 263.0, 8e4, 10.0, constants, parameters)
round(step.radius, sigdigits=4), round(step.deposited_mass, sigdigits=3)

# output
(1.469e-5, 8.57e-12)
```
"""
@inline function depositional_growth_step(r, 𝒮ⁱ, fᵛ, T, p, Δt, constants, parameters, self_interaction=nothing)
    correction = parameters.kinetic_correction
    χ = parameters.capacitance_ratio
    coefficients = depositional_growth_coefficients(T, p, constants, parameters, self_interaction)
    (; Cᵛ, Cᵀ, ℓᵛ, ℓᵀ, Cᴳ) = coefficients

    active = r > 0
    r⁰ = ifelse(active, r, one(r))
    x⁰ = r⁰^2
    Φ⁰, _ = deposition_rate_and_derivative(r⁰, 𝒮ⁱ, fᵛ, coefficients, parameters)

    # Sublimation bracket: the minimum radius
    rᵐⁱⁿ = min(parameters.minimum_radius, r⁰ / 2)
    yᵐⁱⁿ = 2 * log(rᵐⁱⁿ / r⁰)

    # Growth bracket: ψ(r′) ≥ r⁰ / (r⁰ + Δ) for r′ ≥ r⁰, and the self-interaction at most halves the resistance
    _, _, ψᵛᵇ = vapor_kinetic_factor(correction, χ * r⁰, ℓᵛ)
    _, _, ψᵀᵇ = thermal_kinetic_factor(correction, χ * r⁰, ℓᵀ)
    Gᵇ = ifelse(Cᴳ > 0, 2, 1) / (Cᵛ * ψᵛᵇ + Cᵀ * ψᵀᵇ)
    ζᵇ = 2 * Δt * χ * fᵛ * Gᵇ * 𝒮ⁱ / x⁰
    yᵍ = log1p(max(0, ζᵇ))

    growing = 𝒮ⁱ ≥ 0
    y⁻ = ifelse(growing, zero(yᵐⁱⁿ), yᵐⁱⁿ) # residual ≤ 0
    y⁺ = ifelse(growing, yᵍ, zero(yᵐⁱⁿ))   # residual ≥ 0
    ζᴱ = 2 * Δt * Φ⁰ / x⁰
    yᴱ = log1p(ifelse(ζᴱ > -1, ζᴱ, zero(ζᴱ)))

    residual = y -> begin
        rⁿ = r⁰ * exp(y / 2)
        Φ, ∂Φ = deposition_rate_and_derivative(rⁿ, 𝒮ⁱ, fᵛ, coefficients, parameters)
        (expm1(y) - 2 * Δt * Φ / x⁰, exp(y) - Δt * ∂Φ * rⁿ / x⁰)
    end

    # Complete sublimation within the step: the residual is positive at the minimum radius
    Φᵐⁱⁿ, _ = deposition_rate_and_derivative(rᵐⁱⁿ, 𝒮ⁱ, fᵛ, coefficients, parameters)
    vanishing = !growing & (expm1(yᵐⁱⁿ) - 2 * Δt * Φᵐⁱⁿ / x⁰ > 0)

    y = bracketed_newton_solve(residual, parameters.solver, yᴱ, y⁻, y⁺)
    survives = active & !vanishing
    ζ = ifelse(survives, expm1(y), -one(y)) # r⁺² / r⁰² - 1
    r⁺ = ifelse(survives, r⁰ * exp(y / 2), zero(r))

    # r⁺³ - r⁰³ = (r⁺² - r⁰²) (r⁺² + r⁺ r⁰ + r⁰²) / (r⁺ + r⁰)
    Δr³ = ζ * x⁰ * (r⁺^2 + r⁺ * r⁰ + x⁰) / (r⁺ + r⁰)
    Δm = ifelse(active, π * parameters.density * Δr³ * 4 / 3, zero(r))

    return (radius = r⁺, deposited_mass = Δm)
end

"""
$(TYPEDSIGNATURES)

Return the terminal fall speed (m s⁻¹, positive downward) of an ice sphere of equivalent-volume radius `r`
and density `parameters.density` in air at temperature `T` and pressure `p`: the Stokes velocity reduced by
the [`drag_correction`](@ref) of Schiller and Naumann at the particle Reynolds number, valid for
``\\mathrm{Re} ≲ 800``. Non-spherical drag is neglected.
"""
@inline function ice_terminal_velocity(r, T, p, constants, parameters)
    g = constants.gravitational_acceleration
    Rᵈ = dry_air_gas_constant(constants)
    ρ = p / (Rᵈ * T)
    μ = dynamic_viscosity(parameters.dynamic_viscosity, T)
    𝕎ˢ = 2 * (parameters.density - ρ) * g * r^2 / (9 * μ)
    # Damped fixed-point iteration of 𝕎 = 𝕎ˢ / fᴰ(Re(𝕎)), which converges monotonically
    𝕎 = 𝕎ˢ
    for _ in 1:12
        Re = 2 * r * ρ * 𝕎 / μ
        𝕎 = sqrt(𝕎 * 𝕎ˢ / drag_correction(Re))
    end
    return 𝕎
end
