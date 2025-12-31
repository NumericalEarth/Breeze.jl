"""
$(TYPEDSIGNATURES)

Compute the [saturation vapor pressure](https://en.wikipedia.org/wiki/Vapor_pressure)
``pᵛ⁺`` over a surface labeled ``β`` (for example, a planar liquid surface, or curved ice surface)
using the Clausius-Clapeyron relation,

```math
𝖽pᵛ⁺ / 𝖽T = pᵛ⁺ ℒᵝ(T) / (Rᵛ T^2) ,
```

where the temperature-dependent latent heat of the surface is ``ℒᵝ(T)``.

Using a model for the latent heat that is linear in temperature, eg

```math
ℒᵝ = ℒᵝ₀ + Δcᵝ T,
```

where ``ℒᵝ₀ ≡ ℒᵝ(T=0)`` is the latent heat at absolute zero and
``Δcᵝ ≡ cᵖᵛ - cᵝ``  is the constant difference between the vapor specific heat
and the specific heat of phase ``β``.

Note that we typically parameterize the latent heat in terms of a reference
temperature ``T = Tᵣ`` that is well above absolute zero. In that case,
the latent heat is written

```math
ℒᵝ = ℒᵝᵣ + Δcᵝ (T - Tᵣ) \\qquad \\text{and} \\qquad ℒᵝ₀ = ℒᵝᵣ - Δcᵝ Tᵣ .
```

Integrating the Clausius-Clapeyron relation with a temperature-linear latent heat model,
from the triple point pressure and temperature ``(pᵗʳ, Tᵗʳ)`` to pressure ``pᵛ⁺``
and temperature ``T``, we obtain

```math
\\log(pᵛ⁺ / pᵗʳ) = - ℒᵝ₀ / (Rᵛ T) + ℒᵝ₀ / (Rᵛ Tᵗʳ) + \\log \\left[ (Δcᵝ / Rᵛ) (T / Tᵗʳ) \\right] ,
```

which then becomes

```math
pᵛ⁺(T) = pᵗʳ (T / Tᵗʳ)^{Δcᵝ / Rᵛ} \\exp \\left [ (1/Tᵗʳ - 1/T) ℒᵝ₀ / Rᵛ \\right ] .
```

!!! note
    Any reference values for pressure and temperature can be used in principle.
    The advantage of using reference values at the triple point is that the same values
    can then be used for both condensation (vapor → liquid) and deposition (vapor → ice).
"""
@inline function saturation_vapor_pressure(T, constants, surface)
    ℒ₀ = absolute_zero_latent_heat(constants, surface)
    Δcᵝ = specific_heat_difference(constants, surface)

    Tᵗʳ = constants.triple_point_temperature
    pᵗʳ = constants.triple_point_pressure
    Rᵛ = vapor_gas_constant(constants)

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵝ / Rᵛ) * exp((1/Tᵗʳ - 1/T) * ℒ₀ / Rᵛ)
end

@inline function specific_heat_difference(constants, phase::CondensedPhase)
    cᵖᵛ = constants.vapor.heat_capacity
    cᵝ = phase.heat_capacity
    return cᵖᵛ - cᵝ
end

@inline function absolute_zero_latent_heat(constants, phase::CondensedPhase)
    ℒᵣ = phase.reference_latent_heat # at constants.energy_reference_temperature
    Δcᵝ = specific_heat_difference(constants, phase)
    Tᵣ = constants.energy_reference_temperature
    return ℒᵣ - Δcᵝ * Tᵣ
end

struct PlanarLiquidSurface end
struct PlanarIceSurface end

"""
$(TYPEDSIGNATURES)

Return `PlanarMixedPhaseSurface` for computing the saturation vapor pressure over
a surface composed of a mixture of liquid and ice, with a given `liquid_fraction`.
"""
struct PlanarMixedPhaseSurface{FT}
    liquid_fraction :: FT
end

@inline specific_heat_difference(constants, ::PlanarLiquidSurface) = specific_heat_difference(constants, constants.liquid)
@inline specific_heat_difference(constants, ::PlanarIceSurface) = specific_heat_difference(constants, constants.ice)
@inline absolute_zero_latent_heat(constants, ::PlanarLiquidSurface) = absolute_zero_latent_heat(constants, constants.liquid)
@inline absolute_zero_latent_heat(constants, ::PlanarIceSurface) = absolute_zero_latent_heat(constants, constants.ice)

@inline function specific_heat_difference(constants, surf::PlanarMixedPhaseSurface)
    Δcˡ = specific_heat_difference(constants, constants.liquid)
    Δcⁱ = specific_heat_difference(constants, constants.ice)
    λ = surf.liquid_fraction
    return λ * Δcˡ + (1 - λ) * Δcⁱ
end

@inline function absolute_zero_latent_heat(constants, surf::PlanarMixedPhaseSurface)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    λ = surf.liquid_fraction
    return λ * ℒˡ₀ + (1 - λ) * ℒⁱ₀
end

"""
$(TYPEDSIGNATURES)

Compute the saturation specific humidity for a gas at temperature `T`, total
density `ρ`, `constants`dynamics, and over `surface` via:

```math
qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) ,
```

where ``pᵛ⁺`` is the [`saturation_vapor_pressure`](@ref) over `surface`, ``ρ`` is total density,
and ``Rᵛ`` is the specific gas constant for water vapor.

# Examples

First we compute the saturation specific humidity over a liquid surface:

```jldoctest saturation
using Breeze
using Breeze.Thermodynamics: PlanarLiquidSurface, PlanarIceSurface, PlanarMixedPhaseSurface

constants = ThermodynamicConstants()
T = 288.0 # Room temperature (K)
p = 101325.0 # Mean sea-level pressure
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
q = zero(Breeze.Thermodynamics.MoistureMassFractions{Float64})
ρ = Breeze.Thermodynamics.density(T, p, q, constants)
qᵛ⁺ˡ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())

# output
0.010359995391195264
```

Note, this is slightly smaller than the saturation specific humidity over an ice surface:

```jldoctest saturation
julia> qᵛ⁺ˡ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
0.011945100768555072
```

If a medium contains a mixture of 40% water and 60% ice that has (somehow) acquired
thermodynamic equilibrium, we can compute the saturation specific humidity
over the mixed phase surface,

```jldoctest saturation
mixed_surface = PlanarMixedPhaseSurface(0.4)
qᵛ⁺ᵐ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, constants, mixed_surface)

# output
0.01128386068542303
```
"""
@inline function saturation_specific_humidity(T, ρ, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    Rᵛ = vapor_gas_constant(constants)
    return pᵛ⁺ / (ρ * Rᵛ * T)
end

"""
$(TYPEDSIGNATURES)

Compute the supersaturation ``𝒮 = pᵛ/pᵛ⁺ - 1`` over a given `surface`.

- ``𝒮 < 0`` indicates subsaturation (evaporation conditions)
- ``𝒮 = 0`` indicates saturation (equilibrium)
- ``𝒮 > 0`` indicates supersaturation (condensation conditions)

# Arguments
- `T`: Temperature
- `ρ`: Total air density  
- `q`: `MoistureMassFractions` containing vapor, liquid, and ice mass fractions
- `constants`: `ThermodynamicConstants`
- `surface`: Surface type (e.g., `PlanarLiquidSurface()`, `PlanarIceSurface()`)
"""
@inline function supersaturation(T, ρ, q::MoistureMassFractions, constants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, constants, surface)
    pᵛ = vapor_pressure(ρ, T, q.vapor, constants)
    return pᵛ / pᵛ⁺ - 1
end

#####
##### Phase equilibrium types
#####

abstract type AbstractPhaseEquilibrium end

"""
    WarmPhaseEquilibrium()

Represents a warm-phase equilibrium where only liquid water condensate is considered.
The equilibrated surface is always a planar liquid surface.
"""
struct WarmPhaseEquilibrium <: AbstractPhaseEquilibrium end

"""
    equilibrated_surface(equilibrium, T)

Return the appropriate surface type for computing saturation vapor pressure
given the phase `equilibrium` model and temperature `T`.
"""
@inline equilibrated_surface(::WarmPhaseEquilibrium, T) = PlanarLiquidSurface()

"""
    MixedPhaseEquilibrium(; freezing_temperature=273.15, homogeneous_ice_nucleation_temperature=233.15)

Represents a mixed-phase equilibrium where both liquid and ice condensates are considered.
The liquid fraction varies linearly with temperature between the freezing temperature
and the homogeneous ice nucleation temperature.
"""
struct MixedPhaseEquilibrium{FT} <: AbstractPhaseEquilibrium
    freezing_temperature :: FT
    homogeneous_ice_nucleation_temperature :: FT
end

function MixedPhaseEquilibrium(FT::DataType = Float64;
                               freezing_temperature = 273.15,
                               homogeneous_ice_nucleation_temperature = 233.15)

    if freezing_temperature < homogeneous_ice_nucleation_temperature
        throw(ArgumentError("`freezing_temperature` must be greater than `homogeneous_ice_nucleation_temperature`"))
    end

    freezing_temperature = convert(FT, freezing_temperature)
    homogeneous_ice_nucleation_temperature = convert(FT, homogeneous_ice_nucleation_temperature)
    return MixedPhaseEquilibrium(freezing_temperature, homogeneous_ice_nucleation_temperature)
end

@inline function equilibrated_surface(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    T′ = clamp(T, Tʰ, Tᶠ)
    λ = (T′ - Tʰ) / (Tᶠ - Tʰ)
    return PlanarMixedPhaseSurface(λ)
end
