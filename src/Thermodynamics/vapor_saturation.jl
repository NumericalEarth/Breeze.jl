"""
$(TYPEDSIGNATURES)

Compute the [saturation vapor pressure](https://en.wikipedia.org/wiki/Vapor_pressure)
``pᵛ⁺`` over a surface labeled ``β`` (for example, a planar liquid surface, or curved ice surface)
using the Clausius-Clapeyron relation,

```math
dpᵛ⁺ / dT = pᵛ⁺ ℒᵝ(T) / (Rᵛ T^2) ,
```

where the temperature-dependent latent heat of the surfaceis ``ℒᵝ(T)``.

Using a model for the latent heat that is linear in temperature, eg

```math
ℒᵝ = ℒᵝ₀ + Δcᵝ T,
```

where ``ℒᵝ₀ ≡ ℒᵝ(T=0)`` is the latent heat at absolute zero and
``Δcᵝ ≡ (cᵖᵛ - cᵝ)``  is the constant difference between the vapor specific heat
and the specific heat of phase ``β``.

Note that we typically parameterize the latent heat interms of a reference
temperature ``T = Tᵣ`` that is well above absolute zero. In that case,
the latent heat is written
 
```math
ℒᵝ = ℒᵝᵣ + Δcᵝ (T - Tᵣ),
\\qquad \\text{and} \\qquad
ℒᵝ₀ = ℒᵝᵣ - Δcᵝ Tᵣ .
```

Integrating the Clausius-Clapeyron relation with a temperature-linear latent heat model,
from the triple point pressure and temperature ``(pᵗʳ, Tᵗʳ)`` to pressure ``pᵛ⁺``
and temperature ``T``, we obtain

```math
log(pᵛ⁺ / pᵗʳ) = - ℒᵝ₀ / (Rᵛ T) + ℒᵝ₀ / (Rᵛ Tᵗʳ) + log(Δcᵝ / Rᵛ * T / Tᵗʳ)
```

Which then becomes

```math
pᵛ⁺(T) = pᵗʳ \\left ( \\frac{T}{Tᵗʳ} \\right )^{Δcᵝ / Rᵛ} \\exp \\left [ (1/Tᵗʳ - 1/T) ℒᵝ₀ / Rᵛ \\right ] .
```
"""
@inline function saturation_vapor_pressure(T, thermo, surface)
    ℒ₀ = absolute_zero_latent_heat(thermo, surface)
    Δcᵝ = specific_heat_difference(thermo, surface)

    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    Rᵛ = vapor_gas_constant(thermo)

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵝ / Rᵛ) * exp((1/Tᵗʳ - 1/T) * ℒ₀ / Rᵛ)
end

@inline function specific_heat_difference(thermo, phase::CondensedPhase)
    cᵖᵛ = thermo.vapor.heat_capacity
    cᵝ = phase.heat_capacity
    return cᵖᵛ - cᵝ
end

@inline function absolute_zero_latent_heat(thermo, phase::CondensedPhase)
    ℒᵣ = phase.reference_latent_heat # at thermo.energy_reference_temperature
    Δcᵝ = specific_heat_difference(thermo, phase)
    Tᵣ = thermo.energy_reference_temperature
    return ℒᵣ - Δcᵝ * Tᵣ
end

struct PlanarLiquidSurface end
struct PlanarIceSurface end

struct PlanarMixedPhaseSurface{FT}
    liquid_fraction :: FT
end

@inline specific_heat_difference(thermo, ::PlanarLiquidSurface) = specific_heat_difference(thermo, thermo.liquid)
@inline specific_heat_difference(thermo, ::PlanarIceSurface) = specific_heat_difference(thermo, thermo.ice)
@inline absolute_zero_latent_heat(thermo, ::PlanarLiquidSurface) = absolute_zero_latent_heat(thermo, thermo.liquid)
@inline absolute_zero_latent_heat(thermo, ::PlanarIceSurface) = absolute_zero_latent_heat(thermo, thermo.ice)


@inline function specific_heat_difference(thermo, surf::PlanarMixedPhaseSurface)
    Δcˡ = specific_heat_difference(thermo, thermo.liquid)
    Δcⁱ = specific_heat_difference(thermo, thermo.ice)
    λ = surf.liquid_fraction
    return λ * Δcˡ + (1 - λ) * Δcⁱ
end

@inline function absolute_zero_latent_heat(thermo, surf::PlanarMixedPhaseSurface)
    ℒˡ₀ = absolute_zero_latent_heat(thermo, thermo.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(thermo, thermo.ice)
    λ = surf.liquid_fraction
    return λ * ℒˡ₀ + (1 - λ) * ℒⁱ₀
end

"""
$(TYPEDSIGNATURES)

Compute the saturation specific humidity for a gas at temperature `T`, total
density `ρ`, `thermo`dynamics, and over `surface` via:

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

thermo = ThermodynamicConstants()
T = 288.0 # Room temperature (K)
p = 101325.0 # Mean sea-level pressure
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(thermo)
q = zero(Breeze.Thermodynamics.MoistureMassFractions{Float64})
ρ = Breeze.Thermodynamics.density(p, T, q, thermo)
qᵛ⁺ˡ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, PlanarLiquidSurface())

# output
0.010359995391195264
```

Note, this is slightly smaller than the saturation specific humidity over an ice surface:

```jldoctest saturation
julia> qᵛ⁺ˡ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, PlanarIceSurface())
0.011945100768555072
```

If a medium contains a mixture of 40% water and 60% ice that has (somehow) acquired
thermodynamic equilibrium, we can compute the saturation specific humidity
over the mixed phase surface,

```jldoctest saturation
mixed_surface = PlanarMixedPhaseSurface(0.4)
qᵛ⁺ᵐ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, mixed_surface)

# output
0.01128386068542303
```

"""
@inline function saturation_specific_humidity(T, ρ, thermo::ThermodynamicConstants, surface)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, surface)
    Rᵛ = vapor_gas_constant(thermo)
    return pᵛ⁺ / (ρ * Rᵛ * T)
end
