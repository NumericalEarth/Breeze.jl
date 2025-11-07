"""
    saturation_vapor_pressure(T, thermo, phase::CondensedPhase)

Compute the saturation vapor pressure ``pᵛ⁺`` over a planar surface
composed of the "``β``-phase" (liquid, or ice)
using the Clausius-Clapeyron relation,

```math
dpᵛ⁺ / dT = pᵛ⁺ ℒᵝ(T) / (Rᵛ T^2) ,
```

where the latent heat is ``ℒᵝ(T) = ℒᵝ(T=0) + Δcᵝ T``, with ``Δcᵝ ≡ (cᵖᵛ - cᵝ)`` .

The saturation vapor pressure ``pᵛ⁺`` is obtained after integrating the above from
the triple point, i.e., ``p(Tᵗʳ) = pᵗʳ`` to get

```math
pᵛ⁺(T) = pᵗʳ \\left ( \\frac{T}{Tᵗʳ} \\right )^{Δcᵝ / Rᵛ} \\exp \\left [ (1/Tᵗʳ - 1/T) ℒᵝ(T=0) / Rᵛ \\right ] .
```

Note that latent heat ``ℒᵝ(T=0)`` is the difference between the enthalpy of water vapor
and the phase ``β`` when the temperature is ``T = 0``K, or absolute zero.
We define the latent heat using its value ``ℒᵝᵣ = ℒᵝ(T=Tᵣ)`` at the "energy reference temperature"
``T = Tᵣ`` (usually ``Tᵣ ≡ 273.15``K ``= 0^∘``C), such that

```math
ℒᵝ(T) = ℒᵝᵣ + Δcᵝ (T - Tᵣ), \\quad \text{and} \\quad ℒᵝ(T=0) = ℒᵝᵣ - Δcᵝ Tᵣ`` .
```
"""
@inline function saturation_vapor_pressure(T, thermo, phase::CondensedPhase)
    ℒᵣ = phase.reference_latent_heat # at thermo.energy_reference_temperature
    Tᵣ = thermo.energy_reference_temperature

    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    Rᵛ = vapor_gas_constant(thermo)

    cᵖᵛ = thermo.vapor.heat_capacity
    cᵝ = phase.heat_capacity
    Δcᵝ = cᵖᵛ - cᵝ

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵝ / Rᵛ) * exp((1/Tᵗʳ - 1/T) * (ℒᵣ - Δcᵝ * Tᵣ) / Rᵛ)
end

"""
    saturation_specific_humidity(T, ρ, thermo, phase::CondensedPhase)

Compute the saturation specific humidity for a gas at temperature `T`, total
density `ρ`, `thermo`dynamics, and `phase` via:

```math
qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) ,
```

where ``pᵛ⁺`` is the [`saturation_vapor_pressure`](@ref), ``ρ`` is total density,
and ``Rᵛ`` is the specific gas constant for water vapor.
"""
@inline function saturation_specific_humidity(T, ρ, thermo::ThermodynamicConstants, phase::CondensedPhase)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, phase)
    Rᵛ = vapor_gas_constant(thermo)
    return pᵛ⁺ / (ρ * Rᵛ * T)
end
