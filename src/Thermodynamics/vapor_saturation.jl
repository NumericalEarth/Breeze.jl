"""
    saturation_vapor_pressure(T, thermo, phase::CondensedPhase)

Compute the saturation vapor pressure ``pᵛ⁺`` over a fluid surface at
`phase` (i.e., liquid or solid) from the Clausius-Clapeyron relation,

```math
dpᵛ⁺ / dT = pᵛ⁺ ℒᵛ(T) / (Rᵛ T^2) ,
```

where ``ℒˡ(T) = ℒˡ(T=0) + Δcˡ T``, with ``Δcˡ ≡ (cᵖᵛ - cᵖˡ)`` .

The saturation vapor pressure ``pᵛ⁺`` is obtained after integrating the above from
the triple point, i.e., ``p(Tᵗʳ) = pᵗʳ`` to get

```math
pᵛ⁺(T) = pᵗʳ \\left ( \\frac{T}{Tᵗʳ} \\right )^{Δcˡ / Rᵛ} \\exp \\left [ (1/Tᵗʳ - 1/T) ℒˡ(T=0) / Rᵛ \\right ] .
```

Note that latent heat ``ℒ₀`` is at the reference temperature ``T₀``
and that ``ℒ(T=0) = ℒ₀ - Δcˡ T₀``.
"""
@inline function saturation_vapor_pressure(T, thermo, phase::CondensedPhase)
    ℒᵣ = phase.reference_latent_heat # at thermo.energy_reference_temperature
    Tᵣ = thermo.energy_reference_temperature

    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    Rᵛ = vapor_gas_constant(thermo)

    cᵖᵛ = thermo.vapor.heat_capacity
    cᵝ = phase.heat_capacity
    Δc = cᵝ - cᵖᵛ

    return pᵗʳ * (T / Tᵗʳ)^(Δc / Rᵛ) * exp((1/Tᵗʳ - 1/T) * (ℒᵣ - Δc * Tᵣ) / Rᵛ)
end

# Over a liquid surface
"""
    saturation_specific_humidity(T, ρ, thermo, condensed_phase::CondensedPhase)

Compute the saturation specific humidity for a gas at temperature `T`, total
density `ρ`, `thermo`dynamics, and `condensed_phase` via:

```math
qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T) ,
```

where ``pᵛ⁺`` is the [`saturation_vapor_pressure`](@ref), and ``Rᵛ`` is the specific gas
constant for water vapor.
"""
@inline function saturation_specific_humidity(T, ρ, thermo::ThermodynamicConstants, condensed_phase::CondensedPhase)
    pᵛ⁺ = saturation_vapor_pressure(T, thermo, condensed_phase)
    Rᵛ = vapor_gas_constant(thermo)
    return pᵛ⁺ / (ρ * Rᵛ * T)
end
