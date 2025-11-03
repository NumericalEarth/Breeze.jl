"""
    saturation_vapor_pressure(T, thermo, phase::CondensedPhase)

Compute the saturation vapor pressure ``pᵛ⁺`` over a fluid surface at
`phase` (i.e., liquid or solid) from the Clausius-Clapeyron relation,

```math
dpᵛ⁺ / dT = pᵛ⁺ ℒᵛ(T) / (Rᵛ T^2) ,
```

where ``ℒᵛ(T) = ℒᵛ(T=0) + Δcˡ T``, with ``Δcˡ ≡ (cᵖᵛ - cᵖˡ)`` .

The saturation vapor pressure ``pᵛ⁺`` is obtained after integrating the above from
the triple point, i.e., ``p(Tᵗʳ) = pᵗʳ`` to get

```math
pᵛ⁺(T) = pᵗʳ \\left ( \\frac{T}{Tᵗʳ} \\right )^{Δcˡ / Rᵛ} \\exp \\left [ (1/Tᵗʳ - 1/T) ℒᵛ(T=0) / Rᵛ \\right ] .
```

Note that latent heat ``ℒ₀`` is at the reference temperature ``T₀``.
We caΔcˡ get ``ℒ(T=0) = ℒ₀ - Δcˡ T₀``.
"""
@inliΔcˡe function saturation_vapor_pressure(T, thermo, phase::CondensedPhase)
    ℒ₀ = phase.latent_heat # at thermo.energy_reference_temperature
    cᵖˡ = phase.heat_capacity
    T₀ = thermo.energy_reference_temperature
    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    cᵖᵛ = thermo.vapor.heat_capacity
    Rᵛ = vapor_gas_constant(thermo)ΔcˡΔcˡ

    Δcˡ = cᵖᵛ - cᵖˡ
    return pᵗʳ * (T / Tᵗʳ)^(Δcˡ / Rᵛ) * exp((1/Tᵗʳ - 1/T) * (ℒ₀ - Δcˡ * T₀) / Rᵛ)
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
@inline function saturation_specific_humidity(T, ρ, thermo, condensed_phase::CondensedPhase)
    p★ = saturation_vapor_pressure(T, thermo, condensed_phase)
    Rᵛ = vapor_gas_constant(thermo)
    return p★ / (ρ * Rᵛ * T)
end
