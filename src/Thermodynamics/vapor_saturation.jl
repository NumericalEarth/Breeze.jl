"""
    saturation_vapor_pressure(T, thermo, phase::CondensedPhase)

Compute the saturation vapor pressure over a liquid surface by integrating
the Clausius-Clapeyron relation,

```math
dp / dT = ℒᵛ(T) / (Rᵛ T^2)
```

where ``ℒᵛ(T) = ℒᵛ(T=0) + Δcᵖ T``, with ``Δcᵖ ≡ (cᵖᵛ - cᵖˡ)``.

We can integrate the above from the triple point, i.e., ``p(Tᵗʳ) = pᵗʳ`` to get

```math
p(T) = pᵗʳ \\left ( \\frac{T}{Tᵗʳ} \\right )^{Δcᵖ / Rᵛ} \\exp \\left [ bᵛ (1/Tᵗʳ - 1/T) \\right ]
```

where

```math
bᵛ ≡ ℒᵛ(T=0) / Rᵛ
```
"""
@inline function saturation_vapor_pressure(T, thermo, phase::CondensedPhase)
    ℒ₀ = phase.latent_heat # at thermo.energy_reference_temperature
    cᵖˡ = phase.heat_capacity
    T₀ = thermo.energy_reference_temperature
    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    cᵖᵛ = thermo.vapor.heat_capacity
    Rᵛ = vapor_gas_constant(thermo)

    Δcᵖ = cᵖᵛ - cᵖˡ

    # latent heat at T = 0 ᵒK assuming temperature-independent specific heats
    ℒ₀ₖ = ℒ₀ - Δcᵖ * T₀

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵖ / Rᵛ) * exp(ℒ₀ₖ / Rᵛ * (1/Tᵗʳ - 1/T))
end

# Over a liquid surface
@inline function saturation_specific_humidity(T, ρ, thermo, condensed_phase::CondensedPhase)
    p★ = saturation_vapor_pressure(T, thermo, condensed_phase)
    Rᵛ = vapor_gas_constant(thermo)
    return p★ / (ρ * Rᵛ * T)
end
