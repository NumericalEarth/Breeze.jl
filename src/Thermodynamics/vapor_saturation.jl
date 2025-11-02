"""
    saturation_vapor_pressure(T, thermo, phase::CondensedPhase)

Compute the saturation vapor pressure over a liquid surface by integrating
the Clausius-Clapeyron relation,

```math
dp/dT = ℒᵛ / (Rᵛ T^2)
```

which integrates to the expression

```math
p(T) = pᵗʳ \\left ( \\frac{T}{Tᵗʳ} \\right )^{aᵛ} \\exp \\left [ bᵛ (1/Tᵗʳ - 1/T) \\right ]
```

where

```math
aᵛ ≡ (cᵖᵛ - cᵖˡ) / Rᵛ \\\\
bᵛ ≡ ℒ₀ / Rᵛ - aᵛ T₀
```
"""
@inline function saturation_vapor_pressure(T, thermo, phase::CondensedPhase)
    ℒ₀ = phase.latent_heat
    cᵖˡ = phase.heat_capacity
    T₀ = thermo.energy_reference_temperature
    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    cᵖᵛ = thermo.vapor.heat_capacity
    Rᵛ = vapor_gas_constant(thermo)

    aᵛ = (cᵖᵛ - cᵖˡ) / Rᵛ
    bᵛ = ℒ₀ / Rᵛ - aᵛ * T₀

    return pᵗʳ * (T / Tᵗʳ)^aᵛ * exp(bᵛ * (1/Tᵗʳ - 1/T))
end

# Over a liquid surface
@inline function saturation_specific_humidity(T, ρ, thermo, condensed_phase::CondensedPhase)
    p★ = saturation_vapor_pressure(T, thermo, condensed_phase)
    Rᵛ = vapor_gas_constant(thermo)
    return p★ / (ρ * Rᵛ * T)
end
