"""
    saturation_vapor_pressure(T, thermo)

Compute the saturation vapor pressure over a liquid surface by integrating
the Clausius-Clapeyron relation,

```math
dp/dT = ℒᵛ / (Rᵛ T^2)
```
"""
@inline function saturation_vapor_pressure(T, thermo, phase_transition::PhaseTransition)
    ℒ₀  = phase_transition.latent_heat
    cᵖˡ = phase_transition.heat_capacity
    T₀  = thermo.saturation.energy_reference_temperature
    Tᵗʳ = thermo.saturation.triple_point_temperature
    pᵗʳ = thermo.saturation.triple_point_pressure
    cᵖᵛ = thermo.vapor.heat_capacity
    Rᵛ  = vapor_gas_constant(thermo)

    Δcᵖ = cᵖˡ - cᵖᵛ
    Δϰ = Δcᵖ / Rᵛ

    return pᵗʳ * (T / Tᵗʳ)^Δϰ * exp((ℒ₀ - Δcᵖ * T₀) * (1/Tᵗʳ - 1/T) / Rᵛ)
end

# Over a liquid surface
@inline function saturation_specific_humidity(T, ρ, thermo, phase_transition::PhaseTransition)
    p★ = saturation_vapor_pressure(T, thermo, phase_transition)
    Rᵛ = vapor_gas_constant(thermo)
    return p★ / (ρ * Rᵛ * T)
end
