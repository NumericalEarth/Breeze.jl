"""
    ClausiusClapeyronThermodynamicConstants{FT, C, I}

Type alias for `ThermodynamicConstants` using the Clausius-Clapeyron formulation
for saturation vapor pressure calculations.
"""
const ClausiusClapeyronThermodynamicConstants{FT, C, I} = ThermodynamicConstants{FT, C, I, ClausiusClapeyron}

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
\\log(pᵛ⁺ / pᵗʳ) = - ℒᵝ₀ / (Rᵛ T) + ℒᵝ₀ / (Rᵛ Tᵗʳ) + (Δcᵝ / Rᵛ) \\log(T / Tᵗʳ) ,
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
@inline function saturation_vapor_pressure(T, constants::ClausiusClapeyronThermodynamicConstants, surface)
    ℒ₀ = absolute_zero_latent_heat(constants, surface)
    Δcᵝ = specific_heat_difference(constants, surface)

    Tᵗʳ = constants.triple_point_temperature
    pᵗʳ = constants.triple_point_pressure
    Rᵛ = vapor_gas_constant(constants)

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵝ / Rᵛ) * exp((1/Tᵗʳ - 1/T) * ℒ₀ / Rᵛ)
end
