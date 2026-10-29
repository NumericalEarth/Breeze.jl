#####
##### Moist entropy
#####
#
# The specific entropy of moist air is the mass-weighted sum of the partial specific entropies of
# its constituents, with liquid water's entropy zero at the triple point (Tᵗʳ, pᵗʳ),
#
#   η = qᵈ ηᵈ + qᵛ ηᵛ + qˡ ηˡ + qⁱ ηⁱ ,
#
#   ηᵈ = cᵖᵈ ln(T / Tᵗʳ) − Rᵈ ln(pᵈ / pᵗʳ) ,
#   ηᵛ = cᵖᵛ ln(T / Tᵗʳ) − Rᵛ ln(pᵛ / pᵗʳ) + ℒˡ(Tᵗʳ) / Tᵗʳ ,
#   ηˡ = cˡ ln(T / Tᵗʳ) ,
#   ηⁱ = cⁱ ln(T / Tᵗʳ) − ℒᶠ(Tᵗʳ) / Tᵗʳ ,
#
# with the partial pressures pᵈ = ρqᵈ Rᵈ T and pᵛ = ρqᵛ Rᵛ T (Emanuel 1994 §4.5; Pressel et al. 2015).
# The latent heats are Breeze's, ℒˣ(T) = ℒˣ₀ + (cᵖᵛ − cˣ) T with ℒˣ₀ = `absolute_zero_latent_heat`,
# and ℒᶠ = ℒⁱ − ℒˡ. With the constituent enthalpies hᵈ = cᵖᵈ T, hᵛ = cᵖᵛ T + ℒˡ₀, hˡ = cˡ T and
# hⁱ = cⁱ T − (ℒⁱ₀ − ℒˡ₀), which are those of Breeze's energy variables, the specific Gibbs functions
# 𝒢ˣ = hˣ − T ηˣ satisfy
#
#   𝒢ᵛ − 𝒢ˡ = Rᵛ T ln(pᵛ / pᵛ⁺ˡ) ,   𝒢ⁱ − 𝒢ˡ = Rᵛ T ln(pᵛ⁺ⁱ / pᵛ⁺ˡ) ,
#
# exactly, with pᵛ⁺ the Clausius–Clapeyron saturation vapor pressures of Breeze's constants. A
# phase change at saturation is therefore reversible: it leaves η unchanged at fixed enthalpy and
# pressure, or at fixed internal energy and density.
#
# Written per unit mass at fixed temperature, η separates into a temperature part and a part that
# depends on the density and composition only,
#
#   η = cᵛᵐ ln(T / Tᵗʳ) − f(ρ, q) ,
#   f = qᵈ Rᵈ ln(ρqᵈ / ρᵈᵗʳ) + qᵛ Rᵛ ln(ρqᵛ / ρᵛᵗʳ) − qᵛ ℒˡ(Tᵗʳ) / Tᵗʳ + qⁱ ℒᶠ(Tᵗʳ) / Tᵗʳ ,
#
# with cᵛᵐ = cᵖᵐ − Rᵐ (the heat capacity of condensate is c at constant pressure or volume) and the
# triple-point densities ρᵈᵗʳ = pᵗʳ / (Rᵈ Tᵗʳ), ρᵛᵗʳ = pᵗʳ / (Rᵛ Tᵗʳ). At fixed density the
# inversion is therefore closed form, T = Tᵗʳ exp[(η + f) / cᵛᵐ], with or without condensate.
# The vapor term qᵛ ln(ρqᵛ) takes its limit zero at qᵛ → 0 (and for negative qᵛ).

# ℒˣ(Tᵗʳ), the latent heat of phase `phase` at the triple point
@inline triple_point_latent_heat(constants, phase) =
    absolute_zero_latent_heat(constants, phase) + specific_heat_difference(constants, phase) * constants.triple_point_temperature

# The heat of fusion at the triple point, ℒᶠ(Tᵗʳ) = ℒⁱ(Tᵗʳ) − ℒˡ(Tᵗʳ)
@inline triple_point_fusion_heat(constants) =
    triple_point_latent_heat(constants, constants.ice) - triple_point_latent_heat(constants, constants.liquid)

# Dry-air and vapor densities at the triple point pressure and temperature
@inline triple_point_dry_air_density(constants) =
    constants.triple_point_pressure / (dry_air_gas_constant(constants) * constants.triple_point_temperature)

@inline triple_point_vapor_density(constants) =
    constants.triple_point_pressure / (vapor_gas_constant(constants) * constants.triple_point_temperature)

# x ln(a x), continued by its limit zero for x ≤ 0
@inline function x_log_ax(x, a)
    x₊ = max(0, x)
    return x₊ * log(ifelse(x₊ > 0, a * x₊, one(x₊)))
end

"""
$(TYPEDSIGNATURES)

Return the heat capacity at constant volume of moist air with moisture mass fractions `q`,
``cᵛᵐ = cᵖᵐ - Rᵐ = qᵈ cᵛᵈ + qᵛ cᵛᵛ + qˡ cˡ + qⁱ cⁱ``.
"""
@inline mixture_isochoric_heat_capacity(q::MoistureMassFractions, constants) =
    mixture_heat_capacity(q, constants) - mixture_gas_constant(q, constants)

"""
$(TYPEDSIGNATURES)

Return the density- and composition-dependent part ``f(ρ, q)`` of the moist specific entropy,
``η = cᵛᵐ \\ln(T / Tᵗʳ) - f``, for total density `ρ` and moisture mass fractions `q`:

```math
f = qᵈ Rᵈ \\ln \\frac{ρ qᵈ}{ρᵈᵗʳ} + qᵛ Rᵛ \\ln \\frac{ρ qᵛ}{ρᵛᵗʳ} - qᵛ \\frac{ℒˡ(Tᵗʳ)}{Tᵗʳ} + qⁱ \\frac{ℒᶠ(Tᵗʳ)}{Tᵗʳ} ,
```

where ``ρᵈᵗʳ`` and ``ρᵛᵗʳ`` are the dry-air and vapor densities at the triple point. The vapor
term takes its limit zero as ``qᵛ → 0``.
"""
@inline function entropy_composition_term(ρ, q::MoistureMassFractions, constants)
    Tᵗʳ = constants.triple_point_temperature
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    ℒˡᵗʳ = triple_point_latent_heat(constants, constants.liquid)
    ℒᶠᵗʳ = triple_point_fusion_heat(constants)
    qᵈ = dry_air_mass_fraction(q)

    dry = qᵈ * Rᵈ * log(ρ * qᵈ / triple_point_dry_air_density(constants))
    vapor = Rᵛ * x_log_ax(q.vapor, ρ / triple_point_vapor_density(constants))
    latent = (q.ice * ℒᶠᵗʳ - q.vapor * ℒˡᵗʳ) / Tᵗʳ

    return dry + vapor + latent
end

"""
$(TYPEDSIGNATURES)

Return the moist specific entropy ``η`` [J kg⁻¹ K⁻¹] of air at temperature `T`, total density `ρ`
and moisture mass fractions `q`, with the entropy of liquid water zero at the triple point:

```math
η = qᵈ ηᵈ + qᵛ ηᵛ + qˡ ηˡ + qⁱ ηⁱ = cᵛᵐ \\ln(T / Tᵗʳ) - f(ρ, q) ,
```

where ``ηᵈ = cᵖᵈ \\ln(T/Tᵗʳ) - Rᵈ \\ln(pᵈ/pᵗʳ)``, ``ηᵛ = cᵖᵛ \\ln(T/Tᵗʳ) - Rᵛ \\ln(pᵛ/pᵗʳ) + ℒˡ(Tᵗʳ)/Tᵗʳ``,
``ηˡ = cˡ \\ln(T/Tᵗʳ)`` and ``ηⁱ = cⁱ \\ln(T/Tᵗʳ) - ℒᶠ(Tᵗʳ)/Tᵗʳ``, with partial pressures
``pᵈ = ρ qᵈ Rᵈ T`` and ``pᵛ = ρ qᵛ Rᵛ T`` (see [`entropy_composition_term`](@ref)).

# Example

```jldoctest
using Breeze
using Breeze.Thermodynamics: moist_entropy, MoistureMassFractions

constants = ThermodynamicConstants()
q = MoistureMassFractions(0.015, 0.001, 0.0)
η = moist_entropy(295.0, 1.15, q, constants)
round(η, digits=3)

# output
-1220.75
```
"""
@inline function moist_entropy(T, ρ, q::MoistureMassFractions, constants)
    cᵛᵐ = mixture_isochoric_heat_capacity(q, constants)
    Tᵗʳ = constants.triple_point_temperature
    return cᵛᵐ * log(T / Tᵗʳ) - entropy_composition_term(ρ, q, constants)
end

"""
$(TYPEDSIGNATURES)

Return the temperature of air with moist specific entropy `η`, total density `ρ` and moisture mass
fractions `q`: the closed-form inversion ``T = Tᵗʳ \\exp[(η + f(ρ, q)) / cᵛᵐ]`` of [`moist_entropy`](@ref).
"""
@inline function temperature_from_moist_entropy(η, ρ, q::MoistureMassFractions, constants)
    cᵛᵐ = mixture_isochoric_heat_capacity(q, constants)
    Tᵗʳ = constants.triple_point_temperature
    return Tᵗʳ * exp((η + entropy_composition_term(ρ, q, constants)) / cᵛᵐ)
end

#####
##### Partial specific entropies and Gibbs functions of the constituents
#####

"""
$(TYPEDSIGNATURES)

Partial specific entropy of water vapor at temperature `T` and vapor density `ρᵛ`,
``ηᵛ = cᵖᵛ \\ln(T/Tᵗʳ) - Rᵛ \\ln(pᵛ/pᵗʳ) + ℒˡ(Tᵗʳ)/Tᵗʳ`` with ``pᵛ = ρᵛ Rᵛ T``. The vapor density is
floored at the smallest positive float.
"""
@inline function vapor_entropy(T, ρᵛ, constants)
    Tᵗʳ = constants.triple_point_temperature
    Rᵛ = vapor_gas_constant(constants)
    cᵛᵛ = constants.vapor.heat_capacity - Rᵛ
    ρᵛ₊ = max(ρᵛ, floatmin(typeof(ρᵛ)))
    ℒˡᵗʳ = triple_point_latent_heat(constants, constants.liquid)
    return cᵛᵛ * log(T / Tᵗʳ) - Rᵛ * log(ρᵛ₊ / triple_point_vapor_density(constants)) + ℒˡᵗʳ / Tᵗʳ
end

"""
$(TYPEDSIGNATURES)

Specific entropy of liquid water at temperature `T`, ``ηˡ = cˡ \\ln(T/Tᵗʳ)``.
"""
@inline liquid_entropy(T, constants) =
    constants.liquid.heat_capacity * log(T / constants.triple_point_temperature)

"""
$(TYPEDSIGNATURES)

Specific entropy of ice at temperature `T`, ``ηⁱ = cⁱ \\ln(T/Tᵗʳ) - ℒᶠ(Tᵗʳ)/Tᵗʳ``.
"""
@inline function ice_entropy(T, constants)
    Tᵗʳ = constants.triple_point_temperature
    return constants.ice.heat_capacity * log(T / Tᵗʳ) - triple_point_fusion_heat(constants) / Tᵗʳ
end

"""
$(TYPEDSIGNATURES)

Return ``𝒢ᵛ - 𝒢ˡ = Rᵛ T \\ln(pᵛ / pᵛ⁺ˡ)``, the difference of the specific Gibbs functions of vapor
and liquid water consistent with [`moist_entropy`](@ref), at temperature `T` and vapor density `ρᵛ`.
Here ``pᵛ⁺ˡ`` is the Clausius–Clapeyron saturation vapor pressure over liquid of the constants'
latent heats, whatever saturation formula the microphysics uses. Vapor densities below
`eps` times the triple-point vapor density are floored there, which bounds the logarithm.
"""
@inline function vapor_liquid_gibbs_difference(T, ρᵛ, constants)
    Tᵗʳ = constants.triple_point_temperature
    Rᵛ = vapor_gas_constant(constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    Δcˡ = specific_heat_difference(constants, constants.liquid)
    ρᵛᵗʳ = triple_point_vapor_density(constants)
    ρᵛ₊ = max(ρᵛ, eps(typeof(ρᵛ)) * ρᵛᵗʳ)
    # Rᵛ T ln(pᵛ / pᵗʳ) − Rᵛ T ln(pᵛ⁺ˡ / pᵗʳ), with pᵛ / pᵗʳ = (ρᵛ / ρᵛᵗʳ) (T / Tᵗʳ)
    return Rᵛ * T * (log(ρᵛ₊ / ρᵛᵗʳ) + log(T / Tᵗʳ)) - Δcˡ * T * log(T / Tᵗʳ) - ℒˡ₀ * (T / Tᵗʳ - 1)
end

"""
$(TYPEDSIGNATURES)

Return ``𝒢ⁱ - 𝒢ˡ = Rᵛ T \\ln(pᵛ⁺ⁱ / pᵛ⁺ˡ)``, the difference of the specific Gibbs functions of ice
and liquid water consistent with [`moist_entropy`](@ref), at temperature `T`: negative below the
triple point, where ice is the stable phase.
"""
@inline function ice_liquid_gibbs_difference(T, constants)
    Tᵗʳ = constants.triple_point_temperature
    ℒᶠ₀ = absolute_zero_latent_heat(constants, constants.ice) - absolute_zero_latent_heat(constants, constants.liquid)
    Δc = constants.liquid.heat_capacity - constants.ice.heat_capacity
    return Δc * T * log(T / Tᵗʳ) + ℒᶠ₀ * (T / Tᵗʳ - 1)
end

#####
##### Moist entropy state (density-based)
#####

"""
$(TYPEDEF)

Thermodynamic state closed on the moist specific entropy `entropy` (``η``, see [`moist_entropy`](@ref))
and the total density `density` (``ρ``). The temperature follows in closed form,
``T = Tᵗʳ \\exp[(η + f(ρ, q)) / cᵛᵐ]``, and the pressure from the ideal gas law, ``p = ρ Rᵐ T``.
"""
struct MoistEntropyState{FT} <: AbstractThermodynamicState{FT}
    entropy :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    density :: FT
end

@inline is_absolute_zero(𝒰::MoistEntropyState) = 𝒰.density == 0

@inline total_specific_moisture(𝒰::MoistEntropyState) = total_specific_moisture(𝒰.moisture_mass_fractions)

@inline with_moisture(𝒰::MoistEntropyState{FT}, q::MoistureMassFractions{FT}) where FT =
    MoistEntropyState(𝒰.entropy, q, 𝒰.density)

@inline density(𝒰::MoistEntropyState, constants) = 𝒰.density

@inline temperature(𝒰::MoistEntropyState, constants::ThermodynamicConstants) =
    temperature_from_moist_entropy(𝒰.entropy, 𝒰.density, 𝒰.moisture_mass_fractions, constants)

@inline function air_pressure(𝒰::MoistEntropyState, constants)
    Rᵐ = mixture_gas_constant(𝒰.moisture_mass_fractions, constants)
    return 𝒰.density * Rᵐ * temperature(𝒰, constants)
end

@inline function with_temperature(𝒰::MoistEntropyState, T, constants)
    η = moist_entropy(T, 𝒰.density, 𝒰.moisture_mass_fractions, constants)
    return MoistEntropyState(η, 𝒰.moisture_mass_fractions, 𝒰.density)
end
