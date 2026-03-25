abstract type AbstractThermodynamicState{FT} end

@inline Base.eltype(::AbstractThermodynamicState{FT}) where FT = FT

@inline function density(𝒰::AbstractThermodynamicState, constants)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    return density(T, pᵣ, q, constants)
end

@inline function saturation_specific_humidity(𝒰::AbstractThermodynamicState, constants, equil)
    T = temperature(𝒰, constants)
    ρ = density(𝒰, constants)
    return saturation_specific_humidity(T, ρ, constants, equil)
end

#####
##### Liquid-ice potential temperature state
#####

struct LiquidIcePotentialTemperatureState{FT} <: AbstractThermodynamicState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    standard_pressure :: FT # pˢᵗ: reference pressure for potential temperature
    reference_pressure :: FT
end

@inline is_absolute_zero(𝒰::LiquidIcePotentialTemperatureState) = 𝒰.potential_temperature == 0

@inline function exner_function(𝒰::LiquidIcePotentialTemperatureState, constants::ThermodynamicConstants)
    q = 𝒰.moisture_mass_fractions
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    pᵣ = 𝒰.reference_pressure
    pˢᵗ = 𝒰.standard_pressure
    return (pᵣ / pˢᵗ)^(Rᵐ / cᵖᵐ)
end

@inline total_specific_moisture(state::LiquidIcePotentialTemperatureState) =
    total_specific_moisture(state.moisture_mass_fractions)

@inline with_moisture(𝒰::LiquidIcePotentialTemperatureState{FT}, q::MoistureMassFractions{FT}) where FT =
    LiquidIcePotentialTemperatureState{FT}(𝒰.potential_temperature, q, 𝒰.standard_pressure, 𝒰.reference_pressure)

@inline function temperature(𝒰::LiquidIcePotentialTemperatureState, constants::ThermodynamicConstants)
    θ = 𝒰.potential_temperature
    Π = exner_function(𝒰, constants)

    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    return Π * θ + (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ
end

"""
    temperature_from_potential_temperature(θ, p, constants; pˢᵗ=1e5, qᵛ=0)

Compute temperature from potential temperature and pressure.

This is a convenience function that constructs a `LiquidIcePotentialTemperatureState`
with no condensate and computes temperature using the standard thermodynamic relations.

# Arguments
- `θ`: Potential temperature [K]
- `p`: Pressure [Pa]
- `constants`: Thermodynamic constants

# Keyword Arguments
- `pˢᵗ`: Standard pressure for potential temperature definition [Pa], default: 1e5
- `qᵛ`: Specific humidity [kg/kg], default: 0, dry air
"""
@inline function temperature_from_potential_temperature(θ, p, constants; pˢᵗ=1e5, qᵛ=zero(θ))
    q = MoistureMassFractions(qᵛ)  # vapor only, no condensate
    𝒰 = LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, p)
    return temperature(𝒰, constants)
end

@inline function with_temperature(𝒰::LiquidIcePotentialTemperatureState, T, constants)
    Π = exner_function(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    θ = (T - (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ) / Π

    return LiquidIcePotentialTemperatureState(θ, q, 𝒰.standard_pressure, 𝒰.reference_pressure)
end

@inline function density(𝒰::LiquidIcePotentialTemperatureState, constants)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    return density(T, pᵣ, q, constants)
end

#####
##### Moist static energy state (for microphysics interfaces)
#####

struct StaticEnergyState{FT} <: AbstractThermodynamicState{FT}
    static_energy :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    height :: FT
    reference_pressure :: FT
end

@inline total_specific_moisture(state::StaticEnergyState) = total_specific_moisture(state.moisture_mass_fractions)
@inline is_absolute_zero(𝒰::StaticEnergyState) = 𝒰.static_energy == 0

@inline with_moisture(𝒰::StaticEnergyState{FT}, q::MoistureMassFractions{FT}) where FT =
    StaticEnergyState{FT}(𝒰.static_energy, q, 𝒰.height, 𝒰.reference_pressure)

@inline function temperature(𝒰::StaticEnergyState, constants::ThermodynamicConstants)
    e = 𝒰.static_energy
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)

    g = constants.gravitational_acceleration
    z = 𝒰.height

    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    # e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ
    return (e - g * z + ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ
end

@inline function with_temperature(𝒰::StaticEnergyState, T, constants)
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, constants)
    g = constants.gravitational_acceleration
    z = 𝒰.height
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

    return StaticEnergyState(e, q, z, 𝒰.reference_pressure)
end
