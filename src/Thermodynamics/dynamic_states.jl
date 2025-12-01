abstract type AbstractThermodynamicState{FT} end

@inline Base.eltype(::AbstractThermodynamicState{FT}) where FT = FT

struct PotentialTemperatureState{FT} <: AbstractThermodynamicState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    base_pressure :: FT
    reference_pressure :: FT
end

@inline is_absolute_zero(𝒰::PotentialTemperatureState) = 𝒰.potential_temperature == 0

@inline function exner_function(𝒰::PotentialTemperatureState, thermo::ThermodynamicConstants)
    q = 𝒰.moisture_mass_fractions
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    pᵣ = 𝒰.reference_pressure
    p₀ = 𝒰.base_pressure
    return (pᵣ / p₀)^(Rᵐ / cᵖᵐ)
end

@inline total_specific_moisture(state::PotentialTemperatureState) =
    total_specific_moisture(state.moisture_mass_fractions)

@inline with_moisture(𝒰::PotentialTemperatureState{FT}, q::MoistureMassFractions{FT}) where FT =
    PotentialTemperatureState{FT}(𝒰.potential_temperature, q, 𝒰.base_pressure, 𝒰.reference_pressure)

@inline function temperature(𝒰::PotentialTemperatureState, thermo::ThermodynamicConstants)
    θ = 𝒰.potential_temperature
    Π = exner_function(𝒰, thermo)

    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    return Π*θ + (ℒˡᵣ*qˡ + ℒⁱᵣ*qⁱ) / cᵖᵐ 
end

@inline function density(𝒰::PotentialTemperatureState, thermo)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, thermo)
    q = 𝒰.moisture_mass_fractions
    return density(pᵣ, T, q, thermo)
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

@inline function temperature(𝒰::StaticEnergyState, thermo::ThermodynamicConstants)
    e = 𝒰.static_energy
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, thermo)

    g = thermo.gravitational_acceleration
    z = 𝒰.height

    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    # e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ
    return (e - g*z + ℒˡᵣ*qˡ + ℒⁱᵣ*qⁱ) / cᵖᵐ
end

@inline function density(𝒰::AbstractThermodynamicState, thermo)
    pᵣ = 𝒰.reference_pressure
    T = temperature(𝒰, thermo)
    q = 𝒰.moisture_mass_fractions
    return density(pᵣ, T, q, thermo)
end

@inline function saturation_specific_humidity(𝒰::AbstractThermodynamicState, thermo, equil)
    T = temperature(𝒰, thermo)
    ρ = density(𝒰, thermo)
    return saturation_specific_humidity(T, ρ, thermo, equil)
end
