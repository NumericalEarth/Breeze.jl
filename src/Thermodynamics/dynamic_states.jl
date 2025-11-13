abstract type AbstractThermodynamicState{FT} end

struct PotentialTemperatureState{FT} <: AbstractThermodynamicState{FT}
    potential_temperature :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    base_pressure :: FT
    reference_pressure :: FT

    @inline function PotentialTemperatureState{FT}(potential_temperature::FT,
                                                   moisture_mass_fractions::MoistureMassFractions{FT},
                                                   base_pressure::FT,
                                                   reference_pressure::FT) where FT
        return new{FT}(potential_temperature, moisture_mass_fractions, base_pressure, reference_pressure)
    end
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

@inline total_moisture_mass_fraction(state::PotentialTemperatureState) =
    total_moisture_mass_fraction(state.moisture_mass_fractions)

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

    return Π * θ + (ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ 
end

#####
##### Moist static energy state (for microphysics interfaces)
#####

struct MoistStaticEnergyState{FT} <: AbstractThermodynamicState{FT}
    moist_static_energy :: FT
    moisture_mass_fractions :: MoistureMassFractions{FT}
    height :: FT
    reference_pressure :: FT

    @inline function MoistStaticEnergyState{FT}(moist_static_energy::FT,
                                                moisture_mass_fractions::MoistureMassFractions{FT},
                                                height::FT,
                                                reference_pressure::FT) where FT
        return new{FT}(moist_static_energy, moisture_mass_fractions, height, reference_pressure)
    end
end

@inline Base.eltype(::MoistStaticEnergyState{FT}) where FT = FT
@inline total_moisture_mass_fraction(state::MoistStaticEnergyState) = total_moisture_mass_fraction(state.moisture_mass_fractions)
@inline is_absolute_zero(𝒰::MoistStaticEnergyState) = 𝒰.moist_static_energy == 0

@inline with_moisture(𝒰::MoistStaticEnergyState{FT}, q::MoistureMassFractions{FT}) where FT =
    MoistStaticEnergyState{FT}(𝒰.moist_static_energy, q, 𝒰.height, 𝒰.reference_pressure)

@inline function temperature(𝒰::MoistStaticEnergyState, thermo::ThermodynamicConstants)
    e = 𝒰.moist_static_energy
    q = 𝒰.moisture_mass_fractions
    cᵖᵐ = mixture_heat_capacity(q, thermo)

    g = thermo.gravitational_acceleration
    z = 𝒰.height

    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat
    qˡ = q.liquid
    qⁱ = q.ice

    # e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ
    return (e - g * z + ℒˡᵣ * qˡ + ℒⁱᵣ * qⁱ) / cᵖᵐ
end
