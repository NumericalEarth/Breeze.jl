struct PotentialTemperatureState{FT, H, R}
    potential_temperature :: FT
    humidities :: H
    height :: FT
    reference_state :: R
end


@inline function exner_function(𝒰::PotentialTemperatureState, thermo::ThermodynamicConstants)
    q = 𝒰.humidities
    z = 𝒰.height
    ref = 𝒰.reference_state
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    p₀ = ref.base_pressure
    return (pᵣ / p₀)^(Rᵐ / cᵖᵐ)
end

@inline function reference_pressure(z, ref::ReferenceStateConstants, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    g = thermo.gravitational_acceleration
    θᵣ = ref.reference_potential_temperature
    p₀ = ref.base_pressure
    return p₀ * (1 - g * z / (cᵖᵈ * θᵣ))^(Rᵈ / cᵖᵈ)
end

@inline total_specific_humidity(state::PotentialTemperatureState) =
    total_specific_humidity(state.humidities)

#=
@inline function temperature(𝒰::PotentialTemperatureState, thermo::ThermodynamicConstants)
    θ = 𝒰.potential_temperature
    Π = exner_function(𝒰, thermo)
    return Π * θ
end
=#

# TODO: deprecate this
struct AnelasticThermodynamicState{FT}
    potential_temperature :: FT
    humidities :: SpecificHumidities{FT}
    reference_density :: FT
    reference_pressure :: FT
    exner_function :: FT
end

@inline total_specific_humidity(state::AnelasticThermodynamicState) = total_specific_humidity(state.humidities)
