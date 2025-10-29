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
    return exner_function(q, z, ref, thermo)
end

@inline total_specific_humidity(state::PotentialTemperatureState) =
    total_specific_humidity(state.humidities)

@inline function temperature(state::PotentialTemperatureState, thermo::ThermodynamicConstants)
    θ = state.potential_temperature
    q = state.humidities
    z = state.height
    ref = state.reference_state
    Π = exner_function(q, z, ref, thermo)
    return Π * θ
end

struct MoistStaticEnergyState{FT, H}
    moist_static_energy :: FT
    humidities :: H
    height :: FT
end

@inline total_specific_humidity(state::MoistStaticEnergyState) =
    total_specific_humidity(state.humidities)

 # No microphysics: no liquid, only vapor
@inline function temperature(state::MoistStaticEnergyState, thermo::ThermodynamicConstants)
    e = state.moist_static_energy
    q = state.humidities
    z = state.height
    g = thermo.gravitational_acceleration
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    qᵗ = total_specific_humidity(q)
    ℒ₀ = thermo.liquid.latent_heat
    cᵖᵐ_T = e - g * z - qᵗ * ℒ₀
    return cᵖᵐ_T / cᵖᵐ
end
