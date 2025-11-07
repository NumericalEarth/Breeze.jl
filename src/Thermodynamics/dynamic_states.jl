struct PotentialTemperatureState{FT}
    potential_temperature :: FT
    moisture_fractions :: MoistureMassFractions{FT}
    height :: FT
    base_pressure :: FT
    reference_pressure :: FT
    reference_density :: FT
end

@inline function exner_function(𝒰::PotentialTemperatureState, thermo::ThermodynamicConstants)
    q = 𝒰.moisture_fractions
    z = 𝒰.height
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    pᵣ = 𝒰.reference_pressure
    p₀ = 𝒰.base_pressure
    return (pᵣ / p₀)^(Rᵐ / cᵖᵐ)
end

@inline total_specific_humidity(state::PotentialTemperatureState) =
    total_specific_humidity(state.moisture_fractions)

@inline function specific_volume(state::PotentialTemperatureState, ref, thermo)
    pᵣ = state.reference_pressure
    Rᵐ = mixture_gas_constant(state.moisture_fractions, thermo)
    T = state.potential_temperature
    return Rᵐ * T / pᵣ
end

@inline function saturation_specific_humidity(T,
                                              state::PotentialTemperatureState,
                                              thermo::ThermodynamicConstants,
                                              phase::CondensedPhase)
    z = state.height
    ρ = state.reference_density
    return saturation_specific_humidity(T, ρ, thermo, phase)
end

function condensate_specific_humidity(T, state::PotentialTemperatureState, thermo)
    qᵗ = total_specific_humidity(state)
    qᵛ⁺ = saturation_specific_humidity(T, state, thermo, thermo.liquid)
    return max(0, qᵗ - qᵛ⁺)
end

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
    moisture_fractions :: MoistureMassFractions{FT}
    reference_density :: FT
    reference_pressure :: FT
    exner_function :: FT
end

@inline total_specific_humidity(state::AnelasticThermodynamicState) = total_specific_humidity(state.moisture_fractions)
