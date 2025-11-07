struct PotentialTemperatureState{FT, H, R}
    potential_temperature :: FT
    moisture_fractions :: H
    height :: FT
    reference_state :: R
end

@inline function exner_function(𝒰::PotentialTemperatureState, thermo::ThermodynamicConstants)
    q = 𝒰.moisture_fractions
    z = 𝒰.height
    ref = 𝒰.reference_state
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    p₀ = ref.base_pressure
    return (pᵣ / p₀)^(Rᵐ / cᵖᵐ)
end

@inline total_specific_humidity(state::PotentialTemperatureState) =
    total_specific_humidity(state.moisture_fractions)

@inline function specific_volume(state::PotentialTemperatureState, ref, thermo)
    pᵣ = reference_pressure(state.height, ref, thermo)
    Rᵐ = mixture_gas_constant(state.moisture_fractions, thermo)
    T = state.potential_temperature
    return Rᵐ * T / pᵣ
end

@inline function saturation_specific_humidity(T,
                                              state::PotentialTemperatureState,
                                              thermo::ThermodynamicConstants,
                                              phase::CondensedPhase)
    z = state.height
    ref = state.reference_state
    ρ = reference_density(z, ref, thermo)
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
