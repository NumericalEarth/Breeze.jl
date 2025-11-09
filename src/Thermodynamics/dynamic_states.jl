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

@inline total_moisture_fraction(state::PotentialTemperatureState) =
    total_moisture_fraction(state.moisture_fractions)

@inline function with_moisture(𝒰::PotentialTemperatureState, q::MoistureMassFractions)
    return PotentialTemperatureState(𝒰.potential_temperature,
                                     q,
                                     𝒰.height,
                                     𝒰.base_pressure,
                                     𝒰.reference_pressure,
                                     𝒰.reference_density)
end

#####
##### Moist static energy state (for microphysics interfaces)
#####

struct MoistStaticEnergyState{FT}
    moist_static_energy :: FT
    moisture_fractions :: MoistureMassFractions{FT}
    height :: FT
    reference_pressure :: FT
end

@inline Base.eltype(::MoistStaticEnergyState{FT}) where FT = FT
@inline total_moisture_fraction(state::MoistStaticEnergyState) = total_moisture_fraction(state.moisture_fractions)

@inline function with_moisture(𝒰::MoistStaticEnergyState, q::MoistureMassFractions)
    return MoistStaticEnergyState(𝒰.moist_static_energy, q, 𝒰.height, 𝒰.reference_pressure)
end
