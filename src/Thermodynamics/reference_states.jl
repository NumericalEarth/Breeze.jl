using Oceananigans

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceConstants{FT}
    base_pressure :: FT # base pressure: reference pressure at z=0
    reference_potential_temperature :: FT  # constant reference potential temperature
end

function ReferenceConstants(FT = Oceananigans.defaults.FloatType;
                            base_pressure = 101325,
                            potential_temperature = 288)

    return ReferenceConstants{FT}(convert(FT, base_pressure),
                                  convert(FT, potential_temperature))
end

"""
    reference_density(ref, thermo)

Compute the reference density associated with the reference pressure and potential temperature.
The reference density is defined as the density of dry air at the reference pressure and temperature.
"""
@inline function reference_density(z, ref::ReferenceConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    pᵣ = reference_pressure(z, ref, thermo)
    ρ₀ = base_density(ref, thermo)
    p₀ = ref.base_pressure
    return ρ₀ * (pᵣ / p₀)^(1 - Rᵈ / cᵖᵈ)
end

@inline function base_density(ref::ReferenceConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p₀ = ref.base_pressure
    θᵣ = ref.reference_potential_temperature
    return p₀ / (Rᵈ * θᵣ)
end

@inline function reference_specific_volume(z, ref::ReferenceConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    θᵣ = ref.reference_potential_temperature
    return Rᵈ * θᵣ / pᵣ
end

@inline function reference_pressure(z, ref::ReferenceConstants, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    g = thermo.gravitational_acceleration
    θᵣ = ref.reference_potential_temperature
    p₀ = ref.base_pressure
    return p₀ * (1 - g * z / (cᵖᵈ * θᵣ))^(cᵖᵈ / Rᵈ)
end

@inline function saturation_specific_humidity(T, z, ref::ReferenceConstants, thermo, phase_transition)
    ρ = reference_density(z, ref, thermo)
    return saturation_specific_humidity(T, ρ, thermo, phase_transition)
end

function condensate_specific_humidity(T, q, z, ref::ReferenceConstants, thermo)
    qᵛ★ = saturation_specific_humidity(T, z, ref, thermo, thermo.condensation)
    return max(0, q - qᵛ★)
end

function ice_specific_humidity(T, q, z, ref::ReferenceConstants, thermo)
    qi★ = saturation_specific_humidity(T, z, ref, thermo, thermo.deposition)
    return max(0, q - qi★)
end
