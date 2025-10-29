using Oceananigans

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceState{FT}
    base_pressure :: FT # base pressure: reference pressure at z=0
    potential_temperature :: FT  # constant reference potential temperature
end

function Base.summary(ref::ReferenceState)
    return string("ReferenceState(",
                  "p₀=", prettysummary(ref.base_pressure), ", ",
                  "θᵣ=", prettysummary(ref.potential_temperature), ")")
end

Base.show(io::IO, ref::ReferenceState) = print(io, summary(ref))

function ReferenceState(FT = Oceananigans.defaults.FloatType;
                        base_pressure = 101325,
                        potential_temperature = 288)

    return ReferenceState{FT}(convert(FT, base_pressure),
                              convert(FT, potential_temperature))
end

"""
    reference_density(ref, thermo)

Compute the reference density associated with the reference pressure and potential temperature.
The reference density is defined as the density of dry air at the reference pressure and temperature.
"""
@inline function reference_density(z, ref::ReferenceState, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    pᵣ = reference_pressure(z, ref, thermo)
    ρ₀ = base_density(ref, thermo)
    p₀ = ref.base_pressure
    return ρ₀ * (pᵣ / p₀)^(1 - Rᵈ / cᵖᵈ)
end

@inline function base_density(ref::ReferenceState, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p₀ = ref.base_pressure
    θᵣ = ref.potential_temperature
    return p₀ / (Rᵈ * θᵣ)
end

@inline function reference_specific_volume(z, ref::ReferenceState, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    θᵣ = ref.potential_temperature
    return Rᵈ * θᵣ / pᵣ
end

@inline function reference_pressure(z, ref::ReferenceState, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    g = thermo.gravitational_acceleration
    θᵣ = ref.potential_temperature
    p₀ = ref.base_pressure
    return p₀ * (1 - g * z / (cᵖᵈ * θᵣ))^(cᵖᵈ / Rᵈ)
end

@inline function exner_function(q, z, ref, thermo)
    Rᵐ = mixture_gas_constant(q, thermo)
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    inv_ϰᵐ = Rᵐ / cᵖᵐ
    pᵣ = reference_pressure(z, ref, thermo)
    p₀ = ref.base_pressure
    return (pᵣ / p₀)^inv_ϰᵐ
end

@inline function saturation_specific_humidity(T, z, ref::ReferenceState, thermo, condensed_phase)
    ρ = reference_density(z, ref, thermo)
    return saturation_specific_humidity(T, ρ, thermo, condensed_phase)
end
