using Oceananigans: Oceananigans

#####
##### Reference state computations for Boussinesq and Anelastic models
#####

struct ReferenceStateConstants{FT}
    base_pressure :: FT # base pressure: reference pressure at z=0
    reference_potential_temperature :: FT  # constant reference potential temperature
end

function ReferenceStateConstants(FT = Oceananigans.defaults.FloatType;
                            base_pressure = 101325,
                            potential_temperature = 288)

    return ReferenceStateConstants{FT}(convert(FT, base_pressure),
                                  convert(FT, potential_temperature))
end

"""
    reference_density(z, ref::ReferenceStateConstants, thermo)

Compute the reference density at height `z` that associated with the reference pressure and
potential temperature. The reference density is defined as the density of dry air at the
reference pressure and temperature.
"""
@inline function reference_density(z, ref::ReferenceStateConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    pᵣ = reference_pressure(z, ref, thermo)
    ρ₀ = base_density(ref, thermo)
    p₀ = ref.base_pressure
    return ρ₀ * (pᵣ / p₀)^(1 - Rᵈ / cᵖᵈ)
end

@inline reference_specific_volume(z, ref::ReferenceStateConstants, thermo) =
    1 / reference_density(z, ref, thermo)

@inline function base_density(ref::ReferenceStateConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p₀ = ref.base_pressure
    θᵣ = ref.reference_potential_temperature
    return p₀ / (Rᵈ * θᵣ)
end

@inline function reference_pressure(z, ref::ReferenceStateConstants, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    g = thermo.gravitational_acceleration
    θᵣ = ref.reference_potential_temperature
    p₀ = ref.base_pressure
    return p₀ * (1 - g * z / (cᵖᵈ * θᵣ))^(cᵖᵈ / Rᵈ)
end
