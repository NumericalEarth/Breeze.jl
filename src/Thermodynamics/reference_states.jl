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

@inline function base_density(ref::ReferenceStateConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p₀ = ref.base_pressure
    θᵣ = ref.reference_potential_temperature
    return p₀ / (Rᵈ * θᵣ)
end

@inline function reference_specific_volume(z, ref::ReferenceStateConstants, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    θᵣ = ref.reference_potential_temperature
    return Rᵈ * θᵣ / pᵣ
end

@inline function reference_pressure(z, ref::ReferenceStateConstants, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    g = thermo.gravitational_acceleration
    θᵣ = ref.reference_potential_temperature
    p₀ = ref.base_pressure
    return p₀ * (1 - g * z / (cᵖᵈ * θᵣ))^(cᵖᵈ / Rᵈ)
end

@inline function saturation_specific_humidity(T, z, ref::ReferenceStateConstants, thermo, condensed_phase)
    ρ = reference_density(z, ref, thermo)
    return saturation_specific_humidity(T, ρ, thermo, condensed_phase)
end

function condensate_specific_humidity(T, q, z, ref::ReferenceStateConstants, thermo)
    qᵛ★ = saturation_specific_humidity(T, z, ref, thermo, thermo.liquid)
    return max(0, q - qᵛ★)
end

function ice_specific_humidity(T, q, z, ref::ReferenceStateConstants, thermo)
    qi★ = saturation_specific_humidity(T, z, ref, thermo, thermo.solid)
    return max(0, q - qi★)
end

# TODO Do we need these?
#####
##### state thermodynamics for a Boussinesq model
#####

# Organizing information about the state is a WIP
struct ThermodynamicState{FT}
    θ :: FT
    q :: FT
    z :: FT
end

struct ReferenceState{FT}
    p₀ :: FT # base pressure: reference pressure at z=0
    θ :: FT  # constant reference potential temperature
end

Adapt.adapt_structure(to, ref::ReferenceState) =
    ReferenceState(adapt(to, ref.p₀),
                   adapt(to, ref.θ))

function ReferenceState(FT = Oceananigans.defaults.FloatType;
                        base_pressure = 101325,
                        potential_temperature = 288)

    return ReferenceState{FT}(convert(FT, base_pressure),
                              convert(FT, potential_temperature))
end

"""
    reference_density(z, ref, thermo)

Compute the reference density associated with the reference pressure and potential temperature.
The reference density is defined as the density of dry air at the reference pressure and temperature.
"""
@inline function reference_density(z, ref::ReferenceState, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p = reference_pressure(z, ref, thermo)
    return p / (Rᵈ * ref.θ)
end

@inline function base_density(ref::ReferenceState, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    return ref.p₀ / (Rᵈ * ref.θ)
end

@inline function reference_specific_volume(z, ref::ReferenceState, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p = reference_pressure(z, ref, thermo)
    return Rᵈ * ref.θ / p
end

@inline function reference_pressure(z, ref::ReferenceState, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    inv_ϰᵈ = Rᵈ / cᵖᵈ
    g = thermo.gravitational_acceleration
    return ref.p₀ * (1 - g * z / (cᵖᵈ * ref.θ))^inv_ϰᵈ
end

@inline function saturation_specific_humidity(T, z, ref::ReferenceState, thermo, condensed_phase)
    ρ = reference_density(z, ref, thermo)
    return saturation_specific_humidity(T, ρ, thermo, condensed_phase)
end

@inline function exner_function(state::ThermodynamicState, ref::ReferenceState, thermo)
    Rᵐ = mixture_gas_constant(state.q, thermo)
    cᵖᵐ = mixture_heat_capacity(state.q, thermo)
    pᵣ = reference_pressure(state.z, ref, thermo)
    p₀ = ref.base_pressure
    return (pᵣ / p₀)^(Rᵐ / cᵖᵐ)
end

condensate_specific_humidity(T, state::ThermodynamicState, ref::ReferenceState, thermo) =
    condensate_specific_humidity(T, state.q, state.z, ref, thermo)

function condensate_specific_humidity(T, q, z, ref::ReferenceState, thermo)
    qᵛ★ = saturation_specific_humidity(T, z, ref, thermo, thermo.liquid)
    return max(0, q - qᵛ★)
end