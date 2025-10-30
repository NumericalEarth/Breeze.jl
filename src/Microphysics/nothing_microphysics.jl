using ..Thermodynamics:
    ThermodynamicConstants,
    ReferenceState,
    SpecificHumidities,
    exner_function,
    reference_pressure,
    mixture_heat_capacity,
    mixture_gas_constant


#=
# fully compressible case
@inline function temperature(e, U, q, z, thermo::ThermodynamicConstants)
    Π = exner_function(q, z, ref, thermo)
    return Π * θ
end
=#

@inline function specific_volume(T, q::SpecificHumidities, z,
                                 ref::ReferenceState,
                                 thermo::ThermodynamicConstants)

    Rᵐ = mixture_gas_constant(q, thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    return Rᵐ * T / pᵣ
end
