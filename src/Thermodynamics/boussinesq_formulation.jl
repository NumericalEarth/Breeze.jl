struct BoussinesqThermodynamicState{FT}
    potential_temperature :: FT
    specific_humidity :: FT
    height :: FT
end

@inline function specific_volume(state, microphysics, ref, thermo)
    T = temperature(state, microphysics, ref, thermo)
    Rᵐ = mixture_gas_constant(q, thermo)
    pᵣ = reference_pressure(z, ref, thermo)
    return Rᵐ * T / pᵣ
end
