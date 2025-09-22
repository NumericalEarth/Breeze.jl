struct BoussinesqThermodynamicState{FT}
    potential_temperature :: FT
    specific_humidity :: FT
    height :: FT
end

@inline function Base.getproperty(state::BoussinesqThermodynamicState, name::Symbol)
    if name === :θ
        return getfield(state, :potential_temperature)
    elseif name === :q
        return getfield(state, :specific_humidity)
    elseif name === :z
        return getfield(state, :height)
    else
        return getfield(state, name)
    end
end

@inline function specific_volume(state, microphysics, ref, thermo)
    T = temperature(state, microphysics, ref, thermo)
    qᵛ = state.q
    qᵈ = 1 - qᵛ
    Rᵐ = mixture_gas_constant(qᵈ, qᵛ, thermo)
    pᵣ = reference_pressure(state.z, ref, thermo)
    return Rᵐ * T / pᵣ
end
