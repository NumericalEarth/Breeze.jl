
# No microphysics: no liquid, only vapor
@inline function compute_temperature(state::AnelasticThermodynamicState, ::Nothing, thermo)
    e = state.moist_static_energy
    qᵗ = qᵛ = state.specific_humidity # no condenstae
    qᵈ = 1 - qᵗ
    z = state.height
    g = thermo.gravitational_acceleration
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    ℒ₀ = thermo.liquid.latent_heat
    T₀ = thermo.energy_reference_temperature
    h = e - g * z - qᵗ * ℒ₀
    return h / cᵖᵐ
end

# No microphysics: no liquid, only vapor
@inline function compute_temperature(state::BoussinesqThermodynamicState, ::Nothing, thermo)
    θ = state.potential_temperature
    qᵗ = qᵛ = state.specific_humidity # no condenstae
    qᵈ = 1 - qᵗ
    z = state.height
    g = thermo.gravitational_acceleration
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    ℒ₀ = thermo.liquid.latent_heat
    T₀ = thermo.energy_reference_temperature
    h = e - g * z - qᵗ * ℒ₀
    return h / cᵖᵐ
end