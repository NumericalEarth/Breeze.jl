#####
##### Saturation adjustment
#####

using ..Thermodynamics:
    AnelasticThermodynamicState,
    mixture_heat_capacity

# No microphysics: no liquid, only vapor
@inline function compute_temperature(::Nothing, thermo, state::AnelasticThermodynamicState)
    e = state.specific_moist_static_energy
    qᵗ = qᵛ = state.specific_humidity # no condenstae
    qᵈ = 1 - qᵗ
    z = state.height
    g = thermo.gravitational_acceleration
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    ℒ₀ = thermo.liquid.latent_heat
    cᵖᵐ_T = e - g * z - qᵗ * ℒ₀
    return cᵖᵐ_T / cᵖᵐ
end

@inline compute_temperature(::WarmPhaseSaturationAdjustment, thermo, state::AnelasticThermodynamicState) =
    compute_temperature(nothing, thermo, state)
