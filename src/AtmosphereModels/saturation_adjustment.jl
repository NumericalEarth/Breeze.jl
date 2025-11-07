#####
##### Saturation adjustment
#####

function condensate_specific_humidity(T, state::AnelasticThermodynamicState, thermo)
    qᵛ⁺ = saturation_specific_humidity(T, state.reference_density, thermo, thermo.liquid)
    q = state.specific_humidity
    return max(0, q - qᵛ⁺)
end

@inline function compute_temperature(state::AnelasticThermodynamicState{FT}, thermo) where FT
    θ = state.potential_temperature
    θ == 0 && return zero(FT)

    # Generate guess for unsaturated conditions
    Π = state.exner_function
    T₁ = Π * θ
    qˡ₁ = condensate_specific_humidity(T₁, state, thermo)
    qˡ₁ <= 0 && return T₁
    
    # If we made it this far, we have condensation
    r₁ = saturation_adjustment_residual(T₁, qˡ₁, state, thermo)

    q = state.specific_humidity
    ℒ = thermo.liquid.reference_latent_heat
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    T₂ = (T₁ + sqrt(T₁^2 + 4 * ℒ * qˡ₁ / cᵖᵐ)) / 2
    qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
    r₂ = saturation_adjustment_residual(T₂, qˡ₂, state, thermo)

    # Saturation adjustment
    R = sqrt(max(T₂, T₁))
    ϵ = convert(FT, 1e-4)
    δ = ϵ * R 
    iter = 0

    while abs(r₂ - r₁) > δ
        # Compute slope
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        # Store previous values
        r₁ = r₂
        T₁ = T₂

        # Update
        T₂ -= r₂ * ΔTΔr
        qˡ₂ = condensate_specific_humidity(T₂, state, thermo)
        r₂ = saturation_adjustment_residual(T₂, qˡ₂, state, thermo)
        iter += 1
    end

    return T₂
end

@inline function saturation_adjustment_residual(T, qˡ, state, thermo)
    ℒᵛ = thermo.liquid.reference_latent_heat
    q = state.specific_humidity
    θ = state.potential_temperature
    Π = state.exner_function
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    return T^2 - ℒᵛ * qˡ / cᵖᵐ - Π * θ * T
end
