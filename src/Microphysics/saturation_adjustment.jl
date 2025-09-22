#####
##### Microphysics saturation adjustment utilities
#####

# Solve θ = T/Π (1 - ℒ qˡ / (cᵖᵐ T)) for temperature T with qˡ = max(0, q - qᵛ★)
# by iterating on the root of f(T) = T² - Π θ T - ℒ qˡ / cᵖᵐ.
@inline function temperature(state, ref, thermo)
    θ = state.θ
    θ == 0 && return zero(typeof(θ))

    qᵛ = state.q
    qᵈ = one(qᵛ) - qᵛ

    Π = exner_function(state, ref, thermo)
    T₁ = Π * θ
    qˡ₁ = condensate_specific_humidity(T₁, state, ref, thermo)
    qˡ₁ <= 0 && return T₁

    r₁ = saturation_adjustment_residual(T₁, Π, qˡ₁, state, thermo)

    ℒ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    T₂ = (T₁ + sqrt(T₁^2 + 4 * ℒ * qˡ₁ / cᵖᵐ)) / 2
    qˡ₂ = condensate_specific_humidity(T₂, state, ref, thermo)
    r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)

    R = sqrt(max(T₂, T₁))
    ϵ = convert(typeof(T₂), 1e-4)
    δ = ϵ * R

    while abs(r₂ - r₁) > δ
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        r₁ = r₂
        T₁ = T₂

        T₂ -= r₂ * ΔTΔr
        qˡ₂ = condensate_specific_humidity(T₂, state, ref, thermo)
        r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)
    end

    return T₂
end

@inline function saturation_adjustment_residual(T, Π, qˡ, state, thermo)
    ℒᵛ = thermo.liquid.latent_heat
    qᵛ = state.q
    qᵈ = one(qᵛ) - qᵛ
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    return T^2 - ℒᵛ * qˡ / cᵖᵐ - Π * state.θ * T
end


@inline function condensate_specific_humidity(T, state::HeightReferenceThermodynamicState, ref, thermo)
    return condensate_specific_humidity(T, state.q, state.z, ref, thermo)
end

#####
##### Microphysics schemes
#####

# Solve
# θ = T/Π ( 1 - ℒ qˡ / (cᵖᵐ T))
# for temperature T with qˡ = max(0, q - qᵛ★).
# root of: f(T) = T² - Π θ T - ℒ qˡ / cᵖᵐ
@inline function temperature(state::ThermodynamicState{FT}, ref, thermo) where FT
    state.θ == 0 && return zero(FT)

    qᵛ = state.q
    qᵈ = one(qᵛ) - qᵛ

    # Generate guess for unsaturated conditions
    Π = exner_function(state, ref, thermo)
    T₁ = Π * state.θ
    qˡ₁ = condensate_specific_humidity(T₁, state, ref, thermo)
    qˡ₁ <= 0 && return T₁

    # If we made it this far, we have condensation
    r₁ = saturation_adjustment_residual(T₁, Π, qˡ₁, state, thermo)

    ℒ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    T₂ = (T₁ + sqrt(T₁^2 + 4 * ℒ * qˡ₁ / cᵖᵐ)) / 2
    qˡ₂ = condensate_specific_humidity(T₂, state, ref, thermo)
    r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)

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
        qˡ₂ = condensate_specific_humidity(T₂, state, ref, thermo)
        r₂ = saturation_adjustment_residual(T₂, Π, qˡ₂, state, thermo)
        iter += 1
    end

    return T₂
end

@inline function specific_volume(state, ref, thermo)
    T = temperature(state, ref, thermo)
    qᵛ = state.q
    qᵈ = one(qᵛ) - qᵛ
    Rᵐ = mixture_gas_constant(qᵈ, qᵛ, thermo)
    pᵣ = reference_pressure(state.z, ref, thermo)
    return Rᵐ * T / pᵣ
end

@inline specific_volume(state, ::Nothing, ref, thermo) = specific_volume(state, ref, thermo)
