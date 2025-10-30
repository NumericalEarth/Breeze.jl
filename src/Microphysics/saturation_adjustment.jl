using ..Thermodynamics: total_specific_humidity, saturation_specific_humidity

struct SaturationAdjustmentMicrophysics end

#####
##### Microphysics saturation adjustment utilities
#####

# Solve θ = T/Π (1 - ℒ qˡ / (cᵖᵐ T)) for temperature T with qˡ = max(0, q - qᵛ⁺)
# by iterating on the root of f(T) = T² - Π θ T - ℒ qˡ / cᵖᵐ.
@inline function adjust_temperature_and_humidities(𝒰, thermo)
    θ = 𝒰.potential_temperature
    θ == 0 && return zero(θ), 𝒰.humidities

    # qᵈ = dry_air_mass_fraction(𝒰.humidities))
    q = 𝒰.humidities
    qᵗ = total_specific_humidity(q)
    z = 𝒰.height
    ref = 𝒰.reference_state
    Π = exner_function(𝒰.humidities, 𝒰.height, 𝒰.reference_state, thermo)
    T₁ = Π * θ
    qˡ₁ = adjusted_condensate_specific_humidity(T₁, qᵗ, z, ref, thermo)

    if qˡ₁ <= 0
        qᵛ = total_specific_humidity(𝒰.humidities)
        qˡ = zero(qᵛ)
        qˢ = zero(qᵛ)
        q = SpecificHumidities(qᵛ, qˡ, qˢ)
        return T₁, q
    end

    qᵛ₁ = qᵗ - qˡ₁
    q₁ = SpecificHumidities(qᵛ₁, qˡ₁, zero(qᵗ))
    r₁ = saturation_adjustment_residual(T₁, Π, q₁, θ, thermo)

    ℒ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    T₂ = (T₁ + sqrt(T₁^2 + 4 * ℒ * qˡ₁ / cᵖᵐ)) / 2
    qˡ₂ = adjusted_condensate_specific_humidity(T₂, qᵗ, z, ref, thermo)
    qᵛ₂ = qᵗ - qˡ₂
    q₂ = SpecificHumidities(qᵛ₂, qˡ₂, zero(qᵗ))
    r₂ = saturation_adjustment_residual(T₂, Π, q₂, θ, thermo)

    R = sqrt(max(T₂, T₁))
    ϵ = convert(typeof(T₂), 1e-4)
    δ = ϵ * R

    while abs(r₂ - r₁) > δ
        ΔTΔr = (T₂ - T₁) / (r₂ - r₁)

        r₁ = r₂
        T₁ = T₂

        T₂ -= r₂ * ΔTΔr
        qˡ₂ = adjusted_condensate_specific_humidity(T₂, qᵗ, z, ref, thermo)
        q₂ = SpecificHumidities(qᵛ₂, qˡ₂, zero(qᵗ))
        r₂ = saturation_adjustment_residual(T₂, Π, q₂, θ, thermo)
    end

    qᵗ = total_specific_humidity(𝒰.humidities)
    qᵛ = qᵗ - qˡ₂
    qˢ = zero(qˡ₂)
    adjusted_q = SpecificHumidities(qᵛ, qˡ₂, qˢ)

    return T₂, adjusted_q
end

function adjusted_condensate_specific_humidity(T, qᵗ, z, ref::ReferenceState, thermo)
    qᵛ⁺ = saturation_specific_humidity(T, z, ref, thermo, thermo.liquid)
    return max(0, qᵗ - qᵛ⁺)
end

function adjusted_ice_specific_humidity(T, qᵗ, z, ref::ReferenceState, thermo)
    qˢ⁺ = saturation_specific_humidity(T, z, ref, thermo, thermo.solid)
    return max(0, qᵗ - qˢ⁺)
end

@inline function saturation_adjustment_residual(T, Π, q, θ, thermo)
    ℒᵛ = thermo.liquid.latent_heat
    cᵖᵐ = mixture_heat_capacity(q, thermo)
    qˡ = q.liquid
    return T^2 - ℒᵛ * qˡ / cᵖᵐ - Π * θ * T
end


#####
##### Microphysics schemes
#####

#=
# Solve
# θ = T/Π ( 1 - ℒ qˡ / (cᵖᵐ T))
# for temperature T with qˡ = max(0, q - qᵛ⁺).
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
=#
