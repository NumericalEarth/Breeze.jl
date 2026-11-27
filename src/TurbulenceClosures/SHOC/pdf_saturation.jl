function distribution_thermodynamics(model)
    return (constants = model.thermodynamic_constants,
            standard_pressure = AtmosphereModels.standard_pressure(model.dynamics),
            exner_exponent = AtmosphereModels.liquid_ice_exner_exponent(model.formulation),
            phase_equilibrium = condensate_phase_equilibrium(model.microphysics))
end

#####
##### The thermodynamics of the model
#####

@inline held_condensate(thermodynamics::NamedTuple{names}) where names = held_condensate(Val(:held ∈ names), thermodynamics)
@inline held_condensate(::Val{true}, thermodynamics) = thermodynamics.held
@inline held_condensate(::Val{false}, thermodynamics) = nothing

@inline component_moisture(thermodynamics, qᵗ) = moisture_with_held_condensate(held_condensate(thermodynamics), qᵗ)
@inline moisture_with_held_condensate(::Nothing, qᵗ) = MoistureMassFractions(qᵗ)
@inline moisture_with_held_condensate(held, qᵗ) =
    MoistureMassFractions(qᵗ, convert(typeof(qᵗ), held.liquid), convert(typeof(qᵗ), held.ice))

@inline function component_exner_function(thermodynamics, qᵗ, p)
    constants = thermodynamics.constants
    q = component_moisture(thermodynamics, qᵗ)
    κ = exner_exponent(thermodynamics.exner_exponent, q, constants)
    return (p / thermodynamics.standard_pressure)^κ
end

struct IceSurfaceEquilibrium end

ice_saturation_thermodynamics(thermodynamics) = merge(thermodynamics, (; phase_equilibrium = IceSurfaceEquilibrium()))

@inline saturation_surface(equilibrium, T) = equilibrated_surface(equilibrium, T)
@inline saturation_surface(::IceSurfaceEquilibrium, T) = PlanarIceSurface()

@inline condensate_liquid_fraction(::WarmPhaseEquilibrium, T) = (one(T), zero(T))
@inline condensate_liquid_fraction(::IceSurfaceEquilibrium, T) = (zero(T), zero(T))

@inline function condensate_liquid_fraction(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    λ = equilibrated_surface(equilibrium, T).liquid_fraction
    rising = (T > Tʰ) & (T < Tᶠ)
    return λ, ifelse(rising, 1 / (Tᶠ - Tʰ), zero(T))
end

@inline function condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    λ, ∂λ∂T = condensate_liquid_fraction(equilibrium, T)
    ℒˡ = liquid_latent_heat(T, constants)
    ℒⁱ = ice_latent_heat(T, constants)
    ℒ = λ * ℒˡ + (1 - λ) * ℒⁱ
    surface = saturation_surface(equilibrium, T)
    ∂ln_pᵛ⁺∂T = saturation_vapor_pressure_log_slope(T, constants, surface)
    ∂ln_pᵛ⁺∂λ = ifelse(∂λ∂T == 0, zero(T), saturation_vapor_pressure_liquid_fraction_log_slope(T, constants, surface))
    return ℒ, ∂ln_pᵛ⁺∂T + ∂λ∂T * ∂ln_pᵛ⁺∂λ
end

@inline function liquid_water_temperature_slope(thermodynamics, qᵗ, p)
    constants = thermodynamics.constants
    q = component_moisture(thermodynamics, qᵗ)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    return component_exner_function(thermodynamics, qᵗ, p) * (ĉᵖ / cᵖᵐ)
end

@inline liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p) = liquid_water_temperature_slope(thermodynamics, qᵗ, p) * θˡⁱ

@inline function component_saturation(thermodynamics, Tˡ, qᵗ, p)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    surface = saturation_surface(equilibrium, Tˡ)
    q = component_moisture(thermodynamics, qᵗ)
    Rᵐ = mixture_gas_constant(q, constants)
    ρ = p / (Rᵐ * Tˡ)
    qᵛ⁺ = saturation_specific_humidity(Tˡ, ρ, constants, surface)

    ℒ, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, Tˡ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    β = (ℒ / cᵖᵐ) * log_slope
    return qᵛ⁺, β, cᵖᵐ / ℒ, liquid_water_temperature_slope(thermodynamics, qᵗ, p)
end

@inline function buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qᶜ)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    q₀ = component_moisture(thermodynamics, qᵗ)
    ĉᵖ = vapor_equivalent_heat_capacity(q₀, constants)
    R̂ = vapor_equivalent_gas_constant(q₀, constants)
    ∂κ∂qᵗ = TurbulenceClosures.exner_exponent_moisture_derivative(thermodynamics.exner_exponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    Π = component_exner_function(thermodynamics, qᵗ, p)

    Tˡ = liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p)
    λ, _ = condensate_liquid_fraction(equilibrium, Tˡ)
    q = MoistureMassFractions(q₀.vapor - qᶜ, q₀.liquid + λ * qᶜ, q₀.ice + (1 - λ) * qᶜ)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ℒ₀ = λ * absolute_zero_latent_heat(constants, constants.liquid) + (1 - λ) * absolute_zero_latent_heat(constants, constants.ice)
    T = (ĉᵖ * Π * θˡⁱ + ℒ₀ * qᶜ) / cᵖᵐ
    ℒ = λ * liquid_latent_heat(T, constants) + (1 - λ) * ice_latent_heat(T, constants)

    θ̃ = cᵖᵐ * T / (ĉᵖ * Π)
    Aᵗ = θ̃ * ((Rᵛ - Rᵈ) / Rᵐ - (cᵖᵛ - cᵖᵈ) / cᵖᵐ) + θˡⁱ * ((cᵖᵛ - cᵖᵈ) / ĉᵖ + log(p / thermodynamics.standard_pressure) * ∂κ∂qᵗ)
    Aᶜ = ℒ / (ĉᵖ * Π) - θ̃ * Rᵛ / Rᵐ
    return Aᵗ, Aᶜ
end
