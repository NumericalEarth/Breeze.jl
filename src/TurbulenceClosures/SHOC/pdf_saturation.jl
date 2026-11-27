#####
##### The saturation thermodynamics of the assumed PDF
#####
##### The assumed PDF condenses each component where its total water exceeds saturation. It needs, per
##### component, the temperature `Tˡ` of the component with its condensate evaporated, the saturation
##### specific humidity `qᵛ⁺(Tˡ)`, the factor `β` of the linearized saturation excess
#####
#####     s = qᵗ - qᵛ⁺ (1 + β qᵗ) / (1 + β qᵛ⁺) = (qᵗ - qᵛ⁺) / (1 + β qᵛ⁺),    β qᵛ⁺ = (ℒ / ĉᵖ) ∂qᵛ⁺/∂T,
#####
##### the conversion `ĉᵖ / ℒ` and the Exner function `Π` with which the derivative of `qᵛ⁺` with
##### respect to `θˡⁱ` is `(ĉᵖ / ℒ) β qᵛ⁺ Π`, and the two coefficients of the buoyancy flux
#####
#####     w′θᵛ′ = w′θˡⁱ′ + A_q w′qᵗ′ + A_c w′qᶜ′.
#####
##### They are
#####
#####     liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p) -> Tˡ
#####     component_saturation(thermodynamics, Tˡ, qᵗ, p) -> (qᵛ⁺, β, ĉᵖ / ℒ, Π)
#####     buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qᶜ) -> (A_q, A_c)
#####
##### where `thermodynamics` is that of the model that holds the closure (`distribution_thermodynamics`).
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The thermodynamics with which the assumed PDF of the subgrid distribution of `model` condenses: the
`ThermodynamicConstants` of `model` (its saturation vapor pressure and temperature-dependent latent heats), the
standard pressure of its potential temperature, the Exner exponent of its formulation, and the phase equilibrium
of the condensate in the water it distributes (`condensate_phase_equilibrium`): `WarmPhaseEquilibrium()`, liquid
(warm microphysics, and the cloud liquid of P3 and of non-equilibrium cloud formation), or `MixedPhaseEquilibrium()`,
liquid turning to ice between the freezing and homogeneous freezing temperatures, for mixed-phase saturation
adjustment.

Each component condenses over the condensate surface that equilibrium makes at the component temperature, with the
vapor-equivalent heat capacity `ĉᵖ` and Exner function `Π = (p / pˢᵗ)^κ` with which the model defines the
liquid-ice potential temperature `θˡⁱ` that the closure mixes, and, in the buoyancy flux, the potential
temperature of the level in place of a fixed reference.

The saturation excess of a component is linearized about the component's own state. With the component
temperature `Tˡ = Π θˡⁱ` of the component's water all vapor, `ĉᵖ (T - Tˡ) = ℒ(T) qᶜ` exactly in
the model's thermodynamics, and `qᵛ⁺(T) ≈ qᵛ⁺(Tˡ) + (∂qᵛ⁺/∂T)(T - Tˡ)`, so

    qᶜ = (qᵗ - qᵛ⁺(Tˡ)) / (1 + β qᵛ⁺),    β qᵛ⁺ = (ℒ / ĉᵖ) ∂qᵛ⁺/∂T,
    qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ Tˡ),  ρ = p / (Rᵐ Tˡ),    ∂ ln qᵛ⁺ / ∂T = ∂ ln pᵛ⁺ / ∂T,

where `ℒ` is the latent heat of the condensate the equilibrium forms (`λ ℒˡ + (1 - λ) ℒⁱ` for a
liquid fraction `λ`), `Rᵐ` the gas constant of the component's air with its water all vapor (the model's
saturation specific humidity, at fixed pressure and composition) and `∂ ln pᵛ⁺/∂T` includes the change
of `λ` with temperature. The derivative
of `qᵛ⁺` with respect to `θˡⁱ` is `(∂qᵛ⁺/∂T) Π`, and the buoyancy flux follows from linearizing
`θᵛ = θ (1 + (1/ε - 1) qᵛ - qᶜ)` with `θ = θˡⁱ + ℒ qᶜ / (ĉᵖ Π)` about the level mean `θ̄`:

    A_q = ((1 - ε) / ε) θ̄,    A_c = ℒ / (ĉᵖ Π) - θ̄ / ε,    ε = Rᵈ / Rᵛ.

The water of the components is the total water the distribution reads: the moisture of the model and its cloud
liquid (all the water with saturation adjustment; the vapor and the cloud liquid with P3, whose rain and ice
are not distributed).
"""
function distribution_thermodynamics(model)
    return (constants = model.thermodynamic_constants,
            standard_pressure = AtmosphereModels.standard_pressure(model.dynamics),
            exner_exponent = AtmosphereModels.liquid_ice_exner_exponent(model.formulation),
            phase_equilibrium = condensate_phase_equilibrium(model.microphysics))
end

#####
##### The thermodynamics of the model
#####

# The moisture fractions of a component: its distributed water `qᵗ` as vapor (the all-vapor state of the distribution),
# with the condensate the distribution does not distribute (`held`, the rain and ice of P3 at the level) held as
# condensate, so that its heat capacity, gas constant and Exner exponent are those of the model's total water
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

@inline component_heat_capacity(thermodynamics, qᵗ) = vapor_equivalent_heat_capacity(component_moisture(thermodynamics, qᵗ), thermodynamics.constants)

# Saturation over ice at every temperature: the surface of the ice outside the distributed water (P3's ice), for
# the saturation excess of the distributed water over ice (`ice_saturation_thermodynamics`)
struct IceSurfaceEquilibrium end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The `thermodynamics` of the distribution (`distribution_thermodynamics`) with saturation over ice at every
temperature and the latent heat of sublimation: those with which the saturation excess of the distributed water over
ice, ``sⁱ = qʷ - qᵛ⁺ⁱ``, is linearized, for the ice that condenses outside the distributed water.
"""
ice_saturation_thermodynamics(thermodynamics) = merge(thermodynamics, (; phase_equilibrium = IceSurfaceEquilibrium()))

@inline saturation_surface(equilibrium, T) = equilibrated_surface(equilibrium, T)
@inline saturation_surface(::IceSurfaceEquilibrium, T) = PlanarIceSurface()

# The liquid fraction of the condensate, and its derivative with respect to temperature
@inline condensate_liquid_fraction(::WarmPhaseEquilibrium, T) = (one(T), zero(T))
@inline condensate_liquid_fraction(::IceSurfaceEquilibrium, T) = (zero(T), zero(T))

@inline function condensate_liquid_fraction(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    λ = (clamp(T, Tʰ, Tᶠ) - Tʰ) / (Tᶠ - Tʰ)
    ramp = (T > Tʰ) & (T < Tᶠ)
    return λ, ifelse(ramp, 1 / (Tᶠ - Tʰ), zero(T))
end

# The latent heat of the condensate the equilibrium forms at T, and ∂ ln pᵛ⁺ / ∂T. On a surface of
# fixed liquid fraction ln pᵛ⁺ is linear in the fraction, so the change of the fraction with
# temperature adds (∂λ/∂T) ln(pᵛ⁺ˡ / pᵛ⁺ⁱ).
@inline function condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    λ, ∂λ∂T = condensate_liquid_fraction(equilibrium, T)
    Rᵛ = vapor_gas_constant(constants)
    ℒˡ = liquid_latent_heat(T, constants)
    ℒⁱ = ice_latent_heat(T, constants)
    ℒ = λ * ℒˡ + (1 - λ) * ℒⁱ
    pᵛ⁺ˡ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    pᵛ⁺ⁱ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
    log_ratio = ifelse(∂λ∂T == 0, zero(T), log(pᵛ⁺ˡ / pᵛ⁺ⁱ))
    return ℒ, ℒ / (Rᵛ * T * T) + ∂λ∂T * log_ratio
end

@inline function liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p)
    return component_exner_function(thermodynamics, qᵗ, p) * θˡⁱ
end

@inline function component_saturation(thermodynamics, Tˡ, qᵗ, p)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    surface = saturation_surface(equilibrium, Tˡ)
    # The saturation specific humidity of the model (the one the microphysics uses),
    # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T), at the density ρ = p / (Rᵐ T) of the component's air with its total
    # water as vapor
    Rᵐ = mixture_gas_constant(component_moisture(thermodynamics, qᵗ), constants)
    ρ = p / (Rᵐ * Tˡ)
    qᵛ⁺ = saturation_specific_humidity(Tˡ, ρ, constants, surface)

    ℒ, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, Tˡ)
    ĉᵖ = component_heat_capacity(thermodynamics, qᵗ)
    # β qᵛ⁺ = (ℒ / ĉᵖ) ∂qᵛ⁺/∂T at fixed p and composition: ∂ ln qᵛ⁺ / ∂T = ∂ ln pᵛ⁺ / ∂T
    β = (ℒ / ĉᵖ) * log_slope
    return qᵛ⁺, β, ĉᵖ / ℒ, component_exner_function(thermodynamics, qᵗ, p)
end

@inline function buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qˡ)
    constants = thermodynamics.constants
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    Π = component_exner_function(thermodynamics, qᵗ, p)
    Tˡ = Π * θˡⁱ
    ℒ, _ = condensate_latent_heat_and_log_slope(thermodynamics.phase_equilibrium, constants, Tˡ)
    ĉᵖ = component_heat_capacity(thermodynamics, qᵗ)
    heating = ℒ / (ĉᵖ * Π)
    θ = θˡⁱ + heating * qˡ
    return ((1 - ε) / ε) * θ, heating - θ / ε
end
