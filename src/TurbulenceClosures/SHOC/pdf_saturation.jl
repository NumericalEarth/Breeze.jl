#####
##### The saturation thermodynamics of the assumed PDF
#####
##### The assumed PDF condenses each component where its total water exceeds saturation. It needs, per
##### component, the liquid-water temperature `Tˡ` of the component (its temperature with its distributed water all
##### vapor), the saturation specific humidity `qᵛ⁺(Tˡ)`, the factor `β` of the linearized saturation excess
#####
#####     s = qᵗ - qᵛ⁺ (1 + β qᵗ) / (1 + β qᵛ⁺) = (qᵗ - qᵛ⁺) / (1 + β qᵛ⁺),    β qᵛ⁺ = (ℒ / cᵖᵐ) ∂qᵛ⁺/∂T,
#####
##### the conversion `cᵖᵐ / ℒ` and the derivative `∂Tˡ/∂θˡⁱ` with which the derivative of `qᵛ⁺` with
##### respect to `θˡⁱ` is `(cᵖᵐ / ℒ) β qᵛ⁺ ∂Tˡ/∂θˡⁱ`, and the two coefficients of the buoyancy flux
#####
#####     w′θᵛ′ = w′θˡⁱ′ + Aᵗ w′qᵗ′ + Aᶜ w′qᶜ′.
#####
##### They are
#####
#####     liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p) -> Tˡ
#####     component_saturation(thermodynamics, Tˡ, qᵗ, p) -> (qᵛ⁺, β, cᵖᵐ / ℒ, ∂Tˡ/∂θˡⁱ)
#####     buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qᶜ) -> (Aᵗ, Aᶜ)
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
heat capacities, the gas constants and the Exner function `Π = (p / pˢᵗ)^κ` with which the model defines the
liquid-ice potential temperature `θˡⁱ` that the closure mixes, `ĉᵖ Π θˡⁱ = cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ`.

The saturation excess of a component is linearized about the component's own state. With the liquid-water
temperature `Tˡ = (ĉᵖ / cᵖᵐ) Π θˡⁱ` of the component, its temperature with its distributed water all vapor (`cᵖᵐ` the
heat capacity of that state: `ĉᵖ`, and `Tˡ = Π θˡⁱ`, unless condensate outside the distribution is held),
`cᵖᵐ (T - Tˡ) = ℒ(T) qᶜ` exactly in the model's thermodynamics, and `qᵛ⁺(T) ≈ qᵛ⁺(Tˡ) + (∂qᵛ⁺/∂T)(T - Tˡ)`, so

    qᶜ = (qᵗ - qᵛ⁺(Tˡ)) / (1 + β qᵛ⁺),    β qᵛ⁺ = (ℒ / cᵖᵐ) ∂qᵛ⁺/∂T,
    qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ Tˡ),  ρ = p / (Rᵐ Tˡ),    ∂ ln qᵛ⁺ / ∂T = ∂ ln pᵛ⁺ / ∂T,

where `ℒ` is the latent heat of the condensate the equilibrium forms (`λ ℒˡ + (1 - λ) ℒⁱ` for a
liquid fraction `λ`), `Rᵐ` the gas constant of the component's air with its water all vapor (the model's
saturation specific humidity, at fixed pressure and composition) and `∂ ln pᵛ⁺/∂T` includes the change
of `λ` with temperature. The derivative of `qᵛ⁺` with respect to `θˡⁱ` is `(∂qᵛ⁺/∂T) ∂Tˡ/∂θˡⁱ`.

The buoyancy flux is that of the model's buoyancy, a function of `Rᵐ T` at fixed pressure, linearized about the
mean state of the level (its `θˡⁱ`, its water and its mean condensate `qᶜ`) and normalized by its derivative with
respect to `θˡⁱ` (`buoyancy_flux_coefficients`):

    Aᵗ = (∂ ln(Rᵐ T)/∂qᵗ) / (∂ ln(Rᵐ T)/∂θˡⁱ),    Aᶜ = (∂ ln(Rᵐ T)/∂qᶜ) / (∂ ln(Rᵐ T)/∂θˡⁱ),    cᵖᵐ T = ĉᵖ Π θˡⁱ + ℒ₀ qᶜ,

so that `w′b′ = (∂b/∂θˡⁱ) w′θᵛ′`.

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

# The liquid fraction of the model's mixed-phase surface (`equilibrated_surface`), which rises linearly from the
# homogeneous ice nucleation temperature to the freezing temperature, and the slope of that rise
@inline function condensate_liquid_fraction(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    λ = equilibrated_surface(equilibrium, T).liquid_fraction
    rising = (T > Tʰ) & (T < Tᶠ)
    return λ, ifelse(rising, 1 / (Tᶠ - Tʰ), zero(T))
end

# The latent heat of the condensate the equilibrium forms at T, and ∂ ln pᵛ⁺ / ∂T of the model's saturation vapor
# pressure over its surface (`saturation_vapor_pressure_log_slope`); the change of the liquid fraction with temperature
# adds (∂λ/∂T) ∂ ln pᵛ⁺ / ∂λ (`saturation_vapor_pressure_liquid_fraction_log_slope`).
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

# ∂Tˡ/∂θˡⁱ = (ĉᵖ / cᵖᵐ) Π of the model's `ĉᵖ Π θˡⁱ = cᵖᵐ T - ℒˡ₀ qˡ - ℒⁱ₀ qⁱ`, in the state of the component with its
# distributed water all vapor. Its heat capacity `cᵖᵐ` is `ĉᵖ` to the last bit, and the derivative `Π`, unless
# condensate outside the distribution is held.
@inline function liquid_water_temperature_slope(thermodynamics, qᵗ, p)
    constants = thermodynamics.constants
    q = component_moisture(thermodynamics, qᵗ)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    return component_exner_function(thermodynamics, qᵗ, p) * (ĉᵖ / cᵖᵐ)
end

# The liquid-water temperature of a component: its temperature with its distributed water all vapor
@inline liquid_water_temperature(thermodynamics, θˡⁱ, qᵗ, p) = liquid_water_temperature_slope(thermodynamics, qᵗ, p) * θˡⁱ

@inline function component_saturation(thermodynamics, Tˡ, qᵗ, p)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    surface = saturation_surface(equilibrium, Tˡ)
    q = component_moisture(thermodynamics, qᵗ)
    # The saturation specific humidity of the model (the one the microphysics uses),
    # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T), at the density ρ = p / (Rᵐ T) of the component's air with its total
    # water as vapor
    Rᵐ = mixture_gas_constant(q, constants)
    ρ = p / (Rᵐ * Tˡ)
    qᵛ⁺ = saturation_specific_humidity(Tˡ, ρ, constants, surface)

    ℒ, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, Tˡ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    # β qᵛ⁺ = (ℒ / cᵖᵐ) ∂qᵛ⁺/∂T at fixed p and composition: ∂ ln qᵛ⁺ / ∂T = ∂ ln pᵛ⁺ / ∂T
    β = (ℒ / cᵖᵐ) * log_slope
    return qᵛ⁺, β, cᵖᵐ / ℒ, liquid_water_temperature_slope(thermodynamics, qᵗ, p)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The coefficients `(Aᵗ, Aᶜ)` of the buoyancy flux `w′θᵛ′ = w′θˡⁱ′ + Aᵗ w′qᵗ′ + Aᶜ w′qᶜ′` of a level with
liquid-ice potential temperature `θˡⁱ`, distributed water `qᵗ` and mean condensate `qᶜ` at pressure `p`: the
derivatives of the model's `ln(Rᵐ T)` (its buoyancy is a function of `Rᵐ T` at fixed pressure) with respect to the
water and to the condensate, over its derivative with respect to `θˡⁱ`. With `cᵖᵐ T = ĉᵖ Π θˡⁱ + ℒ₀ qᶜ`, the gas
constant `Rᵐ` and the heat capacity `cᵖᵐ` of the level with its condensate, and `θ̃ = cᵖᵐ T / (ĉᵖ Π)`,

    Aᵗ = θ̃ ((Rᵛ - Rᵈ) / Rᵐ - (cᵖᵛ - cᵖᵈ) / cᵖᵐ) + θˡⁱ ((cᵖᵛ - cᵖᵈ) / ĉᵖ + ln(p / pˢᵗ) ∂κ/∂qᵗ),
    Aᶜ = ℒ(T) / (ĉᵖ Π) - θ̃ Rᵛ / Rᵐ,

with `ℒ` the latent heat of the condensate of the phase equilibrium. In dry air without condensate these are
`θˡⁱ (Rᵛ - Rᵈ) / Rᵈ` and `ℒ / (cᵖᵈ Π) - θˡⁱ Rᵛ / Rᵈ`.
"""
@inline function buoyancy_flux_coefficients(thermodynamics, p, θˡⁱ, qᵗ, qᶜ)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    # The level with its distributed water all vapor: ĉᵖ, Π and their dependence on the water
    q₀ = component_moisture(thermodynamics, qᵗ)
    ĉᵖ = vapor_equivalent_heat_capacity(q₀, constants)
    R̂ = vapor_equivalent_gas_constant(q₀, constants)
    ∂κ∂qᵗ = TurbulenceClosures.exner_exponent_moisture_derivative(thermodynamics.exner_exponent, R̂, ĉᵖ, Rᵈ, Rᵛ, cᵖᵈ, cᵖᵛ)
    Π = component_exner_function(thermodynamics, qᵗ, p)

    # The level with its mean condensate, on the phase the equilibrium forms at the liquid-water temperature
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
