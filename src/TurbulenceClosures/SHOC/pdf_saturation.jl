#####
##### The saturation thermodynamics of the assumed PDF
#####
##### The assumed PDF condenses each plume where its total water exceeds saturation. It needs, per
##### plume, the temperature `T_l` of the plume with its condensate evaporated, the saturation
##### specific humidity `q_s(T_l)`, the factor `β` of the linearized saturation excess
#####
#####     s = q_w - q_s (1 + β q_w) / (1 + β q_s) = (q_w - q_s) / (1 + β q_s),    β q_s = (ℒ / c_p) ∂q_s/∂T,
#####
##### the conversion `c_p / ℒ` and the Exner function `Π` with which the derivative of `q_s` with
##### respect to `θ_l` is `(c_p / ℒ) β q_s Π`, and the two coefficients of the buoyancy flux
#####
#####     w'θ_v' = w'θ_l' + A_q w'q_w' + A_c w'q_c'.
#####
##### They are
#####
#####     plume_liquid_water_temperature(thermodynamics, θ_l, q_w, p) -> T_l
#####     plume_saturation(thermodynamics, T_l, q_w, p) -> (q_s, β, c_p / ℒ, Π)
#####     buoyancy_flux_coefficients(thermodynamics, p, θ_l, q_w, q_c) -> (A_q, A_c)
#####
##### where `thermodynamics` is that of the model that holds the closure (`distribution_thermodynamics`).
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The thermodynamics with which the assumed PDF of the subgrid distribution of `model` condenses: the
`ThermodynamicConstants` of `model` (its saturation vapor pressure and temperature-dependent latent heats), the
standard pressure of its potential temperature, the Exner exponent of its formulation, and the phase equilibrium
of the condensate of its microphysics (`condensate_phase_equilibrium`): `WarmPhaseEquilibrium()`, liquid, or
`MixedPhaseEquilibrium()`, liquid turning to ice between the freezing and homogeneous freezing temperatures of the
microphysics, for microphysics with ice (P3, mixed-phase saturation adjustment).

Each plume condenses over the condensate surface that equilibrium makes at the plume temperature, with the
vapor-equivalent heat capacity `ĉᵖ` and Exner function `Π = (p / pˢᵗ)^κ` with which the model defines the
liquid-ice potential temperature `θˡⁱ` that the closure mixes, and, in the buoyancy flux, the potential
temperature of the level in place of a fixed reference.

The saturation excess of a plume is linearized about the plume's own state. With the plume
temperature `T_l = Π θˡⁱ` of the plume's water all vapor, `ĉᵖ (T - T_l) = ℒ(T) q_c` exactly in
the model's thermodynamics, and `q_s(T) ≈ q_s(T_l) + (∂q_s/∂T)(T - T_l)`, so

    q_c = (q_w - q_s(T_l)) / (1 + β q_s),    β q_s = (ℒ / ĉᵖ) ∂q_s/∂T,
    q_s = ε (1 - q_w) e_s / (p - e_s),       ∂ ln q_s / ∂T = (p / (p - e_s)) ∂ ln e_s / ∂T,

where `ℒ` is the latent heat of the condensate the equilibrium forms (`λ ℒˡ + (1 - λ) ℒⁱ` for a
liquid fraction `λ`) and `∂ ln e_s/∂T` includes the change of `λ` with temperature. The derivative
of `q_s` with respect to `θˡⁱ` is `(∂q_s/∂T) Π`, and the buoyancy flux follows from linearizing
`θ_v = θ (1 + (1/ε - 1) q_v - q_c)` with `θ = θˡⁱ + ℒ q_c / (ĉᵖ Π)` about the level mean `θ̄`:

    A_q = ((1 - ε) / ε) θ̄,    A_c = ℒ / (ĉᵖ Π) - θ̄ / ε,    ε = Rᵈ / Rᵛ.

The water of the plumes is the total water the distribution reads: the moisture of the model and its cloud
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

@inline function plume_exner_function(thermodynamics, qw, pval)
    constants = thermodynamics.constants
    q = MoistureMassFractions(qw)
    κ = exner_exponent(thermodynamics.exner_exponent, q, constants)
    return (pval / thermodynamics.standard_pressure)^κ
end

@inline plume_heat_capacity(constants, qw) = vapor_equivalent_heat_capacity(MoistureMassFractions(qw), constants)

# The liquid fraction of the condensate, and its derivative with respect to temperature
@inline condensate_liquid_fraction(::WarmPhaseEquilibrium, T) = (one(T), zero(T))

@inline function condensate_liquid_fraction(equilibrium::MixedPhaseEquilibrium, T)
    Tᶠ = equilibrium.freezing_temperature
    Tʰ = equilibrium.homogeneous_ice_nucleation_temperature
    λ = (clamp(T, Tʰ, Tᶠ) - Tʰ) / (Tᶠ - Tʰ)
    ramp = (T > Tʰ) & (T < Tᶠ)
    return λ, ifelse(ramp, 1 / (Tᶠ - Tʰ), zero(T))
end

# The latent heat of the condensate the equilibrium forms at T, and ∂ ln e_s / ∂T. On a surface of
# fixed liquid fraction ln e_s is linear in the fraction, so the change of the fraction with
# temperature adds (∂λ/∂T) ln(e_s,liquid / e_s,ice).
@inline function condensate_latent_heat_and_log_slope(equilibrium, constants, T)
    λ, dλdT = condensate_liquid_fraction(equilibrium, T)
    Rᵛ = vapor_gas_constant(constants)
    ℒˡ = liquid_latent_heat(T, constants)
    ℒⁱ = ice_latent_heat(T, constants)
    ℒ = λ * ℒˡ + (1 - λ) * ℒⁱ
    eˡ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    eⁱ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
    log_ratio = ifelse(dλdT == 0, zero(T), log(eˡ / eⁱ))
    return ℒ, ℒ / (Rᵛ * T * T) + dλdT * log_ratio
end

@inline function plume_liquid_water_temperature(thermodynamics, thl, qw, pval)
    return plume_exner_function(thermodynamics, qw, pval) * thl
end

@inline function plume_saturation(thermodynamics, Tl, qw, pval)
    constants = thermodynamics.constants
    equilibrium = thermodynamics.phase_equilibrium
    surface = equilibrated_surface(equilibrium, Tl)
    # The saturation specific humidity of the model (the one the microphysics uses),
    # qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T), at the density ρ = p / (Rᵐ T) of the component's air with its total
    # water as vapor
    Rᵐ = mixture_gas_constant(MoistureMassFractions(qw), constants)
    ρ = pval / (Rᵐ * Tl)
    qᵛ⁺ = saturation_specific_humidity(Tl, ρ, constants, surface)

    ℒ, log_slope = condensate_latent_heat_and_log_slope(equilibrium, constants, Tl)
    cᵖ = plume_heat_capacity(constants, qw)
    # β q_s = (ℒ / ĉᵖ) ∂q_s/∂T at fixed p and composition: ∂ ln q_s / ∂T = ∂ ln e_s / ∂T
    beta = (ℒ / cᵖ) * log_slope
    return qᵛ⁺, beta, cᵖ / ℒ, plume_exner_function(thermodynamics, qw, pval)
end

@inline function buoyancy_flux_coefficients(thermodynamics, pval, thl, qw, ql)
    constants = thermodynamics.constants
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    Π = plume_exner_function(thermodynamics, qw, pval)
    Tl = Π * thl
    ℒ, _ = condensate_latent_heat_and_log_slope(thermodynamics.phase_equilibrium, constants, Tl)
    cᵖ = plume_heat_capacity(constants, qw)
    heating = ℒ / (cᵖ * Π)
    θ = thl + heating * ql
    return ((1 - ε) / ε) * θ, heating - θ / ε
end
