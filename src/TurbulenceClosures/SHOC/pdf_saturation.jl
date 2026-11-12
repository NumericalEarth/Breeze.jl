#####
##### Slot: the saturation thermodynamics of the assumed PDF
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
##### A saturation component implements
#####
#####     plume_liquid_water_temperature(saturation, thermodynamics, c, θ_l, q_w, p) -> T_l
#####     plume_saturation(saturation, thermodynamics, c, T_l, q_w, p) -> (q_s, β, c_p / ℒ, Π)
#####     buoyancy_flux_coefficients(saturation, thermodynamics, c, p, θ_l, q_w, q_c) -> (A_q, A_c)
#####
##### where `c` are the `SHOCConstants` and `thermodynamics` is what the model hands the closure
##### (`nothing` outside a model).
#####

"""
$(TYPEDEF)

The saturation thermodynamics of E3SM `shoc.F90` (`shoc_assumed_pdf_compute_temperature`,
`shoc_assumed_pdf_compute_qs`, `shoc_assumed_pdf_compute_s` and the buoyancy flux of
`shoc_assumed_pdf`): saturation over liquid only, with the vapor pressure of Murphy and Koop
(2005) and `q_s = 0.622 e_s / max(e_s, p - e_s)`; the fixed latent heat, heat capacity and gas
constants of [`SHOCConstants`](@ref); the Exner function `(p / basepres)^(rgas / cp)`; and the
reference temperature `basetemp` in the buoyancy flux,

    β = (rgas / rv) (lcond / (rgas T_l)) (lcond / (cp T_l)),
    A_q = ((1 - ε) / ε) basetemp,    A_c = (lcond / cp) / Π - basetemp / ε,    ε = rgas / rv.

This is the default, and the saturation of the original configuration.
"""
struct FixedConstantsLiquidSaturation <: AbstractSHOCComponent end

"""
$(TYPEDEF)

The saturation thermodynamics of the model that holds the closure: its saturation vapor pressure
(of its `ThermodynamicConstants`), over the condensate surface its `phase_equilibrium` makes at
the plume temperature (`WarmPhaseEquilibrium()`, liquid, the cloud of P3; or
`MixedPhaseEquilibrium()`, liquid turning to ice between its freezing and homogeneous nucleation
temperatures); its temperature-dependent latent heats; the vapor-equivalent heat capacity `ĉᵖ`
and Exner function `Π = (p / pˢᵗ)^κ` with which it defines the liquid-ice potential temperature
`θˡⁱ` that SHOC mixes; and, in the buoyancy flux, the potential temperature of the level in
place of a fixed reference.

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

The plume is taken free of rain and ice, as in `shoc.F90`. The model supplies its thermodynamic
constants, the standard pressure and the Exner exponent of its formulation; outside a model
(`thermodynamics = nothing`) this saturation cannot be used.

$(TYPEDFIELDS)
"""
struct ModelThermodynamicsSaturation{E} <: AbstractSHOCComponent
    "The condensate surface: `WarmPhaseEquilibrium()` or `MixedPhaseEquilibrium()`"
    phase_equilibrium :: E
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`ModelThermodynamicsSaturation`](@ref) that condenses on `phase_equilibrium`
(liquid, `WarmPhaseEquilibrium()`, by default).
"""
ModelThermodynamicsSaturation(; phase_equilibrium = WarmPhaseEquilibrium()) =
    ModelThermodynamicsSaturation(phase_equilibrium)

Adapt.adapt_structure(to, s::ModelThermodynamicsSaturation) =
    ModelThermodynamicsSaturation(adapt(to, s.phase_equilibrium))

type_parameter_summary(s::ModelThermodynamicsSaturation) = string("{", nameof(typeof(s.phase_equilibrium)), "}")


"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The thermodynamics of `model` that [`ModelThermodynamicsSaturation`](@ref) uses: its
thermodynamic constants, the standard pressure of its potential temperature and the Exner
exponent of its formulation.
"""
function shoc_saturation_thermodynamics(model)
    return (constants = model.thermodynamic_constants,
            standard_pressure = AtmosphereModels.standard_pressure(model.dynamics),
            exner_exponent = AtmosphereModels.liquid_ice_exner_exponent(model.formulation))
end

#####
##### The fixed constants of shoc.F90
#####

@inline function plume_liquid_water_temperature(::FixedConstantsLiquidSaturation, thermodynamics, c, thl, qw, pval)
    exner_inverse = (c.basepres / pval)^(c.rgas / c.cp)
    return thl / exner_inverse
end

# shoc_assumed_pdf_compute_qs for one plume
@inline function plume_saturation(::FixedConstantsLiquidSaturation, thermodynamics, c, Tl, qw, pval)
    FT = typeof(Tl)
    esval = murphy_koop_liquid_svp(Tl)
    qs = convert(FT, 0.622) * esval / max(esval, pval - esval)
    beta = (c.rgas / c.rv) * (c.lcond / (c.rgas * Tl)) * (c.lcond / (c.cp * Tl))
    return qs, beta, c.cp / c.lcond, (pval / c.basepres)^(c.rgas / c.cp)
end

@inline function buoyancy_flux_coefficients(::FixedConstantsLiquidSaturation, thermodynamics, c, pval, thl, qw, ql)
    epsterm = c.rgas / c.rv
    A_q = ((1 - epsterm) / epsterm) * c.basetemp
    A_c = (c.lcond / c.cp) * (c.basepres / pval)^(c.rgas / c.cp) - (1 / epsterm) * c.basetemp
    return A_q, A_c
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

@inline function plume_liquid_water_temperature(::ModelThermodynamicsSaturation, thermodynamics, c, thl, qw, pval)
    return plume_exner_function(thermodynamics, qw, pval) * thl
end

@inline function plume_saturation(s::ModelThermodynamicsSaturation, thermodynamics, c, Tl, qw, pval)
    constants = thermodynamics.constants
    surface = equilibrated_surface(s.phase_equilibrium, Tl)
    esval = saturation_vapor_pressure(Tl, constants, surface)
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    dry_pressure = max(esval, pval - esval)
    qs = ε * (1 - qw) * esval / dry_pressure

    ℒ, log_slope = condensate_latent_heat_and_log_slope(s.phase_equilibrium, constants, Tl)
    cᵖ = plume_heat_capacity(constants, qw)
    # β q_s = (ℒ / ĉᵖ) ∂q_s/∂T
    beta = (ℒ / cᵖ) * (pval / dry_pressure) * log_slope
    return qs, beta, cᵖ / ℒ, plume_exner_function(thermodynamics, qw, pval)
end

@inline function buoyancy_flux_coefficients(s::ModelThermodynamicsSaturation, thermodynamics, c, pval, thl, qw, ql)
    constants = thermodynamics.constants
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    Π = plume_exner_function(thermodynamics, qw, pval)
    Tl = Π * thl
    ℒ, _ = condensate_latent_heat_and_log_slope(s.phase_equilibrium, constants, Tl)
    cᵖ = plume_heat_capacity(constants, qw)
    heating = ℒ / (cᵖ * Π)
    θ = thl + heating * ql
    return ((1 - ε) / ε) * θ, heating - θ / ε
end
