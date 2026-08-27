#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

#####
##### Coupled condensation/deposition saturation adjustment
#####

struct P3CoupledVaporRates{FT}
    condensation :: FT
    rain_evaporation :: FT
    rain_condensation :: FT
    deposition :: FT
    coating_condensation :: FT
    coating_evaporation :: FT
end

"""
$(TYPEDSIGNATURES)

Bounded Grabowski–Morrison saturation adjustment applied before the
Morrison–Gettelman semi-analytic rates. It aligns `qᵛ`, `qᶜˡ`, `T`, `qᵛ⁺ˡ`, and
`qᵛ⁺ⁱ` with each other before the per-species rate equations are evaluated.

Given the advected liquid supersaturation ``sᵛ⁺ˡ``, the diagnostic local
``qᵛ - qᵛ⁺ˡ``, and the liquid-side psychrometric factor
``ξˡ = 1 + ℒˡ² qᵛ⁺ˡ / (cᵖᵈ Rᵛ T²)``, compute the cloud-water increment

```math
ε = (qᵛ - qᵛ⁺ˡ - sᵛ⁺ˡ) / ξˡ
```

clamped to physical limits: ``ε`` cannot evaporate more cloud than is locally
available (``ε ≥ -qᶜˡ``), and when the advected ``sᵛ⁺ˡ`` is negative
``ε ≤ 0`` (no spurious condensation). The returned ``rate = ε / τ`` is
sized to `sink_limiting_timescale`, so one host step with
``dt = sink_limiting_timescale`` reproduces the one-shot ``ε`` exactly. If
the host integrates with ``dt ≠ τ`` the supersaturation alignment relaxes over multiple
steps rather than landing in one.

When `predict_supersaturation = false`, dispatch bypasses the adjustment and the
local state passes through unchanged.
"""
@inline predicted_supersaturation_adjustment(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, sᵛ⁺ˡ, T, ρ, constants) =
    predicted_supersaturation_adjustment(p3.process_rates, p3, qᶜˡ, qᵛ, qᵛ⁺ˡ,
                                         sᵛ⁺ˡ, T, ρ, constants)

@inline function predicted_supersaturation_adjustment(
    ::ProcessRateParameters{FT, false}, p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, sᵛ⁺ˡ, T, ρ, constants
) where FT
    cloud_water_adjustment = zero(qᶜˡ)
    return (; cloud_water_adjustment,
              rate = zero(qᶜˡ),
              qᶜˡ,
              qᵛ,
              qᵛ⁺ˡ,
              T)
end

@inline function predicted_supersaturation_adjustment(
    parameters::ProcessRateParameters{PFT, true}, p3,
    qᶜˡ, qᵛ, qᵛ⁺ˡ, sᵛ⁺ˡ, T, ρ, constants
) where PFT
    FT = typeof(qᶜˡ)
    τ = max(parameters.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)

    cloud_water_adjustment = (qᵛ - qᵛ⁺ˡ - sᵛ⁺ˡ) / ξˡ
    cloud_water_adjustment = max(cloud_water_adjustment, -max(0, qᶜˡ))
    cloud_water_adjustment = ifelse(sᵛ⁺ˡ < 0,
                                    min(cloud_water_adjustment, zero(FT)),
                                    cloud_water_adjustment)
    cloud_water_adjustment = ifelse(abs(cloud_water_adjustment) <
                                    100 * eps(FT) * max(qᵛ⁺ˡ, qᵛ),
                                    zero(FT), cloud_water_adjustment)
    adjusted_temperature = T + cloud_water_adjustment * ℒˡ / cᵖᵈ
    adjusted_saturation = saturation_specific_humidity(
        adjusted_temperature, ρ, constants, PlanarLiquidSurface())

    return (; cloud_water_adjustment,
              rate = cloud_water_adjustment / τ,
              qᶜˡ = qᶜˡ + cloud_water_adjustment,
              qᵛ = qᵛ - cloud_water_adjustment,
              qᵛ⁺ˡ = adjusted_saturation,
              T = adjusted_temperature)
end

@inline function cloud_vapor_relaxation_coefficient(p3, qᶜˡ, ρ, Dᵛ, μᶜˡ, λᶜˡ,
                                                    nᶜˡ_bounded)
    FT = typeof(qᶜˡ)
    floors = p3.process_rates.floors
    integrated_diameter = nᶜˡ_bounded * (μᶜˡ + 1) / max(λᶜˡ, FT(floors.divisor))
    active = qᶜˡ >= p3.minimum_mass_mixing_ratio
    return ifelse(active, 2 * FT(π) * ρ * Dᵛ * integrated_diameter, zero(FT))
end

@inline function rain_vapor_relaxation_coefficient(p3, qʳ, nʳ, ρ, transport)
    FT = typeof(qʳ)
    qʳ_eff = max(0, qʳ)
    parameters = p3.process_rates
    nʳ_eff = max(nʳ, FT(p3.minimum_number_mixing_ratio))
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    ventilation = rain_ventilation_integral(p3.rain.evaporation, qʳ_eff, nʳ_eff,
                                           transport.ν, transport.Dᵛ, parameters)
    relaxation_coefficient = 2 * FT(π) * ventilation.Nʳ₀ * ρ * transport.Dᵛ *
                             ventilation.integral

    return ifelse(active, relaxation_coefficient, zero(FT))
end

@inline function ice_vapor_relaxation_coefficient(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                                  constants, transport, q)
    FT = typeof(qⁱ)
    parameters = p3.process_rates
    nⁱ_eff = max(nⁱ, FT(p3.minimum_number_mixing_ratio))
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ, parameters.floors)

    Dᵛ = transport.Dᵛ
    ν = transport.ν

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ, parameters.floors)
    ρ_air = density(T, P, q, constants)
    ρ_correction = ice_air_density_correction(parameters, p3.ice.fall_speed.reference_air_density, ρ_air)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                  p3.ice.deposition.ventilation_enhanced,
                                  m_mean, Fᶠ, Fˡ, ρᶠ, parameters, ν, Dᵛ,
                                  ρ_correction)

    # This is the raw inverse relaxation coefficient; the psychrometric correction
    # is applied later through the coupled `ξˡ` / `ξⁱ` factor.
    return 2 * FT(π) * ρ * Dᵛ * nⁱ_eff * C_fv
end

# One species' share of the diffusional-growth budget: its relaxation weight
# `ε / ε_total` against the total driver, plus the transient approach to
# liquid supersaturation, over the psychrometric correction `ξ` for the phase it condenses
# into. All four species use this; only `ε` and `ξ` differ, so a transposed pair
# would otherwise be invisible in four near-identical lines.
@inline function growth_share(relaxation_coefficient, psychrometric_correction,
                              external_driver, total_relaxation, supersaturation,
                              transient_response)
    relaxation_fraction = relaxation_coefficient / total_relaxation
    return (external_driver * relaxation_fraction +
            (supersaturation - external_driver / total_relaxation) *
            relaxation_fraction * transient_response) / psychrometric_correction
end

"""
$(TYPEDSIGNATURES)

Compute cloud, rain, and ice diffusional growth rates using a shared
semi-analytic saturation adjustment, in the `SCF = SPF = 1` limit; the subgrid
cloud/precipitation fraction framework is handled separately.
"""
@inline function coupled_saturation_adjustment_rates(p3, qᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
                                                     qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                                     constants, transport, q,
                                                     μᶜˡ, λᶜˡ, nᶜˡ_bounded,
                                                     temperature_tendency,
                                                     vapor_tendency)
    FT = typeof(qᶜˡ)
    parameters = p3.process_rates
    floors = parameters.floors
    τ = max(parameters.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    ℒⁱ = sublimation_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)

    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    dqᵛ⁺ⁱ_dT = qᵛ⁺ⁱ * ℒⁱ / (Rᵛ * T^2)
    # Psychrometric correction factors over liquid (ξˡ) and ice (ξⁱ) surfaces.
    ξˡ = 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
    ξⁱ = 1 + ℒⁱ * dqᵛ⁺ⁱ_dT / cᵖᵈ

    cloud_relaxation = cloud_vapor_relaxation_coefficient(p3, qᶜˡ, ρ, transport.Dᵛ,
                                                          μᶜˡ, λᶜˡ, nᶜˡ_bounded)
    rain_relaxation = rain_vapor_relaxation_coefficient(p3, qʳ, nʳ, ρ, transport)
    # `qⁱ` is the dry ice mass, so the total ice mass is `qⁱ + qʷⁱ`. Compute
    # `qⁱ_total`/`Fˡ` once here and reuse them for the relaxation gates and the
    # tiny-mass overrides below.
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ, floors)
    # The dry-ice and wet-ice relaxation coefficients share the same
    # `ice_vapor_relaxation_coefficient` and select mutually exclusive liquid-fraction
    # regimes (dry ice below the wet-ice threshold, liquid-coated ice at or above it),
    # so evaluate the coefficient — which carries a `density()` and a
    # ventilation-table lookup — once.
    ice_relaxation_active = qⁱ_total >= p3.minimum_mass_mixing_ratio
    ice_relaxation = ice_vapor_relaxation_coefficient(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                                      constants, transport, q)
    dry_ice_relaxation = ifelse(
        ice_relaxation_active & (Fˡ < p3.process_rates.liquid_fraction_clipping_threshold),
        ice_relaxation, zero(FT))
    # The wet-ice surface condenses vapor as liquid, so it couples through `ξˡ`
    # (like cloud), not through the Bergeron coupling.
    coating_relaxation = ifelse(
        ice_relaxation_active & (Fˡ >= p3.process_rates.liquid_fraction_clipping_threshold),
        ice_relaxation, zero(FT))

    ice_liquid_coupling = (1 + ℒⁱ * dqᵛ⁺ˡ_dT / cᵖᵈ) / ξⁱ
    total_relaxation = max(cloud_relaxation + rain_relaxation +
                           dry_ice_relaxation * ice_liquid_coupling + coating_relaxation,
                           FT(floors.rate_scale))
    transient_response = (1 - exp(-total_relaxation * τ)) / τ
    # `qᵛ`, `qᵛ⁺ˡ`, `qᵛ⁺ⁱ` arrive already adjusted by the G&M step in
    # `compute_p3_process_rates`, so the local diagnostic supersaturation here is
    # the post-G&M value, not the host-advected `sᵛ⁺ˡ`.
    supersaturation = qᵛ - qᵛ⁺ˡ
    bergeron_driver = -(qᵛ⁺ˡ - qᵛ⁺ⁱ) * ice_liquid_coupling * dry_ice_relaxation
    # The external forcing term is the externally driven change in liquid-relative
    # supersaturation: dqᵛ/dt - (dqᵛ⁺ˡ/dT) dT/dt. Breeze currently approximates
    # this with adiabatic cooling, dT/dt = -g w / cᵖᵐ, and dqᵛ/dt = 0. Resolved
    # transport, turbulent mixing, radiation, and user forcing are omitted.
    external_driver = vapor_tendency - dqᵛ⁺ˡ_dT * temperature_tendency
    total_driver = external_driver + bergeron_driver

    raw_cloud_growth = growth_share(cloud_relaxation, ξˡ, total_driver, total_relaxation,
                                    supersaturation, transient_response)
    raw_rain_growth = growth_share(rain_relaxation, ξˡ, total_driver, total_relaxation,
                                   supersaturation, transient_response)
    raw_ice_growth = growth_share(dry_ice_relaxation, ξⁱ, total_driver, total_relaxation,
                                  supersaturation, transient_response) +
                     (qᵛ⁺ˡ - qᵛ⁺ⁱ) * dry_ice_relaxation / ξⁱ
    # Liquid-on-ice coating uses `ξˡ` (like cloud) since the surface condenses
    # vapor as liquid; no Bergeron contribution because the surface is already
    # at liquid saturation.
    raw_coating_growth = growth_share(coating_relaxation, ξˡ, total_driver,
                                      total_relaxation, supersaturation,
                                      transient_response)

    𝒮ˡ = supersaturation / max(qᵛ⁺ˡ, FT(floors.divisor))
    𝒮ⁱ = qᵛ / max(qᵛ⁺ⁱ, FT(floors.divisor)) - 1
    # The tiny-mass clauses below all gate on the total hydrometeor mass
    # (`qⁱ_total`, computed above).
    tiny_mass = parameters.tiny_mass_evaporation_threshold
    subsaturated = -parameters.subsaturation_evaporation_threshold
    raw_cloud_growth = ifelse((𝒮ˡ < subsaturated) & (qᶜˡ < tiny_mass),
                              -qᶜˡ / τ, raw_cloud_growth)
    raw_rain_growth = ifelse((𝒮ˡ < subsaturated) & (qʳ < tiny_mass),
                             -qʳ / τ, raw_rain_growth)
    # Match the cloud/rain branches above: do NOT clamp the prognostic
    # before the sign flip. When advection leaves qⁱ or qʷⁱ slightly negative,
    # the override should produce a positive deposition/coating-condensation
    # rate so the downstream cap (lines 943 / 946) can pull mass back from
    # vapor and restore the field. The qᵛ/τ caps still bound the magnitude.
    raw_ice_growth = ifelse((𝒮ⁱ < subsaturated) & (qⁱ_total < tiny_mass) &
                            (Fˡ < parameters.liquid_fraction_clipping_threshold),
                            -qⁱ / τ, raw_ice_growth)
    # Wet-ice tiny-mass instant evaporation of the liquid coating.
    raw_coating_growth = ifelse((𝒮ⁱ < subsaturated) & (qⁱ_total < tiny_mass) &
                                (Fˡ >= parameters.liquid_fraction_clipping_threshold),
                                -qʷⁱ / τ, raw_coating_growth)

    condensation = clamp(raw_cloud_growth, -max(0, qᶜˡ) / τ, max(0, qᵛ) / τ)
    rain_condensation = min(max(0, raw_rain_growth), max(0, qᵛ) / τ)
    rain_evaporation = min(max(0, -raw_rain_growth), max(0, qʳ) / τ)

    is_sublimation = raw_ice_growth < 0
    calibration = ifelse(is_sublimation,
                         p3.process_rates.calibration_factor_sublimation,
                         p3.process_rates.calibration_factor_deposition)
    deposition_raw = raw_ice_growth * calibration
    # Sublimation is limited to the dry ice mass per unit time, `qⁱ / τ`.
    deposition = clamp(deposition_raw, -max(0, qⁱ) / τ, max(0, qᵛ) / τ)

    coating_condensation = min(max(0, raw_coating_growth), max(0, qᵛ) / τ)
    coating_evaporation = min(max(0, -raw_coating_growth), max(0, qʷⁱ) / τ)

    return P3CoupledVaporRates{FT}(condensation, rain_evaporation, rain_condensation,
                                   deposition, coating_condensation, coating_evaporation)
end
