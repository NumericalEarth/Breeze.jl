#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

using Oceananigans: Oceananigans

using Breeze.Thermodynamics: temperature,
                             adjustment_saturation_specific_humidity,
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             density,
                             liquid_latent_heat,
                             ice_latent_heat,
                             mixture_heat_capacity,
                             vapor_gas_constant,
                             MoistureMassFractions,
                             ThermodynamicConstants
using DocStringExtensions: TYPEDSIGNATURES

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
Morrison–Gettelman semi-analytic rates. Mirrors Fortran `microphy_p3.f90`'s
ssat alignment block (~3940–3989) which runs in-place on `qv`, `qc`, `T`,
`qvs`, and `qvi` before the per-species rate equations.

Given the advected supersaturation ``sˢᵃᵗ``, the diagnostic local
``qᵛ - qᵛ⁺ˡ``, and the liquid-side psychrometric factor
``ξˡ = 1 + ℒˡ² qᵛ⁺ˡ / (cᵖᵈ Rᵛ T²)``, compute the cloud-water increment

```math
ε = (qᵛ - qᵛ⁺ˡ - sˢᵃᵗ) / ξˡ
```

clamped to physical limits: ``ε`` cannot evaporate more cloud than is locally
available (``ε ≥ -qᶜˡ``), and when the advected ``sˢᵃᵗ`` is negative
``ε ≤ 0`` (no spurious condensation). The returned ``rate = ε / τ`` is
sized to `sink_limiting_timescale`, so one host step with
``dt = sink_limiting_timescale`` reproduces the one-shot ``ε`` exactly. If
the host integrates with ``dt ≠ τ`` the supersaturation alignment relaxes over multiple
steps rather than landing in one.

When `predict_supersaturation = false`, ``ε`` is gated to zero so the local
state passes through unchanged.
"""
@inline function predicted_supersaturation_adjustment(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, sˢᵃᵗ, T, constants)
    FT = typeof(qᶜˡ)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)

    ε = (qᵛ - qᵛ⁺ˡ - sˢᵃᵗ) / ξˡ
    ε = max(ε, -clamp_positive(qᶜˡ))
    ε = ifelse(sˢᵃᵗ < 0, min(ε, zero(FT)), ε)
    ε = ifelse(abs(ε) < 100 * eps(FT) * max(qᵛ⁺ˡ, qᵛ), zero(FT), ε)
    ε = ifelse(p3.process_rates.predict_supersaturation, ε, zero(FT))

    return (; ε,
              rate = ε / τ,
              qᶜˡ = qᶜˡ + ε,
              qᵛ = qᵛ - ε,
              T = T + ε * ℒˡ / cᵖᵈ)
end

@inline function cloud_condensation_epsilon(p3, qᶜˡ, ρ, D_v, μ_c, λ_c, nᶜˡ_bounded)
    FT = typeof(qᶜˡ)
    cdist = nᶜˡ_bounded * (μ_c + 1) / max(λ_c, FT(1e-30))
    active = qᶜˡ >= p3.minimum_mass_mixing_ratio
    return ifelse(active, 2 * FT(π) * ρ * D_v * cdist, zero(FT))
end

@inline function rain_condensation_epsilon(p3, qʳ, nʳ, ρ, transport)
    FT = typeof(qʳ)
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(clamp_positive(nʳ), FT(1e-16))
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio
    prp = p3.process_rates

    λ_r = rain_slope_parameter(qʳ_eff, nʳ_eff, prp)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λ_r, prp)
    N₀ = nʳ_bounded * λ_r
    I_VD = p3.rain.evaporation(log10(λ_r))
    I_const = FT(RAIN_F1R) / λ_r^2
    Sc_cbrt = cbrt(transport.nu / max(transport.D_v, FT(1e-10)))
    inv_sqrt_nu = 1 / sqrt(max(transport.nu, FT(1e-10)))
    I_evap = I_const + FT(RAIN_F2R) * Sc_cbrt * inv_sqrt_nu * I_VD
    epsilon_r = 2 * FT(π) * N₀ * ρ * transport.D_v * I_evap

    return ifelse(active, epsilon_r, zero(FT))
end

@inline function ice_relaxation_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                         constants, transport, q, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates
    nⁱ_eff = max(clamp_positive(nⁱ), FT(1e-16))
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)

    D_v = transport.D_v
    nu = transport.nu

    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_air = density(T, P, q, constants)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                  p3.ice.deposition.ventilation_enhanced,
                                  m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v,
                                  ρ_correction, p3, μ)

    # Fortran P3 computes the raw inverse relaxation coefficient here. The
    # psychrometric correction is applied later through the coupled `ξˡ` / `ξⁱ` factor.
    return 2 * FT(π) * ρ * D_v * nⁱ_eff * C_fv
end

# Dry-ice relaxation coefficient (Fortran `epsi(iice)`): active only when liquid
# fraction is below the wet-ice threshold. Fortran gates on total ice mass
# `qitot >= qsmall` (microphy_p3.f90:3298); in Julia `qitot = qⁱ + qʷⁱ`.
@inline function ice_deposition_epsilon(p3, qⁱ, qʷⁱ, nⁱ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                        constants, transport, q, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    active = (qⁱ_total >= p3.minimum_mass_mixing_ratio) & (Fˡ < prp.liquid_fraction_small)
    epsilon_i = ice_relaxation_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                        constants, transport, q, μ)
    return ifelse(active, epsilon_i, zero(FT))
end

# Wet-ice (liquid-coated) relaxation coefficient (Fortran `epsiw(iice)`): active
# only when liquid fraction is at or above the wet-ice threshold. Same formula
# as `ice_deposition_epsilon`; the two are mutually exclusive.
@inline function ice_coating_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                     constants, transport, q, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    active = (qⁱ_total >= p3.minimum_mass_mixing_ratio) & (Fˡ >= prp.liquid_fraction_small)
    epsilon_iw = ice_relaxation_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                         constants, transport, q, μ)
    return ifelse(active, epsilon_iw, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Compute cloud, rain, and ice diffusional growth rates using a shared
semi-analytic saturation adjustment. This mirrors the Fortran P3 structure with
`SCF = SPF = 1`; the subgrid cloud/precipitation fraction framework is handled
separately.
"""
@inline function coupled_saturation_adjustment_rates(p3, qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, qʷⁱ, nⁱ,
                                                     qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                                     constants, transport, q, μ,
                                                     μ_c, λ_c, nᶜˡ_bounded, w)
    FT = typeof(qᶜˡ)
    τ = max(p3.process_rates.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    ℒⁱ = sublimation_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)

    dqᵛ⁺ˡ_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    dqᵛ⁺ⁱ_dT = qᵛ⁺ⁱ * ℒⁱ / (Rᵛ * T^2)
    # Psychrometric correction factors over liquid (ξˡ) and ice (ξⁱ) surfaces.
    ξˡ = 1 + ℒˡ * dqᵛ⁺ˡ_dT / cᵖᵈ
    ξⁱ = 1 + ℒⁱ * dqᵛ⁺ⁱ_dT / cᵖᵈ

    εᶜˡ = cloud_condensation_epsilon(p3, qᶜˡ, ρ, transport.D_v, μ_c, λ_c, nᶜˡ_bounded)
    εʳ = rain_condensation_epsilon(p3, qʳ, nʳ, ρ, transport)
    εⁱ = ice_deposition_epsilon(p3, qⁱ, qʷⁱ, nⁱ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, ρ,
                                constants, transport, q, μ)
    # Fortran `epsiw_tot`: wet-ice surface condenses vapor as liquid, so it
    # couples through `ξˡ` (like cloud), not through the Bergeron coupling.
    εⁱʷ = ice_coating_epsilon(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, T, P, ρ,
                              constants, transport, q, μ)

    ice_liquid_coupling = (1 + ℒⁱ * dqᵛ⁺ˡ_dT / cᵖᵈ) / ξⁱ
    ε_total = max(εᶜˡ + εʳ + εⁱ * ice_liquid_coupling + εⁱʷ, FT(1e-20))
    transient = (1 - exp(-ε_total * τ)) / τ
    # `qᵛ`, `qᵛ⁺ˡ`, `qᵛ⁺ⁱ` arrive already adjusted by the G&M step in
    # `compute_p3_process_rates` (Fortran `microphy_p3.f90` ssat block ~3940–3989),
    # so the local diagnostic supersaturation here is the post-G&M value, not the
    # host-advected `sˢᵃᵗ`.
    ssat_liquid = qᵛ - qᵛ⁺ˡ
    bergeron_driver = -(qᵛ⁺ˡ - qᵛ⁺ⁱ) * ice_liquid_coupling * εⁱ
    # A_w: adiabatic supersaturation forcing from vertical motion.
    # Fortran reference: `aaa = ... - dqsdT*(-dum*g*i_cp) ...` with
    # dum = -cp/g · dT/dt, i.e. dT/dt|_dynamics ≈ -g/cᵖₘ · w. Here cᵖₘ is the
    # moist heat capacity of the local gas mixture, matching Fortran's `i_cp`.
    # We omit the (qᵛ - qᵛ_old)/dt contribution to Fortran's `aaa` because the
    # host does not carry qᵛ_old.
    cᵖₘ = mixture_heat_capacity(q, constants)
    g = constants.gravitational_acceleration
    A_w = (g / cᵖₘ) * dqᵛ⁺ˡ_dT * w
    A_total = A_w + bergeron_driver

    qc_raw = (A_total * εᶜˡ / ε_total + (ssat_liquid - A_total / ε_total) * εᶜˡ / ε_total * transient) / ξˡ
    qr_raw = (A_total * εʳ / ε_total + (ssat_liquid - A_total / ε_total) * εʳ / ε_total * transient) / ξˡ
    qi_raw = (A_total * εⁱ / ε_total + (ssat_liquid - A_total / ε_total) * εⁱ / ε_total * transient) / ξⁱ +
             (qᵛ⁺ˡ - qᵛ⁺ⁱ) * εⁱ / ξⁱ
    # Liquid-on-ice coating uses `ξˡ` (like cloud) since the surface condenses
    # vapor as liquid; no Bergeron contribution because the surface is already
    # at liquid saturation.
    ql_raw = (A_total * εⁱʷ / ε_total + (ssat_liquid - A_total / ε_total) * εⁱʷ / ε_total * transient) / ξˡ

    𝒮ˡ = ssat_liquid / max(qᵛ⁺ˡ, FT(1e-30))
    𝒮ⁱ = qᵛ / max(qᵛ⁺ⁱ, FT(1e-30)) - 1
    # Fortran tiny-mass clauses (3684-3685, 3715-3719, 3753-3756) all gate on
    # total hydrometeor mass. `qⁱ` is the dry ice mass in Julia — equivalent to
    # Fortran's `qitot - qiliq` — so `qⁱ + qʷⁱ` maps to Fortran `qitot`.
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    qc_raw = ifelse((𝒮ˡ < FT(-0.001)) & (qᶜˡ < FT(1e-12)), -qᶜˡ / τ, qc_raw)
    qr_raw = ifelse((𝒮ˡ < FT(-0.001)) & (qʳ < FT(1e-12)), -qʳ / τ, qr_raw)
    # Match the cloud/rain branches above: do NOT clamp_positive the prognostic
    # before the sign flip. When advection leaves qⁱ or qʷⁱ slightly negative,
    # the override should produce a positive deposition/coating-condensation
    # rate so the downstream cap (lines 943 / 946) can pull mass back from
    # vapor and restore the field. The qᵛ/τ caps still bound the magnitude.
    qi_raw = ifelse((𝒮ⁱ < FT(-0.001)) & (qⁱ_total < FT(1e-12)) &
                    (Fˡ < p3.process_rates.liquid_fraction_small),
                    -qⁱ / τ,
                    qi_raw)
    # Wet-ice tiny-mass instant evaporation of the liquid coating (Fortran 3753-3756).
    ql_raw = ifelse((𝒮ⁱ < FT(-0.001)) & (qⁱ_total < FT(1e-12)) &
                    (Fˡ >= p3.process_rates.liquid_fraction_small),
                    -qʷⁱ / τ,
                    ql_raw)

    condensation = ifelse(qc_raw < 0,
                          max(qc_raw, -clamp_positive(qᶜˡ) / τ),
                          min(qc_raw, clamp_positive(qᵛ) / τ))
    rain_condensation = ifelse(qr_raw < 0, zero(FT), min(qr_raw, clamp_positive(qᵛ) / τ))
    rain_evaporation = ifelse(qr_raw < 0,
                              min(-qr_raw, clamp_positive(qʳ) / τ),
                              zero(FT))

    is_sublimation = qi_raw < 0
    calibration = ifelse(is_sublimation,
                         p3.process_rates.calibration_factor_sublimation,
                         p3.process_rates.calibration_factor_deposition)
    deposition_raw = qi_raw * calibration
    # Fortran sublimation limit (3730): `qisub <= (qitot - qiliq)*i_dt` = dry
    # ice mass per unit time, which is `qⁱ / τ` in Julia conventions.
    deposition = ifelse(is_sublimation,
                        max(deposition_raw, -clamp_positive(qⁱ) / τ),
                        min(deposition_raw, clamp_positive(qᵛ) / τ))

    coating_condensation = ifelse(ql_raw < 0, zero(FT),
                                  min(ql_raw, clamp_positive(qᵛ) / τ))
    coating_evaporation = ifelse(ql_raw < 0,
                                 min(-ql_raw, clamp_positive(qʷⁱ) / τ),
                                 zero(FT))

    return P3CoupledVaporRates{FT}(condensation, rain_evaporation, rain_condensation,
                                   deposition, coating_condensation, coating_evaporation)
end
