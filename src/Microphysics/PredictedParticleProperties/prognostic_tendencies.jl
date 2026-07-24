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
##### Individual field tendencies
#####
##### These functions combine process rates into tendencies for each prognostic field.
##### Phase 1 processes: autoconversion, accretion, evaporation, deposition, melting
##### Phase 2 processes: aggregation, riming, shedding, refreezing
#####
##### Sign convention (M7):
##### ─────────────────────
##### All ONE-DIRECTIONAL rate functions return POSITIVE MAGNITUDES.
##### Signs are applied here in the tendency assembly as explicit gain − loss.
#####
##### BIDIRECTIONAL rates (condensation, deposition) retain their natural sign:
###   positive = source (condensation/deposition)
###   negative = sink   (evaporation/sublimation)
##### These are used directly as gains; their negative values contribute as losses.
#####
##### This convention ensures each tendency function reads as:
#####   tendency = ρ × (gains − losses)
##### with no hidden negations inside the rate functions.
#####

@inline liquid_fraction_routing_active(::Nothing) = true
@inline liquid_fraction_routing_active(prp::ProcessRateParameters) = prp.liquid_fraction_active

"""
$(TYPEDSIGNATURES)

Compute cloud liquid mass tendency from P3 process rates.

Cloud liquid gains from:
- Condensation (Phase 1)

Cloud liquid is consumed by:
- Autoconversion (Phase 1)
- Accretion by rain (Phase 1)
- Riming by ice (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
"""
@inline function tendency_ρqᶜˡ(rates::P3ProcessRates, ρ)
    # Phase 1: condensation (positive = cloud forms)
    # CCN activation (vapor → cloud)
    gain = rates.condensation + rates.ccn_activation_mass
    # Phase 1: autoconversion and accretion
    # Phase 2: cloud riming by ice, immersion freezing, homogeneous freezing
    # Above-freezing: cloud collected by melting ice → qʷⁱ
    # Wet growth: retained cloud collection goes to ice or qʷⁱ; in the
    # non-liquid-fraction branch, excess cloud collection is shed to rain.
    loss = rates.autoconversion + rates.accretion + rates.cloud_riming +
           rates.cloud_freezing_mass + rates.cloud_homogeneous_mass +
           rates.cloud_warm_collection + rates.wet_growth_cloud +
           rates.wet_growth_shedding
    return ρ * (gain - loss)
end

"""
$(TYPEDSIGNATURES)

Compute rain mass tendency from P3 process rates.

Rain gains from:
- Autoconversion (Phase 1)
- Accretion (Phase 1)
- Complete melting (Phase 1) - meltwater that sheds from ice
- Shedding (Phase 2) - liquid coating shed from ice (D ≥ 9 mm)
- Wet growth shedding (D8) - excess collection beyond freezing capacity

Rain loses from:
- Evaporation (Phase 1)
- Riming (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
- Rain warm collection by ice (T > T₀) → qʷⁱ
- Wet growth rain rerouting → qʷⁱ
"""
@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ)
    return tendency_ρqʳ(rates, ρ, nothing)
end

@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ, prp::Union{Nothing, ProcessRateParameters})
    # Phase 1: gains from autoconv, accr, complete_melt; loses from evap
    # Phase 2: gains from shedding; loses from riming, freezing, homogeneous freezing
    # Milbrandt et al. (2025): above-freezing collection and wet growth go to qʷⁱ, NOT rain.
    # Rain warm collection is a rain SINK (collected by ice → qʷⁱ).
    # rain condensation (vapor → rain)
    # wet_growth_shedding — excess collection beyond freezing capacity goes to rain.
    # Note: rain_warm_collection is zeroed at rate-assembly time in the non-liquid-
    # fraction branch (Fortran microphy_p3.f90:3055-3066) so it can safely be added
    # here unconditionally.
    cloud_warm_rain_gain = ifelse(liquid_fraction_routing_active(prp),
                                  zero(typeof(ρ)),
                                  rates.cloud_warm_collection)
    gain = rates.autoconversion + rates.accretion + rates.complete_melting +
           rates.shedding + rates.rain_condensation + rates.wet_growth_shedding +
           cloud_warm_rain_gain
    loss = rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass +
           rates.rain_homogeneous_mass + rates.rain_warm_collection + rates.wet_growth_rain
    return ρ * (gain - loss)
end

"""
$(TYPEDSIGNATURES)

Compute rain number tendency from P3 process rates.

Rain number gains from:
- Autoconversion (Phase 1)
- Complete melting (Phase 1) - new rain drops from melted ice
- Breakup (Phase 1) - large drops fragment into smaller ones
- Shedding (Phase 2)
- Shed drops from above-freezing cloud collection (Fortran ncshdc)

Rain number loses from:
- Self-collection (Phase 1)
- Evaporation (Phase 1) - proportional number removal
- Riming (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
- Rain warm collection number (M9, Fortran nrcoll)
"""
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, nⁱ, qⁱ, nʳ, qʳ, p3)
    FT = typeof(ρ)
    prp = p3.process_rates

    # Phase 1: New drops from autoconversion. Seed-drop mass varies by scheme:
    # KK2000 → 25 μm radius (Fortran cons3⁻¹), Kogan2013 → 40 μm (cons8⁻¹),
    # SB2001 → mass = 2/7.6923e9 (Fortran assembles `nr += 0.5 × ncautc`).
    n_from_autoconv = rates.autoconversion / rain_seed_drop_mass(p3)

    # Phase 1: New drops from complete melting (conserve number). The process
    # limiter carries this companion explicitly because whole-particle cleanup
    # can transfer the remaining population even when dry-ice mass is zero.
    n_from_melt = rates.melting_number

    # Phase 1: Evaporation removes rain number proportionally (Fortran nrevp =
    # qrevp·nr/qr, microphy_p3.f90:3698). Consume the value the process operator
    # already budgeted (`rain_evaporation_number`): it is formed from the
    # DSD-bounded nʳ and rescaled by the rain-number sink limiter `f_rain_number`.
    # Recomputing it here from the raw prognostic nʳ/qʳ would bypass both the
    # λ-limiter write-back and f_rain_number, breaking the port's no-over-depletion
    # guarantee and disagreeing with the homogeneous-freezing residual, which
    # already consumes the budgeted value.
    n_from_evap = rates.rain_evaporation_number

    # Gains: shedding produces rain drops
    # cloud_warm_collection → new rain drops from above-freezing cloud
    #      collection (Fortran ncshdc = qcshd × 1.923e6). Only in
    #      non-liquid-fraction path; when liquid fraction is active, collected
    #      mass goes to qʷⁱ, not rain.
    # wet_growth_shedding_number → rain drops from excess wet growth (Fortran nrshdr)
    cloud_warm_rain_n = ifelse(prp.liquid_fraction_active, zero(FT),
                               rates.cloud_warm_collection * FT(1.923e6))
    n_gain = n_from_autoconv + n_from_melt +
             rates.rain_breakup +
             rates.shedding_number +
             cloud_warm_rain_n +
             rates.wet_growth_shedding_number
    # Losses (all positive magnitudes, M7)
    # rain_warm_collection_number → rain number sink from above-freezing rain
    #      collection (Fortran nrcoll)
    n_loss = n_from_evap +
             rates.rain_self_collection +
             rates.rain_riming_number +
             rates.rain_freezing_number +
             rates.rain_homogeneous_number +
             rates.rain_warm_collection_number

    # DSD number correction feedback (Fortran get_rain_dsd2 writes back bounded nr)
    return ρ * (n_gain - n_loss + rates.rain_number_correction)
end

"""
$(TYPEDSIGNATURES)

Compute ice mass tendency from P3 process rates.

Ice gains from:
- Deposition (Phase 1)
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)
- Homogeneous freezing of cloud/rain (Phase 2, T < -40°C)

Ice loses from:
- Partial melting (Phase 1) - becomes liquid coating
- Complete melting (Phase 1) - sheds to rain
"""
@inline function tendency_ρqⁱ(rates::P3ProcessRates, ρ)
    return tendency_ρqⁱ(rates, ρ, nothing)
end

@inline function retained_non_liquid_wet_growth(rates::P3ProcessRates)
    FT = typeof(rates.wet_growth_cloud + rates.wet_growth_rain)
    wg_cloud = clamp_positive(rates.wet_growth_cloud)
    wg_rain = clamp_positive(rates.wet_growth_rain)
    wg_total = wg_cloud + wg_rain
    shed_fraction = safe_divide(rates.wet_growth_shedding, max(wg_total, eps(FT)), zero(FT))
    retained_fraction = clamp(one(FT) - shed_fraction, zero(FT), one(FT))
    retained_cloud = wg_cloud * retained_fraction
    retained_rain = wg_rain * retained_fraction
    return retained_cloud, retained_rain
end

@inline function tendency_ρqⁱ(rates::P3ProcessRates, ρ, prp::Union{Nothing, ProcessRateParameters})
    # Phase 1: deposition, melting (both partial and complete reduce ice mass)
    # Phase 2: riming (cloud + rain), refreezing, nucleation, and freezing.
    # Splintering mass is already part of the riming mass (splinters fragment existing rime),
    # so it is not added separately to the total ice mass tendency.
    retained_cloud, retained_rain = retained_non_liquid_wet_growth(rates)
    retained_wet_growth = ifelse(liquid_fraction_routing_active(prp),
                                 zero(typeof(ρ)),
                                 retained_cloud + retained_rain)
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass + retained_wet_growth
    # Total melting reduces ice mass (partial stays as liquid coating, complete sheds)
    loss = rates.partial_melting + rates.complete_melting
    return ρ * (gain - loss)
end

"""
$(TYPEDSIGNATURES)

Compute ice number tendency from P3 process rates.

Ice number gains from:
- Deposition nucleation (Phase 2)
- Immersion freezing of cloud/rain (Phase 2)
- Rime splintering (Phase 2)
- Homogeneous freezing of cloud/rain (Phase 2, T < -40°C)

Ice number loses from:
- Melting (Phase 1)
- Aggregation (Phase 2)
- Global number limiter (C3, impose_max_Ni)
- Ice λ-limiter correction (Fortran f1pr09/f1pr10 write-back)
"""
@inline function tendency_ρnⁱ(rates::P3ProcessRates, ρ)
    # Gains from nucleation, freezing, splintering, homogeneous freezing
    gain = rates.nucleation_number + rates.cloud_freezing_number +
           rates.rain_freezing_number + rates.splintering_number +
           rates.cloud_homogeneous_number + rates.rain_homogeneous_number
    # Losses (all positive magnitudes, M7)
    # sublimation_number — ice number loss from sublimation (Fortran nisub)
    # ni_limit: C3 global Nᵢ cap (impose_max_Ni); relaxation sink above N_max/ρ.
    loss = rates.melting_number + rates.sublimation_number + rates.aggregation + rates.ni_limit
    return ρ * (gain - loss + rates.ice_number_correction)
end

"""
$(TYPEDSIGNATURES)

Compute rime mass tendency from P3 process rates.

Rime mass gains from:
- Cloud riming (Phase 2)
- Rain riming (Phase 2)
- Refreezing (Phase 2)
- Immersion freezing (frozen cloud/rain becomes rimed ice) (Phase 2)
- Homogeneous freezing (frozen cloud/rain deposits as dense rime) (Phase 2, T < -40°C)

Rime mass loses from:
- Melting (proportional to rime fraction) (Phase 1)
- Sublimation (proportional to rime fraction) (Phase 1)
"""
@inline function tendency_ρqᶠ(rates::P3ProcessRates, ρ, Fᶠ)
    return tendency_ρqᶠ(rates, ρ, Fᶠ, nothing)
end

@inline function tendency_ρqᶠ(rates::P3ProcessRates, ρ, Fᶠ, prp::Union{Nothing, ProcessRateParameters})
    # Phase 2: gains from riming, refreezing, freezing, and homogeneous freezing
    # Frozen cloud/rain becomes fully rimed ice (100% rime fraction for new frozen particles)
    retained_cloud, retained_rain = retained_non_liquid_wet_growth(rates)
    retained_wet_growth = ifelse(liquid_fraction_routing_active(prp),
                                 zero(typeof(ρ)),
                                 retained_cloud + retained_rain)
    gain = rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass +
           rates.wet_growth_densification_mass + retained_wet_growth
    # Ordinary melting and sublimation remove the beginning-of-stage rime
    # fraction. Whole-particle clipping instead drains the explicitly
    # reconstructed residual rime companion, including post-process changes.
    sublimation = clamp_positive(-rates.deposition)
    ordinary_complete_melting =
        max(0, rates.complete_melting - rates.clipping_dry_mass)
    # Splintering (nCat=1): Fortran subtracts splintering from riming then adds it back
    # as qcmul/qrmul, netting to zero effect on rime. Since cloud_riming and rain_riming
    # are the full (unreduced) rates, no splintering subtraction is needed here.
    loss = Fᶠ * (rates.partial_melting + ordinary_complete_melting + sublimation) +
           rates.clipping_rime_mass
    return ρ * (gain - loss)
end

"""
$(TYPEDSIGNATURES)

Compute rime volume tendency from P3 process rates.

Rime volume changes with rime mass: ∂bᶠ/∂t = ∂qᶠ/∂t / ρ_rime.
Includes sublimation loss (M8): sublimation removes rime volume proportionally.
Includes melt-densification (Fortran P3 v5.5.0): during melting, low-density
rime portions melt preferentially, driving the remaining rime toward 917 kg/m³.
"""
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ, qⁱ, prp)
    FT = typeof(ρ)

    ρᶠ_safe = max(ρᶠ, prp.minimum_rime_density)
    ρ_rim_new_safe = max(rates.rime_density_new, prp.minimum_rime_density)

    # Fortran P3 v5.5.0: rho_rimeMax = 900 for rain rime and freezing
    ρ_rimemax = prp.maximum_rime_density
    # Fortran uses rho_rimeMax (900) for homogeneous freezing rime volume, not 917
    ρ_rim_hom = prp.maximum_rime_density

    retained_cloud, retained_rain = retained_non_liquid_wet_growth(rates)
    retained_cloud_volume = ifelse(prp.liquid_fraction_active,
                                   zero(FT),
                                   retained_cloud / ρ_rim_new_safe)
    retained_rain_volume = ifelse(prp.liquid_fraction_active,
                                  zero(FT),
                                  retained_rain / ρ_rimemax)

    # Phase 2: Volume gain from new rime
    # Cloud riming uses Cober-List computed density; rain riming uses rho_rimeMax = 900
    # Immersion freezing uses rho_rimeMax = 900 (Fortran convention, not water density)
    # Refreezing uses rho_rimeMax = 900 (Fortran: qifrz * i_rho_rimeMax, line 4253)
    volume_gain = rates.cloud_riming / ρ_rim_new_safe +
                   rates.rain_riming / ρ_rimemax +
                   rates.refreezing / ρ_rimemax +
                   (rates.cloud_freezing_mass + rates.rain_freezing_mass) / ρ_rimemax +
                   (rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass) / ρ_rim_hom +
                   rates.wet_growth_densification_volume +
                   retained_cloud_volume + retained_rain_volume

    # Ordinary melting and sublimation remove volume proportionally. A whole-
    # particle clip uses the reconstructed companion volume so post-process rime
    # and densification changes are removed exactly.
    sublimation = clamp_positive(-rates.deposition)
    ordinary_complete_melting =
        max(0, rates.complete_melting - rates.clipping_dry_mass)
    ordinary_total_melting = rates.partial_melting + ordinary_complete_melting
    volume_loss = Fᶠ * (ordinary_total_melting + sublimation) / ρᶠ_safe +
                  rates.clipping_rime_volume

    # Melt-densification (Fortran P3 v5.5.0 lines 4309-4313)
    # Low-density rime portions melt first → remaining ice approaches 917 kg/m³.
    # In tendency form: additional volume reduction = bᶠ × (917 - ρᶠ) × |melt| / (ρᶠ × qⁱ)
    # Fortran guards with `.not. log_LiquidFrac`: when liquid fraction is active,
    # melt-densification is skipped because the liquid is tracked explicitly in qʷⁱ.
    # NOTE: The densification target is solid ice density (917), NOT rho_rimeMax (900).
    ρ_solid_ice = prp.pure_ice_density  # 917 kg/m³
    qⁱ_safe = max(qⁱ, FT(1e-12))
    bᶠ = Fᶠ * qⁱ_safe / ρᶠ_safe
    densification = bᶠ * (ρ_solid_ice - ρᶠ_safe) * ordinary_total_melting /
                    (ρᶠ_safe * qⁱ_safe)
    # Only apply when ρᶠ < 917, there is melting, AND liquid fraction is not active
    apply_densification = (ρᶠ_safe < ρ_solid_ice) & !prp.liquid_fraction_active
    densification = ifelse(apply_densification, densification, zero(FT))

    return ρ * (volume_gain - volume_loss - densification)
end

@inline function split_splintering_mass(rates::P3ProcessRates, prp::ProcessRateParameters)
    FT = typeof(rates.splintering_mass)
    # `rime_splintering_rates` scales the cloud branch by `splintering_cloud_riming_scale`
    # (1 for nCat=1, 0 for nCat>1). The reverse split must mirror that scaling so all
    # splinter mass is attributed to the rain branch when the cloud branch is disabled.
    cloud_eff = clamp_positive(rates.cloud_riming) * FT(prp.splintering_cloud_riming_scale)
    rain_eff = clamp_positive(rates.rain_riming)
    total_eff = cloud_eff + rain_eff
    cloud_fraction = safe_divide(cloud_eff, total_eff, zero(FT))
    rain_fraction = safe_divide(rain_eff, total_eff, zero(FT))
    splintering_mass = clamp_positive(rates.splintering_mass)
    return splintering_mass * cloud_fraction, splintering_mass * rain_fraction
end

"""
$(TYPEDSIGNATURES)

Compute cloud-number tendency from P3 process rates.

Activation creates new cloud droplets. Autoconversion, accretion, riming,
freezing, and above-freezing collection remove cloud droplets in proportion
to the cloud mass they consume, following the Fortran `nc` budget structure.
"""
@inline function tendency_ρnᶜˡ(rates::P3ProcessRates, ρ, Nᶜ, qᶜˡ, p3)
    FT = typeof(ρ)
    prp = p3.process_rates
    # Nᶜ is per-volume [#/m³]; dividing by ρ gives per-mass nᶜˡ [#/kg],
    # matching Fortran's nc/qc → [#/kg/s] when multiplied by mass rates.
    number_per_mass = safe_divide(Nᶜ, ρ * qᶜˡ, zero(FT))
    seed_drop_mass = 4 * FT(π) / 3 * prp.liquid_water_density * FT(1e-18)
    activation_number = ifelse(iszero(rates.ccn_activation_number),
                               rates.ccn_activation_mass / seed_drop_mass,
                               rates.ccn_activation_number)

    # Scheme-aware cloud-number loss from autoconversion. SB2001 produces a
    # fixed-mass drizzle drop per unit converted mass; KK2000 and Kogan2013
    # scale by the in-cloud nc/qc ratio (Fortran ncautc = qcaut × nc/qc).
    autoconv_n = cloud_number_loss_from_autoconversion(p3, rates.autoconversion,
                                                       qᶜˡ, Nᶜ, ρ)

    number_loss = autoconv_n +
                  number_per_mass * rates.accretion +
                  rates.cloud_self_collection +
                  rates.cloud_riming_number +
                  rates.cloud_freezing_number +
                  rates.cloud_homogeneous_number +
                  rates.cloud_warm_collection_number

    # DSD number correction feedback (Fortran get_cloud_dsd2 writes back bounded nc)
    return ρ * (activation_number - number_loss + rates.cloud_number_correction)
end

"""
$(TYPEDSIGNATURES)

Compute liquid on ice tendency from P3 process rates.

Following [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction), the
full budget is:

```math
\\frac{dq^{wi}}{dt} = q_{melt,partial} + q_{ccoll} + q_{rcoll} + q_{wgrth1c} + q_{wgrth1r}
                    - q_{lshd} - q_{ifrz}
```

Gains from:
- Partial melting (meltwater stays on ice as liquid coating)
- Above-freezing cloud collection (qccoll: T > T₀, cloud → qʷⁱ)
- Above-freezing rain collection (qrcoll: T > T₀, rain → qʷⁱ)
- Wet growth cloud rerouting (qwgrth1c: excess collection → qʷⁱ)
- Wet growth rain rerouting (qwgrth1r: excess collection → qʷⁱ)

Loses from:
- Shedding (liquid sheds to rain from D ≥ 9 mm particles)
- Refreezing (liquid refreezes to rime)
"""
@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ)
    return tendency_ρqʷⁱ(rates, ρ, nothing)
end

@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ, prp::Union{Nothing, ProcessRateParameters})
    # Include coating condensation/evaporation (Fortran qlcon/qlevp)
    # wet_growth_shedding diverts excess wet growth mass from qʷⁱ to rain.
    # Note: rain_warm_collection is zeroed at rate-assembly time in the non-liquid-
    # fraction branch (Fortran does not transfer rain mass to qʷⁱ in that path), so
    # it can safely be added here unconditionally.
    liquid_fraction_active = liquid_fraction_routing_active(prp)
    cloud_warm_gain = ifelse(liquid_fraction_active, rates.cloud_warm_collection, zero(typeof(ρ)))
    rain_warm_gain = ifelse(liquid_fraction_active, rates.rain_warm_collection, zero(typeof(ρ)))
    wet_growth_cloud_gain = ifelse(liquid_fraction_active, rates.wet_growth_cloud, zero(typeof(ρ)))
    wet_growth_rain_gain = ifelse(liquid_fraction_active, rates.wet_growth_rain, zero(typeof(ρ)))
    wet_growth_shedding_loss = ifelse(liquid_fraction_active, rates.wet_growth_shedding, zero(typeof(ρ)))
    gain = rates.partial_melting +
        cloud_warm_gain +
        rain_warm_gain +
        wet_growth_cloud_gain +
        wet_growth_rain_gain +
        rates.coating_condensation
    loss = rates.shedding + rates.refreezing + rates.coating_evaporation +
        wet_growth_shedding_loss
    return ρ * (gain - loss)
end

"""
$(TYPEDSIGNATURES)

Compute vapor mass tendency from P3 process rates.

Vapor is consumed by:
- Condensation (vapor → cloud liquid)
- Deposition (vapor → ice)
- Deposition nucleation (vapor → ice)

Vapor is produced by:
- Cloud evaporation (negative condensation)
- Rain evaporation
- Sublimation (negative deposition)

When `predict_supersaturation = true`, the G&M one-shot alignment is
folded into `rates.condensation` (= M&G `cond` + `cond_GM`), so vapor and
cloud tendencies pick it up automatically when integrated with
`dt = sink_limiting_timescale`. See `predicted_supersaturation_adjustment`.
"""
@inline function tendency_ρqᵛ(rates::P3ProcessRates, ρ)
    # Condensation: positive = vapor loss (cond), negative = vapor gain (cloud evap)
    # Deposition:   positive = vapor loss (dep),  negative = vapor gain (sublimation)
    # Rain evaporation: positive magnitude (M7) = vapor gain
    # Nucleation: always positive = vapor loss
    # CCN activation, rain condensation, and coating condensation are all vapor sinks;
    #      coating evaporation is a vapor source.
    vapor_loss = rates.condensation + rates.deposition + rates.nucleation_mass +
                 rates.ccn_activation_mass + rates.rain_condensation + rates.coating_condensation
    vapor_gain = rates.rain_evaporation + rates.coating_evaporation
    return ρ * (vapor_gain - vapor_loss)
end

"""
$(TYPEDSIGNATURES)

Compute predicted supersaturation tendency from Grabowski & Morrison (2008).

When `predict_supersaturation = true`, supersaturation ``sˢᵃᵗ = qᵛ - q_{vs}``
is a prognostic variable advected by the dynamical core. The microphysical
tendency reproduces Fortran's post-step recompute ``sˢᵃᵗ = qᵛ - q_{vs}(T)``
(`microphy_p3.f90:5053-5063`). `compute_p3_process_rates` precomputes that
diagnostic tendency from the final local ``qᵛ`` and ``T`` implied by the
Fortran-ordered process rates.

When `predict_supersaturation = false`, returns zero tendency.
"""
@inline function tendency_ρsˢᵃᵗ(rates::P3ProcessRates, ρ, prp)
    raw = ρ * rates.predicted_ssat_tendency
    return ifelse(prp.predict_supersaturation, raw, zero(ρ))
end

"""
$(TYPEDSIGNATURES)

Aerosol-pool tendency: each activated cloud droplet removes one unit from the
unactivated reservoir, so ``∂ρn^a/∂t = -ρ \\, n_{\\text{nuc}}`` with
``n_{\\text{nuc}}`` the same activation rate that sources ``ρn^{cl}``. In the
prescribed-Nᶜ path `rates.ccn_activation_number` is zero, so this returns 0.
"""
@inline tendency_ρnᵃ(rates::P3ProcessRates, ρ) = -ρ * rates.ccn_activation_number

#####
##### Fallback methods for Nothing rates
#####
##### These are safety fallbacks that return zero tendency when rates
##### have not been computed (e.g., during incremental development).
#####

@inline tendency_ρqᶜˡ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqʳ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnᶜˡ(::Nothing, ρ, Nᶜ, qᶜˡ, p3) = zero(ρ)
@inline tendency_ρnʳ(::Nothing, ρ, nⁱ, qⁱ, args...) = zero(ρ)
@inline tendency_ρqⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᶠ(::Nothing, ρ, Fᶠ) = zero(ρ)
@inline tendency_ρbᶠ(::Nothing, ρ, Fᶠ, ρᶠ, prp...) = zero(ρ)
@inline tendency_ρzⁱ(::Nothing, ρ, qⁱ, nⁱ, zⁱ) = zero(ρ)
@inline tendency_ρqʷⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρsˢᵃᵗ(::Nothing, ρ, prp) = zero(ρ)
@inline tendency_ρqᵛ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnᵃ(::Nothing, ρ) = zero(ρ)
