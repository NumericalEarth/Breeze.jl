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
@inline liquid_fraction_routing_active(parameters::ProcessRateParameters) = parameters.liquid_fraction_active

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

@inline function tendency_ρqʳ(rates::P3ProcessRates, ρ, parameters::Union{Nothing, ProcessRateParameters})
    # Phase 1: gains from autoconv, accr, complete_melt; loses from evap
    # Phase 2: gains from shedding; loses from riming, freezing, homogeneous freezing
    # Milbrandt et al. (2025): above-freezing collection and wet growth go to qʷⁱ, NOT rain.
    # Rain warm collection is a rain SINK (collected by ice → qʷⁱ).
    # rain condensation (vapor → rain)
    # wet_growth_shedding — excess collection beyond freezing capacity goes to rain.
    # Note: rain_warm_collection is zeroed at rate-assembly time in the non-liquid-
    # fraction branch, so it can safely be added here unconditionally.
    cloud_warm_rain_gain = ifelse(liquid_fraction_routing_active(parameters),
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
- Shed drops from above-freezing cloud collection

Rain number loses from:
- Self-collection (Phase 1)
- Evaporation (Phase 1) - proportional number removal
- Riming (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
- Rain warm collection number (M9)
"""
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, p3)
    FT = typeof(ρ)
    parameters = p3.process_rates

    # Phase 1: New drops from autoconversion, at the scheme's seed-drop mass
    # (KK2000 → 25 μm radius).
    n_from_autoconv = rates.autoconversion / rain_seed_drop_mass(p3)

    # Phase 1: New drops from complete melting (conserve number). The process
    # limiter carries this companion explicitly because whole-particle cleanup
    # can transfer the remaining population even when dry-ice mass is zero.
    n_from_melt = rates.melting_number

    # Phase 1: Evaporation removes rain number in proportion to the mass it removes,
    # ṅʳ_evap = (nʳ/qʳ) q̇ʳ_evap. Consume the value the process operator already
    # budgeted (`rain_evaporation_number`): it is formed from the DSD-bounded nʳ and
    # rescaled by the rain-number sink limiter `f_rain_number`. Recomputing it here
    # from the raw prognostic nʳ/qʳ would bypass both the λ-limiter write-back and
    # f_rain_number, breaking the no-over-depletion guarantee and disagreeing with
    # the homogeneous-freezing residual, which already consumes the budgeted value.
    n_from_evap = rates.rain_evaporation_number

    # Gains: shedding produces rain drops
    # cloud_warm_collection → new rain drops from above-freezing cloud collection,
    #      one drop per `shed_drop_mass` of collected water. The divisor stays
    #      configurable rather than hardcoded: the rain-number limiter and the
    #      homogeneous-freezing residual both budget this source as
    #      `cloud_warm_collection / shed_drop_mass`, and a literal would disagree
    #      with them for any non-default drop mass. Only in the non-liquid-fraction
    #      path; when liquid fraction is active, collected mass goes to qʷⁱ, not rain.
    # wet_growth_shedding_number → rain drops from excess wet growth
    cloud_warm_rain_n = ifelse(parameters.liquid_fraction_active, zero(FT),
                               rates.cloud_warm_collection / parameters.shed_drop_mass)
    n_gain = n_from_autoconv + n_from_melt +
             rates.rain_breakup +
             rates.shedding_number +
             cloud_warm_rain_n +
             rates.wet_growth_shedding_number
    # Losses (all positive magnitudes, M7)
    # rain_warm_collection_number → rain number sink from above-freezing rain
    #      collection
    n_loss = n_from_evap +
             rates.rain_self_collection +
             rates.rain_riming_number +
             rates.rain_freezing_number +
             rates.rain_homogeneous_number +
             rates.rain_warm_collection_number

    # DSD number correction feedback (the rain PSD diagnosis writes back a bounded nʳ)
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
    # Phase 1: deposition, melting (both partial and complete reduce ice mass)
    # Phase 2: riming (cloud + rain), refreezing, nucleation, and freezing.
    # Splintering mass is already part of the riming mass (splinters fragment existing rime),
    # so it is not added separately to the total ice mass tendency.
    #
    # Wet growth contributes nothing here in either branch. With liquid fraction
    # active, `wet_growth_cloud`/`wet_growth_rain` raise the total ice mass and the
    # coating mass qʷⁱ by the same amount, so the dry ice mass qⁱ is unchanged.
    # Without liquid fraction the collection retained against the wet-growth capacity
    # is already carried by the reduced `cloud_riming`/`rain_riming`
    # (process_rates.jl:466-469), so adding it again would double count.
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass
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
- Global number limiter (C3)
- Ice λ-limiter correction (the tabulated nⁱ bounds write-back)
"""
@inline function tendency_ρnⁱ(rates::P3ProcessRates, ρ)
    # Gains from nucleation, freezing, splintering, homogeneous freezing
    gain = rates.nucleation_number + rates.cloud_freezing_number +
           rates.rain_freezing_number + rates.splintering_number +
           rates.cloud_homogeneous_number + rates.rain_homogeneous_number
    # Losses (all positive magnitudes, M7)
    # sublimation_number — ice number loss from sublimation
    # ni_limit: C3 global Nⁱ cap; relaxation sink above Nⁱ_max/ρ.
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
    # Phase 2: gains from riming, refreezing, freezing, and homogeneous freezing
    # Frozen cloud/rain becomes fully rimed ice (100% rime fraction for new frozen particles)
    #
    # Wet growth contributes no rime mass in either branch: the water it collects
    # stays liquid rather than becoming rime, and the dry-branch retained collection
    # already arrives through `cloud_riming`/`rain_riming`. The dry-branch soaking
    # densification is carried separately by `wet_growth_densification_mass`.
    gain = rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass +
           rates.wet_growth_densification_mass
    # Ordinary melting and sublimation remove the beginning-of-stage rime
    # fraction. Whole-particle clipping instead drains the explicitly
    # reconstructed residual rime companion, including post-process changes.
    sublimation = max(0, -rates.deposition)
    ordinary_complete_melting =
        max(0, rates.complete_melting - rates.clipping_dry_mass)
    # Splintering fragments existing rime rather than creating or destroying it, so it
    # nets to zero here. Since cloud_riming and rain_riming are the full (unreduced)
    # rates, no splintering subtraction is needed.
    loss = Fᶠ * (rates.partial_melting + ordinary_complete_melting + sublimation) +
           rates.clipping_rime_mass
    return ρ * (gain - loss)
end

"""
$(TYPEDSIGNATURES)

Compute rime volume tendency from P3 process rates.

Rime volume changes with rime mass: ∂bᶠ/∂t = ∂qᶠ/∂t / ρ_rime.
Includes sublimation loss (M8): sublimation removes rime volume proportionally.
Includes melt-densification: during melting, low-density rime portions melt
preferentially, driving the remaining rime toward the configured solid-ice density.
"""
@inline function tendency_ρbᶠ(rates::P3ProcessRates, ρ, Fᶠ, ρᶠ, qⁱ, parameters)
    FT = typeof(ρ)

    ρᶠ_safe = max(ρᶠ, parameters.minimum_rime_density)
    ρ_rim_new_safe = max(rates.rime_density_new, parameters.minimum_rime_density)

    # Rain rime and freezing deposit at the maximum rime density
    ρ_rimemax = parameters.maximum_rime_density
    # Homogeneous freezing likewise deposits at the maximum rime density, not at
    # the solid-ice density.
    ρ_rim_hom = parameters.maximum_rime_density

    # Phase 2: Volume gain from new rime
    # Cloud riming uses the Cober-List computed density; rain riming, immersion
    # freezing, and refreezing all deposit at the maximum rime density rather than
    # at the water density.
    #
    # Wet growth adds no rime volume directly. In the dry branch the retained
    # collection is already inside `cloud_riming`/`rain_riming` above, carrying the
    # same fresh-rime / maximum-rime-density split, and the soaking densification
    # comes through `wet_growth_densification_volume`.
    volume_gain = rates.cloud_riming / ρ_rim_new_safe +
                   rates.rain_riming / ρ_rimemax +
                   rates.refreezing / ρ_rimemax +
                   (rates.cloud_freezing_mass + rates.rain_freezing_mass) / ρ_rimemax +
                   (rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass) / ρ_rim_hom +
                   rates.wet_growth_densification_volume

    # Ordinary melting and sublimation remove volume proportionally. A whole-
    # particle clip uses the reconstructed companion volume so post-process rime
    # and densification changes are removed exactly.
    sublimation = max(0, -rates.deposition)
    ordinary_complete_melting =
        max(0, rates.complete_melting - rates.clipping_dry_mass)
    ordinary_total_melting = rates.partial_melting + ordinary_complete_melting
    volume_loss = Fᶠ * (ordinary_total_melting + sublimation) / ρᶠ_safe +
                  rates.clipping_rime_volume

    # Melt-densification. Low-density rime portions melt first, so the remaining ice
    # approaches ρ_solid_ice. In tendency form: additional volume reduction =
    # bᶠ × (ρ_solid_ice - ρᶠ) × |melt| / (ρᶠ × qⁱ).
    # It is skipped when liquid fraction is active, because the meltwater is then
    # tracked explicitly in qʷⁱ. The densification target is the configured solid-ice
    # density, not the maximum rime density.
    ρ_solid_ice = parameters.pure_ice_density
    qⁱ_safe = max(qⁱ, FT(parameters.floors.mass_scale))
    bᶠ = Fᶠ * qⁱ_safe / ρᶠ_safe
    densification = bᶠ * (ρ_solid_ice - ρᶠ_safe) * ordinary_total_melting /
                    (ρᶠ_safe * qⁱ_safe)
    # Apply only below the solid-ice density when liquid fraction is not active.
    apply_densification = (ρᶠ_safe < ρ_solid_ice) & !parameters.liquid_fraction_active
    densification = ifelse(apply_densification, densification, zero(FT))

    return ρ * (volume_gain - volume_loss - densification)
end

"""
$(TYPEDSIGNATURES)

Compute cloud-number tendency from P3 process rates.

Activation creates new cloud droplets. Autoconversion, accretion, riming,
freezing, and above-freezing collection remove cloud droplets in proportion
to the cloud mass they consume.
"""
@inline function tendency_ρnᶜˡ(rates::P3ProcessRates, ρ, Nᶜˡ, qᶜˡ, p3)
    FT = typeof(ρ)
    parameters = p3.process_rates
    number_per_mass = cloud_number_per_cloud_mass(Nᶜˡ, ρ, qᶜˡ)
    seed_drop_mass = activated_droplet_mass(parameters, FT)
    activation_number = ifelse(iszero(rates.ccn_activation_number),
                               rates.ccn_activation_mass / seed_drop_mass,
                               rates.ccn_activation_number)

    # Scheme-aware cloud-number loss from autoconversion. KK2000 scales by the
    # in-cloud nᶜˡ/qᶜˡ ratio.
    autoconv_n = cloud_number_loss_from_autoconversion(p3, rates.autoconversion,
    qᶜˡ, Nᶜˡ, ρ)

    number_loss = autoconv_n +
                  number_per_mass * rates.accretion +
                  rates.cloud_self_collection +
                  rates.cloud_riming_number +
                  rates.cloud_freezing_number +
                  rates.cloud_homogeneous_number +
                  rates.cloud_warm_collection_number

    # DSD number correction feedback (the cloud PSD diagnosis writes back a bounded Nᶜˡ)
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

@inline function tendency_ρqʷⁱ(rates::P3ProcessRates, ρ, parameters::Union{Nothing, ProcessRateParameters})
    # Include condensation onto and evaporation from the liquid coating.
    # wet_growth_shedding diverts excess wet growth mass from qʷⁱ to rain.
    # Note: rain_warm_collection is zeroed at rate-assembly time in the non-liquid-
    # fraction branch, which transfers no rain mass to qʷⁱ, so it can safely be
    # added here unconditionally.
    liquid_fraction_active = liquid_fraction_routing_active(parameters)
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

Compute the liquid supersaturation tendency from Grabowski & Morrison (2008).

When `predict_supersaturation = true`, the liquid supersaturation ``sᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ``
is a prognostic variable advected by the dynamical core. The microphysical
tendency reproduces the post-step diagnosis ``sᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ(T)``.
`compute_p3_process_rates` precomputes that diagnostic tendency from the final
local ``qᵛ`` and ``T`` implied by the ordered process rates.

When `predict_supersaturation = false`, returns zero tendency.
"""
@inline function tendency_ρsᵛ⁺ˡ(rates::P3ProcessRates, ρ, parameters)
    raw = ρ * rates.predicted_supersaturation_tendency
    return gate_predicted_supersaturation(parameters, raw)
end

"""
$(TYPEDSIGNATURES)

Aerosol-pool tendency: each activated cloud droplet removes one unit from the
unactivated reservoir, so ``∂ρn^a/∂t = -ρ \\, n_{\\text{nuc}}`` with
``n_{\\text{nuc}}`` the same activation rate that sources ``ρn^{cl}``. In the
prescribed-Nᶜˡ path `rates.ccn_activation_number` is zero, so this returns 0.
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
@inline tendency_ρnᶜˡ(::Nothing, ρ, Nᶜˡ, qᶜˡ, p3) = zero(ρ)
@inline tendency_ρnʳ(::Nothing, ρ, args...) = zero(ρ)
@inline tendency_ρqⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᶠ(::Nothing, ρ, Fᶠ) = zero(ρ)
@inline tendency_ρbᶠ(::Nothing, ρ, Fᶠ, ρᶠ, parameters...) = zero(ρ)
@inline tendency_ρqʷⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρsᵛ⁺ˡ(::Nothing, ρ, parameters) = zero(ρ)
@inline tendency_ρqᵛ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnᵃ(::Nothing, ρ) = zero(ρ)
