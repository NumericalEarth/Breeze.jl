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
##### Combined P3 tendency calculation
#####

# Derived thermodynamic and PSD state computed during setup of `compute_p3_process_rates`
# and passed to the phase sub-functions to avoid recomputation.
# Internal implementation detail — not part of the public API.
struct P3DerivedState{FT, Q, L}
    # Bounded prognostic state
    nⁱ :: FT        # bounded by maximum_ice_number_density / ρ
    nʳ :: FT        # DSD-bounded rain number
    qᶠ :: FT        # consistent rime mass
    bᶠ :: FT        # consistent rime volume
    Fᶠ :: FT        # rime fraction
    ρᶠ :: FT        # rime density
    # PSD parameters
    Fˡ :: FT        # liquid fraction on ice
    Nᶜˡ :: FT       # effective cloud droplet number concentration
    nᶜˡ :: FT       # DSD-bounded cloud number (for correction)
    μᶜˡ :: FT       # local cloud DSD shape parameter
    λᶜˡ :: FT       # local cloud DSD slope parameter
    # Thermodynamic state
    T :: FT         # temperature [K]
    P :: FT         # pressure [Pa]
    qᵛ :: FT        # vapor mass fraction
    qᵛ⁺ˡ :: FT      # saturation vapor fraction over liquid
    qᵛ⁺ⁱ :: FT      # saturation vapor fraction over ice
    q :: Q          # MoistureMassFractions for heat capacity / density
    # Transport properties
    Dᵛ :: FT       # water vapor diffusivity [m²/s]
    Kᵃ :: FT       # thermal conductivity of air [W/m/K]
    ν :: FT        # kinematic viscosity [m²/s]
    # Table-1 quantities of the bounded ice population, bracketed once per cell
    lookups :: L   # P3IceLookups: coordinate bracket, density correction, ventilation terms
end

@inline function liquid_supersaturation_after_moisture_update(𝒰, qᵛ, qˡ, qⁱ, ρ, constants)
    q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
    𝒰₁ = with_moisture(𝒰, q)
    T = temperature(𝒰₁, constants)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return qᵛ - qᵛ⁺ˡ
end

@inline function final_predicted_supersaturation_tendency(
    ::ProcessRate{FT, false}, 𝒰, qᵛ, qˡ, qⁱ, ρ, constants,
    initial_supersaturation, dt, vapor_to_liquid, vapor_to_ice, liquid_to_ice
) where FT
    return zero(qᵛ)
end

@inline function final_predicted_supersaturation_tendency(
    ::ProcessRate{FT, true}, 𝒰, qᵛ, qˡ, qⁱ, ρ, constants,
    initial_supersaturation, dt, vapor_to_liquid, vapor_to_ice, liquid_to_ice
) where FT
    qᵛ_final = qᵛ - (vapor_to_liquid + vapor_to_ice) * dt
    qˡ_final = qˡ + (vapor_to_liquid - liquid_to_ice) * dt
    qⁱ_final = qⁱ + (vapor_to_ice + liquid_to_ice) * dt
    final_supersaturation = liquid_supersaturation_after_moisture_update(
        𝒰, qᵛ_final, qˡ_final, qⁱ_final, ρ, constants)
    return (final_supersaturation - initial_supersaturation) / dt
end

# Phase 1 process rates: condensation, rain, deposition, and melting.
# Returned by `p3_phase1_rates`. Internal implementation detail.
struct P3Phase1Rates{FT}
    condensation :: FT
    ccn_activation_mass :: FT
    ccn_activation_number :: FT
    autoconversion :: FT
    accretion :: FT
    cloud_self_collection :: FT
    rain_evaporation :: FT
    rain_condensation :: FT
    rain_self_collection :: FT
    rain_breakup :: FT
    deposition :: FT
    coating_condensation :: FT
    coating_evaporation :: FT
    partial_melting :: FT
    complete_melting :: FT
    melting_number :: FT
end

# Phase 2 process rates: aggregation, riming, wet growth, shedding, nucleation, and warm
# collection. Homogeneous freezing is diagnosed separately in `compute_p3_process_rates`
# from the post-process liquid residual. Returned by `p3_phase2_rates`. Internal
# implementation detail.
struct P3Phase2Rates{FT}
    aggregation :: FT
    ni_limit :: FT
    cloud_riming :: FT
    cloud_riming_number :: FT
    rain_riming :: FT
    rain_riming_number :: FT
    rime_density_new :: FT
    wet_growth_cloud :: FT
    wet_growth_rain :: FT
    wet_growth_shedding :: FT
    wet_growth_shedding_number :: FT
    wet_growth_densification_mass :: FT
    wet_growth_densification_volume :: FT
    shedding :: FT
    shedding_number :: FT
    refreezing :: FT
    complete_melting :: FT  # Phase 1 value + M8/M12c clipping
    melting_number :: FT
    whole_particle_clipping :: Bool
    nucleation_mass :: FT
    nucleation_number :: FT
    cloud_freezing_mass :: FT
    cloud_freezing_number :: FT
    rain_freezing_mass :: FT
    rain_freezing_number :: FT
    cloud_warm_collection :: FT
    cloud_warm_collection_number :: FT
    rain_warm_collection :: FT
    rain_warm_collection_number :: FT
end

# Container for the computed P3 process rates: Phase 1 (rain, deposition, melting) and
# Phase 2 (aggregation, riming, shedding, nucleation).
#
# Following Milbrandt et al. (2025), melting is partitioned:
#   - `partial_melting`:  meltwater stays on the ice as liquid coating (large particles)
#   - `complete_melting`: meltwater sheds to rain (small particles)
#
# Sign convention (M7): all one-directional rates store positive magnitudes. Bidirectional
# rates (condensation, deposition) are positive for a source and negative for a sink.
# Signs are applied explicitly in the `tendency_*` functions.
struct P3ProcessRates{FT}
    # Phase 1: Cloud condensation/evaporation (BIDIRECTIONAL: +cond / −evap)
    condensation :: FT             # Vapor ↔ cloud liquid [kg/kg/s] (+cond, −evap)

    # Phase 1: Rain tendencies (all positive magnitudes)
    autoconversion :: FT           # Cloud → rain mass [kg/kg/s]
    accretion :: FT                # Cloud → rain mass (via rain sweep-out) [kg/kg/s]
    cloud_self_collection :: FT    # Cloud number loss from cloud-cloud coalescence [1/kg/s] (0 for KK2000)
    rain_evaporation :: FT         # Rain evaporation magnitude [kg/kg/s]
    rain_evaporation_number :: FT  # Rain number loss from evaporation [1/kg/s]
    # The self-collection/breakup pair is netted (only one is nonzero) so that the
    # rain-number limiter sees a single signed rate (`compute_p3_process_rates`).
    rain_self_collection :: FT     # Net rain number loss from self-collection [1/kg/s]
    rain_breakup :: FT             # Net rain number gain from breakup [1/kg/s]

    # Phase 1: Ice tendencies (BIDIRECTIONAL deposition; positive melting/number)
    deposition :: FT               # Vapor ↔ ice mass [kg/kg/s] (+dep, −sublim)
    partial_melting :: FT          # Ice → liquid coating (stays on ice) [kg/kg/s]
    complete_melting :: FT         # Ice → rain mass (sheds) [kg/kg/s]
    melting_number :: FT           # Ice number loss magnitude from melting [1/kg/s]
    clipping_dry_mass :: FT        # Whole-particle clip contribution to complete melting [kg/kg/s]
    clipping_rime_mass :: FT       # Rime mass removed exactly by whole-particle clips [kg/kg/s]
    clipping_rime_volume :: FT     # Rime volume removed exactly by whole-particle clips [m³/kg/s]
    post_process_clipping :: FT    # One when the post-process liquid-fraction clip fires

    # D2/D1: Ice number loss from vapor-driven sinks (sublimation + coating evaporation)
    sublimation_number :: FT       # Ice number loss magnitude from sublimation / coating evaporation [1/kg/s]

    # Phase 2: Ice aggregation (positive magnitude)
    aggregation :: FT              # Ice number loss magnitude from self-collection [1/kg/s]

    # Global ice number limiter (positive magnitude)
    ni_limit :: FT                 # Ice number excess removal rate [1/kg/s]

    # Phase 2: Riming (all positive magnitudes)
    cloud_riming :: FT             # Cloud → ice via riming [kg/kg/s]
    cloud_riming_number :: FT      # Cloud number loss magnitude [1/kg/s]
    rain_riming :: FT              # Rain → ice via riming [kg/kg/s]
    rain_riming_number :: FT       # Rain number loss magnitude [1/kg/s]
    rime_density_new :: FT         # Density of new rime [kg/m³]

    # Phase 2: Shedding and refreezing
    shedding :: FT                 # Liquid on ice → rain [kg/kg/s]
    shedding_number :: FT          # Rain number from shedding [1/kg/s]
    refreezing :: FT               # Liquid on ice → rime [kg/kg/s]

    # Ice nucleation (deposition + immersion freezing)
    nucleation_mass :: FT          # New ice mass from deposition nucleation [kg/kg/s]
    nucleation_number :: FT        # New ice number from deposition nucleation [1/kg/s]
    cloud_freezing_mass :: FT      # Cloud → ice mass from immersion freezing [kg/kg/s]
    cloud_freezing_number :: FT    # Cloud number to ice number [1/kg/s]
    rain_freezing_mass :: FT       # Rain → ice mass from immersion freezing [kg/kg/s]
    rain_freezing_number :: FT     # Rain number to ice number [1/kg/s]

    # Rime splintering (Hallett-Mossop)
    splintering_mass :: FT         # New ice mass from splintering [kg/kg/s]
    splintering_number :: FT       # New ice number from splintering [1/kg/s]

    # Homogeneous freezing (T < -40°C, instantaneous)
    cloud_homogeneous_mass :: FT   # Cloud → ice from homogeneous freezing [kg/kg/s]
    cloud_homogeneous_number :: FT # Cloud number → ice [1/kg/s]
    rain_homogeneous_mass :: FT    # Rain → ice from homogeneous freezing [kg/kg/s]
    rain_homogeneous_number :: FT  # Rain number → ice [1/kg/s]

    # Above-freezing collection (T > T₀). Cloud collection goes to qʷⁱ in
    # liquid-fraction mode and sheds to rain otherwise; rain mass transfers only to qʷⁱ.
    cloud_warm_collection :: FT        # Cloud collected above T₀ [kg/kg/s]
    cloud_warm_collection_number :: FT # Cloud number loss from warm collection [1/kg/s]
    rain_warm_collection :: FT         # Rain collected above T₀ → qʷⁱ [kg/kg/s]
    rain_warm_collection_number :: FT  # M9: Rain number loss from warm collection [1/kg/s]

    # Liquid-fraction wet growth: collected hydrometeors redirected to qʷⁱ when
    # collection exceeds freezing capacity.
    wet_growth_cloud :: FT             # Cloud collection redirected to qʷⁱ [kg/kg/s]
    wet_growth_rain :: FT              # Rain collection redirected to qʷⁱ [kg/kg/s]

    # Non-liquid-fraction wet growth shedding. Only excess cloud water is a new
    # rain-mass source; number includes excess cloud and recycled rain collection.
    wet_growth_shedding :: FT          # Excess cloud collection → rain mass [kg/kg/s]
    wet_growth_shedding_number :: FT   # Rain number from wet growth shedding [1/kg/s]

    # Warm/mixed-phase budget terms
    ccn_activation_mass :: FT          # CCN activation mass rate (vapor → cloud) [kg/kg/s]
    ccn_activation_number :: FT        # CCN activation number rate [1/kg/s] (prognostic CCN only)
    rain_condensation :: FT            # Rain condensation (vapor → rain) [kg/kg/s]
    coating_condensation :: FT         # Condensation on ice liquid coating [kg/kg/s]
    coating_evaporation :: FT          # Evaporation from ice liquid coating [kg/kg/s]

    # Wet growth rime densification
    # During wet growth, assume total soaking: qᶠ → qⁱ, bᶠ → qⁱ/ρ_rimeMax.
    wet_growth_densification_mass :: FT   # Rime mass source: (qⁱ - qᶠ)/τ [kg/kg/s]
    wet_growth_densification_volume :: FT # Rime volume change: (qⁱ/ρ_max - bᶠ)/τ [m³/kg/s]

    # DSD number correction feedback.
    # After lambda bounding, the DSD-consistent number may differ from the prognostic
    # number. The correction is expressed as a relaxation rate over dt_safety rather
    # than as an instantaneous write-back.
    cloud_number_correction :: FT  # (nᶜˡ_bounded - nᶜˡ) / τ [1/kg/s]
    rain_number_correction :: FT   # (nʳ_bounded - nʳ) / τ [1/kg/s]
    ice_number_correction :: FT    # (nⁱ_lambda_bounded - nⁱ_global_bounded) / τ [1/kg/s]

    # G&M (2008) bounded supersaturation adjustment, also folded into
    # `condensation` so vapor and cloud tendencies include it automatically.
    # Carried separately so callers/tests can inspect the G&M contribution.
    # Sized as `ε / sink_limiting_timescale`, so dt = sink_limiting_timescale
    # integrates the one-shot adjustment exactly (see
    # `predicted_supersaturation_adjustment`).
    predicted_supersaturation_adjustment :: FT
    # End-of-step supersaturation recompute: (qᵛ_final - qᵛ⁺ˡ(T_final) - sᵛ⁺ˡ_initial) / τ.
    # Tied to the same dt = τ assumption.
    predicted_supersaturation_tendency :: FT
end

@inline function p3_phase1_rates(p3, ρ, ℳ, constants, state::P3DerivedState,
                                 temperature_tendency, vapor_tendency)
    FT = typeof(ρ)

    # Unpack derived state (field access on concrete struct — GPU-safe)
    T = state.T
    qᵛ = state.qᵛ
    qᵛ⁺ˡ = state.qᵛ⁺ˡ
    qᵛ⁺ⁱ = state.qᵛ⁺ⁱ
    q = state.q
    Fᶠ = state.Fᶠ
    ρᶠ = state.ρᶠ
    Nᶜˡ = state.Nᶜˡ
    nⁱ = state.nⁱ
    nʳ = state.nʳ
    P = state.P

    # Transport properties (reconstructed as NamedTuple for existing function signatures)
    transport = (; Dᵛ = state.Dᵛ, Kᵃ = state.Kᵃ, ν = state.ν)
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)

    # =========================================================================
    # Coupled cloud/rain/ice vapor growth and decay
    # =========================================================================
    vapor_rates = coupled_saturation_adjustment_rates(p3, ℳ.qᶜˡ, ℳ.qʳ, nʳ,
                                                      ℳ.qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ,
                                                      Fᶠ, ρᶠ, T, P, ρ, constants,
                                                      transport, q,
                                                      state.μᶜˡ, state.λᶜˡ, state.nᶜˡ,
                                                      temperature_tendency, vapor_tendency,
                                                      state.lookups)
    cond = vapor_rates.condensation

    # CCN activation (prescribed or prognostic; depletes ℳ.nᵃ when prognostic)
    ccn = compute_ccn_activation(p3.aerosol, p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.nᵃ,
                                 qᵛ, qᵛ⁺ˡ, T, ρ, constants)
    ccn_activation_mass = ccn.mass
    ccn_activation_number = ccn.number

    # =========================================================================
    # Rain processes
    # =========================================================================
    autoconv = rain_autoconversion_rate(p3, ℳ.qᶜˡ, Nᶜˡ, ρ, ℳ.qʳ)
    accr = rain_accretion_rate(p3, ℳ.qᶜˡ, ℳ.qʳ, ρ)
    cloud_self = cloud_self_collection_rate(p3, ℳ.qᶜˡ, Nᶜˡ, ρ)
    rain_evap = vapor_rates.rain_evaporation
    rain_cond = vapor_rates.rain_condensation
    rain_self = rain_self_collection_rate(p3, ℳ.qʳ, nʳ, ρ)
    rain_br = rain_breakup_rate(p3, ℳ.qʳ, nʳ, rain_self)

    # =========================================================================
    # Ice deposition/sublimation and wet-ice coating condensation/evaporation
    # =========================================================================
    # Both dry-ice deposition and wet-coating vapor exchange are gated on the total
    # ice reservoir qⁱ + qʷⁱ, which includes the liquid coating. Gating on qⁱ alone
    # would incorrectly disable vapor exchange for nearly melted, liquid-coated
    # particles.
    has_total_ice = total_ice_mass(ℳ.qⁱ, qʷⁱ) >= p3.minimum_mass_mixing_ratio
    dep = ifelse(has_total_ice, vapor_rates.deposition, zero(FT))

    liquid_fraction_active = p3.process_rates.liquid_fraction_active
    wet_ice_exchange_active = has_total_ice & liquid_fraction_active
    coat_cond = ifelse(wet_ice_exchange_active,
                       vapor_rates.coating_condensation, zero(FT))
    coat_evap = ifelse(wet_ice_exchange_active,
                       vapor_rates.coating_evaporation, zero(FT))

    melt_rates = ice_melting_rates(p3, ℳ.qⁱ, nⁱ, T, qᵛ, ρ, constants, transport, state.lookups)
    partial_melt = melt_rates.partial_melting
    complete_melt = melt_rates.complete_melting
    complete_melt = ifelse(p3.process_rates.liquid_fraction_active,
                           complete_melt, complete_melt + partial_melt)
    partial_melt = ifelse(p3.process_rates.liquid_fraction_active,
                          partial_melt, zero(FT))
    melt_n = ice_melting_number_rate(ℳ.qⁱ, nⁱ, complete_melt)

    return P3Phase1Rates{FT}(cond, ccn_activation_mass, ccn_activation_number,
                             autoconv, accr, cloud_self,
                             rain_evap, rain_cond, rain_self, rain_br,
                             dep, coat_cond, coat_evap,
                             partial_melt, complete_melt, melt_n)
end

@inline function p3_phase2_rates(p3, ρ, ℳ, constants, state::P3DerivedState,
                                 phase1::P3Phase1Rates)
    FT = typeof(ρ)
    parameters = p3.process_rates
    T₀ = parameters.freezing_temperature

    # Unpack derived state
    (; T, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, qᶠ, bᶠ, Fˡ, Nᶜˡ, μᶜˡ, λᶜˡ, nⁱ, nʳ, lookups) = state
    transport = (; Dᵛ = state.Dᵛ, Kᵃ = state.Kᵃ, ν = state.ν)

    qⁱ = ℳ.qⁱ
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ

    # =========================================================================
    # Aggregation
    # =========================================================================
    agg = ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, qʷⁱ, lookups)

    # Global ice number limiter, expressed as a tendency on the *raw* prognostic
    # ℳ.nⁱ rather than on the locally pre-capped `state.nⁱ`, which is already
    # bounded and would make this limiter dead.
    maximum_ice_number = parameters.maximum_ice_number_density
    ni_lim = max(0, ℳ.nⁱ - maximum_ice_number / ρ) / parameters.sink_limiting_timescale

    # =========================================================================
    # Riming
    # =========================================================================
    # Cloud and rain collection by ice run the same kernels below and above freezing;
    # only the destination of the collected water differs. Each kernel is therefore
    # evaluated once with the gate open and split by temperature here, rather than
    # called twice with complementary gates.
    below_freezing = T <= T₀
    cloud_coll = cloud_collection_mass_rate(p3, qᶜˡ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ, true, qʷⁱ, lookups)
    cloud_rim = ifelse(below_freezing, cloud_coll, zero(FT))
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nᶜˡ, ρ, cloud_rim)
    # Mass and number share one Table 2 read.
    rain_coll_q, rain_coll_n = rain_collection_rates(p3, qʳ, nʳ, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ,
                                                     true, qʷⁱ, lookups)
    rain_rim = ifelse(below_freezing, rain_coll_q, zero(FT))
    rain_rim_n = ifelse(below_freezing, rain_coll_n, zero(FT))

    # Rime density
    # The rime density formula is indexed with the locally diagnosed cloud DSD, not
    # with prescribed cloud parameters, so μᶜˡ and λᶜˡ from `diagnose_cloud_dsd` are
    # passed through to the Cober-List rime density when Nᶜˡ is prognostic.
    # The fall speed is read at the shared Table-1 bracket, which is indexed with the
    # total ice mass.
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    vᵢ = ice_terminal_velocity_mass_weighted(p3, qⁱ_total, lookups)
    ρᶠ_new = rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport, μᶜˡ, λᶜˡ)

    # =========================================================================
    # Wet growth capacity and collection rerouting
    # =========================================================================
    has_hydrometeors = (max(0, qᶜˡ) + max(0, qʳ)) >=
                       parameters.wet_growth_hydrometeor_threshold
    qwgrth_raw = wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, qᵛ, Fᶠ, ρᶠ, ρ,
                                     constants, transport, lookups)
    qwgrth = ifelse(has_hydrometeors, qwgrth_raw, zero(FT))

    total_collection = cloud_rim + rain_rim
    is_wet_growth = has_hydrometeors &
                    (total_collection >= qwgrth + parameters.wet_growth_excess_threshold)

    liquid_fraction_wet_growth = parameters.liquid_fraction_active & is_wet_growth
    dry_wet_growth = !parameters.liquid_fraction_active & is_wet_growth
    retained_fraction = clamp(safe_divide(qwgrth, total_collection, zero(FT)), 0, 1)
    retained_cloud = cloud_rim * retained_fraction
    retained_rain = rain_rim * retained_fraction
    excess_cloud = cloud_rim - retained_cloud
    excess_rain = rain_rim - retained_rain

    # With prognostic liquid fraction, all collection becomes liquid coating. Without
    # it, the freezing-capacity portion becomes dense rime while only excess cloud
    # water is a new rain-mass source; excess collected rain simply returns to rain.
    wg_cloud = liquid_fraction_wet_growth * cloud_rim
    wg_rain = liquid_fraction_wet_growth * rain_rim
    cloud_rim = ifelse(liquid_fraction_wet_growth, zero(FT),
                       ifelse(dry_wet_growth, retained_cloud, cloud_rim))
    rain_rim = ifelse(liquid_fraction_wet_growth, zero(FT),
                      ifelse(dry_wet_growth, retained_rain, rain_rim))
    wg_shed = ifelse(dry_wet_growth, excess_cloud, zero(FT))
    wg_shed_n = ifelse(dry_wet_growth,
                       (excess_cloud + excess_rain) / parameters.shed_drop_mass,
                       zero(FT))
    ρᶠ_new = ifelse(dry_wet_growth, parameters.maximum_rime_density, ρᶠ_new)

    # Wet growth rime densification
    ρ_rimemax = parameters.maximum_rime_density
    τ_densif = parameters.rime_densification_timescale
    qⁱ_safe = max(0, qⁱ)
    bᶠ_safe = max(bᶠ, FT(parameters.floors.mass_scale))
    wg_densif_active = dry_wet_growth & (qⁱ_safe > p3.minimum_mass_mixing_ratio)
    wg_densif_mass = max(0, qⁱ_safe - qᶠ) / τ_densif
    wg_densif_vol = (qⁱ_safe / ρ_rimemax - bᶠ_safe) / τ_densif
    wg_densif_mass = ifelse(wg_densif_active, wg_densif_mass, zero(FT))
    wg_densif_vol  = ifelse(wg_densif_active, wg_densif_vol, zero(FT))

    # =========================================================================
    # Shedding and refreezing
    # =========================================================================
    shed = shedding_rate(p3, qʷⁱ, nⁱ, Fᶠ, Fˡ, lookups)
    shed_n = shedding_number_rate(p3, shed)
    refrz = refreezing_rate_from_capacity(p3, qʷⁱ, qwgrth_raw)
    shed = ifelse(parameters.liquid_fraction_active, shed, zero(FT))
    shed_n = ifelse(parameters.liquid_fraction_active, shed_n, zero(FT))
    refrz = ifelse(parameters.liquid_fraction_active, refrz, zero(FT))

    # Liquid fraction clipping
    Fl_small = parameters.liquid_fraction_clipping_threshold
    τ_clip = parameters.refreezing_timescale
    qʷⁱ_eff = max(0, qʷⁱ)
    clip_freeze = parameters.liquid_fraction_active & (T < T₀) & (Fˡ < Fl_small) & (Fˡ > 0)
    refrz = ifelse(clip_freeze, refrz + qʷⁱ_eff / τ_clip, refrz)

    # Whole-particle liquid-fraction and tiny-warm-ice clips. These predicates can
    # overlap, so form their union and transfer each reservoir exactly once.
    qⁱ_dry = max(0, qⁱ)
    qⁱ_total_clip = qⁱ_dry + qʷⁱ_eff
    has_clip_mass = qⁱ_total_clip >= p3.minimum_mass_mixing_ratio
    warm_liquid_clip = (T >= T₀) & (Fˡ > 1 - Fl_small) & has_clip_mass
    high_liquid_fraction_clip = (Fˡ > parameters.complete_melting_liquid_fraction) & has_clip_mass
    tiny_warm_ice = (T >= T₀) & has_clip_mass &
                    (qⁱ_total_clip < parameters.tiny_ice_to_rain_threshold)
    liquid_fraction_clipping = parameters.liquid_fraction_active &
                               (warm_liquid_clip | high_liquid_fraction_clip)
    whole_particle_clipping = liquid_fraction_clipping | tiny_warm_ice
    complete_melt = ifelse(whole_particle_clipping, qⁱ_dry / τ_clip,
                           phase1.complete_melting)
    melt_n = ifelse(whole_particle_clipping,
                    max(0, ℳ.nⁱ) / τ_clip, phase1.melting_number)
    shed = ifelse(whole_particle_clipping, qʷⁱ_eff / τ_clip, shed)
    shed_n = ifelse(whole_particle_clipping, zero(FT), shed_n)
    refrz = ifelse(whole_particle_clipping, zero(FT), refrz)

    # =========================================================================
    # Ice nucleation
    # =========================================================================
    nucleation_existing_number = ifelse(whole_particle_clipping, zero(FT), nⁱ)
    nuc_q, nuc_n = deposition_nucleation_rate(
        p3, T, qᵛ, qᵛ⁺ⁱ, nucleation_existing_number, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜˡ, T, ρ)
    μʳ = zero(FT)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T, μʳ)

    # Rime splintering is not diagnosed here: `compute_p3_process_rates` recomputes it
    # from the sink-limited riming rates, which is the value every consumer sees.

    # Homogeneous freezing is diagnosed later from the post-process liquid residual
    # (see `compute_p3_process_rates`), so it is not computed here.

    # Above-freezing collection: the complementary halves of the two kernels
    # already evaluated above.
    cloud_warm_q = ifelse(below_freezing, zero(FT), cloud_coll)
    cloud_warm_n = cloud_riming_number_rate(qᶜˡ, Nᶜˡ, ρ, cloud_warm_q)
    # Number sink from above-freezing rain collection fires in both branches.
    rain_warm_q_full = ifelse(below_freezing, zero(FT), rain_coll_q)
    rain_warm_n = ifelse(below_freezing, zero(FT), rain_coll_n)
    # Mass transfer of collected rain into qʷⁱ only happens in the liquid-fraction
    # branch. In the non-liquid path, collection of rain above freezing does not
    # impact total rain mass, so zero out rain_warm_q in that case.
    rain_warm_q = ifelse(parameters.liquid_fraction_active, rain_warm_q_full, zero(FT))

    return P3Phase2Rates{FT}(
        agg, ni_lim,
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        wg_cloud, wg_rain, wg_shed, wg_shed_n, wg_densif_mass, wg_densif_vol,
        shed, shed_n, refrz, complete_melt, melt_n, whole_particle_clipping,
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        cloud_warm_q, cloud_warm_n, rain_warm_q, rain_warm_n
    )
end

"""
$(TYPEDSIGNATURES)

Compute all P3 process rates (Phase 1 and Phase 2) from a microphysical state.

This is the gridless version that accepts a `P3MicrophysicalState` directly,
suitable for use in GPU kernels where grid indexing is handled externally.

# Arguments
- `p3`: P3 microphysics scheme
- `ρ`: Air density [kg/m³]
- `ℳ`: P3MicrophysicalState containing all mixing ratios
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

# Returns
- `P3ProcessRates` containing all computed rates
"""
@inline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    surface_temperature = temperature(𝒰, constants)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing,
                                    surface_temperature, zero(ρ), zero(ρ))
end

@inline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties)
    surface_temperature = temperature(𝒰, constants)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                    surface_temperature, zero(ρ), zero(ρ))
end

@inline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                          surface_temperature)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                    surface_temperature, zero(ρ), zero(ρ))
end

# Everything from here down is `@inline`. A `@noinline` device function would receive `p3`
# (several KB of table handles), `ℳ`, `𝒰`, and `constants` as local-memory copies, and
# ptxas allocates registers over the whole call graph anyway.
@inline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                          surface_temperature,
                                          temperature_tendency,
                                          vapor_tendency)
    FT = typeof(ρ)
    parameters = p3.process_rates
    T₀ = parameters.freezing_temperature

    # === SETUP ===
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nʳ = ℳ.nʳ
    qⁱ = ℳ.qⁱ
    nⁱ_raw = ℳ.nⁱ
    qʷⁱ_prognostic = ℳ.qʷⁱ
    qʷⁱ = active_liquid_on_ice(p3, qʷⁱ_prognostic)
    qʷⁱ_budget = ifelse(parameters.liquid_fraction_active, qʷⁱ, max(0, qʷⁱ_prognostic))

    # The globally capped raw number is the baseline the ice-number correction
    # below relaxes towards; the bounded moments come from `p3_ice_moment_bounds`.
    nⁱ_global = min(max(0, nⁱ_raw), parameters.maximum_ice_number_density / ρ)

    rain_active = qʳ > p3.minimum_mass_mixing_ratio
    qʳ_pos = max(0, qʳ)
    nʳ_floored = max(nʳ, p3.minimum_number_mixing_ratio)
    # rain_slope_parameter and consistent_rime_state are pure functions of (ℳ, parameters);
    # when properties is supplied (hot path from p3_tendency_compute / p3_state_tendencies)
    # we reuse the values already computed in p3_process_properties.
    λʳ = isnothing(properties) ? rain_slope_parameter(qʳ_pos, nʳ_floored, parameters) : properties.λʳ
    nʳ = ifelse(rain_active, rain_number_from_slope(qʳ_pos, λʳ, parameters), zero(FT))

    qᶠ, bᶠ, Fᶠ, ρᶠ = if isnothing(properties)
        rs = consistent_rime_state(p3, qⁱ, ℳ.qᶠ, ℳ.bᶠ)
        rs.qᶠ, rs.bᶠ, rs.Fᶠ, rs.ρᶠ
    else
        properties.qᶠ, properties.bᶠ, properties.Fᶠ, properties.ρᶠ
    end

    # The two branches must agree: `properties` carries exactly what `p3_process_properties`
    # derives from the same two helpers, so the fallback goes through them too rather
    # than restating the bounded-moment recipe.
    Fˡ, qⁱ_total, nⁱ, nⁱ_diagnostic, ρ_mean = if isnothing(properties)
        Fˡ_diagnosed = liquid_fraction_on_ice(qⁱ, qʷⁱ, parameters.floors)
        bounds = p3_ice_moment_bounds(p3, ρ, total_ice_mass(qⁱ, qʷⁱ), nⁱ_raw,
                                      Fᶠ, Fˡ_diagnosed, ρᶠ)
        Fˡ_diagnosed, bounds.qⁱ_total, bounds.nⁱ, bounds.nⁱ_diagnostic, bounds.ρ_mean
    else
        properties.Fˡ, properties.qⁱ_total, properties.nⁱ,
        properties.nⁱ_diagnostic, properties.ρ_mean
    end

    T = temperature(𝒰, constants)
    q_base = 𝒰.moisture_mass_fractions
    qᵛ_base = q_base.vapor
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    P = air_pressure(𝒰, constants)

    supersaturation_adjustment = predicted_supersaturation_adjustment(
        p3, qᶜˡ, qᵛ_base, qᵛ⁺ˡ, ℳ.sᵛ⁺ˡ, T, ρ, constants)
    saturation_alignment_rate = supersaturation_adjustment.rate
    qᶜˡ = supersaturation_adjustment.qᶜˡ
    qᵛ = supersaturation_adjustment.qᵛ
    T = supersaturation_adjustment.T
    q = MoistureMassFractions(qᵛ,
                              q_base.liquid + supersaturation_adjustment.cloud_water_adjustment,
                              q_base.ice)
    qᵛ⁺ˡ = supersaturation_adjustment.qᵛ⁺ˡ
    qᵛ⁺ⁱ = p3_ice_saturation_specific_humidity(T, ρ, constants, T₀, qᵛ⁺ˡ)
    transport = air_transport_properties(T, P, constants)

    cloud = diagnose_cloud_dsd(p3, qᶜˡ, ℳ.nᶜˡ, ρ)
    Nᶜˡ = cloud.Nᶜˡ
    ℳ_adjusted = P3MicrophysicalState(qᶜˡ, ℳ.nᶜˡ, qʳ, ℳ.nʳ, qⁱ, ℳ.nⁱ,
                                      qᶠ, bᶠ, qʷⁱ, qᵛ - qᵛ⁺ˡ, ℳ.nᵃ, ℳ.w)

    # One Table-1 bracket for the bounded population, shared by every ice-side read below.
    lookups = p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, ρ)

    # Build derived state struct (explicit type parameters keep the constructor
    # concrete for GPU compilation). The rate functions that need a heat capacity all use
    # the dry-air `cᵖᵈ` psychrometric convention (`liquid_psychrometric_correction` /
    # `ice_psychrometric_correction`), which is a scheme constant rather than a per-cell
    # quantity, so no cᵖᵐ is carried here.
    state = P3DerivedState{FT, typeof(q), typeof(lookups)}(nⁱ, nʳ, qᶠ, bᶠ, Fᶠ, ρᶠ,
                                                           Fˡ, Nᶜˡ, cloud.nᶜˡ,
                                                           cloud.μᶜˡ, cloud.λᶜˡ,
                                                           T, P, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, q,
                                                           transport.Dᵛ, transport.Kᵃ, transport.ν,
                                                           lookups)

    # === PHASE 1 & 2 RATES ===
    phase1 = p3_phase1_rates(p3, ρ, ℳ_adjusted, constants, state,
                             temperature_tendency, vapor_tendency)
    phase2 = p3_phase2_rates(p3, ρ, ℳ_adjusted, constants, state, phase1)

    # === EXTRACT RATES INTO LOCAL VARIABLES FOR SINK LIMITING ===
    # Phase 1
    cond = phase1.condensation
    ccn_activation_mass = phase1.ccn_activation_mass
    ccn_activation_number = phase1.ccn_activation_number
    autoconv = phase1.autoconversion
    accr = phase1.accretion
    cloud_self = phase1.cloud_self_collection
    rain_evap = phase1.rain_evaporation
    rain_cond = phase1.rain_condensation
    rain_self = phase1.rain_self_collection
    rain_br = phase1.rain_breakup
    dep = phase1.deposition
    coat_cond = phase1.coating_condensation
    coat_evap = phase1.coating_evaporation
    partial_melt = ifelse(phase2.whole_particle_clipping, zero(FT), phase1.partial_melting)
    complete_melt = phase2.complete_melting
    melt_n = phase2.melting_number
    whole_particle_clipping = phase2.whole_particle_clipping

    # Phase 2
    agg = phase2.aggregation
    ni_lim = phase2.ni_limit
    cloud_rim = phase2.cloud_riming
    cloud_rim_n = phase2.cloud_riming_number
    rain_rim = phase2.rain_riming
    rain_rim_n = phase2.rain_riming_number
    ρᶠ_new = phase2.rime_density_new
    wg_cloud = phase2.wet_growth_cloud
    wg_rain = phase2.wet_growth_rain
    wg_shed = phase2.wet_growth_shedding
    wg_shed_n = phase2.wet_growth_shedding_number
    wg_excess_rain = ifelse(whole_particle_clipping, zero(FT),
                            max(0, wg_shed_n * parameters.shed_drop_mass - wg_shed))
    wg_densif_mass = phase2.wet_growth_densification_mass
    wg_densif_vol = phase2.wet_growth_densification_volume
    shed = phase2.shedding
    shed_n = phase2.shedding_number
    inactive_coating_cleanup = ifelse(
        parameters.liquid_fraction_active, zero(FT),
        qʷⁱ_budget / parameters.sink_limiting_timescale)
    shed = shed + inactive_coating_cleanup
    shed_n = shed_n + inactive_coating_cleanup / parameters.shed_drop_mass
    refrz = phase2.refreezing
    nuc_q = phase2.nucleation_mass
    nuc_n = phase2.nucleation_number
    cloud_frz_q = phase2.cloud_freezing_mass
    cloud_frz_n = phase2.cloud_freezing_number
    rain_frz_q = phase2.rain_freezing_mass
    rain_frz_n = phase2.rain_freezing_number
    cloud_warm_q = phase2.cloud_warm_collection
    cloud_warm_n = phase2.cloud_warm_collection_number
    rain_warm_q = phase2.rain_warm_collection
    rain_warm_n = phase2.rain_warm_collection_number

    # These clips are pre-process whole-particle transfers:
    # the original ice particle is gone before collection, vapor growth, or
    # aggregation is evaluated. Retain independent new-ice sources (nucleation and
    # freezing), but suppress every process that requires the clipped particle.
    dep = ifelse(whole_particle_clipping, zero(FT), dep)
    coat_cond = ifelse(whole_particle_clipping, zero(FT), coat_cond)
    coat_evap = ifelse(whole_particle_clipping, zero(FT), coat_evap)
    agg = ifelse(whole_particle_clipping, zero(FT), agg)
    ni_lim = ifelse(whole_particle_clipping, zero(FT), ni_lim)
    cloud_rim = ifelse(whole_particle_clipping, zero(FT), cloud_rim)
    cloud_rim_n = ifelse(whole_particle_clipping, zero(FT), cloud_rim_n)
    rain_rim = ifelse(whole_particle_clipping, zero(FT), rain_rim)
    rain_rim_n = ifelse(whole_particle_clipping, zero(FT), rain_rim_n)
    wg_cloud = ifelse(whole_particle_clipping, zero(FT), wg_cloud)
    wg_rain = ifelse(whole_particle_clipping, zero(FT), wg_rain)
    wg_shed = ifelse(whole_particle_clipping, zero(FT), wg_shed)
    wg_shed_n = ifelse(whole_particle_clipping, zero(FT), wg_shed_n)
    wg_densif_mass = ifelse(whole_particle_clipping, zero(FT), wg_densif_mass)
    wg_densif_vol = ifelse(whole_particle_clipping, zero(FT), wg_densif_vol)
    cloud_warm_q = ifelse(whole_particle_clipping, zero(FT), cloud_warm_q)
    cloud_warm_n = ifelse(whole_particle_clipping, zero(FT), cloud_warm_n)
    rain_warm_q = ifelse(whole_particle_clipping, zero(FT), rain_warm_q)
    rain_warm_n = ifelse(whole_particle_clipping, zero(FT), rain_warm_n)

    # Track clip-only contributions separately so rime mass and volume can be
    # drained exactly instead of assuming the beginning-of-stage rime fraction.
    clipping_dry_mass = ifelse(whole_particle_clipping, complete_melt, zero(FT))
    clipping_rime_mass = ifelse(whole_particle_clipping, qᶠ / parameters.refreezing_timescale,
                                zero(FT))
    clipping_rime_volume = ifelse(whole_particle_clipping, bᶠ / parameters.refreezing_timescale,
                                  zero(FT))

    # === SINK LIMITING ===
    dt_safety = parameters.sink_limiting_timescale

    # --- Vapor sinks ---
    # The saturation-adjustment caps are applied before the per-species conservation
    # budgets, so the cloud/rain/ice budgets below must see the final vapor-limited
    # rates.
    qᵗ = q.vapor + q.liquid + q.ice
    vapor_rates = limit_vapor_rates(cond, ccn_activation_mass, ccn_activation_number,
                                    rain_cond, rain_evap, dep, coat_cond, coat_evap,
                                    nuc_q, nuc_n, qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants,
                                    dt_safety, T₀)
    cond = vapor_rates.cond
    ccn_activation_mass = vapor_rates.ccn_activation_mass
    ccn_activation_number = vapor_rates.ccn_activation_number
    rain_cond = vapor_rates.rain_cond
    rain_evap = vapor_rates.rain_evap
    dep = vapor_rates.dep
    coat_cond = vapor_rates.coat_cond
    coat_evap = vapor_rates.coat_evap
    nuc_q = vapor_rates.nuc_q
    nuc_n = vapor_rates.nuc_n

    # --- Cloud liquid sinks ---
    # The per-species conservation budget splits the signed condensation rate into a
    # non-negative source and a non-negative evaporation sink, and counts the sink in
    # the cloud sink total. Track the negative part of `cond` as a sink magnitude here
    # so it gets rescaled alongside the other cloud sinks when the budget would
    # over-deplete `qᶜˡ`.
    cloud_evap = max(0, -cond)
    cloud_source_total = max(0, cond) + ccn_activation_mass
    # Homogeneous freezing is applied after all ordinary process budgets below,
    # consistent with its place in the process ordering. Do not reserve liquid here:
    # ordinary cloud processes first act on the full cloud reservoir.
    cloud_available = max(0, qᶜˡ) + cloud_source_total * dt_safety
    cloud_sink_total = autoconv + accr + cloud_rim + cloud_frz_q +
                       cloud_warm_q + wg_cloud + wg_shed + cloud_evap
    f_cloud = sink_limiting_factor(cloud_sink_total, cloud_available, dt_safety)
    autoconv      = autoconv * f_cloud
    accr          = accr * f_cloud
    cloud_rim     = cloud_rim * f_cloud
    cloud_rim_n   = cloud_rim_n * f_cloud
    cloud_frz_q   = cloud_frz_q * f_cloud
    cloud_frz_n   = cloud_frz_n * f_cloud
    cloud_warm_q  = cloud_warm_q * f_cloud
    cloud_warm_n  = cloud_warm_n * f_cloud
    wg_cloud      = wg_cloud * f_cloud
    wg_shed       = wg_shed * f_cloud
    cond          = ifelse(cond < 0, cond * f_cloud, cond)

    cloud_warm_to_ice = ifelse(parameters.liquid_fraction_active, cloud_warm_q, zero(FT))
    cloud_warm_to_rain = ifelse(parameters.liquid_fraction_active, zero(FT), cloud_warm_q)

    # Sublimation number loss, taken from the unlimited `dep` before any budget
    # scales it; the loop below then scales `sublim_n` wherever it scales `dep`.
    sublim_mag = max(0, -dep)
    sublim_n = sublim_mag * safe_divide(max(0, nⁱ),
                                        max(qⁱ, FT(parameters.floors.mass_scale)),
                                        zero(FT))

    # Wet-ice coating condensation/evaporation comes from the coupled
    # saturation adjustment (P3CoupledVaporRates). The dry/wet exclusivity is
    # enforced inside that formula via εⁱ / εʷⁱ activation.

    # Rain, dry ice, total ice, and coating exchange mass with one another.
    # A single sequential limiter pass can credit a source that a later donor
    # limiter subsequently reduces. Re-project the four donor budgets a
    # configurable number of times; every projection only reduces rates, so
    # this converges monotonically while remaining allocation-free and GPU-safe.
    #
    # Iteration 0 is the first pass, which historically ran the rain / total-ice
    # / coating budgets before the dry-ice budget joined the cycle. Gating
    # `f_dry_ice` to 1 there reproduces that exactly — multiplying by one is
    # exact — while keeping every budget written once, so a new rate cannot be
    # added to the first pass and forgotten in the re-projections.
    for iteration in 0:parameters.coupled_sink_limiting_iterations
        dry_ice_source_total = max(0, dep) + cloud_rim + rain_rim + refrz +
                               nuc_q + cloud_frz_q + rain_frz_q
        dry_ice_available = max(0, qⁱ) + dry_ice_source_total * dt_safety
        dry_ice_sink_total = partial_melt + complete_melt + max(0, -dep)
        f_dry_ice = ifelse(iteration > 0,
                           sink_limiting_factor(dry_ice_sink_total, dry_ice_available,
                           dt_safety),
                           one(FT))
        partial_melt = partial_melt * f_dry_ice
        complete_melt = complete_melt * f_dry_ice
        melt_n = melt_n * f_dry_ice
        clipping_dry_mass = clipping_dry_mass * f_dry_ice
        clipping_rime_mass = clipping_rime_mass * f_dry_ice
        clipping_rime_volume = clipping_rime_volume * f_dry_ice
        dep = ifelse(dep < 0, dep * f_dry_ice, dep)
        sublim_n = sublim_n * f_dry_ice

        rain_source_total = autoconv + accr + complete_melt + shed + wg_shed +
                            cloud_warm_to_rain + rain_cond
        rain_available = max(0, qʳ) + rain_source_total * dt_safety
        rain_sink_total = rain_rim + rain_frz_q + rain_warm_q + wg_rain + rain_evap
        f_rain = sink_limiting_factor(rain_sink_total, rain_available, dt_safety)
        rain_rim = rain_rim * f_rain
        rain_rim_n = rain_rim_n * f_rain
        rain_frz_q = rain_frz_q * f_rain
        rain_frz_n = rain_frz_n * f_rain
        rain_warm_q = rain_warm_q * f_rain
        rain_warm_n = rain_warm_n * f_rain
        wg_rain = wg_rain * f_rain
        rain_evap = rain_evap * f_rain
        wg_excess_rain = wg_excess_rain * f_rain
        wg_shed_n = (wg_shed + wg_excess_rain) / parameters.shed_drop_mass

        total_ice_source_total = max(0, dep) + cloud_rim + rain_rim +
                                 nuc_q + cloud_frz_q + rain_frz_q +
                                 cloud_warm_to_ice + rain_warm_q +
                                 wg_cloud + wg_rain + coat_cond
        total_ice_available = max(total_ice_mass(qⁱ, qʷⁱ_budget), FT(0)) +
                              total_ice_source_total * dt_safety
        total_ice_sink_total = complete_melt + max(0, -dep) + shed + coat_evap
        f_total_ice = sink_limiting_factor(total_ice_sink_total,
                                           total_ice_available, dt_safety)
        complete_melt = complete_melt * f_total_ice
        melt_n = melt_n * f_total_ice
        clipping_dry_mass = clipping_dry_mass * f_total_ice
        clipping_rime_mass = clipping_rime_mass * f_total_ice
        clipping_rime_volume = clipping_rime_volume * f_total_ice
        dep = ifelse(dep < 0, dep * f_total_ice, dep)
        sublim_n = sublim_n * f_total_ice
        shed = shed * f_total_ice
        shed_n = shed_n * f_total_ice
        coat_evap = coat_evap * f_total_ice

        qwi_source_total = partial_melt + cloud_warm_to_ice + rain_warm_q +
                           wg_cloud + wg_rain + coat_cond
        qwi_available = max(0, qʷⁱ_budget) + qwi_source_total * dt_safety
        qwi_sink_total = shed + refrz + coat_evap
        f_qwi = sink_limiting_factor(qwi_sink_total, qwi_available, dt_safety)
        shed = shed * f_qwi
        shed_n = shed_n * f_qwi
        refrz = refrz * f_qwi
        coat_evap = coat_evap * f_qwi
        complete_melt = ifelse(whole_particle_clipping,
                               complete_melt * f_qwi, complete_melt)
        melt_n = ifelse(whole_particle_clipping, melt_n * f_qwi, melt_n)
        clipping_dry_mass = ifelse(whole_particle_clipping,
                                   clipping_dry_mass * f_qwi,
                                   clipping_dry_mass)
        clipping_rime_mass = ifelse(whole_particle_clipping,
                                    clipping_rime_mass * f_qwi,
                                    clipping_rime_mass)
        clipping_rime_volume = ifelse(whole_particle_clipping,
                                      clipping_rime_volume * f_qwi,
                                      clipping_rime_volume)
    end

    qⁱ_total_coat = max(total_ice_mass(qⁱ, qʷⁱ), FT(parameters.floors.mass_scale))
    coat_evap_n = coat_evap * safe_divide(max(0, nⁱ), qⁱ_total_coat, zero(FT))
    sublim_n = sublim_n + coat_evap_n

    # Recompute splintering from sink-limited riming rates
    diagnostic_mean_mass = qⁱ_total / nⁱ_diagnostic
    D_mean = cbrt(6 * diagnostic_mean_mass / (FT(π) * ρ_mean))
    qᶜˡ_splintering_rate, qʳ_splintering_rate, spl_n = rime_splintering_rates(
        p3, cloud_rim, rain_rim, T, D_mean, Fˡ, surface_temperature, qᶠ)
    qᶜˡ_splintering_rate = min(qᶜˡ_splintering_rate, max(0, cloud_rim))
    qʳ_splintering_rate = min(qʳ_splintering_rate, max(0, rain_rim))
    spl_q = qᶜˡ_splintering_rate + qʳ_splintering_rate

    # Reconstruct the ordinary post-process ice reservoirs. A second Fˡ > 0.99 clip
    # is applied after these processes, so a particle that crosses the threshold
    # during melting must transfer its residual mass and number as a whole. The
    # dry-ice projection above guarantees these residuals are non-negative before
    # the clip is diagnosed.
    dry_ice_source_total = dep + cloud_rim + rain_rim + refrz +
                           nuc_q + cloud_frz_q + rain_frz_q
    dry_ice_sink_total = partial_melt + complete_melt
    dry_ice_remaining = max(0, qⁱ +
                            (dry_ice_source_total - dry_ice_sink_total) * dt_safety)

    qwi_source_total = partial_melt + cloud_warm_to_ice + rain_warm_q +
                       wg_cloud + wg_rain + coat_cond
    qwi_sink_total = shed + refrz + coat_evap
    qwi_remaining = max(0, qʷⁱ_budget +
                        (qwi_source_total - qwi_sink_total) * dt_safety)
    total_ice_remaining = dry_ice_remaining + qwi_remaining
    liquid_fraction_remaining = safe_divide(qwi_remaining, total_ice_remaining,
                                            zero(FT))
    post_process_clipping_active = parameters.liquid_fraction_active &
                                   (total_ice_remaining >= p3.minimum_mass_mixing_ratio) &
                                   (liquid_fraction_remaining > 1 - parameters.liquid_fraction_clipping_threshold)

    # Rime companions are reconstructed with the same formulas used by the
    # prognostic tendencies, excluding homogeneous freezing, which occurs after
    # this clip in the process ordering.
    ordinary_complete_melting = max(0, complete_melt - clipping_dry_mass)
    ordinary_total_melting = partial_melt + ordinary_complete_melting
    sublimation = max(0, -dep)
    rime_mass_gain = cloud_rim + rain_rim + refrz + cloud_frz_q + rain_frz_q + wg_densif_mass
    rime_mass_loss = Fᶠ * (ordinary_total_melting + sublimation) +
                     clipping_rime_mass
    rime_mass_remaining = max(0, qᶠ + (rime_mass_gain - rime_mass_loss) * dt_safety)

    rime_density_safe = max(ρᶠ, parameters.minimum_rime_density)
    new_rime_density_safe = max(ρᶠ_new, parameters.minimum_rime_density)
    maximum_rime_density = parameters.maximum_rime_density
    rime_volume_gain = cloud_rim / new_rime_density_safe +
                       (rain_rim + refrz + cloud_frz_q + rain_frz_q) /
                       maximum_rime_density + wg_densif_vol
    rime_volume_loss = Fᶠ * (ordinary_total_melting + sublimation) /
                       rime_density_safe + clipping_rime_volume
    dry_ice_safe = max(qⁱ, FT(parameters.floors.mass_scale))
    bᶠ_safe = Fᶠ * dry_ice_safe / rime_density_safe
    melt_densification = bᶠ_safe * (parameters.pure_ice_density - rime_density_safe) *
                          ordinary_total_melting / (rime_density_safe * dry_ice_safe)
    densification_active = (rime_density_safe < parameters.pure_ice_density) &
                           !parameters.liquid_fraction_active
    melt_densification = ifelse(densification_active, melt_densification, zero(FT))
    rime_volume_after_ordinary_processes = max(
        0, bᶠ + (rime_volume_gain - rime_volume_loss) * dt_safety)
    maximum_melt_densification = rime_volume_after_ordinary_processes / dt_safety
    melt_densification = min(melt_densification, maximum_melt_densification)
    rime_volume_remaining = max(
        0, rime_volume_after_ordinary_processes - melt_densification * dt_safety)

    ni_correction = ifelse(whole_particle_clipping, zero(FT),
                           (nⁱ - nⁱ_global) / dt_safety)
    ice_number_source_total = nuc_n + cloud_frz_n + rain_frz_n + spl_n
    # Global and DSD corrections conceptually replace the raw population with
    # the bounded one before physical processes act. Give particle-removing
    # melting/sublimation priority, then limit number-only aggregation to the
    # population that remains. A pre-process whole-particle clip instead drains
    # the raw population directly and suppresses both corrections.
    number_after_correction = ifelse(whole_particle_clipping, nⁱ_raw, nⁱ)
    particle_sink_capacity = max(
        0, number_after_correction / dt_safety + ice_number_source_total)
    melt_n = min(melt_n, particle_sink_capacity)
    particle_sink_capacity = max(0, particle_sink_capacity - melt_n)
    sublim_n = min(sublim_n, particle_sink_capacity)
    number_available_for_aggregation = max(0, particle_sink_capacity - sublim_n)
    agg = min(agg, number_available_for_aggregation)
    ice_number_sink_total = melt_n + sublim_n + agg
    ice_number_remaining = max(0, number_after_correction +
                               (ice_number_source_total - ice_number_sink_total) *
                               dt_safety)

    post_clip_dry_mass = ifelse(post_process_clipping_active,
                                dry_ice_remaining / dt_safety, zero(FT))
    post_clip_coating = ifelse(post_process_clipping_active,
                               qwi_remaining / dt_safety, zero(FT))
    post_clip_number = ifelse(post_process_clipping_active,
                              ice_number_remaining / dt_safety, zero(FT))
    post_clip_rime_mass = ifelse(post_process_clipping_active,
                                 rime_mass_remaining / dt_safety, zero(FT))
    post_clip_rime_volume = ifelse(post_process_clipping_active,
                                   rime_volume_remaining / dt_safety, zero(FT))

    complete_melt = complete_melt + post_clip_dry_mass
    shed = shed + post_clip_coating
    melt_n = melt_n + post_clip_number
    clipping_dry_mass = clipping_dry_mass + post_clip_dry_mass
    clipping_rime_mass = clipping_rime_mass + post_clip_rime_mass
    clipping_rime_volume = clipping_rime_volume + post_clip_rime_volume
    post_process_clipping = ifelse(post_process_clipping_active, one(FT), zero(FT))

    # Reserve the immersion-frozen drops first: their number companion must retain
    # the same species-budget ratio as rain_freezing_mass.
    # Project the remaining number-only sinks onto the population left afterward.
    cloud_warm_rain_number = ifelse(
        parameters.liquid_fraction_active, zero(FT), cloud_warm_q / parameters.shed_drop_mass)

    # Net the self-collection/breakup pair before it enters the number budget.
    # Physically this is one signed rate: a base self-collection rate reduced by the
    # Verlinde-Cotton breakup modifier. Breeze reports the two directions separately,
    # so they arrive here as a sink/source pair that must be collapsed first:
    # rescaling only the sink half by `f_rain_number` below would leave the breakup
    # source at full strength against a limited sink, turning the net into spurious
    # rain-number production once Dʳ exceeds the breakup
    # threshold. Netting here also keeps `f_rain_number` a positivity guarantee —
    # everything in `rain_number_source_total` stays unscaled, so the limited
    # sinks cannot outrun the number the budget promised them.
    net_rain_self = rain_self - rain_br
    rain_self = max(0, net_rain_self)
    rain_br = max(0, -net_rain_self)

    rain_number_source_total = autoconv / rain_seed_drop_mass(p3) + melt_n +
                               rain_br + shed_n + cloud_warm_rain_number + wg_shed_n
    rain_evap_n = safe_divide(nʳ * rain_evap, qʳ, zero(FT))
    rain_number_available = max(0, nʳ) + rain_number_source_total * dt_safety
    rain_number_available_after_freezing =
        max(0, rain_number_available - rain_frz_n * dt_safety)
    rain_number_sink_total = rain_evap_n + rain_self + rain_rim_n + rain_warm_n
    f_rain_number = sink_limiting_factor(
        rain_number_sink_total, rain_number_available_after_freezing, dt_safety)
    rain_evap_n = rain_evap_n * f_rain_number
    rain_self = rain_self * f_rain_number
    rain_rim_n = rain_rim_n * f_rain_number
    rain_warm_n = rain_warm_n * f_rain_number

    # --- Homogeneous freezing of post-process liquid ---
    # Homogeneous freezing acts after the ordinary process updates and after
    # sedimentation. Sedimentation is advanced by the host model in Breeze, but
    # within the local process operator the essential ordering is preserved: first
    # finalize every ordinary limiter above, then freeze the liquid that remains.
    # Re-diagnosing the rate from the residual also captures liquid created by
    # condensation, melting, or shedding during this interval.
    cloud_sink_total = autoconv + accr + cloud_rim + cloud_frz_q +
                       cloud_warm_q + wg_cloud + wg_shed + max(0, -cond)
    cloud_remaining = max(0, max(0, qᶜˡ) +
                          (cloud_source_total - cloud_sink_total) * dt_safety)

    rain_source_total = autoconv + accr + complete_melt + shed + wg_shed +
                        cloud_warm_to_rain + rain_cond
    rain_sink_total = rain_rim + rain_frz_q + rain_warm_q + wg_rain + rain_evap
    rain_remaining = max(0, max(0, qʳ) +
                         (rain_source_total - rain_sink_total) * dt_safety)

    # Diagnose the post-process number reservoirs as well, so frozen liquid carries
    # the number left by collection, breakup, melting, and activation rather than the
    # beginning-of-stage number. In the prescribed-Nᶜˡ path, cloud number is reset
    # to its prescribed value immediately before homogeneous freezing.
    cloud_number_tendency = cloud_number_tendency_before_homogeneous_freezing(
        p3, ρ, qᶜˡ, Nᶜˡ, ccn_activation_mass, ccn_activation_number,
        autoconv, accr, cloud_self, cloud_rim_n, cloud_frz_n, cloud_warm_n)
    prognostic_cloud_number = max(0, cloud.nᶜˡ +
                                  cloud_number_tendency * dt_safety)
    prescribed_cloud_number = p3.cloud.number_concentration / ρ
    cloud_number_remaining = ifelse(isnothing(p3.aerosol), prescribed_cloud_number,
                                    prognostic_cloud_number)

    rain_number_tendency = rain_number_tendency_before_homogeneous_freezing(
        p3, autoconv, melt_n, rain_evap_n, rain_self,
        rain_br, rain_rim_n, rain_frz_n, shed_n, cloud_warm_q, rain_warm_n, wg_shed_n)
    rain_number_remaining = max(0, nʳ +
                                rain_number_tendency * dt_safety)

    cloud_hom_q, cloud_hom_n = homogeneous_freezing_rate(
        p3, cloud_remaining, cloud_number_remaining, T)
    rain_hom_q, rain_hom_n = homogeneous_freezing_rate(
        p3, rain_remaining, max(0, rain_number_remaining), T)

    # `homogeneous_freezing_timescale` and `sink_limiting_timescale` are
    # independently configurable. Cap both mass and number consistently so one
    # limiter interval can never remove more than the residual reservoir.
    f_cloud_hom = sink_limiting_factor(cloud_hom_q, cloud_remaining, dt_safety)
    cloud_hom_q = cloud_hom_q * f_cloud_hom
    cloud_hom_n = cloud_hom_n * f_cloud_hom
    f_rain_hom = sink_limiting_factor(rain_hom_q, rain_remaining, dt_safety)
    rain_hom_q = rain_hom_q * f_rain_hom
    rain_hom_n = rain_hom_n * f_rain_hom

    # DSD number correction feedback
    ncl_correction = (cloud.nᶜˡ - ℳ.nᶜˡ) / dt_safety
    nr_correction = (nʳ - ℳ.nʳ) / dt_safety

    # The supersaturation is re-diagnosed as sᵛ⁺ˡ = qᵛ - qᵛ⁺ˡ(T) at the end of the
    # step. That final T must come from the same conserved thermodynamic variable
    # (`ρθ` or `ρs`) that the host model advances, not from a standalone cᵖᵈ
    # latent-heating estimate.
    vapor_to_liquid = cond + ccn_activation_mass + rain_cond + coat_cond - rain_evap - coat_evap
    vapor_to_ice = dep + nuc_q
    liquid_to_ice = cloud_rim + rain_rim + cloud_frz_q + rain_frz_q +
                    cloud_hom_q + rain_hom_q + refrz -
                    complete_melt - partial_melt
    supersaturation_tendency = final_predicted_supersaturation_tendency(
        parameters, 𝒰, qᵛ, q.liquid, q.ice, ρ, constants, ℳ.sᵛ⁺ˡ, dt_safety,
        vapor_to_liquid, vapor_to_ice, liquid_to_ice)
    # `saturation_alignment_rate` is intentionally NOT rescaled by the cloud sink limiter: the
    # G&M alignment is its own one-shot saturation adjustment with a local
    # `ε ≥ -qᶜˡ` cap, and the cloud budget at the limiter starts from the already
    # adjusted qᶜˡ + ε — so ε is absorbed into `cloud_available`, not the
    # source/sink list.
    cond_total = cond + saturation_alignment_rate

    return P3ProcessRates{FT}(
        cond_total,
        autoconv, accr, cloud_self, rain_evap, rain_evap_n, rain_self, rain_br,
        dep, partial_melt, complete_melt, melt_n,
        clipping_dry_mass, clipping_rime_mass, clipping_rime_volume,
        post_process_clipping,
        sublim_n,
        agg, ni_lim,
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        shed, shed_n, refrz,
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        spl_q, spl_n,
        cloud_hom_q, cloud_hom_n, rain_hom_q, rain_hom_n,
        cloud_warm_q, cloud_warm_n, rain_warm_q, rain_warm_n,
        wg_cloud, wg_rain,
        wg_shed, wg_shed_n,
        ccn_activation_mass, ccn_activation_number, rain_cond,
        coat_cond, coat_evap,
        wg_densif_mass, wg_densif_vol,
        ncl_correction, nr_correction, ni_correction,
        saturation_alignment_rate, supersaturation_tendency,
    )
end
