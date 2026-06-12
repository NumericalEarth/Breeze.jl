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
                             with_moisture,
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
##### Combined P3 tendency calculation
#####

"""
Derived thermodynamic and PSD state computed during setup of `compute_p3_process_rates`.
Passed to `@noinline` sub-functions to avoid recomputation.
Internal implementation detail — not part of the public API.
"""
struct P3DerivedState{FT, Q}
    # Bounded prognostic state
    nⁱ :: FT        # bounded by maximum_ice_number_density / ρ
    nʳ :: FT        # DSD-bounded rain number
    qᶠ :: FT        # consistent rime mass
    bᶠ :: FT        # consistent rime volume
    Fᶠ :: FT        # rime fraction
    ρᶠ :: FT        # rime density
    # PSD parameters
    μ_ice :: FT     # ice shape parameter
    Fˡ_mu :: FT     # liquid fraction for μ lookup
    Nᶜ :: FT        # effective cloud droplet number concentration
    nᶜˡ :: FT       # DSD-bounded cloud number (for correction)
    μ_c :: FT       # local cloud DSD shape parameter (Fortran mu_c)
    λ_c :: FT       # local cloud DSD slope parameter (Fortran lamc)
    # Thermodynamic state
    T :: FT         # temperature [K]
    P :: FT         # pressure [Pa]
    qᵛ :: FT        # vapor mass fraction
    qᵛ⁺ˡ :: FT      # saturation vapor fraction over liquid
    qᵛ⁺ⁱ :: FT      # saturation vapor fraction over ice
    q :: Q          # MoistureMassFractions for heat capacity / density
    # Transport properties
    D_v :: FT       # water vapor diffusivity [m²/s]
    K_a :: FT       # thermal conductivity of air [W/m/K]
    nu :: FT        # kinematic viscosity [m²/s]
    # Mixture heat capacity (hoisted to avoid recomputation in phase-1 sub-functions)
    cᵖᵐ :: FT       # moist mixture heat capacity [J/kg/K]
end

@inline function liquid_supersaturation_after_moisture_update(𝒰, qᵛ, qˡ, qⁱ, ρ, constants)
    q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
    𝒰₁ = with_moisture(𝒰, q)
    T = temperature(𝒰₁, constants)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return qᵛ - qᵛ⁺ˡ
end

"""
Phase 1 process rates: condensation, rain, deposition, and melting.
Returned by `_p3_phase1_rates`. Internal implementation detail.
"""
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

"""
Phase 2 process rates: aggregation, riming, wet growth, shedding,
nucleation, homogeneous freezing, and warm collection.
Returned by `_p3_phase2_rates`. Internal implementation detail.
"""
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
    nucleation_mass :: FT
    nucleation_number :: FT
    cloud_freezing_mass :: FT
    cloud_freezing_number :: FT
    rain_freezing_mass :: FT
    rain_freezing_number :: FT
    splintering_mass :: FT
    splintering_number :: FT
    cloud_homogeneous_mass :: FT
    cloud_homogeneous_number :: FT
    rain_homogeneous_mass :: FT
    rain_homogeneous_number :: FT
    cloud_warm_collection :: FT
    cloud_warm_collection_number :: FT
    rain_warm_collection :: FT
    rain_warm_collection_number :: FT
    D_mean :: FT    # needed by wrapper for splintering recomputation
    Fˡ :: FT        # needed by wrapper for splintering recomputation
end

"""
    P3ProcessRates

Container for computed P3 process rates.
Includes Phase 1 (rain, deposition, melting), Phase 2 (aggregation, riming, shedding, nucleation).

Following Milbrandt et al. (2025), melting is partitioned:
- `partial_melting`: Meltwater stays on ice as liquid coating (large particles)
- `complete_melting`: Meltwater sheds to rain (small particles)

Sign convention (M7): All one-directional rates store **positive magnitudes**.
Bidirectional rates (condensation, deposition) are positive for source, negative for sink.
Signs are applied explicitly in the `tendency_*` functions.
"""
struct P3ProcessRates{FT}
    # Phase 1: Cloud condensation/evaporation (BIDIRECTIONAL: +cond / −evap)
    condensation :: FT             # Vapor ↔ cloud liquid [kg/kg/s] (+cond, −evap)

    # Phase 1: Rain tendencies (all positive magnitudes)
    autoconversion :: FT           # Cloud → rain mass [kg/kg/s]
    accretion :: FT                # Cloud → rain mass (via rain sweep-out) [kg/kg/s]
    cloud_self_collection :: FT    # Cloud number loss from cloud-cloud coalescence [1/kg/s] (SB2001 only; 0 for KK2000/K2013)
    rain_evaporation :: FT         # Rain evaporation magnitude [kg/kg/s]
    rain_self_collection :: FT     # Rain number loss magnitude [1/kg/s]
    rain_breakup :: FT             # Rain number gain from breakup [1/kg/s]

    # Phase 1: Ice tendencies (BIDIRECTIONAL deposition; positive melting/number)
    deposition :: FT               # Vapor ↔ ice mass [kg/kg/s] (+dep, −sublim)
    partial_melting :: FT          # Ice → liquid coating (stays on ice) [kg/kg/s]
    complete_melting :: FT         # Ice → rain mass (sheds) [kg/kg/s]
    melting_number :: FT           # Ice number loss magnitude from melting [1/kg/s]

    # D2/D1: Ice number loss from vapor-driven sinks (Fortran nisub + nlevp)
    sublimation_number :: FT       # Ice number loss magnitude from sublimation / coating evaporation [1/kg/s]

    # Phase 2: Ice aggregation (positive magnitude)
    aggregation :: FT              # Ice number loss magnitude from self-collection [1/kg/s]

    # Global ice number limiter — Fortran impose_max_Ni (positive magnitude)
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

    # Above-freezing collection (T > T₀): collected hydrometeors → qʷⁱ
    # (Milbrandt et al. 2025; Fortran qccoll/qrcoll pathway)
    cloud_warm_collection :: FT        # Cloud collected above T₀ → qʷⁱ [kg/kg/s]
    cloud_warm_collection_number :: FT # Cloud number loss from warm collection [1/kg/s]
    rain_warm_collection :: FT         # Rain collected above T₀ → qʷⁱ [kg/kg/s]
    rain_warm_collection_number :: FT  # M9: Rain number loss from warm collection [1/kg/s]

    # Wet growth: collected hydrometeors redirected to qʷⁱ when collection
    # exceeds freezing capacity (Milbrandt et al. 2025; Fortran qwgrth1c/qwgrth1r)
    wet_growth_cloud :: FT             # Cloud collection redirected to qʷⁱ [kg/kg/s]
    wet_growth_rain :: FT              # Rain collection redirected to qʷⁱ [kg/kg/s]

    # Wet growth shedding — excess collection beyond freezing capacity goes to rain
    # (Fortran nrshdr/qcshd: mass that can't freeze sheds as 1 mm rain drops)
    wet_growth_shedding :: FT          # Excess collection → rain mass [kg/kg/s]
    wet_growth_shedding_number :: FT   # Rain number from wet growth shedding [1/kg/s]

    # Warm/mixed-phase budget terms
    ccn_activation_mass :: FT          # CCN activation mass rate (vapor → cloud) [kg/kg/s]
    ccn_activation_number :: FT        # CCN activation number rate [1/kg/s] (prognostic CCN only)
    rain_condensation :: FT            # Rain condensation (vapor → rain) [kg/kg/s]
    coating_condensation :: FT         # Condensation on ice liquid coating [kg/kg/s]
    coating_evaporation :: FT          # Evaporation from ice liquid coating [kg/kg/s]

    # Wet growth rime densification (Fortran lines 4303-4307)
    # During wet growth, assume total soaking: qᶠ → qⁱ, bᶠ → qⁱ/ρ_rimeMax.
    wet_growth_densification_mass :: FT   # Rime mass source: (qⁱ - qᶠ)/τ [kg/kg/s]
    wet_growth_densification_volume :: FT # Rime volume change: (qⁱ/ρ_max - bᶠ)/τ [m³/kg/s]

    # DSD number correction feedback (Fortran get_cloud_dsd2/get_rain_dsd2)
    # After lambda bounding, the DSD-consistent number may differ from the prognostic
    # number. Fortran writes the bounded value back instantaneously; here we express
    # the correction as a relaxation rate over dt_safety.
    cloud_number_correction :: FT  # (nᶜˡ_bounded - nᶜˡ) / τ [1/kg/s]
    rain_number_correction :: FT   # (nʳ_bounded - nʳ) / τ [1/kg/s]

    # G&M (2008) bounded supersaturation adjustment, also folded into
    # `condensation` so vapor and cloud tendencies include it automatically.
    # Carried separately so callers/tests can inspect the G&M contribution.
    # Sized as `ε / sink_limiting_timescale`, so dt = sink_limiting_timescale
    # integrates the one-shot adjustment exactly (see
    # `predicted_supersaturation_adjustment`).
    predicted_ssat_adjustment :: FT
    # End-of-step ssat recompute: (qᵛ_final - qᵛ⁺ˡ(T_final) - sˢᵃᵗ_initial) / τ.
    # Tied to the same dt = τ assumption.
    predicted_ssat_tendency :: FT
end

@noinline function _p3_phase1_rates(p3, ρ, ℳ, constants, state::P3DerivedState)
    FT = typeof(ρ)

    # Unpack derived state (field access on concrete struct — GPU-safe)
    T = state.T
    qᵛ = state.qᵛ
    qᵛ⁺ˡ = state.qᵛ⁺ˡ
    qᵛ⁺ⁱ = state.qᵛ⁺ⁱ
    q = state.q
    Fᶠ = state.Fᶠ
    ρᶠ = state.ρᶠ
    μ_ice = state.μ_ice
    Fˡ_mu = state.Fˡ_mu
    Nᶜ = state.Nᶜ
    nⁱ = state.nⁱ
    nʳ = state.nʳ
    P = state.P

    # Transport properties (reconstructed as NamedTuple for existing function signatures)
    transport = (; D_v = state.D_v, K_a = state.K_a, nu = state.nu)
    cᵖᵐ = state.cᵖᵐ

    # =========================================================================
    # Coupled cloud/rain/ice vapor growth and decay
    # =========================================================================
    vapor_rates = coupled_saturation_adjustment_rates(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.qʳ, nʳ,
                                                      ℳ.qⁱ, ℳ.qʷⁱ, nⁱ, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ,
                                                      Fᶠ, ρᶠ, T, P, ρ, constants,
                                                      transport, q, μ_ice,
                                                      state.μ_c, state.λ_c, state.nᶜˡ, ℳ.w, cᵖᵐ)
    cond = vapor_rates.condensation

    # CCN activation (prescribed or prognostic; depletes ℳ.nᵃ when prognostic)
    ccn = compute_ccn_activation(p3.aerosol, p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.nᵃ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants, cᵖᵐ)
    ccn_act = ccn.mass
    ccn_act_n = ccn.number

    # =========================================================================
    # Rain processes
    # =========================================================================
    autoconv = rain_autoconversion_rate(p3, ℳ.qᶜˡ, Nᶜ, ρ, ℳ.qʳ)
    accr = rain_accretion_rate(p3, ℳ.qᶜˡ, ℳ.qʳ, ρ)
    cloud_self = cloud_self_collection_rate(p3, ℳ.qᶜˡ, Nᶜ, ρ)
    rain_evap = vapor_rates.rain_evaporation
    rain_cond = vapor_rates.rain_condensation
    rain_self = rain_self_collection_rate(p3, ℳ.qʳ, nʳ, ρ)
    rain_br = rain_breakup_rate(p3, ℳ.qʳ, nʳ, rain_self)

    # =========================================================================
    # Ice deposition/sublimation and wet-ice coating condensation/evaporation
    # =========================================================================
    dep = vapor_rates.deposition
    dep = ifelse(ℳ.qⁱ > FT(1e-20), dep, zero(FT))

    coat_cond = ifelse(ℳ.qⁱ > FT(1e-20), vapor_rates.coating_condensation, zero(FT))
    coat_evap = ifelse(ℳ.qⁱ > FT(1e-20), vapor_rates.coating_evaporation, zero(FT))

    melt_rates = ice_melting_rates(p3, ℳ.qⁱ, nⁱ, ℳ.qʷⁱ, T, P, qᵛ, qᵛ⁺ˡ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)
    partial_melt = melt_rates.partial_melting
    complete_melt = melt_rates.complete_melting
    melt_n = ice_melting_number_rate(ℳ.qⁱ, nⁱ, complete_melt)

    return P3Phase1Rates{FT}(cond, ccn_act, ccn_act_n,
                             autoconv, accr, cloud_self,
                             rain_evap, rain_cond, rain_self, rain_br,
                             dep, coat_cond, coat_evap,
                             partial_melt, complete_melt, melt_n)
end

@noinline function _p3_phase2_rates(p3, ρ, ℳ, constants, state::P3DerivedState, phase1::P3Phase1Rates)
    FT = typeof(ρ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # Unpack derived state
    T = state.T
    P = state.P
    qᵛ = state.qᵛ
    qᵛ⁺ˡ = state.qᵛ⁺ˡ
    qᵛ⁺ⁱ = state.qᵛ⁺ⁱ
    q = state.q
    Fᶠ = state.Fᶠ
    ρᶠ = state.ρᶠ
    qᶠ = state.qᶠ
    bᶠ = state.bᶠ
    μ_ice = state.μ_ice
    Fˡ_mu = state.Fˡ_mu
    Nᶜ = state.Nᶜ
    μ_c = state.μ_c
    λ_c = state.λ_c
    nⁱ = state.nⁱ
    nʳ = state.nʳ
    transport = (; D_v = state.D_v, K_a = state.K_a, nu = state.nu)

    qⁱ = ℳ.qⁱ
    qʷⁱ = ℳ.qʷⁱ
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ

    # =========================================================================
    # Aggregation
    # =========================================================================
    agg = ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)

    # Global ice number limiter — Fortran impose_max_Ni hard-clamps the prognostic
    # nitot at multiple driver points (microphy_p3.f90:2812, 4390, 4937). Mirror that as
    # a tendency by using the *raw* prognostic ℳ.nⁱ rather than the locally pre-capped
    # `state.nⁱ`, which would always be ≤ N_max/ρ and make this limiter dead.
    N_max = prp.maximum_ice_number_density
    ni_lim = clamp_positive(ℳ.nⁱ - N_max / ρ) / prp.sink_limiting_timescale

    # =========================================================================
    # Riming
    # =========================================================================
    cloud_rim = cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nᶜ, ρ, cloud_rim)
    rain_rim = rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    rain_rim_n = rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)

    # Rime density
    # Fortran p3_main indexes the rime density formula with the locally diagnosed
    # cloud DSD (mu_c, lamc from get_cloud_dsd2 — microphy_p3.f90:2801, 3380-3388),
    # not prescribed cloud parameters. Pass μ_c and λ_c from `diagnose_cloud_dsd`
    # through to match Fortran's Cober-List rime density when Nᶜ is prognostic.
    # Use total ice mass for terminal velocity to match the table-axis convention.
    qⁱ_total = total_ice_mass(qⁱ, qʷⁱ)
    vᵢ = ice_terminal_velocity_mass_weighted(p3, qⁱ_total, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=Fˡ_mu, μ=μ_ice)
    ρᶠ_new = rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport, μ_c, λ_c)

    # =========================================================================
    # Wet growth capacity and collection rerouting
    # =========================================================================
    has_hydrometeors = (clamp_positive(qᶜˡ) + clamp_positive(qʳ)) >= FT(1e-6)
    qwgrth_raw = wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)
    qwgrth = ifelse(has_hydrometeors, qwgrth_raw, zero(FT))

    total_collection = cloud_rim + rain_rim
    is_wet_growth = has_hydrometeors & (total_collection > qwgrth + FT(1e-10))

    wg_cloud = ifelse(is_wet_growth, cloud_rim, zero(FT))
    wg_rain  = ifelse(is_wet_growth, rain_rim, zero(FT))
    cloud_rim   = ifelse(is_wet_growth, zero(FT), cloud_rim)
    rain_rim    = ifelse(is_wet_growth, zero(FT), rain_rim)

    # Wet growth shedding
    shed_active = !prp.liquid_fraction_active & is_wet_growth
    wg_shed   = ifelse(shed_active, clamp_positive(total_collection - qwgrth), zero(FT))
    wg_shed_n = wg_shed / prp.shed_drop_mass

    # Wet growth rime densification
    ρ_rimemax = prp.maximum_rime_density
    τ_densif = prp.rime_densification_timescale
    qⁱ_safe = clamp_positive(qⁱ)
    bᶠ_safe = max(bᶠ, FT(1e-20))
    wg_densif_active = is_wet_growth & !prp.liquid_fraction_active & (qⁱ_safe > FT(1e-14))
    wg_densif_mass = clamp_positive(qⁱ_safe - qᶠ) / τ_densif
    wg_densif_vol = (qⁱ_safe / ρ_rimemax - bᶠ_safe) / τ_densif
    wg_densif_mass = ifelse(wg_densif_active, wg_densif_mass, zero(FT))
    wg_densif_vol  = ifelse(wg_densif_active, wg_densif_vol, zero(FT))

    # =========================================================================
    # Shedding and refreezing
    # =========================================================================
    qⁱ_total = max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20))
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    D_mean = first(mean_ice_particle_diameter(m_mean, Fᶠ, Fˡ, ρᶠ, prp))

    shed = shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean, μ_ice)
    shed_n = shedding_number_rate(p3, shed)
    refrz = refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)

    # Liquid fraction clipping
    Fl_small = prp.liquid_fraction_small
    τ_clip = prp.refreezing_timescale
    qʷⁱ_eff = clamp_positive(qʷⁱ)
    clip_freeze = (T < T₀) & (Fˡ < Fl_small) & (Fˡ > 0)
    clip_melt   = (T >= T₀) & (Fˡ > 1 - Fl_small)
    refrz = ifelse(clip_freeze, refrz + qʷⁱ_eff / τ_clip, refrz)
    shed = ifelse(clip_melt, shed + qʷⁱ_eff / τ_clip, shed)

    # Filiq > 0.99 safety clipping
    qⁱ_dry = clamp_positive(qⁱ)
    clip_high_fl = (Fˡ > FT(0.99)) & (qⁱ_dry > 0)
    complete_melt = phase1.complete_melting  # start from Phase 1 value
    complete_melt = ifelse(clip_high_fl, complete_melt + qⁱ_dry / τ_clip, complete_melt)
    shed = ifelse(clip_high_fl, shed + qʷⁱ_eff / τ_clip, shed)

    # M12(c): Melt tiny ice at T >= T₀
    qⁱ_total_clip = qⁱ_dry + qʷⁱ_eff
    tiny_warm_ice = (T >= T₀) & (qⁱ_total_clip >= FT(1e-14)) & (qⁱ_total_clip < prp.qsmall_dry)
    complete_melt = ifelse(tiny_warm_ice, complete_melt + qⁱ_dry / τ_clip, complete_melt)
    shed = ifelse(tiny_warm_ice, shed + qʷⁱ_eff / τ_clip, shed)

    # =========================================================================
    # Ice nucleation
    # =========================================================================
    nuc_q, nuc_n = deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    μ_r = zero(FT)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T, μ_r)

    # Rime splintering
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T, D_mean, Fˡ, T, qᶠ)

    # Homogeneous freezing
    cloud_hom_q, cloud_hom_n = homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    rain_hom_q, rain_hom_n = homogeneous_freezing_rain_rate(p3, qʳ, nʳ, T)

    # Above-freezing collection
    cloud_warm_q, cloud_warm_n_raw = cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    cloud_warm_n = cloud_warm_n_raw
    rain_warm_q_full = rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    # Number sink from above-freezing rain collection fires in both branches
    # (Fortran nrcoll for liquid-fraction, nrcol for non-liquid-fraction).
    rain_warm_n = safe_divide(nʳ * rain_warm_q_full, qʳ, zero(FT))
    # Mass transfer of collected rain into qʷⁱ only happens in the liquid-fraction
    # branch (Fortran qrcoll). The non-liquid path explicitly leaves rain mass alone
    # — see microphy_p3.f90:3055-3066, "collection of rain above freezing does not
    # impact total rain mass" — so zero out rain_warm_q in that case.
    rain_warm_q = ifelse(prp.liquid_fraction_active, rain_warm_q_full, zero(FT))

    return P3Phase2Rates{FT}(
        agg, ni_lim,
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        wg_cloud, wg_rain, wg_shed, wg_shed_n, wg_densif_mass, wg_densif_vol,
        shed, shed_n, refrz, complete_melt,
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        spl_q, spl_n,
        cloud_hom_q, cloud_hom_n, rain_hom_q, rain_hom_n,
        cloud_warm_q, cloud_warm_n, rain_warm_q, rain_warm_n,
        D_mean, Fˡ
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
@noinline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants,
                                            props = nothing)
    FT = typeof(ρ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # === SETUP ===
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nʳ = ℳ.nʳ
    qⁱ = ℳ.qⁱ
    nⁱ = ℳ.nⁱ
    qʷⁱ = ℳ.qʷⁱ

    nⁱ = min(nⁱ, prp.maximum_ice_number_density / ρ)

    rain_active = (qʳ > FT(1e-14)) & (nʳ > FT(1e-16))
    qʳ_pos = clamp_positive(qʳ)
    # rain_slope_parameter and consistent_rime_state are pure functions of (ℳ, prp);
    # when props is supplied (hot path from _p3_scalar_compute / p3_rates_and_properties)
    # we reuse the values already computed in p3_ice_properties.
    λ_r = isnothing(props) ? rain_slope_parameter(qʳ_pos, clamp_positive(nʳ), prp) : props.λ_r
    nʳ = ifelse(rain_active, rain_number_from_slope(qʳ_pos, λ_r, prp), nʳ)

    qᶠ, bᶠ, Fᶠ, ρᶠ = if isnothing(props)
        rs = consistent_rime_state(p3, qⁱ, ℳ.qᶠ, ℳ.bᶠ, qʷⁱ)
        rs.qᶠ, rs.bᶠ, rs.Fᶠ, rs.ρᶠ
    else
        props.qᶠ, props.bᶠ, props.Fᶠ, props.ρᶠ
    end

    qⁱ_total_mu = isnothing(props) ?
                  max(clamp_positive(qⁱ) + clamp_positive(qʷⁱ), FT(1e-20)) :
                  props.qⁱ_total
    Fˡ_mu = isnothing(props) ? (clamp_positive(qʷⁱ) / qⁱ_total_mu) : props.Fˡ
    # μ_ice is still recomputed here because props.μ_ice uses props.nⁱ which is
    # zeroed in the no-ice case, whereas the local nⁱ above is just clamp-capped.
    # The two values agree in cells with ice (the cells that matter for rates).
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_total_mu, nⁱ, ℳ.zⁱ, Fᶠ, Fˡ_mu, ρᶠ)

    T = temperature(𝒰, constants)
    q_base = 𝒰.moisture_mass_fractions
    qᵛ_base = q_base.vapor
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    P = 𝒰.reference_pressure

    ssat_adjustment = predicted_supersaturation_adjustment(p3, qᶜˡ, qᵛ_base, qᵛ⁺ˡ, ℳ.sˢᵃᵗ, T, constants)
    cond_GM = ssat_adjustment.rate
    qᶜˡ = ssat_adjustment.qᶜˡ
    qᵛ = ssat_adjustment.qᵛ
    T = ssat_adjustment.T
    q = MoistureMassFractions(qᵛ, q_base.liquid + ssat_adjustment.ε, q_base.ice)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    transport = air_transport_properties(T, P)

    cloud = diagnose_cloud_dsd(p3, qᶜˡ, ℳ.nᶜˡ, ρ)
    Nᶜ = cloud.Nᶜ
    ℳ_adjusted = P3MicrophysicalState(qᶜˡ, ℳ.nᶜˡ, qʳ, ℳ.nʳ, qⁱ, ℳ.nⁱ,
                                      qᶠ, bᶠ, ℳ.zⁱ, qʷⁱ, qᵛ - qᵛ⁺ˡ, ℳ.nᵃ, ℳ.w)

    # Hoist cᵖᵐ once; shared by coupled_saturation_adjustment_rates and ccn_activation_rate.
    cᵖᵐ = mixture_heat_capacity(q, constants)

    # Build derived state struct (explicit type parameters to avoid
    # jl_f_throw_methoderror in @noinline GPU compilation)
    state = P3DerivedState{FT, typeof(q)}(nⁱ, nʳ, qᶠ, bᶠ, Fᶠ, ρᶠ,
                                          μ_ice, Fˡ_mu, Nᶜ, cloud.nᶜˡ,
                                          cloud.μ_c, cloud.λ_c,
                                          T, P, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, q,
                                          transport.D_v, transport.K_a, transport.nu, cᵖᵐ)

    # === PHASE 1 & 2 RATES (delegated to @noinline sub-functions) ===
    ph1 = _p3_phase1_rates(p3, ρ, ℳ_adjusted, constants, state)
    ph2 = _p3_phase2_rates(p3, ρ, ℳ_adjusted, constants, state, ph1)

    # === EXTRACT RATES INTO LOCAL VARIABLES FOR SINK LIMITING ===
    # Phase 1
    cond = ph1.condensation
    ccn_act = ph1.ccn_activation_mass
    ccn_act_n = ph1.ccn_activation_number
    autoconv = ph1.autoconversion
    accr = ph1.accretion
    cloud_self = ph1.cloud_self_collection
    rain_evap = ph1.rain_evaporation
    rain_cond = ph1.rain_condensation
    rain_self = ph1.rain_self_collection
    rain_br = ph1.rain_breakup
    dep = ph1.deposition
    coat_cond = ph1.coating_condensation
    coat_evap = ph1.coating_evaporation
    partial_melt = ph1.partial_melting
    complete_melt = ph2.complete_melting  # NOTE: Phase 2 modified this with clipping
    melt_n = ph1.melting_number

    # Phase 2
    agg = ph2.aggregation
    ni_lim = ph2.ni_limit
    cloud_rim = ph2.cloud_riming
    cloud_rim_n = ph2.cloud_riming_number
    rain_rim = ph2.rain_riming
    rain_rim_n = ph2.rain_riming_number
    ρᶠ_new = ph2.rime_density_new
    wg_cloud = ph2.wet_growth_cloud
    wg_rain = ph2.wet_growth_rain
    wg_shed = ph2.wet_growth_shedding
    wg_shed_n = ph2.wet_growth_shedding_number
    wg_densif_mass = ph2.wet_growth_densification_mass
    wg_densif_vol = ph2.wet_growth_densification_volume
    shed = ph2.shedding
    shed_n = ph2.shedding_number
    refrz = ph2.refreezing
    nuc_q = ph2.nucleation_mass
    nuc_n = ph2.nucleation_number
    cloud_frz_q = ph2.cloud_freezing_mass
    cloud_frz_n = ph2.cloud_freezing_number
    rain_frz_q = ph2.rain_freezing_mass
    rain_frz_n = ph2.rain_freezing_number
    spl_q = ph2.splintering_mass
    spl_n = ph2.splintering_number
    cloud_hom_q = ph2.cloud_homogeneous_mass
    cloud_hom_n = ph2.cloud_homogeneous_number
    rain_hom_q = ph2.rain_homogeneous_mass
    rain_hom_n = ph2.rain_homogeneous_number
    cloud_warm_q = ph2.cloud_warm_collection
    cloud_warm_n = ph2.cloud_warm_collection_number
    rain_warm_q = ph2.rain_warm_collection
    rain_warm_n = ph2.rain_warm_collection_number

    # === SINK LIMITING ===
    dt_safety = prp.sink_limiting_timescale

    # --- Vapor sinks ---
    # Fortran applies the saturation-adjustment caps before the per-species
    # conservation budgets (microphy_p3.f90:3990-4055, then 4061 onward), so
    # cloud/rain/ice budgets below must see the final vapor-limited rates.
    qᵗ = q.vapor + q.liquid + q.ice
    vapor_rates = limit_vapor_rates(cond, ccn_act, ccn_act_n, rain_cond, rain_evap,
                                    dep, coat_cond, coat_evap, nuc_q, nuc_n,
                                    qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, dt_safety)
    cond = vapor_rates.cond
    ccn_act = vapor_rates.ccn_act
    ccn_act_n = vapor_rates.ccn_act_n
    rain_cond = vapor_rates.rain_cond
    rain_evap = vapor_rates.rain_evap
    dep = vapor_rates.dep
    coat_cond = vapor_rates.coat_cond
    coat_evap = vapor_rates.coat_evap
    nuc_q = vapor_rates.nuc_q
    nuc_n = vapor_rates.nuc_n

    # --- Cloud liquid sinks ---
    # Match Fortran's per-species conservation budget (microphy_p3.f90:4060-4083),
    # which splits signed `qccon` into non-negative `qccon` (source) and `qcevp`
    # (sink) and includes `qcevp` in the cloud sink total. Track the negative
    # part of `cond` as a sink magnitude here so it gets rescaled alongside the
    # other cloud sinks when the budget would over-deplete `qᶜˡ`.
    cloud_evap = clamp_positive(-cond)
    cloud_source_total = clamp_positive(cond) + ccn_act
    cloud_available = max(0, qᶜˡ) + cloud_source_total * dt_safety
    cloud_sink_total = autoconv + accr + cloud_rim + cloud_frz_q +
                       cloud_hom_q + cloud_warm_q + wg_cloud + cloud_evap
    f_cloud = sink_limiting_factor(cloud_sink_total, cloud_available, dt_safety)
    autoconv      = autoconv * f_cloud
    accr          = accr * f_cloud
    cloud_rim     = cloud_rim * f_cloud
    cloud_rim_n   = cloud_rim_n * f_cloud
    cloud_frz_q   = cloud_frz_q * f_cloud
    cloud_frz_n   = cloud_frz_n * f_cloud
    cloud_hom_q   = cloud_hom_q * f_cloud
    cloud_hom_n   = cloud_hom_n * f_cloud
    cloud_warm_q  = cloud_warm_q * f_cloud
    cloud_warm_n  = cloud_warm_n * f_cloud
    wg_cloud      = wg_cloud * f_cloud
    cond          = ifelse(cond < 0, cond * f_cloud, cond)

    # --- Rain sinks ---
    rain_source_total = autoconv + accr + complete_melt + shed + wg_shed + rain_cond
    rain_available = max(0, qʳ) + rain_source_total * dt_safety
    rain_sink_total = rain_rim + rain_frz_q + rain_hom_q + rain_warm_q + wg_rain + rain_evap
    f_rain = sink_limiting_factor(rain_sink_total, rain_available, dt_safety)
    rain_rim      = rain_rim * f_rain
    rain_rim_n    = rain_rim_n * f_rain
    rain_frz_q    = rain_frz_q * f_rain
    rain_frz_n    = rain_frz_n * f_rain
    rain_hom_q    = rain_hom_q * f_rain
    rain_hom_n    = rain_hom_n * f_rain
    rain_warm_q   = rain_warm_q * f_rain
    rain_warm_n   = rain_warm_n * f_rain
    wg_rain       = wg_rain * f_rain
    rain_evap     = rain_evap * f_rain

    # Sublimation number loss
    sublim_mag = clamp_positive(-dep)
    sublim_n = sublim_mag * safe_divide(clamp_positive(nⁱ), max(clamp_positive(qⁱ), FT(1e-20)), zero(FT))

    # Wet-ice coating condensation/evaporation comes from the coupled
    # saturation adjustment (P3CoupledVaporRates). The dry/wet exclusivity is
    # enforced inside that formula via εⁱ / εⁱʷ activation.

    # --- Total-ice (qⁱ + qʷⁱ) sink limiting ---
    # Matches Fortran's single qitot budget at microphy_p3.f90:4106-4136. The
    # paired qʷⁱ-only budget below mirrors Fortran's qiliq budget at 4138-4170,
    # so `shed` / `coat_evap` are deliberately scaled in both stages (`qlshd`
    # / `qlevp` are sinks of both qitot and qiliq in Fortran). `partial_melt`
    # is not scaled here because `qimlt` is invisible to qitot in Fortran (it
    # transfers mass dry → coating without changing the total).
    total_ice_source_total = max(0, dep) + cloud_rim + rain_rim +
                             nuc_q + cloud_frz_q + rain_frz_q +
                             cloud_hom_q + rain_hom_q +
                             cloud_warm_q + rain_warm_q +
                             wg_cloud + wg_rain + coat_cond
    total_ice_available = max(total_ice_mass(qⁱ, qʷⁱ), FT(0)) + total_ice_source_total * dt_safety
    total_ice_sink_total = complete_melt + clamp_positive(-dep) + shed + coat_evap
    f_total_ice = sink_limiting_factor(total_ice_sink_total, total_ice_available, dt_safety)
    complete_melt = complete_melt * f_total_ice
    melt_n        = melt_n * f_total_ice
    dep           = ifelse(dep < 0, dep * f_total_ice, dep)
    sublim_n      = sublim_n * f_total_ice
    shed          = shed * f_total_ice
    shed_n        = shed_n * f_total_ice
    coat_evap     = coat_evap * f_total_ice

    # --- qʷⁱ sinks ---
    qwi_source_total = partial_melt + cloud_warm_q + rain_warm_q + wg_cloud + wg_rain + coat_cond
    qwi_available = max(0, qʷⁱ) + qwi_source_total * dt_safety
    qwi_sink_total = shed + refrz + wg_shed + coat_evap
    f_qwi = sink_limiting_factor(qwi_sink_total, qwi_available, dt_safety)
    shed      = shed * f_qwi
    shed_n    = shed_n * f_qwi
    refrz     = refrz * f_qwi
    wg_shed   = wg_shed * f_qwi
    wg_shed_n = wg_shed_n * f_qwi
    coat_evap = coat_evap * f_qwi

    qⁱ_total_coat = max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20))
    coat_evap_n = coat_evap * safe_divide(clamp_positive(nⁱ), qⁱ_total_coat, zero(FT))
    sublim_n = sublim_n + coat_evap_n

    # Recompute splintering from sink-limited riming rates
    D_mean = ph2.D_mean
    Fˡ = ph2.Fˡ
    cloud_spl_q, rain_spl_q, spl_n = rime_splintering_rates(p3, cloud_rim, rain_rim, T, D_mean, Fˡ, T, qᶠ)
    cloud_spl_q = min(cloud_spl_q, clamp_positive(cloud_rim))
    rain_spl_q = min(rain_spl_q, clamp_positive(rain_rim))
    spl_q = cloud_spl_q + rain_spl_q

    # DSD number correction feedback
    ncl_correction = (cloud.nᶜˡ - ℳ.nᶜˡ) / dt_safety
    nr_correction = (nʳ - ℳ.nʳ) / dt_safety

    # Fortran's `microphy_p3.f90:5053-5063` recomputes ssat = qv - qvs(T)
    # at the end of the substep. Breeze must diagnose that final T from the
    # same conserved thermodynamic variable (`ρθ` or `ρe`) that the host model
    # advances, not from a standalone cᵖᵈ latent-heating estimate.
    vapor_to_liquid = cond + ccn_act + rain_cond + coat_cond - rain_evap - coat_evap
    vapor_to_ice = dep + nuc_q
    liquid_to_ice = cloud_rim + rain_rim + cloud_frz_q + rain_frz_q +
                    cloud_hom_q + rain_hom_q + refrz -
                    complete_melt - partial_melt
    qᵛ_final = qᵛ - (vapor_to_liquid + vapor_to_ice) * dt_safety
    qˡ_final = q.liquid + (vapor_to_liquid - liquid_to_ice) * dt_safety
    qⁱ_final = q.ice + (vapor_to_ice + liquid_to_ice) * dt_safety
    ssat_final = liquid_supersaturation_after_moisture_update(𝒰, qᵛ_final, qˡ_final,
                                                              qⁱ_final, ρ, constants)
    ssat_tendency = (ssat_final - ℳ.sˢᵃᵗ) / dt_safety
    ssat_tendency = ifelse(prp.predict_supersaturation, ssat_tendency, zero(FT))
    # `cond_GM` is intentionally NOT rescaled by the cloud sink limiter: the
    # G&M alignment is its own one-shot saturation adjustment with a local
    # `ε ≥ -qᶜˡ` cap, and the cloud budget at the limiter sees `qᶜˡ_adjusted`
    # (= qᶜˡ + ε) as its starting state — so ε is absorbed into
    # `cloud_available`, not the source/sink list.
    cond_total = cond + cond_GM

    return P3ProcessRates{FT}(
        cond_total,
        autoconv, accr, cloud_self, rain_evap, rain_self, rain_br,
        dep, partial_melt, complete_melt, melt_n,
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
        ccn_act, ccn_act_n, rain_cond,
        coat_cond, coat_evap,
        wg_densif_mass, wg_densif_vol,
        ncl_correction, nr_correction,
        cond_GM, ssat_tendency,
    )
end
