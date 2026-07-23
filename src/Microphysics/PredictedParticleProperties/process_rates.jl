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
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             PlanarLiquidSurface,
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
nucleation, and warm collection. Homogeneous freezing is diagnosed
separately in `compute_p3_process_rates` from the post-process liquid residual.
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
    melting_number :: FT
    whole_particle_clipping :: Bool
    nucleation_mass :: FT
    nucleation_number :: FT
    cloud_freezing_mass :: FT
    cloud_freezing_number :: FT
    rain_freezing_mass :: FT
    rain_freezing_number :: FT
    splintering_mass :: FT
    splintering_number :: FT
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
    rain_evaporation_number :: FT  # Rain number loss from evaporation [1/kg/s]
    rain_self_collection :: FT     # Rain number loss magnitude [1/kg/s]
    rain_breakup :: FT             # Rain number gain from breakup [1/kg/s]

    # Phase 1: Ice tendencies (BIDIRECTIONAL deposition; positive melting/number)
    deposition :: FT               # Vapor ↔ ice mass [kg/kg/s] (+dep, −sublim)
    partial_melting :: FT          # Ice → liquid coating (stays on ice) [kg/kg/s]
    complete_melting :: FT         # Ice → rain mass (sheds) [kg/kg/s]
    melting_number :: FT           # Ice number loss magnitude from melting [1/kg/s]
    clipping_dry_mass :: FT        # Whole-particle clip contribution to complete melting [kg/kg/s]
    clipping_rime_mass :: FT       # Rime mass removed exactly by whole-particle clips [kg/kg/s]
    clipping_rime_volume :: FT     # Rime volume removed exactly by whole-particle clips [m³/kg/s]
    post_process_clipping :: FT    # One when the post-process liquid-fraction clip fires

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

    # Above-freezing collection (T > T₀). Cloud collection goes to qʷⁱ in
    # liquid-fraction mode and sheds to rain otherwise; rain mass transfers only to qʷⁱ.
    cloud_warm_collection :: FT        # Cloud collected above T₀ [kg/kg/s]
    cloud_warm_collection_number :: FT # Cloud number loss from warm collection [1/kg/s]
    rain_warm_collection :: FT         # Rain collected above T₀ → qʷⁱ [kg/kg/s]
    rain_warm_collection_number :: FT  # M9: Rain number loss from warm collection [1/kg/s]

    # Liquid-fraction wet growth: collected hydrometeors redirected to qʷⁱ when
    # collection exceeds freezing capacity (Fortran qwgrth1c/qwgrth1r).
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
    ice_number_correction :: FT    # (nⁱ_lambda_bounded - nⁱ_global_bounded) / τ [1/kg/s]

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

# Preserve the original 54-rate positional constructor used by downstream
# diagnostics. New number-limiter and clip-accounting fields default to zero in
# that compatibility path.
function P3ProcessRates{FT}(first_rate::Number,
                            remaining_rates::Vararg{Number, 53}) where FT
    converted_rates = map(FT, remaining_rates)
    return P3ProcessRates{FT}(FT(first_rate), converted_rates[1:4]..., zero(FT),
                              converted_rates[5:10]...,
                              zero(FT), zero(FT), zero(FT), zero(FT),
                              converted_rates[11:53]...)
end

P3ProcessRates(first_rate::FT, remaining_rates::Vararg{FT, 53}) where FT =
    P3ProcessRates{FT}(first_rate, remaining_rates...)

@noinline function _p3_phase1_rates(p3, ρ, ℳ, constants, state::P3DerivedState,
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
    μ_ice = state.μ_ice
    Fˡ_mu = state.Fˡ_mu
    Nᶜ = state.Nᶜ
    nⁱ = state.nⁱ
    nʳ = state.nʳ
    P = state.P

    # Transport properties (reconstructed as NamedTuple for existing function signatures)
    transport = (; D_v = state.D_v, K_a = state.K_a, nu = state.nu)
    cᵖᵐ = state.cᵖᵐ
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)

    # =========================================================================
    # Coupled cloud/rain/ice vapor growth and decay
    # =========================================================================
    vapor_rates = coupled_saturation_adjustment_rates(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.qʳ, nʳ,
                                                      ℳ.qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ,
                                                      Fᶠ, ρᶠ, T, P, ρ, constants,
                                                      transport, q, μ_ice,
                                                      state.μ_c, state.λ_c, state.nᶜˡ,
                                                      temperature_tendency, vapor_tendency)
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
    # Fortran gates both dry-ice deposition and wet-coating vapor exchange on
    # qitot, which includes the liquid coating. In Julia that reservoir is
    # qⁱ + qʷⁱ; gating on qⁱ alone incorrectly disables vapor exchange
    # for nearly melted, liquid-coated particles.
    has_total_ice = total_ice_mass(ℳ.qⁱ, qʷⁱ) >= p3.minimum_mass_mixing_ratio
    dep = ifelse(has_total_ice, vapor_rates.deposition, zero(FT))

    liquid_fraction_active = p3.process_rates.liquid_fraction_active
    wet_ice_exchange_active = has_total_ice & liquid_fraction_active
    coat_cond = ifelse(wet_ice_exchange_active,
                       vapor_rates.coating_condensation, zero(FT))
    coat_evap = ifelse(wet_ice_exchange_active,
                       vapor_rates.coating_evaporation, zero(FT))

    melt_rates = ice_melting_rates(p3, ℳ.qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺ˡ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)
    partial_melt = melt_rates.partial_melting
    complete_melt = melt_rates.complete_melting
    complete_melt = ifelse(p3.process_rates.liquid_fraction_active,
                           complete_melt, complete_melt + partial_melt)
    partial_melt = ifelse(p3.process_rates.liquid_fraction_active,
                          partial_melt, zero(FT))
    melt_n = ice_melting_number_rate(ℳ.qⁱ, nⁱ, complete_melt)

    return P3Phase1Rates{FT}(cond, ccn_act, ccn_act_n,
                             autoconv, accr, cloud_self,
                             rain_evap, rain_cond, rain_self, rain_br,
                             dep, coat_cond, coat_evap,
                             partial_melt, complete_melt, melt_n)
end

@noinline function _p3_phase2_rates(p3, ρ, ℳ, constants, state::P3DerivedState,
                                   phase1::P3Phase1Rates)
    return _p3_phase2_rates(p3, ρ, ℳ, constants, state, phase1, state.T)
end

@noinline function _p3_phase2_rates(p3, ρ, ℳ, constants, state::P3DerivedState,
                                   phase1::P3Phase1Rates, surface_temperature)
    nⁱ_global = min(clamp_positive(ℳ.nⁱ),
                    p3.process_rates.maximum_ice_number_density / ρ)
    nⁱ_diagnostic = max(nⁱ_global, p3.minimum_number_mixing_ratio)
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
    qⁱ_total = max(total_ice_mass(ℳ.qⁱ, qʷⁱ), typeof(ρ)(1e-20))
    ρ_mean = ice_mean_density(p3, qⁱ_total, nⁱ_diagnostic, ℳ.zⁱ,
                              state.Fᶠ, state.Fˡ_mu, state.ρᶠ, state.μ_ice)
    return _p3_phase2_rates(p3, ρ, ℳ, constants, state, phase1,
                            surface_temperature, nⁱ_diagnostic, ρ_mean)
end

@noinline function _p3_phase2_rates(p3, ρ, ℳ, constants, state::P3DerivedState,
                                   phase1::P3Phase1Rates, surface_temperature,
                                   nⁱ_diagnostic, ρ_mean)
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
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
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

    liquid_fraction_wet_growth = prp.liquid_fraction_active & is_wet_growth
    dry_wet_growth = !prp.liquid_fraction_active & is_wet_growth
    retained_fraction = clamp(safe_divide(qwgrth, total_collection, zero(FT)), 0, 1)
    retained_cloud = cloud_rim * retained_fraction
    retained_rain = rain_rim * retained_fraction
    excess_cloud = cloud_rim - retained_cloud
    excess_rain = rain_rim - retained_rain

    # With prognostic liquid fraction, all collection becomes liquid coating. Without
    # it, the freezing-capacity portion becomes dense rime while only excess cloud
    # water is a new rain-mass source; excess collected rain simply returns to rain.
    wg_cloud = ifelse(liquid_fraction_wet_growth, cloud_rim, zero(FT))
    wg_rain = ifelse(liquid_fraction_wet_growth, rain_rim, zero(FT))
    cloud_rim = ifelse(liquid_fraction_wet_growth, zero(FT),
                       ifelse(dry_wet_growth, retained_cloud, cloud_rim))
    rain_rim = ifelse(liquid_fraction_wet_growth, zero(FT),
                      ifelse(dry_wet_growth, retained_rain, rain_rim))
    wg_shed = ifelse(dry_wet_growth, excess_cloud, zero(FT))
    wg_shed_n = ifelse(dry_wet_growth,
                       (excess_cloud + excess_rain) / prp.shed_drop_mass,
                       zero(FT))
    ρᶠ_new = ifelse(dry_wet_growth, prp.maximum_rime_density, ρᶠ_new)

    # Wet growth rime densification
    ρ_rimemax = prp.maximum_rime_density
    τ_densif = prp.rime_densification_timescale
    qⁱ_safe = clamp_positive(qⁱ)
    bᶠ_safe = max(bᶠ, FT(1e-20))
    wg_densif_active = dry_wet_growth & (qⁱ_safe > FT(1e-14))
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
    # Fortran diam_ice is the volume-equivalent diameter diagnosed from the
    # tabulated bulk mean density f1pr16, not a single-particle inversion of
    # the piecewise mass law. Table 3 supplies f1pr16 in three-moment mode;
    # Table 1 supplies it otherwise.
    diagnostic_mean_mass = qⁱ_total / nⁱ_diagnostic
    D_mean = cbrt(6 * diagnostic_mean_mass / (FT(π) * ρ_mean))

    shed = shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean, μ_ice)
    shed_n = shedding_number_rate(p3, shed)
    refrz = refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)
    shed = ifelse(prp.liquid_fraction_active, shed, zero(FT))
    shed_n = ifelse(prp.liquid_fraction_active, shed_n, zero(FT))
    refrz = ifelse(prp.liquid_fraction_active, refrz, zero(FT))

    # Liquid fraction clipping
    Fl_small = prp.liquid_fraction_clipping_threshold
    τ_clip = prp.refreezing_timescale
    qʷⁱ_eff = clamp_positive(qʷⁱ)
    clip_freeze = prp.liquid_fraction_active & (T < T₀) & (Fˡ < Fl_small) & (Fˡ > 0)
    refrz = ifelse(clip_freeze, refrz + qʷⁱ_eff / τ_clip, refrz)

    # Whole-particle liquid-fraction and tiny-warm-ice clips. These predicates can
    # overlap, so form their union and transfer each reservoir exactly once.
    qⁱ_dry = clamp_positive(qⁱ)
    qⁱ_total_clip = qⁱ_dry + qʷⁱ_eff
    has_clip_mass = qⁱ_total_clip >= p3.minimum_mass_mixing_ratio
    warm_liquid_clip = (T >= T₀) & (Fˡ > 1 - Fl_small) & has_clip_mass
    high_liquid_fraction_clip = (Fˡ > FT(0.99)) & has_clip_mass
    tiny_warm_ice = (T >= T₀) & has_clip_mass &
                    (qⁱ_total_clip < prp.tiny_ice_to_rain_threshold)
    liquid_fraction_clipping = prp.liquid_fraction_active &
                               (warm_liquid_clip | high_liquid_fraction_clip)
    whole_particle_clipping = liquid_fraction_clipping | tiny_warm_ice
    complete_melt = ifelse(whole_particle_clipping, qⁱ_dry / τ_clip,
                           phase1.complete_melting)
    melt_n = ifelse(whole_particle_clipping,
                    clamp_positive(ℳ.nⁱ) / τ_clip, phase1.melting_number)
    shed = ifelse(whole_particle_clipping, qʷⁱ_eff / τ_clip, shed)
    shed_n = ifelse(whole_particle_clipping, zero(FT), shed_n)
    refrz = ifelse(whole_particle_clipping, zero(FT), refrz)

    # =========================================================================
    # Ice nucleation
    # =========================================================================
    nucleation_existing_number = ifelse(whole_particle_clipping, zero(FT), nⁱ)
    nuc_q, nuc_n = deposition_nucleation_rate(
        p3, T, qᵛ, qᵛ⁺ⁱ, nucleation_existing_number, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    μ_r = zero(FT)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T, μ_r)

    # Rime splintering
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T, D_mean, Fˡ,
                                        surface_temperature, qᶠ)

    # Homogeneous freezing is diagnosed later from the post-process liquid residual
    # (see `compute_p3_process_rates`), so it is not computed here.

    # Above-freezing collection
    cloud_warm_q, _ = cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    cloud_warm_n = cloud_riming_number_rate(qᶜˡ, Nᶜ, ρ, cloud_warm_q)
    rain_warm_q_full = rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    # Number sink from above-freezing rain collection fires in both branches
    # (Fortran nrcoll for liquid-fraction, nrcol for non-liquid-fraction).
    rain_warm_n = rain_warm_collection_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    # Mass transfer of collected rain into qʷⁱ only happens in the liquid-fraction
    # branch (Fortran qrcoll). The non-liquid path explicitly leaves rain mass alone
    # — see microphy_p3.f90:3055-3066, "collection of rain above freezing does not
    # impact total rain mass" — so zero out rain_warm_q in that case.
    rain_warm_q = ifelse(prp.liquid_fraction_active, rain_warm_q_full, zero(FT))

    return P3Phase2Rates{FT}(
        agg, ni_lim,
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        wg_cloud, wg_rain, wg_shed, wg_shed_n, wg_densif_mass, wg_densif_vol,
        shed, shed_n, refrz, complete_melt, melt_n, whole_particle_clipping,
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        spl_q, spl_n,
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
@noinline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    surface_temperature = temperature(𝒰, constants)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing,
                                    surface_temperature, zero(ρ), zero(ρ))
end

@noinline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props)
    surface_temperature = temperature(𝒰, constants)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props,
                                    surface_temperature, zero(ρ), zero(ρ))
end

@noinline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props,
                                            surface_temperature)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props,
                                    surface_temperature, zero(ρ), zero(ρ))
end

@noinline function compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props,
                                            surface_temperature,
                                            temperature_tendency,
                                            vapor_tendency)
    FT = typeof(ρ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # === SETUP ===
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nʳ = ℳ.nʳ
    qⁱ = ℳ.qⁱ
    nⁱ_raw = ℳ.nⁱ
    qʷⁱ_prognostic = ℳ.qʷⁱ
    qʷⁱ = active_liquid_on_ice(p3, qʷⁱ_prognostic)
    qʷⁱ_budget = ifelse(prp.liquid_fraction_active, qʷⁱ,
                             clamp_positive(qʷⁱ_prognostic))

    nⁱ_global = min(clamp_positive(nⁱ_raw),
                    prp.maximum_ice_number_density / ρ)

    rain_active = qʳ > FT(1e-14)
    qʳ_pos = clamp_positive(qʳ)
    nʳ_floored = max(clamp_positive(nʳ), FT(1e-16))
    # rain_slope_parameter and consistent_rime_state are pure functions of (ℳ, prp);
    # when props is supplied (hot path from _p3_scalar_compute / p3_rates_and_properties)
    # we reuse the values already computed in p3_ice_properties.
    λ_r = isnothing(props) ? rain_slope_parameter(qʳ_pos, nʳ_floored, prp) : props.λ_r
    nʳ = ifelse(rain_active, rain_number_from_slope(qʳ_pos, λ_r, prp), zero(FT))

    qᶠ, bᶠ, Fᶠ, ρᶠ = if isnothing(props)
        rs = consistent_rime_state(p3, qⁱ, ℳ.qᶠ, ℳ.bᶠ, qʷⁱ)
        rs.qᶠ, rs.bᶠ, rs.Fᶠ, rs.ρᶠ
    else
        props.qᶠ, props.bᶠ, props.Fᶠ, props.ρᶠ
    end

    if isnothing(props)
        qⁱ_total_mu = max(clamp_positive(qⁱ) + clamp_positive(qʷⁱ), FT(1e-20))
        Fˡ_mu = clamp_positive(qʷⁱ) / qⁱ_total_mu
        nⁱ_diagnostic = max(nⁱ_global, p3.minimum_number_mixing_ratio)
        μ_ice = compute_ice_shape_parameter(p3, qⁱ_total_mu, nⁱ_diagnostic,
                                            ℳ.zⁱ, Fᶠ, Fˡ_mu, ρᶠ)
        ρ_mean = ice_mean_density(p3, qⁱ_total_mu, nⁱ_diagnostic, ℳ.zⁱ,
                                  Fᶠ, Fˡ_mu, ρᶠ, μ_ice)
        nⁱ = bounded_ice_number(p3, qⁱ_total_mu, nⁱ_diagnostic,
                                Fᶠ, Fˡ_mu, ρᶠ, μ_ice)
    else
        qⁱ_total_mu = props.qⁱ_total
        Fˡ_mu = props.Fˡ
        nⁱ = props.nⁱ
        nⁱ_diagnostic = props.nⁱ_diagnostic
        ρ_mean = props.ρ_mean
        μ_ice = props.μ_ice
    end

    T = temperature(𝒰, constants)
    q_base = 𝒰.moisture_mass_fractions
    qᵛ_base = q_base.vapor
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    P = p3_air_pressure(𝒰, constants)

    ssat_adjustment = predicted_supersaturation_adjustment(p3, qᶜˡ, qᵛ_base, qᵛ⁺ˡ, ℳ.sˢᵃᵗ, T, constants)
    cond_GM = ssat_adjustment.rate
    qᶜˡ = ssat_adjustment.qᶜˡ
    qᵛ = ssat_adjustment.qᵛ
    T = ssat_adjustment.T
    q = MoistureMassFractions(qᵛ, q_base.liquid + ssat_adjustment.ε, q_base.ice)
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = p3_ice_saturation_specific_humidity(T, ρ, constants, T₀, qᵛ⁺ˡ)
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
    ph1 = _p3_phase1_rates(p3, ρ, ℳ_adjusted, constants, state,
                           temperature_tendency, vapor_tendency)
    ph2 = _p3_phase2_rates(p3, ρ, ℳ_adjusted, constants, state, ph1,
                           surface_temperature, nⁱ_diagnostic, ρ_mean)

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
    partial_melt = ifelse(ph2.whole_particle_clipping, zero(FT), ph1.partial_melting)
    complete_melt = ph2.complete_melting  # NOTE: Phase 2 modified this with clipping
    melt_n = ph2.melting_number
    whole_particle_clipping = ph2.whole_particle_clipping

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
    wg_excess_rain = ifelse(whole_particle_clipping, zero(FT),
                            max(0, wg_shed_n * prp.shed_drop_mass - wg_shed))
    wg_densif_mass = ph2.wet_growth_densification_mass
    wg_densif_vol = ph2.wet_growth_densification_volume
    shed = ph2.shedding
    shed_n = ph2.shedding_number
    inactive_coating_cleanup = ifelse(
        prp.liquid_fraction_active, zero(FT),
        qʷⁱ_budget / prp.sink_limiting_timescale)
    shed = shed + inactive_coating_cleanup
    shed_n = shed_n + inactive_coating_cleanup / prp.shed_drop_mass
    refrz = ph2.refreezing
    nuc_q = ph2.nucleation_mass
    nuc_n = ph2.nucleation_number
    cloud_frz_q = ph2.cloud_freezing_mass
    cloud_frz_n = ph2.cloud_freezing_number
    rain_frz_q = ph2.rain_freezing_mass
    rain_frz_n = ph2.rain_freezing_number
    spl_q = ph2.splintering_mass
    spl_n = ph2.splintering_number
    cloud_warm_q = ph2.cloud_warm_collection
    cloud_warm_n = ph2.cloud_warm_collection_number
    rain_warm_q = ph2.rain_warm_collection
    rain_warm_n = ph2.rain_warm_collection_number

    # These clips correspond to Fortran's pre-process whole-particle transfers:
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
    spl_q = ifelse(whole_particle_clipping, zero(FT), spl_q)
    spl_n = ifelse(whole_particle_clipping, zero(FT), spl_n)
    cloud_warm_q = ifelse(whole_particle_clipping, zero(FT), cloud_warm_q)
    cloud_warm_n = ifelse(whole_particle_clipping, zero(FT), cloud_warm_n)
    rain_warm_q = ifelse(whole_particle_clipping, zero(FT), rain_warm_q)
    rain_warm_n = ifelse(whole_particle_clipping, zero(FT), rain_warm_n)

    # Track clip-only contributions separately so rime mass and volume can be
    # drained exactly instead of assuming the beginning-of-stage rime fraction.
    clipping_dry_mass = ifelse(whole_particle_clipping, complete_melt, zero(FT))
    clipping_rime_mass = ifelse(whole_particle_clipping, qᶠ / prp.refreezing_timescale,
                                zero(FT))
    clipping_rime_volume = ifelse(whole_particle_clipping, bᶠ / prp.refreezing_timescale,
                                  zero(FT))

    # === SINK LIMITING ===
    dt_safety = prp.sink_limiting_timescale

    # --- Vapor sinks ---
    # Fortran applies the saturation-adjustment caps before the per-species
    # conservation budgets (microphy_p3.f90:3990-4055, then 4061 onward), so
    # cloud/rain/ice budgets below must see the final vapor-limited rates.
    qᵗ = q.vapor + q.liquid + q.ice
    vapor_rates = limit_vapor_rates(cond, ccn_act, ccn_act_n, rain_cond, rain_evap,
                                    dep, coat_cond, coat_evap, nuc_q, nuc_n,
                                    qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, dt_safety, T₀)
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
    # Homogeneous freezing is applied after all ordinary process budgets below,
    # matching its ordering in Fortran P3. Do not reserve liquid for it here:
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

    cloud_warm_to_ice = ifelse(prp.liquid_fraction_active, cloud_warm_q, zero(FT))
    cloud_warm_to_rain = ifelse(prp.liquid_fraction_active, zero(FT), cloud_warm_q)

    # --- Rain sinks ---
    # As with cloud above, limit ordinary rain processes against the full rain
    # reservoir. Homogeneous freezing is diagnosed from the residual later.
    rain_source_total = autoconv + accr + complete_melt + shed + wg_shed +
                        cloud_warm_to_rain + rain_cond
    rain_available = max(0, qʳ) + rain_source_total * dt_safety
    rain_sink_total = rain_rim + rain_frz_q + rain_warm_q + wg_rain + rain_evap
    f_rain = sink_limiting_factor(rain_sink_total, rain_available, dt_safety)
    rain_rim      = rain_rim * f_rain
    rain_rim_n    = rain_rim_n * f_rain
    rain_frz_q    = rain_frz_q * f_rain
    rain_frz_n    = rain_frz_n * f_rain
    rain_warm_q   = rain_warm_q * f_rain
    rain_warm_n   = rain_warm_n * f_rain
    wg_rain       = wg_rain * f_rain
    rain_evap     = rain_evap * f_rain
    wg_excess_rain = wg_excess_rain * f_rain
    wg_shed_n = (wg_shed + wg_excess_rain) / prp.shed_drop_mass

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
                             cloud_warm_to_ice + rain_warm_q +
                             wg_cloud + wg_rain + coat_cond
    total_ice_available = max(total_ice_mass(qⁱ, qʷⁱ_budget), FT(0)) + total_ice_source_total * dt_safety
    total_ice_sink_total = complete_melt + clamp_positive(-dep) + shed + coat_evap
    f_total_ice = sink_limiting_factor(total_ice_sink_total, total_ice_available, dt_safety)
    complete_melt = complete_melt * f_total_ice
    melt_n        = melt_n * f_total_ice
    clipping_dry_mass = clipping_dry_mass * f_total_ice
    clipping_rime_mass = clipping_rime_mass * f_total_ice
    clipping_rime_volume = clipping_rime_volume * f_total_ice
    dep           = ifelse(dep < 0, dep * f_total_ice, dep)
    sublim_n      = sublim_n * f_total_ice
    shed          = shed * f_total_ice
    shed_n        = shed_n * f_total_ice
    coat_evap     = coat_evap * f_total_ice

    # --- qʷⁱ sinks ---
    qwi_source_total = partial_melt + cloud_warm_to_ice + rain_warm_q +
                       wg_cloud + wg_rain + coat_cond
    qwi_available = max(0, qʷⁱ_budget) + qwi_source_total * dt_safety
    qwi_sink_total = shed + refrz + coat_evap
    f_qwi = sink_limiting_factor(qwi_sink_total, qwi_available, dt_safety)
    shed      = shed * f_qwi
    shed_n    = shed_n * f_qwi
    refrz     = refrz * f_qwi
    coat_evap = coat_evap * f_qwi

    # A whole-particle clip transfers dry mass, coating, and number together. If
    # competition for coating water imposes an additional limiter, apply that same
    # factor to the dry-mass and number companions.
    complete_melt = ifelse(whole_particle_clipping, complete_melt * f_qwi,
                           complete_melt)
    melt_n = ifelse(whole_particle_clipping, melt_n * f_qwi, melt_n)
    clipping_dry_mass = ifelse(whole_particle_clipping,
                               clipping_dry_mass * f_qwi, clipping_dry_mass)
    clipping_rime_mass = ifelse(whole_particle_clipping,
                                clipping_rime_mass * f_qwi, clipping_rime_mass)
    clipping_rime_volume = ifelse(whole_particle_clipping,
                                  clipping_rime_volume * f_qwi,
                                  clipping_rime_volume)

    # Rain, dry ice, total ice, and coating exchange mass with one another.
    # A single sequential limiter pass can credit a source that a later donor
    # limiter subsequently reduces. Re-project the four donor budgets a
    # configurable number of times; every projection only reduces rates, so
    # this converges monotonically while remaining allocation-free and GPU-safe.
    for _ in 1:prp.coupled_sink_limiting_iterations
        dry_ice_source_total = max(0, dep) + cloud_rim + rain_rim + refrz +
                               nuc_q + cloud_frz_q + rain_frz_q
        dry_ice_available = max(0, qⁱ) + dry_ice_source_total * dt_safety
        dry_ice_sink_total = partial_melt + complete_melt + clamp_positive(-dep)
        f_dry_ice = sink_limiting_factor(dry_ice_sink_total, dry_ice_available,
                                         dt_safety)
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
        wg_shed_n = (wg_shed + wg_excess_rain) / prp.shed_drop_mass

        total_ice_source_total = max(0, dep) + cloud_rim + rain_rim +
                                 nuc_q + cloud_frz_q + rain_frz_q +
                                 cloud_warm_to_ice + rain_warm_q +
                                 wg_cloud + wg_rain + coat_cond
        total_ice_available = max(total_ice_mass(qⁱ, qʷⁱ_budget), FT(0)) +
                              total_ice_source_total * dt_safety
        total_ice_sink_total = complete_melt + clamp_positive(-dep) + shed + coat_evap
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

    qⁱ_total_coat = max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20))
    coat_evap_n = coat_evap * safe_divide(clamp_positive(nⁱ), qⁱ_total_coat, zero(FT))
    sublim_n = sublim_n + coat_evap_n

    # Recompute splintering from sink-limited riming rates
    D_mean = ph2.D_mean
    Fˡ = ph2.Fˡ
    cloud_spl_q, rain_spl_q, spl_n = rime_splintering_rates(
        p3, cloud_rim, rain_rim, T, D_mean, Fˡ, surface_temperature, qᶠ)
    cloud_spl_q = min(cloud_spl_q, clamp_positive(cloud_rim))
    rain_spl_q = min(rain_spl_q, clamp_positive(rain_rim))
    spl_q = cloud_spl_q + rain_spl_q

    # Reconstruct the ordinary post-process ice reservoirs. Fortran applies a
    # second Fˡ > 0.99 clip after these processes, so a particle that crosses
    # the threshold during melting must transfer its residual mass and number as
    # a whole. The dry-ice projection above guarantees these residuals are
    # non-negative before the clip is diagnosed.
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
    post_process_clipping_active = prp.liquid_fraction_active &
                                   (total_ice_remaining >= p3.minimum_mass_mixing_ratio) &
                                   (liquid_fraction_remaining > 1 - prp.liquid_fraction_clipping_threshold)

    # Rime companions are reconstructed with the same formulas used by the
    # prognostic tendencies, excluding homogeneous freezing, which occurs after
    # this clip in the Fortran ordering.
    ordinary_complete_melting = max(0, complete_melt - clipping_dry_mass)
    ordinary_total_melting = partial_melt + ordinary_complete_melting
    sublimation = clamp_positive(-dep)
    rime_mass_gain = cloud_rim + rain_rim + refrz + cloud_frz_q + rain_frz_q +
                     wg_densif_mass
    rime_mass_loss = Fᶠ * (ordinary_total_melting + sublimation) +
                     clipping_rime_mass
    rime_mass_remaining = max(0, qᶠ +
                                  (rime_mass_gain - rime_mass_loss) * dt_safety)

    rime_density_safe = max(ρᶠ, prp.minimum_rime_density)
    new_rime_density_safe = max(ρᶠ_new, prp.minimum_rime_density)
    maximum_rime_density = prp.maximum_rime_density
    rime_volume_gain = cloud_rim / new_rime_density_safe +
                       (rain_rim + refrz + cloud_frz_q + rain_frz_q) /
                       maximum_rime_density + wg_densif_vol
    rime_volume_loss = Fᶠ * (ordinary_total_melting + sublimation) /
                       rime_density_safe + clipping_rime_volume
    dry_ice_safe = max(qⁱ, FT(1e-12))
    rime_volume = Fᶠ * dry_ice_safe / rime_density_safe
    melt_densification = rime_volume * (prp.pure_ice_density - rime_density_safe) *
                          ordinary_total_melting / (rime_density_safe * dry_ice_safe)
    densification_active = (rime_density_safe < prp.pure_ice_density) &
                           !prp.liquid_fraction_active
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
    # the same species-budget ratio as rain_freezing_mass (Fortran qrheti/nrheti).
    # Project the remaining number-only sinks onto the population left afterward.
    cloud_warm_rain_number = ifelse(
        prp.liquid_fraction_active, zero(FT), cloud_warm_q / prp.shed_drop_mass)
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
    # Fortran applies homogeneous freezing after the ordinary process updates and
    # sedimentation (`microphy_p3.f90:4650-4757`). Sedimentation is advanced by the
    # host model in Breeze, but within the local process operator we can preserve
    # the essential ordering: first finalize every ordinary limiter above, then
    # freeze the liquid that remains. Re-diagnosing the rate from the residual also
    # captures liquid created by condensation, melting, or shedding during this
    # interval.
    cloud_sink_total = autoconv + accr + cloud_rim + cloud_frz_q +
                       cloud_warm_q + wg_cloud + wg_shed + clamp_positive(-cond)
    cloud_remaining = max(0, max(0, qᶜˡ) +
                                (cloud_source_total - cloud_sink_total) * dt_safety)

    rain_source_total = autoconv + accr + complete_melt + shed + wg_shed +
                        cloud_warm_to_rain + rain_cond
    rain_sink_total = rain_rim + rain_frz_q + rain_warm_q + wg_rain + rain_evap
    rain_remaining = max(0, max(0, qʳ) +
                               (rain_source_total - rain_sink_total) * dt_safety)

    # Diagnose the post-process number reservoirs as well, so frozen liquid carries
    # the number left by collection, breakup, melting, and activation rather than the
    # beginning-of-stage number. In the prescribed-Nᶜ path Fortran resets cloud
    # number to its prescribed value immediately before homogeneous freezing.
    cloud_number_tendency = cloud_number_tendency_before_homogeneous_freezing(
        p3, ρ, qᶜˡ, Nᶜ, ccn_act, ccn_act_n, autoconv, accr, cloud_self,
        cloud_rim_n, cloud_frz_n, cloud_warm_n)
    prognostic_cloud_number = max(0, cloud.nᶜˡ +
                                     cloud_number_tendency * dt_safety)
    prescribed_cloud_number = p3.cloud.number_concentration / ρ
    cloud_number_remaining = ifelse(isnothing(p3.aerosol), prescribed_cloud_number,
                                    prognostic_cloud_number)

    rain_number_tendency = rain_number_tendency_before_homogeneous_freezing(
        p3, nⁱ, qⁱ, nʳ, qʳ, autoconv, melt_n, rain_evap_n, rain_self,
        rain_br, rain_rim_n, rain_frz_n, shed_n, cloud_warm_q, rain_warm_n, wg_shed_n)
    rain_number_remaining = max(0, nʳ +
                                   rain_number_tendency * dt_safety)

    cloud_hom_q, cloud_hom_n = homogeneous_freezing_cloud_rate(
        p3, cloud_remaining, ρ * cloud_number_remaining, T, ρ)
    rain_hom_q, rain_hom_n = homogeneous_freezing_rain_rate(
        p3, rain_remaining, rain_number_remaining, T)

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
        ccn_act, ccn_act_n, rain_cond,
        coat_cond, coat_evap,
        wg_densif_mass, wg_densif_vol,
        ncl_correction, nr_correction, ni_correction,
        cond_GM, ssat_tendency,
    )
end
