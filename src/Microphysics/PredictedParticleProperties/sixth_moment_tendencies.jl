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

@inline function group2_ice_sixth_moment_tendency(rates::P3ProcessRates, prp::ProcessRateParameters, μ_r)
    FT = typeof(rates.nucleation_mass + μ_r)
    cloud_splintering_mass, rain_splintering_mass = split_splintering_mass(rates, prp)
    # Fortran P3 v5.5.0 uses diagnosed μ_r uniformly for group-2 initiated ice
    # (microphy_p3.f90 update_zi_proc2 calls around line 4483).
    return initiated_ice_sixth_moment_tendency(rates.nucleation_mass, rates.nucleation_number, μ_r) +
           initiated_ice_sixth_moment_tendency(rates.cloud_freezing_mass, rates.cloud_freezing_number, μ_r) +
           initiated_ice_sixth_moment_tendency(rates.rain_freezing_mass, rates.rain_freezing_number, μ_r) +
           initiated_ice_sixth_moment_tendency(rain_splintering_mass, rates.splintering_number, μ_r) +
           initiated_ice_sixth_moment_tendency(cloud_splintering_mass, rates.splintering_number, μ_r) +
           initiated_ice_sixth_moment_tendency(rates.cloud_homogeneous_mass, rates.cloud_homogeneous_number, μ_r) +
           initiated_ice_sixth_moment_tendency(rates.rain_homogeneous_mass, rates.rain_homogeneous_number, μ_r)
end

@inline function active_ice_sixth_moment_tendency(ice_table::P3IceIntegralsTable, p3, rates::P3ProcessRates,
                                                  ρ, qⁱ, qʷⁱ, nⁱ, qᶠ, bᶠ, zⁱ,
                                                  μ_ice, μ_r)
    FT = typeof(ρ)
    prp = p3.process_rates
    τ = max(prp.sink_limiting_timescale, eps(FT))

    splintering_mass = clamp_positive(rates.splintering_mass)

    # Fortran subtracts qcmul/qrmul from the group-1 reconstruction and adds
    # them back through update_zi_proc2 as group-2 initiated ice.
    group2_mass = rates.nucleation_mass + rates.cloud_freezing_mass +
                  rates.rain_freezing_mass + splintering_mass +
                  rates.cloud_homogeneous_mass +
                  rates.rain_homogeneous_mass
    group2_number = rates.nucleation_number + rates.cloud_freezing_number +
                    rates.rain_freezing_number + rates.splintering_number +
                    rates.cloud_homogeneous_number + rates.rain_homogeneous_number
    group2_rime_mass = rates.cloud_freezing_mass + rates.rain_freezing_mass +
                       splintering_mass + rates.cloud_homogeneous_mass +
                       rates.rain_homogeneous_mass
    group2_rime_volume = group2_rime_mass / max(prp.maximum_rime_density, eps(FT))
    current_rime_state = consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, qʷⁱ)

    qⁱ_rate = tendency_ρqⁱ(rates, one(FT)) - group2_mass
    qʷⁱ_rate = tendency_ρqʷⁱ(rates, one(FT))
    nⁱ_rate = tendency_ρnⁱ(rates, one(FT)) - group2_number
    qᶠ_rate = tendency_ρqᶠ(rates, one(FT), current_rime_state.Fᶠ) - group2_rime_mass
    bᶠ_rate = tendency_ρbᶠ(rates, one(FT), current_rime_state.Fᶠ, current_rime_state.ρᶠ, qⁱ, prp) -
              group2_rime_volume

    qⁱ_new = max(0, qⁱ + τ * qⁱ_rate)
    qʷⁱ_new = max(0, qʷⁱ + τ * qʷⁱ_rate)
    nⁱ_new = max(nⁱ + τ * nⁱ_rate, eps(FT))
    qᶠ_new = max(0, qᶠ + τ * qᶠ_rate)
    bᶠ_new = max(0, bᶠ + τ * bᶠ_rate)

    rime_state = consistent_rime_state(p3, qⁱ_new, qᶠ_new, bᶠ_new, qʷⁱ_new)
    Fˡ_new = liquid_fraction_on_ice(qⁱ_new, qʷⁱ_new)
    qⁱ_total_new = max(total_ice_mass(qⁱ_new, qʷⁱ_new), FT(1e-20))
    ρ_bulk_new = ice_mean_density_for_bounds(ice_table, qⁱ_total_new, nⁱ_new,
                                             rime_state.Fᶠ, Fˡ_new, rime_state.ρᶠ, μ_ice)
    M₃_new = FT(6) * qⁱ_total_new / (FT(π) * max(ρ_bulk_new, eps(FT)))
    zⁱ_new_raw = g_of_mu(μ_ice) * M₃_new^2 / max(nⁱ_new, eps(FT))
    has_group1_ice = (qⁱ_new > FT(1e-20)) & (nⁱ_new > eps(FT))
    zⁱ_new = ifelse(has_group1_ice, max(FT(1e-35), zⁱ_new_raw), zero(FT))

    z_group1 = (zⁱ_new - max(0, zⁱ)) / τ
    z_group2 = group2_ice_sixth_moment_tendency(rates, prp, μ_r)

    return ρ * (z_group1 + z_group2)
end

"""
$(TYPEDSIGNATURES)

Compute ice sixth moment tendency from P3 process rates.

The sixth moment (reflectivity) changes with:
- Deposition (growth) (Phase 1)
- Melting (loss) (Phase 1)
- Riming (growth) (Phase 2)
- Nucleation (growth) (Phase 2)
- Aggregation (redistribution) (Phase 2)

This simplified version uses proportional scaling (Z/q ratio).
For three-moment P3 with lookup tables, use `active_ice_sixth_moment_tendency`,
which mirrors Fortran's active fixed-μ reconstruction path.
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, prp::ProcessRateParameters, μ_r = zero(typeof(ρ)))
    FT = typeof(ρ)
    # Simplified: Z changes proportionally to mass changes
    # More accurate version would use full integral formulation
    ratio = safe_divide(zⁱ, qⁱ, zero(FT))

    # Net mass change for ice
    # Total melting (partial + complete) reduces ice mass
    total_melting = rates.partial_melting + rates.complete_melting
    mass_change = rates.deposition - total_melting +
                  rates.cloud_riming + rates.rain_riming + rates.refreezing +
                  rates.coating_condensation - rates.coating_evaporation
    z_group2 = group2_ice_sixth_moment_tendency(rates, prp, μ_r)

    return ρ * (ratio * mass_change + z_group2)
end

@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)
    prp = ProcessRateParameters(FT)
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, prp)
end

"""
$(TYPEDSIGNATURES)

Compute ice sixth moment tendency using tabulated integrals when available.

Following Milbrandt et al. (2021, 2024), the sixth moment tendency is
computed by integrating the contribution of each process over the
size distribution, properly accounting for how different processes
affect particles of different sizes.

Uses pre-computed lookup tables via `tabulate(p3, arch)` for
PSD-integrated sixth moment changes per process.

# Arguments
- `rates`: P3ProcessRates containing mass tendencies
- `ρ`: Air density [kg/m³]
- `qⁱ`: Ice mass mixing ratio [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `zⁱ`: Ice sixth moment [m⁶/kg]
- `Fᶠ`: Rime fraction [-]
- `Fˡ`: Liquid fraction [-]
- `p3`: P3 microphysics scheme (for accessing tabulated integrals)
- `nu`: Kinematic viscosity [m²/s]
- `D_v`: Water vapor diffusivity [m²/s]

# Returns
- Tendency of density-weighted sixth moment [kg/m³ × m⁶/kg / s]
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3, nu, D_v, μ, μ_cloud, λ_r = nothing)
    FT = typeof(ρ)

    # Mean ice particle mass for table lookup
    m̄ = safe_divide(qⁱ, nⁱ, FT(1e-20))
    log_mean_mass = log10(max(m̄, FT(1e-20)))

    # Schmidt number correction for enhanced ventilation integrals
    # Table stores 0.44 × ∫ C(D)√(V×D) N'(D) dD; runtime applies
    # Sc^(1/3) × √ρ_fac / √ν.
    # (see ventilation_sc_correction and deposition_ventilation for dimensional derivation)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
    sc_correction = ventilation_sc_correction(nu, D_v, ρ_correction)

    z_tendency = tabulated_z_tendency(
        p3.ice, log_mean_mass, Fᶠ, Fˡ, ρᶠ, rates, ρ, qⁱ, nⁱ, zⁱ,
        p3.process_rates, sc_correction, p3, μ, μ_cloud, λ_r
    )

    return z_tendency
end

# Tabulated version: use TabulatedFunction5D lookups for Z tendencies.
#
# Table convention (D33 verified: both Julia-native and Fortran-read paths are consistent):
# - Single-term processes (rime, aggregation, shedding): table stores raw dG (the
#   PSD-integrated sixth-moment kernel). Runtime: z_table × mass_rate / Nⁱ.
# - Two-term ventilation processes (deposition, sublimation, melting): table stores
#   raw dG/dt values. Runtime extracts environmental factors from mass_rate / mass_table
#   to avoid cross-term errors from the constant + enhanced ventilation split.
#
# Fortran convention:
#   epsiz = (m6dep + S_c*m6dep1) × 2πρDv              (dep/sub: raw dG × env)
#   zimlt = (vdepm1*m6mlt1 + vdepm2*m6mlt2*S_c) × thermo  (melt: mass_table × dG × env)
#   zqccol = m6rime × env_factors                      (rime: dG × env, single term)
@inline function tabulated_z_tendency(ice::IceProperties{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                          M6, <:Any, <:Any},
                                        log_m, Fᶠ, Fˡ, ρᶠ, rates, ρ, qⁱ, nⁱ, zⁱ,
                                        prp::ProcessRateParameters, sc_correction, p3, μ, μ_cloud, λ_r = nothing) where {M6 <: IceSixthMoment{<:P3Table5D}}
    FT = typeof(ρ)
    ice_table = ice_integrals_table(p3)
    rain_ice_table = rain_ice_collection_table(p3)
    sixth = ice_table.sixth_moment
    dep = ice_table.deposition

    inv_nⁱ = safe_divide(one(FT), nⁱ, eps(FT))

    # All Fortran Table 1 entries (deposition.* and sixth_moment.*) share the same
    # 5D axes, so we compute interpolation indices/weights once at this cell's
    # (log_m, Fᶠ, Fˡ, ρᶠ, μ) and reuse them for every table read below.
    prep = prepare_5d(dep.ventilation, log_m, Fᶠ, Fˡ, ρᶠ, μ)

    # --- Deposition / Sublimation ---
    # Z tables store raw dG/dt. Extract env factor from mass_rate / (mass_table × Nⁱ).
    # Fortran: epsiz = (m6dep + S_c×m6dep1) × 2πρDv
    #          epsi  = (vdep  + S_c×vdep1 ) × 2πρDv × Nⁱ
    # where S_c = Sc^(1/3) × √ρ_fac / √ν.
    # Deposition (c=2) and sublimation (c=1) use different dG normalization,
    # but the SAME mass integrals (vdep/vdep1). Separate via sign of rates.deposition.
    mass_dep_combined = evaluate_at(dep.ventilation, prep) +
                        sc_correction * evaluate_at(dep.ventilation_enhanced, prep)
    env_dep = safe_divide(abs(rates.deposition), max(nⁱ * mass_dep_combined, eps(FT)), zero(FT))

    z_dep_combined = evaluate_at(sixth.deposition, prep) +
                     sc_correction * evaluate_at(sixth.deposition1, prep)
    z_sub_combined = evaluate_at(sixth.sublimation, prep) +
                     sc_correction * evaluate_at(sixth.sublimation1, prep)

    # Diffusional growth is monotonic in particle size: deposition and coating
    # condensation cannot destroy the sixth moment, while sublimation and coating
    # evaporation cannot create it. Near the lower table bound, interpolation can
    # return a contribution with the wrong sign; preserve the process sign here.
    z_dep_combined = max(0, z_dep_combined)
    z_sub_combined = max(0, z_sub_combined)

    is_deposition = rates.deposition > zero(FT)
    z_dep_sub_rate = ifelse(is_deposition, z_dep_combined, -z_sub_combined) * env_dep

    env_coat_cond = safe_divide(rates.coating_condensation, max(nⁱ * mass_dep_combined, eps(FT)), zero(FT))
    env_coat_evap = safe_divide(rates.coating_evaporation, max(nⁱ * mass_dep_combined, eps(FT)), zero(FT))
    z_coat_rate = z_dep_combined * env_coat_cond - z_sub_combined * env_coat_evap

    # --- Melting ---
    # Fortran: zimlt = (vdepm1×m6mlt1 + vdepm2×m6mlt2×S_c) × thermo
    # Z is the mass-weighted combination of Z tables. Extract thermo from mass_rate / (mass_combined × Nⁱ).
    mass_melt_const = evaluate_at(dep.small_ice_ventilation_constant, prep)
    mass_melt_enh   = evaluate_at(dep.small_ice_ventilation_reynolds, prep)
    mass_melt_combined = mass_melt_const + sc_correction * mass_melt_enh
    complete_melting = rates.complete_melting
    env_melt = safe_divide(complete_melting, max(nⁱ * mass_melt_combined, eps(FT)), zero(FT))

    # Fortran uses D ≤ D_crit filtered tables (f1pr32/f1pr33) for Z melting when
    # liquid fraction is active. The non-liquid-fraction zimlt path in Fortran reuses
    # deposition tables and is gated by log_full3mom=.false. (dead code in P3 v5.5.0).
    # Julia provides all-D melt integrands for completeness via melt_all1/melt_all2.
    z_melt1 = ifelse(prp.liquid_fraction_active,
                     evaluate_at(sixth.melt1, prep),
                     evaluate_at(sixth.melt_all1, prep))
    z_melt2 = ifelse(prp.liquid_fraction_active,
                     evaluate_at(sixth.melt2, prep),
                     evaluate_at(sixth.melt_all2, prep))
    # Mass-weighted Z: each term multiplied by its own mass table (Fortran convention)
    z_melt_numerator = mass_melt_const * z_melt1 + sc_correction * mass_melt_enh * z_melt2
    z_melt_rate = z_melt_numerator * env_melt

    # --- Riming ---
    z_cloud_rime = evaluate_at(sixth.rime, prep)
    z_cloud_rime_rate = z_cloud_rime * rates.cloud_riming * inv_nⁱ
    # Per-unit-rain-riming sixth-moment factor — identical 6D-table inputs across
    # the standard and wet-growth rain-riming paths, so compute once and reuse.
    z_rain_rime_factor = rain_riming_sixth_moment_factor(rain_ice_table, log_m, Fᶠ, Fˡ, ρᶠ, μ, λ_r,
                                                          inv_nⁱ, z_cloud_rime)
    z_rain_rime_rate = z_rain_rime_factor * rates.rain_riming

    # Wet growth Z contribution.
    # During wet growth, rates.cloud_riming and rates.rain_riming are zeroed (redirected
    # to wet_growth_cloud/wet_growth_rain), so z_cloud_rime_rate and z_rain_rime_rate
    # are zero. The Z contribution must come from the wet growth rates instead.
    #
    # Fortran (microphy_p3.f90 lines 3250-3280):
    # - log_LiquidFrac=.TRUE.  (lines 3257-3259): zqccol/zqrcol keep full values
    #   ("both ice riming and retention of liquid increase zitot")
    # - log_LiquidFrac=.FALSE. (lines 3269-3280): zqccol/zqrcol reduced by shedding
    #   fraction (1 - shed/total_collection)
    z_wg_cloud_rate = z_cloud_rime * rates.wet_growth_cloud * inv_nⁱ
    z_wg_rain_rate = z_rain_rime_factor * rates.wet_growth_rain
    wg_total = rates.wet_growth_cloud + rates.wet_growth_rain
    shed_frac = safe_divide(rates.wet_growth_shedding, max(wg_total, eps(FT)), zero(FT))
    z_wg_rate = ifelse(prp.liquid_fraction_active,
                       z_wg_cloud_rate + z_wg_rain_rate,
                       (z_wg_cloud_rate + z_wg_rain_rate) * (1 - shed_frac))

    # --- Aggregation (single term): z_table = dG/nagg ---
    z_agg = evaluate_at(sixth.aggregation, prep)
    z_agg_rate = z_agg * rates.aggregation * inv_nⁱ

    # --- Shedding (single term): z_table = dG_kernel/M3 ---
    z_shed = evaluate_at(sixth.shedding, prep)
    z_shed_rate = z_shed * rates.shedding * inv_nⁱ

    z_group2 = group2_ice_sixth_moment_tendency(rates, prp, zero(FT))

    # Total Z rate
    z_rate = z_dep_sub_rate +
             z_coat_rate +
             z_cloud_rime_rate +
             z_rain_rime_rate +
             z_wg_rate +
             z_agg_rate -
             z_shed_rate -
             z_melt_rate
    z_rate = z_rate + z_group2

    return ρ * z_rate
end

# Fallback for 2-moment ice (no m6 tables): use simplified proportional Z tendency
@inline function tabulated_z_tendency(ice::IceProperties,
                                        log_m, Fᶠ, Fˡ, ρᶠ, rates, ρ, qⁱ, nⁱ, zⁱ,
                                        prp::ProcessRateParameters, sc_correction, p3, μ, μ_cloud, λ_r = nothing)
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, prp)
end

@inline function rain_riming_sixth_moment_factor(::Nothing, log_m, Fᶠ, Fˡ, ρᶠ, μ, λ_r,
                                                  inv_nⁱ, z_cloud_rime)
    return z_cloud_rime * inv_nⁱ
end

@inline function rain_riming_sixth_moment_factor(rain_ice_table::P3RainIceCollectionTable, log_m, Fᶠ, Fˡ, ρᶠ, μ,
                                                  ::Nothing, inv_nⁱ, z_cloud_rime)
    return z_cloud_rime * inv_nⁱ
end

@inline function rain_riming_sixth_moment_factor(rain_ice_table::P3RainIceCollectionTable, log_m, Fᶠ, Fˡ, ρᶠ, μ,
                                                  λ_r, inv_nⁱ, z_cloud_rime)
    FT = typeof(log_m)
    log_λ_r = log10(max(FT(λ_r), FT(1e-20)))
    # `sixth_moment` and `mass` share axes; prep once and reuse.
    prep = prepare_6d(rain_ice_table.mass, log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    z_rain_rime = evaluate_at(rain_ice_table.sixth_moment, prep)
    # Fortran convention: zqrcol = N0r × m6collr × env (no 10^f1pr08 factor).
    # Since rain_riming = N0r × ni × env × 10^f1pr08 (mass kernel),
    # divide by the mass kernel to recover: z = m6collr × rain_riming / (ni × mass_kernel).
    mass_kernel = exp10(evaluate_at(rain_ice_table.mass, prep))
    inv_mass_kernel = safe_divide(one(FT), mass_kernel, zero(FT))
    return z_rain_rime * inv_nⁱ * inv_mass_kernel
end

@inline function initiated_ice_sixth_moment_tendency(mass_tendency, number_tendency, μ_new)
    FT = typeof(mass_tendency + number_tendency + μ_new)
    q_source = clamp_positive(mass_tendency)
    n_source = clamp_positive(number_tendency)
    has_source = (q_source > zero(FT)) & (n_source > zero(FT))
    mom3_tendency = q_source * FT(6) / (FT(900) * FT(π))
    z_source = g_of_mu(μ_new) * mom3_tendency^2 / max(n_source, eps(FT))
    return ifelse(has_source, z_source, zero(FT))
end

@inline function nucleation_sixth_moment_tendency(nucleation_number, prp::ProcessRateParameters)
    FT = typeof(nucleation_number)
    nucleation_mass = nucleation_number * prp.nucleated_ice_mass
    return initiated_ice_sixth_moment_tendency(nucleation_mass, nucleation_number, zero(FT))
end
