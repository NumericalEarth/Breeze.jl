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
##### Utility functions
#####

@inline clamp_positive(x) = max(0, x)

"""
$(TYPEDSIGNATURES)

Compute proportional rescaling factor for sink rates so that
`total_sink × dt_safety` does not exceed `available_mass`.

Returns 1 when sinks are within budget, or `available_mass / (total_sink × dt_safety)`
when they exceed it. All arguments must be positive or zero.
GPU-compatible: uses `ifelse` instead of branching.
"""
@inline function sink_limiting_factor(total_sink, available_mass, dt_safety)
    projected = total_sink * dt_safety
    return ifelse(projected > available_mass,
                  available_mass / max(projected, eps(typeof(available_mass))),
                  one(typeof(available_mass)))
end

"""
$(TYPEDSIGNATURES)

Cap vapor sinks and sources against the moist-adiabatic saturation-adjustment
budget, matching Fortran P3 v5.5.0 `qcon_satadj` / `qevp_satadj` /
`qdep_satadj` (`microphy_p3.f90:3990-4055`).

Defining `qsatadj_ℓ = (qᵛ - qᵛ⁺ˡ) / ξˡ` with the moist-static feedback
factor `ξˡ = 1 + ℒˡ² qᵛ⁺ˡ / (cᵖᵈ Rᵛ T²)`:

- Liquid-phase condensation sinks (`cond > 0`, `ccn_act`, `rain_cond`,
  `coat_cond`) cannot exceed `max(0, qsatadj_ℓ)` (Fortran 3997-4012).
- Liquid-phase evaporation sources (`cond < 0`, `rain_evap`, `coat_evap`)
  cannot exceed `max(0, -qsatadj_ℓ)` (Fortran 4014-4028).

The rescaled liquid tendencies are then carried into a post-liquid state
`(qᵛ_after, T_after)` (Fortran's `qv_tmp` / `t_tmp`), and `qᵛ⁺ⁱ_after`
is recomputed at `T_after` to evaluate
`ξⁱ_after = 1 + ℒⁱ_after² qᵛ⁺ⁱ_after / (cᵖᵈ Rᵛ T_after²)`. With
`qsatadj_ᵢ = (qᵛ_after - qᵛ⁺ⁱ_after) / ξⁱ_after`:

- Ice-phase deposition sinks (`dep > 0`, `nuc_q`) cannot exceed
  `max(0, qsatadj_ᵢ)` (Fortran 4037-4049).
- Ice-phase sublimation sources (`dep < 0`) cannot exceed
  `max(0, -qsatadj_ᵢ)` (Fortran 4050-4055).

Number rates `ccn_act_n` and `nuc_n` are scaled by the same factor as their
companion mass rates to preserve mean particle mass.

Returns a NamedTuple of the possibly-rescaled rates.
"""
@inline function limit_vapor_rates(cond, ccn_act, ccn_act_n, rain_cond, rain_evap,
                                   dep, coat_cond, coat_evap, nuc_q, nuc_n,
                                   qᵛ, qᵛ⁺ˡ, T, P, qᵗ, constants, dt_safety)
    FT = typeof(qᵛ)
    Rᵛ = FT(vapor_gas_constant(constants))
    ℒˡ = vaporization_latent_heat(constants, T)
    ξˡ = liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)

    # Liquid-phase saturation-adjustment caps (Fortran 3991-4028). In the
    # SCF=1 limit `qcon_satadj` and `qevp_satadj` collapse to the same signed
    # value `qsatadj_ℓ`; condensation sees the positive part, evaporation the
    # negative part.
    qsatadj_ℓ = (qᵛ - qᵛ⁺ˡ) / ξˡ
    qcon_cap = max(0, qsatadj_ℓ)
    qevp_cap = max(0, -qsatadj_ℓ)

    # Condensation cap (Fortran 3997-4012)
    cond_sink_total = clamp_positive(cond) + ccn_act + rain_cond + coat_cond
    f_cond = sink_limiting_factor(cond_sink_total, qcon_cap, dt_safety)

    cond_pos_scaled = ifelse(cond > 0, cond * f_cond, cond)
    ccn_act = ccn_act * f_cond
    ccn_act_n = ccn_act_n * f_cond
    rain_cond = rain_cond * f_cond
    coat_cond = coat_cond * f_cond

    # Evaporation cap (Fortran 4014-4028): zero when supersaturated, otherwise
    # rescale the lumped evaporation rates to fit within `qevp_cap`.
    evp_total = clamp_positive(-cond) + rain_evap + coat_evap
    f_evp = sink_limiting_factor(evp_total, qevp_cap, dt_safety)

    cond = ifelse(cond < 0, cond * f_evp, cond_pos_scaled)
    rain_evap = rain_evap * f_evp
    coat_evap = coat_evap * f_evp

    # Ice-phase cap, after netting the rescaled liquid tendencies into qᵛ and T
    # (Fortran 4031-4035 `qv_tmp` / `t_tmp`).
    net_liquid = clamp_positive(cond) + ccn_act + rain_cond + coat_cond -
                 rain_evap - coat_evap - clamp_positive(-cond)
    qᵛ_after = qᵛ - net_liquid * dt_safety
    T_after = T + net_liquid * ℒˡ * dt_safety / cᵖᵈ
    qᵛ⁺ⁱ_after = adjustment_saturation_specific_humidity(T_after, P, qᵗ, constants, PlanarIceSurface())
    ℒⁱ_after = sublimation_latent_heat(constants, T_after)
    ξⁱ_after = ice_psychrometric_correction(constants, ℒⁱ_after, qᵛ⁺ⁱ_after, Rᵛ, T_after)

    # Ice-phase deposition / sublimation caps (Fortran 4037-4055).
    qsatadj_ᵢ = (qᵛ_after - qᵛ⁺ⁱ_after) / ξⁱ_after
    qdep_cap = max(0, qsatadj_ᵢ)
    qsub_cap = max(0, -qsatadj_ᵢ)

    # Deposition cap (Fortran 4037-4049)
    dep_sink_total = clamp_positive(dep) + nuc_q
    f_dep = sink_limiting_factor(dep_sink_total, qdep_cap, dt_safety)

    dep_pos_scaled = ifelse(dep > 0, dep * f_dep, dep)
    nuc_q = nuc_q * f_dep
    nuc_n = nuc_n * f_dep

    # Sublimation cap (Fortran 4050-4055)
    sub_total = clamp_positive(-dep)
    f_sub = sink_limiting_factor(sub_total, qsub_cap, dt_safety)

    dep = ifelse(dep < 0, dep * f_sub, dep_pos_scaled)

    return (; cond, ccn_act, ccn_act_n, rain_cond, rain_evap,
              dep, coat_cond, coat_evap, nuc_q, nuc_n)
end

@inline function safe_divide(a, b, default)
    return ifelse(iszero(b), default, a / b)
end

@inline safe_divide(a, b) = safe_divide(a, b, zero(a))

@inline function ice_air_density_correction(reference_air_density, air_density)
    FT = typeof(reference_air_density)
    return (reference_air_density / max(air_density, FT(0.01)))^FT(0.54)
end

@inline function mean_ice_distribution_state(FT, Fᶠ, Fˡ, ρᶠ, prp)
    mass = IceMassPowerLaw(FT)

    return IceSizeDistributionState(FT;
                                    intercept = one(FT),
                                    shape = zero(FT),
                                    slope = one(FT),
                                    rime_fraction = Fᶠ,
                                    liquid_fraction = Fˡ,
                                    rime_density = ρᶠ,
                                    mass_coefficient = mass.coefficient,
                                    mass_exponent = mass.exponent,
                                    ice_density = mass.ice_density,
                                    reference_air_density = prp.reference_air_density,
                                    air_density = prp.reference_air_density)
end

@inline function mean_ice_particle_diameter(m_mean, Fᶠ, Fˡ, ρᶠ, prp)
    FT = typeof(m_mean)
    state = mean_ice_distribution_state(FT, Fᶠ, Fˡ, ρᶠ, prp)
    thresholds = regime_thresholds_from_state(FT, state)
    D_mean = particle_diameter_ice_only(clamp_positive(m_mean), state, thresholds)

    return D_mean, state, thresholds
end

"""
$(TYPEDSIGNATURES)

Apply the Fortran `calc_bulkRhoRime` consistency pass to the prognostic rime
state. Returns corrected `qᶠ`, `bᶠ`, rime fraction `Fᶠ`, and rime density `ρᶠ`.
"""
@inline function consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, qʷⁱ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_available = clamp_positive(qⁱ)
    # Julia's qⁱ is already dry ice (= Fortran qitot - qiliq), so
    # qⁱ_dry = qⁱ_available (no need to subtract qʷⁱ again).
    qⁱ_dry = qⁱ_available
    qᶠ_raw = clamp_positive(qᶠ)
    bᶠ_raw = clamp_positive(bᶠ)

    has_rime_volume = bᶠ_raw >= FT(1e-15)
    ρᶠ_raw = safe_divide(qᶠ_raw, bᶠ_raw, zero(FT))
    ρᶠ_bounded = clamp(ρᶠ_raw, prp.minimum_rime_density, prp.maximum_rime_density)

    qᶠ_after_volume = ifelse(has_rime_volume, qᶠ_raw, zero(FT))
    bᶠ_after_volume = ifelse(has_rime_volume,
                             safe_divide(qᶠ_after_volume, ρᶠ_bounded, zero(FT)),
                             zero(FT))
    ρᶠ = ifelse(has_rime_volume, ρᶠ_bounded, zero(FT))

    rime_not_small = qᶠ_after_volume >= p3.minimum_mass_mixing_ratio
    qᶠ_after_small = ifelse(rime_not_small, qᶠ_after_volume, zero(FT))
    bᶠ_after_small = ifelse(rime_not_small, bᶠ_after_volume, zero(FT))

    # bound rime mass by dry ice mass, not total ice mass
    exceeds_dry_ice = (qᶠ_after_small > qⁱ_dry) & (ρᶠ > zero(FT))
    qᶠ_consistent = ifelse(exceeds_dry_ice, qⁱ_dry, qᶠ_after_small)
    bᶠ_consistent = ifelse(exceeds_dry_ice,
                           safe_divide(qᶠ_consistent, ρᶠ, zero(FT)),
                           bᶠ_after_small)
    Fᶠ = safe_divide(qᶠ_consistent, qⁱ_dry, zero(FT))

    return (; qᶠ = qᶠ_consistent, bᶠ = bᶠ_consistent, Fᶠ, ρᶠ)
end

@inline ice_integrals_table(p3) = lookup_field(p3.ice.lookup_tables, Val(:ice_integrals))
@inline rain_ice_collection_table(p3) = lookup_field(p3.ice.lookup_tables, Val(:rain_ice_collection))
@inline three_moment_shape_table(p3) = lookup_field(p3.ice.lookup_tables, Val(:three_moment_shape))

@inline lookup_field(tables::P3LookupTables, ::Val{:ice_integrals}) = tables.ice_integrals
@inline lookup_field(tables::P3LookupTables, ::Val{:rain_ice_collection}) = tables.rain_ice_collection
@inline lookup_field(tables::P3LookupTables, ::Val{:three_moment_shape}) = tables.three_moment_shape
@inline lookup_field(::Nothing, ::Val) = nothing

@inline total_ice_mass(qⁱ, qʷⁱ) = clamp_positive(qⁱ) + clamp_positive(qʷⁱ)

@inline function liquid_fraction_on_ice(qⁱ, qʷⁱ)
    FT = typeof(qⁱ)
    qⁱ_total = max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20))
    return clamp_positive(qʷⁱ) / qⁱ_total
end

@inline function mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    FT = typeof(qⁱ)
    return safe_divide(max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20)),
                       max(clamp_positive(nⁱ), FT(1e-16)),
                       FT(1e-20))
end

@inline function rain_slope_parameter(qʳ, nʳ, prp)
    FT = typeof(qʳ + nʳ)
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    λ_r_cubed = FT(π) * prp.liquid_water_density * nʳ_eff / max(qʳ_eff, FT(1e-20))
    return clamp(cbrt(λ_r_cubed), prp.rain_lambda_min, prp.rain_lambda_max)
end

@inline function rain_number_from_slope(qʳ, λ_r, prp)
    FT = typeof(qʳ + λ_r)
    qʳ_eff = clamp_positive(qʳ)
    return qʳ_eff * λ_r^3 / (FT(π) * prp.liquid_water_density)
end

@inline function bounded_rain_number(nʳ, qʳ, prp)
    FT = typeof(qʳ + nʳ)
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    λ_r_uncapped = cbrt(FT(π) * prp.liquid_water_density * nʳ_eff /
                        max(qʳ_eff, FT(1e-20)))
    λ_r = clamp(λ_r_uncapped, prp.rain_lambda_min, prp.rain_lambda_max)
    nʳ_bounded = rain_number_from_slope(qʳ_eff, λ_r, prp)
    needs_adjustment = (λ_r_uncapped < prp.rain_lambda_min) | (λ_r_uncapped > prp.rain_lambda_max)
    return ifelse(needs_adjustment, nʳ_bounded, nʳ_eff)
end

@inline function ice_mean_density_for_bounds(ice_table::P3IceIntegralsTable, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    FT = typeof(qⁱ_total)
    m̄ = safe_divide(max(qⁱ_total, FT(1e-20)), max(nⁱ, FT(1e-16)), FT(1e-20))
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    return ice_table.bulk_properties.mean_density(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
end

@inline function bound_ice_sixth_moment(ice_table::P3IceIntegralsTable, qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    FT = typeof(qⁱ_total)
    has_ice = (qⁱ_total > FT(1e-20)) & (nⁱ > FT(1e-16))
    ρ_bulk = ice_mean_density_for_bounds(ice_table, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    μ_bounds = ThreeMomentClosure(FT)
    z_bounded = enforce_z_bounds(clamp_positive(zⁱ), qⁱ_total, nⁱ, ρ_bulk, μ_bounds.μmin, μ_bounds.μmax)
    return ifelse(has_ice, z_bounded, zero(FT))
end

@inline bound_ice_sixth_moment(::Nothing, qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ) = clamp_positive(zⁱ)

@inline function bound_ice_sixth_moment(p3, qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    return bound_ice_sixth_moment(ice_integrals_table(p3), qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
end

#####
##### Ice shape parameter (μ) from Table 3
#####
##### For 3-moment P3, μ is diagnosed from (qⁱ, nⁱ, zⁱ) via Table 3.
##### For 2-moment P3, μ is looked up from Table 1 (mu_i_save column).
#####

"""
$(TYPEDSIGNATURES)

Compute the ice PSD shape parameter μ from lookup tables.

For 3-moment P3, μ is diagnosed from the ratio Z/L (sixth moment to mass)
using the pre-tabulated closure in Table 3.

For 2-moment P3 (Table 3 absent), μ is looked up directly from Table 1
(`bulk_properties.shape`), which stores the `mu_i_save` value computed
during Fortran table generation.
"""
@inline function compute_ice_shape_parameter(p3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ)
    FT = typeof(qⁱ)
    shape_table_3mom = three_moment_shape_table(p3)
    return ice_shape_parameter(shape_table_3mom, p3.ice.bulk_properties.shape,
                                qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, FT)
end

# 3-moment: diagnose μ from Table 3 (independent of mu axis)
@inline function ice_shape_parameter(shape_table_3mom::P3ThreeMomentShapeTable, shape_table,
                                      qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, FT)
    qⁱ_safe = max(qⁱ, eps(FT))
    nⁱ_safe = max(nⁱ, eps(FT))
    zⁱ_safe = max(zⁱ, eps(FT))
    return shape_parameter_lookup(shape_table_3mom, qⁱ_safe, nⁱ_safe, zⁱ_safe, Fᶠ, Fˡ, ρᶠ)
end

# 2-moment with tables: look up μ from Table 1 (mu_i_save)
@inline function ice_shape_parameter(::Nothing, shape_table::P3Table5D,
                                      qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, FT)
    log_m = log10(max(qⁱ / max(nⁱ, eps(FT)), eps(FT)))
    return shape_table(log_m, Fᶠ, Fˡ, ρᶠ, zero(FT))
end


#####
##### Thermodynamic latent heat helpers (H1)
#####
##### When thermodynamic constants are available, use T-dependent latent heats
##### for energy-budget consistency with the condensation path. When `nothing`
##### is passed (backward-compatible path), fall back to the Fortran P3 v5.5.0
##### hardcoded constants.
#####

@inline sublimation_latent_heat(::Nothing, T) = typeof(T)(2.835e6)
@inline sublimation_latent_heat(constants, T) = ice_latent_heat(T, constants)

@inline vaporization_latent_heat(::Nothing, T) = typeof(T)(2.5e6)
@inline vaporization_latent_heat(constants, T) = liquid_latent_heat(T, constants)

@inline p3_dry_air_heat_capacity(::Nothing, FT) = FT(1005)
@inline p3_dry_air_heat_capacity(constants, FT) = FT(constants.dry_air.heat_capacity)

# GPU-compatible fallbacks: use precomputed module constants when constants=nothing.
# The original code used `isnothing(constants) ? ThermodynamicConstants(FT) : constants`
# but ThermodynamicConstants() allocates and cannot run on GPU. These dispatches provide
# the same default values without allocation.
using Breeze.Thermodynamics: Thermodynamics
@inline Thermodynamics.vapor_gas_constant(::Nothing) = VAPOR_GAS_CONSTANT
@inline Thermodynamics.dry_air_gas_constant(::Nothing) = DRY_AIR_GAS_CONSTANT
@inline Thermodynamics.density(T, P, q::MoistureMassFractions, ::Nothing) =
    Thermodynamics.density(T, P, q, ThermodynamicConstants(typeof(T)))
@inline Thermodynamics.mixture_heat_capacity(q::MoistureMassFractions, ::Nothing) =
    Thermodynamics.mixture_heat_capacity(q, ThermodynamicConstants(typeof(q.vapor)))

@inline fusion_latent_heat(constants, T) = sublimation_latent_heat(constants, T) - vaporization_latent_heat(constants, T)

#####
##### Psychrometric corrections ξˡ, ξⁱ
#####
##### Account for the latent-heat feedback that reduces the effective
##### supersaturation drive during condensation (ξˡ, Fortran "ab") and ice
##### deposition (ξⁱ, Fortran "abi"). Both share the form
##### ξ = 1 + ℒ² qᵛ⁺ / (cᵖᵈ Rᵛ T²) with the appropriate latent heat.
#####

@inline function liquid_psychrometric_correction(constants, ℒˡ, qᵛ⁺ˡ, Rᵛ, T)
    FT = typeof(T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    return 1 + ℒˡ^2 * qᵛ⁺ˡ / (Rᵛ * T^2 * cᵖᵈ)
end

@inline function ice_psychrometric_correction(constants, ℒⁱ, qᵛ⁺ⁱ, Rᵛ, T)
    FT = typeof(T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    return 1 + ℒⁱ^2 * qᵛ⁺ⁱ / (Rᵛ * T^2 * cᵖᵈ)
end

#####
##### Saturation vapor pressure at freezing (M6)
#####
##### When thermodynamic constants are available, derive e_s(T₀) from the
##### Clausius-Clapeyron or Tetens formula. When `nothing` is passed, fall back
##### to the Fortran P3 v5.5.0 hardcoded 611 Pa (≈ e_s at 273.15 K).
#####

@inline saturation_vapor_pressure_at_freezing(::Nothing, T₀) = typeof(T₀)(611)
@inline function saturation_vapor_pressure_at_freezing(constants, T₀)
    return saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
end
