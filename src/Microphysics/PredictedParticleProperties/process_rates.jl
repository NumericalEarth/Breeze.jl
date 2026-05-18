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

"""
$(TYPEDSIGNATURES)

Return max(0, x) for numerical stability.
"""
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

"""
$(TYPEDSIGNATURES)

Safe division returning `default` when b ≈ 0.
All arguments must be positional (GPU kernel compatibility).
"""
@inline function safe_divide(a, b, default)
    return ifelse(iszero(b), default, a / b)
end

# Convenience overload for common case
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
    # D14: Julia's qⁱ is already dry ice (= Fortran qitot - qiliq), so
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

    # M5: bound rime mass by dry ice mass, not total ice mass
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
    FT = typeof(qʳ)
    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = clamp_positive(nʳ)
    λ_r_cubed = FT(π) * prp.liquid_water_density * nʳ_eff / max(qʳ_eff, FT(1e-20))
    return clamp(cbrt(λ_r_cubed), prp.rain_lambda_min, prp.rain_lambda_max)
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
    cₚᵈ = p3_dry_air_heat_capacity(constants, FT)
    return 1 + ℒˡ^2 * qᵛ⁺ˡ / (Rᵛ * T^2 * cₚᵈ)
end

@inline function ice_psychrometric_correction(constants, ℒⁱ, qᵛ⁺ⁱ, Rᵛ, T)
    FT = typeof(T)
    cₚᵈ = p3_dry_air_heat_capacity(constants, FT)
    return 1 + ℒⁱ^2 * qᵛ⁺ⁱ / (Rᵛ * T^2 * cₚᵈ)
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

#####
##### Ventilation Sc correction (H4)
#####
##### The ventilation-enhanced table stores 0.44 × ∫ C(D)√(V×D) N'(D) dD
##### with dimensions [m² s^(-1/2)]. At runtime, multiplying by
##### Sc^(1/3) × √ρ_fac / √ν restores the correct dimensions [m].
##### This helper centralizes the
##### correction so that all call sites (deposition, Z-tendency) stay in sync.
#####

"""
$(TYPEDSIGNATURES)

Schmidt number correction factor for ventilation-enhanced table values.

The P3 lookup table stores the ventilation-enhanced integral without the
`Sc^{1/3} √rhofaci / √ν` factor (matching the Fortran convention). This function
computes the correction that must be applied at runtime:

```math
f_{Sc} = \\frac{Sc^{1/3} \\sqrt{\\rho_{fac}}}{\\sqrt{\\nu}}
```

See `quadrature.jl` for the table storage convention.
"""
@inline function ventilation_sc_correction(nu, D_v, ρ_correction = one(typeof(nu)))
    FT = typeof(nu)
    Sc = nu / max(D_v, FT(1e-30))
    return cbrt(Sc) * sqrt(ρ_correction) / sqrt(nu)
end

#####
##### PSD-integrated process rate helpers (tabulated)
#####

"""
$(TYPEDSIGNATURES)

Compute per-particle ventilation integral C(D) × f_v(D) for deposition
using PSD-integrated lookup tables.
"""
@inline function deposition_ventilation(vent::P3Table5D,
                                          vent_e::P3Table5D,
                                          m_mean, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
    FT = typeof(m_mean)
    return deposition_ventilation(vent, vent_e, m_mean, Fᶠ, zero(FT), ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
end

@inline function deposition_ventilation(vent::P3Table5D,
                                          vent_e::P3Table5D,
                                          m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    # vent stores the constant ventilation term (0.65 × ∫ C(D) N'(D) dD)
    # vent_e stores the enhanced term (0.44 × ∫ C(D)√(V×D) N'(D) dD)  [m² s^(-1/2)]
    # Runtime correction via ventilation_sc_correction:
    # Sc^(1/3) × √ρ_fac / √ν [s^(1/2) m^(-1)]
    # Dimensional check: table [m² s^(-1/2)] × correction [s^(1/2)/m] = [m]
    return vent(log_m, Fᶠ, Fˡ, ρᶠ, μ) + ventilation_sc_correction(nu, D_v, ρ_correction) * vent_e(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute per-particle ventilation integral C(D) × f_v(D) for melting
using PSD-integrated lookup tables, blending ice (0.65, 0.44) and rain
(0.78, 0.28) ventilation coefficients weighted by liquid fraction Fl.
"""
@inline function melting_ventilation(vent::P3Table5D,
                                       vent_e::P3Table5D,
                                       m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    return vent(log_m, Fᶠ, Fl, ρᶠ, μ) + ventilation_sc_correction(nu, D_v, ρ_correction) * vent_e(log_m, Fᶠ, Fl, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute per-particle collection kernel ⟨A × V⟩ for riming.
Returns PSD-integrated ∫ V(D) A(D) N'(D) dD (per particle) from lookup table.
"""
@inline function collection_kernel_per_particle(coll::P3Table5D,
                                                  m_mean, Fᶠ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    return collection_kernel_per_particle(coll, m_mean, Fᶠ, zero(FT), ρᶠ, prp, p3, μ)
end

@inline function collection_kernel_per_particle(coll::P3Table5D,
                                                  m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    return coll(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute aggregation kernel for self-collection using PSD-integrated
kernel from lookup table.
"""
@inline function aggregation_kernel(coll::P3Table5D,
                                      m_mean, Fᶠ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    return aggregation_kernel(coll, m_mean, Fᶠ, zero(FT), ρᶠ, prp, p3, μ)
end

@inline function aggregation_kernel(coll::P3Table5D,
                                      m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    # Table stores the half-integral (Fortran convention):
    # (1/2) ∫∫ (√A₁+√A₂)² |V₁-V₂| N₁ N₂ dD₁ dD₂
    # No E_agg — collection efficiency is applied by the caller.
    return coll(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end

#####
##### CCN activation
#####

"""
$(TYPEDSIGNATURES)

Compute CCN activation rate for the 1-moment (prescribed Nᶜ) case.

Following Fortran P3 v5.5.0 (lines 3953-3963): when the air is supersaturated
and the cloud mass is below the minimum threshold for the prescribed droplet
concentration, a seed mass is created. The target cloud mass is
``N_c / ρ × m_{\\text{drop}}`` where ``m_{\\text{drop}} = (4π/3) ρ_w r^3``
for ``r = 1`` μm. The rate is limited by the available supersaturation.

# Returns
- Rate of vapor → cloud liquid conversion from CCN activation [kg/kg/s]
"""
@inline function ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    FT = typeof(qᶜˡ)
    prp = p3.process_rates

    # Mass of a newly formed cloud droplet (Fortran cons7: radius 1 μm)
    cons7 = FT(4 * FT(π) / 3 * 1000 * (1e-6)^3)

    # Target cloud mass for prescribed droplet concentration
    target_qc = Nᶜ / ρ * cons7

    # Deficit: how much mass is needed to reach the minimum
    deficit = clamp_positive(target_qc - clamp_positive(qᶜˡ))

    # Psychrometric correction (liquid saturation)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT

    # Limit by available supersaturation (Fortran: min(tmp1, (Qv_cld-dumqvs)/ab))
    max_from_ss = clamp_positive((qᵛ - qᵛ⁺ˡ) / Γˡ)
    rate = min(deficit, max_from_ss) / prp.sink_limiting_timescale

    # Only activate when supersaturated (Fortran threshold: sup_cld > 1e-6)
    is_supersaturated = (qᵛ - qᵛ⁺ˡ) / max(qᵛ⁺ˡ, FT(1e-10)) > FT(1e-6)
    return ifelse(is_supersaturated, rate, zero(FT))
end

"""
$(TYPEDSIGNATURES)

Dispatch CCN activation: prescribed (Nothing) or prognostic (AerosolActivation).
Returns `(; mass, number)` named tuple.
"""
@inline function compute_ccn_activation(::Nothing, p3, qᶜˡ, nᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    FT = typeof(qᶜˡ)
    # Prescribed-Nᶜ path (Fortran `log_predictNc = .false.`, `nc = nccnst_2`):
    # the activation target is the scheme parameter, not the DSD-diagnosed `Nᶜ`.
    # When `qᶜˡ` is below the mass threshold, `diagnose_cloud_dsd` clamps the
    # returned `Nᶜ` toward zero — using that value would collapse `target_qc`
    # and block any seed mass from forming in a warm-bubble parcel.
    target_Nᶜ = p3.cloud.number_concentration
    mass = ccn_activation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, target_Nᶜ, constants)
    return (; mass, number = zero(FT))
end

@inline function compute_ccn_activation(aerosol::AerosolActivation, p3, qᶜˡ, nᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    result = prognostic_ccn_activation_rate(aerosol, nᶜˡ, qᵛ, qᵛ⁺ˡ, T)
    return (; mass = result.qcnuc, number = result.ncnuc)
end

#####
##### Ice deposition and sublimation
#####

"""
$(TYPEDSIGNATURES)

Compute ventilation-enhanced ice deposition/sublimation rate with latent-heat
psychrometric correction.

Following Morrison & Milbrandt (2015a) Eq. 30, the single-particle growth rate is:

```math
\\frac{dm}{dt} = \\frac{4πC f_v (S_i - 1)}{Γⁱ \\left[\\frac{ℒⁱ}{K_a T}\\left(\\frac{ℒⁱ}{R_v T} - 1\\right) + \\frac{R_v T}{e_{si} D_v}\\right]}
```

where ``Γⁱ = 1 + ℒⁱ^2 q^{v+i} / (R_v T^2 c_p^m)`` is the latent-heat psychrometric
correction (analogous to Fortran P3's `abi` factor). It accounts for the reduction
in the effective supersaturation drive caused by latent heat released during
deposition and is consistent with Breeze's `SaturationAdjustment` Jacobian
linearisation.

The bulk rate integrates over the size distribution:

```math
\\frac{dq^i}{dt} = \\int \\frac{dm}{dt}(D)\\, N'(D)\\, dD
```

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Dry ice mass fraction [kg/kg]
- `qʷⁱ`: Liquid water on ice [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ⁱ`: Saturation vapor mass fraction over ice [kg/kg]
- `Fᶠ`: Rime fraction [-]
- `ρᶠ`: Rime density [kg/m³]
- `T`: Temperature [K]
- `P`: Pressure [Pa]
- `constants`: Thermodynamic constants (or `nothing` for Fortran-matched hardcoded values)
- `transport`: Pre-computed air transport properties `(; D_v, K_a, nu)`
- `q`: Moisture mass fractions used to compute mixture heat capacity for ``Γⁱ``

# Returns
- Rate of vapor → ice conversion [kg/kg/s] (positive = deposition)
"""
function ventilation_enhanced_deposition(p3, qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P,
                                                  constants, transport, q, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)

    # When runtime thermodynamic constants are provided, use their gas constants
    # consistently with the latent heat and saturation calculations.
    Rᵛ = FT(vapor_gas_constant(constants))
    Rᵈ = FT(dry_air_gas_constant(constants))
    ℒⁱ = sublimation_latent_heat(constants, T)
    # T,P-dependent transport properties (pre-computed or computed on demand)
    K_a = transport.K_a       # Thermal conductivity of air [W/m/K]
    D_v = transport.D_v       # Diffusivity of water vapor [m²/s]
    nu  = transport.nu        # Kinematic viscosity [m²/s]

    # Saturation vapor pressure over ice
    # Derived from qᵛ⁺ⁱ: qᵛ⁺ⁱ = ε × e_si / (P - (1-ε) × e_si)
    # Rearranging: e_si = P × qᵛ⁺ⁱ / (ε + qᵛ⁺ⁱ × (1 - ε))
    ε = Rᵈ / Rᵛ
    qᵛ⁺ⁱ_safe = max(qᵛ⁺ⁱ, FT(1e-30))
    e_si = P * qᵛ⁺ⁱ_safe / (ε + qᵛ⁺ⁱ_safe * (1 - ε))

    # Supersaturation ratio with respect to ice
    S_i = qᵛ / max(qᵛ⁺ⁱ, FT(1e-10))

    # Mean particle mass
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)

    ρ_air = density(T, P, q, constants)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)

    # PSD-integrated ventilation integral C(D) × f_v(D) from lookup table.
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

    # Denominator: thermodynamic resistance terms (Mason 1971)
    # A = ℒⁱ/(K_a × T) × (ℒⁱ/(R_v × T) - 1)
    # B = R_v × T / (e_si × D_v)
    A = ℒⁱ / (K_a * T) * (ℒⁱ / (Rᵛ * T) - 1)
    B = Rᵛ * T / (e_si * D_v)
    thermodynamic_factor = A + B

    # Latent-heat psychrometric correction Γⁱ (Fortran P3 "abi"):
    # Reduces the effective supersaturation drive to account for the
    # warming produced by the latent heat of deposition.
    # Γⁱ = 1 + Lₛ² qᵛ⁺ⁱ / (Rᵛ T² cᵖᵈ)  ≡  1 + (Lₛ/cᵖᵈ) dqᵛ⁺ⁱ/dT
    Γⁱ = ice_psychrometric_correction(constants, ℒⁱ, qᵛ⁺ⁱ_safe, Rᵛ, T)

    # Deposition rate per particle (Eq. 30 from MM15a)
    # Uses 2π (not 4π) because the ventilation integral stores capm = cap × D
    # (P3 Fortran convention), which is 2× the physical capacitance C = D/2.
    # The product 2π × capm = 2π × 2C = 4πC is physically correct.
    dm_dt = 2 * FT(π) * C_fv * (S_i - 1) / (Γⁱ * thermodynamic_factor)

    # Scale by number concentration
    dep_rate = nⁱ_eff * dm_dt

    # Apply calibration factors (Fortran P3 v5.5.0 clbfact_dep, clbfact_sub).
    # These ad hoc multipliers account for uncertainty in ice capacitance.
    is_sublimation = S_i < 1
    cal = ifelse(is_sublimation, prp.calibration_factor_sublimation,
                                 prp.calibration_factor_deposition)
    dep_rate = dep_rate * cal

    # Limit sublimation to available ice
    τ_dep = prp.ice_deposition_timescale
    max_sublim = -qⁱ_eff / τ_dep

    return ifelse(is_sublimation, max(dep_rate, max_sublim), dep_rate)
end

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
    N₀ = nʳ_eff * λ_r
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
                                                     μ_c, λ_c, nᶜˡ_bounded)
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

    qc_raw = (bergeron_driver * εᶜˡ / ε_total + (ssat_liquid - bergeron_driver / ε_total) * εᶜˡ / ε_total * transient) / ξˡ
    qr_raw = (bergeron_driver * εʳ / ε_total + (ssat_liquid - bergeron_driver / ε_total) * εʳ / ε_total * transient) / ξˡ
    qi_raw = (bergeron_driver * εⁱ / ε_total + (ssat_liquid - bergeron_driver / ε_total) * εⁱ / ε_total * transient) / ξⁱ +
             (qᵛ⁺ˡ - qᵛ⁺ⁱ) * εⁱ / ξⁱ
    # Liquid-on-ice coating uses `ξˡ` (like cloud) since the surface condenses
    # vapor as liquid; no Bergeron contribution because the surface is already
    # at liquid saturation.
    ql_raw = (bergeron_driver * εⁱʷ / ε_total + (ssat_liquid - bergeron_driver / ε_total) * εⁱʷ / ε_total * transient) / ξˡ

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

    # C3: Global ice number limiter — Fortran impose_max_Ni (positive magnitude)
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

    # D8: Wet growth shedding — excess collection beyond freezing capacity goes to rain
    # (Fortran nrshdr/qcshd: mass that can't freeze sheds as 1 mm rain drops)
    wet_growth_shedding :: FT          # Excess collection → rain mass [kg/kg/s]
    wet_growth_shedding_number :: FT   # Rain number from wet growth shedding [1/kg/s]

    # M9: Warm/mixed-phase budget terms
    ccn_activation_mass :: FT          # CCN activation mass rate (vapor → cloud) [kg/kg/s]
    ccn_activation_number :: FT        # CCN activation number rate [1/kg/s] (prognostic CCN only)
    rain_condensation :: FT            # Rain condensation (vapor → rain) [kg/kg/s]
    coating_condensation :: FT         # Condensation on ice liquid coating [kg/kg/s]
    coating_evaporation :: FT          # Evaporation from ice liquid coating [kg/kg/s]

    # H9: Wet growth rime densification (Fortran lines 4303-4307)
    # During wet growth, assume total soaking: qᶠ → qⁱ, bᶠ → qⁱ/ρ_rimeMax.
    wet_growth_densification_mass :: FT   # Rime mass source: (qⁱ - qᶠ)/τ [kg/kg/s]
    wet_growth_densification_volume :: FT # Rime volume change: (qⁱ/ρ_max - bᶠ)/τ [m³/kg/s]

    # M6: DSD number correction feedback (Fortran get_cloud_dsd2/get_rain_dsd2)
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

    # =========================================================================
    # Coupled cloud/rain/ice vapor growth and decay
    # =========================================================================
    vapor_rates = coupled_saturation_adjustment_rates(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.qʳ, nʳ,
                                                      ℳ.qⁱ, ℳ.qʷⁱ, nⁱ, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ,
                                                      Fᶠ, ρᶠ, T, P, ρ, constants,
                                                      transport, q, μ_ice,
                                                      state.μ_c, state.λ_c, state.nᶜˡ)
    cond = vapor_rates.condensation

    # CCN activation (prescribed or prognostic)
    ccn = compute_ccn_activation(p3.aerosol, p3, ℳ.qᶜˡ, ℳ.nᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, ρ, Nᶜ, constants)
    ccn_act = ccn.mass
    ccn_act_n = ccn.number

    # =========================================================================
    # Rain processes
    # =========================================================================
    autoconv = rain_autoconversion_rate(p3, ℳ.qᶜˡ, Nᶜ, ρ)
    accr = rain_accretion_rate(p3, ℳ.qᶜˡ, ℳ.qʳ)
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
                             autoconv, accr, rain_evap, rain_cond, rain_self, rain_br,
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

    # C3: Global ice number limiter — Fortran impose_max_Ni hard-clamps the prognostic
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
    is_wet_growth = total_collection > qwgrth + FT(1e-10)

    wg_cloud = ifelse(is_wet_growth, cloud_rim, zero(FT))
    wg_rain  = ifelse(is_wet_growth, rain_rim, zero(FT))
    cloud_rim   = ifelse(is_wet_growth, zero(FT), cloud_rim)
    cloud_rim_n = ifelse(is_wet_growth, zero(FT), cloud_rim_n)
    rain_rim    = ifelse(is_wet_growth, zero(FT), rain_rim)
    rain_rim_n  = ifelse(is_wet_growth, zero(FT), rain_rim_n)

    # D8: Wet growth shedding
    shed_active = !prp.liquid_fraction_active & is_wet_growth
    wg_shed   = ifelse(shed_active, clamp_positive(total_collection - qwgrth), zero(FT))
    wg_shed_n = wg_shed / prp.shed_drop_mass

    # H9: Wet growth rime densification
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

    # M8: Filiq > 0.99 safety clipping
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
    nʳ = ifelse(rain_active, qʳ_pos * λ_r^3 / (FT(π) * prp.liquid_water_density), nʳ)

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
                                      qᶠ, bᶠ, ℳ.zⁱ, qʷⁱ, qᵛ - qᵛ⁺ˡ)

    # Build derived state struct (explicit type parameters to avoid
    # jl_f_throw_methoderror in @noinline GPU compilation)
    state = P3DerivedState{FT, typeof(q)}(nⁱ, nʳ, qᶠ, bᶠ, Fᶠ, ρᶠ,
                                          μ_ice, Fˡ_mu, Nᶜ, cloud.nᶜˡ,
                                          cloud.μ_c, cloud.λ_c,
                                          T, P, qᵛ, qᵛ⁺ˡ, qᵛ⁺ⁱ, q,
                                          transport.D_v, transport.K_a, transport.nu)

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

    # D2: Sublimation number loss
    sublim_mag = clamp_positive(-dep)
    sublim_n = sublim_mag * safe_divide(clamp_positive(nⁱ), max(clamp_positive(qⁱ), FT(1e-20)), zero(FT))

    # D1: Wet-ice coating condensation/evaporation comes from the coupled
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

    # M6: DSD number correction feedback
    ncl_correction = (cloud.nᶜˡ - ℳ.nᶜˡ) / dt_safety
    nr_correction = (nʳ - ℳ.nʳ) / dt_safety

    # Fortran's `microphy_p3.f90:5053-5063` recomputes ssat = qv - qvs(T)
    # at the end of the substep. Reconstruct that final diagnostic from the
    # locally adjusted (post-G&M) state and the sink-limited M&G rates. `cond`
    # here is the M&G part only; the one-shot G&M alignment is in `qᵛ`, `T`,
    # and `qᶜˡ` (already shifted by ε) and is folded into `cond_total` below.
    #
    # The `/ dt_safety` denominator (`sink_limiting_timescale`) ties this
    # tendency to the same time-step assumption as `cond_GM`: when the host
    # integrates with dt = sink_limiting_timescale the prognostic `ρsˢᵃᵗ`
    # lands exactly at `ρ(qᵛ_final - qᵛ⁺ˡ(T_final))`. For dt ≠ τ the alignment
    # relaxes over multiple steps.
    ℒˡ = vaporization_latent_heat(constants, T)
    ℒⁱ = sublimation_latent_heat(constants, T)
    ℒᶠ = fusion_latent_heat(constants, T)
    cᵖᵈ = p3_dry_air_heat_capacity(constants, FT)
    vapor_to_liquid = cond + ccn_act + rain_cond + coat_cond - rain_evap - coat_evap
    vapor_to_ice = dep + nuc_q
    liquid_to_ice = cloud_rim + rain_rim + cloud_frz_q + rain_frz_q +
                    cloud_hom_q + rain_hom_q + refrz -
                    complete_melt - partial_melt
    qᵛ_final = qᵛ - (vapor_to_liquid + vapor_to_ice) * dt_safety
    T_final = T + (ℒˡ * vapor_to_liquid + ℒⁱ * vapor_to_ice + ℒᶠ * liquid_to_ice) *
              dt_safety / cᵖᵈ
    qᵛ⁺ˡ_final = saturation_specific_humidity(T_final, ρ, constants, PlanarLiquidSurface())
    ssat_tendency = (qᵛ_final - qᵛ⁺ˡ_final - ℳ.sˢᵃᵗ) / dt_safety
    ssat_tendency = ifelse(prp.predict_supersaturation, ssat_tendency, zero(FT))
    # `cond_GM` is intentionally NOT rescaled by the cloud sink limiter: the
    # G&M alignment is its own one-shot saturation adjustment with a local
    # `ε ≥ -qᶜˡ` cap, and the cloud budget at the limiter sees `qᶜˡ_adjusted`
    # (= qᶜˡ + ε) as its starting state — so ε is absorbed into
    # `cloud_available`, not the source/sink list.
    cond_total = cond + cond_GM

    return P3ProcessRates{FT}(
        cond_total,
        autoconv, accr, rain_evap, rain_self, rain_br,
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
    # M9: CCN activation (vapor → cloud)
    gain = rates.condensation + rates.ccn_activation_mass
    # Phase 1: autoconversion and accretion
    # Phase 2: cloud riming by ice, immersion freezing, homogeneous freezing
    # Above-freezing: cloud collected by melting ice → qʷⁱ
    # Wet growth: cloud collection redirected to qʷⁱ
    loss = rates.autoconversion + rates.accretion + rates.cloud_riming +
           rates.cloud_freezing_mass + rates.cloud_homogeneous_mass +
           rates.cloud_warm_collection + rates.wet_growth_cloud
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
    # Phase 1: gains from autoconv, accr, complete_melt; loses from evap
    # Phase 2: gains from shedding; loses from riming, freezing, homogeneous freezing
    # Milbrandt et al. (2025): above-freezing collection and wet growth go to qʷⁱ, NOT rain.
    # Rain warm collection is a rain SINK (collected by ice → qʷⁱ).
    # M9: rain condensation (vapor → rain)
    # D8: wet_growth_shedding — excess collection beyond freezing capacity goes to rain.
    # Note: rain_warm_collection is zeroed at rate-assembly time in the non-liquid-
    # fraction branch (Fortran microphy_p3.f90:3055-3066) so it can safely be added
    # here unconditionally.
    gain = rates.autoconversion + rates.accretion + rates.complete_melting +
           rates.shedding + rates.rain_condensation + rates.wet_growth_shedding
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
- Cloud warm collection number (M9, Fortran ncshdc)

Rain number loses from:
- Self-collection (Phase 1)
- Evaporation (Phase 1) - proportional number removal
- Riming (Phase 2)
- Immersion freezing (Phase 2)
- Homogeneous freezing (Phase 2, T < -40°C)
- Rain warm collection number (M9, Fortran nrcoll)
"""
@inline function tendency_ρnʳ(rates::P3ProcessRates, ρ, nⁱ, qⁱ, nʳ, qʳ, prp::ProcessRateParameters)
    FT = typeof(ρ)

    # Phase 1: New drops from autoconversion
    n_from_autoconv = rates.autoconversion / prp.initial_rain_drop_mass

    # Phase 1: New drops from complete melting (conserve number)
    # Only complete_melting produces new rain drops; partial_melting stays on ice
    n_from_melt = safe_divide(nⁱ * rates.complete_melting, qⁱ, zero(FT))

    # Phase 1: Evaporation removes rain number proportionally (Fortran P3 v5.5.0)
    # rain_evaporation is positive magnitude (M7); proportional number loss is positive.
    n_from_evap = safe_divide(nʳ * rates.rain_evaporation, qʳ, zero(FT))

    # Gains: shedding produces rain drops
    # M9: cloud_warm_collection_number → new rain drops from above-freezing cloud
    #      collection (Fortran ncshdc). Only in non-liquid-fraction path;
    #      when liquid fraction is active, collected mass goes to qʷⁱ, not rain.
    # D8: wet_growth_shedding_number → rain drops from excess wet growth (Fortran nrshdr)
    cloud_warm_rain_n = ifelse(prp.liquid_fraction_active, zero(FT),
                               rates.cloud_warm_collection_number)
    n_gain = n_from_autoconv + n_from_melt +
             rates.rain_breakup +
             rates.shedding_number +
             cloud_warm_rain_n +
             rates.wet_growth_shedding_number
    # Losses (all positive magnitudes, M7)
    # M9: rain_warm_collection_number → rain number sink from above-freezing rain
    #      collection (Fortran nrcoll)
    n_loss = n_from_evap +
             rates.rain_self_collection +
             rates.rain_riming_number +
             rates.rain_freezing_number +
             rates.rain_homogeneous_number +
             rates.rain_warm_collection_number

    # M6: DSD number correction feedback (Fortran get_rain_dsd2 writes back bounded nr)
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
- Global number limiter (C3, impose_max_Ni)
"""
@inline function tendency_ρnⁱ(rates::P3ProcessRates, ρ)
    # Gains from nucleation, freezing, splintering, homogeneous freezing
    gain = rates.nucleation_number + rates.cloud_freezing_number +
           rates.rain_freezing_number + rates.splintering_number +
           rates.cloud_homogeneous_number + rates.rain_homogeneous_number
    # Losses (all positive magnitudes, M7)
    # D2: sublimation_number — ice number loss from sublimation (Fortran nisub)
    # ni_limit: C3 global Nᵢ cap (impose_max_Ni); relaxation sink above N_max/ρ.
    loss = rates.melting_number + rates.sublimation_number + rates.aggregation + rates.ni_limit
    return ρ * (gain - loss)
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
    gain = rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass +
           rates.wet_growth_densification_mass
    # Phase 1: melts and sublimates proportionally with ice mass
    # M8: sublimation (negative deposition) also removes rime proportionally
    sublimation = clamp_positive(-rates.deposition)
    # Splintering (nCat=1): Fortran subtracts splintering from riming then adds it back
    # as qcmul/qrmul, netting to zero effect on rime. Since cloud_riming and rain_riming
    # are the full (unreduced) rates, no splintering subtraction is needed here.
    loss = Fᶠ * (rates.partial_melting + rates.complete_melting + sublimation)
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

    ρᶠ_safe = max(ρᶠ, FT(100))
    ρ_rim_new_safe = max(rates.rime_density_new, FT(100))

    # Fortran P3 v5.5.0: rho_rimeMax = 900 for rain rime and freezing
    ρ_rimemax = prp.maximum_rime_density
    # D6: Fortran uses rho_rimeMax (900) for homogeneous freezing rime volume, not 917
    ρ_rim_hom = prp.maximum_rime_density

    # Phase 2: Volume gain from new rime
    # Cloud riming uses Cober-List computed density; rain riming uses rho_rimeMax = 900
    # Immersion freezing uses rho_rimeMax = 900 (Fortran convention, not water density)
    # Refreezing uses rho_rimeMax = 900 (Fortran: qifrz * i_rho_rimeMax, line 4253)
    volume_gain = rates.cloud_riming / ρ_rim_new_safe +
                   rates.rain_riming / ρ_rimemax +
                   rates.refreezing / ρ_rimemax +
                   (rates.cloud_freezing_mass + rates.rain_freezing_mass) / ρ_rimemax +
                   (rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass) / ρ_rim_hom +
                   rates.wet_growth_densification_volume

    # Phase 1: Volume loss from melting and sublimation (proportional to rime fraction)
    # M8: sublimation (negative deposition) also removes rime volume proportionally
    sublimation = clamp_positive(-rates.deposition)
    total_melting = rates.partial_melting + rates.complete_melting
    volume_loss = Fᶠ * (total_melting + sublimation) / ρᶠ_safe

    # M3: Melt-densification (Fortran P3 v5.5.0 lines 4309-4313)
    # Low-density rime portions melt first → remaining ice approaches 917 kg/m³.
    # In tendency form: additional volume reduction = bᶠ × (917 - ρᶠ) × |melt| / (ρᶠ × qⁱ)
    # Fortran guards with `.not. log_LiquidFrac`: when liquid fraction is active,
    # melt-densification is skipped because the liquid is tracked explicitly in qʷⁱ.
    # NOTE: The densification target is solid ice density (917), NOT rho_rimeMax (900).
    ρ_solid_ice = prp.pure_ice_density  # 917 kg/m³
    qⁱ_safe = max(qⁱ, FT(1e-12))
    bᶠ = Fᶠ * qⁱ_safe / ρᶠ_safe
    densification = bᶠ * (ρ_solid_ice - ρᶠ_safe) * total_melting / (ρᶠ_safe * qⁱ_safe)
    # Only apply when ρᶠ < 917, there is melting, AND liquid fraction is not active
    apply_densification = (ρᶠ_safe < ρ_solid_ice) & !prp.liquid_fraction_active
    densification = ifelse(apply_densification, densification, zero(FT))

    return ρ * (volume_gain - volume_loss - densification)
end

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

    # D32: Fortran uses D ≤ D_crit filtered tables (f1pr32/f1pr33) for Z melting when
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

    # D31: Wet growth Z contribution.
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
    z_rain_rime = rain_ice_table.sixth_moment(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    # Fortran convention: zqrcol = N0r × m6collr × env (no 10^f1pr08 factor).
    # Since rain_riming = N0r × ni × env × 10^f1pr08 (mass kernel),
    # divide by the mass kernel to recover: z = m6collr × rain_riming / (ni × mass_kernel).
    mass_kernel = exp10(rain_ice_table.mass(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ))
    inv_mass_kernel = safe_divide(one(FT), mass_kernel, zero(FT))
    return z_rain_rime * inv_nⁱ * inv_mass_kernel
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

"""
$(TYPEDSIGNATURES)

Compute cloud-number tendency from P3 process rates.

Activation creates new cloud droplets. Autoconversion, accretion, riming,
freezing, and above-freezing collection remove cloud droplets in proportion
to the cloud mass they consume, following the Fortran `nc` budget structure.
"""
@inline function tendency_ρnᶜˡ(rates::P3ProcessRates, ρ, Nᶜ, qᶜˡ, prp::ProcessRateParameters)
    FT = typeof(ρ)
    # Nᶜ is per-volume [#/m³]; dividing by ρ gives per-mass nᶜˡ [#/kg],
    # matching Fortran's nc/qc → [#/kg/s] when multiplied by mass rates.
    number_per_mass = safe_divide(Nᶜ, ρ * qᶜˡ, zero(FT))
    seed_drop_mass = 4 * FT(π) / 3 * prp.liquid_water_density * FT(1e-18)
    activation_number = ifelse(iszero(rates.ccn_activation_number),
                               rates.ccn_activation_mass / seed_drop_mass,
                               rates.ccn_activation_number)

    number_loss = number_per_mass * (rates.autoconversion + rates.accretion) +
                  rates.cloud_riming_number +
                  rates.cloud_freezing_number +
                  rates.cloud_homogeneous_number +
                  rates.cloud_warm_collection_number

    # M6: DSD number correction feedback (Fortran get_cloud_dsd2 writes back bounded nc)
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
    # D1: Include coating condensation/evaporation (Fortran qlcon/qlevp)
    # D8: wet_growth_shedding diverts excess wet growth mass from qʷⁱ to rain.
    # Note: rain_warm_collection is zeroed at rate-assembly time in the non-liquid-
    # fraction branch (Fortran does not transfer rain mass to qʷⁱ in that path), so
    # it can safely be added here unconditionally.
    gain = rates.partial_melting +
           rates.cloud_warm_collection +
           rates.rain_warm_collection +
           rates.wet_growth_cloud +
           rates.wet_growth_rain +
           rates.coating_condensation
    loss = rates.shedding + rates.refreezing + rates.coating_evaporation +
           rates.wet_growth_shedding
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
    # M9: CCN activation, rain condensation, and coating condensation are all vapor sinks;
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

#####
##### Fallback methods for Nothing rates
#####
##### These are safety fallbacks that return zero tendency when rates
##### have not been computed (e.g., during incremental development).
#####

@inline tendency_ρqᶜˡ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqʳ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnᶜˡ(::Nothing, ρ, Nᶜ, qᶜˡ, prp) = zero(ρ)
@inline tendency_ρnʳ(::Nothing, ρ, nⁱ, qⁱ, args...) = zero(ρ)
@inline tendency_ρqⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᶠ(::Nothing, ρ, Fᶠ) = zero(ρ)
@inline tendency_ρbᶠ(::Nothing, ρ, Fᶠ, ρᶠ, prp...) = zero(ρ)
@inline tendency_ρzⁱ(::Nothing, ρ, qⁱ, nⁱ, zⁱ) = zero(ρ)
@inline tendency_ρqʷⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρsˢᵃᵗ(::Nothing, ρ, prp) = zero(ρ)
@inline tendency_ρqᵛ(::Nothing, ρ) = zero(ρ)
