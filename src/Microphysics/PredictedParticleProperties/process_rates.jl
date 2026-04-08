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
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             density,
                             liquid_latent_heat,
                             ice_latent_heat,
                             mixture_heat_capacity,
                             vapor_gas_constant,
                             MoistureMassFractions

#####
##### Utility functions
#####

"""
    clamp_positive(x)

Return max(0, x) for numerical stability.
"""
@inline clamp_positive(x) = max(0, x)

"""
    sink_limiting_factor(total_sink, available_mass, dt_safety)

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
    safe_divide(a, b, default)

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
    consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, qʷⁱ)

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

@inline lookup_table_1(p3) = _lookup_field(p3.ice.lookup_tables, Val(:table_1))
@inline lookup_table_2(p3) = _lookup_field(p3.ice.lookup_tables, Val(:table_2))
@inline lookup_table_3(p3) = _lookup_field(p3.ice.lookup_tables, Val(:table_3))

@inline _lookup_field(tables::P3LookupTables, ::Val{:table_1}) = tables.table_1
@inline _lookup_field(tables::P3LookupTables, ::Val{:table_2}) = tables.table_2
@inline _lookup_field(tables::P3LookupTables, ::Val{:table_3}) = tables.table_3
@inline _lookup_field(::Nothing, ::Val) = nothing

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

@inline function ice_mean_density_for_bounds(table1::P3LookupTable1, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    FT = typeof(qⁱ_total)
    m̄ = safe_divide(max(qⁱ_total, FT(1e-20)), max(nⁱ, FT(1e-16)), FT(1e-20))
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    return table1.bulk_properties.mean_density(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
end

@inline function bound_ice_sixth_moment(table1::P3LookupTable1, qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    FT = typeof(qⁱ_total)
    has_ice = (qⁱ_total > FT(1e-20)) & (nⁱ > FT(1e-16))
    ρ_bulk = ice_mean_density_for_bounds(table1, qⁱ_total, nⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    μ_bounds = ThreeMomentClosure(FT)
    z_bounded = enforce_z_bounds(clamp_positive(zⁱ), qⁱ_total, nⁱ, ρ_bulk, μ_bounds.μmin, μ_bounds.μmax)
    return ifelse(has_ice, z_bounded, zero(FT))
end

@inline bound_ice_sixth_moment(::Nothing, qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ) = clamp_positive(zⁱ)

@inline function bound_ice_sixth_moment(p3, qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
    return bound_ice_sixth_moment(lookup_table_1(p3), qⁱ_total, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, μ)
end

#####
##### Ice shape parameter (μ) from Table 3
#####
##### For 3-moment P3, μ is diagnosed from (qⁱ, nⁱ, zⁱ) via Table 3.
##### For 2-moment (no Table 3), μ is diagnosed from the P3Closure μ-λ
##### relationship using an analytical λ approximation for unrimed aggregates.
#####

"""
    compute_ice_shape_parameter(p3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ)

Compute the ice PSD shape parameter μ.

For 3-moment P3, μ is diagnosed from the ratio Z/L (sixth moment to mass)
using the pre-tabulated closure in Table 3.

For 2-moment P3 (Table 3 absent), μ is computed from the P3Closure
μ-λ diagnostic relationship. The slope parameter λ is approximated from
the mean particle mass m̄ = qⁱ/nⁱ assuming the unrimed aggregate regime
(m(D) = α Dᵝ with exponential PSD μ=0 as starting estimate):
log λ ≈ (log α + log Γ(β+1) - log m̄) / β. This approximation is
GPU-compatible and allocation-free (no incomplete gamma functions).

All inputs are specific (per kg of air); the Table 3 lookup normalizes
by ratios so units cancel, and the 2-moment closure is also scale-free.
"""
@inline function compute_ice_shape_parameter(p3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ)
    FT = typeof(qⁱ)
    table3 = lookup_table_3(p3)
    return _ice_shape_parameter(table3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, FT)
end

@inline function _ice_shape_parameter(table3::P3LookupTable3, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, FT)
    # Clamp to physical range to avoid table lookup at unphysical values
    qⁱ_safe = max(qⁱ, eps(FT))
    nⁱ_safe = max(nⁱ, eps(FT))
    zⁱ_safe = max(zⁱ, eps(FT))
    return shape_parameter_lookup(table3, qⁱ_safe, nⁱ_safe, zⁱ_safe, Fᶠ, Fˡ, ρᶠ)
end

# Fallback: no Table 3 (2-moment mode). Compute μ from the P3Closure μ-λ diagnostic.
# λ is approximated from m_mean = qⁱ/nⁱ assuming the unrimed aggregate regime (regime 2)
# with exponential PSD (μ=0) as initial estimate:
#   m_mean = α Γ(β+1) / λ^β  →  logλ = (log α + loggamma(β+1) - log m_mean) / β
# μ is then computed from the P3Closure small-particle (Field et al. 2007) μ-λ relation
# via TwoMomentClosure (which is identical to the P3Closure small-particle branch).
# This is GPU-safe (no gamma_inc), allocation-free, and scale-free.
@inline function _ice_shape_parameter(::Nothing, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, FT)
    qⁱ_eff = max(qⁱ, FT(1e-20))
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m_mean = qⁱ_eff / nⁱ_eff

    mass = IceMassPowerLaw(FT)
    α = mass.coefficient
    β = mass.exponent

    # logλ from mean mass via unrimed aggregate mass-diameter relation
    logλ = (log(α) + loggamma(β + 1) - log(m_mean)) / β
    logλ = clamp(logλ, log(FT(10)), log(FT(1e7)))

    # TwoMomentClosure = P3Closure small-particle branch (Field et al. 2007):
    # μ = clamp(a λ^b - c, 0, μmax)
    closure = TwoMomentClosure(FT)
    return shape_parameter(closure, logλ)
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

@inline fusion_latent_heat(constants, T) = sublimation_latent_heat(constants, T) - vaporization_latent_heat(constants, T)

#####
##### Ice psychrometric correction Γⁱ
#####
##### Accounts for the latent-heat feedback that reduces the effective
##### supersaturation drive during ice deposition (Fortran P3 "abi" factor).
##### Γⁱ = 1 + Lₛ² qᵛ⁺ⁱ / (Rᵛ T² cᵖ)
##### Analogous to Γˡ in cloud_condensation_rate; both are linearisations
##### of the saturation adjustment Jacobian as used in SaturationAdjustment.
#####

@inline function ice_psychrometric_correction(::Nothing, q, L_s, qᵛ⁺ⁱ, Rᵛ, T)
    FT = typeof(T)
    cₚᵈ = FT(1004.64)   # Fortran P3 dry-air heat capacity [J/(kg·K)]
    return 1 + L_s^2 * qᵛ⁺ⁱ / (Rᵛ * T^2 * cₚᵈ)
end

@inline function ice_psychrometric_correction(constants, q, L_s, qᵛ⁺ⁱ, Rᵛ, T)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    return 1 + L_s^2 * qᵛ⁺ⁱ / (Rᵛ * T^2 * cᵖᵐ)
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
    ventilation_sc_correction(nu, D_v, ρ_correction)

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
    deposition_ventilation(vent, vent_e, m_mean, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

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
    melting_ventilation(vent, vent_e, m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

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
    collection_kernel_per_particle(coll, m_mean, Fᶠ, ρᶠ, prp, p3, μ)

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
    aggregation_kernel(coll, m_mean, Fᶠ, ρᶠ, prp, p3, μ)

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
##### Cloud condensation/evaporation
#####

"""
    cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)

Compute cloud liquid condensation/evaporation rate using relaxation-to-saturation.

When the air is supersaturated (qᵛ > qᵛ⁺ˡ), excess vapor condenses onto cloud
droplets. When subsaturated, cloud liquid evaporates back to vapor. The rate
follows a relaxation timescale with a thermodynamic (psychrometric) correction
factor that accounts for latent heating during phase change.

# Arguments
- `p3`: P3 microphysics scheme (provides condensation timescale)
- `qᶜˡ`: Cloud liquid mass fraction [kg/kg]
- `qᵛ`: Vapor mass fraction [kg/kg]
- `qᵛ⁺ˡ`: Saturation vapor mass fraction over liquid [kg/kg]
- `T`: Temperature [K]
- `q`: Moisture mass fractions (vapor, liquid, ice)
- `constants`: Thermodynamic constants

# Returns
- Rate of vapor → cloud liquid conversion [kg/kg/s]
  (positive = condensation, negative = evaporation)
"""
@inline function cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)
    FT = typeof(qᶜˡ)
    τᶜˡ = p3.cloud.condensation_timescale

    # Thermodynamic adjustment factor (psychrometric correction)
    ℒˡ = liquid_latent_heat(T, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Rᵛ = vapor_gas_constant(constants)
    dqᵛ⁺_dT = qᵛ⁺ˡ * ℒˡ / (Rᵛ * T^2)
    Γˡ = 1 + (ℒˡ / cᵖᵐ) * dqᵛ⁺_dT

    # Relaxation toward saturation
    Sᶜᵒⁿᵈ = (qᵛ - qᵛ⁺ˡ) / (Γˡ * τᶜˡ)

    # Limit evaporation to available cloud liquid (include Γˡ for consistency)
    Sᶜᵒⁿᵈ_min = -max(0, qᶜˡ) / (Γˡ * τᶜˡ)
    return max(Sᶜᵒⁿᵈ, Sᶜᵒⁿᵈ_min)
end

#####
##### Ice deposition and sublimation
#####

"""
    ventilation_enhanced_deposition(p3, qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, constants, transport, q, μ)

Compute ventilation-enhanced ice deposition/sublimation rate with latent-heat
psychrometric correction.

Following Morrison & Milbrandt (2015a) Eq. 30, the single-particle growth rate is:

```math
\\frac{dm}{dt} = \\frac{4πC f_v (S_i - 1)}{Γⁱ \\left[\\frac{L_s}{K_a T}\\left(\\frac{L_s}{R_v T} - 1\\right) + \\frac{R_v T}{e_{si} D_v}\\right]}
```

where ``Γⁱ = 1 + L_s^2 q^{v+i} / (R_v T^2 c_p^m)`` is the latent-heat psychrometric
correction (analogous to Fortran P3's `abi` factor and to ``Γˡ`` in
[`cloud_condensation_rate`](@ref)). It accounts for the reduction in the effective
supersaturation drive caused by latent heat released during deposition and is
consistent with Breeze's `SaturationAdjustment` Jacobian linearisation.

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
@inline function ventilation_enhanced_deposition(p3, qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P,
                                                  constants, transport, q, μ)
    FT = typeof(qⁱ)
    prp = p3.process_rates

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = clamp_positive(nⁱ)
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)

    # When runtime thermodynamic constants are provided, use their gas constants
    # consistently with the latent heat and saturation calculations.
    thermodynamic_constants = isnothing(constants) ? ThermodynamicConstants(FT) : constants
    Rᵛ = FT(vapor_gas_constant(thermodynamic_constants))
    Rᵈ = FT(dry_air_gas_constant(thermodynamic_constants))
    L_s = sublimation_latent_heat(constants, T)
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

    ρ_air = density(T, P, q, thermodynamic_constants)
    ρ_correction = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ_air)

    # PSD-integrated ventilation integral C(D) × f_v(D) from lookup table.
    C_fv = deposition_ventilation(p3.ice.deposition.ventilation,
                                    p3.ice.deposition.ventilation_enhanced,
                                    m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)

    # Denominator: thermodynamic resistance terms (Mason 1971)
    # A = L_s/(K_a × T) × (L_s/(R_v × T) - 1)
    # B = R_v × T / (e_si × D_v)
    A = L_s / (K_a * T) * (L_s / (Rᵛ * T) - 1)
    B = Rᵛ * T / (e_si * D_v)
    thermodynamic_factor = A + B

    # Latent-heat psychrometric correction Γⁱ (Fortran P3 "abi"):
    # Reduces the effective supersaturation drive to account for the
    # warming produced by the latent heat of deposition.
    # Γⁱ = 1 + Lₛ² qᵛ⁺ⁱ / (Rᵛ T² cᵖᵐ)  ≡  1 + (Lₛ/cᵖᵐ) dqᵛ⁺ⁱ/dT
    Γⁱ = ice_psychrometric_correction(constants, q, L_s, qᵛ⁺ⁱ_safe, Rᵛ, T)

    # Deposition rate per particle (Eq. 30 from MM15a)
    # Uses 2π (not 4π) because the ventilation integral stores capm = cap × D
    # (P3 Fortran convention), which is 2× the physical capacitance C = D/2.
    # The product 2π × capm = 2π × 2C = 4πC is physically correct.
    dm_dt = FT(2π) * C_fv * (S_i - 1) / (Γⁱ * thermodynamic_factor)

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
##### Combined P3 tendency calculation
#####

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

    # M9: Warm/mixed-phase budget terms (stubs for Fortran parity)
    ccn_activation :: FT               # CCN activation (vapor → cloud) [kg/kg/s]
    rain_condensation :: FT            # Rain condensation (vapor → rain) [kg/kg/s]
    coating_condensation :: FT         # Condensation on ice liquid coating [kg/kg/s]
    coating_evaporation :: FT          # Evaporation from ice liquid coating [kg/kg/s]
end

"""
    compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

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
    FT = typeof(ρ)
    prp = p3.process_rates
    T₀ = prp.freezing_temperature

    # Extract from microphysical state (already specific, not density-weighted)
    qᶜˡ = ℳ.qᶜˡ
    qʳ = ℳ.qʳ
    nʳ = ℳ.nʳ
    qⁱ = ℳ.qⁱ
    nⁱ = ℳ.nⁱ
    qʷⁱ = ℳ.qʷⁱ

    # D10: Fortran impose_max_Ni — hard cap on ice number before PSD solve.
    # Fortran: nitot = min(nitot, max_Ni / rho) where max_Ni = 2e6 [1/m³].
    # This prevents unphysical PSD parameters from extreme number concentrations.
    nⁱ = min(nⁱ, prp.maximum_ice_number_density / ρ)

    # H4: Rain DSD lambda bounds and Nr adjustment (Fortran get_rain_dsd2).
    # When λ_r hits DSD bounds, recompute nʳ to stay mass-consistent with qʳ.
    # λ_r = (π ρ_w nʳ / qʳ)^(1/3)  →  nʳ = qʳ λ_r³ / (π ρ_w)
    rain_active = (qʳ > FT(1e-14)) & (nʳ > FT(1e-16))
    qʳ_pos = clamp_positive(qʳ)
    nʳ_pos = clamp_positive(nʳ)
    λ_r = rain_slope_parameter(qʳ_pos, nʳ_pos, prp)
    nʳ = ifelse(rain_active, qʳ_pos * λ_r^3 / (FT(π) * prp.liquid_water_density), nʳ)

    # Rime properties (Fortran calc_bulkRhoRime consistency)
    rime_state = consistent_rime_state(p3, qⁱ, ℳ.qᶠ, ℳ.bᶠ, qʷⁱ)
    qᶠ = rime_state.qᶠ
    bᶠ = rime_state.bᶠ
    Fᶠ = rime_state.Fᶠ
    ρᶠ = rime_state.ρᶠ

    # Ice PSD shape parameter μ from Table 3 (3-moment) or 0 (2-moment fallback).
    # Liquid fraction for Table 3 lookup uses qʷⁱ / (qⁱ + qʷⁱ).
    qⁱ_total_mu = max(clamp_positive(qⁱ) + clamp_positive(qʷⁱ), FT(1e-20))
    Fˡ_mu = clamp_positive(qʷⁱ) / qⁱ_total_mu
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_total_mu, nⁱ, ℳ.zⁱ, Fᶠ, Fˡ_mu, ρᶠ)

    # Thermodynamic state
    T = temperature(𝒰, constants)
    qᵛ = 𝒰.moisture_mass_fractions.vapor

    # Saturation vapor mixing ratios using Breeze thermodynamics
    qᵛ⁺ˡ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())

    # Moisture mass fractions for thermodynamic calculations
    q = 𝒰.moisture_mass_fractions

    # Cloud droplet number concentration
    Nᶜ = p3.cloud.number_concentration

    # =========================================================================
    # Phase 1: Cloud condensation/evaporation
    # =========================================================================
    cond = cloud_condensation_rate(p3, qᶜˡ, qᵛ, qᵛ⁺ˡ, T, q, constants)

    # =========================================================================
    # Phase 1: Rain processes
    # =========================================================================
    P = 𝒰.reference_pressure

    # Compute T,P-dependent transport properties once (Fortran P3 v5.5.0 formulas)
    # — shared by deposition, melting, and rain evaporation (eliminates triple computation)
    transport = air_transport_properties(T, P)

    autoconv = rain_autoconversion_rate(p3, qᶜˡ, Nᶜ, ρ)
    accr = rain_accretion_rate(p3, qᶜˡ, qʳ)
    rain_evap = rain_evaporation_rate(p3, qʳ, nʳ, qᵛ, qᵛ⁺ˡ, T, ρ, P, transport)
    rain_self = rain_self_collection_rate(p3, qʳ, nʳ, ρ)
    rain_br = rain_breakup_rate(p3, qʳ, nʳ, rain_self)

    # =========================================================================
    # Phase 1: Ice deposition/sublimation and melting
    # =========================================================================
    dep = ventilation_enhanced_deposition(p3, qⁱ, qʷⁱ, nⁱ, qᵛ, qᵛ⁺ⁱ, Fᶠ, ρᶠ, T, P, constants, transport, q, μ_ice)
    dep = ifelse(qⁱ > FT(1e-20), dep, zero(FT))

    # Partitioned melting: partial stays on ice, complete goes to rain
    melt_rates = ice_melting_rates(p3, qⁱ, nⁱ, qʷⁱ, T, P, qᵛ, qᵛ⁺ˡ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)
    partial_melt = melt_rates.partial_melting
    complete_melt = melt_rates.complete_melting
    # Only complete melting removes ice particles; partial melting keeps particles as ice
    melt_n = ice_melting_number_rate(qⁱ, nⁱ, complete_melt)

    # =========================================================================
    # Phase 2: Ice aggregation
    # =========================================================================
    agg = ice_aggregation_rate(p3, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)

    # =========================================================================
    # C3: Global ice number limiter (Fortran impose_max_Ni)
    # =========================================================================
    # Relaxation sink: remove excess Nᵢ above N_max/ρ over one dt_safety period.
    # Using a relaxation rate (not a hard clamp) maintains GPU-kernel compatibility
    # and is consistent with the existing sink_limiting_factor pattern.
    N_max = prp.maximum_ice_number_density
    ni_lim = clamp_positive(nⁱ - N_max / ρ) / prp.sink_limiting_timescale

    # =========================================================================
    # Phase 2: Riming
    # =========================================================================
    cloud_rim = cloud_riming_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    cloud_rim_n = cloud_riming_number_rate(qᶜˡ, Nᶜ, cloud_rim)

    # C5: Pass nʳ so rain_riming_rate can apply the rain-DSD cross-section correction.
    rain_rim = rain_riming_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    rain_rim_n = rain_riming_number_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)

    # Rime density for new rime (use actual ice fall speed, not placeholder)
    # D16: Pass actual Fˡ so mixed-phase particles use correct (denser/faster) fall speed.
    vᵢ = ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=Fˡ_mu, μ=μ_ice)
    ρᶠ_new = rime_density(p3, qᶜˡ, cloud_rim, T, vᵢ, ρ, constants, transport)

    # =========================================================================
    # Phase 2: Wet growth capacity and collection rerouting
    # (Milbrandt et al. 2025; Fortran qwgrth/qwgrth1c/qwgrth1r)
    # =========================================================================
    # When collection rate exceeds the freezing capacity (wet growth),
    # all collected hydrometeors stay liquid and are redirected to qʷⁱ.
    # D4: Fortran guards wet growth with (qc+qr) >= 1e-6 (microphy_p3.f90 line 3241)
    has_hydrometeors = (clamp_positive(qᶜˡ) + clamp_positive(qʳ)) >= FT(1e-6)
    qwgrth_raw = wet_growth_capacity(p3, qⁱ, qʷⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)
    qwgrth = ifelse(has_hydrometeors, qwgrth_raw, zero(FT))

    # Check if total riming exceeds wet growth capacity
    total_collection = cloud_rim + rain_rim
    is_wet_growth = total_collection > qwgrth + FT(1e-10)

    # During wet growth: redirect ALL collection to qʷⁱ, zero out rime
    wg_cloud = ifelse(is_wet_growth, cloud_rim, zero(FT))
    wg_rain  = ifelse(is_wet_growth, rain_rim, zero(FT))
    cloud_rim   = ifelse(is_wet_growth, zero(FT), cloud_rim)
    cloud_rim_n = ifelse(is_wet_growth, zero(FT), cloud_rim_n)
    rain_rim    = ifelse(is_wet_growth, zero(FT), rain_rim)
    rain_rim_n  = ifelse(is_wet_growth, zero(FT), rain_rim_n)

    # D8: Wet growth shedding (Fortran nrshdr/qcshd).
    # Fortran only sheds excess when log_LiquidFrac = .FALSE.; when liquid fraction
    # is active, ALL collection goes to qiliq with no shedding (lines 3254-3264).
    # Shed drops are 1 mm diameter (Fortran 1.923e6 drops/kg).
    shed_active = !prp.liquid_fraction_active & is_wet_growth
    wg_shed   = ifelse(shed_active, clamp_positive(total_collection - qwgrth), zero(FT))
    wg_shed_n = wg_shed * FT(1.923e6)

    # =========================================================================
    # Phase 2: Shedding and refreezing
    # =========================================================================
    # Liquid fraction for shedding (Fl = qʷⁱ / (qⁱ + qʷⁱ))
    qⁱ_total = max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20))
    Fˡ = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    D_mean = first(mean_ice_particle_diameter(m_mean, Fᶠ, Fˡ, ρᶠ, prp))

    shed = shedding_rate(p3, qʷⁱ, qⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, m_mean, μ_ice)
    shed_n = shedding_number_rate(p3, shed)
    refrz = refreezing_rate(p3, qʷⁱ, qⁱ, nⁱ, T, P, qᵛ, Fᶠ, ρᶠ, ρ, constants, transport, μ_ice)

    # Liquid fraction clipping (Fortran freeze_tiny_liqfrac, lines 11620-11624):
    # When Fl < liqfracsmall below freezing, drain all qʷⁱ → rime.
    # When Fl > 1-liqfracsmall above freezing, drain all qʷⁱ → rain (complete melt).
    # Implemented as relaxation over refreezing_timescale (Fortran does it instantaneously).
    Fl_small = prp.liquid_fraction_small
    τ_clip = prp.refreezing_timescale
    qʷⁱ_eff = clamp_positive(qʷⁱ)
    clip_freeze = (T < T₀) & (Fˡ < Fl_small) & (Fˡ > 0)
    clip_melt   = (T >= T₀) & (Fˡ > 1 - Fl_small)
    refrz = ifelse(clip_freeze, refrz + qʷⁱ_eff / τ_clip, refrz)
    complete_melt = ifelse(clip_melt, complete_melt + qʷⁱ_eff / τ_clip, complete_melt)

    # =========================================================================
    # Ice nucleation (deposition nucleation and immersion freezing)
    # =========================================================================
    nuc_q, nuc_n = deposition_nucleation_rate(p3, T, qᵛ, qᵛ⁺ⁱ, nⁱ, ρ)
    cloud_frz_q, cloud_frz_n = immersion_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    rain_frz_q, rain_frz_n = immersion_freezing_rain_rate(p3, qʳ, nʳ, T)

    # =========================================================================
    # Rime splintering (Hallett-Mossop secondary ice production)
    # =========================================================================
    spl_q, spl_n = rime_splintering_rate(p3, cloud_rim, rain_rim, T, D_mean, Fˡ, T, qᶠ)

    # =========================================================================
    # Homogeneous freezing (T < -40°C, instantaneous conversion)
    # =========================================================================
    cloud_hom_q, cloud_hom_n = homogeneous_freezing_cloud_rate(p3, qᶜˡ, Nᶜ, T, ρ)
    rain_hom_q, rain_hom_n = homogeneous_freezing_rain_rate(p3, qʳ, nʳ, T)

    # =========================================================================
    # Above-freezing collection (Milbrandt et al. 2025: qccoll/qrcoll → qʷⁱ)
    # =========================================================================
    cloud_warm_q, cloud_warm_n_raw = cloud_warm_collection_rate(p3, qᶜˡ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    # D19: In the liquid-fraction path, above-freezing cloud collection sends mass
    # to qʷⁱ (not rain), so no rain number should be created. The ncshdc formula
    # only exists in Fortran's non-liquid-fraction path.
    cloud_warm_n = ifelse(prp.liquid_fraction_active, zero(FT), cloud_warm_n_raw)
    rain_warm_q = rain_warm_collection_rate(p3, qʳ, nʳ, qⁱ, nⁱ, T, Fᶠ, ρᶠ, ρ, μ_ice, qʷⁱ)
    # M9: Rain number loss from above-freezing collection (Fortran nrcoll).
    # Proportional to mass collected: Δnʳ/nʳ = Δqʳ/qʳ
    rain_warm_n = safe_divide(nʳ * rain_warm_q, qʳ, zero(FT))

    # =========================================================================
    # Sink limiting: rescale sink rates so total sinks × dt_safety ≤ available
    # mass for each species. Prevents negative mixing ratios with explicit
    # time integration.
    # M16: source-inclusive limiting (Fortran convention):
    #   available = current_mass + source_rates × dt_safety
    # This accounts for mass produced within the timestep.
    # =========================================================================
    dt_safety = prp.sink_limiting_timescale

    # --- Cloud liquid sinks ---
    cloud_source_total = max(0, cond)
    cloud_available = max(0, qᶜˡ) + cloud_source_total * dt_safety
    cloud_sink_total = autoconv + accr + cloud_rim + cloud_frz_q +
                       cloud_hom_q + cloud_warm_q + wg_cloud
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

    # --- Rain sinks ---
    # D17: Include rain_evap in sink coordination (Fortran jointly limits all sinks).
    rain_source_total = autoconv + accr + complete_melt + shed + wg_shed
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

    # --- Dry-ice sinks ---
    # Julia carries dry ice and liquid-on-ice separately, so keep a dry-pool
    # limiter to prevent partial/complete melting and sublimation from
    # overdrawing qⁱ before the total-ice coordination below.
    ice_source_total = max(0, dep) + cloud_rim + rain_rim + refrz +
                       nuc_q + cloud_frz_q + rain_frz_q +
                       cloud_hom_q + rain_hom_q
    ice_available = max(0, qⁱ) + ice_source_total * dt_safety
    ice_sink_total = partial_melt + complete_melt + clamp_positive(-dep)
    f_ice = sink_limiting_factor(ice_sink_total, ice_available, dt_safety)
    partial_melt  = partial_melt * f_ice
    complete_melt = complete_melt * f_ice
    melt_n        = melt_n * f_ice
    dep           = ifelse(dep < 0, dep * f_ice, dep)

    # --- Vapor sinks ---
    vapor_source_total = rain_evap + clamp_positive(-dep) + clamp_positive(-cond)
    vapor_available = max(0, qᵛ) + vapor_source_total * dt_safety
    vapor_sink_total = max(0, cond) + max(0, dep) + nuc_q
    f_vapor = sink_limiting_factor(vapor_sink_total, vapor_available, dt_safety)
    cond  = ifelse(cond > 0, cond * f_vapor, cond)
    dep   = ifelse(dep > 0, dep * f_vapor, dep)
    nuc_q = nuc_q * f_vapor
    nuc_n = nuc_n * f_vapor

    # D2: Sublimation number loss (Fortran nisub = qisub * ni/qi).
    # D1 later adds the nlevp-equivalent coating-evaporation number sink to
    # this same carried loss term.
    sublim_mag = clamp_positive(-dep)
    sublim_n = sublim_mag * safe_divide(clamp_positive(nⁱ), max(clamp_positive(qⁱ), FT(1e-20)), zero(FT))

    # D1: Coating condensation/evaporation on ice liquid fraction
    # Vapor condenses onto (or evaporates from) the liquid coating of ice particles,
    # using the same ventilation integral as deposition but driven by liquid saturation.
    # Fortran: qlcon/qlevp (microphy_p3.f90 lines 3746-3771).
    #
    # Fortran exclusive branching: when Fl >= 1%, ALL diffusional growth goes to
    # the liquid coating (epsiw) and deposition (epsi) is zeroed. When Fl < 1%,
    # all goes to deposition and coating is zeroed. They never operate simultaneously.
    qⁱ_total_coat = max(total_ice_mass(qⁱ, qʷⁱ), FT(1e-20))
    Fˡ_coat = liquid_fraction_on_ice(qⁱ, qʷⁱ)
    has_coating = Fˡ_coat >= FT(0.01)  # Fortran threshold: qiliq/qitot >= 0.01
    # Compute liquid saturation ratio S_l = qv / qv_sat_liq
    S_l = qᵛ / max(qᵛ⁺ˡ, FT(1e-10))
    # Use same ventilation integral as deposition
    m_mean_coat = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ)
    ρ_correction_coat = ice_air_density_correction(p3.ice.fall_speed.reference_air_density, ρ)
    C_fv_coat = deposition_ventilation(p3.ice.deposition.ventilation,
                                        p3.ice.deposition.ventilation_enhanced,
                                        m_mean_coat, Fᶠ, Fˡ_coat, ρᶠ, prp, transport.nu, transport.D_v,
                                        ρ_correction_coat, p3, μ_ice)
    thermodynamic_constants_coat = isnothing(constants) ? ThermodynamicConstants(FT) : constants
    Rᵛ_coat = FT(vapor_gas_constant(thermodynamic_constants_coat))
    Rᵈ_coat = FT(dry_air_gas_constant(thermodynamic_constants_coat))
    L_v_coat = vaporization_latent_heat(constants, T)
    ε_coat = Rᵈ_coat / Rᵛ_coat
    qᵛ⁺ˡ_safe = max(qᵛ⁺ˡ, FT(1e-30))
    e_sl = P * qᵛ⁺ˡ_safe / (ε_coat + qᵛ⁺ˡ_safe * (1 - ε_coat))
    A_coat = L_v_coat / (transport.K_a * T) * (L_v_coat / (Rᵛ_coat * T) - 1)
    B_coat = Rᵛ_coat * T / (e_sl * transport.D_v)
    Γˡ = 1 + L_v_coat^2 * qᵛ⁺ˡ_safe / (Rᵛ_coat * T^2 * FT(1005))
    # D15: Fortran uses unscaled rate (binary Fl >= 0.01 switch, not continuous scaling).
    coat_rate = FT(2π) * C_fv_coat * (S_l - 1) / (Γˡ * (A_coat + B_coat)) * clamp_positive(nⁱ)
    coat_cond = ifelse(has_coating, clamp_positive(coat_rate), zero(FT))
    # Cap evaporation at available liquid coating (Fortran: min(qlevp, qiliq*i_dt))
    τ_coat = prp.sink_limiting_timescale
    coat_evap_raw = ifelse(has_coating, clamp_positive(-coat_rate), zero(FT))
    coat_evap = min(coat_evap_raw, clamp_positive(qʷⁱ) / τ_coat)

    # Fortran exclusive branching: when coating is active, zero out ice deposition
    # and sublimation number to avoid double-counting vapor consumption
    # (Fortran: epsi=0 when epsiw>0; nisub=0 follows from qisub=0).
    dep = ifelse(has_coating, zero(FT), dep)
    sublim_n = ifelse(has_coating, zero(FT), sublim_n)

    # D17: Fortran also limits sinks against total ice qitot = qⁱ + qʷⁱ.
    # Keep the split-pool guards above, then apply this shared limiter to the
    # processes that genuinely remove total ice to rain/vapor.
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

    # --- Liquid on ice (qʷⁱ) sinks ---
    # D8: wg_shed diverts incoming wet growth mass from qʷⁱ to rain.
    # D17: Include coat_evap in qʷⁱ sinks (Fortran jointly limits qlevp + qlshd + qifrz).
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

    coat_evap_n = coat_evap * safe_divide(clamp_positive(nⁱ), qⁱ_total_coat, zero(FT))
    sublim_n = sublim_n + coat_evap_n

    # Recompute splintering from sink-limited riming rates.
    cloud_spl_q, rain_spl_q, spl_n = rime_splintering_rates(p3, cloud_rim, rain_rim, T, D_mean, Fˡ, T, qᶠ)
    cloud_spl_q = min(cloud_spl_q, clamp_positive(cloud_rim))
    rain_spl_q = min(rain_spl_q, clamp_positive(rain_rim))
    spl_q = cloud_spl_q + rain_spl_q

    return P3ProcessRates(
        # Phase 1: Condensation
        cond,
        # Phase 1: Rain
        autoconv, accr, rain_evap, rain_self, rain_br,
        # Phase 1: Ice
        dep, partial_melt, complete_melt, melt_n,
        # D2: Sublimation number loss
        sublim_n,
        # Phase 2: Aggregation + C3 global Nᵢ limiter
        agg, ni_lim,
        # Phase 2: Riming
        cloud_rim, cloud_rim_n, rain_rim, rain_rim_n, ρᶠ_new,
        # Phase 2: Shedding and refreezing
        shed, shed_n, refrz,
        # Ice nucleation
        nuc_q, nuc_n, cloud_frz_q, cloud_frz_n, rain_frz_q, rain_frz_n,
        # Rime splintering
        spl_q, spl_n,
        # Homogeneous freezing
        cloud_hom_q, cloud_hom_n, rain_hom_q, rain_hom_n,
        # Above-freezing collection → qʷⁱ
        cloud_warm_q, cloud_warm_n, rain_warm_q, rain_warm_n,
        # Wet growth collection → qʷⁱ
        wg_cloud, wg_rain,
        # D8: Wet growth shedding → rain
        wg_shed, wg_shed_n,
        # M9: CCN activation and rain condensation stubs
        zero(FT), zero(FT),
        # D1: Coating condensation/evaporation
        coat_cond, coat_evap
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
    tendency_ρqᶜˡ(rates)

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
    gain = rates.condensation + rates.ccn_activation
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
    tendency_ρqʳ(rates)

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
    gain = rates.autoconversion + rates.accretion + rates.complete_melting +
           rates.shedding + rates.rain_condensation + rates.wet_growth_shedding
    loss = rates.rain_evaporation + rates.rain_riming + rates.rain_freezing_mass +
           rates.rain_homogeneous_mass + rates.rain_warm_collection + rates.wet_growth_rain
    return ρ * (gain - loss)
end

"""
    tendency_ρnʳ(rates, ρ, nⁱ, qⁱ, nʳ, qʳ, prp)

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
    #      collection (Fortran ncshdc)
    # D8: wet_growth_shedding_number → rain drops from excess wet growth (Fortran nrshdr)
    n_gain = n_from_autoconv + n_from_melt +
             rates.rain_breakup +
             rates.shedding_number +
             rates.cloud_warm_collection_number +
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

    return ρ * (n_gain - n_loss)
end

"""
    tendency_ρqⁱ(rates)

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
    # Phase 2: riming (cloud + rain), refreezing, nucleation, freezing, splintering
    # Splintering mass is already part of the riming mass (splinters fragment existing rime),
    # so it is NOT added here. Instead, it is subtracted from rime mass in tendency_ρqᶠ.
    gain = rates.deposition + rates.cloud_riming + rates.rain_riming + rates.refreezing +
           rates.nucleation_mass + rates.cloud_freezing_mass + rates.rain_freezing_mass +
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass
    # Total melting reduces ice mass (partial stays as liquid coating, complete sheds)
    loss = rates.partial_melting + rates.complete_melting
    return ρ * (gain - loss)
end

"""
    tendency_ρnⁱ(rates)

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
    tendency_ρqᶠ(rates)

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
           rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass
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
    tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, qⁱ, prp)

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
                   (rates.cloud_homogeneous_mass + rates.rain_homogeneous_mass) / ρ_rim_hom

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

"""
    tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ)

Compute ice sixth moment tendency from P3 process rates.

The sixth moment (reflectivity) changes with:
- Deposition (growth) (Phase 1)
- Melting (loss) (Phase 1)
- Riming (growth) (Phase 2)
- Nucleation (growth) (Phase 2)
- Aggregation (redistribution) (Phase 2)

This simplified version uses proportional scaling (Z/q ratio).
For more accurate 3-moment treatment, use the version that accepts
the p3 scheme to access tabulated sixth moment integrals.
"""
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, prp::ProcessRateParameters)
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
    z_group2 = initiated_ice_sixth_moment_tendency(rates.nucleation_mass, rates.nucleation_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.cloud_freezing_mass, rates.cloud_freezing_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.rain_freezing_mass, rates.rain_freezing_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.splintering_mass, rates.splintering_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.cloud_homogeneous_mass, rates.cloud_homogeneous_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.rain_homogeneous_mass, rates.rain_homogeneous_number, zero(FT))

    return ρ * (ratio * mass_change + z_group2)
end

@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ)
    FT = typeof(ρ)
    prp = ProcessRateParameters(FT)
    return tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, prp)
end

"""
    tendency_ρzⁱ(rates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3, nu, D_v, μ)

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
@inline function tendency_ρzⁱ(rates::P3ProcessRates, ρ, qⁱ, nⁱ, zⁱ, Fᶠ, Fˡ, ρᶠ, p3, nu, D_v, μ, λ_r = nothing)
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
        p3.process_rates, sc_correction, p3, μ, λ_r
    )

    return z_tendency
end

# Tabulated version: use TabulatedFunction5D lookups for Z tendencies.
#
# Table convention:
# - Single-term processes (rime, aggregation, shedding): table stores dG/mass_integral.
#   Runtime: z_table × mass_rate / Nⁱ (exact).
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
                                        prp::ProcessRateParameters, sc_correction, p3, μ, λ_r = nothing) where {M6 <: IceSixthMoment{<:TabulatedFunction5D}}
    FT = typeof(ρ)
    lt1 = lookup_table_1(p3)
    lt2 = lookup_table_2(p3)
    sixth = lt1.sixth_moment
    dep = lt1.deposition

    inv_nⁱ = safe_divide(one(FT), nⁱ, eps(FT))

    # --- Deposition / Sublimation ---
    # Z tables store raw dG/dt. Extract env factor from mass_rate / (mass_table × Nⁱ).
    # Fortran: epsiz = (m6dep + S_c×m6dep1) × 2πρDv
    #          epsi  = (vdep  + S_c×vdep1 ) × 2πρDv × Nⁱ
    # where S_c = Sc^(1/3) × √ρ_fac / √ν.
    # Deposition (c=2) and sublimation (c=1) use different dG normalization,
    # but the SAME mass integrals (vdep/vdep1). Separate via sign of rates.deposition.
    mass_dep_combined = dep.ventilation(log_m, Fᶠ, Fˡ, ρᶠ, μ) +
                        sc_correction * dep.ventilation_enhanced(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    env_dep = safe_divide(abs(rates.deposition), max(nⁱ * mass_dep_combined, eps(FT)), zero(FT))

    z_dep_combined = sixth.deposition(log_m, Fᶠ, Fˡ, ρᶠ, μ) +
                     sc_correction * sixth.deposition1(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    z_sub_combined = sixth.sublimation(log_m, Fᶠ, Fˡ, ρᶠ, μ) +
                     sc_correction * sixth.sublimation1(log_m, Fᶠ, Fˡ, ρᶠ, μ)

    is_deposition = rates.deposition > zero(FT)
    z_dep_sub_rate = ifelse(is_deposition, z_dep_combined, -z_sub_combined) * env_dep

    env_coat_cond = safe_divide(rates.coating_condensation, max(nⁱ * mass_dep_combined, eps(FT)), zero(FT))
    env_coat_evap = safe_divide(rates.coating_evaporation, max(nⁱ * mass_dep_combined, eps(FT)), zero(FT))
    z_coat_rate = z_dep_combined * env_coat_cond - z_sub_combined * env_coat_evap

    # --- Melting ---
    # Fortran: zimlt = (vdepm1×m6mlt1 + vdepm2×m6mlt2×S_c) × thermo
    # Z is the mass-weighted combination of Z tables. Extract thermo from mass_rate / (mass_combined × Nⁱ).
    mass_melt_const = dep.small_ice_ventilation_constant(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    mass_melt_enh   = dep.small_ice_ventilation_reynolds(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    mass_melt_combined = mass_melt_const + sc_correction * mass_melt_enh
    complete_melting = rates.complete_melting
    env_melt = safe_divide(complete_melting, max(nⁱ * mass_melt_combined, eps(FT)), zero(FT))

    z_melt1 = sixth.melt1(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    z_melt2 = sixth.melt2(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    # Mass-weighted Z: each term multiplied by its own mass table (Fortran convention)
    z_melt_numerator = mass_melt_const * z_melt1 + sc_correction * mass_melt_enh * z_melt2
    z_melt_rate = z_melt_numerator * env_melt

    # --- Riming ---
    z_cloud_rime = sixth.rime(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    z_cloud_rime_rate = z_cloud_rime * rates.cloud_riming * inv_nⁱ
    z_rain_rime_rate = rain_riming_sixth_moment_tendency(lt2, log_m, Fᶠ, Fˡ, ρᶠ, μ, λ_r,
                                                          rates.rain_riming, inv_nⁱ, z_cloud_rime)

    # --- Aggregation (single term): z_table = dG/nagg ---
    z_agg = sixth.aggregation(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    z_agg_rate = z_agg * rates.aggregation * inv_nⁱ

    # --- Shedding (single term): z_table = dG_kernel/M3 ---
    z_shed = sixth.shedding(log_m, Fᶠ, Fˡ, ρᶠ, μ)
    z_shed_rate = z_shed * rates.shedding * inv_nⁱ

    cloud_spl_q, rain_spl_q = split_splintering_mass(rates)
    μ_c = liu_daum_shape_parameter(p3.cloud.number_concentration)
    z_group2 = initiated_ice_sixth_moment_tendency(rates.nucleation_mass, rates.nucleation_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.cloud_freezing_mass, rates.cloud_freezing_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.rain_freezing_mass, rates.rain_freezing_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rain_spl_q, rates.splintering_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(cloud_spl_q, rates.splintering_number, zero(FT)) +
               initiated_ice_sixth_moment_tendency(rates.cloud_homogeneous_mass, rates.cloud_homogeneous_number, μ_c) +
               initiated_ice_sixth_moment_tendency(rates.rain_homogeneous_mass, rates.rain_homogeneous_number, zero(FT))

    # Total Z rate
    z_rate = z_dep_sub_rate +
             z_coat_rate +
             z_cloud_rime_rate +
             z_rain_rime_rate +
             z_agg_rate -
             z_shed_rate -
             z_melt_rate
    z_rate = z_rate + z_group2

    return ρ * z_rate
end

@inline function rain_riming_sixth_moment_tendency(::Nothing, log_m, Fᶠ, Fˡ, ρᶠ, μ, λ_r,
                                                   rain_riming, inv_nⁱ, z_cloud_rime)
    return z_cloud_rime * rain_riming * inv_nⁱ
end

@inline function rain_riming_sixth_moment_tendency(table2::P3LookupTable2, log_m, Fᶠ, Fˡ, ρᶠ, μ,
                                                   ::Nothing, rain_riming, inv_nⁱ, z_cloud_rime)
    return z_cloud_rime * rain_riming * inv_nⁱ
end

@inline function rain_riming_sixth_moment_tendency(table2::P3LookupTable2, log_m, Fᶠ, Fˡ, ρᶠ, μ,
                                                   λ_r, rain_riming, inv_nⁱ, z_cloud_rime)
    FT = typeof(log_m)
    log_λ_r = log10(max(FT(λ_r), FT(1e-20)))
    z_rain_rime = table2.sixth_moment(log_m, log_λ_r, Fᶠ, Fˡ, ρᶠ, μ)
    return z_rain_rime * rain_riming * inv_nⁱ
end

@inline function split_splintering_mass(rates::P3ProcessRates)
    FT = typeof(rates.splintering_mass)
    total_riming = clamp_positive(rates.cloud_riming) + clamp_positive(rates.rain_riming)
    cloud_fraction = safe_divide(clamp_positive(rates.cloud_riming), total_riming, zero(FT))
    rain_fraction = safe_divide(clamp_positive(rates.rain_riming), total_riming, zero(FT))
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
    tendency_ρqʷⁱ(rates)

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
    tendency_ρqᵛ(rates)

Compute vapor mass tendency from P3 process rates.

Vapor is consumed by:
- Condensation (vapor → cloud liquid)
- Deposition (vapor → ice)
- Deposition nucleation (vapor → ice)

Vapor is produced by:
- Cloud evaporation (negative condensation)
- Rain evaporation
- Sublimation (negative deposition)
"""
@inline function tendency_ρqᵛ(rates::P3ProcessRates, ρ)
    # Condensation: positive = vapor loss (cond), negative = vapor gain (cloud evap)
    # Deposition:   positive = vapor loss (dep),  negative = vapor gain (sublimation)
    # Rain evaporation: positive magnitude (M7) = vapor gain
    # Nucleation: always positive = vapor loss
    # M9: CCN activation, rain condensation, and coating condensation are all vapor sinks;
    #      coating evaporation is a vapor source.
    vapor_loss = rates.condensation + rates.deposition + rates.nucleation_mass +
                 rates.ccn_activation + rates.rain_condensation + rates.coating_condensation
    vapor_gain = rates.rain_evaporation + rates.coating_evaporation
    return ρ * (vapor_gain - vapor_loss)
end

#####
##### Fallback methods for Nothing rates
#####
##### These are safety fallbacks that return zero tendency when rates
##### have not been computed (e.g., during incremental development).
#####

@inline tendency_ρqᶜˡ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqʳ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnʳ(::Nothing, ρ, nⁱ, qⁱ, args...) = zero(ρ)
@inline tendency_ρqⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρnⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᶠ(::Nothing, ρ, Fᶠ) = zero(ρ)
@inline tendency_ρbᶠ(::Nothing, ρ, Fᶠ, ρᶠ, prp...) = zero(ρ)
@inline tendency_ρzⁱ(::Nothing, ρ, qⁱ, nⁱ, zⁱ) = zero(ρ)
@inline tendency_ρqʷⁱ(::Nothing, ρ) = zero(ρ)
@inline tendency_ρqᵛ(::Nothing, ρ) = zero(ρ)
