#####
##### Phase 3: Terminal velocities
#####
##### Terminal velocity calculations for rain and ice sedimentation.
##### Rain uses the P3 piecewise Gunn-Kinzer/Beard law with air density correction.
#####

"""
    rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)

Compute mass-weighted terminal velocity for rain.

Dispatches on `p3.rain.velocity_mass`:

- **Tabulated** (`TabulatedFunction1D`): Looks up the PSD-integrated velocity
  at `log10(λ_r)` and applies the air density correction `(ρ₀/ρ)^0.54`.
- **Mean-mass** (`RainVelocityMass`): Uses the same piecewise rain fall-speed
    law evaluated at the volume-mean drop diameter.

See [Seifert and Beheng (2006)](@cite SeifertBeheng2006).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Mass-weighted fall speed [m/s] (positive downward)
"""
@inline function rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density
    v_min = prp.rain_velocity_min
    v_max = prp.rain_velocity_max

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1e-16))

    # Density correction factor (Foote & du Toit 1969; Fortran P3 uses 0.54)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    vₜ = tabulated_rain_mass_weighted_velocity(p3.rain.velocity_mass,
                                                 qʳ_eff, nʳ_eff, ρ_correction,
                                                 ρʷ, prp, FT)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated path: look up PSD-integrated mass-weighted velocity
@inline function tabulated_rain_mass_weighted_velocity(table::TabulatedFunction1D,
                                                         qʳ, nʳ, ρ_correction, ρʷ, prp, FT)
    m̄  = qʳ / nʳ
    λ_r = cbrt(FT(π) * ρʷ / (6 * max(m̄, FT(1e-15))))
    # H6: Clamp λ_r to Fortran P3 bounds (prevents unphysical lookup)
    λ_r = clamp(λ_r, prp.rain_lambda_min, prp.rain_lambda_max)
    log_λ = log10(λ_r)
    vₜ_ref = table(log_λ)
    return vₜ_ref * ρ_correction
end

# Mean-mass fallback path
# Uses the same 4-regime Gunn-Kinzer/Beard piecewise formula as the tabulated
# path (rain_fall_speed in quadrature.jl), ensuring consistent V(D) across both
# code paths. Previously used the single power law V = ar × D^br.
@inline function tabulated_rain_mass_weighted_velocity(::AbstractRainIntegral,
                                                         qʳ, nʳ, ρ_correction, ρʷ, prp, FT)
    D_min = prp.rain_diameter_min
    D_max = prp.rain_diameter_max

    m̄ = qʳ / nʳ
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * ρʷ))
    D̄ₘ_clamped = clamp(D̄ₘ, D_min, D_max)
    return rain_fall_speed(D̄ₘ_clamped, ρ_correction)
end

"""
    rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)

Compute number-weighted terminal velocity for rain.

Dispatches on `p3.rain.velocity_number`:

- **Tabulated** (`TabulatedFunction1D`): Looks up the PSD-integrated number-
  weighted velocity at `log10(λ_r)` with air density correction.
- **Mean-mass** (`RainVelocityNumber`): Uses a fixed ratio to the mass-weighted
  velocity.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
@inline function rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density
    v_min = prp.rain_velocity_min
    v_max = prp.rain_velocity_max

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1e-16))

    # Density correction factor (Foote & du Toit 1969; Fortran P3 uses 0.54)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    vₜ = tabulated_rain_number_weighted_velocity(p3.rain.velocity_number,
                                                   p3.rain.velocity_mass,
                                                   qʳ_eff, nʳ_eff, ρ_correction,
                                                   ρʷ, prp, FT)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated path: look up PSD-integrated number-weighted velocity
@inline function tabulated_rain_number_weighted_velocity(table::TabulatedFunction1D,
                                                           ::AbstractRainIntegral,
                                                           qʳ, nʳ, ρ_correction, ρʷ, prp, FT)
    m̄  = qʳ / nʳ
    λ_r = cbrt(FT(π) * ρʷ / (6 * max(m̄, FT(1e-15))))
    # H6: Clamp λ_r to Fortran P3 bounds
    λ_r = clamp(λ_r, prp.rain_lambda_min, prp.rain_lambda_max)
    log_λ = log10(λ_r)
    vₜ_ref = table(log_λ)
    return vₜ_ref * ρ_correction
end

# Mean-mass fallback: use fixed ratio to mass-weighted velocity
@inline function tabulated_rain_number_weighted_velocity(::AbstractRainIntegral,
                                                           vel_mass_field,
                                                           qʳ, nʳ, ρ_correction, ρʷ, prp, FT)
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = tabulated_rain_mass_weighted_velocity(vel_mass_field, qʳ, nʳ,
                                                  ρ_correction, ρʷ, prp, FT)
    return ratio * vₘ
end

"""
    ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))

Compute mass-weighted terminal velocity for ice.

When tabulated integrals are available (via `tabulate(p3, arch)`), uses
pre-computed lookup tables for accurate size-distribution integration.
Otherwise, uses regime-dependent fall speeds following [Mitchell (1996)](@cite Mitchell1996powerlaws)
and [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `Fˡ`: Liquid fraction (optional, for tabulated lookup)

# Returns
- Mass-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    # Density correction factor (Heymsfield et al. 2006)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = tabulated_mass_weighted_fall_speed(fs.mass_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction4D lookup (includes rime density axis)
@inline function tabulated_mass_weighted_fall_speed(table::TabulatedFunction4D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    log_mean_mass = log10(max(m̄, p3.minimum_mass_mixing_ratio))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ)
    return vₜ_norm * ρ_correction
end

# Fallback: use analytical approximation when not tabulated
@inline function tabulated_mass_weighted_fall_speed(::AbstractFallSpeedIntegral, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)

    ρ_eff_unrimed = prp.ice_effective_density_unrimed
    D_threshold = prp.ice_diameter_threshold
    D_min = prp.ice_diameter_min
    D_max = prp.ice_diameter_max
    ρᶠ_min = prp.minimum_rime_density
    ρᶠ_max = prp.maximum_rime_density

    a_unrimed = prp.ice_fall_speed_coefficient_unrimed
    b_unrimed = prp.ice_fall_speed_exponent_unrimed
    a_rimed = prp.ice_fall_speed_coefficient_rimed
    b_rimed = prp.ice_fall_speed_exponent_rimed
    c_small = prp.ice_small_particle_coefficient

    # Effective density depends on riming
    Fᶠ_clamped = clamp(Fᶠ, FT(0), FT(1))
    ρᶠ_clamped = clamp(ρᶠ, ρᶠ_min, ρᶠ_max)
    ρ_eff = ρ_eff_unrimed + Fᶠ_clamped * (ρᶠ_clamped - ρ_eff_unrimed)

    # Effective diameter
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * ρ_eff))
    D_clamped = clamp(D̄ₘ, D_min, D_max)

    # Coefficients interpolated based on riming
    a = a_unrimed + Fᶠ_clamped * (a_rimed - a_unrimed)
    b = b_unrimed + Fᶠ_clamped * (b_rimed - b_unrimed)

    # Terminal velocity (large particle regime)
    vₜ_large = a * D_clamped^b * ρ_correction

    # Small particle (Stokes) regime
    vₜ_small = c_small * D_clamped^2 * ρ_correction

    # Mass-weighted PSD correction (analytical fallback only — the tabulated
    # path already returns PSD-integrated values). For an inverse exponential
    # PSD (μ=0) with m ~ D³, the mass-weighted velocity exceeds the velocity
    # at the mean-mass diameter by Γ(4+b)/Γ(4), where b is the velocity-diameter
    # exponent in V = a × D^b.
    # NOTE: This uses the Γ(4+b)/Γ(4) convention (D = 1/λ reference), consistent
    # with Fortran P3 v5.5.0. D̄ₘ in this code is actually 6^(1/3)/λ, introducing
    # a 6^(b/3) factor (~1.29 for b=0.4, ~3.3 for b=2). Both regimes use the
    # same approximation for internal consistency.
    # Large particles (b ≈ 0.44): Γ(4.44)/Γ(4) ≈ 1.787 (Fortran P3 convention)
    # Stokes regime (b = 2):      Γ(6)/Γ(4) = 5 × 4 = 20
    mass_weight_factor_large = FT(1.787)
    mass_weight_factor_small = FT(20)

    # Blend between regimes (apply regime-dependent PSD correction)
    vₜ = ifelse(D_clamped < D_threshold,
                vₜ_small * mass_weight_factor_small,
                vₜ_large * mass_weight_factor_large)
end

"""
    ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ)

Compute number-weighted terminal velocity for ice.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = tabulated_number_weighted_fall_speed(fs.number_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction4D lookup (includes rime density axis)
@inline function tabulated_number_weighted_fall_speed(table::TabulatedFunction4D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    log_mean_mass = log10(max(m̄, p3.minimum_mass_mixing_ratio))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ)
    return vₜ_norm * ρ_correction
end

# Fallback: use ratio to mass-weighted velocity
@inline function tabulated_number_weighted_fall_speed(::AbstractFallSpeedIntegral, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = tabulated_mass_weighted_fall_speed(MassWeightedFallSpeed(), m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    return ratio * vₘ
end

"""
    ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=0)

Compute reflectivity-weighted (Z-weighted) terminal velocity for ice.

Needed for the sixth moment (reflectivity) sedimentation in 3-moment P3.
When tabulated integrals are available, uses pre-computed lookup tables.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `Fˡ`: Liquid fraction (optional, for tabulated lookup)

# Returns
- Reflectivity-weighted fall speed [m/s] (positive downward)
"""
@inline function ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = tabulated_reflectivity_weighted_fall_speed(fs.reflectivity_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction4D lookup (includes rime density axis)
@inline function tabulated_reflectivity_weighted_fall_speed(table::TabulatedFunction4D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    log_mean_mass = log10(max(m̄, p3.minimum_mass_mixing_ratio))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ)
    return vₜ_norm * ρ_correction
end

# Fallback: use ratio to mass-weighted velocity
@inline function tabulated_reflectivity_weighted_fall_speed(::AbstractFallSpeedIntegral, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    ratio = prp.velocity_ratio_reflectivity_to_mass
    vₘ = tabulated_mass_weighted_fall_speed(MassWeightedFallSpeed(), m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    return ratio * vₘ
end

"""
    ice_terminal_velocities(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))

Compute all three ice terminal velocities (mass-, number-, and reflectivity-weighted)
in a single call, sharing the mean particle mass and air density correction computation.

This is a performance convenience wrapper over the individual
`ice_terminal_velocity_mass_weighted`, `ice_terminal_velocity_number_weighted`, and
`ice_terminal_velocity_reflectivity_weighted` functions. The individual functions
remain available for cases where only one velocity is needed.

See [Heymsfield et al. (2006)](@cite HeymsfieldEtAl2006) for the density correction exponent
and [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) for the P3 fall
speed framework.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters and lookup tables)
- `qⁱ`: Ice mass fraction [kg/kg]
- `nⁱ`: Ice number concentration [1/kg]
- `Fᶠ`: Rime mass fraction (qᶠ/qⁱ)
- `ρᶠ`: Rime density [kg/m³]
- `ρ`: Air density [kg/m³]
- `Fˡ`: Liquid fraction (optional, for tabulated lookup)

# Returns
- `NamedTuple` with fields `mass_weighted`, `number_weighted`, `reflectivity_weighted` [m/s]
  (all positive downward)
"""
@inline function ice_terminal_velocities(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density
    v_min = prp.ice_velocity_min
    v_max = prp.ice_velocity_max

    # --- Shared computation (done once instead of three times) ---
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m̄ = qⁱ_eff / nⁱ_eff

    # Density correction factor (Heymsfield et al. 2006, exponent 0.54 for ice)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # --- Dispatch into the tabulated / analytical internals ---
    vₜ_mass = tabulated_mass_weighted_fall_speed(
        fs.mass_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    vₜ_number = tabulated_number_weighted_fall_speed(
        fs.number_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    vₜ_refl = tabulated_reflectivity_weighted_fall_speed(
        fs.reflectivity_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return (mass_weighted        = clamp(vₜ_mass,   v_min, v_max),
            number_weighted      = clamp(vₜ_number, v_min, v_max),
            reflectivity_weighted = clamp(vₜ_refl,   v_min, v_max))
end
