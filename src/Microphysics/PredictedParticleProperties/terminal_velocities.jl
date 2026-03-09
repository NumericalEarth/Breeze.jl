#####
##### Phase 3: Terminal velocities
#####
##### Terminal velocity calculations for rain and ice sedimentation.
##### Uses power-law relationships with air density correction.
#####

"""
    rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)

Compute mass-weighted terminal velocity for rain.

Uses the power-law relationship v(D) = a × D^b × √(ρ₀/ρ).
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

    a = prp.rain_fall_speed_coefficient
    b = prp.rain_fall_speed_exponent
    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density
    D_min = prp.rain_diameter_min
    D_max = prp.rain_diameter_max
    v_min = prp.rain_velocity_min
    v_max = prp.rain_velocity_max

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1))

    # Mean rain drop mass
    m̄ = qʳ_eff / nʳ_eff

    # Mass-weighted mean diameter: m = (π/6) ρʷ D³
    D̄ₘ = cbrt(6 * m̄ / (FT(π) * ρʷ))

    # Density correction factor (Heymsfield et al. 2006)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Clamp diameter to physical range
    D̄ₘ_clamped = clamp(D̄ₘ, D_min, D_max)

    # Terminal velocity
    vₜ = a * D̄ₘ_clamped^b * ρ_correction

    return clamp(vₜ, v_min, v_max)
end

"""
    rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)

Compute number-weighted terminal velocity for rain.

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

    # Number-weighted velocity is smaller than mass-weighted
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)

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
    nⁱ_eff = max(nⁱ, FT(1))

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    # Density correction factor (Heymsfield et al. 2006)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = _tabulated_mass_weighted_fall_speed(fs.mass_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction3D lookup
@inline function _tabulated_mass_weighted_fall_speed(table::TabulatedFunction3D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)
    # Compute log mean mass (guarding against log(0))
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    # Look up normalized velocity from table
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ)
    return vₜ_norm * ρ_correction
end

# Fallback: use analytical approximation when not tabulated
@inline function _tabulated_mass_weighted_fall_speed(::Any, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
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
    # PSD (μ=0), the mass-weighted velocity is Γ(4+b)/(Γ(4)×λ^(-b)) ≈ 1.9×
    # the single-particle velocity at D_mean. Correction = Γ(4+b)/(6×1.817^b).
    mass_weight_factor = FT(1.9)

    # Blend between regimes
    vₜ = ifelse(D_clamped < D_threshold, vₜ_small, vₜ_large)
    return vₜ * mass_weight_factor
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
    nⁱ_eff = max(nⁱ, FT(1))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = _tabulated_number_weighted_fall_speed(fs.number_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction3D lookup
@inline function _tabulated_number_weighted_fall_speed(table::TabulatedFunction3D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ)
    return vₜ_norm * ρ_correction
end

# Fallback: use ratio to mass-weighted velocity
@inline function _tabulated_number_weighted_fall_speed(::Any, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    ratio = prp.velocity_ratio_number_to_mass
    vₘ = _tabulated_mass_weighted_fall_speed(nothing, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
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
    nⁱ_eff = max(nⁱ, FT(1))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # Try to use tabulated fall speed if available
    vₜ = _tabulated_reflectivity_weighted_fall_speed(fs.reflectivity_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)

    return clamp(vₜ, v_min, v_max)
end

# Tabulated version: use TabulatedFunction3D lookup
@inline function _tabulated_reflectivity_weighted_fall_speed(table::TabulatedFunction3D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    FT = typeof(m̄)
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ)
    return vₜ_norm * ρ_correction
end

# Fallback: use ratio to mass-weighted velocity
@inline function _tabulated_reflectivity_weighted_fall_speed(::Any, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    ratio = prp.velocity_ratio_reflectivity_to_mass
    vₘ = _tabulated_mass_weighted_fall_speed(nothing, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp)
    return ratio * vₘ
end
