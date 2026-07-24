#####
##### Phase 3: Terminal velocities
#####
##### Terminal velocity calculations for rain and ice sedimentation.
##### Rain uses the P3 piecewise Gunn-Kinzer/Beard law with air density correction.
#####

"""
$(TYPEDSIGNATURES)

Compute mass-weighted terminal velocity for rain.

Looks up the PSD-integrated velocity from a tabulated `TabulatedFunction1D`
at `log10(λ_r)` and applies the air density correction `(ρ₀/ρ)^0.54`.

See [Seifert and Beheng (2006)](@cite SeifertBeheng2006).

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Mass-weighted fall speed [m/s] (positive downward)
"""
function rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1e-16))

    # Density correction factor (Foote & du Toit 1969; Fortran P3 uses 0.54)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # m9: Fortran applies no velocity clamping; table bounds are sufficient.
    velocity = tabulated_rain_mass_weighted_velocity(p3.rain.velocity_mass,
                                                       qʳ_eff, nʳ_eff, ρ_correction,
                                                       ρʷ, prp, FT)
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, velocity, zero(FT))
end

# Tabulated path: look up PSD-integrated mass-weighted velocity
@inline function tabulated_rain_mass_weighted_velocity(table::TabulatedFunction1D,
                                                         qʳ, nʳ, ρ_correction, ρʷ, prp, FT)
    m̄  = qʳ / nʳ
    # For exponential PSD (μ_r=0): <m> = π ρ_w / λ³, so λ = (π ρ_w / m̄)^(1/3)
    λ_r = cbrt(FT(π) * ρʷ / max(m̄, FT(1e-15)))
    # Clamp λ_r to Fortran P3 bounds (prevents unphysical lookup).
    # m10: Fortran get_rain_dsd2 also recomputes nr when λ is clamped;
    # that adjustment is done in compute_p3_process_rates (H4) and does not
    # affect the velocity lookup which depends only on λ_r.
    λ_r = clamp(λ_r, prp.rain_lambda_min, prp.rain_lambda_max)
    log_λ = log10(λ_r)
    vₜ_ref = table(log_λ)
    return vₜ_ref * ρ_correction
end

"""
$(TYPEDSIGNATURES)

Compute number-weighted terminal velocity for rain.

Looks up the PSD-integrated number-weighted velocity from a tabulated
`TabulatedFunction1D` at `log10(λ_r)` with air density correction.

# Arguments
- `p3`: P3 microphysics scheme (provides parameters)
- `qʳ`: Rain mass fraction [kg/kg]
- `nʳ`: Rain number concentration [1/kg]
- `ρ`: Air density [kg/m³]

# Returns
- Number-weighted fall speed [m/s] (positive downward)
"""
function rain_terminal_velocity_number_weighted(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates

    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1e-16))

    # Density correction factor (Foote & du Toit 1969; Fortran P3 uses 0.54)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # m9: Fortran applies no velocity clamping; table bounds are sufficient.
    velocity = tabulated_rain_number_weighted_velocity(p3.rain.velocity_number,
                                                         p3.rain.velocity_mass,
                                                         qʳ_eff, nʳ_eff, ρ_correction,
                                                         ρʷ, prp, FT)
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, velocity, zero(FT))
end

# Tabulated path: look up PSD-integrated number-weighted velocity
@inline function tabulated_rain_number_weighted_velocity(table::TabulatedFunction1D,
                                                           _velocity_mass,
                                                           qʳ, nʳ, ρ_correction, ρʷ, prp, FT)
    m̄  = qʳ / nʳ
    # For exponential PSD (μ_r=0): <m> = π ρ_w / λ³, so λ = (π ρ_w / m̄)^(1/3)
    λ_r = cbrt(FT(π) * ρʷ / max(m̄, FT(1e-15)))
    # Clamp λ_r to Fortran P3 bounds.
    # m10: nr adjustment handled by H4 in compute_p3_process_rates.
    λ_r = clamp(λ_r, prp.rain_lambda_min, prp.rain_lambda_max)
    log_λ = log10(λ_r)
    vₜ_ref = table(log_λ)
    return vₜ_ref * ρ_correction
end

# GPU-safe concrete struct (NamedTuple complicates the GPU compiler's NoInline boundaries).
struct RainTerminalVelocities{FT}
    mass_weighted :: FT
    number_weighted :: FT
end

struct CloudTerminalVelocities{FT}
    mass_weighted :: FT
    number_weighted :: FT
end

# `μ_c` and `λ_c` are the cloud-DSD shape/slope diagnosed by `diagnose_cloud_dsd`;
# the caller passes the values already computed in `p3_ice_properties`
# (`props.μ_cloud`/`props.λ_cloud`) so the fall-speed kernel does not re-diagnose them.
@inline function cloud_terminal_velocities(p3, qᶜˡ, ρ, ν, μ_c, λ_c)
    FT = typeof(qᶜˡ + ρ + ν + μ_c + λ_c)
    μ_air = ν * ρ
    a_cn = FT(9.81) * p3.process_rates.liquid_water_density /
           (FT(18) * max(μ_air, FT(1e-20)))
    inverse_λ_squared = inv(λ_c^2)
    active = qᶜˡ >= p3.minimum_mass_mixing_ratio
    mass_weighted = a_cn * (μ_c + 5) * (μ_c + 4) * inverse_λ_squared
    number_weighted = a_cn * (μ_c + 2) * (μ_c + 1) * inverse_λ_squared
    return CloudTerminalVelocities{FT}(ifelse(active, mass_weighted, zero(FT)),
                                       ifelse(active, number_weighted, zero(FT)))
end

"""
$(TYPEDSIGNATURES)

Compute mass- and number-weighted rain terminal velocities together, sharing the
slope-parameter, ρ-correction, and `log10(λ_r)` computations between the two
table lookups. The result is equivalent to calling `rain_terminal_velocity_mass_weighted`
and `rain_terminal_velocity_number_weighted` separately.

# Returns
- `RainTerminalVelocities` with fields `mass_weighted`, `number_weighted` [m/s] (positive downward)
"""
@inline function rain_terminal_velocities(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates
    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, FT(1e-16))
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    m̄  = qʳ_eff / nʳ_eff
    λ_r = cbrt(FT(π) * ρʷ / max(m̄, FT(1e-15)))
    λ_r = clamp(λ_r, prp.rain_lambda_min, prp.rain_lambda_max)
    log_λ = log10(λ_r)

    vₘ = p3.rain.velocity_mass(log_λ) * ρ_correction
    vₙ = p3.rain.velocity_number(log_λ) * ρ_correction
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    return RainTerminalVelocities{FT}(ifelse(active, vₘ, zero(FT)),
                                       ifelse(active, vₙ, zero(FT)))
end

"""
$(TYPEDSIGNATURES)

Compute mass-weighted terminal velocity for ice.

Uses pre-computed lookup tables for accurate size-distribution integration.
See [Mitchell (1996)](@cite Mitchell1996powerlaws)
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
@inline function ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)), μ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    # Density correction factor (Heymsfield et al. 2007)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # m9: Fortran applies no velocity clamping; table bounds are sufficient.
    velocity = tabulated_mass_weighted_fall_speed(fs.mass_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, velocity, zero(FT))
end

# Tabulated version: use TabulatedFunction5D lookup (includes rime density and mu axes)
@inline function tabulated_mass_weighted_fall_speed(table::P3Table5D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    # m̄ = qⁱ/nⁱ is a per-particle mass [kg]; floor it only with a tiny log-guard,
    # NOT the bulk mass-mixing-ratio threshold `minimum_mass_mixing_ratio` (kg/kg).
    # The table clamps the coordinate to its mass axis (min ≈ 1.56e-15 kg), matching
    # Fortran's clamp of the lookup index to 1 (find_lookupTable_indices_1a).
    log_mean_mass = log10(max(m̄, oftype(m̄, 1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return vₜ_norm * ρ_correction
end

# Prepared-index variant: reuse precomputed interpolation indices and skip the log/clamp setup.
@inline function tabulated_mass_weighted_fall_speed(table::P3Table5D,
                                                    prep::Prepared5DInterpolation, ρ_correction)
    return evaluate_at(table, prep) * ρ_correction
end

"""
$(TYPEDSIGNATURES)

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
@inline function ice_terminal_velocity_number_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)), μ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # m9: Fortran applies no velocity clamping; table bounds are sufficient.
    velocity = tabulated_number_weighted_fall_speed(fs.number_weighted, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, velocity, zero(FT))
end

# Tabulated version: use TabulatedFunction5D lookup (includes rime density and mu axes)
@inline function tabulated_number_weighted_fall_speed(table::P3Table5D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_mass_weighted_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, oftype(m̄, 1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return vₜ_norm * ρ_correction
end

@inline function tabulated_number_weighted_fall_speed(table::P3Table5D,
                                                      prep::Prepared5DInterpolation, ρ_correction)
    return evaluate_at(table, prep) * ρ_correction
end

"""
$(TYPEDSIGNATURES)

Compute reflectivity-weighted (Z-weighted) terminal velocity for ice.

Needed for the sixth moment (reflectivity) sedimentation in 3-moment P3.
Uses pre-computed lookup tables for accurate PSD integration.

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
@inline function ice_terminal_velocity_reflectivity_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)), μ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m̄ = qⁱ_eff / nⁱ_eff
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    # m9: Fortran applies no velocity clamping; table bounds are sufficient.
    velocity = tabulated_reflectivity_weighted_fall_speed(fs.reflectivity_weighted,
                                                            m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction,
                                                            p3, prp, μ)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, velocity, zero(FT))
end

# Tabulated version: use TabulatedFunction5D lookup (includes rime density and mu axes)
@inline function tabulated_reflectivity_weighted_fall_speed(table::P3Table5D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_mass_weighted_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, oftype(m̄, 1e-20)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return vₜ_norm * ρ_correction
end

@inline function tabulated_reflectivity_weighted_fall_speed(table::P3Table5D,
                                                            prep::Prepared5DInterpolation, ρ_correction)
    return evaluate_at(table, prep) * ρ_correction
end

# 2-moment ice closure: reflectivity table is `nothing` because Z_i is not a
# prognostic moment. Return zero so callers (e.g. `ρz̃ⁱ` advection velocity)
# stay well-defined without forcing them to branch on the closure mode.
@inline tabulated_reflectivity_weighted_fall_speed(::Nothing, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ) =
    zero(typeof(m̄))
@inline tabulated_reflectivity_weighted_fall_speed(::Nothing, prep::Prepared5DInterpolation, ρ_correction) =
    zero(typeof(ρ_correction))

"""
$(TYPEDSIGNATURES)

Compute all three ice terminal velocities (mass-, number-, and reflectivity-weighted)
in a single call, sharing the mean particle mass and air density correction computation.

This is a performance convenience wrapper over the individual
`ice_terminal_velocity_mass_weighted`, `ice_terminal_velocity_number_weighted`, and
`ice_terminal_velocity_reflectivity_weighted` functions. The individual functions
remain available for cases where only one velocity is needed.

See [Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007) for the density correction exponent
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
# GPU-safe concrete struct (NamedTuple complicates the GPU compiler's NoInline boundaries).
struct IceTerminalVelocities{FT}
    mass_weighted :: FT
    number_weighted :: FT
    reflectivity_weighted :: FT
end

function ice_terminal_velocities(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)), μ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    # --- Shared computation (done once instead of three times) ---
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, FT(1e-16))
    m̄ = qⁱ_eff / nⁱ_eff

    # Density correction factor (Heymsfield et al. 2007, exponent 0.54 for ice)
    ρ_correction = (ρ₀ / ρ)^FT(0.54)

    velocities = fused_fall_speeds(fs.mass_weighted, fs.number_weighted, fs.reflectivity_weighted,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return IceTerminalVelocities{FT}(
        ifelse(active, velocities.mass_weighted, zero(FT)),
        ifelse(active, velocities.number_weighted, zero(FT)),
        ifelse(active, velocities.reflectivity_weighted, zero(FT)),
    )
end

# Fast path: all three tables are 5D (the supported P3 configuration with loaded tables).
# Interpolation indices for (log_m, Fᶠ, Fˡ, ρᶠ, μ) are shared across the three reads.
@inline function fused_fall_speeds(mass_table::P3Table5D, number_table::P3Table5D, refl_table::P3Table5D,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    FT = typeof(m̄)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_mass_weighted_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    prep = prepare_5d(mass_table, log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return IceTerminalVelocities{FT}(
        tabulated_mass_weighted_fall_speed(mass_table, prep, ρ_correction),
        tabulated_number_weighted_fall_speed(number_table, prep, ρ_correction),
        tabulated_reflectivity_weighted_fall_speed(refl_table, prep, ρ_correction),
    )
end

# 2-moment fast path: reflectivity table is unused, so skip its lookup entirely
# but still share the (log_m, Fᶠ, Fˡ, ρᶠ, μ) prep across the mass/number tables.
@inline function fused_fall_speeds(mass_table::P3Table5D, number_table::P3Table5D, refl_table::Nothing,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    FT = typeof(m̄)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_mass_weighted_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, FT(1e-20)))
    prep = prepare_5d(mass_table, log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return IceTerminalVelocities{FT}(
        tabulated_mass_weighted_fall_speed(mass_table, prep, ρ_correction),
        tabulated_number_weighted_fall_speed(number_table, prep, ρ_correction),
        zero(FT),
    )
end

# Fallback for non-5D fall speed tables (quadrature path, mixed types).
@inline function fused_fall_speeds(mass_table, number_table, refl_table,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    FT = typeof(m̄)
    return IceTerminalVelocities{FT}(
        tabulated_mass_weighted_fall_speed(mass_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ),
        tabulated_number_weighted_fall_speed(number_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ),
        tabulated_reflectivity_weighted_fall_speed(refl_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ),
    )
end
