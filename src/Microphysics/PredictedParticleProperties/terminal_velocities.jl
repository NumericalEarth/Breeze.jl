#####
##### Phase 3: Terminal velocities
#####
##### Terminal velocity calculations for rain and ice sedimentation.
##### Rain uses the P3 piecewise Gunn-Kinzer/Beard law with air density correction.
#####

# GPU-safe concrete struct (NamedTuple complicates the GPU compiler's NoInline boundaries).
struct RainTerminalVelocities{FT}
    mass_weighted :: FT
    number_weighted :: FT
end

struct CloudTerminalVelocities{FT}
    mass_weighted :: FT
    number_weighted :: FT
end

# Stokes-regime cloud-droplet fall speed, `v(D) = a_cn D²`. Fortran `get_cloud_dsd2`
# sets `acn = g ρʷ / (18 η)` with `bcn = 2`, so the PSD-weighted moments follow from
# Γ(μ+b+4)/Γ(μ+4) = (μ+5)(μ+4) for mass and Γ(μ+b+1)/Γ(μ+1) = (μ+2)(μ+1) for number.
# `rime_density` needs the same mass-weighted speed to form the Cober-List rime-impact
# parameter, so both call these helpers rather than repeating the arithmetic with their
# own gravitational acceleration.
@inline cloud_stokes_prefactor(g, ρᴸ, η) =
    g * ρᴸ / (18 * max(η, oftype(η, DEFAULT_FLOORS.divisor)))

@inline cloud_mass_weighted_stokes_velocity(a_cn, μ_c, λ_c) =
    a_cn * (μ_c + 5) * (μ_c + 4) / λ_c^2

@inline cloud_number_weighted_stokes_velocity(a_cn, μ_c, λ_c) =
    a_cn * (μ_c + 2) * (μ_c + 1) / λ_c^2

# `μ_c` and `λ_c` are the cloud-DSD shape/slope diagnosed by `diagnose_cloud_dsd`;
# the caller passes the values already computed in `p3_ice_properties`
# (`props.μ_cloud`/`props.λ_cloud`) so the fall-speed kernel does not re-diagnose them.
@inline function cloud_terminal_velocities(p3, qᶜˡ, ρ, ν, μ_c, λ_c, constants)
    FT = typeof(qᶜˡ + ρ + ν + μ_c + λ_c)
    η = ν * ρ
    g = p3_gravitational_acceleration(constants, FT)
    a_cn = cloud_stokes_prefactor(g, p3.process_rates.liquid_water_density, η)
    active = qᶜˡ >= p3.minimum_mass_mixing_ratio
    mass_weighted = cloud_mass_weighted_stokes_velocity(a_cn, μ_c, λ_c)
    number_weighted = cloud_number_weighted_stokes_velocity(a_cn, μ_c, λ_c)
    return CloudTerminalVelocities{FT}(ifelse(active, mass_weighted, zero(FT)),
                                       ifelse(active, number_weighted, zero(FT)))
end

"""
$(TYPEDSIGNATURES)

Compute mass- and number-weighted rain terminal velocities together, sharing the
slope-parameter, ρ-correction, and `log10(λ_r)` computations between the two
table lookups.

# Returns
- `RainTerminalVelocities` with fields `mass_weighted`, `number_weighted` [m/s] (positive downward)
"""
@inline function rain_terminal_velocities(p3, qʳ, nʳ, ρ)
    FT = typeof(qʳ)
    prp = p3.process_rates
    ρ₀ = prp.reference_air_density
    ρʷ = prp.liquid_water_density

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, p3.minimum_number_mixing_ratio)
    ρ_correction = ice_air_density_correction(ρ₀, ρ)

    m̄  = qʳ_eff / nʳ_eff
    λ_r = cbrt(FT(π) * ρʷ / max(m̄, FT(prp.floors.mass_scale)))
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
    nⁱ_eff = max(nⁱ, p3.minimum_number_mixing_ratio)

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    ρ_correction = ice_air_density_correction(ρ₀, ρ)

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
    log_mean_mass = log10(max(m̄, oftype(m̄, prp.floors.mass_scale)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return vₜ_norm * ρ_correction
end

# Prepared-index variant: reuse precomputed interpolation indices and skip the log/clamp setup.
@inline function tabulated_mass_weighted_fall_speed(table::P3Table5D,
                                                    prep::Prepared5DInterpolation, ρ_correction)
    return evaluate_at(table, prep) * ρ_correction
end

# Tabulated version: use TabulatedFunction5D lookup (includes rime density and mu axes)
@inline function tabulated_number_weighted_fall_speed(table::P3Table5D, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_mass_weighted_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, oftype(m̄, prp.floors.mass_scale)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return vₜ_norm * ρ_correction
end

@inline function tabulated_number_weighted_fall_speed(table::P3Table5D,
                                                      prep::Prepared5DInterpolation, ρ_correction)
    return evaluate_at(table, prep) * ρ_correction
end

"""
$(TYPEDSIGNATURES)

Compute both ice terminal velocities (mass- and number-weighted) in a single call,
sharing the mean particle mass, the air density correction, and the 5D
interpolation indices between the two table reads.

[`ice_terminal_velocity_mass_weighted`](@ref) remains available for the one
caller that needs the mass-weighted speed alone.

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
- `NamedTuple` with fields `mass_weighted`, `number_weighted` [m/s]
  (both positive downward)
"""
# GPU-safe concrete struct (NamedTuple complicates the GPU compiler's NoInline boundaries).
struct IceTerminalVelocities{FT}
    mass_weighted :: FT
    number_weighted :: FT
end

function ice_terminal_velocities(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)), μ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    prp = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    # --- Shared computation (done once instead of twice) ---
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, p3.minimum_number_mixing_ratio)
    m̄ = qⁱ_eff / nⁱ_eff

    ρ_correction = ice_air_density_correction(ρ₀, ρ)

    velocities = fused_fall_speeds(fs.mass_weighted, fs.number_weighted,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return IceTerminalVelocities{FT}(
        ifelse(active, velocities.mass_weighted, zero(FT)),
        ifelse(active, velocities.number_weighted, zero(FT)),
    )
end

# Fast path: both tables are 5D (the supported P3 configuration with loaded tables).
# Interpolation indices for (log_m, Fᶠ, Fˡ, ρᶠ, μ) are shared across the two reads.
@inline function fused_fall_speeds(mass_table::P3Table5D, number_table::P3Table5D,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    FT = typeof(m̄)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_mass_weighted_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, FT(prp.floors.mass_scale)))
    prep = prepare_5d(mass_table, log_mean_mass, Fᶠ, Fˡ, ρᶠ, μ)
    return IceTerminalVelocities{FT}(
        tabulated_mass_weighted_fall_speed(mass_table, prep, ρ_correction),
        tabulated_number_weighted_fall_speed(number_table, prep, ρ_correction),
    )
end

# Fallback for non-5D fall speed tables (quadrature path, mixed types).
@inline function fused_fall_speeds(mass_table, number_table,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ)
    FT = typeof(m̄)
    return IceTerminalVelocities{FT}(
        tabulated_mass_weighted_fall_speed(mass_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ),
        tabulated_number_weighted_fall_speed(number_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, prp, μ),
    )
end
