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

# Stokes-regime cloud-droplet fall speed, `v(D) = a_cn D²`, with
# `a_cn = g ρʷ / (18 η)` and exponent 2, so the PSD-weighted moments follow from
# Γ(μᶜˡ+b+4)/Γ(μᶜˡ+4) = (μᶜˡ+5)(μᶜˡ+4) for mass and
# Γ(μᶜˡ+b+1)/Γ(μᶜˡ+1) = (μᶜˡ+2)(μᶜˡ+1) for number.
# `rime_density` needs the same mass-weighted speed to form the Cober-List rime-impact
# parameter, so both call these helpers rather than repeating the arithmetic with their
# own gravitational acceleration.
@inline cloud_stokes_prefactor(g, ρᴸ, η) =
    g * ρᴸ / (18 * max(η, oftype(η, DEFAULT_FLOORS.divisor)))

@inline cloud_mass_weighted_stokes_velocity(stokes_prefactor, μᶜˡ, λᶜˡ) =
    stokes_prefactor * (μᶜˡ + 5) * (μᶜˡ + 4) / λᶜˡ^2

@inline cloud_number_weighted_stokes_velocity(stokes_prefactor, μᶜˡ, λᶜˡ) =
    stokes_prefactor * (μᶜˡ + 2) * (μᶜˡ + 1) / λᶜˡ^2

# `μᶜˡ` and `λᶜˡ` are the cloud-DSD shape/slope diagnosed by `diagnose_cloud_dsd`;
# the caller passes the values already computed in `p3_ice_properties`
# (`properties.μᶜˡ`/`properties.λᶜˡ`) so the fall-speed kernel does not re-diagnose them.
@inline function cloud_terminal_velocities(p3, qᶜˡ, ρ, ν, μᶜˡ, λᶜˡ, constants)
    FT = typeof(qᶜˡ + ρ + ν + μᶜˡ + λᶜˡ)
    η = ν * ρ
    g = p3_gravitational_acceleration(constants, FT)
    stokes_prefactor = cloud_stokes_prefactor(g, p3.process_rates.liquid_water_density, η)
    active = qᶜˡ >= p3.minimum_mass_mixing_ratio
    mass_weighted = cloud_mass_weighted_stokes_velocity(stokes_prefactor, μᶜˡ, λᶜˡ)
    number_weighted = cloud_number_weighted_stokes_velocity(stokes_prefactor, μᶜˡ, λᶜˡ)
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
    parameters = p3.process_rates
    ρ₀ = parameters.reference_air_density
    ρʷ = parameters.liquid_water_density

    qʳ_eff = clamp_positive(qʳ)
    nʳ_eff = max(nʳ, p3.minimum_number_mixing_ratio)
    ρ_correction = ice_air_density_correction(ρ₀, ρ)

    m̄  = qʳ_eff / nʳ_eff
    λʳ = cbrt(FT(π) * ρʷ / max(m̄, FT(parameters.floors.mass_scale)))
    λʳ = clamp(λʳ, parameters.minimum_rain_slope, parameters.maximum_rain_slope)
    log_slope = log10(λʳ)

    mass_weighted_velocity = p3.rain.velocity_mass(log_slope) * ρ_correction
    number_weighted_velocity = p3.rain.velocity_number(log_slope) * ρ_correction
    active = qʳ_eff >= p3.minimum_mass_mixing_ratio

    return RainTerminalVelocities{FT}(ifelse(active, mass_weighted_velocity, zero(FT)),
                                      ifelse(active, number_weighted_velocity, zero(FT)))
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
@inline function ice_terminal_velocity_mass_weighted(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ;
                                                     Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    parameters = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, p3.minimum_number_mixing_ratio)

    # Mean ice particle mass
    m̄ = qⁱ_eff / nⁱ_eff

    ρ_correction = ice_air_density_correction(ρ₀, ρ)

    # m9: no velocity clamping is applied; the table bounds are sufficient.
    velocity = tabulated_fall_speed(fs.mass_weighted, m̄, Fᶠ, Fˡ,
                                    ρᶠ, ρ_correction, parameters)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return ifelse(active, velocity, zero(FT))
end

# Tabulated version: use the four-dimensional ice lookup table. The mass- and
# number-weighted speeds differ only in which table is passed in.
@inline function tabulated_fall_speed(table::P3Table4D, m̄, Fᶠ, Fˡ,
                                      ρᶠ, ρ_correction, parameters)
    # m̄ = qⁱ/nⁱ is a per-particle mass [kg]; floor it only with a tiny log-guard,
    # NOT the bulk mass-mixing-ratio threshold `minimum_mass_mixing_ratio` (kg/kg).
    # The table clamps the coordinate to its mass axis (min ≈ 1.56e-15 kg) rather
    # than extrapolating below it.
    log_mean_mass = log10(max(m̄, oftype(m̄, parameters.floors.mass_scale)))
    vₜ_norm = table(log_mean_mass, Fᶠ, Fˡ, ρᶠ)
    return vₜ_norm * ρ_correction
end

# Prepared-index variant: reuse precomputed interpolation indices and skip the log/clamp setup.
@inline tabulated_fall_speed(table::P3Table4D, prep::PreparedInterpolation, ρ_correction) =
    evaluate_at(table, prep) * ρ_correction

"""
$(TYPEDSIGNATURES)

Compute both ice terminal velocities (mass- and number-weighted) in a single call,
sharing the mean particle mass, the air density correction, and the 4D
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

function ice_terminal_velocities(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=zero(typeof(qⁱ)))
    FT = typeof(qⁱ)
    parameters = p3.process_rates
    fs = p3.ice.fall_speed

    ρ₀ = fs.reference_air_density

    # --- Shared computation (done once instead of twice) ---
    qⁱ_eff = clamp_positive(qⁱ)
    nⁱ_eff = max(nⁱ, p3.minimum_number_mixing_ratio)
    m̄ = qⁱ_eff / nⁱ_eff

    ρ_correction = ice_air_density_correction(ρ₀, ρ)

    velocities = fused_fall_speeds(fs.mass_weighted, fs.number_weighted,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, parameters)
    active = qⁱ_eff >= p3.minimum_mass_mixing_ratio
    return IceTerminalVelocities{FT}(
        ifelse(active, velocities.mass_weighted, zero(FT)),
        ifelse(active, velocities.number_weighted, zero(FT)),
    )
end

# Fast path: both tables are 4D (the supported P3 configuration with loaded tables).
# Interpolation indices for (log_m, Fᶠ, Fˡ, ρᶠ) are shared across the two reads.
@inline function fused_fall_speeds(mass_table::P3Table4D, number_table::P3Table4D,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, parameters)
    FT = typeof(m̄)
    # Per-particle-mass log-guard; the table clamps its mass axis (see
    # tabulated_fall_speed), not the bulk qmin.
    log_mean_mass = log10(max(m̄, FT(parameters.floors.mass_scale)))
    prep = prepare_interpolation(mass_table, log_mean_mass, Fᶠ, Fˡ, ρᶠ)
    return IceTerminalVelocities{FT}(
        tabulated_fall_speed(mass_table, prep, ρ_correction),
        tabulated_fall_speed(number_table, prep, ρ_correction),
    )
end

# Fallback for non-4D fall speed tables (quadrature path, mixed types).
@inline function fused_fall_speeds(mass_table, number_table,
                                    m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, p3, parameters)
    FT = typeof(m̄)
    return IceTerminalVelocities{FT}(
        tabulated_fall_speed(mass_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, parameters),
        tabulated_fall_speed(number_table, m̄, Fᶠ, Fˡ, ρᶠ, ρ_correction, parameters),
    )
end
