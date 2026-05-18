#####
##### Gamma distribution moment integrals
#####

"""
$(TYPEDSIGNATURES)

Compute log(scale × ∫₀^∞ D^k G(D) dD) where G(D) = D^μ exp(-λD).

The integral equals Γ(k+μ+1) / λ^(k+μ+1).
"""
function log_gamma_moment(μ, logλ; k = 0, scale = 1)
    FT = typeof(μ)
    z = k + μ + 1
    return -z * logλ + loggamma(z) + log(FT(scale))
end

"""
$(TYPEDSIGNATURES)

Compute log(scale × ∫_{D₁}^{D₂} D^k G(D) dD) using incomplete gamma functions.
"""
function log_gamma_inc_moment(D₁, D₂, μ, logλ; k = 0, scale = 1)
    FT = typeof(μ)
    D₁ < D₂ || return log(zero(FT))

    z = k + μ + 1
    λ = exp(logλ)

    (_, q₁) = gamma_inc(z, λ * D₁)
    (_, q₂) = gamma_inc(z, λ * D₂)

    # Use a tiny floor (eps²) instead of eps to avoid amplification by large
    # scale factors (e.g. α/(1-Fr) at Fr≈1).  The old eps floor produced
    # spurious regime-4 contributions when multiplied by ~1e15 scale.
    Δq = max(q₁ - q₂, eps(FT)^2)

    return -z * logλ + loggamma(z) + log(Δq) + log(FT(scale))
end

"""
$(TYPEDSIGNATURES)

Compute log(exp(a) + exp(b)) stably.
"""
function logaddexp(a, b)
    # Compute both forms and select the stable one
    result_a_larger = a + log1p(exp(b - a))
    result_b_larger = b + log1p(exp(a - b))
    return ifelse(a > b, result_a_larger, result_b_larger)
end

"""
$(TYPEDSIGNATURES)

Compute log(∫₀^∞ Dⁿ m(D) N'(D) dD / N₀) over the piecewise mass-diameter relationship.

When `liquid_fraction` Fˡ > 0, the total mass includes a liquid coating term:
`m(D) = (1 - Fˡ) × m_ice(D) + Fˡ × ρᴸ × π/6 × D³`,
matching the Fortran convention in `create_p3_lookupTable_1.f90`.
"""
function log_mass_moment(mass::IceMassPowerLaw, rime_fraction, rime_density, μ, logλ;
                         n = 0, liquid_fraction = zero(typeof(μ)))
    FT = typeof(μ)
    Fᶠ = rime_fraction
    Fˡ = liquid_fraction

    thresholds = ice_regime_thresholds(mass, rime_fraction, rime_density)
    α = mass.coefficient
    β = mass.exponent
    ρᵢ = mass.ice_density

    # Regime 1: small spherical ice [0, D_spherical)
    a₁ = ρᵢ * FT(π) / 6
    log_M₁ = log_gamma_inc_moment(zero(FT), thresholds.spherical, μ, logλ; k = 3 + n, scale = a₁)

    # Compute unrimed case: aggregates [D_spherical, ∞)
    log_M₂_unrimed = log_gamma_inc_moment(thresholds.spherical, FT(Inf), μ, logλ; k = β + n, scale = α)
    unrimed_result = logaddexp(log_M₁, log_M₂_unrimed)

    # Compute rimed case (regimes 2-4)
    # Clamp Fᶠ away from 0 and 1 to avoid 0*Inf=NaN in IEEE arithmetic
    Fᶠ_safe = clamp(Fᶠ, eps(FT), 1 - eps(FT))

    # Regime 2: rimed aggregates [D_spherical, D_graupel)
    log_M₂ = log_gamma_inc_moment(thresholds.spherical, thresholds.graupel, μ, logλ; k = β + n, scale = α)

    # Regime 3: graupel [D_graupel, D_partial)
    a₃ = thresholds.ρ_graupel * FT(π) / 6
    log_M₃ = log_gamma_inc_moment(thresholds.graupel, thresholds.partial_rime, μ, logλ; k = 3 + n, scale = a₃)

    # Regime 4: partially rimed [D_partial, ∞)
    a₄ = α / (1 - Fᶠ_safe)
    log_M₄ = log_gamma_inc_moment(thresholds.partial_rime, FT(Inf), μ, logλ; k = β + n, scale = a₄)

    rimed_result = logaddexp(logaddexp(log_M₁, log_M₂), logaddexp(log_M₃, log_M₄))

    # Select ice-only mass moment based on whether ice is rimed
    log_M_ice = ifelse(iszero(Fᶠ), unrimed_result, rimed_result)

    # Add liquid mass contribution: Fˡ × ρᴸ × π/6 × D³
    # Total mass = (1 - Fˡ) × m_ice(D) + Fˡ × ρᴸ × π/6 × D³
    # In log-space: logaddexp(log(1-Fˡ) + log_M_ice, log(Fˡ) + log_M_liquid)
    ρᴸ = FT(1000)
    a_liquid = ρᴸ * FT(π) / 6
    log_M_liquid = log_gamma_moment(μ, logλ; k = 3 + n, scale = a_liquid)

    Fˡ_safe = clamp(Fˡ, eps(FT), 1 - eps(FT))
    log_total = logaddexp(log(1 - Fˡ_safe) + log_M_ice, log(Fˡ_safe) + log_M_liquid)

    # If no liquid, return ice-only result
    return ifelse(Fˡ < eps(FT), log_M_ice, log_total)
end

#####
##### Lambda solver (two-moment)
#####

"""
$(TYPEDSIGNATURES)

Compute log(L_ice / N_ice) as a function of logλ for two-moment closure.
Includes L_ice and N_ice arguments to support the TwoMomentClosure D_mvd diagnostic.
"""
function log_mass_number_ratio(mass::IceMassPowerLaw,
                               closure,
                               rime_fraction, rime_density, liquid_fraction, logλ, L_ice, N_ice)
    μ = shape_parameter(closure, logλ, L_ice, N_ice, rime_fraction, rime_density, liquid_fraction, mass)
    log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ;
                                     liquid_fraction)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return log_L_over_N₀ - log_N_over_N₀
end

#####
##### Lambda solver (three-moment)
#####

"""
$(TYPEDSIGNATURES)

Compute log(λ) from log(Z/N) given shape parameter μ.
"""
function log_lambda_from_reflectivity(μ, log_Z_over_N)
    # From Z/N = Γ(μ+7) / (Γ(μ+1) λ⁶)
    # λ⁶ = Γ(μ+7) / (Γ(μ+1) × Z/N)
    # log(λ) = [loggamma(μ+7) - loggamma(μ+1) - log(Z/N)] / 6
    return (loggamma(μ + 7) - loggamma(μ + 1) - log_Z_over_N) / 6
end

"""
$(TYPEDSIGNATURES)

Approximate the three-moment ice shape parameter using the Fortran P3 `G(μ)` fit.

This matches `compute_mu_3mom_1` in the reference P3 Fortran code:
it forms ``G = M₀ M₆ / M₃²`` and applies the piecewise polynomial inversion
used by `solve_mui`.
"""
@inline function shape_parameter_from_moments(M₀, M₃, M₆, μmax)
    FT = promote_type(typeof(M₀), typeof(M₃), typeof(M₆), typeof(μmax))
    M₃_min = FT(1e-20)

    M₃ <= M₃_min && return FT(μmax)

    # Promote to Float64 for piecewise polynomial evaluation (Fortran uses
    # double precision). Near breakpoints, Float32 rounding assigns G to the
    # wrong segment, producing incorrect μ.
    G64 = Float64(M₀ / M₃) * Float64(M₆ / M₃)
    G²64 = G64 * G64

    μ64 = if G64 >= 20.0
        0.0
    elseif G64 >= 13.31
        3.3638e-3 * G²64 - 1.7152e-1 * G64 + 2.0857
    elseif G64 >= 7.123
        1.5900e-2 * G²64 - 4.8202e-1 * G64 + 4.0108
    elseif G64 >= 4.2
        1.0730e-1 * G²64 - 1.7481 * G64 + 8.4246
    elseif G64 >= 2.946
        5.9070e-1 * G²64 - 5.7918 * G64 + 16.919
    elseif G64 >= 1.793
        4.3966 * G²64 - 26.659 * G64 + 45.477
    elseif G64 >= 1.472
        47.552 * G²64 - 179.58 * G64 + 181.26
    else
        Float64(μmax)
    end

    return FT(min(max(μ64, 0.0), Float64(μmax)))
end

"""
$(TYPEDSIGNATURES)

Compute the full three-moment mass residual for the exact closure.
"""
function mass_residual_three_moment(mass::IceMassPowerLaw,
                                    rime_fraction, rime_density,
                                    μ, log_Z_over_N, log_L_over_N)
    logλ = log_lambda_from_reflectivity(μ, log_Z_over_N)
    log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    computed_log_L_over_N = log_L_over_N₀ - log_N_over_N₀

    return computed_log_L_over_N - log_L_over_N
end

"""
$(TYPEDSIGNATURES)

Compute G(μ) = Γ(μ+7)Γ(μ+1) / Γ(μ+4)² for the three-moment μ-Z constraint.
Simplifies to (μ+6)(μ+5)(μ+4) / ((μ+3)(μ+2)(μ+1)).
Matches Fortran `G_of_mu`.
"""
@inline function g_of_mu(μ)
    return (μ + 6) * (μ + 5) * (μ + 4) / ((μ + 3) * (μ + 2) * (μ + 1))
end

"""
$(TYPEDSIGNATURES)

Bound Z_ice to a physically consistent range based on the μ bounds.
Matches Fortran `apply_mui_bounds_to_zi` and basic zsmall/zlarge clamps.
"""
@inline function enforce_z_bounds(Z_ice, L_ice, N_ice, ρ_bulk, μmin, μmax)
    FT = typeof(Z_ice)
    # Basic magnitude bounds (Fortran zsmall/zlarge)
    Z_clamped = clamp(Z_ice, FT(1e-35), FT(1))

    # Moment-based bounds: G(μ_max) × M₃²/N ≤ Z ≤ G(μ_min) × M₃²/N
    M₃ = FT(6) * L_ice / (FT(π) * max(ρ_bulk, eps(FT)))
    M₃²_over_N = M₃^2 / max(N_ice, eps(FT))

    G_min = g_of_mu(μmin)  # upper Z bound (wide distribution)
    G_max = g_of_mu(μmax)  # lower Z bound (narrow distribution)

    Z_clamped = min(Z_clamped, G_min * M₃²_over_N)
    Z_clamped = max(Z_clamped, G_max * M₃²_over_N)

    return Z_clamped
end
