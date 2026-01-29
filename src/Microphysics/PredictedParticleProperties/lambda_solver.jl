#####
##### Lambda Solver for P3 Ice Size Distribution
#####
##### Given prognostic moments and ice properties (rime fraction, rime density),
##### solve for the gamma distribution parameters (N₀, λ, μ).
#####
##### The solver handles the piecewise mass-diameter relationship with four regimes
##### from Morrison & Milbrandt (2015a) Equations 1-5.
#####
##### Two closures are available:
##### 1. Two-moment: Uses μ-λ relationship from Field et al. (2007)
##### 2. Three-moment: Uses sixth moment Z to determine μ independently
#####

#####
##### Mass-diameter relationship parameters
#####

"""
    IceMassPowerLaw

Power law for ice particle mass. See [`IceMassPowerLaw()`](@ref) constructor.
"""
struct IceMassPowerLaw{FT}
    coefficient :: FT
    exponent :: FT
    ice_density :: FT
end

"""
$(TYPEDSIGNATURES)

Construct power law parameters for ice particle mass: ``m(D) = α D^β``.

For vapor-grown aggregates (regime 2 in P3), the mass-diameter relationship
follows a power law with empirically-determined coefficients. This captures
the fractal nature of ice crystal aggregates, which have effective densities
much lower than pure ice.

# Physical Interpretation

The exponent ``β ≈ 1.9`` (less than 3) means density decreases with size:
- Small particles: closer to solid ice density
- Large aggregates: fluffy, low effective density

This is the key to P3's smooth transitions—as particles grow and aggregate,
their properties evolve continuously without discrete category jumps.

# Keyword Arguments

- `coefficient`: α in m(D) = α D^β [kg/m^β], default 0.0121
- `exponent`: β in m(D) = α D^β [-], default 1.9
- `ice_density`: Pure ice density [kg/m³], default 917

# References

Default parameters from [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)
supplementary material, based on aircraft observations.
"""
function IceMassPowerLaw(FT = Oceananigans.defaults.FloatType;
                         coefficient = 0.0121,
                         exponent = 1.9,
                         ice_density = 917)
    return IceMassPowerLaw(FT(coefficient), FT(exponent), FT(ice_density))
end

#####
##### μ-λ relationship
#####

#####
##### Two-moment closure: μ-λ relationship
#####

"""
    TwoMomentClosure

μ-λ closure for two-moment PSD. See [`TwoMomentClosure()`](@ref) constructor.
"""
struct TwoMomentClosure{FT}
    a :: FT
    b :: FT
    c :: FT
    μmax :: FT
end

"""
$(TYPEDSIGNATURES)

Construct the μ-λ relationship for gamma size distribution closure.

With only two prognostic moments (mass and number), we need a closure
to determine the three-parameter gamma distribution (N₀, μ, λ). P3 uses
an empirical power-law relating shape parameter μ to slope parameter λ:

```math
μ = \\text{clamp}(a λ^b - c, 0, μ_{max})
```

This relationship was fitted to aircraft observations of ice particle
size distributions by [Field et al. (2007)](@cite FieldEtAl2007).

# Physical Interpretation

- **Small λ** (large particles): μ → 0, giving an exponential distribution
- **Large λ** (small particles): μ increases, narrowing the distribution

The clamping to [0, μmax] ensures physical distributions with non-negative
shape parameter and prevents unrealistically narrow distributions.

# Keyword Arguments

- `a`: Coefficient in μ = a λ^b - c, default 0.00191
- `b`: Exponent in μ = a λ^b - c, default 0.8
- `c`: Offset in μ = a λ^b - c, default 2
- `μmax`: Maximum shape parameter, default 6

# References

From [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 27,
based on [Field et al. (2007)](@cite FieldEtAl2007) observations.
"""
function TwoMomentClosure(FT = Oceananigans.defaults.FloatType;
                          a = 0.00191,
                          b = 0.8,
                          c = 2,
                          μmax = 6)
    return TwoMomentClosure(FT(a), FT(b), FT(c), FT(μmax))
end

# Backwards compatibility alias
const ShapeParameterRelation = TwoMomentClosure

"""
    P3Closure

Updated μ-λ closure for P3, including the large-particle diagnostic.
See [`P3Closure()`](@ref) constructor.
"""
struct P3Closure{FT}
    # Constants for small particle regime (Field et al. 2007)
    a :: FT
    b :: FT
    c :: FT
    μmax_small :: FT
    # Constants for large particle regime
    μmax_large :: FT
    D_threshold :: FT # Threshold diameter [m] (0.2 mm)
end

"""
$(TYPEDSIGNATURES)

Construct the P3 μ-λ closure which includes a diagnostic for large rimed particles.

This closure matches the logic in the official P3 Fortran code (lookup table generation).
It uses the Field et al. (2007) relation for small particles, but switches to
a diagnostic based on mean volume diameter (D_mvd) for large particles to account
for riming effects.

# Logic

1. Compute mean volume diameter ``D_{mvd} = (L / (\\frac{\\pi}{6} \\rho_g))^{1/3}``
2. If ``D_{mvd} \\le 0.2`` mm:
   Use Field et al. (2007) relation: ``\\mu = 0.076 \\lambda^{0.8} - 2`` (clamped [0, 6])
3. If ``D_{mvd} > 0.2`` mm:
   ``\\mu = 0.25 (D_{mvd} - 0.2) f_\\rho F^f`` (clamped [0, 20])
   where ``f_\\rho = \\max(1, 1 + 0.00842(\\rho_g - 400))``

# Keyword Arguments

- `a`, `b`, `c`: Constants for small regime (same as TwoMomentClosure)
- `μmax_small`: Max μ for small regime (default 6)
- `μmax_large`: Max μ for large regime (default 20)
- `D_threshold`: Threshold D_mvd [m] (default 2e-4)
"""
function P3Closure(FT = Oceananigans.defaults.FloatType;
                   a = 0.00191,
                   b = 0.8,
                   c = 2,
                   μmax_small = 6,
                   μmax_large = 20,
                   D_threshold = 2e-4)
    return P3Closure(FT(a), FT(b), FT(c), FT(μmax_small), FT(μmax_large), FT(D_threshold))
end

"""
    shape_parameter(closure, logλ, L_ice, rime_fraction, rime_density, mass_params)

Compute shape parameter μ.
"""
function shape_parameter(closure::TwoMomentClosure, logλ, args...)
    λ = exp(logλ)
    μ = closure.a * λ^closure.b - closure.c
    return clamp(μ, zero(μ), closure.μmax)
end

function shape_parameter(closure::P3Closure, logλ, L_ice, rime_fraction, rime_density, mass::IceMassPowerLaw)
    FT = typeof(closure.a)
    λ = exp(logλ)

    # 1. Compute graupel density (rho_g)
    if iszero(rime_fraction)
        ρ_g = mass.ice_density
    else
        ρ_dep = deposited_ice_density(mass, rime_fraction, rime_density)
        ρ_g = graupel_density(rime_fraction, rime_density, ρ_dep)
    end

    # 2. Compute D_mvd (Mean Volume Diameter)
    # D_mvd = (L / ((pi/6) * rho_g))^(1/3)
    val = L_ice / (FT(π)/6 * ρ_g)
    # Handle L=0 case
    if val <= 0
        D_mvd = zero(FT)
    else
        D_mvd = val^(1/3) # in meters
    end

    # 3. Branch based on D_mvd
    if D_mvd <= closure.D_threshold
        # Small regime: Heymsfield 2003
        μ = closure.a * λ^closure.b - closure.c
        μ = clamp(μ, zero(FT), closure.μmax_small)
    else
        # Large regime
        D_mvd_mm = D_mvd * 1000
        D_thres_mm = closure.D_threshold * 1000

        # Density adjustment factor
        f_ρ = max(one(FT), one(FT) + FT(0.00842) * (ρ_g - 400))

        # μ = 0.25 * (D_mvd_mm - 0.2) * f_rho * F^f
        μ = FT(0.25) * (D_mvd_mm - D_thres_mm) * f_ρ * rime_fraction

        μ = clamp(μ, zero(FT), closure.μmax_large)
    end

    return μ
end

#####
##### Three-moment closure: Z/N constraint
#####

"""
    ThreeMomentClosure

Three-moment closure using reflectivity. See [`ThreeMomentClosure()`](@ref) constructor.
"""
struct ThreeMomentClosure{FT}
    μmin :: FT
    μmax :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a three-moment closure for gamma size distribution.

With three prognostic moments (mass L, number N, and reflectivity Z),
the shape parameter μ can be diagnosed directly from the moment ratios,
without requiring an empirical μ-λ relationship.

# Three-Moment Approach

For a gamma distribution ``N'(D) = N₀ D^μ e^{-λD}``, the moments are:
- ``M_0 = N = N₀ Γ(μ+1) / λ^{μ+1}``
- ``M_6 = Z = N₀ Γ(μ+7) / λ^{μ+7}``

The sixth-to-zeroth moment ratio gives:

```math
Z/N = Γ(μ+7) / (Γ(μ+1) λ^6)
```

Combined with the mass constraint, this provides two equations for two
unknowns (μ, λ), eliminating the need for the empirical μ-λ closure.

# Advantages

- **Physical basis**: μ evolves based on actual size distribution changes
- **Better representation of size sorting**: Differential sedimentation
  can narrow or broaden distributions independently of total mass/number
- **Improved hail simulation**: Crucial for representing the distinct
  size distributions of large, heavily rimed particles

# Keyword Arguments

- `μmin`: Minimum shape parameter, default 0 (exponential distribution)
- `μmax`: Maximum shape parameter, default 20

# References

[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) introduced three-moment ice,
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) refined the implementation.
"""
function ThreeMomentClosure(FT = Oceananigans.defaults.FloatType;
                            μmin = 0,
                            μmax = 20)
    return ThreeMomentClosure(FT(μmin), FT(μmax))
end

#####
##### Diameter thresholds between particle regimes
#####

"""
    regime_threshold(α, β, ρ)

Diameter threshold from mass power law: D = (6α / πρ)^(1/(3-β))

Used to determine boundaries between spherical ice, aggregates, and graupel.
"""
function regime_threshold(α, β, ρ)
    FT = typeof(α)
    return (6 * α / (FT(π) * ρ))^(1 / (3 - β))
end

"""
    deposited_ice_density(mass, rime_fraction, rime_density)

Density of the vapor-deposited (unrimed) portion of ice particles.
Equation 16 in [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
function deposited_ice_density(mass::IceMassPowerLaw, rime_fraction, rime_density)
    β = mass.exponent
    Fᶠ = rime_fraction
    ρᶠ = rime_density
    FT = typeof(β)

    # Handle unrimed case to avoid division by zero
    if Fᶠ <= eps(FT)
        return mass.ice_density
    end

    k = (1 - Fᶠ)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ
    den = (β - 2) * (k - 1) / ((1 - Fᶠ) * k - 1) - (1 - Fᶠ)
    return num / den
end

"""
    graupel_density(rime_fraction, rime_density, deposited_density)

Bulk density of graupel particles (rime + deposited ice).
"""
function graupel_density(rime_fraction, rime_density, deposited_density)
    return rime_fraction * rime_density + (1 - rime_fraction) * deposited_density
end

"""
    IceRegimeThresholds

Diameter thresholds between P3 ice regimes. See [`ice_regime_thresholds`](@ref).
"""
struct IceRegimeThresholds{FT}
    spherical :: FT
    graupel :: FT
    partial_rime :: FT
    ρ_graupel :: FT
end

"""
$(TYPEDSIGNATURES)

Compute diameter thresholds separating the four P3 ice particle regimes.

P3's key innovation is a piecewise mass-diameter relationship that
transitions smoothly between ice particle types:

1. **Small spherical** (D < D_th): Dense, nearly solid ice crystals
2. **Vapor-grown aggregates** (D_th ≤ D < D_gr): Fractal aggregates, m ∝ D^β
3. **Graupel** (D_gr ≤ D < D_cr): Compact, heavily rimed particles
4. **Partially rimed** (D ≥ D_cr): Large aggregates with rimed cores

The thresholds depend on rime fraction and rime density, so they evolve
as particles rime—no ad-hoc category conversions needed.

# Arguments

- `mass`: Power law parameters for vapor-grown aggregates
- `rime_fraction`: Fraction of particle mass that is rime (0 to 1)
- `rime_density`: Density of rime layer [kg/m³]

# Returns

[`IceRegimeThresholds`](@ref) with fields:
- `spherical`: D_th threshold [m]
- `graupel`: D_gr threshold [m]
- `partial_rime`: D_cr threshold [m]
- `ρ_graupel`: Bulk density of graupel [kg/m³]

# References

See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Equations 12-14.
"""
function ice_regime_thresholds(mass::IceMassPowerLaw, rime_fraction, rime_density)
    α = mass.coefficient
    β = mass.exponent
    ρᵢ = mass.ice_density
    Fᶠ = rime_fraction
    ρᶠ = rime_density
    FT = typeof(α)

    D_spherical = regime_threshold(α, β, ρᵢ)

    # Compute rimed case thresholds (will be ignored if unrimed)
    # Use safe values to avoid division by zero when Fᶠ = 0
    Fᶠ_safe = max(Fᶠ, eps(FT))
    ρ_dep = deposited_ice_density(mass, Fᶠ_safe, ρᶠ)
    ρ_g = graupel_density(Fᶠ_safe, ρᶠ, ρ_dep)

    D_graupel = regime_threshold(α, β, ρ_g)
    D_partial = regime_threshold(α, β, ρ_g * (1 - Fᶠ_safe))

    # For unrimed ice (Fᶠ = 0), use Inf thresholds; otherwise use computed values
    is_unrimed = iszero(Fᶠ)
    D_graupel_out = ifelse(is_unrimed, FT(Inf), D_graupel)
    D_partial_out = ifelse(is_unrimed, FT(Inf), D_partial)
    ρ_g_out = ifelse(is_unrimed, ρᵢ, ρ_g)

    return IceRegimeThresholds(D_spherical, D_graupel_out, D_partial_out, ρ_g_out)
end

"""
    ice_mass_coefficients(mass, rime_fraction, rime_density, D)

Return (a, b) for ice mass at diameter D: m(D) = a D^b.

The relationship is piecewise across four regimes:
1. D < D_spherical: small spheres, m = (π/6)ρᵢ D³
2. D_spherical ≤ D < D_graupel: aggregates, m = α D^β
3. D_graupel ≤ D < D_partial: graupel, m = (π/6)ρ_g D³
4. D ≥ D_partial: partially rimed, m = α/(1-Fᶠ) D^β
"""
function ice_mass_coefficients(mass::IceMassPowerLaw, rime_fraction, rime_density, D)
    FT = typeof(D)
    α = mass.coefficient
    β = mass.exponent
    ρᵢ = mass.ice_density
    Fᶠ = rime_fraction

    thresholds = ice_regime_thresholds(mass, rime_fraction, rime_density)

    # Regime 1: small spheres
    a₁ = ρᵢ * FT(π) / 6
    b₁ = FT(3)

    # Regime 2: aggregates (also used for unrimed large particles)
    a₂ = FT(α)
    b₂ = FT(β)

    # Regime 3: graupel
    a₃ = thresholds.ρ_graupel * FT(π) / 6
    b₃ = FT(3)

    # Regime 4: partially rimed (avoid division by zero)
    Fᶠ_safe = min(Fᶠ, 1 - eps(FT))
    a₄ = FT(α) / (1 - Fᶠ_safe)
    b₄ = FT(β)

    # Determine which regime applies (work backwards from regime 4)
    is_regime_4 = D ≥ thresholds.partial_rime
    is_regime_3 = D ≥ thresholds.graupel
    is_regime_2 = D ≥ thresholds.spherical

    # Select coefficients: start with regime 4, override with 3, 2, 1 as conditions apply
    a = ifelse(is_regime_4, a₄, a₃)
    b = ifelse(is_regime_4, b₄, b₃)

    a = ifelse(is_regime_3, a, a₂)
    b = ifelse(is_regime_3, b, b₂)

    a = ifelse(is_regime_2, a, a₁)
    b = ifelse(is_regime_2, b, b₁)

    return (a, b)
end

"""
    ice_mass(mass, rime_fraction, rime_density, D)

Compute ice particle mass at diameter D.
"""
function ice_mass(mass::IceMassPowerLaw, rime_fraction, rime_density, D)
    (a, b) = ice_mass_coefficients(mass, rime_fraction, rime_density, D)
    return a * D^b
end

#####
##### Gamma distribution moment integrals
#####

"""
    log_gamma_moment(μ, logλ; k=0, scale=1)

Compute log(scale × ∫₀^∞ D^k G(D) dD) where G(D) = D^μ exp(-λD).

The integral equals Γ(k+μ+1) / λ^(k+μ+1).
"""
function log_gamma_moment(μ, logλ; k = 0, scale = 1)
    FT = typeof(μ)
    z = k + μ + 1
    return -z * logλ + loggamma(z) + log(FT(scale))
end

"""
    log_gamma_inc_moment(D₁, D₂, μ, logλ; k=0, scale=1)

Compute log(scale × ∫_{D₁}^{D₂} D^k G(D) dD) using incomplete gamma functions.
"""
function log_gamma_inc_moment(D₁, D₂, μ, logλ; k = 0, scale = 1)
    FT = typeof(μ)
    D₁ < D₂ || return log(zero(FT))

    z = k + μ + 1
    λ = exp(logλ)

    (_, q₁) = gamma_inc(z, λ * D₁)
    (_, q₂) = gamma_inc(z, λ * D₂)

    Δq = max(q₁ - q₂, eps(FT))

    return -z * logλ + loggamma(z) + log(Δq) + log(FT(scale))
end

"""
    logaddexp(a, b)

Compute log(exp(a) + exp(b)) stably.
"""
function logaddexp(a, b)
    # Compute both forms and select the stable one
    result_a_larger = a + log1p(exp(b - a))
    result_b_larger = b + log1p(exp(a - b))
    return ifelse(a > b, result_a_larger, result_b_larger)
end

"""
    log_mass_moment(mass, rime_fraction, rime_density, μ, logλ; n=0)

Compute log(∫₀^∞ Dⁿ m(D) N'(D) dD / N₀) over the piecewise mass-diameter relationship.
"""
function log_mass_moment(mass::IceMassPowerLaw, rime_fraction, rime_density, μ, logλ; n = 0)
    FT = typeof(μ)
    Fᶠ = rime_fraction

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
    # Use safe rime fraction to avoid division by zero
    Fᶠ_safe = max(Fᶠ, eps(FT))

    # Regime 2: rimed aggregates [D_spherical, D_graupel)
    log_M₂ = log_gamma_inc_moment(thresholds.spherical, thresholds.graupel, μ, logλ; k = β + n, scale = α)

    # Regime 3: graupel [D_graupel, D_partial)
    a₃ = thresholds.ρ_graupel * FT(π) / 6
    log_M₃ = log_gamma_inc_moment(thresholds.graupel, thresholds.partial_rime, μ, logλ; k = 3 + n, scale = a₃)

    # Regime 4: partially rimed [D_partial, ∞)
    a₄ = α / (1 - Fᶠ_safe)
    log_M₄ = log_gamma_inc_moment(thresholds.partial_rime, FT(Inf), μ, logλ; k = β + n, scale = a₄)

    rimed_result = logaddexp(logaddexp(log_M₁, log_M₂), logaddexp(log_M₃, log_M₄))

    # Select result based on whether ice is rimed
    return ifelse(iszero(Fᶠ), unrimed_result, rimed_result)
end

#####
##### Lambda solver (two-moment)
#####

"""
    log_mass_number_ratio(mass, closure, rime_fraction, rime_density, logλ, L_ice)

Compute log(L_ice / N_ice) as a function of logλ for two-moment closure.
Includes L_ice argument to support the P3Closure diagnostic.
"""
function log_mass_number_ratio(mass::IceMassPowerLaw,
                               closure,
                               rime_fraction, rime_density, logλ, L_ice)
    μ = shape_parameter(closure, logλ, L_ice, rime_fraction, rime_density, mass)
    log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return log_L_over_N₀ - log_N_over_N₀
end

#####
##### Lambda solver (three-moment)
#####

"""
    log_sixth_moment(μ, logλ)

Compute log(M₆/N₀) = log(∫₀^∞ D⁶ N'(D) dD / N₀) for a gamma distribution.

The sixth moment integral equals Γ(μ+7) / λ^(μ+7).
"""
function log_sixth_moment(μ, logλ)
    return log_gamma_moment(μ, logλ; k = 6)
end

"""
    log_reflectivity_number_ratio(μ, logλ)

Compute log(Z/N) for a gamma distribution.

```math
Z/N = Γ(μ+7) / (Γ(μ+1) λ^6)
```
"""
function log_reflectivity_number_ratio(μ, logλ)
    log_Z_over_N₀ = log_sixth_moment(μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return log_Z_over_N₀ - log_N_over_N₀
end

"""
    lambda_from_reflectivity(μ, Z_ice, N_ice)

Compute λ from the Z/N ratio given a fixed shape parameter μ.

From Z/N = Γ(μ+7) / (Γ(μ+1) λ⁶), we get:
```math
λ = \\left( \\frac{Γ(μ+7) N}{Γ(μ+1) Z} \\right)^{1/6}
```
"""
function lambda_from_reflectivity(μ, Z_ice, N_ice)
    FT = typeof(μ)
    (iszero(Z_ice) || iszero(N_ice)) && return FT(Inf)

    log_ratio = loggamma(μ + 7) - loggamma(μ + 1) + log(N_ice) - log(Z_ice)
    return exp(log_ratio / 6)
end

"""
    log_lambda_from_reflectivity(μ, log_Z_over_N)

Compute log(λ) from log(Z/N) given shape parameter μ.
"""
function log_lambda_from_reflectivity(μ, log_Z_over_N)
    # From Z/N = Γ(μ+7) / (Γ(μ+1) λ⁶)
    # λ⁶ = Γ(μ+7) / (Γ(μ+1) × Z/N)
    # log(λ) = [loggamma(μ+7) - loggamma(μ+1) - log(Z/N)] / 6
    return (loggamma(μ + 7) - loggamma(μ + 1) - log_Z_over_N) / 6
end

"""
    mass_residual_three_moment(mass, rime_fraction, rime_density, μ, log_Z_over_N, log_L_over_N)

Compute the mass constraint residual for three-moment solving.

Given μ and log(Z/N), we can compute λ. Then the residual is:
  computed log(L/N) - target log(L/N)

This should be zero at the correct μ.
"""
function mass_residual_three_moment(mass::IceMassPowerLaw,
                                    rime_fraction, rime_density,
                                    μ, log_Z_over_N, log_L_over_N)
    # Compute λ from Z/N constraint
    logλ = log_lambda_from_reflectivity(μ, log_Z_over_N)

    # Compute L/N at this (μ, λ)
    log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    computed_log_L_over_N = log_L_over_N₀ - log_N_over_N₀

    return computed_log_L_over_N - log_L_over_N
end

"""
    solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                          mass = IceMassPowerLaw(),
                          closure = ThreeMomentClosure(),
                          max_iterations = 50,
                          tolerance = 1e-10)

Solve for shape parameter μ using the three-moment constraint.

The algorithm:
1. For each candidate μ, compute λ from the Z/N constraint
2. Check if the resulting (μ, λ) satisfies the L/N constraint
3. Use bisection to find the μ that satisfies both constraints

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment / reflectivity [m⁶/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]

# Returns
- Shape parameter μ
"""
function solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                               mass = IceMassPowerLaw(),
                               closure = ThreeMomentClosure(),
                               max_iterations = 50,
                               tolerance = 1e-10)
    FT = typeof(L_ice)

    # Handle edge cases
    (iszero(N_ice) || iszero(L_ice) || iszero(Z_ice)) && return closure.μmin

    # Target ratios
    log_Z_over_N = log(Z_ice) - log(N_ice)
    log_L_over_N = log(L_ice) - log(N_ice)

    # Residual function
    f(μ) = mass_residual_three_moment(mass, rime_fraction, rime_density,
                                       μ, log_Z_over_N, log_L_over_N)

    # Bisection method over [μmin, μmax]
    μ_lo = closure.μmin
    μ_hi = closure.μmax

    f_lo = f(μ_lo)
    f_hi = f(μ_hi)

    # Check if solution is in bounds
    # If residuals have same sign, clamp to appropriate bound
    same_sign = f_lo * f_hi > 0
    is_below = f_lo > 0  # Both residuals positive means μ is too small

    # If no sign change, return boundary value
    if same_sign
        return ifelse(is_below, closure.μmax, closure.μmin)
    end

    # Bisection iteration
    for _ in 1:max_iterations
        μ_mid = (μ_lo + μ_hi) / 2
        f_mid = f(μ_mid)

        abs(f_mid) < tolerance && return μ_mid
        (μ_hi - μ_lo) < tolerance * μ_mid && return μ_mid

        # Update bounds
        if f_lo * f_mid < 0
            μ_hi = μ_mid
            f_hi = f_mid
        else
            μ_lo = μ_mid
            f_lo = f_mid
        end
    end

    return (μ_lo + μ_hi) / 2
end

"""
    solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                 mass = IceMassPowerLaw(),
                 closure = P3Closure(),
                 logλ_bounds = (log(10), log(1e7)),
                 max_iterations = 50,
                 tolerance = 1e-10)

Solve for slope parameter λ given ice mass and number concentrations.

Uses the secant method to find logλ such that the computed L/N ratio
matches the observed ratio. This is the two-moment solver using the
μ-λ closure relationship.

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]

# Keyword Arguments
- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Two-moment closure (default: `P3Closure()`)

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                      mass = IceMassPowerLaw(),
                      closure = P3Closure(),
                      shape_relation = nothing,  # deprecated, for backwards compatibility
                      logλ_bounds = (log(10), log(1e7)),
                      max_iterations = 50,
                      tolerance = 1e-10,
                      kwargs...)

    # Handle deprecated keyword
    actual_closure = isnothing(shape_relation) ? closure : shape_relation

    FT = typeof(L_ice)
    if L_ice <= 0 || N_ice <= 0
        # No ice mass or number: return upper bound to avoid unphysical λ = 0.
        return FT(logλ_bounds[2])
    end

    target = log(L_ice) - log(N_ice)
    # Pass L_ice to log_mass_number_ratio for P3 closure diagnostic
    f(logλ) = log_mass_number_ratio(mass, actual_closure, rime_fraction, rime_density, logλ, L_ice) - target

    # Secant method
    x₀, x₁ = FT.(logλ_bounds)
    f₀, f₁ = f(x₀), f(x₁)

    for _ in 1:max_iterations
        Δx = f₁ * (x₁ - x₀) / (f₁ - f₀)
        x₂ = clamp(x₁ - Δx, FT(logλ_bounds[1]), FT(logλ_bounds[2]))

        abs(Δx) < tolerance * abs(x₁) && return x₂

        x₀, f₀ = x₁, f₁
        x₁, f₁ = x₂, f(x₂)
    end

    return x₁
end

"""
    solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ;
                 mass = IceMassPowerLaw(),
                 logλ_bounds = (log(10), log(1e7)),
                 max_iterations = 50,
                 tolerance = 1e-10)

Solve for slope parameter λ given a fixed shape parameter μ (three-moment).

For three-moment ice, μ is determined from the Z/N constraint, so this
function finds λ that satisfies the L/N constraint at that μ.

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment [m⁶/m³] (used for initial guess)
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]
- `μ`: Shape parameter (determined from three-moment solver)

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ;
                      mass = IceMassPowerLaw(),
                      logλ_bounds = (log(10), log(1e7)),
                      max_iterations = 50,
                      tolerance = 1e-10,
                      kwargs...)

    FT = typeof(L_ice)
    if L_ice <= 0 || N_ice <= 0
        # No ice mass or number: return upper bound to avoid unphysical λ = 0.
        return FT(logλ_bounds[2])
    end

    target = log(L_ice) - log(N_ice)

    function f(logλ)
        log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
        log_N_over_N₀ = log_gamma_moment(μ, logλ)
        return (log_L_over_N₀ - log_N_over_N₀) - target
    end

    # Use Z/N constraint for initial guess if Z is available
    if !iszero(Z_ice)
        logλ_guess = log_lambda_from_reflectivity(μ, log(Z_ice) - log(N_ice))
        logλ_guess = clamp(logλ_guess, FT(logλ_bounds[1]), FT(logλ_bounds[2]))
    else
        logλ_guess = (FT(logλ_bounds[1]) + FT(logλ_bounds[2])) / 2
    end

    # Secant method starting from Z/N guess
    x₀ = FT(logλ_bounds[1])
    x₁ = logλ_guess
    f₀, f₁ = f(x₀), f(x₁)

    for _ in 1:max_iterations
        denom = f₁ - f₀
        abs(denom) < eps(FT) && return x₁

        Δx = f₁ * (x₁ - x₀) / denom
        x₂ = clamp(x₁ - Δx, FT(logλ_bounds[1]), FT(logλ_bounds[2]))

        abs(Δx) < tolerance * abs(x₁) && return x₂

        x₀, f₀ = x₁, f₁
        x₁, f₁ = x₂, f(x₂)
    end

    return x₁
end

"""
    intercept_parameter(N_ice, μ, logλ)

Compute N₀ from the normalization: N = N₀ × ∫ D^μ exp(-λD) dD.
"""
function intercept_parameter(N_ice, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return N_ice / exp(log_N_over_N₀)
end

"""
    log_intercept_parameter(N_ice, μ, logλ)

Compute log(N₀) from normalization.
"""
function log_intercept_parameter(N_ice, μ, logλ)
    return log(N_ice) - log_gamma_moment(μ, logλ)
end

"""
    DiameterBounds

Physical bounds on ice particle diameters for the lambda solver.
See [`DiameterBounds()`](@ref) constructor.
"""
struct DiameterBounds{FT}
    D_min :: FT
    D_max :: FT
end

"""
$(TYPEDSIGNATURES)

Construct diameter bounds for the lambda solver.

The P3 scheme constrains the size distribution such that the mean diameter
remains within physical limits. This prevents unphysical distributions with
extremely small or large particles.

For a gamma distribution N'(D) = N₀ D^μ exp(-λD), the mean diameter is:
  D_mean = (μ + 1) / λ

To enforce D_min ≤ D_mean ≤ D_max:
  (μ + 1) / D_max ≤ λ ≤ (μ + 1) / D_min

# Keyword Arguments

- `D_min`: Minimum mean diameter [m], default 2 μm
- `D_max`: Maximum mean diameter [m], default 40 mm

# Example

```julia
bounds = DiameterBounds(; D_min=5e-6, D_max=20e-3)  # 5 μm to 20 mm
```
"""
function DiameterBounds(FT = Float64; D_min = FT(2e-6), D_max = FT(40e-3))
    return DiameterBounds(FT(D_min), FT(D_max))
end

"""
    lambda_bounds_from_diameter(μ, bounds::DiameterBounds)

Compute λ bounds from diameter bounds for a given shape parameter μ.

For D_mean = (μ + 1) / λ:
- λ_min = (μ + 1) / D_max
- λ_max = (μ + 1) / D_min

Returns (λ_min, λ_max).
"""
@inline function lambda_bounds_from_diameter(μ, bounds::DiameterBounds)
    FT = typeof(μ)
    λ_min = (μ + 1) / bounds.D_max
    λ_max = (μ + 1) / bounds.D_min
    return (λ_min, λ_max)
end

"""
    enforce_diameter_bounds(λ, μ, bounds::DiameterBounds)

Clamp λ to ensure the mean diameter stays within physical bounds.

Returns the clamped λ value.
"""
@inline function enforce_diameter_bounds(λ, μ, bounds::DiameterBounds)
    (λ_min, λ_max) = lambda_bounds_from_diameter(μ, bounds)
    return clamp(λ, λ_min, λ_max)
end

"""
    IceDistributionParameters

Result of [`distribution_parameters`](@ref). Fields: `N₀`, `λ`, `μ`.
"""
struct IceDistributionParameters{FT}
    N₀ :: FT
    λ :: FT
    μ :: FT
end

"""
$(TYPEDSIGNATURES)

Solve for gamma size distribution parameters from two prognostic moments (L, N).

This is the two-moment closure for P3: given the prognostic ice mass ``L`` and
number ``N`` concentrations, plus the predicted rime properties, compute
the complete gamma distribution:

```math
N'(D) = N₀ D^μ e^{-λD}
```

The solution proceeds in three steps:

1. **Solve for λ**: Secant method finds the slope parameter satisfying
   the L/N ratio constraint with piecewise m(D)
2. **Compute μ**: Shape parameter from μ-λ relationship
3. **Compute N₀**: Intercept from number normalization

# Arguments

- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `rime_fraction`: Mass fraction of rime [-] (0 = unrimed, 1 = fully rimed)
- `rime_density`: Density of the rime layer [kg/m³]

# Keyword Arguments

- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Two-moment closure (default: `P3Closure()`)

# Returns

[`IceDistributionParameters`](@ref) with fields `N₀`, `λ`, `μ`.

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

# Typical ice cloud conditions
L_ice = 1e-4  # 0.1 g/m³
N_ice = 1e5   # 100,000 particles/m³

params = distribution_parameters(L_ice, N_ice, 0.0, 400.0)
# IceDistributionParameters(N₀=..., λ=..., μ=...)
```

# References

See [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function distribution_parameters(L_ice, N_ice, rime_fraction, rime_density;
                                  mass = IceMassPowerLaw(),
                                  closure = P3Closure(),
                                  diameter_bounds = nothing,
                                  shape_relation = nothing,  # deprecated
                                  kwargs...)
    FT = typeof(L_ice)

    # Handle deprecated keyword
    actual_closure = isnothing(shape_relation) ? closure : shape_relation

    logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density; mass, closure=actual_closure, kwargs...)
    λ = exp(logλ)
    μ = shape_parameter(actual_closure, logλ, L_ice, rime_fraction, rime_density, mass)

    # Enforce diameter bounds if provided
    if !isnothing(diameter_bounds)
        λ = enforce_diameter_bounds(λ, μ, diameter_bounds)
    end

    N₀ = intercept_parameter(N_ice, μ, log(λ))

    return IceDistributionParameters(N₀, λ, μ)
end

"""
$(TYPEDSIGNATURES)

Solve for gamma size distribution parameters from three prognostic moments (L, N, Z).

This is the three-moment solver for P3: given the prognostic ice mass ``L``,
number ``N``, and sixth moment ``Z`` concentrations, compute the complete
gamma distribution without needing an empirical μ-λ closure:

```math
N'(D) = N₀ D^μ e^{-λD}
```

The solution uses:
1. **Z/N constraint**: Determines λ as a function of μ
2. **L/N constraint**: Used to solve for the correct μ
3. **Normalization**: N₀ from the number integral

# Advantages of Three-Moment

- Shape parameter μ evolves physically based on actual size distribution
- Better representation of size sorting during sedimentation
- Improved simulation of hail and large, heavily rimed particles
- No need for empirical μ-λ parameterization

# Arguments

- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment / reflectivity [m⁶/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of the rime layer [kg/m³]

# Keyword Arguments

- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Three-moment closure (default: `ThreeMomentClosure()`)

# Returns

[`IceDistributionParameters`](@ref) with fields `N₀`, `λ`, `μ`.

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

# Ice with reflectivity constraint
L_ice = 1e-4   # 0.1 g/m³
N_ice = 1e5    # 100,000 particles/m³
Z_ice = 1e-12  # Sixth moment [m⁶/m³]

params = distribution_parameters(L_ice, N_ice, Z_ice, 0.0, 400.0)
# IceDistributionParameters(N₀=..., λ=..., μ=...)
```

# References

[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) introduced three-moment ice,
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) refined the approach.
"""
function distribution_parameters(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                                  mass = IceMassPowerLaw(),
                                  closure = ThreeMomentClosure(),
                                  diameter_bounds = nothing,
                                  kwargs...)

    FT = typeof(L_ice)

    # Handle edge cases
    if iszero(N_ice) || iszero(L_ice)
        return IceDistributionParameters(zero(FT), zero(FT), zero(FT))
    end

    # If Z is zero or negative, fall back to two-moment with μ at lower bound
    if Z_ice ≤ 0
        μ = closure.μmin
        logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ; mass, kwargs...)
        λ = exp(logλ)

        # Enforce diameter bounds if provided
        if !isnothing(diameter_bounds)
            λ = enforce_diameter_bounds(λ, μ, diameter_bounds)
        end

        N₀ = intercept_parameter(N_ice, μ, log(λ))
        return IceDistributionParameters(N₀, λ, μ)
    end

    # Solve for μ using three-moment constraint
    μ = solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density; mass, closure, kwargs...)

    # Solve for λ at this μ
    logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ; mass, kwargs...)
    λ = exp(logλ)

    # Enforce diameter bounds if provided
    if !isnothing(diameter_bounds)
        λ = enforce_diameter_bounds(λ, μ, diameter_bounds)
    end

    # Compute N₀ from normalization
    N₀ = intercept_parameter(N_ice, μ, log(λ))

    return IceDistributionParameters(N₀, λ, μ)
end
