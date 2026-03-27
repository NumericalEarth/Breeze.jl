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
- `ice_density`: Pure ice density [kg/m³], default 900

# References

Default parameters from [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization)
supplementary material, based on aircraft observations.
"""
function IceMassPowerLaw(FT = Oceananigans.defaults.FloatType;
                         coefficient = 0.0121,
                         exponent = 1.9,
                         ice_density = 900)
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
    FixedShapeParameter

Fixed shape parameter closure: always returns a constant μ regardless of λ.
Used for tabulation with exponential PSD (μ=0) to match Fortran Table 1.
See [`FixedShapeParameter()`](@ref) constructor.
"""
struct FixedShapeParameter{FT}
    μ :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a fixed shape parameter closure.

This bypasses the empirical μ-λ relationship and uses a constant μ for all λ values.
The primary use case is tabulation: the Fortran P3 Table 1 is generated with μ=0
(exponential PSD), so using `FixedShapeParameter(0)` produces tables that match
the Fortran reference values.

# Keyword Arguments

- `μ`: Fixed shape parameter value, default 0 (exponential distribution)
"""
function FixedShapeParameter(FT::Type{<:AbstractFloat} = Float64; μ = 0)
    return FixedShapeParameter(FT(μ))
end

@inline function shape_parameter(closure::FixedShapeParameter, logλ, args...)
    return closure.μ
end

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
    shape_parameter(closure, logλ, L_ice, rime_fraction, rime_density, liquid_fraction, mass_params)

Compute shape parameter μ.
"""
@inline function shape_parameter(closure::TwoMomentClosure, logλ, args...)
    λ = exp(logλ)
    μ = closure.a * λ^closure.b - closure.c
    return clamp(μ, 0, closure.μmax)
end

@inline function shape_parameter(closure::P3Closure, logλ, L_ice, rime_fraction, rime_density, liquid_fraction, mass::IceMassPowerLaw)
    FT = typeof(closure.a)
    λ = exp(logλ)

    # 1. Compute graupel density (rho_g)
    ρ_dep = deposited_ice_density(mass, rime_fraction, rime_density)
    ρ_g_rimed = graupel_density(rime_fraction, rime_density, ρ_dep)
    ρ_g_dry = ifelse(iszero(rime_fraction), mass.ice_density, ρ_g_rimed)
    # M12: blend liquid water density into bulk density (Fortran diagnostic_mui_Fl)
    ρ_g = (1 - liquid_fraction) * ρ_g_dry + liquid_fraction * FT(1000)

    # 2. Compute D_mvd (Mean Volume Diameter)
    # D_mvd = (L / ((pi/6) * rho_g))^(1/3)
    val = L_ice / (FT(π)/6 * ρ_g)
    D_mvd = ifelse(val <= 0, FT(0), val^(1/3))

    # 3. Compute both regimes, select based on D_mvd
    # Small regime: Heymsfield 2003
    μ_small = closure.a * λ^closure.b - closure.c
    μ_small = clamp(μ_small, 0, closure.μmax_small)

    # Large regime
    D_mvd_mm = D_mvd * 1000
    D_thres_mm = closure.D_threshold * 1000
    f_ρ = max(1, 1 + FT(0.00842) * (ρ_g - 400))
    μ_large = FT(0.25) * (D_mvd_mm - D_thres_mm) * f_ρ * rime_fraction
    μ_large = clamp(μ_large, 0, closure.μmax_large)

    return ifelse(D_mvd <= closure.D_threshold, μ_small, μ_large)
end

#####
##### Three-moment closure: Z/N constraint
#####

abstract type AbstractThreeMomentClosure end

"""
    ThreeMomentLookupClosure

Three-moment closure that uses lookup tables for shape parameter μ and slope
parameter λ. See [`ThreeMomentLookupClosure()`](@ref) constructor.
"""
struct ThreeMomentLookupClosure{TABLE} <: AbstractThreeMomentClosure
    table :: TABLE
end

"""
    ThreeMomentClosure

Fortran-parity three-moment closure using the upstream P3 `solve_mui` approximation.
See [`ThreeMomentClosure()`](@ref) constructor.
"""
struct ThreeMomentClosure{FT} <: AbstractThreeMomentClosure
    μmin :: FT
    μmax :: FT
end

"""
$(TYPEDSIGNATURES)

Construct the Fortran-parity three-moment closure for gamma size distribution.

This closure follows the current upstream P3 implementation: it iterates on bulk
ice density, approximates the third diameter moment as spherical, and applies the
piecewise-polynomial `G(μ)` inversion used by `solve_mui`.

Use this closure when Fortran parity is the priority.

# Keyword Arguments

- `μmin`: Minimum shape parameter, default 0 (exponential distribution)
- `μmax`: Maximum shape parameter, default 20

# References

[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Morrison et al. (2025)](@cite Morrison2025complete3moment).
"""
function ThreeMomentClosure(FT = Oceananigans.defaults.FloatType;
                            μmin = 0,
                            μmax = 20)
    return ThreeMomentClosure(FT(μmin), FT(μmax))
end

"""
    ThreeMomentClosureExact

Three-moment closure that solves the full Breeze moment constraints against the
piecewise mass-diameter relation. See [`ThreeMomentClosureExact()`](@ref) constructor.
"""
struct ThreeMomentClosureExact{FT} <: AbstractThreeMomentClosure
    μmin :: FT
    μmax :: FT
end

"""
$(TYPEDSIGNATURES)

Construct the exact three-moment closure for gamma size distribution.

With three prognostic moments (mass L, number N, and reflectivity Z), the shape
parameter μ is diagnosed by solving the full Breeze mass and reflectivity
constraints using the same piecewise mass-diameter relation employed elsewhere
in the P3 implementation.

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
function ThreeMomentClosureExact(FT = Oceananigans.defaults.FloatType;
                                 μmin = 0,
                                 μmax = 20)
    return ThreeMomentClosureExact(FT(μmin), FT(μmax))
end

#####
##### Diameter thresholds between particle regimes
#####

"""
    regime_threshold(α, β, ρ)

Diameter threshold from mass power law: D = (6α / πρ)^(1/(3-β))

Used to determine boundaries between spherical ice, aggregates, and graupel.
"""
@inline function regime_threshold(α, β, ρ)
    FT = typeof(α)
    return (6 * α / (FT(π) * ρ))^(1 / (3 - β))
end

"""
    deposited_ice_density(mass, rime_fraction, rime_density)

Density of the vapor-deposited (unrimed) portion of ice particles.
Equation 16 in [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
@inline function deposited_ice_density(mass::IceMassPowerLaw, rime_fraction, rime_density)
    β = mass.exponent
    Fᶠ = rime_fraction
    ρᶠ = rime_density
    FT = typeof(β)

    # Compute rimed density (use safe Fᶠ to avoid division by zero)
    Fᶠ_safe = max(Fᶠ, eps(FT))
    k = (1 - Fᶠ_safe)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ_safe
    den = (β - 2) * (k - 1) / ((1 - Fᶠ_safe) * k - 1) - (1 - Fᶠ_safe)
    ρ_dep_rimed = num / max(den, eps(FT))

    # Return ice_density for unrimed case, computed density otherwise
    return ifelse(Fᶠ <= eps(FT), mass.ice_density, ρ_dep_rimed)
end

"""
    graupel_density(rime_fraction, rime_density, deposited_density)

Bulk density of graupel particles (rime + deposited ice).
"""
@inline function graupel_density(rime_fraction, rime_density, deposited_density)
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
@inline function ice_regime_thresholds(mass::IceMassPowerLaw, rime_fraction, rime_density)
    # NOTE (M3): This duplicates regime_thresholds_from_state() in quadrature.jl.
    # Both must produce identical thresholds. If one is changed, update the other.
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
    # Note: same logic and ordering as particle_mass_ice_only in quadrature.jl
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
    log_mass_number_ratio(mass, closure, rime_fraction, rime_density, liquid_fraction, logλ, L_ice)

Compute log(L_ice / N_ice) as a function of logλ for two-moment closure.
Includes L_ice argument to support the P3Closure diagnostic.
"""
function log_mass_number_ratio(mass::IceMassPowerLaw,
                               closure,
                               rime_fraction, rime_density, liquid_fraction, logλ, L_ice)
    μ = shape_parameter(closure, logλ, L_ice, rime_fraction, rime_density, liquid_fraction, mass)
    log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return log_L_over_N₀ - log_N_over_N₀
end

#####
##### Lambda solver (three-moment)
#####

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
    shape_parameter_from_moments(mom0, mom3, mom6, μmax)

Approximate the three-moment ice shape parameter using the Fortran P3 `G(μ)` fit.

This matches `compute_mu_3mom_1` in the reference P3 Fortran code:
it forms ``G = M₀ M₆ / M₃²`` and applies the piecewise polynomial inversion
used by `solve_mui`.
"""
@inline function shape_parameter_from_moments(mom0, mom3, mom6, μmax)
    FT = promote_type(typeof(mom0), typeof(mom3), typeof(mom6), typeof(μmax))
    eps_m3 = FT(1e-20)

    mom3 <= eps_m3 && return FT(μmax)

    G = (mom0 / mom3) * (mom6 / mom3)
    G² = G * G

    μ = if G >= FT(20)
        FT(0)
    elseif G >= FT(13.31)
        FT(3.3638e-3) * G² - FT(1.7152e-1) * G + FT(2.0857)
    elseif G >= FT(7.123)
        FT(1.5900e-2) * G² - FT(4.8202e-1) * G + FT(4.0108)
    elseif G >= FT(4.2)
        FT(1.0730e-1) * G² - FT(1.7481) * G + FT(8.4246)
    elseif G >= FT(2.946)
        FT(5.9070e-1) * G² - FT(5.7918) * G + FT(16.919)
    elseif G >= FT(1.793)
        FT(4.3966) * G² - FT(26.659) * G + FT(45.477)
    elseif G >= FT(1.472)
        FT(47.552) * G² - FT(179.58) * G + FT(181.26)
    else
        FT(μmax)
    end

    return min(max(μ, FT(0)), FT(μmax))
end

"""
    mass_residual_three_moment(mass, rime_fraction, rime_density, μ, log_Z_over_N, log_L_over_N)

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
    g_of_mu(μ)

Compute G(μ) = Γ(μ+7)Γ(μ+1) / Γ(μ+4)² for the three-moment μ-Z constraint.
Simplifies to (μ+6)(μ+5)(μ+4) / ((μ+3)(μ+2)(μ+1)).
Matches Fortran `G_of_mu`.
"""
@inline function g_of_mu(μ)
    return (μ + 6) * (μ + 5) * (μ + 4) / ((μ + 3) * (μ + 2) * (μ + 1))
end

"""
    enforce_z_bounds(Z_ice, L_ice, N_ice, ρ_bulk, μmin, μmax)

Bound Z_ice to a physically consistent range based on the μ bounds.
Matches Fortran `apply_mui_bounds_to_zi` and basic zsmall/zlarge clamps.
"""
@inline function enforce_z_bounds(Z_ice, L_ice, N_ice, ρ_bulk, μmin, μmax)
    FT = typeof(Z_ice)
    # Basic magnitude bounds (Fortran zsmall/zlarge)
    Z_clamped = clamp(Z_ice, FT(1e-35), FT(1))

    # Moment-based bounds: G(μ_max) × mom3²/N ≤ Z ≤ G(μ_min) × mom3²/N
    mom3 = FT(6) * L_ice / (FT(π) * max(ρ_bulk, eps(FT)))
    tmp = mom3^2 / max(N_ice, eps(FT))

    G_min = g_of_mu(μmin)  # upper Z bound (wide distribution)
    G_max = g_of_mu(μmax)  # lower Z bound (narrow distribution)

    Z_clamped = min(Z_clamped, G_min * tmp)
    Z_clamped = max(Z_clamped, G_max * tmp)

    return Z_clamped
end

"""
    solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                          mass = IceMassPowerLaw(),
                          closure = ThreeMomentClosure(),
                          max_iterations = nothing,
                          tolerance = nothing,
                          density_quadrature_points = 64)

Solve for shape parameter μ using the selected three-moment closure.

Supported closures:
- `ThreeMomentClosure()`: Fortran-parity `solve_mui` iteration
- `ThreeMomentClosureExact()`: full Breeze residual solve

Closure-specific defaults:
- `ThreeMomentClosure`: `max_iterations = 5`, `tolerance = 0.25`
- `ThreeMomentClosureExact`: `max_iterations = 50`, `tolerance = 1e-10`

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `Z_ice`: Ice sixth moment / reflectivity [m⁶/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]
- `liquid_fraction`: Liquid water fraction from partial melting [-] (H15)

# Returns
- Shape parameter μ
"""
function solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                               liquid_fraction = zero(typeof(L_ice)),
                               mass = IceMassPowerLaw(),
                               closure = ThreeMomentClosure(),
                               max_iterations = nothing,
                               tolerance = nothing,
                               density_quadrature_points = 64)
    return solve_shape_parameter_with_closure(closure, L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                                              liquid_fraction, mass, max_iterations, tolerance, density_quadrature_points)
end

function solve_shape_parameter_with_closure(closure::ThreeMomentClosure,
                                            L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                                            liquid_fraction = zero(typeof(L_ice)),
                                            mass = IceMassPowerLaw(),
                                            max_iterations = nothing,
                                            tolerance = nothing,
                                            density_quadrature_points = 64)
    FT = typeof(L_ice)
    max_iterations = isnothing(max_iterations) ? 5 : max_iterations
    tolerance = FT(isnothing(tolerance) ? 0.25 : tolerance)

    # Handle edge cases
    (iszero(N_ice) || iszero(L_ice) || iszero(Z_ice)) && return closure.μmin

    nodes, weights = chebyshev_gauss_nodes_weights(FT, density_quadrature_points)
    μ_old = clamp(FT(0.5), closure.μmin, closure.μmax)
    μ = μ_old

    # M15: enforce Z bounds before solve (Fortran apply_mui_bounds_to_zi)
    logλ_init = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ_old; mass)
    N₀_init = intercept_parameter(N_ice, μ_old, logλ_init)
    state_init = IceSizeDistributionState(FT;
        intercept = N₀_init, shape = μ_old, slope = exp(logλ_init),
        rime_fraction = rime_fraction, rime_density = rime_density,
        liquid_fraction = liquid_fraction,
        mass_coefficient = mass.coefficient, mass_exponent = mass.exponent,
        ice_density = mass.ice_density)
    ρ_bulk_dry = max(evaluate_quadrature(MeanDensity(), state_init, nodes, weights), eps(FT))
    # H15: Blend liquid water density (Fortran: rhom = (1-Fl)*cgp(i_rhor) + Fl*1000*π/6)
    ρ_bulk_init = (1 - liquid_fraction) * ρ_bulk_dry + liquid_fraction * FT(1000)
    ρ_bulk_init = max(ρ_bulk_init, eps(FT))
    Z_ice = enforce_z_bounds(Z_ice, L_ice, N_ice, ρ_bulk_init, closure.μmin, closure.μmax)

    for _ in 1:max_iterations
        logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ_old; mass)
        λ = exp(logλ)
        N₀ = intercept_parameter(N_ice, μ_old, logλ)

        state = IceSizeDistributionState(FT;
            intercept = N₀,
            shape = μ_old,
            slope = λ,
            rime_fraction = rime_fraction,
            rime_density = rime_density,
            liquid_fraction = liquid_fraction,
            mass_coefficient = mass.coefficient,
            mass_exponent = mass.exponent,
            ice_density = mass.ice_density)

        ρ_bulk_dry = max(evaluate_quadrature(MeanDensity(), state, nodes, weights), eps(FT))
        # H15: Blend liquid water density into bulk density diagnostic
        ρ_bulk = (1 - liquid_fraction) * ρ_bulk_dry + liquid_fraction * FT(1000)
        ρ_bulk = max(ρ_bulk, eps(FT))
        mom3 = FT(6) * L_ice / (ρ_bulk * FT(π))
        μ = shape_parameter_from_moments(N_ice, mom3, Z_ice, closure.μmax)
        μ = clamp(μ, closure.μmin, closure.μmax)

        abs(μ_old - μ) < tolerance && return μ
        μ_old = μ
    end

    return μ
end

function solve_shape_parameter_with_closure(closure::ThreeMomentClosureExact,
                                            L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                                            liquid_fraction = zero(typeof(L_ice)),
                                            mass = IceMassPowerLaw(),
                                            max_iterations = nothing,
                                            tolerance = nothing,
                                            density_quadrature_points = 64)
    FT = typeof(L_ice)
    max_iterations = isnothing(max_iterations) ? 50 : max_iterations
    tolerance = FT(isnothing(tolerance) ? 1e-10 : tolerance)

    (iszero(N_ice) || iszero(L_ice) || iszero(Z_ice)) && return closure.μmin

    # M15: enforce Z bounds before solve
    # Use ice density as a conservative estimate for moment-based Z bounding
    ρ_estimate = FT(mass.ice_density)
    Z_ice = enforce_z_bounds(Z_ice, L_ice, N_ice, ρ_estimate, closure.μmin, closure.μmax)

    log_Z_over_N = log(Z_ice) - log(N_ice)
    log_L_over_N = log(L_ice) - log(N_ice)

    f(μ) = mass_residual_three_moment(mass, rime_fraction, rime_density,
                                      μ, log_Z_over_N, log_L_over_N)

    μ_lo = closure.μmin
    μ_hi = closure.μmax
    f_lo = f(μ_lo)
    f_hi = f(μ_hi)

    same_sign = f_lo * f_hi > 0
    is_below = f_lo > 0

    if same_sign
        return ifelse(is_below, closure.μmax, closure.μmin)
    end

    for _ in 1:max_iterations
        μ_mid = (μ_lo + μ_hi) / 2
        f_mid = f(μ_mid)

        abs(f_mid) < tolerance && return μ_mid
        (μ_hi - μ_lo) < tolerance * max(μ_mid, one(FT)) && return μ_mid

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
                 liquid_fraction = zero(typeof(L_ice)),
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
- `liquid_fraction`: Liquid water fraction [-] (default 0)
- `mass`: Power law parameters (default: `IceMassPowerLaw()`)
- `closure`: Two-moment closure (default: `P3Closure()`)

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                      liquid_fraction = zero(typeof(L_ice)),
                      mass = IceMassPowerLaw(),
                      closure = P3Closure(),
                      logλ_bounds = (log(10), log(1e7)),
                      max_iterations = 50,
                      tolerance = 1e-10)

    FT = typeof(L_ice)
    if L_ice <= 0 || N_ice <= 0
        # No ice mass or number: return upper bound to avoid unphysical λ = 0.
        return FT(logλ_bounds[2])
    end

    target = log(L_ice) - log(N_ice)
    # Pass L_ice to log_mass_number_ratio for P3 closure diagnostic
    f(logλ) = log_mass_number_ratio(mass, closure, rime_fraction, rime_density, liquid_fraction, logλ, L_ice) - target

    # Secant method
    x₀, x₁ = FT.(logλ_bounds)
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
                      tolerance = 1e-10)

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
    DiameterBounds

Physical bounds on ice particle diameters for the lambda solver.
See [`DiameterBounds()`](@ref) constructor.
"""
struct DiameterBounds{FT}
    D_min :: FT
    D_max :: FT
end

# Fortran P3 lambda limiter constants (create_p3_lookupTable_3.f90, lines 77-79)
const P3_DM_MAX_BASE = 5e-3    # 5 mm  (Fortran Dm_max1 = 5000e-6)
const P3_DM_MAX_RIME = 20e-3   # 20 mm (Fortran Dm_max2 = 20000e-6)
const P3_DM_MIN      = 2e-6    # 2 μm  (Fortran Dm_min  = 2e-6)

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
$(TYPEDSIGNATURES)

Construct Fr-dependent diameter bounds matching the Fortran P3 lambda limiter.

The maximum mean diameter depends on rime fraction Fr:
  D_max = 5 mm + 20 mm × Fr²

This ranges from 5 mm (unrimed, Fr=0) to 25 mm (fully rimed, Fr=1),
matching `create_p3_lookupTable_3.f90` lines 313-315. The previous fixed
default of 40 mm was too permissive for unrimed ice.

# Arguments

- `FT`: Float type
- `rime_fraction`: Rime mass fraction Fr ∈ [0, 1]
"""
@inline function DiameterBounds(FT, rime_fraction)
    D_min = FT(P3_DM_MIN)
    D_max = FT(P3_DM_MAX_BASE) + FT(P3_DM_MAX_RIME) * rime_fraction^2
    return DiameterBounds(D_min, D_max)
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
                                  liquid_fraction = zero(typeof(L_ice)),
                                  mass = IceMassPowerLaw(),
                                  closure = P3Closure(),
                                  diameter_bounds = nothing)
    FT = typeof(L_ice)

    logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density; liquid_fraction, mass, closure)
    λ = exp(logλ)
    μ = shape_parameter(closure, logλ, L_ice, rime_fraction, rime_density, liquid_fraction, mass)

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
                                  liquid_fraction = zero(typeof(L_ice)),
                                  mass = IceMassPowerLaw(),
                                  closure = ThreeMomentClosure(),
                                  diameter_bounds = nothing)

    FT = typeof(L_ice)

    # Handle edge cases
    if iszero(N_ice) || iszero(L_ice)
        return IceDistributionParameters(zero(FT), zero(FT), zero(FT))
    end

    # If Z is zero or negative, fall back to two-moment with μ at lower bound
    if Z_ice ≤ 0
        μ = closure.μmin
        logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ; mass)
        λ = exp(logλ)

        # Enforce diameter bounds if provided
        if !isnothing(diameter_bounds)
            λ = enforce_diameter_bounds(λ, μ, diameter_bounds)
        end

        N₀ = intercept_parameter(N_ice, μ, log(λ))
        return IceDistributionParameters(N₀, λ, μ)
    end

    # H15: Solve for μ using three-moment constraint with liquid fraction
    μ = solve_shape_parameter(L_ice, N_ice, Z_ice, rime_fraction, rime_density;
                              liquid_fraction, mass, closure)

    # Solve for λ at this μ
    logλ = solve_lambda(L_ice, N_ice, Z_ice, rime_fraction, rime_density, μ; mass)
    λ = exp(logλ)

    # Enforce diameter bounds if provided
    if !isnothing(diameter_bounds)
        λ = enforce_diameter_bounds(λ, μ, diameter_bounds)
    end

    # Compute N₀ from normalization
    N₀ = intercept_parameter(N_ice, μ, log(λ))

    return IceDistributionParameters(N₀, λ, μ)
end
