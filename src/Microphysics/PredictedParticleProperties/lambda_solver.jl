#####
##### Lambda Solver for P3 Ice Size Distribution
#####
##### Given prognostic moments (L_ice, N_ice) and ice properties (rime fraction, rime density),
##### solve for the gamma distribution parameters (N₀, λ, μ).
#####
##### The solver handles the piecewise mass-diameter relationship with four regimes
##### from Morrison & Milbrandt (2015a) Equations 1-5. The μ-λ relationship is from
##### Morrison & Milbrandt (2015a) Equation 27, based on Field et al. (2007) observations.
#####
##### For three-moment ice (Milbrandt et al. 2021, 2024), the sixth moment Z can provide
##### an additional constraint to determine μ independently of the μ-λ relationship.
##### This is a TODO for future implementation.
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

"""
    ShapeParameterRelation

μ-λ closure for two-moment PSD. See [`ShapeParameterRelation()`](@ref) constructor.
"""
struct ShapeParameterRelation{FT}
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

# Three-Moment Alternative

With three-moment ice (tracking reflectivity Z), μ can be diagnosed
independently from the Z/N ratio, making this closure unnecessary.
See [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021).

# Keyword Arguments

- `a`: Coefficient in μ = a λ^b - c, default 0.00191
- `b`: Exponent in μ = a λ^b - c, default 0.8
- `c`: Offset in μ = a λ^b - c, default 2
- `μmax`: Maximum shape parameter, default 6

# References

From [Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 27,
based on [Field et al. (2007)](@cite FieldEtAl2007) observations.
"""
function ShapeParameterRelation(FT = Oceananigans.defaults.FloatType;
                                 a = 0.00191,
                                 b = 0.8,
                                 c = 2,
                                 μmax = 6)
    return ShapeParameterRelation(FT(a), FT(b), FT(c), FT(μmax))
end

"""
    shape_parameter(relation, logλ)

Compute μ from log(λ) using the power law relationship.
"""
function shape_parameter(relation::ShapeParameterRelation, logλ)
    λ = exp(logλ)
    μ = relation.a * λ^relation.b - relation.c
    return clamp(μ, zero(μ), relation.μmax)
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
Equation 16 in [Morrison2015parameterization](@citet).
"""
function deposited_ice_density(mass::IceMassPowerLaw, rime_fraction, rime_density)
    β = mass.exponent
    Fᶠ = rime_fraction
    ρᶠ = rime_density
    
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
    
    # For unrimed ice, only the spherical threshold matters
    if iszero(Fᶠ)
        return IceRegimeThresholds(D_spherical, FT(Inf), FT(Inf), ρᵢ)
    end
    
    ρ_dep = deposited_ice_density(mass, Fᶠ, ρᶠ)
    ρ_g = graupel_density(Fᶠ, ρᶠ, ρ_dep)
    
    D_graupel = regime_threshold(α, β, ρ_g)
    D_partial = regime_threshold(α, β, ρ_g * (1 - Fᶠ))
    
    return IceRegimeThresholds(D_spherical, D_graupel, D_partial, ρ_g)
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
    
    if D < thresholds.spherical
        return (ρᵢ * FT(π) / 6, FT(3))
    elseif iszero(Fᶠ) || D < thresholds.graupel
        return (FT(α), FT(β))
    elseif D < thresholds.partial_rime
        return (thresholds.ρ_graupel * FT(π) / 6, FT(3))
    else
        return (FT(α) / (1 - Fᶠ), FT(β))
    end
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
    a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))
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
    
    if iszero(Fᶠ)
        # Unrimed: aggregates [D_spherical, ∞)
        log_M₂ = log_gamma_inc_moment(thresholds.spherical, FT(Inf), μ, logλ; k = β + n, scale = α)
        return logaddexp(log_M₁, log_M₂)
    end
    
    # Regime 2: rimed aggregates [D_spherical, D_graupel)
    log_M₂ = log_gamma_inc_moment(thresholds.spherical, thresholds.graupel, μ, logλ; k = β + n, scale = α)
    
    # Regime 3: graupel [D_graupel, D_partial)
    a₃ = thresholds.ρ_graupel * FT(π) / 6
    log_M₃ = log_gamma_inc_moment(thresholds.graupel, thresholds.partial_rime, μ, logλ; k = 3 + n, scale = a₃)
    
    # Regime 4: partially rimed [D_partial, ∞)
    a₄ = α / (1 - Fᶠ)
    log_M₄ = log_gamma_inc_moment(thresholds.partial_rime, FT(Inf), μ, logλ; k = β + n, scale = a₄)
    
    return logaddexp(logaddexp(log_M₁, log_M₂), logaddexp(log_M₃, log_M₄))
end

#####
##### Lambda solver
#####

"""
    log_mass_number_ratio(mass, shape_relation, rime_fraction, rime_density, logλ)

Compute log(L_ice / N_ice) as a function of logλ.
"""
function log_mass_number_ratio(mass::IceMassPowerLaw, 
                                shape_relation::ShapeParameterRelation, 
                                rime_fraction, rime_density, logλ)
    μ = shape_parameter(shape_relation, logλ)
    log_L_over_N₀ = log_mass_moment(mass, rime_fraction, rime_density, μ, logλ)
    log_N_over_N₀ = log_gamma_moment(μ, logλ)
    return log_L_over_N₀ - log_N_over_N₀
end

"""
    solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                 mass = IceMassPowerLaw(),
                 shape_relation = ShapeParameterRelation(),
                 logλ_bounds = (log(10), log(1e7)),
                 max_iterations = 50,
                 tolerance = 1e-10)

Solve for slope parameter λ given ice mass and number concentrations.

Uses the secant method to find logλ such that the computed L/N ratio
matches the observed ratio.

# Arguments
- `L_ice`: Ice mass concentration [kg/m³]
- `N_ice`: Ice number concentration [1/m³]
- `rime_fraction`: Mass fraction of rime [-]
- `rime_density`: Density of rime [kg/m³]

# Returns
- `logλ`: Log of slope parameter
"""
function solve_lambda(L_ice, N_ice, rime_fraction, rime_density;
                      mass = IceMassPowerLaw(),
                      shape_relation = ShapeParameterRelation(),
                      logλ_bounds = (log(10), log(1e7)),
                      max_iterations = 50,
                      tolerance = 1e-10)
    
    FT = typeof(L_ice)
    (iszero(N_ice) || iszero(L_ice)) && return log(zero(FT))
    
    target = log(L_ice) - log(N_ice)
    f(logλ) = log_mass_number_ratio(mass, shape_relation, rime_fraction, rime_density, logλ) - target
    
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

Solve for gamma size distribution parameters from prognostic moments.

This is the core closure for P3: given the prognostic ice mass ``L`` and 
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
- `shape_relation`: μ-λ relationship (default: `ShapeParameterRelation()`)

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
                                  shape_relation = ShapeParameterRelation(),
                                  kwargs...)
    logλ = solve_lambda(L_ice, N_ice, rime_fraction, rime_density; mass, shape_relation, kwargs...)
    λ = exp(logλ)
    μ = shape_parameter(shape_relation, logλ)
    N₀ = intercept_parameter(N_ice, μ, logλ)
    
    return IceDistributionParameters(N₀, λ, μ)
end
