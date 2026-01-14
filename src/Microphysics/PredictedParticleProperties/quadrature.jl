#####
##### Quadrature Evaluation of P3 Integrals
#####
##### Numerical integration over the ice size distribution using
##### Chebyshev-Gauss quadrature on a transformed domain.
#####

export evaluate, chebyshev_gauss_nodes_weights

#####
##### Chebyshev-Gauss quadrature
#####

"""
    chebyshev_gauss_nodes_weights(FT, n)

Compute Chebyshev-Gauss quadrature nodes and weights for n points.

Chebyshev-Gauss quadrature is particularly well-suited for smooth
integrands over unbounded domains after transformation. The nodes
cluster near the boundaries, which helps capture rapidly-varying
contributions near D = 0.

Returns `(nodes, weights)` for approximating:

```math
∫_{-1}^{1} f(x) dx ≈ ∑ᵢ wᵢ f(xᵢ)
```

These are then transformed to diameter space using [`transform_to_diameter`](@ref).
"""
function chebyshev_gauss_nodes_weights(FT::Type{<:AbstractFloat}, n::Int)
    nodes = zeros(FT, n)
    weights = fill(FT(π / n), n)
    
    for i in 1:n
        nodes[i] = cos(FT((2i - 1) * π / (2n)))
    end
    
    return nodes, weights
end

chebyshev_gauss_nodes_weights(n::Int) = chebyshev_gauss_nodes_weights(Float64, n)

#####
##### Domain transformation
#####
##### Transform from x ∈ [-1, 1] to D ∈ [0, ∞) using exponential mapping
#####

"""
    transform_to_diameter(x, λ; scale=10)

Transform Chebyshev node x ∈ [-1, 1] to diameter D ∈ [0, ∞).

Uses the mapping:
```math
D = \\frac{s}{\\lambda} \\cdot \\frac{1 + x}{1 - x + \\epsilon}
```

where s is a scale factor (default 10) that controls the integration range
relative to the characteristic size 1/λ.
"""
@inline function transform_to_diameter(x, λ; scale=10)
    ε = eps(typeof(x))
    return scale / λ * (1 + x) / (1 - x + ε)
end

"""
    jacobian_diameter_transform(x, λ; scale=10)

Jacobian dD/dx for the diameter transformation.
"""
@inline function jacobian_diameter_transform(x, λ; scale=10)
    ε = eps(typeof(x))
    denom = (1 - x + ε)^2
    return scale / λ * 2 / denom
end

#####
##### Generic integration interface
#####

"""
    evaluate(integral, state; n_quadrature=64)

Evaluate a P3 integral over the ice size distribution using quadrature.

This is the core numerical integration routine for computing bulk properties
and process rates from the gamma size distribution. Each integral type
dispatches to its own `integrand` function.

**Algorithm:**

1. Generate Chebyshev-Gauss nodes on [-1, 1]
2. Transform to diameter space D ∈ [0, ∞) using exponential mapping
3. Evaluate integrand at each quadrature point
4. Sum weighted contributions with Jacobian correction

# Arguments

- `integral`: Integral type (e.g., `MassWeightedFallSpeed()`)
- `state`: [`IceSizeDistributionState`](@ref) with N₀, μ, λ and rime properties
- `n_quadrature`: Number of quadrature points (default 64, sufficient for most integrals)

# Returns

The evaluated integral value with the same floating-point type as `state.slope`.

# Example

```julia
using Breeze.Microphysics.PredictedParticleProperties

state = IceSizeDistributionState(Float64; intercept=1e6, shape=0.0, slope=1000.0)
Vn = evaluate(NumberWeightedFallSpeed(), state)
```
"""
function evaluate(integral::AbstractP3Integral, state::IceSizeDistributionState; 
                  n_quadrature::Int = 64)
    FT = typeof(state.slope)
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_quadrature)
    
    λ = state.slope
    result = zero(FT)
    
    for i in 1:n_quadrature
        x = nodes[i]
        w = weights[i]
        
        D = transform_to_diameter(x, λ)
        J = jacobian_diameter_transform(x, λ)
        
        # Compute integrand at this diameter
        f = integrand(integral, D, state)
        
        result += w * f * J
    end
    
    return result
end

#####
##### Integrand functions for each integral type
#####

# Default fallback
integrand(::AbstractP3Integral, D, state) = zero(D)

#####
##### Fall speed integrals
#####

"""
Terminal velocity V(D) for ice particles.

Follows power law: V(D) = a_V * D^b_V

with adjustments for particle regime (small ice, unrimed, rimed, graupel).
"""
@inline function terminal_velocity(D, state; 
                                    a_V = 11.72, 
                                    b_V = 0.41)
    # Simplified power law for now
    # Full P3 uses regime-dependent coefficients
    FT = typeof(D)
    return FT(a_V) * D^FT(b_V)
end

# Number-weighted fall speed: ∫ V(D) N'(D) dD
@inline function integrand(::NumberWeightedFallSpeed, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    Np = size_distribution(D, state)
    return V * Np
end

# Mass-weighted fall speed: ∫ V(D) m(D) N'(D) dD
@inline function integrand(::MassWeightedFallSpeed, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    m = particle_mass(D, state)
    Np = size_distribution(D, state)
    return V * m * Np
end

# Reflectivity-weighted fall speed: ∫ V(D) D^6 N'(D) dD
@inline function integrand(::ReflectivityWeightedFallSpeed, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    Np = size_distribution(D, state)
    return V * D^6 * Np
end

#####
##### Particle mass
#####

"""
Particle mass m(D) as a function of diameter.

The mass-dimension relationship depends on the particle regime:
- Small spherical ice: m = (π/6) ρⁱ D³
- Unrimed aggregates: m = α D^β
- Partially rimed: interpolation
- Fully rimed (graupel): m = (π/6) ρᶠ D³
"""
@inline function particle_mass(D, state::IceSizeDistributionState)
    # Simplified form using effective density
    # Full P3 uses regime-dependent formulation
    FT = typeof(D)
    ρⁱ = FT(917)  # kg/m³, pure ice density
    ρᶠ = state.rime_density
    Fᶠ = state.rime_fraction
    
    # Effective density: interpolate between ice and rime
    ρ_eff = (1 - Fᶠ) * ρⁱ * FT(0.1) + Fᶠ * ρᶠ  # 0.1 factor for aggregate density
    
    return FT(π) / 6 * ρ_eff * D^3
end

#####
##### Deposition/ventilation integrals
#####

"""
Ventilation factor ``fᵛᵉ`` for vapor diffusion enhancement.

Following Hall and Pruppacher (1976):
- For D ≤ 100 μm: fᵛᵉ = 1.0
- For D > 100 μm: fᵛᵉ = 0.65 + 0.44 * (V*D)^0.5
"""
@inline function ventilation_factor(D, state; constant_term=true)
    V = terminal_velocity(D, state)
    D_threshold = typeof(D)(100e-6)
    is_small = D ≤ D_threshold
    
    # Small particles: constant_term → 1, otherwise → 0
    small_value = ifelse(constant_term, one(D), zero(D))
    # Large particles: constant_term → 0.65, otherwise → 0.44 * sqrt(V * D)
    large_value = ifelse(constant_term, typeof(D)(0.65), typeof(D)(0.44) * sqrt(V * D))
    
    return ifelse(is_small, small_value, large_value)
end

# Basic ventilation: ∫ fᵛᵉ(D) C(D) N'(D) dD
@inline function integrand(::Ventilation, D, state::IceSizeDistributionState)
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    return fᵛᵉ * C * Np
end

@inline function integrand(::VentilationEnhanced, D, state::IceSizeDistributionState)
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    return fᵛᵉ * C * Np
end

# Size-regime-specific ventilation for melting
@inline function integrand(::SmallIceVentilationConstant, D, state::IceSizeDistributionState)
    D_crit = critical_diameter_small_ice(state.rime_fraction)
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::SmallIceVentilationReynolds, D, state::IceSizeDistributionState)
    D_crit = critical_diameter_small_ice(state.rime_fraction)
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationConstant, D, state::IceSizeDistributionState)
    D_crit = critical_diameter_small_ice(state.rime_fraction)
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationReynolds, D, state::IceSizeDistributionState)
    D_crit = critical_diameter_small_ice(state.rime_fraction)
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

"""
Capacitance C(D) for vapor diffusion.

For spheres: C = D/2
For non-spherical particles: C ≈ 0.48 * D (plates/dendrites)
"""
@inline function capacitance(D, state::IceSizeDistributionState)
    D_crit = critical_diameter_small_ice(state.rime_fraction)
    sphere_capacitance = D / 2
    nonspherical_capacitance = typeof(D)(0.48) * D
    return ifelse(D ≤ D_crit, sphere_capacitance, nonspherical_capacitance)
end

#####
##### Bulk property integrals
#####

# Effective radius: ∫ D³ N'(D) dD / ∫ D² N'(D) dD
# (computed as ratio of two integrals - here we return numerator)
@inline function integrand(::EffectiveRadius, D, state::IceSizeDistributionState)
    Np = size_distribution(D, state)
    return D^3 * Np
end

# Mean diameter: ∫ D m(D) N'(D) dD
@inline function integrand(::MeanDiameter, D, state::IceSizeDistributionState)
    m = particle_mass(D, state)
    Np = size_distribution(D, state)
    return D * m * Np
end

# Mean density: ∫ ρ(D) m(D) N'(D) dD
@inline function integrand(::MeanDensity, D, state::IceSizeDistributionState)
    m = particle_mass(D, state)
    ρ = particle_density(D, state)
    Np = size_distribution(D, state)
    return ρ * m * Np
end

# Reflectivity: ∫ D^6 N'(D) dD
@inline function integrand(::Reflectivity, D, state::IceSizeDistributionState)
    Np = size_distribution(D, state)
    return D^6 * Np
end

# Slope parameter λ - diagnostic, not an integral
@inline integrand(::SlopeParameter, D, state::IceSizeDistributionState) = zero(D)

# Shape parameter μ - diagnostic, not an integral
@inline integrand(::ShapeParameter, D, state::IceSizeDistributionState) = zero(D)

# Shedding rate: integral over particles above melting threshold
@inline function integrand(::SheddingRate, D, state::IceSizeDistributionState)
    m = particle_mass(D, state)
    Np = size_distribution(D, state)
    Fˡ = state.liquid_fraction
    return Fˡ * m * Np  # Simplified: liquid fraction times mass
end

"""
Particle density ρ(D) as a function of diameter.
"""
@inline function particle_density(D, state::IceSizeDistributionState)
    FT = typeof(D)
    ρⁱ = FT(917)  # kg/m³, pure ice density
    Fᶠ = state.rime_fraction
    ρᶠ = state.rime_density
    
    # Effective density: interpolate
    return (1 - Fᶠ) * ρⁱ * FT(0.1) + Fᶠ * ρᶠ
end

#####
##### Collection integrals
#####

# Aggregation number: ∫∫ K(D₁,D₂) N'(D₁) N'(D₂) dD₁ dD₂
# Simplified single integral form
@inline function integrand(::AggregationNumber, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)
    return V * A * Np^2  # Simplified self-collection
end

# Rain collection by ice
@inline function integrand(::RainCollectionNumber, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)
    return V * A * Np
end

"""
Particle cross-sectional area A(D).
"""
@inline function particle_area(D, state::IceSizeDistributionState)
    FT = typeof(D)
    return FT(π) / 4 * D^2  # Simplified: sphere
end

#####
##### Sixth moment integrals
#####

# Sixth moment rime tendency
@inline function integrand(::SixthMomentRime, D, state::IceSizeDistributionState)
    Np = size_distribution(D, state)
    return D^6 * Np
end

# Sixth moment deposition tendencies
@inline function integrand(::SixthMomentDeposition, D, state::IceSizeDistributionState)
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    return 6 * D^5 * fᵛᵉ * C * Np
end

@inline function integrand(::SixthMomentDeposition1, D, state::IceSizeDistributionState)
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    return 6 * D^5 * fᵛᵉ * C * Np
end

# Sixth moment melting tendencies
@inline function integrand(::SixthMomentMelt1, D, state::IceSizeDistributionState)
    Np = size_distribution(D, state)
    return 6 * D^5 * Np
end

@inline function integrand(::SixthMomentMelt2, D, state::IceSizeDistributionState)
    m = particle_mass(D, state)
    Np = size_distribution(D, state)
    return D^6 / m * Np
end

# Sixth moment aggregation
@inline function integrand(::SixthMomentAggregation, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)
    return D^6 * V * A * Np^2
end

# Sixth moment shedding
@inline function integrand(::SixthMomentShedding, D, state::IceSizeDistributionState)
    Np = size_distribution(D, state)
    Fˡ = state.liquid_fraction
    return Fˡ * D^6 * Np
end

# Sixth moment sublimation tendencies
@inline function integrand(::SixthMomentSublimation, D, state::IceSizeDistributionState)
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    return 6 * D^5 * fᵛᵉ * C * Np
end

@inline function integrand(::SixthMomentSublimation1, D, state::IceSizeDistributionState)
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    return 6 * D^5 * fᵛᵉ * C * Np
end

#####
##### Lambda limiter integrals
#####

@inline function integrand(::NumberMomentLambdaLimit, D, state::IceSizeDistributionState)
    Np = size_distribution(D, state)
    return Np
end

@inline function integrand(::MassMomentLambdaLimit, D, state::IceSizeDistributionState)
    m = particle_mass(D, state)
    Np = size_distribution(D, state)
    return m * Np
end

#####
##### Rain integrals
#####

@inline integrand(::RainShapeParameter, D, state) = zero(D)
@inline integrand(::RainVelocityNumber, D, state) = zero(D)
@inline integrand(::RainVelocityMass, D, state) = zero(D)
@inline integrand(::RainEvaporation, D, state) = zero(D)

#####
##### Ice-rain collection integrals
#####

@inline function integrand(::IceRainMassCollection, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    m = particle_mass(D, state)
    Np = size_distribution(D, state)
    return V * A * m * Np
end

@inline function integrand(::IceRainNumberCollection, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)
    return V * A * Np
end

@inline function integrand(::IceRainSixthMomentCollection, D, state::IceSizeDistributionState)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)
    return D^6 * V * A * Np
end

