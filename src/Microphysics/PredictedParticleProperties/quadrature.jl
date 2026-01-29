#####
##### Quadrature Evaluation of P3 Integrals
#####
##### Numerical integration over the ice size distribution using
##### Chebyshev-Gauss quadrature on a transformed domain.
#####
#####
##### References:
##### - Morrison & Milbrandt (2015a): Fall speed constants, Best number formulation
##### - Mitchell & Heymsfield (2005): Drag coefficients
##### - Heymsfield et al. (2006): Density correction
#####

export evaluate, chebyshev_gauss_nodes_weights

# Constants from P3 Fortran implementation (create_p3_lookupTable_1.f90)
# Reference conditions for fall speed parameterization
const P3_REF_T = 253.15       # Reference temperature [K]
const P3_REF_P = 60000.0      # Reference pressure [Pa]
const P3_REF_RHO = P3_REF_P / (287.15 * P3_REF_T) # ≈ 0.825 kg/m³

# Dynamic viscosity at reference conditions (Sutherland's law)
# μ = 1.496e-6 * T^1.5 / (T + 120)
const P3_REF_ETA = 1.496e-6 * P3_REF_T^1.5 / (P3_REF_T + 120.0) # ≈ 1.62e-5 Pa s

# Kinematic viscosity at reference conditions
const P3_REF_NU = P3_REF_ETA / P3_REF_RHO

# Mitchell & Heymsfield (2005) surface roughness parameters
const MH_δ₀ = 5.83
const MH_C₀ = 0.6
const MH_C₁ = 4 / (MH_δ₀^2 * sqrt(MH_C₀))
const MH_C₂ = MH_δ₀^2 / 4

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
    terminal_velocity(D, state)

Terminal velocity V(D) for ice particles following Mitchell and Heymsfield (2005).

The fall speed is calculated using the Best number formulation, which accounts for
particle mass, projected area, and air properties. A density correction factor
`(ρ₀/ρ)^0.54` is applied following Heymsfield et al. (2006).

For mixed-phase particles (with liquid fraction Fˡ), the velocity is a linear
interpolation between the ice fall speed and rain fall speed:
`V = Fˡ * V_rain + (1 - Fˡ) * V_ice`
"""
@inline function terminal_velocity(D, state::IceSizeDistributionState)
    FT = typeof(D)
    Fˡ = state.liquid_fraction
    
    # Calculate ice fall speed (Mitchell & Heymsfield 2005)
    # Uses mass/area of the ice portion only
    m_ice = particle_mass_ice_only(D, state)
    A_ice = particle_area_ice_only(D, state)
    V_ice = ice_fall_speed_mh2005(D, state, m_ice, A_ice)
    
    # Apply density correction to ice fall speed
    ρ = state.air_density
    ρ₀ = state.reference_air_density
    ρ_correction = (ρ₀ / max(ρ, FT(0.1)))^FT(0.54)
    V_ice_corr = V_ice * ρ_correction
    
    # Calculate rain fall speed (if needed)
    if Fˡ > eps(FT)
        # Rain fall speed includes density correction internally
        V_rain = rain_fall_speed(D, ρ_correction)
        return Fˡ * V_rain + (1 - Fˡ) * V_ice_corr
    else
        return V_ice_corr
    end
end

"""
    ice_fall_speed_mh2005(D, state, m, A)

Compute terminal velocity of ice particle using Mitchell & Heymsfield (2005).
Calculates velocity at reference conditions (P3_REF_T, P3_REF_P).
"""
@inline function ice_fall_speed_mh2005(D, state::IceSizeDistributionState, m, A)
    FT = typeof(D)
    g = FT(9.81)
    
    # Reference properties
    ρ_ref = FT(P3_REF_RHO)
    η_ref = FT(P3_REF_ETA) # dynamic
    ν_ref = FT(P3_REF_NU)  # kinematic
    
    # Avoid division by zero
    A_safe = max(A, eps(FT))
    
    # Best number X at reference conditions
    # X = 2 m g ρ D^2 / (A η^2)
    X = 2 * m * g * ρ_ref * D^2 / (A_safe * η_ref^2)
    
    # Limit X for numerical stability (and to match Fortran checks?)
    X = max(X, FT(1e-20))
    
    # MH2005 drag terms (a0=0, b0=0 branch for aggregates)
    X_sqrt = sqrt(X)
    C1_X_sqrt = MH_C₁ * X_sqrt
    term = sqrt(1 + C1_X_sqrt)
    
    # b₁ = (C₁ √X) / (2 (√(1+C₁√X)-1) √(1+C₁√X))
    denom_b = 2 * (term - 1) * term
    b₁ = C1_X_sqrt / max(denom_b, eps(FT))
    
    # a₁ = C₂ (√(1+C₁√X)-1)² / X^b₁
    # Note: X^b1 can be small.
    # Fortran computes `xx**b1` then `a1 = ... / xx**b1`
    
    # If X is very small (Stokes regime), b1 -> 1, a1 -> ?
    # Let's handle small X explicitly to avoid singularities
    if X < 1e-5
        # Stokes flow: V = m g / (3 π η D)
        # We can just return Stokes velocity
        return m * g / (3 * FT(π) * η_ref * D)
    end
    
    a₁ = MH_C₂ * (term - 1)^2 / X^b₁
    
    # Velocity formula derived from MH2005 power law fit Re = a X^b
    # V = a₁ * ν^(1-2b₁) * (2 m g / (ρ A))^b₁ * D^(2b₁ - 1)
    
    term_bracket = 2 * m * g / (ρ_ref * A_safe)
    
    V_ref = a₁ * ν_ref^(1 - 2*b₁) * term_bracket^b₁ * D^(2*b₁ - 1)
    
    return V_ref
end

"""
    rain_fall_speed(D, ρ_correction)

Compute rain fall speed using piecewise power laws from P3 Fortran.
"""
@inline function rain_fall_speed(D, ρ_correction)
    FT = typeof(D)
    
    # Mass of water sphere in GRAMS for the formula
    # ρ_w = 997 kg/m³
    m_kg = (FT(π)/6) * FT(997) * D^3
    m_g = m_kg * 1000
    
    # Formulas give V in cm/s
    if D <= 134.43e-6
        V_cm = 4.5795e5 * m_g^(2/3)
    elseif D < 1511.64e-6
        V_cm = 4.962e3 * m_g^(1/3)
    elseif D < 3477.84e-6
        V_cm = 1.732e3 * m_g^(1/6)
    else
        V_cm = FT(917.0)
    end
    
    return V_cm * FT(0.01) * ρ_correction
end

"""
    particle_mass_ice_only(D, state)

Mass of the ice portion of the particle (ignoring liquid water).
Used for fall speed calculation of the ice component.
"""
@inline function particle_mass_ice_only(D, state::IceSizeDistributionState)
    FT = typeof(D)
    α = state.mass_coefficient
    β = state.mass_exponent
    ρᵢ = state.ice_density
    
    thresholds = regime_thresholds_from_state(D, state)
    
    # Regime 1: small spheres
    a₁ = ρᵢ * FT(π) / 6
    b₁ = FT(3)
    
    # Regime 2: aggregates
    a₂ = FT(α)
    b₂ = FT(β)
    
    # Regime 3: graupel
    a₃ = thresholds.ρ_graupel * FT(π) / 6
    b₃ = FT(3)
    
    # Regime 4: partially rimed
    # Use safe rime fraction for coefficient calculation
    Fᶠ_safe = min(state.rime_fraction, FT(1) - eps(FT))
    a₄ = FT(α) / (1 - Fᶠ_safe)
    b₄ = FT(β)
    
    is_regime_4 = D ≥ thresholds.partial_rime
    is_regime_3 = D ≥ thresholds.graupel
    is_regime_2 = D ≥ thresholds.spherical
    
    a = a₁
    b = b₁
    
    a = ifelse(is_regime_2, a₂, a)
    b = ifelse(is_regime_2, b₂, b)
    
    a = ifelse(is_regime_3, a₃, a)
    b = ifelse(is_regime_3, b₃, b)
    
    a = ifelse(is_regime_4, a₄, a)
    b = ifelse(is_regime_4, b₄, b)
    
    return a * D^b
end

"""
    particle_area_ice_only(D, state)

Projected area of the ice portion of the particle.
"""
@inline function particle_area_ice_only(D, state::IceSizeDistributionState)
    FT = typeof(D)
    Fᶠ = state.rime_fraction
    
    thresholds = regime_thresholds_from_state(D, state)
    
    # Spherical area
    A_sphere = FT(π) / 4 * D^2
    
    # Aggregate area
    γ = FT(0.2285)
    σ = FT(1.88)
    A_aggregate = γ * D^σ
    
    is_small = D < thresholds.spherical
    is_graupel = D ≥ thresholds.graupel
    
    A_intermediate = (1 - Fᶠ) * A_aggregate + Fᶠ * A_sphere
    
    A = ifelse(is_small, A_sphere, A_intermediate)
    A = ifelse(is_graupel, A_sphere, A)
    
    return A
end

"""
    regime_thresholds_from_state(D, state)

Compute ice regime thresholds from the state's mass-diameter parameters.
Returns an IceRegimeThresholds struct with spherical, graupel, partial_rime thresholds.
"""
@inline function regime_thresholds_from_state(D, state::IceSizeDistributionState)
    FT = typeof(D)
    α = state.mass_coefficient
    β = state.mass_exponent
    ρᵢ = state.ice_density
    Fᶠ = state.rime_fraction
    ρᶠ = state.rime_density

    # Regime 1 threshold: D where power law equals sphere
    # (π/6) ρᵢ D³ = α D^β  →  D = (6α / (π ρᵢ))^(1/(3-β))
    D_spherical = (6 * α / (FT(π) * ρᵢ))^(1 / (3 - β))

    # For unrimed ice, graupel and partial rime thresholds are infinite
    is_unrimed = Fᶠ < FT(1e-10)

    # Safe rime fraction for rimed calculations
    Fᶠ_safe = max(Fᶠ, FT(1e-10))

    # Deposited ice density (Eq. 16 from MM15a)
    k = (1 - Fᶠ_safe)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ_safe
    den = (β - 2) * (k - 1) / ((1 - Fᶠ_safe) * k - 1) - (1 - Fᶠ_safe)
    ρ_dep = num / max(den, FT(1e-10))

    # Graupel density
    ρ_g = Fᶠ_safe * ρᶠ + (1 - Fᶠ_safe) * ρ_dep

    # Graupel threshold
    D_graupel_calc = (6 * α / (FT(π) * ρ_g))^(1 / (3 - β))

    # Partial rime threshold
    D_partial_calc = (6 * α / (FT(π) * ρ_g * (1 - Fᶠ_safe)))^(1 / (3 - β))

    D_graupel = ifelse(is_unrimed, FT(Inf), D_graupel_calc)
    D_partial = ifelse(is_unrimed, FT(Inf), D_partial_calc)
    ρ_graupel = ifelse(is_unrimed, ρᵢ, ρ_g)

    return (spherical = D_spherical, graupel = D_graupel, partial_rime = D_partial, ρ_graupel = ρ_graupel)
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
    particle_mass(D, state)

Particle mass m(D) as a function of diameter.

Includes the mass of the ice portion (from P3 4-regime m-D relationships)
plus any liquid water coating (from liquid fraction Fˡ).

`m(D) = (1 - Fˡ) * m_ice(D) + Fˡ * m_liquid(D)`

where m_liquid is the mass of a water sphere.
"""
@inline function particle_mass(D, state::IceSizeDistributionState)
    FT = typeof(D)
    Fˡ = state.liquid_fraction
    
    # Calculate ice mass (unmodified by liquid fraction)
    m_ice = particle_mass_ice_only(D, state)
    
    # Liquid mass (sphere)
    # ρ_w = 1000 kg/m³ (from P3 Fortran)
    m_liquid = FT(π)/6 * 1000 * D^3
    
    return (1 - Fˡ) * m_ice + Fˡ * m_liquid
end


#####
##### Deposition/ventilation integrals
#####

"""
    ventilation_factor(D, state, constant_term)

Ventilation factor f_v for vapor diffusion enhancement following Hall & Pruppacher (1976).

The ventilation factor accounts for enhanced mass transfer due to air flow
around falling particles. For a particle of diameter D falling at velocity V:

f_v = a_v + b_v × Re^(1/2) × Sc^(1/3)

where Re = V×D/ν is the Reynolds number and Sc = ν/D_v is the Schmidt number.
For typical atmospheric conditions with Sc^(1/3) ≈ 0.9:

- Small particles (D ≤ 100 μm): f_v ≈ 1.0 (diffusion-limited)
- Large particles (D > 100 μm): f_v = 0.65 + 0.44 × √(V × D / ν)

This function returns either the constant term (0.65) or the Reynolds-dependent
term (0.44 × √(V×D)) depending on the `constant_term` argument, allowing
separation for integral evaluation.
"""
@inline function ventilation_factor(D, state::IceSizeDistributionState, constant_term)
    FT = typeof(D)
    V = terminal_velocity(D, state)

    # Kinematic viscosity of air (approximately 1.5e-5 m²/s at typical conditions)
    ν = FT(1.5e-5)

    D_threshold = FT(100e-6)
    is_small = D ≤ D_threshold

    # Small particles: no ventilation enhancement (f_v = 1)
    # constant_term=true → 1, constant_term=false → 0
    small_value = ifelse(constant_term, one(FT), zero(FT))

    # Large particles: f_v = 0.65 + 0.44 × √(V × D / ν)
    # constant_term=true → 0.65, constant_term=false → 0.44 × √(V × D / ν)
    Re_term = sqrt(V * D / ν)
    large_value = ifelse(constant_term, FT(0.65), FT(0.44) * Re_term)

    return ifelse(is_small, small_value, large_value)
end

# Backwards compatibility wrapper
@inline ventilation_factor(D, state; constant_term=true) = ventilation_factor(D, state, constant_term)

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
    thresholds = regime_thresholds_from_state(D, state)
    D_crit = thresholds.spherical
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::SmallIceVentilationReynolds, D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    D_crit = thresholds.spherical
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationConstant, D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    D_crit = thresholds.spherical
    fᵛᵉ = ventilation_factor(D, state; constant_term=true)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationReynolds, D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    D_crit = thresholds.spherical
    fᵛᵉ = ventilation_factor(D, state; constant_term=false)
    C = capacitance(D, state)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

"""
    capacitance(D, state)

Capacitance C(D) for vapor diffusion following regime-dependent formulation.

The capacitance determines the rate of vapor exchange with ice particles:
- Small spherical ice (D < D_th): C = D/2 (sphere)
- Large ice crystals/aggregates: C ≈ 0.48 × D (non-spherical)
- Heavily rimed (graupel): C = D/2 (approximately spherical)

For non-spherical particles, the capacitance is approximated as that of
an oblate spheroid with aspect ratio typical of vapor-grown crystals.
See Pruppacher & Klett (1997) Chapter 13.
"""
@inline function capacitance(D, state::IceSizeDistributionState)
    FT = typeof(D)
    Fᶠ = state.rime_fraction

    # Get regime thresholds
    thresholds = regime_thresholds_from_state(D, state)

    # Sphere capacitance
    C_sphere = D / 2

    # Non-spherical capacitance (oblate spheroid approximation)
    # Typical aspect ratio of 0.6 gives C ≈ 0.48 D
    C_nonspherical = FT(0.48) * D

    # Small spherical ice
    is_small = D < thresholds.spherical

    # Heavily rimed particles become more spherical
    # Graupel and heavily rimed particles: use spherical capacitance
    is_graupel = D ≥ thresholds.graupel

    # Interpolate based on rime fraction for intermediate regime
    # More rime → more spherical
    C_intermediate = (1 - Fᶠ) * C_nonspherical + Fᶠ * C_sphere

    # Select based on regime
    C = ifelse(is_small, C_sphere, C_intermediate)
    C = ifelse(is_graupel, C_sphere, C)

    return C
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
    particle_density(D, state)

Particle effective density ρ(D) as a function of diameter.

The density is computed from the mass and volume:
ρ_eff(D) = m(D) / V(D) = m(D) / [(π/6) D³]

This gives regime-dependent effective densities:
- Small spherical ice: ρ_eff = ρᵢ = 917 kg/m³
- Aggregates: ρ_eff = 6α D^(β-3) / π (decreases with size for β < 3)
- Graupel: ρ_eff = ρ_g
- Partially rimed: ρ_eff = 6α D^(β-3) / [π(1-Fᶠ)]
"""
@inline function particle_density(D, state::IceSizeDistributionState)
    FT = typeof(D)

    # Get particle mass from regime-dependent formulation
    m = particle_mass(D, state)

    # Particle volume (sphere)
    V = FT(π) / 6 * D^3

    # Effective density = mass / volume
    # Clamp to avoid unrealistic values
    ρ_eff = m / max(V, eps(FT))

    # Clamp to physical range [50, 1000] kg/m³ (upper bound 1000 for liquid water)
    return clamp(ρ_eff, FT(50), FT(1000))
end

#####
##### Collection integrals
#####

"""
    collision_kernel(D₁, D₂, state, E_coll)

Collision kernel K(D₁,D₂) for ice-ice aggregation following Morrison & Milbrandt (2015a).

K(D₁,D₂) = E_coll × (π/4)(D₁+D₂)² × |V(D₁) - V(D₂)|

where:
- E_coll is the collection efficiency (typically 0.1-1.0 for aggregation)
- (π/4)(D₁+D₂)² is the geometric sweep-out cross-section
- |V(D₁) - V(D₂)| is the differential fall speed
"""
@inline function collision_kernel(D₁, D₂, state::IceSizeDistributionState, E_coll)
    FT = typeof(D₁)

    # Terminal velocities at each diameter
    V₁ = terminal_velocity(D₁, state)
    V₂ = terminal_velocity(D₂, state)

    # Differential fall speed
    ΔV = abs(V₁ - V₂)

    # Geometric sweep-out area: π/4 × (D₁ + D₂)²
    A_sweep = FT(π) / 4 * (D₁ + D₂)^2

    return E_coll * A_sweep * ΔV
end

"""
    evaluate_double_integral(state, kernel_func; n_quadrature=32)

Evaluate a double integral ∫∫ kernel_func(D₁, D₂, state) N'(D₁) N'(D₂) dD₁ dD₂
using 2D Chebyshev-Gauss quadrature.

This is used for collection integrals (aggregation, self-collection) that require
integration over pairs of particle sizes.
"""
function evaluate_double_integral(state::IceSizeDistributionState, kernel_func;
                                   n_quadrature::Int = 32)
    FT = typeof(state.slope)
    nodes, weights = chebyshev_gauss_nodes_weights(FT, n_quadrature)

    λ = state.slope
    result = zero(FT)

    for i in 1:n_quadrature
        x₁ = nodes[i]
        w₁ = weights[i]
        D₁ = transform_to_diameter(x₁, λ)
        J₁ = jacobian_diameter_transform(x₁, λ)
        N₁ = size_distribution(D₁, state)

        for j in 1:n_quadrature
            x₂ = nodes[j]
            w₂ = weights[j]
            D₂ = transform_to_diameter(x₂, λ)
            J₂ = jacobian_diameter_transform(x₂, λ)
            N₂ = size_distribution(D₂, state)

            # Kernel value
            K = kernel_func(D₁, D₂, state)

            result += w₁ * w₂ * K * N₁ * N₂ * J₁ * J₂
        end
    end

    return result
end

"""
    aggregation_kernel(D₁, D₂, state)

Aggregation kernel for ice-ice self-collection.
"""
@inline function aggregation_kernel(D₁, D₂, state::IceSizeDistributionState)
    FT = typeof(D₁)
    E_agg = FT(0.1)  # Default aggregation efficiency (temperature-dependent in full model)
    return collision_kernel(D₁, D₂, state, E_agg)
end

# Aggregation number: ∫∫ K(D₁,D₂) N'(D₁) N'(D₂) dD₁ dD₂
# Using approximation from Wisner et al. (1972) for computational efficiency:
# I_agg ≈ ∫ V(D) A(D) N(D)² dD × scale_factor
# This is the self-collection form used in most bulk schemes
@inline function integrand(::AggregationNumber, D, state::IceSizeDistributionState)
    FT = typeof(D)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)

    # Aggregation efficiency (simplified)
    E_agg = FT(0.1)

    # Self-collection approximation: ∫ E_agg × V × A × N² dD
    return E_agg * V * A * Np^2
end

"""
    evaluate_aggregation_integral(state; n_quadrature=32)

Evaluate the full 2D aggregation integral using proper collision kernel.
This is more accurate than the 1D approximation but slower.

∫∫ (1/2) K_agg(D₁,D₂) N'(D₁) N'(D₂) dD₁ dD₂

The factor of 1/2 avoids double-counting symmetric collisions.
"""
function evaluate_aggregation_integral(state::IceSizeDistributionState;
                                        n_quadrature::Int = 32)
    return evaluate_double_integral(state, aggregation_kernel; n_quadrature) / 2
end

# Rain collection by ice (riming kernel)
# ∫ E_rim × V(D) × A(D) × N'(D) dD
@inline function integrand(::RainCollectionNumber, D, state::IceSizeDistributionState)
    FT = typeof(D)
    V = terminal_velocity(D, state)
    A = particle_area(D, state)
    Np = size_distribution(D, state)

    # Collection efficiency for rain-ice (typically higher than ice-ice)
    E_rim = FT(1.0)

    return E_rim * V * A * Np
end

"""
    riming_kernel(D_ice, D_drop, V_ice, V_drop, E_rim)

Riming kernel for ice-droplet collection.

K = E_rim × (π/4)(D_ice + D_drop)² × |V_ice - V_drop|

For riming, the collection efficiency E_rim ≈ 1 for large ice collecting
small cloud droplets, but decreases for small ice or large rain drops.
"""
@inline function riming_kernel(D_ice, D_drop, V_ice, V_drop, E_rim)
    FT = typeof(D_ice)
    A_sweep = FT(π) / 4 * (D_ice + D_drop)^2
    ΔV = abs(V_ice - V_drop)
    return E_rim * A_sweep * ΔV
end

"""
    particle_area(D, state)

Projected cross-sectional area A(D) for ice particles.

Includes liquid fraction weighting for mixed-phase particles.
"""
@inline function particle_area(D, state::IceSizeDistributionState)
    FT = typeof(D)
    Fˡ = state.liquid_fraction
    
    # Calculate ice area (unmodified by liquid fraction)
    A_ice = particle_area_ice_only(D, state)
    
    # Liquid area (sphere)
    A_liquid = FT(π)/4 * D^2
    
    return (1 - Fˡ) * A_ice + Fˡ * A_liquid
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
