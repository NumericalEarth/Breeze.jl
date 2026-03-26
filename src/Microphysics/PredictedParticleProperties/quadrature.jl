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
const P3_REF_RHO = P3_REF_P / (dry_air_gas_constant(ThermodynamicConstants()) * P3_REF_T) # ≈ 0.825 kg/m³

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
    weights = zeros(FT, n)

    for i in 1:n
        x = cos(FT((2i - 1) * π / (2n)))
        nodes[i] = x
        # Chebyshev-Gauss type 1 computes ∫ f(x)/√(1-x²) dx with weight π/n.
        # For regular integrals ∫ f(x) dx, multiply by √(1-x²).
        weights[i] = FT(π / n) * sqrt(1 - x^2)
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

    # Precompute regime thresholds once (independent of D)
    thresholds = regime_thresholds_from_state(FT, state)

    for i in 1:n_quadrature
        x = nodes[i]
        w = weights[i]

        D = transform_to_diameter(x, λ)
        J = jacobian_diameter_transform(x, λ)

        # Compute integrand at this diameter with precomputed thresholds
        f = integrand(integral, D, state, thresholds)

        result += w * f * J
    end

    return result
end

#####
##### Integrand functions for each integral type
#####

# Default fallback (4-argument form with precomputed thresholds)
integrand(::AbstractP3Integral, D, state, thresholds) = zero(D)

# Backward-compatible 3-argument fallback: compute thresholds on the fly
@inline function integrand(integral::AbstractP3Integral, D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return integrand(integral, D, state, thresholds)
end

#####
##### Fall speed integrals
#####

"""
    terminal_velocity(D, state, thresholds)

Terminal velocity V(D) for ice particles following Mitchell and Heymsfield (2005).

The fall speed is calculated using the Best number formulation, which accounts for
particle mass, projected area, and air properties. A density correction factor
`(ρ₀/ρ)^0.54` is applied following Heymsfield et al. (2006).

For mixed-phase particles (with liquid fraction Fˡ), the velocity is a linear
interpolation between the ice fall speed and rain fall speed:
`V = Fˡ * V_rain + (1 - Fˡ) * V_ice`

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function terminal_velocity(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    Fˡ = state.liquid_fraction

    # Calculate ice fall speed (Mitchell & Heymsfield 2005)
    # Uses mass/area of the ice portion only
    m_ice = particle_mass_ice_only(D, state, thresholds)
    A_ice = particle_area_ice_only(D, state, thresholds)
    V_ice = ice_fall_speed_mh2005(D, state, m_ice, A_ice)

    # Apply density correction to ice fall speed
    # ρ₀ must match the reference conditions at which V_ice was computed (P3_REF_RHO ≈ 0.825)
    ρ = state.air_density
    ρ₀ = FT(P3_REF_RHO)
    ρ_correction = (ρ₀ / max(ρ, FT(0.1)))^FT(0.54)
    V_ice_corr = V_ice * ρ_correction

    # Calculate rain fall speed and blend with ice
    V_rain = rain_fall_speed(D, ρ_correction)
    V_blend = Fˡ * V_rain + (1 - Fˡ) * V_ice_corr
    return ifelse(Fˡ > eps(FT), V_blend, V_ice_corr)
end

# Backward-compatible method: compute thresholds on the fly
@inline function terminal_velocity(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return terminal_velocity(D, state, thresholds)
end

"""
    ice_fall_speed_mh2005(D, state, m, A)

Compute terminal velocity of ice particle using Mitchell & Heymsfield (2005).
Calculates velocity at reference conditions (P3_REF_T, P3_REF_P).
"""
@inline function ice_fall_speed_mh2005(D, state::IceSizeDistributionState, m, A)
    FT = typeof(D)
    g = FT(ThermodynamicConstants().gravitational_acceleration)

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

    # Stokes regime (small X): V = m g / (3 π η D)
    V_stokes = m * g / (3 * FT(π) * η_ref * D)

    a₁ = MH_C₂ * (term - 1)^2 / max(X^b₁, eps(FT))

    # Velocity formula derived from MH2005 power law fit Re = a X^b
    # V = a₁ * ν^(1-2b₁) * (2 m g / (ρ A))^b₁ * D^(2b₁ - 1)
    term_bracket = 2 * m * g / (ρ_ref * A_safe)
    V_mh = a₁ * ν_ref^(1 - 2*b₁) * term_bracket^b₁ * D^(2*b₁ - 1)

    return ifelse(X < FT(1e-5), V_stokes, V_mh)
end

"""
    rain_fall_speed(D, ρ_correction)

Compute rain fall speed using piecewise power laws from P3 Fortran.
"""
@inline function rain_fall_speed(D, ρ_correction)
    FT = typeof(D)

    # Mass of water sphere in GRAMS for the formula
    # ρ_w = 1000 kg/m³ (consistent with ProcessRateParameters.liquid_water_density)
    m_kg = (FT(π)/6) * FT(1000) * D^3
    m_g = m_kg * 1000

    # Piecewise power law (Gunn-Kinzer/Beard), V in cm/s
    V_cm = ifelse(D <= FT(134.43e-6),  FT(4.5795e5) * m_g^(FT(2)/FT(3)),
           ifelse(D <  FT(1511.64e-6), FT(4.962e3)  * m_g^(FT(1)/FT(3)),
           ifelse(D <  FT(3477.84e-6), FT(1.732e3)  * m_g^(FT(1)/FT(6)),
                                       FT(917.0))))

    return V_cm * FT(0.01) * ρ_correction
end

"""
    particle_mass_ice_only(D, state, thresholds)

Mass of the ice portion of the particle (ignoring liquid water).
Used for fall speed calculation of the ice component.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function particle_mass_ice_only(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    α = state.mass_coefficient
    β = state.mass_exponent
    ρᵢ = state.ice_density

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

    # Determine which regime applies (work backwards from regime 4)
    # Note: same logic and ordering as ice_mass_coefficients in lambda_solver.jl
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

    return a * D^b
end

# Backward-compatible method: compute thresholds on the fly
@inline function particle_mass_ice_only(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_mass_ice_only(D, state, thresholds)
end

@inline function partially_rimed_mass_weight(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    α = state.mass_coefficient
    β = state.mass_exponent

    m_actual = particle_mass_ice_only(D, state, thresholds)
    m_unrimed = α * D^β
    m_graupel = thresholds.ρ_graupel * FT(π) / 6 * D^3
    Δm = m_graupel - m_unrimed
    weight = ifelse(abs(Δm) > eps(FT), (m_actual - m_unrimed) / Δm, one(FT))

    return clamp(weight, zero(FT), one(FT))
end

"""
    particle_diameter_ice_only(m_ice, state, thresholds)

Invert the piecewise ice-only mass-diameter relation to recover diameter from mass.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function particle_diameter_ice_only(m_ice, state::IceSizeDistributionState, thresholds)
    FT = typeof(m_ice)
    α = state.mass_coefficient
    β = state.mass_exponent
    ρᵢ = state.ice_density

    Fᶠ_safe = min(state.rime_fraction, one(FT) - eps(FT))
    m_positive = max(0, m_ice)

    D_spherical = cbrt(6 * m_positive / (FT(π) * ρᵢ))
    D_aggregate = (m_positive / α)^(1 / β)
    D_graupel = cbrt(6 * m_positive / (FT(π) * thresholds.ρ_graupel))
    D_partial = ((1 - Fᶠ_safe) * m_positive / α)^(1 / β)

    m_spherical = ρᵢ * FT(π) / 6 * thresholds.spherical^3
    m_graupel = thresholds.ρ_graupel * FT(π) / 6 * thresholds.graupel^3
    m_partial = α / (1 - Fᶠ_safe) * thresholds.partial_rime^β

    is_regime_4 = m_positive ≥ m_partial
    is_regime_3 = m_positive ≥ m_graupel
    is_regime_2 = m_positive ≥ m_spherical

    D = ifelse(is_regime_4, D_partial, D_graupel)
    D = ifelse(is_regime_3, D, D_aggregate)
    D = ifelse(is_regime_2, D, D_spherical)

    return D
end

@inline function particle_diameter_ice_only(m_ice, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(m_ice, state)
    return particle_diameter_ice_only(m_ice, state, thresholds)
end

"""
    particle_area_ice_only(D, state, thresholds)

Projected area of the ice portion of the particle.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function particle_area_ice_only(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)

    # Spherical area
    A_sphere = FT(π) / 4 * D^2

    # Aggregate area: A = γ D^σ (Mitchell 1996)
    # Original coefficients are CGS (cm for D, cm² for A): γ_cgs = 0.2285, σ = 1.88
    # Convert to MKS: γ_mks = γ_cgs × 100^σ / 100² = 0.2285 × 100^(1.88-2)
    σ = FT(1.88)
    γ = FT(0.2285) * FT(100)^(σ - 2)
    A_aggregate = γ * D^σ

    is_small = D < thresholds.spherical
    is_partially_rimed = D ≥ thresholds.partial_rime
    is_graupel = (D ≥ thresholds.graupel) & !is_partially_rimed

    partial_rime_weight = partially_rimed_mass_weight(D, state, thresholds)
    A_partial = A_aggregate + partial_rime_weight * (A_sphere - A_aggregate)

    A = A_aggregate
    A = ifelse(is_partially_rimed, A_partial, A)
    A = ifelse(is_graupel, A_sphere, A)
    A = ifelse(is_small, A_sphere, A)

    return A
end

# Backward-compatible method: compute thresholds on the fly
@inline function particle_area_ice_only(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_area_ice_only(D, state, thresholds)
end

"""
    regime_thresholds_from_state(FT, state)

Compute ice regime thresholds from the state's mass-diameter parameters.
Returns a NamedTuple with spherical, graupel, partial_rime thresholds and ρ_graupel.

Thresholds depend only on the state's rime properties and mass-diameter
parameters — they are independent of particle diameter D. Computing them
once per quadrature evaluation (rather than per quadrature point) avoids
redundant cube-root and deposited-ice-density calculations.
"""
@inline function regime_thresholds_from_state(::Type{FT}, state::IceSizeDistributionState) where {FT}
    α = state.mass_coefficient
    β = state.mass_exponent
    ρᵢ = state.ice_density
    Fᶠ = state.rime_fraction
    ρᶠ = state.rime_density

    # Regime 1 threshold: D where power law equals sphere
    # (π/6) ρᵢ D³ = α D^β  →  D = (6α / (π ρᵢ))^(1/(3-β))
    D_spherical = (6 * α / (FT(π) * ρᵢ))^(1 / (3 - β))

    # For unrimed ice, graupel and partial rime thresholds are infinite
    # NOTE (M3): This duplicates ice_regime_thresholds() in lambda_solver.jl.
    # Both must produce identical thresholds. If one is changed, update the other.
    is_unrimed = Fᶠ < eps(FT)

    # Safe rime fraction for rimed calculations
    Fᶠ_safe = max(Fᶠ, eps(FT))

    # Deposited ice density (Eq. 16 from MM15a)
    k = (1 - Fᶠ_safe)^(-1 / (3 - β))
    num = ρᶠ * Fᶠ_safe
    den = (β - 2) * (k - 1) / ((1 - Fᶠ_safe) * k - 1) - (1 - Fᶠ_safe)
    ρ_dep = num / max(den, eps(FT))

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

# Backward-compatible method: extract FT from the diameter argument
@inline regime_thresholds_from_state(D, state::IceSizeDistributionState) =
    regime_thresholds_from_state(typeof(D), state)

# Number-weighted fall speed: ∫ V(D) N'(D) dD
@inline function integrand(::NumberWeightedFallSpeed, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    Np = size_distribution(D, state)
    return V * Np
end

# Mass-weighted fall speed: ∫ V(D) m(D) N'(D) dD
@inline function integrand(::MassWeightedFallSpeed, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return V * m * Np
end

# Reflectivity-weighted fall speed: ∫ V(D) D^6 N'(D) dD
@inline function integrand(::ReflectivityWeightedFallSpeed, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    Np = size_distribution(D, state)
    return V * D^6 * Np
end

#####
##### Particle mass
#####

"""
    particle_mass(D, state, thresholds)

Particle mass m(D) as a function of diameter.

Includes the mass of the ice portion (from P3 4-regime m-D relationships)
plus any liquid water coating (from liquid fraction Fˡ).

`m(D) = (1 - Fˡ) * m_ice(D) + Fˡ * m_liquid(D)`

where m_liquid is the mass of a water sphere.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function particle_mass(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    Fˡ = state.liquid_fraction

    # Calculate ice mass (unmodified by liquid fraction)
    m_ice = particle_mass_ice_only(D, state, thresholds)

    # Liquid mass (sphere)
    # ρ_w = 1000 kg/m³ (from P3 Fortran)
    m_liquid = FT(π)/6 * 1000 * D^3

    return (1 - Fˡ) * m_ice + Fˡ * m_liquid
end

# Backward-compatible method: compute thresholds on the fly
@inline function particle_mass(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_mass(D, state, thresholds)
end

"""
    particle_mass_derivative(D, state, thresholds)

Derivative dm/dD of the total (ice + liquid) particle mass with respect to diameter.

For the piecewise power law `m_ice = a D^b`, `dm_ice/dD = a b D^(b-1)`.
Including liquid fraction Fˡ:

`dm/dD = (1 - Fˡ) dm_ice/dD + Fˡ × (π/2) × 1000 × D²`

This Jacobian is used in the sixth-moment integrand normalization to convert
mass growth rate to diameter growth rate: `dD/dt = (dm/dt) / (dm/dD)`.
Matches the Fortran `dmdD` computation in `create_p3_lookupTable_1.f90`.
"""
@inline function particle_mass_derivative(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    α = state.mass_coefficient
    β = state.mass_exponent
    ρᵢ = state.ice_density
    Fˡ = state.liquid_fraction

    # dm_ice/dD for each regime (derivative of a D^b → a b D^(b-1))
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
    Fᶠ_safe = min(state.rime_fraction, FT(1) - eps(FT))
    a₄ = FT(α) / (1 - Fᶠ_safe)
    b₄ = FT(β)

    is_regime_4 = D ≥ thresholds.partial_rime
    is_regime_3 = D ≥ thresholds.graupel
    is_regime_2 = D ≥ thresholds.spherical

    a = ifelse(is_regime_4, a₄, a₃)
    b = ifelse(is_regime_4, b₄, b₃)
    a = ifelse(is_regime_3, a, a₂)
    b = ifelse(is_regime_3, b, b₂)
    a = ifelse(is_regime_2, a, a₁)
    b = ifelse(is_regime_2, b, b₁)

    dmdD_ice = a * b * D^(b - 1)

    # Liquid sphere: m_liquid = (π/6) × 1000 × D³ → dm/dD = (π/2) × 1000 × D²
    dmdD_liquid = FT(π) / 2 * 1000 * D^2

    dmdD = (1 - Fˡ) * dmdD_ice + Fˡ * dmdD_liquid
    return max(dmdD, eps(FT))
end

@inline function particle_mass_derivative(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_mass_derivative(D, state, thresholds)
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
@inline function ventilation_factor(D, state::IceSizeDistributionState, constant_term, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)

    D_threshold = FT(100e-6)
    is_small = D ≤ D_threshold

    # Small particles: no ventilation enhancement (f_v = 1)
    # constant_term=true → 1, constant_term=false → 0
    small_value = ifelse(constant_term, one(FT), zero(FT))

    # Large particles (Fortran table convention):
    # The table stores the PSD-dependent part without Sc^(1/3) × √ρ_fac / √ν.
    # At runtime, the deposition rate multiplies by Sc^(1/3) × √ρ_fac / √ν.
    # constant_term=true → 0.65
    # constant_term=false → 0.44 × √(V × D)
    VD_term = sqrt(max(V * D, zero(FT)))
    large_value = ifelse(constant_term, FT(0.65), FT(0.44) * VD_term)

    return ifelse(is_small, small_value, large_value)
end

# Backward-compatible methods: compute thresholds on the fly
@inline function ventilation_factor(D, state::IceSizeDistributionState, constant_term)
    thresholds = regime_thresholds_from_state(D, state)
    return ventilation_factor(D, state, constant_term, thresholds)
end

@inline ventilation_factor(D, state; constant_term=true) = ventilation_factor(D, state, constant_term)

"""
    melt_ventilation_factor(D, state, constant_term, thresholds)

Fortran P3 melting ventilation factor used for `vdepm*` and `m6mlt*` tables.

For `D ≤ 100 μm`, melting uses the diffusion-limited coefficients `fac1 = 1`
and `fac2 = 0`. For larger particles it blends the ice and rain ventilation
branches by liquid fraction Fˡ:

- constant term: `(1 - Fˡ) × 0.65 + Fˡ × 0.78`
- Reynolds term: `(1 - Fˡ) × 0.44 × √(V_ice D) + Fˡ × 0.28 × √(V_rain D)`

This matches `create_p3_lookupTable_1.f90` lines 1967-1985.
"""
@inline function melt_ventilation_factor(D, state::IceSizeDistributionState, constant_term, thresholds)
    FT = typeof(D)
    Fˡ = state.liquid_fraction

    D_threshold = FT(100e-6)
    is_small = D ≤ D_threshold
    small_value = ifelse(constant_term, one(FT), zero(FT))

    m_ice = particle_mass_ice_only(D, state, thresholds)
    A_ice = particle_area_ice_only(D, state, thresholds)
    V_ice = ice_fall_speed_mh2005(D, state, m_ice, A_ice)

    ρ = state.air_density
    ρ₀ = FT(P3_REF_RHO)
    ρ_correction = (ρ₀ / max(ρ, FT(0.1)))^FT(0.54)

    V_ice_corr = V_ice * ρ_correction
    V_rain = rain_fall_speed(D, ρ_correction)

    large_constant = (one(FT) - Fˡ) * FT(0.65) + Fˡ * FT(0.78)
    large_reynolds = (one(FT) - Fˡ) * FT(0.44) * sqrt(max(V_ice_corr * D, zero(FT))) +
                     Fˡ * FT(0.28) * sqrt(max(V_rain * D, zero(FT)))
    large_value = ifelse(constant_term, large_constant, large_reynolds)

    return ifelse(is_small, small_value, large_value)
end

@inline function melt_ventilation_factor(D, state::IceSizeDistributionState, constant_term)
    thresholds = regime_thresholds_from_state(D, state)
    return melt_ventilation_factor(D, state, constant_term, thresholds)
end

# Basic ventilation: ∫ fᵛᵉ(D) C(D) N'(D) dD
@inline function integrand(::Ventilation, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    return fᵛᵉ * C * Np
end

@inline function integrand(::VentilationEnhanced, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    return fᵛᵉ * C * Np
end

# Size-regime-specific ventilation for melting
@inline function integrand(::SmallIceVentilationConstant, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::SmallIceVentilationReynolds, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationConstant, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationReynolds, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

"""
    capacitance(D, state, thresholds)

Capacitance C(D) for vapor diffusion following the P3 Fortran convention.

Returns `capm = cap × D` (Fortran convention), where:
- Small spherical ice (D < D_th): capm = D (cap=1, physical C = D/2)
- Large ice crystals/aggregates: capm = 0.48 × D (cap=0.48)
- Heavily rimed (graupel): capm = D (cap=1, approximately spherical)

Rate equations use `2π × capm` (not `4π × C`) so that
`2π × D = 4π × D/2 = 4πC_physical` is correct.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.

See Pruppacher & Klett (1997) Chapter 13.
"""
@inline function capacitance(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    Fˡ = state.liquid_fraction

    # Sphere capacitance (P3 Fortran convention: cap=1.0, capm = cap × D)
    # Physical capacitance is D/2; the extra factor of 2 is absorbed by using
    # 2π instead of 4π in the deposition/melting rate equations.
    C_sphere = D

    # Non-spherical capacitance (oblate spheroid approximation)
    # P3 Fortran: cap=0.48, capm = 0.48 × D
    C_nonspherical = FT(0.48) * D

    is_small = D < thresholds.spherical
    is_partially_rimed = D ≥ thresholds.partial_rime
    is_graupel = (D ≥ thresholds.graupel) & !is_partially_rimed

    partial_rime_weight = partially_rimed_mass_weight(D, state, thresholds)
    C_ice = C_nonspherical + partial_rime_weight * (C_sphere - C_nonspherical)

    C = C_nonspherical
    C = ifelse(is_partially_rimed, C_ice, C)
    C = ifelse(is_graupel, C_sphere, C)
    C = ifelse(is_small, C_sphere, C)

    return (1 - Fˡ) * C + Fˡ * C_sphere
end

# Backward-compatible method: compute thresholds on the fly
@inline function capacitance(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return capacitance(D, state, thresholds)
end

#####
##### Bulk property integrals
#####

# Effective radius (Fortran convention): eff = 3 ∫m N'dD / (4 ρ_ice ∫A N'dD)
# Integrand computes the numerator: m(D) N'(D)
# Normalization in tabulation.jl divides by area integral × (4/3) ρ_ice
@inline function integrand(::EffectiveRadius, D, state::IceSizeDistributionState, thresholds)
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return m * Np
end

# Mean diameter: ∫ D m(D) N'(D) dD
@inline function integrand(::MeanDiameter, D, state::IceSizeDistributionState, thresholds)
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return D * m * Np
end

# Mean density: ∫ ρ(D) m(D) N'(D) dD
@inline function integrand(::MeanDensity, D, state::IceSizeDistributionState, thresholds)
    m = particle_mass(D, state, thresholds)
    ρ = particle_density(D, state, thresholds)
    Np = size_distribution(D, state)
    return ρ * m * Np
end

# Reflectivity (Fortran Rayleigh convention):
# refl = ∫ 0.1892 × (6/(π ρ_ice))² × m(D)² × N'(D) dD
# where 0.1892 ≈ π⁵|K_w|²/λ⁴ Rayleigh prefactor
@inline function integrand(::Reflectivity, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    ρ_ice = FT(916.7)
    K_refl = FT(0.1892) * (6 / (FT(π) * ρ_ice))^2
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return K_refl * m^2 * Np
end

# Slope parameter λ - diagnostic, not an integral
@inline integrand(::SlopeParameter, D, state::IceSizeDistributionState, thresholds) = zero(D)

# Shape parameter μ - diagnostic, not an integral
@inline integrand(::ShapeParameter, D, state::IceSizeDistributionState, thresholds) = zero(D)

# Shedding rate: integral over particles with D ≥ 9 mm (Rasmussen et al. 2011).
# Fortran P3 f1pr28: ∫_{D≥9mm} m(D) N'(D) dD, normalized per particle.
# Uses Fl-blended mass ((1-Fl)*m_ice + Fl*m_liquid), matching Fortran table
# generation (create_p3_lookupTable_1.f90 line 1600). The rime fraction Fr
# and liquid fraction Fl multiplier are applied at runtime.
@inline function integrand(::SheddingRate, D, state::IceSizeDistributionState, thresholds)
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    # Only particles with D ≥ 9 mm can shed (Rasmussen et al. 2011)
    return ifelse(D >= typeof(D)(0.009), m * Np, zero(D))
end

"""
    particle_density(D, state, thresholds)

Particle effective density ρ(D) as a function of diameter.

The density is computed from the mass and volume:
ρ_eff(D) = m(D) / V(D) = m(D) / [(π/6) D³]

This gives regime-dependent effective densities:
- Small spherical ice: ρ_eff = ρᵢ = 900 kg/m³
- Aggregates: ρ_eff = 6α D^(β-3) / π (decreases with size for β < 3)
- Graupel: ρ_eff = ρ_g
- Partially rimed: ρ_eff = 6α D^(β-3) / [π(1-Fᶠ)]

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function particle_density(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)

    # Get particle mass from regime-dependent formulation
    m = particle_mass(D, state, thresholds)

    # Particle volume (sphere)
    V = FT(π) / 6 * D^3

    # Effective density = mass / volume
    # No clamping — the P3 m-D relationship already constrains density
    # through the four regimes (sphere, aggregate, graupel, partial rime)
    return m / max(V, eps(FT))
end

# Backward-compatible method: compute thresholds on the fly
@inline function particle_density(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_density(D, state, thresholds)
end

#####
##### Collection integrals
#####

# Aggregation number: ∫∫ (√A₁+√A₂)² |V₁-V₂| N'(D₁) N'(D₂) dD₁ dD₂
# WARNING: This single-integral integrand uses the Wisner (1972) approximation
# (V × A × N'²) which has different magnitude than the double integral stored
# in tables. Do NOT use evaluate(AggregationNumber(), state) for runtime
# computation — use _aggregation_kernel() dispatch instead.
# For tabulation, evaluate_quadrature is specialized to compute the full double integral.
@inline function integrand(::AggregationNumber, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return V * A * Np^2
end

# Rain collection by ice (riming kernel)
# ∫ V(D) × A(D) × N'(D) dD for D ≥ 100 μm
# Fortran P3: only ice particles with D ≥ 100 μm contribute to riming collection.
# Collection efficiency is applied at runtime (not in this integral).
@inline function integrand(::RainCollectionNumber, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return ifelse(D < FT(100e-6), zero(FT), V * A * Np)
end


"""
    particle_area(D, state, thresholds)

Projected cross-sectional area A(D) for ice particles.

Includes liquid fraction weighting for mixed-phase particles.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold computations within a quadrature loop.
"""
@inline function particle_area(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    Fˡ = state.liquid_fraction

    # Calculate ice area (unmodified by liquid fraction)
    A_ice = particle_area_ice_only(D, state, thresholds)

    # Liquid area (sphere)
    A_liquid = FT(π)/4 * D^2

    return (1 - Fˡ) * A_ice + Fˡ * A_liquid
end

# Backward-compatible method: compute thresholds on the fly
@inline function particle_area(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_area(D, state, thresholds)
end

#####
##### Sixth moment integrals
#####

# Sixth moment rime tendency
# Fortran (create_p3_lookupTable_1.f90 line 1552):
#   sum2 = ∫ 6*D^5 * A * V * N'(D) / dmdD dD  (for D ≥ 100 μm)
# The collection kernel (A × V) and Jacobian (1/dmdD) convert the mass-based
# riming rate into a diameter growth rate for the D^6 moment.
@inline function integrand(::SixthMomentRime, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    contribution = 6 * D^5 * A * V * Np / dmdD
    return ifelse(D >= FT(100e-6), contribution, zero(FT))
end

# Sixth moment deposition tendencies
# Fortran (line 2107-2116): includes fv × C × 6D^5 / dmdD
@inline function integrand(::SixthMomentDeposition, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * fᵛᵉ * C * Np / dmdD
end

@inline function integrand(::SixthMomentDeposition1, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * fᵛᵉ * C * Np / dmdD
end

# Sixth moment melting tendencies
# Fortran (line 1991): sum5 = ∫ capm × 6D^5 × fac1 × N'(D) / dmdD dD  (D ≤ D_crit)
# melt1 = constant ventilation, melt2 = enhanced ventilation (for small ice, D ≤ D_crit)
@inline function integrand(::SixthMomentMelt1, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    contribution = 6 * D^5 * fᵛᵉ * C * Np / dmdD
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::SixthMomentMelt2, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    contribution = 6 * D^5 * fᵛᵉ * C * Np / dmdD
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

# Sixth moment aggregation
# The single-integral integrand is retained as a fallback. For tabulation,
# evaluate_quadrature is specialized to compute the full double integral (M9 fix).
@inline function integrand(::SixthMomentAggregation, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return D^6 * V * A * Np^2
end

# Sixth moment shedding
# Fortran (line 1649): 6*D^5 * (sum3/sum4) * D^bb × N'(D) / dmdD for D ≥ 9 mm
# where bb=3 (line 348). The (sum3/sum4) ratio is constant w.r.t. D and factors
# out of the integral; D^bb = D^3 does NOT factor out and must be in the integrand.
# The integrand is thus 6*D^(5+3) = 6*D^8 for M6, and 3*D^(2+3) = 3*D^5 for M3.
# The normalization divides by sum4 = M3 moment (see normalize_integral).
@inline function integrand(::SixthMomentShedding, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    # D^(5+bb) with bb=3 → D^8
    contribution = 6 * D^8 * Np / dmdD
    return ifelse(D >= FT(0.009), contribution, zero(FT))
end

# Sixth moment sublimation tendencies
# Identical integrands to deposition (Fortran line 2132-2134 confirms same sums);
# the difference is in the normalization coefficient (factor 1 vs 2 in the M3 term).
@inline function integrand(::SixthMomentSublimation, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * fᵛᵉ * C * Np / dmdD
end

@inline function integrand(::SixthMomentSublimation1, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * fᵛᵉ * C * Np / dmdD
end

#####
##### Lambda limiter integrals
#####

@inline function integrand(::NumberMomentLambdaLimit, D, state::IceSizeDistributionState, thresholds)
    Np = size_distribution(D, state)
    return Np
end

@inline function integrand(::MassMomentLambdaLimit, D, state::IceSizeDistributionState, thresholds)
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return m * Np
end

#####
##### Rain integrals
#####

@inline integrand(::RainShapeParameter, D, state, thresholds) = zero(D)
@inline integrand(::RainVelocityNumber, D, state, thresholds) = zero(D)
@inline integrand(::RainVelocityMass, D, state, thresholds) = zero(D)
@inline integrand(::RainEvaporation, D, state, thresholds) = zero(D)

#####
##### Ice-rain collection integrals
#####

# Ice-rain collection integrands: only ice particles with D ≥ 100 μm contribute
# to riming collection (Fortran P3 v5.5.0: create_p3_lookupTable_1.f90, line 1548).
@inline function integrand(::IceRainMassCollection, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return ifelse(D < FT(100e-6), zero(FT), V * A * m * Np)
end

@inline function integrand(::IceRainNumberCollection, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return ifelse(D < FT(100e-6), zero(FT), V * A * Np)
end

@inline function integrand(::IceRainSixthMomentCollection, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return ifelse(D < FT(100e-6), zero(FT), D^6 * V * A * Np)
end
