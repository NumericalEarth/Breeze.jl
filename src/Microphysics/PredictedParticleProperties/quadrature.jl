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

# Gravitational acceleration matching the Fortran P3 lookup table code
# (create_p3_lookupTable_1.f90 line 325: g = 9.861)
const P3_REF_G = 9.861

# Slinn (1983) collection efficiency constants
# (create_p3_lookupTable_1.f90 lines 335-338)
const P3_BOLTZMANN = 1.3806503e-23   # Boltzmann constant [J/K]
const P3_MEAN_FREE_PATH = 0.0256e-6  # Mean free path of air [m]
const P3_DAW = 0.04e-6               # Water-friendly aerosol diameter [m]
const P3_DAI = 0.8e-6                # Ice-friendly aerosol diameter [m]

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

"""
    MidpointNode{FT}

Wrapper around a diameter value for midpoint-rule quadrature.

When used with [`transform_to_diameter`](@ref) and [`jacobian_diameter_transform`](@ref),
returns identity values (D itself and Jacobian = 1), so all existing Chebyshev integration
loops work unchanged via dispatch.
"""
struct MidpointNode{FT}
    D :: FT
end

"""
$(TYPEDSIGNATURES)

Construct midpoint-rule quadrature nodes and weights matching the Fortran P3 convention.

Returns `(nodes::Vector{MidpointNode}, weights::Vector{FT})` where each node stores
a diameter value `D_i = (i - 0.5) × bin_width` and each weight equals `bin_width`.

Fortran defaults: `bin_width = 2μm`, `num_bins = 40000` for single integrals;
`bin_width = 50μm`, `num_bins = 1500` for collection (double) integrals.
"""
function midpoint_nodes_weights(FT::Type{<:AbstractFloat} = Float64;
                                bin_width::Real = 2e-6,
                                num_bins::Int = 40000)
    dd = FT(bin_width)
    nodes   = [MidpointNode(FT((i - 0.5) * dd)) for i in 1:num_bins]
    weights = fill(dd, num_bins)
    return nodes, weights
end

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

# Identity transform for midpoint-rule nodes (already in diameter space)
@inline transform_to_diameter(node::MidpointNode, λ; scale=10) = node.D

"""
    jacobian_diameter_transform(x, λ; scale=10)

Jacobian dD/dx for the diameter transformation.
"""
@inline function jacobian_diameter_transform(x, λ; scale=10)
    ε = eps(typeof(x))
    denom = (1 - x + ε)^2
    return scale / λ * 2 / denom
end

# Identity Jacobian for midpoint-rule nodes (weight already equals bin width)
@inline jacobian_diameter_transform(::MidpointNode{FT}, λ; scale=10) where {FT} = one(FT)

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
    g = FT(P3_REF_G)

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

    a₁ = MH_C₂ * (term - 1)^2 / max(X^b₁, eps(FT))

    # Velocity formula derived from MH2005 power law fit Re = a X^b
    # V = a₁ * ν^(1-2b₁) * (2 m g / (ρ A))^b₁ * D^(2b₁ - 1)
    # Fortran always uses MH2005 (no Stokes regime switch)
    term_bracket = 2 * m * g / (ρ_ref * A_safe)
    V = a₁ * ν_ref^(1 - 2*b₁) * term_bracket^b₁ * D^(2*b₁ - 1)

    return V
end

"""
    rain_fall_speed(D, ρ_correction)

Compute rain fall speed using piecewise power laws from P3 Fortran.
"""
@inline function rain_fall_speed(D, ρ_correction)
    FT = typeof(D)

    # Mass of water sphere in GRAMS for the formula
    # m3: ρ_w = 997 kg/m³ (Fortran rhow = 997)
    m_kg = (FT(π)/6) * FT(997) * D^3
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

    # Clamp Fᶠ away from 0 and 1 to avoid 0*Inf=NaN in IEEE arithmetic
    Fᶠ_safe = clamp(Fᶠ, eps(FT), 1 - eps(FT))

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
# H12: Fortran P3 v5.5.0 (create_p3_lookupTable_1.f90, lines 1985-2006) uses
# the **dry ice PSD** (n0d, mu_id, lamd) for melting integrals, NOT the wet (total) PSD.
# The dry PSD from Cholette et al. (2019) represents only the ice portion when Fl > 0.
# Uses `dry_size_distribution(D, state)` which analytically adjusts λ and N₀ for dry mass.
# For Fl = 0, dry = wet (no change). For Fl > 0, correctly excludes liquid water.
@inline function integrand(::SmallIceVentilationConstant, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::SmallIceVentilationReynolds, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationConstant, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    contribution = fᵛᵉ * C * Np
    return ifelse(D > D_crit, contribution, zero(D))
end

@inline function integrand(::LargeIceVentilationReynolds, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
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

# Reflectivity (Fortran `refl` / sum5 convention):
# refl = ∫ (6/(π ρ_ice))² × m(D)² × N'(D) dD
# No Rayleigh prefactor — matches Fortran Table 1 `refl` (sum5).
@inline function integrand(::Reflectivity, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    ρ_ice = FT(917.0)
    K_refl = (6 / (FT(π) * ρ_ice))^2
    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)
    return K_refl * m^2 * Np
end

# M9: Rayleigh-scattering reflectivity (Fortran `refl2` convention):
# Includes the dielectric factor |K_w|² = 0.1892 for liquid water at 10 cm wavelength.
# Fortran (create_p3_lookupTable_1.f90, lines 1227-1268) uses three regimes:
#   Fl = 0:           0.1892 × (6/(π×917))² × m² × N'(D)
#   Fl intermediate:  rayleigh_soak_wetice (Maxwell-Garnett mixing)
#   Fl = 1:           D⁶ × N'(D)  (pure water drops)
# The mixed-phase branch matches the Fortran Maxwell-Garnett dielectric mixing.
@inline function ray_complex_water(FT, λ, T_celsius)
    πFT = FT(π)
    epsinf = FT(5.27137) + FT(0.02164740) * T_celsius - FT(0.00131198) * T_celsius^2
    epss = FT(78.54)
    α = FT(-16.8129) / (T_celsius + FT(273.16)) + FT(0.0609265)
    λs = FT(0.00033836) * exp(FT(2513.98) / (T_celsius + FT(273.16))) * FT(1e-2)
    ratio = λs / λ
    denom = FT(1) + FT(2) * ratio^(FT(1) - α) * sin(α * πFT / FT(2)) + ratio^(FT(2) - FT(2) * α)
    epsr = epsinf + (epss - epsinf) * (ratio^(FT(1) - α) * sin(α * πFT / FT(2)) + FT(1)) / denom
    epsi = (epss - epsinf) * ratio^(FT(1) - α) * cos(α * πFT / FT(2)) / denom +
           λ * FT(1.25664) / FT(1.88496)
    return sqrt(complex(epsr, -epsi))
end

@inline function maetzler_complex_ice(FT, λ, T_celsius)
    c = FT(2.99e8)
    T_kelvin = T_celsius + FT(273.16)
    f = c / λ * FT(1e-9)
    B1 = FT(0.0207)
    B2 = FT(1.16e-11)
    b = FT(335)
    delta_beta = exp(FT(-10.02) + FT(0.0364) * (T_kelvin - FT(273.16)))
    beta_m = (B1 / T_kelvin) * (exp(b / T_kelvin) / (exp(b / T_kelvin) - FT(1))^2) + B2 * f^2
    beta = beta_m + delta_beta
    theta = FT(300) / T_kelvin - FT(1)
    alpha = (FT(0.00504) + FT(0.0062) * theta) * exp(FT(-22.1) * theta)
    ε = complex(FT(3.1884) + FT(9.1e-4) * (T_kelvin - FT(273.16)), alpha / f + beta * f)
    return sqrt(conj(ε))
end

@inline function maxwell_garnett_refractive_index(m1, m2, m3, vol1, vol2, vol3, inclusion::Symbol)
    m1_squared = m1^2
    m2_squared = m2^2
    m3_squared = m3^2

    if inclusion === :spherical
        β2 = 3 * m1_squared / (m2_squared + 2 * m1_squared)
        β3 = 3 * m1_squared / (m3_squared + 2 * m1_squared)
    else
        β2 = 2 * m1_squared / (m2_squared - m1_squared) *
             (m2_squared / (m2_squared - m1_squared) * log(m2_squared / m1_squared) - 1)
        β3 = 2 * m1_squared / (m3_squared - m1_squared) *
             (m3_squared / (m3_squared - m1_squared) * log(m3_squared / m1_squared) - 1)
    end

    return sqrt(((1 - vol2 - vol3) * m1_squared + vol2 * β2 * m2_squared + vol3 * β3 * m3_squared) /
                (1 - vol2 - vol3 + vol2 * β2 + vol3 * β3))
end

@inline function wet_ice_rayleigh_factor(D, total_mass, liquid_fraction)
    FT = typeof(D + total_mass + liquid_fraction)
    λ_radar = FT(0.10)
    m_air = complex(one(FT), zero(FT))
    m_water = ray_complex_water(FT, λ_radar, zero(FT))
    m_ice = maetzler_complex_ice(FT, λ_radar, zero(FT))
    K_w = abs((m_water^2 - 1) / (m_water^2 + 2))^2

    mass_water = liquid_fraction * total_mass
    volume_total = FT(π) / FT(6) * D^3
    vol_ice = (total_mass - mass_water) / (volume_total * FT(900))
    vol_water = mass_water / (FT(1000) * volume_total)
    vol_air = FT(1) - vol_ice - vol_water

    # Rayleigh_soak_wetice: inner mix uses water matrix + ice inclusions
    # (matrix='water', inclusion='spheroidal'), outer uses icewater matrix + air inclusions
    # (hostmatrix='icewater', hostinclusion='spheroidal').
    vol_ice_frac = vol_ice / max(vol_ice + vol_water, FT(1e-10))
    vol_water_frac = FT(1) - vol_ice_frac
    # Step 1: ice inclusions in water matrix (Fortran get_m_mix with matrix='water')
    icewater_mix = maxwell_garnett_refractive_index(m_water, m_air, m_ice, FT(0), FT(0), vol_ice_frac, :spheroidal)
    # Step 2: air inclusions in icewater matrix (Fortran get_m_mix with matrix='ice' via hostmatrix='icewater')
    particle_mix = maxwell_garnett_refractive_index(icewater_mix, m_air, 2 * m_air, FT(0), vol_air, FT(0), :spheroidal)

    return abs((particle_mix^2 - 1) / (particle_mix^2 + 2))^2 / K_w * D^6
end

@inline function integrand(::RayleighReflectivity, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    ρ_ice = FT(917.0)
    Fˡ = state.liquid_fraction

    m = particle_mass(D, state, thresholds)
    Np = size_distribution(D, state)

    # Dry ice: |K_w|² × (6/(π ρ_ice))² × m²
    K_w_sq = FT(0.1892)
    K_refl = (6 / (FT(π) * ρ_ice))^2
    refl_ice = K_w_sq * K_refl * m^2 * Np

    # Pure water: D⁶ (equivalent reflectivity for water drops)
    refl_water = D^6 * Np

    refl_mixed = wet_ice_rayleigh_factor(D, m, Fˡ) * Np

    return ifelse(Fˡ <= zero(FT), refl_ice, ifelse(Fˡ >= one(FT), refl_water, refl_mixed))
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

    # Effective density = mass / volume.
    # D is always positive in quadrature, so V > 0.  Guard only against
    # exact zero (not eps(FT), which is ~2e-16 and larger than the volume
    # of micrometer-scale particles).
    return m / max(V, FT(1e-100))
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
# (V × A × N'²) which has different magnitude than the Fortran double integral.
# At runtime, the Fortran-generated lookup tables provide the correct double-integral
# values (nagg column). This single-integral integrand is NOT used for production;
# it exists only as a fallback for unit testing without tables.
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

# Slinn (1983) collection efficiency for aerosol scavenging by ice particles.
# Faithfully translates Fortran P3 create_p3_lookupTable_1.f90 lines 1560-1586.
# All constants (mu, rho, etc.) match the Fortran reference conditions used for
# lookup-table generation.
#
# Arguments:
#   D_ice   — ice particle diameter [m]
#   V_ice   — ice particle terminal velocity [m/s]
#   D_aer   — aerosol (or cloud droplet) diameter [m]
@inline function slinn_collection_efficiency(D_ice, V_ice, D_aer)
    FT = typeof(D_ice)

    # Reference conditions matching Fortran table generation
    ρ  = FT(P3_REF_RHO)           # air density [kg/m³]
    ν  = FT(P3_REF_NU)            # kinematic viscosity [m²/s] (Fortran "mu")
    T  = FT(P3_REF_T)             # temperature [K]
    kB = FT(P3_BOLTZMANN)         # Boltzmann constant [J/K]
    λ_mfp = FT(P3_MEAN_FREE_PATH) # mean free path [m]

    # Reynolds number (Fortran: Re = 0.5*rho*d1*fall1(jj)/mu)
    Re = FT(0.5) * ρ * D_ice * V_ice / ν

    # Cunningham slip correction (Fortran: wcc)
    wcc = 1 + 2 * λ_mfp / D_aer * (FT(1.257) + FT(0.4) * exp(FT(-0.55) * D_aer / λ_mfp))

    # Brownian diffusivity (Fortran: diffin = boltzman*t*wcc/(3.*pi*mu*Daw))
    D_B = kB * T * wcc / (3 * FT(π) * ν * D_aer)

    # Schmidt number (Fortran: Sc = mu/(rho*diffin))
    Sc = ν / (ρ * D_B)

    # Stokes number (Fortran: St = Daw*Daw*fall1(jj)*1000.*wcc/(9.*mu*d1))
    St = D_aer^2 * V_ice * FT(1000) * wcc / (9 * ν * D_ice)

    # Critical Stokes number (Fortran: aval, St2)
    aval = log(1 + Re)
    St2 = (FT(1.2) + aval / 12) / (1 + aval)

    # Slinn (1983) collection efficiency
    # Brownian diffusion + interception terms
    Eff = 4 / (Re * Sc) * (1 + FT(0.4) * Re^FT(0.5) * Sc^FT(0.3333) +
              FT(0.16) * Re^FT(0.5) * Sc^FT(0.5)) +
          4 * D_aer / D_ice * (FT(0.02) + D_aer / D_ice * (1 + 2 * Re^FT(0.5)))

    # Inertial impaction term (only when St > St2)
    # Use max(0, ...) to keep the base non-negative, since ifelse evaluates both branches.
    ΔSt = max(0, St - St2)
    Eff = Eff + ifelse(St > St2, (ΔSt / (ΔSt + FT(0.666667)))^FT(1.5), zero(FT))

    # Clamp to [1e-5, 1]
    return clamp(Eff, FT(1e-5), one(FT))
end

# Slinn (1983) aerosol collection by water-friendly aerosol: ∫ E(D) × V(D) × A(D) × N'(D) dD
# Fortran P3: nawcol. Uses Daw = 0.04 μm (water-friendly aerosol diameter).
@inline function integrand(::CloudAerosolCollection, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    Eff = slinn_collection_efficiency(D, V, FT(P3_DAW))
    return V * A * Eff * Np
end

# Slinn (1983) aerosol collection by ice-friendly aerosol: ∫ E(D) × V(D) × A(D) × N'(D) dD
# Fortran P3: naicol. Uses Dai = 0.8 μm (ice-friendly aerosol diameter).
@inline function integrand(::IceAerosolCollection, D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    Eff = slinn_collection_efficiency(D, V, FT(P3_DAI))
    return V * A * Eff * Np
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
# H12: Uses dry PSD for melting (same fix as mass melting integrands above).
@inline function integrand(::SixthMomentMelt1, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    contribution = 6 * D^5 * fᵛᵉ * C * Np / dmdD
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

@inline function integrand(::SixthMomentMelt2, D, state::IceSizeDistributionState, thresholds)
    D_crit = thresholds.spherical
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    contribution = 6 * D^5 * fᵛᵉ * C * Np / dmdD
    return ifelse(D ≤ D_crit, contribution, zero(D))
end

# D32: All-D sixth moment melting (Fortran f1pr30/f1pr31).
# Used in the non-liquid-fraction path where all particles contribute to Z melting.
@inline function integrand(::SixthMomentMeltAll1, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = melt_ventilation_factor(D, state, true, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * fᵛᵉ * C * Np / dmdD
end

@inline function integrand(::SixthMomentMeltAll2, D, state::IceSizeDistributionState, thresholds)
    fᵛᵉ = melt_ventilation_factor(D, state, false, thresholds)
    C = capacitance(D, state, thresholds)
    Np = dry_size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * fᵛᵉ * C * Np / dmdD
end

# Sixth moment aggregation (L9 parity note)
# Fortran (create_p3_lookupTable_1.f90:1415-1470) computes m6agg via a double
# integral with three-term normalization: mom6/mom3² × nagg + 1/mom3² × (sum2+sum3)
# - 2×mom6/mom3³ × (sum4+sum5). At runtime, the Fortran-generated lookup tables
# provide the correct double-integral m6agg values. This single-integral Wisner
# approximation (D⁶ × V × A × N'²) is NOT used for production; it exists only
# as a fallback for unit testing without tables.
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

# M8 PARITY NOTE: These are **single-integral approximations** over the ice PSD.
# The Fortran P3 (create_p3_lookupTable_1.f90, lines 1820-1870) computes a DOUBLE
# integral over both the ice PSD and a binned rain distribution, using the full
# gravitational collection kernel: K = (√A_ice + √A_rain)² × |V_ice - V_rain|.
# These single-integral integrands omit: (1) rain PSD integration, (2) rain
# cross-section, (3) differential fall speed. They are retained as fallback
# integrands for unit tests; the runtime uses Fortran-generated Table 2 data.
# The 100 μm threshold applies only to cloud-riming (Fortran line 1548), NOT to
# ice-rain collection which integrates over ALL ice sizes.

# Rain mass collected by ice (Fortran sum2/qrrain/f1pr08):
# Single-integral approximation: ∫ V(D) A(D) N'(D) dD (ice-side kernel only)
@inline function integrand(::IceRainMassCollection, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return V * A * Np
end

# Rain number collected by ice (Fortran sum1/nrrain/f1pr07):
# Same as mass but without rain mass weighting (handled by double integral).
@inline function integrand(::IceRainNumberCollection, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    return V * A * Np
end

# Sixth moment change from rain collection (Fortran sum3/m6collr):
# Fortran: ∫∫ K × 6D^5 × m_rain × N'_ice × N'_rain / dmdD dD1 dD2
# Single-integral approximation: use D^5 / dmdD (M6 Jacobian) instead.
@inline function integrand(::IceRainSixthMomentCollection, D, state::IceSizeDistributionState, thresholds)
    V = terminal_velocity(D, state, thresholds)
    A = particle_area(D, state, thresholds)
    Np = size_distribution(D, state)
    dmdD = particle_mass_derivative(D, state, thresholds)
    return 6 * D^5 * V * A * Np / dmdD
end
