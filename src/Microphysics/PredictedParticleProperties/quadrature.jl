#####
##### Ice geometry and quadrature helpers
#####
##### Geometric primitives for ice particles (mass, area, diameter, capacitance,
##### regime thresholds) plus shared Chebyshev–Gauss quadrature nodes/weights
##### and the piecewise rain fall-speed law used by the rain tabulation path.
#####

export chebyshev_gauss_nodes_weights

#####
##### Chebyshev–Gauss quadrature
#####

"""
$(TYPEDSIGNATURES)

Compute Chebyshev–Gauss quadrature nodes and weights for `n` points.

Returns `(nodes, weights)` for approximating

```math
∫_{-1}^{1} f(x) dx ≈ ∑ᵢ wᵢ f(xᵢ)
```

The nodes cluster near the boundaries, which helps capture rapidly-varying
contributions of size-distribution integrands.
"""
function chebyshev_gauss_nodes_weights(FT::Type{<:AbstractFloat}, n::Int)
    nodes = zeros(FT, n)
    weights = zeros(FT, n)

    for i in 1:n
        x = cos(FT((2i - 1) * π / (2n)))
        nodes[i] = x
        # Chebyshev–Gauss type 1 computes ∫ f(x)/√(1-x²) dx with weight π/n.
        # For regular integrals ∫ f(x) dx, multiply by √(1-x²).
        weights[i] = FT(π / n) * sqrt(1 - x^2)
    end

    return nodes, weights
end

chebyshev_gauss_nodes_weights(n::Int) = chebyshev_gauss_nodes_weights(Float64, n)

#####
##### Domain transform from x ∈ [-1, 1] (Chebyshev–Gauss nodes) to D ∈ [0, ∞)
#####

"""
$(TYPEDSIGNATURES)

Map a Chebyshev–Gauss node `x ∈ [-1, 1]` to a particle diameter `D ∈ [0, ∞)`
using `D = (scale/λ) (1+x)/(1-x+ε)`. The default `scale = 10` covers more than
99.99% of an exponential tail with decay length 1/λ.
"""
@inline function transform_to_diameter(x, λ; scale=10)
    FT = typeof(x)
    return FT(scale) / λ * (1 + x) / (1 - x + eps(FT))
end

"""
$(TYPEDSIGNATURES)

Jacobian `dD/dx` of the diameter transform used by [`transform_to_diameter`](@ref).
"""
@inline function jacobian_diameter_transform(x, λ; scale=10)
    FT = typeof(x)
    return FT(scale) / λ * (2 + eps(FT)) / (1 - x + eps(FT))^2
end

#####
##### Rain fall speed (Gunn–Kinzer / Beard piecewise power law)
#####

"""
$(TYPEDSIGNATURES)

Piecewise Gunn–Kinzer / Beard rain terminal velocity. Captures the Stokes-drag
regime below D ≈ 100 μm and the terminal-velocity plateau above D ≈ 5 mm.
Used by the rain quadrature tabulation path.
"""
@inline function rain_fall_speed(D, ρ_correction)
    FT = typeof(D)

    # Mass of water sphere in GRAMS for the formula
    # m3: ρ_w = 997 kg/m³ (Fortran rhow = 997)
    m_kg = (FT(π)/6) * FT(997) * D^3
    m_g = m_kg * 1000

    # Piecewise power law (Gunn–Kinzer/Beard), V in cm/s
    V_cm = ifelse(D <= FT(134.43e-6),  FT(4.5795e5) * m_g^(FT(2)/FT(3)),
           ifelse(D <  FT(1511.64e-6), FT(4.962e3)  * m_g^(FT(1)/FT(3)),
           ifelse(D <  FT(3477.84e-6), FT(1.732e3)  * m_g^(FT(1)/FT(6)),
                                       FT(917.0))))

    return V_cm * FT(0.01) * ρ_correction
end

#####
##### Ice regime thresholds and geometric primitives
#####

"""
$(TYPEDSIGNATURES)

Compute ice regime thresholds from the state's mass-diameter parameters.
Returns a NamedTuple with spherical, graupel, partial_rime thresholds and ρ_graupel.

Thresholds depend only on the state's rime properties and mass-diameter
parameters — they are independent of particle diameter D, so callers can
compute them once and reuse them across many evaluations of the geometric
helpers below.
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

@inline regime_thresholds_from_state(D, state::IceSizeDistributionState) =
    regime_thresholds_from_state(typeof(D), state)

"""
$(TYPEDSIGNATURES)

Mass of the ice portion of the particle (ignoring liquid water).

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref)
to avoid redundant threshold recomputation in tight loops.
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
    Fᶠ_safe = min(state.rime_fraction, FT(1) - eps(FT))
    a₄ = FT(α) / (1 - Fᶠ_safe)
    b₄ = FT(β)

    # Determine which regime applies (work backwards from regime 4)
    # Note: same logic and ordering as ice_mass_coefficients in lambda_solver.jl
    is_regime_4 = D ≥ thresholds.partial_rime
    is_regime_3 = D ≥ thresholds.graupel
    is_regime_2 = D ≥ thresholds.spherical

    a = ifelse(is_regime_4, a₄, a₃)
    b = ifelse(is_regime_4, b₄, b₃)

    a = ifelse(is_regime_3, a, a₂)
    b = ifelse(is_regime_3, b, b₂)

    a = ifelse(is_regime_2, a, a₁)
    b = ifelse(is_regime_2, b, b₁)

    return a * D^b
end

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
$(TYPEDSIGNATURES)

Invert the piecewise ice-only mass-diameter relation to recover diameter from mass.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref).
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
$(TYPEDSIGNATURES)

Projected area of the ice portion of the particle.

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref).
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

@inline function particle_area_ice_only(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return particle_area_ice_only(D, state, thresholds)
end

"""
$(TYPEDSIGNATURES)

Particle capacitance for vapor diffusion. P3 Fortran convention: cap=1.0 for
spheres and 0.48 for non-spherical, with `capm = cap × D` (i.e. 2× the
physical capacitance C = D/2 — the extra factor of 2 is absorbed by using
2π instead of 4π in the deposition / melting rate equations).

Accepts precomputed `thresholds` from [`regime_thresholds_from_state`](@ref).
See Pruppacher & Klett (1997) Chapter 13.
"""
@inline function capacitance(D, state::IceSizeDistributionState, thresholds)
    FT = typeof(D)
    Fˡ = state.liquid_fraction

    C_sphere = D
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

@inline function capacitance(D, state::IceSizeDistributionState)
    thresholds = regime_thresholds_from_state(D, state)
    return capacitance(D, state, thresholds)
end
