#####
##### Quadrature helpers
#####
##### Chebyshev–Gauss nodes/weights, the diameter transform they integrate over,
##### and the piecewise rain fall-speed law — together these tabulate the 1D rain
##### integrals at startup (`tabulate_rain_from_quadrature`). The ice-side
##### integrals are read from the P3 lookup tables and need none of this.
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
        x = cos(FT((2i - 1) * FT(π) / (2n)))
        nodes[i] = x
        # Chebyshev–Gauss type 1 computes ∫ f(x)/√(1-x²) dx with weight π/n.
        # For regular integrals ∫ f(x) dx, multiply by √(1-x²).
        weights[i] = FT(π) / n * sqrt(1 - x^2)
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

    # The fit is stated for a drop mass in grams, at ρʷ = 997 kg/m³
    m_kg = (FT(π)/6) * FT(997) * D^3
    m_g = m_kg * 1000

    # Piecewise power law (Gunn–Kinzer/Beard): branch edges in m, coefficients in
    # cm/s per g^exponent, and a terminal plateau above the largest edge. A published
    # fit, so the numbers stay with the formula they belong to.
    V_cm = ifelse(D <= FT(134.43e-6),  FT(4.5795e5) * m_g^(FT(2)/FT(3)),
           ifelse(D <  FT(1511.64e-6), FT(4.962e3)  * m_g^(FT(1)/FT(3)),
           ifelse(D <  FT(3477.84e-6), FT(1.732e3)  * m_g^(FT(1)/FT(6)),
                                       FT(917.0))))

    # cm/s → m/s
    return V_cm / 100 * ρ_correction
end
