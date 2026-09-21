"""
Small, scheme-independent helpers shared across Breeze: guarded arithmetic, the
Chebyshev–Gauss quadrature used to tabulate size-distribution integrals, and the
`@adapt_architecture` macro that generates architecture-transfer methods for
container structs.
"""
module Utils

export safe_divide,
       chebyshev_gauss_nodes_weights,
       transform_to_diameter,
       jacobian_diameter_transform,
       @adapt_architecture

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture

#####
##### Guarded arithmetic
#####

"""
$(TYPEDSIGNATURES)

Return `a / b`, or `default` where `b` is exactly zero. Branch-free, so it is
safe inside GPU kernels.

```jldoctest
using Breeze.Utils: safe_divide
safe_divide(1.0, 0.0, -1.0)

# output
-1.0
```
"""
@inline safe_divide(a, b, default) = ifelse(iszero(b), default, a / b)

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

```jldoctest
using Breeze.Utils: chebyshev_gauss_nodes_weights
nodes, weights = chebyshev_gauss_nodes_weights(Float64, 4)
round(sum(weights), digits=4)

# output
2.0523
```
"""
function chebyshev_gauss_nodes_weights(FT::DataType, n::Int)
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

chebyshev_gauss_nodes_weights(n::Int) =
    chebyshev_gauss_nodes_weights(Oceananigans.defaults.FloatType, n)

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
##### Architecture transfer for container structs
#####

"""
    @adapt_architecture T

Generate `Adapt.adapt_structure` and `Oceananigans.Architectures.on_architecture`
methods for `T` that walk every field of `T` and reconstruct via the positional
constructor. `T` must already be defined when the macro is expanded.

A field-by-field walk leaves scalars untouched, because both `Adapt.adapt` and
`on_architecture` fall back to the identity for types without specific methods
(`adapt_storage(to, x) = x`, `on_architecture(arch, a) = a`), so only the array
fields are actually transferred.

The invoking module must have `Adapt` and `Oceananigans` in scope.
"""
macro adapt_architecture(T)
    fields = fieldnames(getfield(__module__, T))
    adapt_args = [:(Adapt.adapt(to, x.$f)) for f in fields]
    on_arch_args = [:(on_architecture(arch, x.$f)) for f in fields]
    return esc(quote
        Adapt.adapt_structure(to, x::$T) = $T($(adapt_args...))
        Oceananigans.Architectures.on_architecture(arch, x::$T) = $T($(on_arch_args...))
    end)
end

end # module Utils
