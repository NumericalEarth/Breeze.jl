"""
Small, scheme-independent helpers shared across Breeze: property sums, guarded arithmetic, the
Chebyshev–Gauss quadrature used to tabulate size-distribution integrals, and the
`@adapt_architecture` macro that generates architecture-transfer methods for
container structs.
"""
module Utils

export sum_properties,
       safe_divide,
       zero_numerator_quotient,
       chebyshev_gauss_nodes_weights,
       transform_to_diameter,
       jacobian_diameter_transform,
       @adapt_architecture

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture

#####
##### Sums of selected properties
#####

"""
$(TYPEDSIGNATURES)

Return the sum of the properties of `object` selected by the nonempty tuple `names`.
Properties may be scalars, fields, or other values supporting `+`.

A single selected property is returned unchanged. No scalar zero is added, so the
sum also supports lazy field expressions. Callers handle empty selections.

```jldoctest
using Breeze.Utils: sum_properties
sum_properties((a=1, b=2, c=4), (:a, :c))

# output
5
```
"""
@inline sum_properties(object, names::Tuple{Symbol}) = getproperty(object, first(names))

@inline function sum_properties(object, names::Tuple{Symbol, Vararg})
    value = getproperty(object, first(names))
    return value + sum_properties(object, Base.tail(names))
end

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

"""
$(TYPEDSIGNATURES)

Return `a / b`, bit for bit, formed so that a zero numerator does not reach the divider.

On a GPU a single-precision division is a reciprocal, a few fused multiply-adds and a check of the operands;
a zero numerator (like an infinite or a denormal one) fails the check, and a subroutine of some forty
instructions then runs for the whole warp. Where the numerator is a field that is exactly zero in most cells —
the mass of a hydrometeor that is absent, the latent shift of an unsaturated state — that subroutine can be a
tenth of a kernel. Here the divider is given `1` in place of the zero and the zero is restored by a product:

| `a`                  | `b`                         | result                                           |
|:---------------------|:----------------------------|:-------------------------------------------------|
| `±0`                 | normal, either sign         | `a * (1 / b)`: a zero with the sign of `a / b`   |
| `±0`                 | `±Inf`                      | `a * (±0)`: a zero with the sign of `a / b`      |
| `±0`                 | `±0`, denormal or `NaN`     | `a / b` itself                                   |
| non-zero, `±Inf`, `NaN` | anything                 | `a / b` itself                                   |

In the first two rows `1 / b` is finite (`|1 / b| ≤ 1 / floatmin < floatmax`) and, for a finite `b`, not zero
(`1 / floatmax` is above the smallest denormal), so the product of the zero `a` with it is a zero whose sign is
`sign(a) ⊻ sign(b)` — the IEEE quotient `a / b`. Both are exact, so nothing is rounded; a fused multiply-add
that absorbed the product into a following addition would add the same exact zero. A denormal `b` is left to
the divider because `1 / b` would overflow and `0 * Inf` is `NaN`, where `0 / b` is `0`.

Branchless: two comparisons, two selects and a product (seven instructions) at a site whose numerator is not
zero, so it belongs only where the numerator is zero in most cells. A select around the division
(`ifelse(a == 0, a, a / b)`) does not help — the division is still evaluated with the zero — and the compiler
rewrites `ifelse(z, a, ifelse(z, 1, a) / b)` into `a / ifelse(z, 1, b)`, which puts the zero back in the divider;
the product cannot be simplified away without knowing the sign of the zero.

For any other number type (dual numbers, traced numbers) the quotient is formed as it is written.

```jldoctest
using Breeze.Utils: zero_numerator_quotient
zero_numerator_quotient(-0.0f0, 2.0f0), zero_numerator_quotient(0.0f0, -2.0f0), zero_numerator_quotient(3.0f0, 2.0f0)

# output
(-0.0f0, -0.0f0, 1.5f0)
```
"""
@inline function zero_numerator_quotient(a::FT, b::FT) where FT <: Union{Float32, Float64}
    z = (a == 0) & (abs(b) ≥ floatmin(FT))   # a zero numerator over a normal or infinite denominator
    q = ifelse(z, one(FT), a) / b            # the divider sees 1 in place of the zero
    return ifelse(z, a * q, q)               # ±0 × (1 / b): the zero of a / b, sign included
end

@inline zero_numerator_quotient(a, b) = a / b

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

#####
##### Register-capped launches (handoff 582e072d, e490c5ed, fc4445df; CUDA method in BreezeCUDAExt)
#####

"""
$(TYPEDSIGNATURES)

Launch the KernelAbstractions kernel object `kernel!(args...; ndrange)`, on CUDA devices with at most `maxregs`
registers per thread when `maxregs > 0` (`BreezeCUDAExt`); elsewhere, or with `maxregs = 0`, as is.
"""
launch_with_maxregs!(arch, kernel!, maxregs, args...; ndrange) = (kernel!(args...; ndrange); nothing)

"""
Whether `launch_with_maxregs!` on CUDA compiles once per kernel and argument types and launches without
re-converting (bit-identical; host cost only). `BREEZE_CACHED_MAXREGS_LAUNCH=false` at load turns it off.
"""
const CACHED_MAXREGS_LAUNCH = Ref(true)

"""
Whether register caps (`launch_with_maxregs!` and the table entries `kernel_launch_maxregs` of
`kernel_launch_configurations.jl`) apply. `BREEZE_LAUNCH_MAXREGS=false` at load ignores them all.
"""
const LAUNCH_MAXREGS = Ref(true)

"""
Whether Breeze's per-kernel launch table (`kernel_launch_configurations.jl`: workgroups, k-tiles, register caps,
static shapes) applies. `BREEZE_KERNEL_LAUNCH_TABLE=false` at load makes every entry return the fallback.
"""
const KERNEL_LAUNCH_TABLE = Ref(true)

env_switch(name, default) = lowercase(get(ENV, name, string(default))) != "false"

function __init__()
    CACHED_MAXREGS_LAUNCH[] = env_switch("BREEZE_CACHED_MAXREGS_LAUNCH", true)
    LAUNCH_MAXREGS[] = env_switch("BREEZE_LAUNCH_MAXREGS", true)
    KERNEL_LAUNCH_TABLE[] = env_switch("BREEZE_KERNEL_LAUNCH_TABLE", true)
    return nothing
end

end # module Utils
