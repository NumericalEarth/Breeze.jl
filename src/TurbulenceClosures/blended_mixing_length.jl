#####
##### Mixing length for `TKEBasedTurbulenceClosure`
#####
##### The mixing length is a dispatched component of the closure: kernels call one hook,
##### `mixing_lengthᶜᶜᶠ`, and alternative formulations are new component types selected through the
##### closure constructor. `BlendedMixingLength` composes independent length-scale branches with a
##### blending rule, so published schemes are expressed as data rather than as separate code paths.
#####

"""
$(TYPEDEF)

Supertype for the individual length scales combined by a [`BlendedMixingLength`](@ref).

A branch is a callable over `(i, j, k, grid, q, N², state)` through `length_scaleᶜᶜᶠ` and
`length_scaleᶜᶜᶜ`. Branches ignore the arguments they do not need, so that every branch presents
the same interface to the blend regardless of what it depends on.
"""
abstract type AbstractLengthScale end

"""
$(TYPEDEF)

Supertype for rules that reduce a tuple of length scales to one master length scale.
"""
abstract type AbstractLengthScaleBlend end

#####
##### Branches
#####

"""
$(TYPEDEF)

Distance to the surface, ``ℓᵍ = κ (z + ℓʳ)``, offset by a roughness length so that it stays finite
as ``z → 0``.

The plain form is deliberate. It carries the von Kármán constant explicitly rather than absorbing
it into ``Cᴷ``, so the neutral log-layer constraint on the closure coefficients (see
`stress_coefficient`) is *defaulted to* and then checked, rather than imposed by construction.

This branch carries no stability correction; [`SurfaceLayerLengthScale`](@ref) is the same scale
with MYNN's correction applied, and the two agree exactly in neutral conditions.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct GeometricLengthScale{FT} <: AbstractLengthScale
    "von Kármán constant, setting the slope ``ℓᵍ = κ(z + ℓʳ)``"
    κ :: FT = 0.4
    "roughness length ``ℓʳ``, which keeps ``ℓᵍ > 0`` at the surface, m"
    ℓʳ :: FT = 0.1
end

"""
$(TYPEDEF)

Depth over which the column is turbulent, ``ℓᵗ = Cᵗ ∫ q z \\, dz / ∫ q \\, dz``, the ``q``-weighted
centroid of the column ([Nakanishi and Niino (2009)](@cite NakanishiNiino2009), their Eq. 54).

It is not a boundary-layer depth and needs no ``zᵢ`` diagnostic, which avoids the degeneracies of
threshold detectors — a bulk-Richardson threshold has nothing to bite on in neutral air, and a
stress threshold has nothing to bite on in free convection.

This is the only non-local branch: it is a column integral, computed once per column by
`_compute_turbulence_length_scale!` and read back with no ``k`` index.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct TurbulenceLengthScale{FT} <: AbstractLengthScale
    "coefficient of ``ℓᵗ``; 0.23 in MYNN Eq. 54"
    Cᵗ :: FT = 0.23
end

"""
$(TYPEDEF)

Distance a parcel travels against stable stratification, ``ℓᵇ = Cᵇ q / N``, active only where the
column is stably stratified. The positive part of ``N²`` is taken smoothly (see `smooth_positive`),
so the branch engages continuously; in neutral or unstable air it returns a very large length and
drops out of any blend that selects the smallest scale.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct BuoyancyLengthScale{FT} <: AbstractLengthScale
    "coefficient of ``ℓᵇ = Cᵇ q / N``. The default is Deardorff's ``0.76 \\sqrt{e} / N`` rewritten
     in ``q = \\sqrt{2e}``; MYNN's realizability bound ``ℓ/q ≤ 1/N`` corresponds to the looser
     ``Cᵇ = 1``"
    Cᵇ :: FT = 0.53
    "smoothing scale for the positive part of ``N²``, s⁻². Only ``|N²| ≲ N²ᵐⁱⁿ`` is affected, where
     the branch is inactive anyway"
    N²ᵐⁱⁿ :: FT = 1e-9
    "coefficient of MYNN's convective enhancement (their Eq. 55), which lengthens ``ℓᵇ`` in the
     upper half of a convective boundary layer, where plain ``q/N`` underestimates it because TKE
     there is supplied by transport rather than by the local gradient. MYNN use 5; the default of
     zero leaves the branch as plain ``Cᵇ q / N``"
    Cᶜᵇ :: FT = 0
end

"""
$(TYPEDEF)

Distance to the surface with MYNN's stability correction
([Nakanishi and Niino (2009)](@cite NakanishiNiino2009), their Eq. 53):

```math
ℓˢ = \\begin{cases}
  ℓᵍ / Cⁿ            & ζ ≥ 1 \\\\
  ℓᵍ / (1 + Cˢ ζ)    & 0 ≤ ζ < 1 \\\\
  ℓᵍ (1 - Cᶜ ζ)^{nᶜ} & ζ < 0
\\end{cases}
```

with ``ℓᵍ = κ(z + ℓʳ)`` and ``ζ = z/L`` the Monin–Obukhov stability parameter,
``L = -u_\\star³ / (κ \\langle w'b' \\rangle)``, and ``ζ`` floored at ``ζᵐⁱⁿ`` in the unstable
branch. That floor matters: at zero mean wind ``u_\\star = 0`` exactly, so ``L = 0`` and ``ζ = -∞``
at *every* height, and the unstable branch — the only one of the three that diverges rather than
saturating — would remove the wall constraint altogether.

This is [`GeometricLengthScale`](@ref) plus the correction it omits. The two agree exactly at
``ζ = 0``, so replacing one with the other leaves a neutral column untouched; in stable air the
branch shrinks by up to ``Cⁿ``, and in unstable air it grows up to the ceiling set by ``ζᵐⁱⁿ``. The
pieces join continuously but not smoothly — the derivative in ``ζ`` jumps across ``ζ = 0``.

The fit behind Eq. 53 spans ``ζ ∈ [-3.13, 0.44]`` ([Nakanishi 2001](@cite Nakanishi2001), his
Table I and Fig. 3), so both tails are extrapolations: MYNN bound the stable one at ``ζ = 1`` and
left the unstable one open, which `ζᵐⁱⁿ` closes.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct SurfaceLayerLengthScale{FT} <: AbstractLengthScale
    "von Kármán constant"
    κ :: FT = 0.4
    "roughness length ``ℓʳ``, m"
    ℓʳ :: FT = 0.1
    "strongly-stable divisor"
    Cⁿ :: FT = 3.7
    "weakly-stable slope"
    Cˢ :: FT = 2.7
    "unstable coefficient"
    Cᶜ :: FT = 100
    "unstable exponent"
    nᶜ :: FT = 0.2
    "floor on ``ζ``, bounding the unstable branch. Nakanishi (2001)'s LES spans ``ζ ∈ [-3.13, 0.44]``
     (his Table I and Fig. 3), and the unstable branch is the only piece of Eq. 53 that diverges
     rather than saturating"
    ζᵐⁱⁿ :: FT = -4
end

#####
##### Blends
#####

"""
$(TYPEDEF)

``ℓ = \\min(ℓ₁, ℓ₂, …)`` — the simplest rule: the most restrictive branch wins outright.
"""
struct MinimumBlend <: AbstractLengthScaleBlend end

"""
$(TYPEDEF)

``1/ℓ = 1/ℓ₁ + 1/ℓ₂ + …`` — the first-power harmonic blend of
[Nakanishi and Niino (2009)](@cite NakanishiNiino2009) (their Eq. 52), after Blackadar (1962).
Smooth, and strictly smaller than every branch.

Note that it approaches the smallest branch only at *first order* in the ratio of the branches, so
where one scale is meant to dominate — the surface layer, say — the others still contaminate it
proportionally. [`PowerBlend`](@ref) is the same rule with that order as a parameter.
"""
struct HarmonicBlend <: AbstractLengthScaleBlend end

"""
$(TYPEDEF)

``ℓ^{-p} = ℓ₁^{-p} + ℓ₂^{-p} + …``, the ``p``-norm blend of
[Mason and Thomson (1992)](@cite MasonThomson1992), who match a wall scale ``κ(z + ℓʳ)`` to an
outer scale with ``n = 2``.

``p = 1`` reproduces [`HarmonicBlend`](@ref) and ``p → ∞`` approaches [`MinimumBlend`](@ref).
``p = 2`` is both the literature value and the cheapest: it needs one reciprocal square root rather than a general power.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct PowerBlend{FT} <: AbstractLengthScaleBlend
    "blend exponent"
    p :: FT = 2
end

@inline (::MinimumBlend)(ℓs) = minimum(ℓs)
@inline (::HarmonicBlend)(ℓs) = inv(sum(inv, ℓs))
@inline (blend::PowerBlend)(ℓs) = sum(ℓ -> ℓ^(-blend.p), ℓs)^(-inv(blend.p))

## `Base.@kwdef` on a `{FT}` struct generates a constructor requiring every field to share one type,
## so setting a single coefficient to an integer — `BuoyancyLengthScale(Cᵇ = 1)` — would otherwise
## find no method. Promote instead; `convert_eltype` later fixes the element type to the model's.
GeometricLengthScale(κ, ℓʳ) = GeometricLengthScale(promote(κ, ℓʳ)...)
BuoyancyLengthScale(Cᵇ, N²ᵐⁱⁿ, Cᶜᵇ) = BuoyancyLengthScale(promote(Cᵇ, N²ᵐⁱⁿ, Cᶜᵇ)...)
SurfaceLayerLengthScale(κ, ℓʳ, Cⁿ, Cˢ, Cᶜ, nᶜ, ζᵐⁱⁿ) =
    SurfaceLayerLengthScale(promote(κ, ℓʳ, Cⁿ, Cˢ, Cᶜ, nᶜ, ζᵐⁱⁿ)...)

#####
##### The composed mixing length
#####

"""
$(TYPEDEF)

Master mixing length: a tuple of [`AbstractLengthScale`](@ref) branches reduced by an
[`AbstractLengthScaleBlend`](@ref).

```julia
BlendedMixingLength(branches...; blend = MinimumBlend())
```

The default of `MinimumBlend` is the plainest rule; pass `blend` for anything else. Published
schemes are then data:

```jldoctest
using Breeze

deardorff = BlendedMixingLength(BuoyancyLengthScale(Cᵇ = 0.53))
mynn = BlendedMixingLength(GeometricLengthScale(), TurbulenceLengthScale(),
                           BuoyancyLengthScale(Cᵇ = 1); blend = HarmonicBlend())

length(mynn.branches)

# output
3
```

Two structural departures from MYNN, both toward smoothness: ``ℓᵇ`` is a *branch* rather than the
hard realizability clip ``ℓ/q ≤ 1/N``, and ``N²`` enters through a smooth positive part.

MYNN's own configuration is available but is not the default — it is
[`SurfaceLayerLengthScale`](@ref) rather than [`GeometricLengthScale`](@ref), a
[`BuoyancyLengthScale`](@ref) with `Cᵇ = 1` and `Cᶜᵇ = 5`, and a [`HarmonicBlend`](@ref).

Fields
======

$(TYPEDFIELDS)
"""
struct BlendedMixingLength{B, L}
    "the rule reducing the branches to one length"
    blend :: B
    "tuple of length-scale branches"
    branches :: L
end

BlendedMixingLength(branches::AbstractLengthScale...; blend = MinimumBlend()) =
    BlendedMixingLength(blend, branches)

"""
$(TYPEDSIGNATURES)

The mixing length of [Nakanishi and Niino (2009)](@cite NakanishiNiino2009): their three branches
(Eqs. 53–55) under the harmonic blend of their Eq. 52.

```julia
closure = TKEBasedTurbulenceClosure(mixing_length = NakanishiNiinoLengthScale(), Cq = 3)
```

Note that a closure built this way is *not* the MYNN model. The algebraic stability functions
``S_M(G_M, G_H)`` and ``S_H(G_M, G_H)`` of the Mellor–Yamada hierarchy are replaced in
[`TKEBasedTurbulenceClosure`](@ref) by a constant ``Cᴷ`` and a Richardson-dependent Prandtl number,
which makes it a ``k``–``ℓ`` closure borrowing their parameterizations rather than a level-2.5
model. What is genuinely theirs is the length scale, and that is what this constructor names.

``Cᵇ`` and ``Cᶜᵇ`` are given explicitly because they are where MYNN differs from this package's own
branch defaults — Deardorff's ``Cᵇ = 0.53`` and no convective enhancement. For anything else, build
a [`BlendedMixingLength`](@ref) directly.
"""
NakanishiNiinoLengthScale(; ℓʳ = 0.1, Cᵇ = 1, Cᶜᵇ = 5) =
    BlendedMixingLength(SurfaceLayerLengthScale(; ℓʳ),
                        TurbulenceLengthScale(),
                        BuoyancyLengthScale(; Cᵇ, Cᶜᵇ);
                        blend = HarmonicBlend())

Base.summary(ℓ::AbstractLengthScale) = string(nameof(typeof(ℓ)))
Base.summary(blend::AbstractLengthScaleBlend) = string(nameof(typeof(blend)))
Base.summary(blend::PowerBlend) = string("PowerBlend(p = ", prettysummary(blend.p), ")")

Base.summary(ml::BlendedMixingLength) =
    string("BlendedMixingLength(", join(summary.(ml.branches), ", "),
           "; blend = ", summary(ml.blend), ")")

function Base.show(io::IO, ml::BlendedMixingLength)
    print(io, "BlendedMixingLength with", '\n',
              "├── blend: ", summary(ml.blend), '\n')
    for (n, branch) in enumerate(ml.branches)
        prefix = n == length(ml.branches) ? "└── " : "├── "
        print(io, prefix, summary(branch), ": ")
        fields = propertynames(branch)
        print(io, join((string(f, " = ", prettysummary(getproperty(branch, f))) for f in fields),
                       ", "), '\n')
    end
end

#####
##### Branch evaluation. Each branch takes the full argument list and ignores what it does not use,
##### so the blend sees a uniform interface.
#####

"""
$(TYPEDSIGNATURES)

Smooth positive part of `x` with transition scale `δ > 0`: `(x + √(x² + δ²)) / 2`. Agrees with
`max(x, 0)` to within `δ/2` and is differentiable everywhere, unlike `max`.
"""
@inline smooth_positive(x, δ) = (x + sqrt(x^2 + δ^2)) / 2

@inline length_scaleᶜᶜᶠ(i, j, k, grid, ℓ::GeometricLengthScale, q, N², state) =
    ℓ.κ * (height_above_bottomᶜᶜᶠ(i, j, k, grid) + ℓ.ℓʳ)

@inline length_scaleᶜᶜᶜ(i, j, k, grid, ℓ::GeometricLengthScale, q, N², state) =
    ℓ.κ * (height_above_bottomᶜᶜᶜ(i, j, k, grid) + ℓ.ℓʳ)

@inline length_scaleᶜᶜᶠ(i, j, k, grid, ::TurbulenceLengthScale, q, N², state) =
    @inbounds state.ℓᵗ[i, j, 1]
@inline length_scaleᶜᶜᶜ(i, j, k, grid, ::TurbulenceLengthScale, q, N², state) =
    @inbounds state.ℓᵗ[i, j, 1]

@inline function length_scaleᶜᶜᶠ(i, j, k, grid, ℓ::BuoyancyLengthScale, q, N², state)
    N = sqrt(smooth_positive(N², ℓ.N²ᵐⁱⁿ))

    # MYNN Eq. 55: [1 + Cᶜᵇ √(qᶜ / (ℓᵗ N))] q/N under an unstable surface, with the convective
    # velocity qᶜ = (Jᵇ ℓᵗ)^{1/3}. Written as Jᵇ^{1/3} ℓᵗ^{-2/3} / N rather than qᶜ/(ℓᵗ N), which
    # would be Inf/Inf for the unbounded ℓᵗ of a quiescent column.
    Jᵇ = @inbounds state.Jᵇ[i, j, 1]
    ℓᵗ = @inbounds state.ℓᵗ[i, j, 1]
    ratio = cbrt(max(Jᵇ, zero(Jᵇ))) / (cbrt(ℓᵗ)^2 * N)
    enhancement = 1 + ℓ.Cᶜᵇ * sqrt(ratio)

    return enhancement * ℓ.Cᵇ * q / N
end

## `q` and `N²` are supplied at the caller's location, so the buoyancy branch needs no separate
## centered form.
@inline length_scaleᶜᶜᶜ(i, j, k, grid, ℓ::BuoyancyLengthScale, q, N², state) =
    length_scaleᶜᶜᶠ(i, j, k, grid, ℓ, q, N², state)

"""
$(TYPEDSIGNATURES)

MYNN Eq. 53. The three pieces are formed unconditionally and selected afterwards, because `ifelse`
evaluates both arms: the unstable form raises ``1 - Cᶜ ζ`` to a fractional power, which is a domain
error for ``ζ > 1/Cᶜ``, so its argument is clipped to the side of ``ζ = 0`` on which it applies.
"""
@inline function surface_layer_length(ℓ::SurfaceLayerLengthScale, z, i, j, state)
    u★² = @inbounds state.u★²[i, j, 1]
    Jᵇ = @inbounds state.Jᵇ[i, j, 1]
    ℓᵍ = ℓ.κ * (z + ℓ.ℓʳ)

    # ζ = z/L with L = -u★³/(κ Jᵇ), formed as a product so that u★ → 0 — free convection, where the
    # shear scale is irrelevant rather than zero — sends ζ → -∞ and the branch to infinity.
    u★³ = u★² * sqrt(u★²)
    ζ = -ℓ.κ * z * Jᵇ / max(u★³, eps(typeof(ℓᵍ)))

    ζ⁺ = max(ζ, 0)
    ζ⁻ = clamp(ζ, ℓ.ζᵐⁱⁿ, 0)
    weakly_stable = ℓᵍ / (1 + ℓ.Cˢ * ζ⁺)
    strongly_stable = ℓᵍ / ℓ.Cⁿ
    unstable = ℓᵍ * (1 - ℓ.Cᶜ * ζ⁻)^ℓ.nᶜ

    return ifelse(ζ ≥ 1, strongly_stable, ifelse(ζ ≥ 0, weakly_stable, unstable))
end

@inline length_scaleᶜᶜᶠ(i, j, k, grid, ℓ::SurfaceLayerLengthScale, q, N², state) =
    surface_layer_length(ℓ, height_above_bottomᶜᶜᶠ(i, j, k, grid), i, j, state)

@inline length_scaleᶜᶜᶜ(i, j, k, grid, ℓ::SurfaceLayerLengthScale, q, N², state) =
    surface_layer_length(ℓ, height_above_bottomᶜᶜᶜ(i, j, k, grid), i, j, state)

"""
$(TYPEDSIGNATURES)

Master mixing length at (Center, Center, Face): every branch evaluated, then blended.
"""
@inline function mixing_lengthᶜᶜᶠ(i, j, k, grid, ml::BlendedMixingLength, q, N², state)
    ℓs = map(branch -> length_scaleᶜᶜᶠ(i, j, k, grid, branch, q, N², state), ml.branches)
    return ml.blend(ℓs)
end

"""
$(TYPEDSIGNATURES)

`mixing_lengthᶜᶜᶠ` at cell centers, for the dissipation ``ε = Cᵋ e^{3/2}/ℓ``, which lives with ``e``
at centers.

Evaluating the geometric branch here rather than interpolating ``ℓ`` down from the faces matters in
the first cell: the face value at the surface is masked to zero, so an interpolated ``ℓ`` would be
halved (or, ignoring the masked face, doubled) exactly where the log-layer balance is judged.
"""
@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, ml::BlendedMixingLength, q, N², state)
    ℓs = map(branch -> length_scaleᶜᶜᶜ(i, j, k, grid, branch, q, N², state), ml.branches)
    return ml.blend(ℓs)
end

"""
$(TYPEDSIGNATURES)

Height above the bottom boundary, at faces and at centers — **unfloored**.

Oceananigans' same-named functions floor the height at a cell thickness
(`TurbulenceClosures.jl:190-201`), which suits an ocean bottom boundary layer but is not wanted
here. These are *new functions in this module*, not methods on the upstream ones: their signature
`(i, j, k, grid)` carries no Breeze-owned type, so extending them would be type piracy and would
silently change the mixing length of every Oceananigans closure that calls them, CATKE included.
"""
@inline height_above_bottomᶜᶜᶠ(i, j, k, grid) =
    clip(znode(i, j, k, grid, Center(), Center(), Face()) - z_bottom(i, j, grid))

@inline height_above_bottomᶜᶜᶜ(i, j, k, grid) =
    clip(znode(i, j, k, grid, Center(), Center(), Center()) - z_bottom(i, j, grid))

@inline clip(x) = max(0, x)

#####
##### The column integral behind ℓᵗ
#####

"""
$(TYPEDSIGNATURES)

``Cᵗ`` of the `TurbulenceLengthScale` branch, or `nothing` if the mixing length has none. Resolved
by dispatch on the branch tuple, so the search costs nothing at run time.
"""
@inline turbulence_length_coefficient(ml::BlendedMixingLength) =
    turbulence_length_coefficient(ml.branches)

@inline turbulence_length_coefficient(::Tuple{}) = nothing
@inline turbulence_length_coefficient(branches::Tuple{TurbulenceLengthScale, Vararg}) =
    first(branches).Cᵗ
@inline turbulence_length_coefficient(branches::Tuple) =
    turbulence_length_coefficient(Base.tail(branches))

"""
$(TYPEDSIGNATURES)

The von Kármán constant carried by whichever branch measures distance to the surface —
[`GeometricLengthScale`](@ref) or [`SurfaceLayerLengthScale`](@ref) — or `nothing` if there is
neither.

A surface drag law that is meant to be consistent with the closure's own log layer must use the
same ``κ`` the mixing length does; reading it back rather than repeating the literal is what keeps
the two from drifting apart.
"""
@inline von_karman_constant(ml::BlendedMixingLength) = von_karman_constant(ml.branches)

@inline von_karman_constant(::Tuple{}) = nothing
@inline von_karman_constant(branches::Tuple{GeometricLengthScale, Vararg}) = first(branches).κ
@inline von_karman_constant(branches::Tuple{SurfaceLayerLengthScale, Vararg}) = first(branches).κ
@inline von_karman_constant(branches::Tuple) = von_karman_constant(Base.tail(branches))

"""
$(TYPEDSIGNATURES)

Compute the ``q``-weighted centroid ``ℓᵗ = Cᵗ ∫ q z \\, dz / ∫ q \\, dz`` for every column. Launched
only when the mixing length carries a `TurbulenceLengthScale` branch; otherwise ``ℓᵗ`` keeps the
`Inf` it was constructed with, which drops out of every blend.

The integrand is ``q - qᵐⁱⁿ`` rather than ``q``. With ``e`` floored at a small positive value the
free atmosphere contributes to both integrals, and the ``z`` weighting amplifies that contribution
in the numerator: on a 2 km column the error is a fraction of a percent, but on a 20 km column it is
comparable to the boundary-layer signal, and ``ℓᵗ`` would then grow with domain height rather than
with the turbulence. Subtracting the floor removes the quiescent contribution exactly.
"""
@kernel function _compute_turbulence_length_scale!(ℓᵗ, grid, closure, e)
    i, j = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    Cᵗ = turbulence_length_coefficient(closure_ij.mixing_length)
    eᵐⁱⁿ = closure_ij.eᵐⁱⁿ

    # `q = √(2e)` weights both integrals, so the √2 cancels out of the ratio and only √e is formed.
    qᵐⁱⁿ = sqrt(eᵐⁱⁿ)

    FT = eltype(grid)

    ∫qz = zero(grid)
    ∫q  = zero(grid)

    for k in 1:size(grid, 3)
        eᵢ = @inbounds e[i, j, k]
        q  = sqrt(max(eᵐⁱⁿ, eᵢ))
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        z  = height_above_bottomᶜᶜᶜ(i, j, k, grid)

        active = !inactive_cell(i, j, k, grid)
        w = (q - qᵐⁱⁿ) * Δz * active

        ∫qz += w * z
        ∫q  += w
    end

    # A quiescent column has ∫q = 0; the remaining branches then set ℓ on their own, which every
    # blend achieves if ℓᵗ is large rather than zero.
    ℓ = Cᵗ * ∫qz / ∫q
    @inbounds ℓᵗ[i, j, 1] = ifelse(∫q > 0, FT(ℓ), FT(Inf))
end
