#####
##### Mixing length for `TKEBasedTurbulenceClosure`
#####
##### The mixing length is a dispatched component of the closure, mirroring `Smagorinsky.coefficient`:
##### kernels call one hook, `mixing_lengthᶜᶜᶠ`, and alternative formulations are new component
##### types selected through the closure constructor. `MesoscaleLengthScale` is the only
##### formulation in v1.
#####

"""
$(TYPEDEF)

Mixing length for a mesoscale, ensemble-mean boundary layer, in which no turbulent motion is
resolved. Structurally this is the three-branch harmonic blend of MYNN
([Nakanishi and Niino (2009)](@cite NakanishiNiino2009), their Eqs. 52–55),

```math
1/ℓ = 1/ℓᵍ + 1/ℓᵗ + 1/ℓᵇ
```

with

```math
ℓᵍ = κ (z + ℓʳ), \\qquad ℓᵗ = Cᵗ ∫ q z \\, dz / ∫ q \\, dz, \\qquad ℓᵇ = Cᵇ q / N
```

where ``q = \\sqrt{2e}``. The three branches are the distance to the surface, the depth over which
turbulence is organized, and the distance a parcel travels against stable stratification.

No filter width appears anywhere: ``ℓ`` is a *flow* scale, not a grid scale. That is deliberate —
``ℓ ∝ Δz`` is only defensible when the grid spacing lies inside the inertial subrange, which is not
the mesoscale regime this component targets. A filter-width sibling component is the natural way to
add the large-eddy limit later, selected by dispatch rather than by a blending function.

Two departures from MYNN, both in the direction of smoothness:

  - ``ℓᵇ`` is a *branch of the harmonic blend* rather than the hard realizability clip
    ``ℓ/q ≤ 1/N``, and ``N²`` enters through a smooth positive part, so the stable limit engages
    continuously instead of switching on.

  - ``ℓᵗ`` is the ``q``-weighted centroid of the column times one literature constant. It is not a
    boundary-layer depth and needs no ``zᵢ`` diagnostic, which avoids the degeneracies of threshold
    detectors — a bulk-Richardson threshold has nothing to bite on in neutral air, and a
    stress-threshold has nothing to bite on in free convection.

Fields
======

$(TYPEDFIELDS)
"""
Base.@kwdef struct MesoscaleLengthScale{FT}
    "von Kármán constant, setting the slope of the geometric branch ``ℓᵍ = κ(z + ℓʳ)``"
    κ :: FT = 0.4
    "coefficient of the turbulence length scale ``ℓᵗ``; 0.23 in MYNN Eq. 54"
    Cᵗ :: FT = 0.23
    "coefficient of the buoyancy length scale ``ℓᵇ = Cᵇ q / N``. The default is Deardorff's
     ``0.76 \\sqrt{e} / N`` rewritten in ``q = \\sqrt{2e}``; MYNN's realizability bound
     ``ℓ/q ≤ 1/N`` corresponds to the looser ``Cᵇ = 1``"
    Cᵇ :: FT = 0.53
    "roughness length ``ℓʳ``, which keeps ``ℓᵍ`` finite at the surface, m"
    ℓʳ :: FT = 0.1
    "smoothing scale for the positive part of ``N²`` in the buoyancy branch, s⁻². Only
     ``|N²| ≲ N²ᵐⁱⁿ`` is affected, where the branch is inactive anyway"
    N²ᵐⁱⁿ :: FT = 1e-9
end

Base.summary(::MesoscaleLengthScale) = "MesoscaleLengthScale"

Base.show(io::IO, ℓ::MesoscaleLengthScale) =
    print(io, "MesoscaleLengthScale with", '\n',
              "├── κ:     ", prettysummary(ℓ.κ), " (von Kármán)", '\n',
              "├── Cᵗ:    ", prettysummary(ℓ.Cᵗ), " (turbulence length scale)", '\n',
              "├── Cᵇ:    ", prettysummary(ℓ.Cᵇ), " (buoyancy length scale)", '\n',
              "├── ℓʳ:    ", prettysummary(ℓ.ℓʳ), " m (roughness length)", '\n',
              "└── N²ᵐⁱⁿ: ", prettysummary(ℓ.N²ᵐⁱⁿ), " s⁻² (buoyancy-branch smoothing)")

#####
##### The three branches, each separately testable
#####

"""
$(TYPEDSIGNATURES)

Smooth positive part of `x` with transition scale `δ > 0`: `(x + √(x² + δ²)) / 2`. Agrees with
`max(x, 0)` to within `δ/2` and is differentiable everywhere, unlike `max`.
"""
@inline smooth_positive(x, δ) = (x + sqrt(x^2 + δ^2)) / 2

"""
$(TYPEDSIGNATURES)

Geometric branch ``ℓᵍ = κ (z + ℓʳ)``: the distance to the surface, offset by the roughness length
so that ``ℓᵍ`` stays finite as ``z → 0``.

The plain form is kept on purpose. It carries the von Kármán constant explicitly rather than
absorbing it into ``Cᴷ``, so that the neutral log-layer constraint on the coefficients (see
`stress_coefficient`) is *defaulted to* and then checked, rather than imposed by construction.
"""
@inline function geometric_length_scaleᶜᶜᶠ(i, j, k, grid, mixing_length)
    z = height_above_surfaceᶜᶜᶠ(i, j, k, grid)
    return mixing_length.κ * (z + mixing_length.ℓʳ)
end

"""
$(TYPEDSIGNATURES)

`geometric_length_scaleᶜᶜᶠ` at cell centers.
"""
@inline function geometric_length_scaleᶜᶜᶜ(i, j, k, grid, mixing_length)
    z = height_above_surfaceᶜᶜᶜ(i, j, k, grid)
    return mixing_length.κ * (z + mixing_length.ℓʳ)
end

"""
$(TYPEDSIGNATURES)

Height above the surface, at faces and at centers.

Oceananigans' `height_above_bottomᶜᶜᶠ`/`ᶜᶜᶜ` are *not* usable here: they floor the result at one
cell thickness, which suits an ocean bottom boundary layer but destroys ``ℓᵍ = κ(z + ℓʳ)`` in
exactly the cells where the log law is judged, and makes ``ℓᵍ(z₁)`` grid-dependent. The roughness
length is what keeps ``ℓᵍ`` finite at the surface, so no floor is wanted.
"""
@inline height_above_surfaceᶜᶜᶠ(i, j, k, grid) =
    clip(znode(i, j, k, grid, Center(), Center(), Face()) - z_bottom(i, j, grid))

@inline height_above_surfaceᶜᶜᶜ(i, j, k, grid) =
    clip(znode(i, j, k, grid, Center(), Center(), Center()) - z_bottom(i, j, grid))

@inline clip(x) = max(zero(x), x)

"""
$(TYPEDSIGNATURES)

Buoyancy branch ``ℓᵇ = Cᵇ q / N``, active only where the column is stably stratified. The positive
part of ``N²`` is taken smoothly (see `smooth_positive`), so the branch engages continuously; in
neutral or unstable air it returns a very large length and drops out of the harmonic blend.
"""
@inline function buoyancy_length_scaleᶜᶜᶠ(i, j, k, grid, mixing_length, q, N²)
    N²⁺ = smooth_positive(N², mixing_length.N²ᵐⁱⁿ)
    return mixing_length.Cᵇ * q / sqrt(N²⁺)
end

"""
$(TYPEDSIGNATURES)

Turbulence branch ``ℓᵗ = Cᵗ ∫ q z \\, dz / ∫ q \\, dz``, the ``q``-weighted centroid of the column.
Read from the column field computed by `compute_turbulence_length_scale!`, which is constant within
a column and therefore carries no `k` index.
"""
@inline turbulence_length_scaleᶜᶜᶠ(i, j, k, grid, ℓᵗ) = @inbounds ℓᵗ[i, j, 1]

"""
$(TYPEDSIGNATURES)

Master mixing length, the first-power harmonic blend of the three branches. Working with the
reciprocals keeps the blend finite when any single branch is unbounded — which is the normal case
for ``ℓᵇ`` in neutral air.
"""
@inline function mixing_lengthᶜᶜᶠ(i, j, k, grid, mixing_length::MesoscaleLengthScale, q, N², ℓᵗ)
    ℓᵍ⁻¹ = 1 / geometric_length_scaleᶜᶜᶠ(i, j, k, grid, mixing_length)
    ℓᵇ⁻¹ = 1 / buoyancy_length_scaleᶜᶜᶠ(i, j, k, grid, mixing_length, q, N²)
    ℓᵗ⁻¹ = 1 / turbulence_length_scaleᶜᶜᶠ(i, j, k, grid, ℓᵗ)
    return 1 / (ℓᵍ⁻¹ + ℓᵗ⁻¹ + ℓᵇ⁻¹)
end

"""
$(TYPEDSIGNATURES)

`mixing_lengthᶜᶜᶠ` at cell centers, for the dissipation ``ε = Cᵋ e^{3/2}/ℓ``, which lives with
``e`` at centers.

Evaluating the geometric branch here rather than interpolating ``ℓ`` down from the faces matters
in the first cell: the face value at the surface is masked to zero, so an interpolated ``ℓ`` would
be halved (or, ignoring the masked face, doubled) exactly where the log-layer balance is judged.
"""
@inline function mixing_lengthᶜᶜᶜ(i, j, k, grid, mixing_length::MesoscaleLengthScale, q, N², ℓᵗ)
    ℓᵍ⁻¹ = 1 / geometric_length_scaleᶜᶜᶜ(i, j, k, grid, mixing_length)
    ℓᵇ⁻¹ = 1 / buoyancy_length_scaleᶜᶜᶠ(i, j, k, grid, mixing_length, q, N²)
    ℓᵗ⁻¹ = 1 / turbulence_length_scaleᶜᶜᶠ(i, j, k, grid, ℓᵗ)
    return 1 / (ℓᵍ⁻¹ + ℓᵗ⁻¹ + ℓᵇ⁻¹)
end

#####
##### The column integral behind ℓᵗ
#####

"""
$(TYPEDSIGNATURES)

Compute the ``q``-weighted centroid ``ℓᵗ = Cᵗ ∫ q z \\, dz / ∫ q \\, dz`` for every column.

The integrand is ``q - qᵐⁱⁿ`` rather than ``q``. With ``e`` floored at a small positive value the
free atmosphere contributes to both integrals, and the ``z`` weighting amplifies that contribution
in the numerator: on a 2 km column the error is a fraction of a percent, but on a 20 km column it
is comparable to the boundary-layer signal, and ``ℓᵗ`` would then grow with domain height rather
than with the turbulence. Subtracting the floor removes the quiescent contribution exactly.
"""
@kernel function _compute_turbulence_length_scale!(ℓᵗ, grid, closure, e)
    i, j = @index(Global, NTuple)

    closure_ij = getclosure(i, j, closure)
    mixing_length = closure_ij.mixing_length
    eᵐⁱⁿ = closure_ij.eᵐⁱⁿ
    qᵐⁱⁿ = sqrt(2 * eᵐⁱⁿ)

    ∫qz = zero(grid)
    ∫q  = zero(grid)

    for k in 1:size(grid, 3)
        eᵢ = @inbounds e[i, j, k]
        q  = sqrt(2 * max(eᵐⁱⁿ, eᵢ))
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        z  = height_above_surfaceᶜᶜᶜ(i, j, k, grid)

        active = !inactive_cell(i, j, k, grid)
        w = (q - qᵐⁱⁿ) * Δz * active

        ∫qz += w * z
        ∫q  += w
    end

    # A quiescent column has ∫q = 0; the geometric branch then sets ℓ on its own, which the
    # harmonic blend achieves if ℓᵗ is large rather than zero.
    FT = eltype(grid)
    ℓ = mixing_length.Cᵗ * ∫qz / ∫q
    @inbounds ℓᵗ[i, j, 1] = ifelse(∫q > 0, FT(ℓ), FT(Inf))
end
