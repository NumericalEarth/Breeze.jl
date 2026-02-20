#####
##### Terrain metric terms
#####
##### For basic terrain-following coordinates, the terrain slope at height ζ is
#####
#####   (∂z/∂x)_ζ = (∂h/∂x) * (1 - ζ / z_top)
#####
##### These metric terms are needed for:
##### 1. Computing the contravariant vertical velocity Ω̃
##### 2. Correcting horizontal pressure gradients
#####

#####
##### Pressure gradient stencil types
#####
##### The terrain-corrected horizontal pressure gradient requires interpolating
##### ∂p/∂ζ to the velocity point and multiplying by the terrain slope. These
##### two types control the order of interpolation and multiplication:
#####
##### SlopeOutsideInterpolation (default):
#####   slope(i,j,k) * ℑz(ℑx(∂z(p')))
#####   — slope is evaluated at the target (Face, Center, Center) point and
#####     multiplied after averaging the vertical pressure derivative.
#####
##### SlopeInsideInterpolation:
#####   ℑz(ℑx(slope(i,j,k) * ∂z(p')))
#####   — slope is evaluated at each (Center, Center, Face) stencil point and
#####     multiplied before averaging, closer to the CM1 approach.
#####
##### The two stencils differ at O(Δx·Δz) on a terrain-deformed grid because
##### interpolation and pointwise multiplication do not commute when the
##### multiplier varies spatially.
#####

"""
    SlopeOutsideInterpolation

Terrain pressure gradient stencil where the slope is multiplied outside
the interpolation of ``∂p'/∂ζ``:

```math
\\text{correction} = s(i,j,k) \\, \\overline{\\overline{\\partial_\\zeta p'}^x}^z
```

This is the default stencil.
"""
struct SlopeOutsideInterpolation end

"""
    SlopeInsideInterpolation

Terrain pressure gradient stencil where the slope is multiplied inside
the interpolation of ``∂p'/∂ζ``:

```math
\\text{correction} = \\overline{\\overline{s \\, \\partial_\\zeta p'}^x}^z
```

The slope is evaluated at each `(Center, Center, Face)` stencil point before
averaging to `(Face, Center, Center)`. This stencil is closer to the CM1
approach where the metric term sits inside the 4-point average.
"""
struct SlopeInsideInterpolation end

"""
    TerrainMetrics{H, SX, SY, FT, PG}

Pre-computed terrain derivative fields and model top height.

Fields
======

- `topography`: 2D `CenterField` storing ``h(x, y)``
- `∂x_h`: 2D field storing ``\\partial h / \\partial x`` at ``(Face, Center)``
- `∂y_h`: 2D field storing ``\\partial h / \\partial y`` at ``(Center, Face)``
- `z_top`: Height of the model top (top of the reference coordinate)
- `pressure_gradient_stencil`: Stencil type for the terrain-corrected horizontal
  pressure gradient ([`SlopeOutsideInterpolation`](@ref) or [`SlopeInsideInterpolation`](@ref))
"""
struct TerrainMetrics{H, SX, SY, FT, PG}
    topography :: H
    ∂x_h :: SX
    ∂y_h :: SY
    z_top :: FT
    pressure_gradient_stencil :: PG
end

Adapt.adapt_structure(to, m::TerrainMetrics) =
    TerrainMetrics(adapt(to, m.topography),
                   adapt(to, m.∂x_h),
                   adapt(to, m.∂y_h),
                   m.z_top,
                   m.pressure_gradient_stencil)

"""
    terrain_slope_x(i, j, k, grid, metrics, ℓz)

Compute ``(\\partial z / \\partial x)_\\zeta`` at horizontal location ``(Face, Center)``
and vertical location `ℓz` (either `Center()` or `Face()`).

For basic terrain-following coordinates:
```math
\\left(\\frac{\\partial z}{\\partial x}\\right)_\\zeta
= \\frac{\\partial h}{\\partial x} \\left(1 - \\frac{\\zeta}{z_{top}}\\right)
```
"""
@inline function terrain_slope_x(i, j, k, grid, metrics, ℓz)
    ζ = rnode(k, grid, ℓz)
    z_top = metrics.z_top
    @inbounds ∂x_h = metrics.∂x_h[i, j, 1]
    return ∂x_h * (1 - ζ / z_top)
end

"""
    terrain_slope_y(i, j, k, grid, metrics, ℓz)

Compute ``(\\partial z / \\partial y)_\\zeta`` at horizontal location ``(Center, Face)``
and vertical location `ℓz` (either `Center()` or `Face()`).
"""
@inline function terrain_slope_y(i, j, k, grid, metrics, ℓz)
    ζ = rnode(k, grid, ℓz)
    z_top = metrics.z_top
    @inbounds ∂y_h = metrics.∂y_h[i, j, 1]
    return ∂y_h * (1 - ζ / z_top)
end
