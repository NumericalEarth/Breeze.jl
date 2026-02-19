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

"""
    TerrainMetrics{H, SX, SY, FT}

Pre-computed terrain derivative fields and model top height.

Fields
======

- `topography`: 2D `CenterField` storing ``h(x, y)``
- `∂x_h`: 2D field storing ``\\partial h / \\partial x`` at ``(Face, Center)``
- `∂y_h`: 2D field storing ``\\partial h / \\partial y`` at ``(Center, Face)``
- `z_top`: Height of the model top (top of the reference coordinate)
"""
struct TerrainMetrics{H, SX, SY, FT}
    topography :: H
    ∂x_h :: SX
    ∂y_h :: SY
    z_top :: FT
end

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
