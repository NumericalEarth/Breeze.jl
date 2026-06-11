using Oceananigans.Utils: prettysummary

#####
##### Terrain metric terms
#####
##### For basic terrain-following coordinates, the terrain slope at height r is
#####
#####   (∂z/∂x)_r = (∂h/∂x) * (1 - r / z_top)
#####
##### These metric terms are needed for:
##### 1. Computing the contravariant vertical velocity w̃
##### 2. Correcting horizontal pressure gradients
#####

#####
##### Pressure gradient stencil types
#####
##### The terrain-corrected horizontal pressure gradient requires interpolating
##### ∂p/∂r to the velocity point and multiplying by the terrain slope. These
##### two types control the order of interpolation and multiplication:
#####
##### SlopeOutsideInterpolation (default):
#####   slope(i, j, k) * ℑz(ℑx(∂z(p')))
#####   — slope is evaluated at the target (Face, Center, Center) point and
#####     multiplied after averaging the vertical pressure derivative.
#####
##### SlopeInsideInterpolation:
#####   ℑz(ℑx(slope(i, j, k) * ∂z(p')))
#####   — slope is evaluated at each (Center, Center, Face) stencil point and
#####     multiplied before averaging.
#####
##### The two stencils differ at O(Δx·Δz) on a terrain-deformed grid because
##### interpolation and pointwise multiplication do not commute when the
##### multiplier varies spatially.
#####

"""
$(TYPEDEF)

Terrain pressure gradient stencil where the slope is multiplied outside
the interpolation of ``∂p'/∂r``:

```math
\\text{correction} = s(i,j,k) \\, \\overline{\\overline{\\partial_r p'}^x}^z
```

This is the default stencil.
"""
struct SlopeOutsideInterpolation end

"""
$(TYPEDEF)

Terrain pressure gradient stencil where the slope is multiplied inside
the interpolation of ``∂p'/∂r``:

```math
\\text{correction} = \\overline{\\overline{s \\, \\partial_r p'}^x}^z
```

The slope is evaluated at each `(Center, Center, Face)` stencil point before
averaging to `(Face, Center, Center)`.
"""
struct SlopeInsideInterpolation end

"""
$(TYPEDEF)

Pre-computed terrain derivative fields and model top height.

Fields
======

- `topography`: 2D field storing ``h(x, y)`` at `(Center, Center)`
- `∂x_h`: 2D field storing ``\\partial h / \\partial x`` at `(Face, Center)`
- `∂y_h`: 2D field storing ``\\partial h / \\partial y`` at `(Center, Face)`
- `z_top`: Height of the model top (top of the reference coordinate)
- `pressure_gradient_stencil`: Stencil type for the terrain-corrected horizontal
  pressure gradient ([`SlopeOutsideInterpolation`](@ref) or
  [`SlopeInsideInterpolation`](@ref))
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

Base.summary(tf::TerrainMetrics) = "TerrainMetrics for $(summary(tf.topography)) using $(summary(tf.pressure_gradient_stencil))"

function Base.show(io::IO, tm::TerrainMetrics)
    print(io, "TerrainMetrics", '\n')
    print(io, "├── topography: ", summary(tm.topography), '\n')
    print(io, "├── ∂x_h: ", prettysummary(tm.∂x_h), '\n')
    print(io, "├── ∂y_h: ", prettysummary(tm.∂y_h), '\n')
    print(io, "├── z_top: ", prettysummary(tm.z_top), '\n')
    print(io, "└── pressure_gradient_stencil: ", prettysummary(tm.pressure_gradient_stencil))
end
