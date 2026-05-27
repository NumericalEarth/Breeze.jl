"""
    TerrainFollowingDiscretization

Module implementing terrain-following vertical coordinates using
Oceananigans' `MutableVerticalDiscretization`.

The basic terrain-following (BTF) coordinate maps the physical vertical
coordinate ``z`` to a computational coordinate ``\\zeta`` via

```math
z(x, y, \\zeta) = \\zeta \\, \\sigma(x, y) + h(x, y)
```

where ``h(x, y)`` is the terrain height, ``\\sigma = (z_{top} - h) / z_{top}``
is the column scaling factor, and ``\\zeta \\in [0, z_{top}]`` is the reference
(computational) coordinate.

This module is designed to be self-contained so that it can eventually be
ported to Oceananigans.
"""
module TerrainFollowingDiscretization

export follow_terrain!, TerrainMetrics, BasicTerrainFollowing,
       GridFittedTerrain, FaceSampledTerrain,
       SlopeOutsideInterpolation, SlopeInsideInterpolation,
       TerrainFollowingVerticalDiscretization, LinearDecay, SLEVE,
       materialize_terrain!, build_terrain_metrics, ∂z∂x, ∂z∂y, TFVDRG

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using GPUArraysCore: @allowscalar
using KernelAbstractions: @kernel, @index
using Oceananigans
using Oceananigans: Center, Face, CenterField, XFaceField, YFaceField, interior, set!
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.Grids: xnode, ynode, rnode, AbstractGrid
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: δxᶠᶜᶜ, δyᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxyᶠᶠᵃ

include("terrain_smoothing.jl")
include("terrain_metrics.jl")
include("follow_terrain.jl")
include("terrain_formulations.jl")
include("terrain_following_vertical_discretization.jl")
include("materialize_terrain.jl")

end # module
