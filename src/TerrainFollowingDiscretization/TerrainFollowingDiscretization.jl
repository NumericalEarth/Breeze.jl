"""
    TerrainFollowingDiscretization

Module implementing terrain-following vertical coordinates via the
[`TerrainFollowingVerticalDiscretization`](@ref) (TFVD) grid type.

TFVD stores a uniform reference vertical coordinate ``r`` and a
formulation (e.g. [`LinearDecay`](@ref) or [`TwoLevelDecay`](@ref)) that defines
the physical altitude

```math
z(x, y, r) = r + h(x, y) \\, b(r) ,
```

where ``h(x, y)`` is the terrain and ``b(r)`` is a decay basis
satisfying ``b(0) = 1`` and ``b(z_\\text{top}) = 0``.

Public API:
  - [`TerrainFollowingVerticalDiscretization`](@ref) — pass as the `z`
    argument to `RectilinearGrid` / `LatitudeLongitudeGrid`.
  - [`LinearDecay`](@ref) (Gal-Chen & Somerville 1975).
  - [`TwoLevelDecay`](@ref) (Schär et al. 2002).
  - [`materialize_terrain!`](@ref) — fill the formulation's terrain fields
    from a `h(x, y)` function once the grid is built.
  - [`build_terrain_metrics`](@ref) — attach a pressure-gradient stencil.

See `docs/src/terrain_following_coordinates.md` for the math, discrete
operators, well-balancing reference state, and worked examples.
"""
module TerrainFollowingDiscretization

export TerrainMetrics,
       SlopeOutsideInterpolation, SlopeInsideInterpolation,
       TerrainFollowingVerticalDiscretization, LinearDecay, TwoLevelDecay,
       materialize_terrain!, build_terrain_metrics, ∂z∂x, ∂z∂y, TerrainFollowingGrid

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, Center, Face, Field, set!
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: znode, rnode, ξnode, ηnode
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxyᶠᶠᵃ

include("terrain_metrics.jl")
include("terrain_formulations.jl")
include("terrain_following_vertical_discretization.jl")
include("terrain_amg_operators.jl")
include("materialize_terrain.jl")

end # module
