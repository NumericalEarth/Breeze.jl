#####
##### follow_terrain! — set up terrain-following coordinates on a MutableVerticalDiscretization grid
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: surface_kernel_parameters

abstract type AbstractTerrainInterpretation end

"""
$(TYPEDEF)

Default terrain interpretation for [`follow_terrain!`](@ref). The topography
function is evaluated at the physical horizontal cell centers reported by the
Oceananigans grid.
"""
struct GridFittedTerrain <: AbstractTerrainInterpretation end

"""
$(TYPEDEF)

Terrain interpretation that samples the topography function at the
upper x- and y-face of each cell, equivalent to
`xnode(i, Center()) + Δx/2` and `ynode(j, Center()) + Δy/2`,
rather than at the cell centre.

Use this when reproducing setups whose analytic topography is defined
on faces of the grid — for example, cross-model validation against
codes that evaluate the topography at integer-indexed nodes
(`dx · i`, `dy · j`) rather than at cell centres
(`dx · (i − 1/2)`, `dy · (j − 1/2)`), so that the apex of an analytic
mountain lands exactly on one cell rather than between cells.
"""
struct FaceSampledTerrain <: AbstractTerrainInterpretation end

"""
$(TYPEDSIGNATURES)

Transform `grid` into a terrain-following grid by setting the vertical
coordinate scaling factors ``σ`` and surface displacement ``η``
on the grid's `MutableVerticalDiscretization`.

Returns a [`TerrainMetrics`](@ref) object containing pre-computed terrain
slopes needed for metric term corrections in the momentum and scalar equations.

Arguments
=========

- `grid`: A grid with `MutableVerticalDiscretization` (created by passing
  `MutableVerticalDiscretization(z_faces)` as the `z` argument to `RectilinearGrid`).

- `topography`: A function `h(x, y)` returning the terrain height at each
  horizontal location.

Keyword Arguments
=================

- `smoothing`: Decay function controlling how terrain influence decreases with height.
  Default: [`BasicTerrainFollowing`](@ref) (linear decay to zero at model top).

- `terrain_interpretation`: Location at which function-valued topography is
  evaluated. Default: [`GridFittedTerrain`](@ref) (cell centres). Pass
  [`FaceSampledTerrain`](@ref) to evaluate the topography at the upper
  x- and y-face of each cell instead — useful when reproducing setups
  whose analytic topography is defined on grid-index nodes rather than
  cell centres.

Example
=======

```jldoctest
using Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization
using Breeze: follow_terrain!, TerrainMetrics

z_faces = MutableVerticalDiscretization(collect(range(0, 10000, length=41)))
grid = RectilinearGrid(size=(64, 40), x=(-50000, 50000), z=z_faces,
                       topology=(Periodic, Flat, Bounded))

h(x, y) = 500 * exp(-x^2 / 5000^2)
metrics = follow_terrain!(grid, h)
metrics.z_top

# output

10000.0
```
"""
function follow_terrain!(grid::MutableGridOfSomeKind, topography;
                         smoothing = BasicTerrainFollowing(),
                         pressure_gradient_stencil = SlopeOutsideInterpolation(),
                         terrain_interpretation = GridFittedTerrain())
    return follow_terrain!(grid, topography, smoothing, pressure_gradient_stencil,
                           terrain_interpretation)
end

# Dispatch on smoothing type
function follow_terrain!(grid, topography, ::BasicTerrainFollowing, pressure_gradient_stencil,
                         terrain_interpretation)
    terrain_interpretation isa AbstractTerrainInterpretation ||
        throw(ArgumentError("`terrain_interpretation` must be a terrain interpretation such as `GridFittedTerrain()` or `FaceSampledTerrain()`"))

    arch = architecture(grid)

    # Get the model top from the reference coordinate (scalar access is fine here —
    # this is a one-time setup operation, not a hot path).
    Nz = size(grid, 3)
    z_top = @allowscalar rnode(Nz + 1, grid, Face())

    # Create a 2D CenterField to store the topography
    h_field = CenterField(grid, indices=(:, :, 1))

    # Set topography values and fill halos
    is_flat = set_topography!(h_field, grid, topography, terrain_interpretation)
    fill_halo_regions!(h_field)

    # Compute sigma and eta on the grid
    kp = surface_kernel_parameters(grid)
    launch!(arch, grid, kp, _set_btf_sigma!, grid, h_field, z_top)

    # Also set σᶜᶜ⁻ = σᶜᶜⁿ and ∂t_σ = 0 (static terrain)
    parent(grid.z.σᶜᶜ⁻) .= parent(grid.z.σᶜᶜⁿ)
    fill!(grid.z.∂t_σ, 0)

    # Compute terrain slopes
    ∂x_h = XFaceField(grid, indices=(:, :, 1))
    ∂y_h = YFaceField(grid, indices=(:, :, 1))

    launch!(arch, grid, kp, _compute_terrain_slopes!, ∂x_h, ∂y_h, grid, h_field)

    return TerrainMetrics(h_field, ∂x_h, ∂y_h, z_top, pressure_gradient_stencil, Val(is_flat))
end

# Set topography from a function: always evaluate on CPU, then copy to device.
# This supports arbitrary user-defined functions (including those that reference
# non-const globals) without requiring GPU-compatible code.
# Note: Oceananigans' set!(field, func) requires func(x, y, z) and evaluates on-device,
# which would fail for non-GPU-compatible user functions. The manual copyto! pattern here
# is intentionally more general.
function set_topography!(h_field, grid, topography::Function, terrain_interpretation)
    Nx, Ny = size(grid, 1), size(grid, 2)
    cpu_h = [topography(topography_xnode(i, grid, terrain_interpretation),
                        topography_ynode(j, grid, terrain_interpretation))
              for i in 1:Nx, j in 1:Ny]
    copyto!(interior(h_field, :, :, 1), cpu_h)
    return all(iszero, cpu_h)
end

@inline topography_xnode(i, grid, ::GridFittedTerrain) = xnode(i, grid, Center())
@inline topography_ynode(j, grid, ::GridFittedTerrain) = ynode(j, grid, Center())

@inline function topography_xnode(i, grid, ::FaceSampledTerrain)
    Nx = size(grid, 1)
    x = xnode(i, grid, Center())
    Nx == 1 && return x

    if i < Nx
        return x + (xnode(i + 1, grid, Center()) - x) / 2
    else
        return x + (x - xnode(i - 1, grid, Center())) / 2
    end
end

@inline function topography_ynode(j, grid, ::FaceSampledTerrain)
    Ny = size(grid, 2)
    y = ynode(j, grid, Center())
    Ny == 1 && return y

    if j < Ny
        return y + (ynode(j + 1, grid, Center()) - y) / 2
    else
        return y + (y - ynode(j - 1, grid, Center())) / 2
    end
end

@kernel function _set_btf_sigma!(grid, h_field, z_top)
    i, j = @index(Global, NTuple)

    # Terrain height interpolated to each stagger location
    @inbounds hᶜᶜ = h_field[i, j, 1]
    hᶠᶜ = ℑxᶠᵃᵃ(i, j, 1, grid, h_field)
    hᶜᶠ = ℑyᵃᶠᵃ(i, j, 1, grid, h_field)
    hᶠᶠ = ℑxyᶠᶠᵃ(i, j, 1, grid, h_field)

    # Basic terrain-following: σ = (z_top - h) / z_top
    @inbounds begin
        grid.z.σᶜᶜⁿ[i, j, 1] = (z_top - hᶜᶜ) / z_top
        grid.z.σᶠᶜⁿ[i, j, 1] = (z_top - hᶠᶜ) / z_top
        grid.z.σᶜᶠⁿ[i, j, 1] = (z_top - hᶜᶠ) / z_top
        grid.z.σᶠᶠⁿ[i, j, 1] = (z_top - hᶠᶠ) / z_top
        grid.z.ηⁿ[i, j, 1] = hᶜᶜ
    end
end

@kernel function _compute_terrain_slopes!(∂x_h, ∂y_h, grid, h_field)
    i, j = @index(Global, NTuple)
    # Use basic finite differences (δx · Δx⁻¹) instead of generalized derivatives (∂x).
    # On MutableVerticalDiscretization grids, the generalized ∂x includes a chain-rule
    # vertical correction (∂z/∂x · ∂ϕ/∂z) that accesses neighboring k-levels.
    # Terrain slopes are purely horizontal, so we need the simple difference only.
    @inbounds ∂x_h[i, j, 1] = δxᶠᶜᶜ(i, j, 1, grid, h_field) * Δx⁻¹ᶠᶜᶜ(i, j, 1, grid)
    @inbounds ∂y_h[i, j, 1] = δyᶜᶠᶜ(i, j, 1, grid, h_field) * Δy⁻¹ᶜᶠᶜ(i, j, 1, grid)
end
