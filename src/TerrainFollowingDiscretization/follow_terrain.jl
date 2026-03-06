#####
##### follow_terrain! — set up terrain-following coordinates on a MutableVerticalDiscretization grid
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: surface_kernel_parameters

"""
    follow_terrain!(grid, topography; smoothing=BasicTerrainFollowing())

Transform `grid` into a terrain-following grid by setting the vertical
coordinate scaling factors ``\\sigma`` and surface displacement ``\\eta``
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
                          pressure_gradient_stencil = SlopeOutsideInterpolation())
    return _follow_terrain!(grid, topography, smoothing, pressure_gradient_stencil)
end

# Dispatch on smoothing type
function _follow_terrain!(grid, topography, ::BasicTerrainFollowing, pressure_gradient_stencil)
    arch = architecture(grid)

    # Get the model top from the reference coordinate (scalar access is fine here —
    # this is a one-time setup operation, not a hot path).
    Nz = size(grid, 3)
    z_top = @allowscalar rnode(Nz + 1, grid, Face())

    # Create a 2D CenterField to store the topography
    h_field = CenterField(grid, indices=(:, :, 1))

    # Set topography values and fill halos
    _set_topography!(h_field, grid, topography)
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

    return TerrainMetrics(h_field, ∂x_h, ∂y_h, z_top, pressure_gradient_stencil)
end

# Set topography from a function: always evaluate on CPU, then copy to device.
# This supports arbitrary user-defined functions (including those that reference
# non-const globals) without requiring GPU-compatible code.
# Note: Oceananigans' set!(field, func) requires func(x, y, z) and evaluates on-device,
# which would fail for non-GPU-compatible user functions. The manual copyto! pattern here
# is intentionally more general.
function _set_topography!(h_field, grid, topography::Function)
    Nx, Ny = size(grid, 1), size(grid, 2)
    cpu_h = [topography(xnode(i, grid, Center()), ynode(j, grid, Center()))
              for i in 1:Nx, j in 1:Ny]
    copyto!(interior(h_field, :, :, 1), cpu_h)
    return nothing
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
