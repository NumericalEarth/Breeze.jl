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

- `topography`: Either a function `h(x, y)` or a 2D array giving the terrain
  height at each horizontal location.

Keyword Arguments
=================

- `smoothing`: Decay function controlling how terrain influence decreases with height.
  Default: [`BasicTerrainFollowing`](@ref) (linear decay to zero at model top).

Example
=======

```julia
using Oceananigans
using Oceananigans.Grids: MutableVerticalDiscretization

z_faces = MutableVerticalDiscretization(collect(range(0, 10000, length=41)))
grid = RectilinearGrid(size=(64, 40), x=(-50000, 50000), z=z_faces,
                       topology=(Periodic, Flat, Bounded))

h(x, y) = 500 * exp(-x^2 / 5000^2)
metrics = follow_terrain!(grid, h)
```
"""
function follow_terrain!(grid::MutableGridOfSomeKind, topography;
                          smoothing = BasicTerrainFollowing(),
                          pressure_gradient_stencil = SlopeOutsideInterpolation())
    return _follow_terrain!(grid, topography, smoothing, pressure_gradient_stencil)
end

# Dispatch on smoothing type
function _follow_terrain!(grid, topography, ::BasicTerrainFollowing, pressure_gradient_stencil)
    arch = grid.architecture

    # Get the model top from the reference coordinate
    Nz = size(grid, 3)
    z_top = rnode(Nz + 1, grid, Face())

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

# Set topography from a function
function _set_topography!(h_field, grid, topography::Function)
    arch = grid.architecture
    kp = surface_kernel_parameters(grid)
    launch!(arch, grid, kp, _set_topography_from_function!, h_field, grid, topography)
    return nothing
end

@kernel function _set_topography_from_function!(h_field, grid, topography)
    i, j = @index(Global, NTuple)
    x = xnode(i, grid, Center())
    y = ynode(j, grid, Center())
    @inbounds h_field[i, j, 1] = topography(x, y)
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
    @inbounds ∂x_h[i, j, 1] = ∂xᶠᶜᶜ(i, j, 1, grid, h_field)
    @inbounds ∂y_h[i, j, 1] = ∂yᶜᶠᶜ(i, j, 1, grid, h_field)
end
