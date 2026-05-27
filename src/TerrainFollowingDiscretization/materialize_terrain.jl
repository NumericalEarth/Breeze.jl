#####
##### Allocation + materialization of the terrain-following coordinate.
#####
##### `allocate_formulation` is called from `generate_coordinate` (before the
##### grid exists) and allocates zero-filled raw data arrays for the terrain
##### components, exactly as `MutableVerticalDiscretization` allocates σ via
##### `new_data`. `materialize_terrain!` is called once the grid exists and
##### fills those arrays in place from the topography — the skeleton →
##### materialize pattern. It replaces the old `follow_terrain!` (which also
##### returned a Dynamics-side `TerrainMetrics`); here all geometry stays in
##### `grid.z`.
#####

using Oceananigans.Grids: new_data

@inline _cc(FT, arch, topo, sz, halo) = new_data(FT, arch, (Center, Center, Nothing), topo, sz, halo)
@inline _fc(FT, arch, topo, sz, halo) = new_data(FT, arch, (Face,   Center, Nothing), topo, sz, halo)
@inline _cf(FT, arch, topo, sz, halo) = new_data(FT, arch, (Center, Face,   Nothing), topo, sz, halo)

function allocate_formulation(f::LinearDecay, FT, arch, sz, halo, topo, z_top)
    h    = _cc(FT, arch, topo, sz, halo); fill!(h, 0)
    ∂x_h = _fc(FT, arch, topo, sz, halo); fill!(∂x_h, 0)
    ∂y_h = _cf(FT, arch, topo, sz, halo); fill!(∂y_h, 0)
    return LinearDecay(convert(FT, z_top), h, ∂x_h, ∂y_h)
end

function allocate_formulation(f::SLEVE, FT, arch, sz, halo, topo, z_top)
    h₁ = _cc(FT, arch, topo, sz, halo); h₂ = _cc(FT, arch, topo, sz, halo)
    ∂x_h₁ = _fc(FT, arch, topo, sz, halo); ∂x_h₂ = _fc(FT, arch, topo, sz, halo)
    ∂y_h₁ = _cf(FT, arch, topo, sz, halo); ∂y_h₂ = _cf(FT, arch, topo, sz, halo)
    for a in (h₁, h₂, ∂x_h₁, ∂x_h₂, ∂y_h₁, ∂y_h₂); fill!(a, 0); end
    return SLEVE(convert(FT, z_top),
                 convert(FT, f.large_scale_height), convert(FT, f.small_scale_height),
                 h₁, h₂, ∂x_h₁, ∂x_h₂, ∂y_h₁, ∂y_h₂)
end

"""
$(TYPEDSIGNATURES)

Fill the terrain components of a `TerrainFollowingVerticalDiscretization` grid
in place from `topography(x, y)`. Must be called after the grid is built (the
horizontal nodes are needed to evaluate the topography). For `SLEVE`, the
terrain is split into large- and small-scale parts by horizontal smoothing.
"""
function materialize_terrain!(grid, topography; terrain_interpretation = GridFittedTerrain())
    materialize_formulation!(grid.z.formulation, grid, topography, terrain_interpretation)
    return grid
end

# Evaluate topography into a temporary CenterField, then copy into the raw array.
function _fill_terrain_height!(h_raw, grid, topography, terrain_interpretation)
    arch = architecture(grid)
    h_field = CenterField(grid, indices=(:, :, 1))
    set_topography!(h_field, grid, topography, terrain_interpretation)
    fill_halo_regions!(h_field)
    parent(h_raw) .= parent(h_field)
    return h_field
end

# Compute the surface slopes in the interior, then fill_halo_regions! to wrap
# the (periodic) halos correctly, then copy into the raw slope arrays. The grid
# slope operator ∂z∂x interpolates to i+1 at i=Nx, so the slope halos MUST be
# valid — an unfilled (zero) halo seeds a boundary instability. Computing the
# slopes over the halo region directly (e.g. surface_kernel_parameters) instead
# reads unfilled h at the outermost halo and produces garbage, so go through a
# Field + fill_halo_regions!.
function _fill_terrain_slopes!(∂x_raw, ∂y_raw, h_field, grid)
    arch = architecture(grid)
    ∂x = XFaceField(grid, indices = (:, :, 1))
    ∂y = YFaceField(grid, indices = (:, :, 1))
    launch!(arch, grid, (size(grid, 1), size(grid, 2)),
            _compute_terrain_slopes!, ∂x, ∂y, grid, h_field)
    fill_halo_regions!(∂x)
    fill_halo_regions!(∂y)
    parent(∂x_raw) .= parent(∂x)
    parent(∂y_raw) .= parent(∂y)
    return nothing
end

function materialize_formulation!(f::LinearDecay, grid, topography, terrain_interpretation)
    h_field = _fill_terrain_height!(f.h, grid, topography, terrain_interpretation)
    _fill_terrain_slopes!(f.∂x_h, f.∂y_h, h_field, grid)
    return nothing
end

function materialize_formulation!(f::SLEVE, grid, topography, terrain_interpretation)
    arch = architecture(grid)
    # Full terrain into a temp field, then split: h₁ = smooth(h) (large scale),
    # h₂ = h − h₁ (small scale). Store the parts in the formulation's arrays.
    h_field  = CenterField(grid, indices=(:, :, 1))
    set_topography!(h_field, grid, topography, terrain_interpretation)
    fill_halo_regions!(h_field)

    h₁_field = CenterField(grid, indices=(:, :, 1))
    smooth_horizontally!(h₁_field, h_field, grid; passes = 8)   # large-scale part
    fill_halo_regions!(h₁_field)

    h₂_field = CenterField(grid, indices=(:, :, 1))
    parent(h₂_field) .= parent(h_field) .- parent(h₁_field)     # small-scale residual
    fill_halo_regions!(h₂_field)

    parent(f.h₁) .= parent(h₁_field)
    parent(f.h₂) .= parent(h₂_field)
    _fill_terrain_slopes!(f.∂x_h₁, f.∂y_h₁, h₁_field, grid)
    _fill_terrain_slopes!(f.∂x_h₂, f.∂y_h₂, h₂_field, grid)
    return nothing
end

# Iterated 1-2-1 horizontal smoothing (separable, periodic in x) to isolate the
# large-scale terrain. `passes` controls the cutoff: more passes removes finer
# scales into h₂.
function smooth_horizontally!(out, in_field, grid; passes = 8)
    arch = architecture(grid)
    parent(out) .= parent(in_field)
    tmp = CenterField(grid, indices=(:, :, 1))
    for _ in 1:passes
        fill_halo_regions!(out)
        launch!(arch, grid, (size(grid, 1), size(grid, 2)), _smooth_121!, tmp, out, grid)
        parent(out) .= parent(tmp)
    end
    return out
end

@kernel function _smooth_121!(out, in_field, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        sx = (in_field[i-1, j, 1] + 2 * in_field[i, j, 1] + in_field[i+1, j, 1]) / 4
        out[i, j, 1] = sx
    end
end

"""
$(TYPEDSIGNATURES)

Build a `TerrainMetrics` for a materialized `TerrainFollowingVerticalDiscretization`
grid. On such grids the terrain *slope* used by the dynamics comes from the grid
`∂z∂x` operator (formulation decay), so this object only carries the
`pressure_gradient_stencil`, `z_top`, and a representative terrain field for
diagnostics. Lets the existing `TerrainCompressibleDynamics` construct unchanged.
"""
build_terrain_metrics(grid, stencil) = _build_terrain_metrics(grid.z.formulation, stencil)

_build_terrain_metrics(f::LinearDecay, stencil) =
    TerrainMetrics(f.h, f.∂x_h, f.∂y_h, f.z_top, stencil, Val(false))

_build_terrain_metrics(f::SLEVE, stencil) =
    TerrainMetrics(f.h₁, f.∂x_h₁, f.∂y_h₁, f.z_top, stencil, Val(false))
