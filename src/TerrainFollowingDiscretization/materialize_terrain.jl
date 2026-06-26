#####
##### Allocation + materialization of the terrain-following coordinate.
#####
##### `allocate_formulation` is called from `generate_coordinate` (before the
##### grid exists) and allocates zero-filled raw data arrays for the terrain
##### components, exactly as `MutableVerticalDiscretization` allocates σ via
##### `new_data`. `materialize_terrain!` is called once the grid exists and
##### fills those arrays in place from the topography — the skeleton →
##### materialize pattern. All geometry stays in `grid.z`.
#####

using Oceananigans.Grids: new_data
using Oceananigans.Operators: δxᶠᶜᶜ, δyᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ

@inline centered_data(FT, arch, topo, sz, halo) = new_data(FT, arch, (Center, Center, Nothing), topo, sz, halo)
@inline xface_data(FT, arch, topo, sz, halo) = new_data(FT, arch, (Face,   Center, Nothing), topo, sz, halo)
@inline yface_data(FT, arch, topo, sz, halo) = new_data(FT, arch, (Center, Face,   Nothing), topo, sz, halo)

function allocate_formulation(f::LinearDecay, FT, arch, sz, halo, topo, z_top)
    # If the user-supplied formulation is already materialised (h ≠ nothing —
    # typically because Oceananigans is reconstructing the grid via
    # `on_architecture`), preserve the data and just move it to `arch` rather
    # than allocating fresh zero-filled arrays.
    if f.h !== nothing
        return Oceananigans.Architectures.on_architecture(arch,
            LinearDecay(convert(FT, z_top), f.h, f.∂x_h, f.∂y_h))
    end
    h    = centered_data(FT, arch, topo, sz, halo); fill!(h, 0)
    ∂x_h = xface_data(FT, arch, topo, sz, halo); fill!(∂x_h, 0)
    ∂y_h = yface_data(FT, arch, topo, sz, halo); fill!(∂y_h, 0)
    return LinearDecay(convert(FT, z_top), h, ∂x_h, ∂y_h)
end

# 1D-in-z basis component (Center or Face in z, singleton x, y), zero-filled.
@inline zcenter_data(FT, arch, topo, sz, halo) = new_data(FT, arch, (Nothing, Nothing, Center), topo, sz, halo)
@inline zface_data(FT, arch, topo, sz, halo)   = new_data(FT, arch, (Nothing, Nothing, Face),   topo, sz, halo)

function allocate_two_level_basis(FT, arch, sz, halo, topo)
    zc() = (a = zcenter_data(FT, arch, topo, sz, halo); fill!(a, 0); a)
    zf() = (a = zface_data(FT, arch, topo, sz, halo);   fill!(a, 0); a)
    return TwoLevelBasis(zc(), zf(), zc(), zf(), zc(), zf(), zc(), zf())
end

function allocate_formulation(f::TwoLevelDecay, FT, arch, sz, halo, topo, z_top)
    # Preserve already-materialised data on rebuild (see LinearDecay variant).
    if f.h₁ !== nothing
        return Oceananigans.Architectures.on_architecture(arch,
            TwoLevelDecay(convert(FT, z_top),
                          convert(FT, f.large_scale_height), convert(FT, f.small_scale_height),
                          f.h₁, f.h₂, f.∂x_h₁, f.∂x_h₂, f.∂y_h₁, f.∂y_h₂, f.basis))
    end
    h₁ = centered_data(FT, arch, topo, sz, halo); h₂ = centered_data(FT, arch, topo, sz, halo)
    ∂x_h₁ = xface_data(FT, arch, topo, sz, halo); ∂x_h₂ = xface_data(FT, arch, topo, sz, halo)
    ∂y_h₁ = yface_data(FT, arch, topo, sz, halo); ∂y_h₂ = yface_data(FT, arch, topo, sz, halo)
    for a in (h₁, h₂, ∂x_h₁, ∂x_h₂, ∂y_h₁, ∂y_h₂); fill!(a, 0); end
    basis = allocate_two_level_basis(FT, arch, sz, halo, topo)
    return TwoLevelDecay(convert(FT, z_top),
                 convert(FT, f.large_scale_height), convert(FT, f.small_scale_height),
                 h₁, h₂, ∂x_h₁, ∂x_h₂, ∂y_h₁, ∂y_h₂, basis)
end

"""
$(TYPEDSIGNATURES)

Fill the terrain components of a `TerrainFollowingVerticalDiscretization` grid
in place from `topography(x, y)`. Must be called after the grid is built (the
horizontal nodes are needed to evaluate the topography). For `TwoLevelDecay`, the
terrain is split into large- and small-scale parts by horizontal smoothing.

The topography is evaluated at the horizontal cell-centre nodes. Its arguments
follow the grid's horizontal coordinates with Flat dimensions dropped, matching
every other `set!` initialiser: `topography(x, y)` on a RectilinearGrid,
`topography(λ, φ)` on a LatitudeLongitudeGrid, and e.g. `topography(x)` when y is
Flat.
"""
function materialize_terrain!(grid, topography)
    materialize_formulation!(grid.z.formulation, grid, topography)
    return grid
end

# Evaluate the topography into a fresh 2D (… , Nothing) field. The field's
# vertical location is `Nothing`, so `set!` evaluates `topography` at the
# horizontal nodes only (with Flat dimensions dropped), without a spurious
# vertical coordinate.
function terrain_height_field(grid, topography)
    h_field = Field((Center(), Center(), nothing), grid)
    set!(h_field, topography)
    fill_halo_regions!(h_field)
    return h_field
end

# Surface slope at face stagger via basic finite difference. We use `δx · Δx⁻¹`
# directly rather than the generalised `∂x` operator: on a TFVD grid `∂x` adds
# a chain-rule vertical correction that accesses neighbouring k-levels — the
# terrain slope is a purely horizontal quantity, so the simple difference is
# what we want.
@kernel function _compute_terrain_slopes!(∂x_h, ∂y_h, grid, h_field)
    i, j = @index(Global, NTuple)
    @inbounds ∂x_h[i, j, 1] = δxᶠᶜᶜ(i, j, 1, grid, h_field) * Δx⁻¹ᶠᶜᶜ(i, j, 1, grid)
    @inbounds ∂y_h[i, j, 1] = δyᶜᶠᶜ(i, j, 1, grid, h_field) * Δy⁻¹ᶜᶠᶜ(i, j, 1, grid)
end

# Compute the surface slopes on the interior, then fill_halo_regions! to wrap
# the (periodic) halos correctly, then copy into the raw slope arrays. The
# grid slope operator ∂z∂x interpolates to i+1 at i=Nx, so the slope halos
# must be valid — an unfilled (zero) halo seeds a boundary instability.
function fill_terrain_slopes!(∂x_raw, ∂y_raw, h_field, grid)
    arch = architecture(grid)
    ∂x = Field((Face(), Center(), nothing), grid)
    ∂y = Field((Center(), Face(), nothing), grid)
    launch!(arch, grid, (size(grid, 1), size(grid, 2)),
            _compute_terrain_slopes!, ∂x, ∂y, grid, h_field)
    fill_halo_regions!(∂x)
    fill_halo_regions!(∂y)
    parent(∂x_raw) .= parent(∂x)
    parent(∂y_raw) .= parent(∂y)
    return nothing
end

function materialize_formulation!(f::LinearDecay, grid, topography)
    h_field = terrain_height_field(grid, topography)
    parent(f.h) .= parent(h_field)
    fill_terrain_slopes!(f.∂x_h, f.∂y_h, h_field, grid)
    return nothing
end

function materialize_formulation!(f::TwoLevelDecay, grid, topography)
    # Full terrain into a temp field, then split: h₁ = smooth(h) (large scale),
    # h₂ = h − h₁ (small scale). Store the parts in the formulation's arrays.
    h_field = terrain_height_field(grid, topography)

    h₁_field = Field((Center(), Center(), nothing), grid)
    smooth_horizontally!(h₁_field, h_field, grid; passes = 8)   # large-scale part
    fill_halo_regions!(h₁_field)

    h₂_field = Field((Center(), Center(), nothing), grid)
    parent(h₂_field) .= parent(h_field) .- parent(h₁_field)     # small-scale residual
    fill_halo_regions!(h₂_field)

    parent(f.h₁) .= parent(h₁_field)
    parent(f.h₂) .= parent(h₂_field)
    fill_terrain_slopes!(f.∂x_h₁, f.∂y_h₁, h₁_field, grid)
    fill_terrain_slopes!(f.∂x_h₂, f.∂y_h₂, h₂_field, grid)
    fill_two_level_basis!(f.basis, grid, f.z_top, f.large_scale_height, f.small_scale_height)
    return nothing
end

# Evaluate bₙ(r), bₙ′(r) at the reference centre/face nodes (grid.z.cᵃᵃᶜ/cᵃᵃᶠ) and
# store them. Broadcasting over the raw coordinate arrays covers the full halo'd
# k-range and works on CPU and GPU (sinh/cosh broadcast on either architecture).
set_basis_component!(dest, rcoord, fn, z_top, s) =
    (vec(parent(dest)) .= fn.(parent(rcoord), z_top, s); nothing)

function fill_two_level_basis!(basis, grid, z_top, s₁, s₂)
    rᶜ = grid.z.cᵃᵃᶜ   # reference centre coordinate (1D in z, with halos)
    rᶠ = grid.z.cᵃᵃᶠ   # reference face coordinate
    set_basis_component!(basis.b₁ᶜ, rᶜ, b_two_level, z_top, s₁)
    set_basis_component!(basis.b₁ᶠ, rᶠ, b_two_level, z_top, s₁)
    set_basis_component!(basis.b₂ᶜ, rᶜ, b_two_level, z_top, s₂)
    set_basis_component!(basis.b₂ᶠ, rᶠ, b_two_level, z_top, s₂)
    set_basis_component!(basis.∂b₁ᶜ, rᶜ, b′_two_level, z_top, s₁)
    set_basis_component!(basis.∂b₁ᶠ, rᶠ, b′_two_level, z_top, s₁)
    set_basis_component!(basis.∂b₂ᶜ, rᶜ, b′_two_level, z_top, s₂)
    set_basis_component!(basis.∂b₂ᶠ, rᶠ, b′_two_level, z_top, s₂)
    return nothing
end

# Iterated 1-2-1 horizontal smoothing to isolate the large-scale terrain.
# `passes` controls the cutoff: more passes removes finer scales into h₂.
function smooth_horizontally!(out, in_field, grid; passes = 8)
    arch = architecture(grid)
    parent(out) .= parent(in_field)
    tmp = Field((Center(), Center(), nothing), grid)
    for _ in 1:passes
        fill_halo_regions!(out)
        launch!(arch, grid, (size(grid, 1), size(grid, 2)), _smooth_x_121!, tmp, out, grid)

        if size(grid, 2) == 1
            parent(out) .= parent(tmp)
        else
            fill_halo_regions!(tmp)
            launch!(arch, grid, (size(grid, 1), size(grid, 2)), _smooth_y_121!, out, tmp, grid)
        end
    end
    return out
end

@kernel function _smooth_x_121!(out, in_field, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        out[i, j, 1] = (in_field[i - 1, j, 1] + 2 * in_field[i, j, 1] + in_field[i + 1, j, 1]) / 4
    end
end

@kernel function _smooth_y_121!(out, in_field, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        out[i, j, 1] = (in_field[i, j - 1, 1] + 2 * in_field[i, j, 1] + in_field[i, j + 1, 1]) / 4
    end
end

"""
$(TYPEDSIGNATURES)

Build a `TerrainMetrics` for a materialized `TerrainFollowingVerticalDiscretization`
grid. On such grids the terrain *slope* used by the dynamics comes from the grid
`∂z∂x` operator (formulation decay), so this object only carries the
`pressure_gradient_stencil`, `z_top`, and a representative terrain field.
"""
build_terrain_metrics(grid, stencil) = terrain_metrics_for_formulation(grid.z.formulation, stencil)

terrain_metrics_for_formulation(f::LinearDecay, stencil) =
    TerrainMetrics(f.h, f.∂x_h, f.∂y_h, f.z_top, stencil)

terrain_metrics_for_formulation(f::TwoLevelDecay, stencil) =
    TerrainMetrics(f.h₁, f.∂x_h₁, f.∂y_h₁, f.z_top, stencil)
