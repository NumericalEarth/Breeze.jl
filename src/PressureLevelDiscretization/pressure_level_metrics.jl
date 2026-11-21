#####
##### Metrics of the hybrid sigma-pressure grid, derived on the fly from A, B, pˢ and ρ
#####
##### Layer k lies between face k (below) and face k + 1 (above). Its pressure thickness is
##### Δp = (A[k] - A[k+1]) + (B[k] - B[k+1]) pˢ and its physical depth Δz = Δp / (g ρ). The depth
##### at a z-face is the mean of the depths of the two layers it separates, as the spacing of the
##### face-centered reference coordinate is the mean of the cell spacings. Horizontal faces take the
##### horizontal average of the cell depths, as the terrain-following grid takes its slopes.
#####
##### The dynamics' own conservation kernels use the layer mass Δp / g, never ρ Δz: in floating point
##### ρ (Δp / (g ρ)) differs from Δp / g by a rounding, and the positivity, telescoping and ledger
##### identities hold exactly only in Δp. The physical depth is what the physics reads.
#####

const PLG = PressureLevelGrid

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Pressure on the z-face `k` of column `(i, j)` of a [`PressureLevelGrid`](@ref): `A[k] + B[k] pˢ`.
"""
@inline interface_pressure(i, j, k, grid::PLG) =
    @inbounds grid.z.pressure_coefficient[k] + grid.z.surface_pressure_coefficient[k] * grid.z.surface_pressure[i, j, 1]

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Pressure thickness of layer `k` of column `(i, j)` of a [`PressureLevelGrid`](@ref) at the
current surface pressure: `ΔA + ΔB pˢ`, positive.
"""
@inline layer_pressure_thickness(i, j, k, grid::PLG) =
    @inbounds (grid.z.pressure_coefficient[k] - grid.z.pressure_coefficient[k+1]) +
              (grid.z.surface_pressure_coefficient[k] - grid.z.surface_pressure_coefficient[k+1]) * grid.z.surface_pressure[i, j, 1]

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Pressure thickness of layer `k` of column `(i, j)` at the surface pressure of the start of the
stage (`previous_surface_pressure`).
"""
@inline previous_layer_pressure_thickness(i, j, k, grid::PLG) =
    @inbounds (grid.z.pressure_coefficient[k] - grid.z.pressure_coefficient[k+1]) +
              (grid.z.surface_pressure_coefficient[k] - grid.z.surface_pressure_coefficient[k+1]) * grid.z.previous_surface_pressure[i, j, 1]

# Physical depth of layer k: Δp / (g ρ)
@inline layer_depthᶜᶜᶜ(i, j, k, grid::PLG) =
    @inbounds layer_pressure_thickness(i, j, k, grid) / (grid.z.gravitational_acceleration * grid.z.density[i, j, k])

@inline previous_layer_depthᶜᶜᶜ(i, j, k, grid::PLG) =
    @inbounds previous_layer_pressure_thickness(i, j, k, grid) / (grid.z.gravitational_acceleration * grid.z.density[i, j, k])

# Depth at a z-face: the mean of the two layers it separates
@inline layer_depthᶜᶜᶠ(i, j, k, grid::PLG) = (layer_depthᶜᶜᶜ(i, j, k - 1, grid) + layer_depthᶜᶜᶜ(i, j, k, grid)) / 2

for (LX, LY, horizontal) in ((:ᶜ, :ᶜ, nothing), (:ᶠ, :ᶜ, :ℑxᶠᵃᵃ), (:ᶜ, :ᶠ, :ℑyᵃᶠᵃ), (:ᶠ, :ᶠ, :ℑxyᶠᶠᵃ)), LZ in (:ᶜ, :ᶠ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rcp_zspacing = Symbol(:Δz⁻¹, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)
    column_depth = Symbol(:layer_depth, :ᶜᶜ, LZ)
    ℓx = LX == :ᶜ ? :Center : :Face
    ℓy = LY == :ᶜ ? :Center : :Face
    ℓz = LZ == :ᶜ ? :Center : :Face

    depth = isnothing(horizontal) ? :($column_depth(i, j, k, grid)) : :(Operators.$horizontal(i, j, k, grid, $column_depth))

    @eval begin
        @inline Operators.$zspacing(i, j, k, grid::PLG) = $depth
        @inline Operators.$rcp_zspacing(i, j, k, grid::PLG) = 1 / Operators.$zspacing(i, j, k, grid)
        @inline Operators.σⁿ(i, j, k, grid::PLG, ::$ℓx, ::$ℓy, ::$ℓz) = Operators.$zspacing(i, j, k, grid) / Operators.$rspacing(i, j, k, grid)
    end
end

# The scaling at the start of the stage, at cell centers (as `MutableVerticalDiscretization` keeps σ⁻ only there).
# The density is that of the current state: the grid keeps one density.
@inline Operators.σ⁻(i, j, k, grid::PLG, ::Center, ::Center, ::Center) =
    previous_layer_depthᶜᶜᶜ(i, j, k, grid) / Operators.Δrᶜᶜᶜ(i, j, k, grid)

#####
##### Heights above the ground
#####

# Height of z-face k above the ground (z = 0 at face 1); halo faces below the ground are negative
@inline function face_height(i, j, k, grid::PLG)
    z = zero(grid)
    for m in 1:k-1
        z += layer_depthᶜᶜᶜ(i, j, m, grid)
    end
    for m in k:0
        z -= layer_depthᶜᶜᶜ(i, j, m, grid)
    end
    return z
end

@inline Grids.znode(i, j, k, grid::PLG, ℓx, ℓy, ::Face) = face_height(i, j, k, grid)
@inline Grids.znode(i, j, k, grid::PLG, ℓx, ℓy, ::Center) = face_height(i, j, k, grid) + layer_depthᶜᶜᶜ(i, j, k, grid) / 2

# `set!(field, f)` evaluates `f` at `_node`, whose vertical entry is `rnode` in Oceananigans; on this grid the
# vertical entry is the height above the ground. A copy of Oceananigans' `_node` methods with `rnode` → `znode`,
# as the terrain-following grid has.
#
# TODO (upstream Oceananigans): if stock `_node` used `znode` instead of `rnode` (a no-op for every existing
# grid, since `znode` defaults to `rnode`), the override of `znode` above would flow through `set!` and this
# block could be deleted (the same TODO as in TerrainFollowingDiscretization).
@inline Grids._node(i, j, k, grid::PLG, ℓx, ℓy, ℓz) =
    (Grids.ξnode(i, j, k, grid, ℓx, ℓy, ℓz), Grids.ηnode(i, j, k, grid, ℓx, ℓy, ℓz), Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx::Nothing, ℓy, ℓz) = (Grids.ηnode(i, j, k, grid, ℓx, ℓy, ℓz), Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx, ℓy::Nothing, ℓz) = (Grids.ξnode(i, j, k, grid, ℓx, ℓy, ℓz), Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx::Nothing, ℓy::Nothing, ℓz) = tuple(Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx, ℓy, ℓz::Nothing) = (Grids.ξnode(i, j, k, grid, ℓx, ℓy, ℓz), Grids.ηnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx, ℓy::Nothing, ℓz::Nothing) = tuple(Grids.ξnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx::Nothing, ℓy, ℓz::Nothing) = tuple(Grids.ηnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline Grids._node(i, j, k, grid::PLG, ℓx::Nothing, ℓy::Nothing, ℓz::Nothing) = tuple()
