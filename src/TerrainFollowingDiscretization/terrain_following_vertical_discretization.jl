#####
##### TerrainFollowingVerticalDiscretization — a grid-owned terrain-following
##### vertical coordinate.
#####
##### The coordinate map is
#####
#####   z(x, y, r) = r + Σₙ hₙ(x, y) · bₙ(r)
#####
##### with the formulation supplying the terrain components hₙ and the decay
##### profiles bₙ(r). Unlike `MutableVerticalDiscretization` (which *stores* the
##### scaling σ and has no horizontal slope), this type stores the *generator*
##### — the terrain components + decay law — and DERIVES the Jacobian
##### σ = ∂z/∂r and the slopes ∂z/∂x, ∂z/∂y in the operators, so σ and the
##### slope cannot drift out of consistency. `LinearDecay` (Gal-Chen) and
##### `TwoLevelDecay` (Schär et al. 2002) are two formulations of the one type.
#####

using Oceananigans.Grids: AbstractVerticalCoordinate, AbstractUnderlyingGrid

struct TerrainFollowingVerticalDiscretization{C, D, E, F, FM} <: AbstractVerticalCoordinate
    "Face-centered reference coordinate r"
    cᵃᵃᶠ :: C
    "Cell-centered reference coordinate r"
    cᵃᵃᶜ :: D
    "Face-centered reference spacing Δr"
    Δᵃᵃᶠ :: E
    "Cell-centered reference spacing Δr"
    Δᵃᵃᶜ :: F
    "Terrain-decay formulation (LinearDecay | TwoLevelDecay) — the generator of σ and the slopes"
    formulation :: FM
end

"""
    TerrainFollowingVerticalDiscretization(r_faces; formulation=LinearDecay())

Skeleton constructor. `r_faces` is the reference (computational) r face
specification — a range, vector, or function — exactly as for a static z grid.
The terrain components inside `formulation` are filled later by
[`materialize_terrain!`](@ref) once the horizontal grid exists.
"""
TerrainFollowingVerticalDiscretization(r_faces; formulation = LinearDecay()) =
    TerrainFollowingVerticalDiscretization(r_faces, nothing, nothing, nothing, formulation)

const TFVD = TerrainFollowingVerticalDiscretization
const RegularTerrainFollowingVerticalDiscretization = TerrainFollowingVerticalDiscretization{<:Any, <:Any, <:Any, <:Number}

Oceananigans.Grids.coordinate_summary(::Oceananigans.Grids.Bounded,
                                      z::RegularTerrainFollowingVerticalDiscretization,
                                      name) =
    string("regularly spaced with terrain-following Δr=",
           Oceananigans.Utils.prettysummary(z.Δᵃᵃᶜ))

Oceananigans.Grids.coordinate_summary(::Oceananigans.Grids.Bounded, z::TFVD, name) =
    string("variably spaced with terrain-following min(Δr)=",
           Oceananigans.Utils.prettysummary(minimum(parent(z.Δᵃᵃᶜ))),
           ", max(Δr)=",
           Oceananigans.Utils.prettysummary(maximum(parent(z.Δᵃᵃᶜ))))

# Validate the reference (r) face specification, keeping the formulation; the
# reference coordinate arrays are built later by `generate_coordinate`.
function Oceananigans.Grids.validate_dimension_specification(T, ξ::TFVD, dir, N, FT)
    cᶠ = Oceananigans.Grids.validate_dimension_specification(T, ξ.cᵃᵃᶠ, dir, N, FT)
    return TerrainFollowingVerticalDiscretization(cᶠ, ξ.cᵃᵃᶜ, ξ.Δᵃᵃᶠ, ξ.Δᵃᵃᶜ, ξ.formulation)
end

Adapt.adapt_structure(to, z::TFVD) =
    TerrainFollowingVerticalDiscretization(Adapt.adapt(to, z.cᵃᵃᶠ),
                                           Adapt.adapt(to, z.cᵃᵃᶜ),
                                           Adapt.adapt(to, z.Δᵃᵃᶠ),
                                           Adapt.adapt(to, z.Δᵃᵃᶜ),
                                           Adapt.adapt(to, z.formulation))

Oceananigans.Architectures.on_architecture(arch, z::TFVD) =
    TerrainFollowingVerticalDiscretization(Oceananigans.Architectures.on_architecture(arch, z.cᵃᵃᶠ),
                                           Oceananigans.Architectures.on_architecture(arch, z.cᵃᵃᶜ),
                                           Oceananigans.Architectures.on_architecture(arch, z.Δᵃᵃᶠ),
                                           Oceananigans.Architectures.on_architecture(arch, z.Δᵃᵃᶜ),
                                           Oceananigans.Architectures.on_architecture(arch, z.formulation))

# Build the reference (r) coordinate arrays exactly like a static z grid; the
# formulation's terrain components start as `nothing` (skeleton) and are filled
# by `materialize_terrain!` once the grid's horizontal nodes exist.
function Oceananigans.Grids.generate_coordinate(FT, topo, size, halo, coordinate::TFVD, coordinate_name, dim::Int, arch)
    dim == 3 || throw(ArgumentError("TerrainFollowingVerticalDiscretization is only supported in the z (third) dimension"))
    coordinate_name == :z || throw(ArgumentError("TerrainFollowingVerticalDiscretization is only supported for the z-coordinate"))

    Nz = size[3]; Hz = halo[3]
    r_faces = coordinate.cᵃᵃᶠ
    Lz, rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ = Oceananigans.Grids.generate_coordinate(FT, topo[3](), Nz, Hz, r_faces, :r, arch)

    # Allocate the formulation's terrain-component arrays (zero-filled; the
    # bottom of the domain is taken as z = 0, so z_top = Lz). Filled later by
    # `materialize_terrain!`.
    formulation = allocate_formulation(coordinate.formulation, FT, arch, size, halo, topo, Lz)
    z = TerrainFollowingVerticalDiscretization(rᵃᵃᶠ, rᵃᵃᶜ, Δrᵃᵃᶠ, Δrᵃᵃᶜ, formulation)
    return Lz, z
end

#####
##### Metric operators — derived from the formulation
#####
##### σⁿ = ∂z/∂r (Jacobian) and znode = z(x,y,r) are computed from the
##### formulation's terrain components + decay profiles. Δz spacings follow as
##### Δr · σⁿ. Defining σⁿ over (i,j,k) is what makes the Jacobian k-dependent
##### for TwoLevelDecay without any 3-D σ storage.
#####

# A terrain-following coordinate lives in the vertical-coordinate slot (`CZ`, the
# 5th type parameter) of any `AbstractUnderlyingGrid`, so this single alias covers
# both `RectilinearGrid` and `LatitudeLongitudeGrid`. `z` must be `Bounded`. This
# mirrors Oceananigans' own `AbstractMutableGrid` alias for `MutableVerticalDiscretization`.
const TerrainFollowingGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Oceananigans.Grids.Bounded, <:TFVD}

# Preserve the materialised terrain components when Oceananigans reconstructs
# the grid (e.g. `on_architecture(CPU(), gpu_grid)` inside `set_to_function!`).
# The default `cpu_face_constructor_z` returns only r-face coordinates, which
# would make the downstream `generate_coordinate` allocate a fresh skeleton
# formulation with zero h / ∂x_h / ∂y_h — and the `node()` override would then
# return r instead of physical altitude. By wrapping the full TFVD here, the
# formulation arrays survive the rebuild.
@inline Oceananigans.Grids.cpu_face_constructor_z(grid::TerrainFollowingGrid) =
    TerrainFollowingVerticalDiscretization(Oceananigans.Grids.cpu_face_constructor_r(grid);
                                            formulation = Oceananigans.Architectures.on_architecture(Oceananigans.Architectures.CPU(), grid.z.formulation))

@inline Oceananigans.Operators.σⁿ(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy, ℓz) =
    terrain_following_σ(i, j, k, grid, grid.z.formulation, ℓx, ℓy, ℓz)

# The terrain is static, so the previous-step scaling equals the current one.
# (The generic fallback returns `one(grid)` ≡ a flat grid, which makes any
# consumer of σ⁻ see spurious grid motion and blow up.)
@inline Oceananigans.Operators.σ⁻(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy, ℓz) = Oceananigans.Operators.σⁿ(i, j, k, grid, ℓx, ℓy, ℓz)

@inline Oceananigans.Grids.znode(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy, ℓz) =
    rnode(i, j, k, grid, ℓx, ℓy, ℓz) +
    terrain_following_Δz_surface(i, j, k, grid, grid.z.formulation, ℓx, ℓy, ℓz)

# `set!(field, f)` evaluates the initialiser at `node(i, j, k, grid, ℓx, ℓy, ℓz)`,
# which nullifies Flat dimensions and then delegates to `Oceananigans.Grids._node`
# to drop `Nothing`-located dimensions. Stock `_node` returns `rnode` (the reference
# coordinate r) as the vertical entry — on a terrain-following grid that is *not*
# the physical altitude. We extend `_node` (not `node`) on grids whose vertical
# discretisation is a TFVD so the vertical entry is `znode` (z = r + h(x,y)·b(r),
# the actual cell-centre altitude), making `set!(field, (x, y, z) -> f(z))`
# evaluate `f` at the true altitude. Extending `_node` reuses Oceananigans' Flat
# handling unchanged — we only mirror its `Nothing`-dropping layer.
#
# These are a copy of Oceananigans' eight `_node` methods with `rnode` → `znode`.
# The four that carry no vertical entry (ℓz === nothing) are identical to stock and
# exist only to resolve dispatch ambiguity between our grid-typed methods and
# Oceananigans' `Nothing`-location-typed ones.
#
# TODO (upstream Oceananigans): if stock `_node` used `znode` instead of `rnode`
# (a no-op for every existing grid, since `znode` defaults to `rnode` in
# vertical_discretization.jl), our `znode` override above would flow through `set!`
# automatically and this entire block could be deleted.
@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy, ℓz) =
    (ξnode(i, j, k, grid, ℓx, ℓy, ℓz),
     ηnode(i, j, k, grid, ℓx, ℓy, ℓz),
     Oceananigans.Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx::Nothing, ℓy, ℓz) =
    (ηnode(i, j, k, grid, ℓx, ℓy, ℓz), Oceananigans.Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy::Nothing, ℓz) =
    (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), Oceananigans.Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx::Nothing, ℓy::Nothing, ℓz) =
    tuple(Oceananigans.Grids.znode(i, j, k, grid, ℓx, ℓy, ℓz))

# ℓz === nothing: no vertical entry, identical to stock `_node` (disambiguators only).
@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy, ℓz::Nothing) =
    (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx, ℓy::Nothing, ℓz::Nothing) =
    tuple(ξnode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx::Nothing, ℓy, ℓz::Nothing) =
    tuple(ηnode(i, j, k, grid, ℓx, ℓy, ℓz))

@inline Oceananigans.Grids._node(i, j, k, grid::TerrainFollowingGrid, ℓx::Nothing, ℓy::Nothing, ℓz::Nothing) =
    tuple()

# Vertical spacing = reference spacing × Jacobian, mirroring the mutable-grid
# operators but dispatching on the terrain-following grid type.
for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)
    ℓx = LX == :ᶜ ? :Center : :Face
    ℓy = LY == :ᶜ ? :Center : :Face
    ℓz = LZ == :ᶜ ? :Center : :Face
    @eval @inline Oceananigans.Operators.$zspacing(i, j, k, grid::TerrainFollowingGrid) =
        Oceananigans.Operators.$rspacing(i, j, k, grid) * Oceananigans.Operators.σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
end

# Horizontal slope of the coordinate surfaces, (∂z/∂x)_r and (∂z/∂y)_r, at the
# requested vertical location. Used by the terrain pressure-gradient force.
@inline ∂z∂x(i, j, k, grid::TerrainFollowingGrid, ℓz) = terrain_following_∂z∂x(i, j, k, grid, grid.z.formulation, ℓz)
@inline ∂z∂y(i, j, k, grid::TerrainFollowingGrid, ℓz) = terrain_following_∂z∂y(i, j, k, grid, grid.z.formulation, ℓz)
