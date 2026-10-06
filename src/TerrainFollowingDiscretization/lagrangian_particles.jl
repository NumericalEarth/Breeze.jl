#####
##### Lagrangian particles on terrain-following grids
#####
##### Oceananigans stores particle positions in physical coordinates, but
##### interpolates against the grid's one-dimensional vertical coordinate r
##### (`FractionalIndices` measures the vertical entry of a node with `rnodes`).
##### On a TerrainFollowingGrid the two differ,
#####
#####   z(x, y, r) = r + Σₙ hₙ(x, y) bₙ(r),
#####
##### so the interpolation node must carry r, obtained by inverting z(x, y, ·).
#####
##### That inversion is injected through `flattened_node`, the hook Oceananigans
##### already uses to build the particle interpolation node (see `advect_particle`
##### and `update_property!`, its only callers). Overriding this one function makes
##### both velocity interpolation and `tracked_fields` sampling terrain-aware
##### while leaving the generic `Fields.interpolate` contract untouched — its
##### third coordinate stays r for every other caller, including the wall-model
##### reference height in `BoundaryConditions/filtered_surface_state.jl`.
#####
##### Positions are still advanced with the physical velocity components,
##### dx/dt = u, dy/dt = v, dz/dt = w, and the vertical walls are the bounding
##### coordinate surfaces: the local terrain z(x, y, r_bottom) below and the flat
##### lid z(x, y, r_top) above.
#####

# `TerrainFollowingGrid` with the horizontal topologies exposed as parameters. The
# four `flattened_node` methods below need these so each is strictly more specific
# than the corresponding Oceananigans method (`XFlatGrid`, `YFlatGrid`,
# `XYFlatGrid`) and no ambiguity arises. z is always `Bounded` on these grids, so
# the Z-Flat variants have an empty intersection and cannot be ambiguous.
const TerrainFollowingGridTopology{TX, TY} =
    AbstractUnderlyingGrid{<:Any, TX, TY, <:Oceananigans.Grids.Bounded, <:TFVD}

# Immersed boundaries wrap the underlying grid, so neither the `flattened_node`
# override nor `advect_particle` below dispatches on them: particle tracking would
# silently fall back to the generic (flat-grid) implementation and interpret the
# stored physical altitude as r. Reject the combination instead of failing quietly.
const ImmersedTerrainFollowingGrid = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TerrainFollowingGrid}

AtmosphereModels.validate_particles(::AbstractLagrangianParticles, ::ImmersedTerrainFollowingGrid) =
    throw(ArgumentError("Lagrangian particles are not supported on an ImmersedBoundaryGrid whose " *
                        "underlying grid uses a TerrainFollowingVerticalDiscretization, because " *
                        "particle interpolation and the immersed-boundary bounce have not been " *
                        "reconciled with the terrain coordinate map. Use the terrain-following " *
                        "grid directly (its lower boundary already follows the terrain)."))

#####
##### The coordinate map z(x, y, r) and its inverse, at a particle position
#####

# Horizontal (bi)linear interpolation of a terrain component hₙ to the particle
# position. The vertical location is `nothing`, so only the horizontal fractional
# indices are formed and the dummy third entry is discarded. Interpolating hₙ at
# `(Center, Center)` is consistent with the staggered `terrain_at_stagger` used by
# `znode`: at a Face node the two agree exactly, since a Face lies midway between
# the two Centers that the linear interpolant blends.
@inline function terrain_component_at_particle(x, y, grid, component)
    location = (Center(), Center(), nothing)
    indices = FractionalIndices((x, y, 0), grid, location...)
    return interpolate(indices, component, location, grid)
end

# The r faces bounding the domain. Deriving the walls and the Newton bracket from
# these — rather than from `formulation.z_top`, which is the domain *height* — keeps
# the assumption that r starts at zero out of the particle code.
@inline function terrain_reference_bounds(grid)
    Nz = size(grid, 3)
    return rnode(1, grid, Face()), rnode(Nz + 1, grid, Face())
end

@inline function terrain_physical_height(x, y, r, grid, f::LinearDecay)
    h = terrain_component_at_particle(x, y, grid, f.h)
    return r + h * b_linear(r, f.z_top)
end

@inline function terrain_physical_height(x, y, r, grid, f::TwoLevelDecay)
    h₁ = terrain_component_at_particle(x, y, grid, f.h₁)
    h₂ = terrain_component_at_particle(x, y, grid, f.h₂)
    return r + h₁ * b_two_level(r, f.z_top, f.large_scale_height) +
               h₂ * b_two_level(r, f.z_top, f.small_scale_height)
end

@inline terrain_physical_height(x, y, r, grid::TerrainFollowingGrid) =
    terrain_physical_height(x, y, r, grid, grid.z.formulation)

"""
$(TYPEDSIGNATURES)

Return the physical altitudes `(z_bottom, z_top)` of the bounding coordinate
surfaces above the horizontal position `(x, y)`: the terrain surface
`z(x, y, r_bottom)` and the lid `z(x, y, r_top)`. These are the walls a particle
bounces off, and they are evaluated with the same interpolated terrain components
as [`terrain_reference_coordinate`](@ref), so a particle sitting exactly on either
wall inverts to exactly `r_bottom` or `r_top`.
"""
@inline function terrain_wall_heights(x, y, grid::TerrainFollowingGrid)
    r_bottom, r_top = terrain_reference_bounds(grid)
    z_bottom = terrain_physical_height(x, y, r_bottom, grid)
    z_top = terrain_physical_height(x, y, r_top, grid)
    return z_bottom, z_top
end

# Linear decay is affine in r, z = r (1 − h/z_top) + h, so the inverse is exact.
@inline function terrain_reference_coordinate(x, y, z, grid, f::LinearDecay)
    h = terrain_component_at_particle(x, y, grid, f.h)
    r_bottom, r_top = terrain_reference_bounds(grid)
    r = (z - h) / (1 - h / f.z_top)
    # Clamp so a position outside the vertical domain cannot form a fractional
    # index outside the grid, matching the bracketed two-level solve below.
    return clamp(r, r_bottom, r_top)
end

# Number of Newton iterations in the two-level inversion. With the maintained
# bracket every iteration either takes a Newton step or bisects, so the root is
# located to within (r_top − r_bottom) / 2^newton_iterations even in the worst
# case; in practice Newton reaches roundoff (≈1e-13 m) in about four iterations
# because z(r) is smooth and monotonic.
const newton_iterations = 8

@inline function terrain_reference_coordinate(x, y, z, grid, f::TwoLevelDecay)
    h₁ = terrain_component_at_particle(x, y, grid, f.h₁)
    h₂ = terrain_component_at_particle(x, y, grid, f.h₂)
    h = h₁ + h₂
    z_top = f.z_top
    s₁ = f.large_scale_height
    s₂ = f.small_scale_height

    # The sinh bases have no closed-form inverse. The linear-decay inverse is a
    # close initial estimate; the bracket keeps the fixed-iteration solve inside
    # the physical domain (z(r) is monotonically increasing because σ = ∂z/∂r > 0).
    r_lower, r_upper = terrain_reference_bounds(grid)
    r = clamp((z - h) / (1 - h / z_top), r_lower, r_upper)

    # Hoist the two basis normalizations: they are independent of r, and each
    # iteration is otherwise a handful of transcendentals per particle.
    n₁ = b_two_level_normalization(z_top, s₁)
    n₂ = b_two_level_normalization(z_top, s₂)

    for _ in 1:newton_iterations
        b₁ = b_two_level(r, z_top, s₁, n₁)
        b₂ = b_two_level(r, z_top, s₂, n₂)
        mapped_z = r + h₁ * b₁ + h₂ * b₂
        residual = mapped_z - z

        r_lower = ifelse(residual ≤ 0, r, r_lower)
        r_upper = ifelse(residual ≥ 0, r, r_upper)

        ∂z∂r = 1 + h₁ * b′_two_level(r, z_top, s₁, n₁) + h₂ * b′_two_level(r, z_top, s₂, n₂)

        newton_r = r - residual / ∂z∂r
        newton_is_bracketed = (newton_r > r_lower) & (newton_r < r_upper)
        r = ifelse(newton_is_bracketed, newton_r, (r_lower + r_upper) / 2)
    end

    return r
end

"""
$(TYPEDSIGNATURES)

Invert the terrain coordinate map for the reference coordinate `r` of the physical
position `(x, y, z)`, clamped to the grid's bounding r faces.
"""
@inline terrain_reference_coordinate(x, y, z, grid::TerrainFollowingGrid) =
    terrain_reference_coordinate(x, y, z, grid, grid.z.formulation)

#####
##### The particle interpolation node
#####

# One method per horizontal-topology combination, mirroring Oceananigans'
# `flattened_node`, so that every particle interpolation on a terrain-following
# grid is performed at the reference coordinate. `terrain_reference_coordinate` is
# evaluated exactly once per call; `advect_particle` below reuses the result for
# all three velocity components.
@inline LagrangianParticleTracking.flattened_node((x, y, z), grid::TerrainFollowingGridTopology) =
    (x, y, terrain_reference_coordinate(x, y, z, grid))

@inline LagrangianParticleTracking.flattened_node((x, y, z), grid::TerrainFollowingGridTopology{Flat}) =
    (y, terrain_reference_coordinate(x, y, z, grid))

@inline LagrangianParticleTracking.flattened_node((x, y, z), grid::TerrainFollowingGridTopology{<:Any, Flat}) =
    (x, terrain_reference_coordinate(x, y, z, grid))

@inline LagrangianParticleTracking.flattened_node((x, y, z), grid::TerrainFollowingGridTopology{Flat, Flat}) =
    tuple(terrain_reference_coordinate(x, y, z, grid))

#####
##### Advection
#####

# TODO (upstream Oceananigans): this mirrors `LagrangianParticleTracking.advect_particle`
# and differs from it in exactly one respect — the vertical walls. Stock uses
# `rnode(i, j, 1, …)` and `rnode(i, j, kᴿ, …)`, which are reference coordinates,
# whereas particle positions are physical altitudes. If those bounds came from a grid
# hook (say `particle_vertical_bounds(x, y, grid)`, defaulting to those two `rnode`s)
# this method could be deleted and the `flattened_node` override above would be the
# only extension needed. Until then, upstream changes to `advect_particle` — new
# features as well as bug fixes — have to be mirrored here.
@inline function LagrangianParticleTracking.advect_particle((x, y, z), particles, p, restitution,
                                                            grid::TerrainFollowingGrid, Δt, velocities)
    # Inverts z(x, y, r) for r once; the fractional indices formed below reuse it,
    # so the two-level Newton solve runs exactly once per particle per step.
    X = LagrangianParticleTracking.flattened_node((x, y, z), grid)

    # Current particle indices, from the cell interfaces.
    face_indices = FractionalIndices(X, grid, Face(), Face(), Face())
    i, _, _ = interpolator(face_indices.i)
    j, _, _ = interpolator(face_indices.j)
    k, _, _ = interpolator(face_indices.k)

    ℓu = (Face(), Center(), Center())
    ℓv = (Center(), Face(), Center())
    ℓw = (Center(), Center(), Face())

    u = interpolate(FractionalIndices(X, grid, ℓu...), velocities.u, ℓu, grid)
    v = interpolate(FractionalIndices(X, grid, ℓv...), velocities.v, ℓv, grid)
    w = interpolate(FractionalIndices(X, grid, ℓw...), velocities.w, ℓw, grid)

    u = LagrangianParticleTracking.particle_u_velocity(particles, p, u)
    v = LagrangianParticleTracking.particle_v_velocity(particles, p, v)
    w = LagrangianParticleTracking.particle_w_velocity(particles, p, w)

    # Positions are physical, so no vertical metric factor appears; the horizontal
    # metrics convert to degrees on a LatitudeLongitudeGrid and are unity otherwise.
    x = x + LagrangianParticleTracking.x_metric(i, j, grid) * u * Δt
    y = y + LagrangianParticleTracking.y_metric(i, j, grid) * v * Δt
    z = z + w * Δt

    tx, ty, tz = map(instantiate, topology(grid))
    Nx, Ny, _ = size(grid)
    i_right = LagrangianParticleTracking.rightmost_interface_index(tx, Nx)
    j_right = LagrangianParticleTracking.rightmost_interface_index(ty, Ny)

    x_left = ξnode(1, j, k, grid, Face(), Face(), Face())
    y_left = ηnode(i, 1, k, grid, Face(), Face(), Face())
    x_right = ξnode(i_right, j, k, grid, Face(), Face(), Face())
    y_right = ηnode(i, j_right, k, grid, Face(), Face(), Face())

    x = LagrangianParticleTracking.enforce_boundary_conditions(tx, x, x_left, x_right, restitution)
    y = LagrangianParticleTracking.enforce_boundary_conditions(ty, y, y_left, y_right, restitution)

    # The walls are evaluated at the *new* horizontal position, so the particle
    # bounces off the terrain it has actually moved over (and, on a Periodic grid,
    # off the terrain on the far side after wrapping).
    #
    # Note this reflects the vertical coordinate alone: on a sloping surface it is a
    # vertical bounce, not a specular reflection about the surface normal, so neither
    # the horizontal position nor the velocity direction is altered. That is exact
    # for the flat lid and for `restitution = 0` (the particle settles onto the
    # ground), and approximate for an elastic bounce off a steep slope.
    z_bottom, z_top = terrain_wall_heights(x, y, grid)
    z = LagrangianParticleTracking.enforce_boundary_conditions(tz, z, z_bottom, z_top, restitution)

    return (x, y, z)
end
