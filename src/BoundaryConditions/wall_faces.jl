#####
##### Wall faces
#####
##### The bulk fluxes may be placed on any of the six boundaries of a bounded domain.
##### Oceananigans evaluates a boundary condition with the two indices tangential to its
##### face: (i, j) on the bottom and top, (j, k) on the west and east, and (i, k) on the
##### south and north. Everything that distinguishes the faces — the near-wall cell, the
##### orientation of a flux, the tangential wind, the wall distance — is collected here
##### so that each flux formula is written once.
#####

const HorizontalWall = Union{Bottom, Top}
const XNormalWall = Union{West, East}
const YNormalWall = Union{South, North}
const VerticalWall = Union{XNormalWall, YNormalWall}
const LeftWall = Union{West, South, Bottom}
const RightWall = Union{East, North, Top}

# Indices (i, j, k) of the near-wall cell from the boundary-tangential indices (ℓ, m)
@inline near_wall_indices(ℓ, m, grid, ::Bottom) = (ℓ, m, 1)
@inline near_wall_indices(ℓ, m, grid, ::Top) = (ℓ, m, grid.Nz)
@inline near_wall_indices(ℓ, m, grid, ::West) = (1, ℓ, m)
@inline near_wall_indices(ℓ, m, grid, ::East) = (grid.Nx, ℓ, m)
@inline near_wall_indices(ℓ, m, grid, ::South) = (ℓ, 1, m)
@inline near_wall_indices(ℓ, m, grid, ::North) = (ℓ, grid.Ny, m)

# Fluxes point along the positive coordinate direction, so a positive flux enters the
# domain through a left wall and leaves it through a right wall. A flux of magnitude F
# directed out of the domain is therefore -F on a left wall and +F on a right wall.
@inline outward_flux_sign(::LeftWall) = -1
@inline outward_flux_sign(::RightWall) = 1

# The velocity component normal to a wall, which cannot receive drag there
wall_normal_direction(::XNormalWall) = XDirection()
wall_normal_direction(::YNormalWall) = YDirection()
wall_normal_direction(::HorizontalWall) = ZDirection()

# Wall state (temperature, humidity) at the boundary-tangential indices (ℓ, m), from a
# number or from a two-dimensional field living on the wall
@inline wall_value(ℓ, m, grid, side, x::Number, clock) = x
@inline wall_value(ℓ, m, grid, ::HorizontalWall, field::AbstractArray, clock) = @inbounds field[ℓ, m, 1]
@inline wall_value(ℓ, m, grid, ::XNormalWall,    field::AbstractArray, clock) = @inbounds field[1, ℓ, m]
@inline wall_value(ℓ, m, grid, ::YNormalWall,    field::AbstractArray, clock) = @inbounds field[ℓ, 1, m]

# A function is evaluated at the centre of the wall face with the non-`Flat` coordinates of the
# wall followed by the time, as Oceananigans evaluates boundary-condition and forcing functions:
# `f(x, y, t)` on the bottom and top, `f(y, z, t)` on the west and east, `f(x, z, t)` on the
# south and north, with the coordinate of a `Flat` direction dropped (`node` does that)
@inline wall_value(ℓ, m, grid, ::HorizontalWall, f::Function, clock) = f(node(ℓ, m, 1, grid, Center(), Center(), nothing)..., clock.time)
@inline wall_value(ℓ, m, grid, ::XNormalWall,    f::Function, clock) = f(node(1, ℓ, m, grid, nothing, Center(), Center())..., clock.time)
@inline wall_value(ℓ, m, grid, ::YNormalWall,    f::Function, clock) = f(node(ℓ, 1, m, grid, Center(), nothing, Center())..., clock.time)

# Wall-normal distance from the wall to the near-wall cell centre: the height of the
# first cell centre for the bottom wall, and half the cell width otherwise
@inline wall_distance(i, j, k, grid, ::Bottom) = znode(i, j, k, grid, Center(), Center(), Center())
@inline wall_distance(i, j, k, grid, ::Top) = Δzᶜᶜᶜ(i, j, k, grid) / 2
@inline wall_distance(i, j, k, grid, ::XNormalWall) = Δxᶜᶜᶜ(i, j, k, grid) / 2
@inline wall_distance(i, j, k, grid, ::YNormalWall) = Δyᶜᶜᶜ(i, j, k, grid) / 2

# Height of the wall next to the near-wall cell, for the potential energy in the static energy
@inline wall_height(i, j, k, grid, ::Bottom) = znode(i, j, k,     grid, Center(), Center(), Face())
@inline wall_height(i, j, k, grid, ::Top) = znode(i, j, k + 1, grid, Center(), Center(), Face())
@inline wall_height(i, j, k, grid, ::VerticalWall) = znode(i, j, k,     grid, Center(), Center(), Center())

# Buoyancy stabilizes or destabilizes a surface layer on a horizontal wall only, and the
# bulk Richardson number changes sign under the top wall (cold above warm is unstable)
@inline stability_sign(::Bottom) = 1
@inline stability_sign(::Top) = -1

#####
##### Tangential wind at the near-wall cell
#####
##### `direction` is `nothing` for a scalar flux, evaluated at (Center, Center, Center),
##### or the `XDirection`, `YDirection`, or `ZDirection` of the momentum component that
##### receives drag, evaluated at that component's face.
#####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline near_wall_velocity(i, j, k, grid, ::XDirection, fields) = @inbounds fields.u[i, j, k]
@inline near_wall_velocity(i, j, k, grid, ::YDirection, fields) = @inbounds fields.v[i, j, k]
@inline near_wall_velocity(i, j, k, grid, ::ZDirection, fields) = @inbounds fields.w[i, j, k]

# Horizontal walls: the tangential wind is (u, v)
@inline tangential_speed²(i, j, k, grid, ::HorizontalWall, ::Nothing, fields) =
    ℑxᶜᵃᵃ(i, j, k, grid, ϕ², fields.u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², fields.v)

@inline tangential_speed²(i, j, k, grid, ::HorizontalWall, ::XDirection, fields) =
    ϕ²(i, j, k, grid, fields.u) + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², fields.v)

@inline tangential_speed²(i, j, k, grid, ::HorizontalWall, ::YDirection, fields) =
    ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², fields.u) + ϕ²(i, j, k, grid, fields.v)

# Walls normal to x: the tangential wind is (v, w)
@inline tangential_speed²(i, j, k, grid, ::XNormalWall, ::Nothing, fields) =
    ℑyᵃᶜᵃ(i, j, k, grid, ϕ², fields.v) + ℑzᵃᵃᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(i, j, k, grid, ::XNormalWall, ::YDirection, fields) =
    ϕ²(i, j, k, grid, fields.v) + ℑyzᵃᶠᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(i, j, k, grid, ::XNormalWall, ::ZDirection, fields) =
    ℑyzᵃᶜᶠ(i, j, k, grid, ϕ², fields.v) + ϕ²(i, j, k, grid, fields.w)

# Walls normal to y: the tangential wind is (u, w)
@inline tangential_speed²(i, j, k, grid, ::YNormalWall, ::Nothing, fields) =
    ℑxᶜᵃᵃ(i, j, k, grid, ϕ², fields.u) + ℑzᵃᵃᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(i, j, k, grid, ::YNormalWall, ::XDirection, fields) =
    ϕ²(i, j, k, grid, fields.u) + ℑxzᶠᵃᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(i, j, k, grid, ::YNormalWall, ::ZDirection, fields) =
    ℑxzᶜᵃᶠ(i, j, k, grid, ϕ², fields.u) + ϕ²(i, j, k, grid, fields.w)

#####
##### Near-wall wind, with or without the temporally filtered surface state
#####
##### Filtering (`FilteredSurfaceVelocities`, `filtered_surface_state.jl`) is defined for
##### the bottom wall only; materialization rejects it elsewhere.
#####

@inline near_wall_velocity(i, j, k, grid, side, direction, fields, ::Nothing) =
    near_wall_velocity(i, j, k, grid, direction, fields)

@inline wall_wind_speed²(i, j, k, grid, side, direction, fields, ::Nothing) =
    tangential_speed²(i, j, k, grid, side, direction, fields)

@inline near_wall_velocity(i, j, k, grid, ::Bottom, ::XDirection, fields, fv::FilteredSurfaceVelocities) = @inbounds fv.u[i, j, 1]
@inline near_wall_velocity(i, j, k, grid, ::Bottom, ::YDirection, fields, fv::FilteredSurfaceVelocities) = @inbounds fv.v[i, j, 1]

@inline wall_wind_speed²(i, j, k, grid, ::Bottom, ::XDirection, fields, fv::FilteredSurfaceVelocities) = wind_speed²ᶠᶜᶜ(i, j, grid, fields, fv)
@inline wall_wind_speed²(i, j, k, grid, ::Bottom, ::YDirection, fields, fv::FilteredSurfaceVelocities) = wind_speed²ᶜᶠᶜ(i, j, grid, fields, fv)
@inline wall_wind_speed²(i, j, k, grid, ::Bottom, ::Nothing,    fields, fv::FilteredSurfaceVelocities) = wind_speed²ᶜᶜᶜ(i, j, grid, fields, fv)

validate_wall_filtering(side, ::Nothing) = nothing
validate_wall_filtering(::Bottom, ::FilteredSurfaceVelocities) = nothing
validate_wall_filtering(side, ::FilteredSurfaceVelocities) =
    throw(ArgumentError("Temporally filtered surface state is only supported on the bottom boundary, not on the $(typeof(side)) boundary"))
