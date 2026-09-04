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

# Indices (i, j, k) of the near-wall cell from the boundary-tangential indices (a, b)
@inline near_wall_indices(::Bottom, a, b, grid) = (a, b, 1)
@inline near_wall_indices(::Top,    a, b, grid) = (a, b, grid.Nz)
@inline near_wall_indices(::West,   a, b, grid) = (1, a, b)
@inline near_wall_indices(::East,   a, b, grid) = (grid.Nx, a, b)
@inline near_wall_indices(::South,  a, b, grid) = (a, 1, b)
@inline near_wall_indices(::North,  a, b, grid) = (a, grid.Ny, b)

# Fluxes point along the positive coordinate direction, so a positive flux enters the
# domain through a left wall and leaves it through a right wall. A flux of magnitude F
# directed out of the domain is therefore -F on a left wall and +F on a right wall.
@inline outward_flux_sign(::LeftWall) = -1
@inline outward_flux_sign(::RightWall) = 1

# The velocity component normal to a wall, which cannot receive drag there
wall_normal_direction(::XNormalWall) = XDirection()
wall_normal_direction(::YNormalWall) = YDirection()
wall_normal_direction(::HorizontalWall) = ZDirection()

# Wall state (temperature, humidity) at the boundary-tangential indices (a, b), from a
# number or from a two-dimensional field living on the wall
@inline wall_value(side, a, b, x::Number) = x
@inline wall_value(::HorizontalWall, a, b, field::AbstractArray) = @inbounds field[a, b, 1]
@inline wall_value(::XNormalWall,    a, b, field::AbstractArray) = @inbounds field[1, a, b]
@inline wall_value(::YNormalWall,    a, b, field::AbstractArray) = @inbounds field[a, 1, b]

# A two-dimensional field on the wall, for wall states given as functions of the two
# wall coordinates: (x, y) on the bottom and top, (y, z) on the west and east, (x, z) on
# the south and north
wall_field(grid, ::HorizontalWall) = Field{Center, Center, Nothing}(grid)
wall_field(grid, ::XNormalWall)    = Field{Nothing, Center, Center}(grid)
wall_field(grid, ::YNormalWall)    = Field{Center, Nothing, Center}(grid)

# Wall-normal distance from the wall to the near-wall cell centre: the height of the
# first cell centre for the bottom wall, and half the cell width otherwise
@inline wall_distance(::Bottom,       i, j, k, grid) = znode(i, j, k, grid, Center(), Center(), Center())
@inline wall_distance(::Top,          i, j, k, grid) = Δzᶜᶜᶜ(i, j, k, grid) / 2
@inline wall_distance(::XNormalWall,  i, j, k, grid) = Δxᶜᶜᶜ(i, j, k, grid) / 2
@inline wall_distance(::YNormalWall,  i, j, k, grid) = Δyᶜᶜᶜ(i, j, k, grid) / 2

# Height of the wall next to the near-wall cell, for the potential energy in the static energy
@inline wall_height(::Bottom,       i, j, k, grid) = znode(i, j, k,     grid, Center(), Center(), Face())
@inline wall_height(::Top,          i, j, k, grid) = znode(i, j, k + 1, grid, Center(), Center(), Face())
@inline wall_height(::VerticalWall, i, j, k, grid) = znode(i, j, k,     grid, Center(), Center(), Center())

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

@inline near_wall_velocity(::XDirection, i, j, k, fields) = @inbounds fields.u[i, j, k]
@inline near_wall_velocity(::YDirection, i, j, k, fields) = @inbounds fields.v[i, j, k]
@inline near_wall_velocity(::ZDirection, i, j, k, fields) = @inbounds fields.w[i, j, k]

# Horizontal walls: the tangential wind is (u, v)
@inline tangential_speed²(::HorizontalWall, ::Nothing, i, j, k, grid, fields) =
    ℑxᶜᵃᵃ(i, j, k, grid, ϕ², fields.u) + ℑyᵃᶜᵃ(i, j, k, grid, ϕ², fields.v)

@inline tangential_speed²(::HorizontalWall, ::XDirection, i, j, k, grid, fields) =
    ϕ²(i, j, k, grid, fields.u) + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², fields.v)

@inline tangential_speed²(::HorizontalWall, ::YDirection, i, j, k, grid, fields) =
    ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², fields.u) + ϕ²(i, j, k, grid, fields.v)

# Walls normal to x: the tangential wind is (v, w)
@inline tangential_speed²(::XNormalWall, ::Nothing, i, j, k, grid, fields) =
    ℑyᵃᶜᵃ(i, j, k, grid, ϕ², fields.v) + ℑzᵃᵃᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(::XNormalWall, ::YDirection, i, j, k, grid, fields) =
    ϕ²(i, j, k, grid, fields.v) + ℑyzᵃᶠᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(::XNormalWall, ::ZDirection, i, j, k, grid, fields) =
    ℑyzᵃᶜᶠ(i, j, k, grid, ϕ², fields.v) + ϕ²(i, j, k, grid, fields.w)

# Walls normal to y: the tangential wind is (u, w)
@inline tangential_speed²(::YNormalWall, ::Nothing, i, j, k, grid, fields) =
    ℑxᶜᵃᵃ(i, j, k, grid, ϕ², fields.u) + ℑzᵃᵃᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(::YNormalWall, ::XDirection, i, j, k, grid, fields) =
    ϕ²(i, j, k, grid, fields.u) + ℑxzᶠᵃᶜ(i, j, k, grid, ϕ², fields.w)

@inline tangential_speed²(::YNormalWall, ::ZDirection, i, j, k, grid, fields) =
    ℑxzᶜᵃᶠ(i, j, k, grid, ϕ², fields.u) + ϕ²(i, j, k, grid, fields.w)

#####
##### Near-wall wind, with or without the temporally filtered surface state
#####
##### Filtering (`FilteredSurfaceVelocities`, `filtered_surface_state.jl`) is defined for
##### the bottom wall only; materialization rejects it elsewhere.
#####

@inline near_wall_velocity(side, direction, i, j, k, fields, ::Nothing) =
    near_wall_velocity(direction, i, j, k, fields)

@inline wall_wind_speed²(side, direction, i, j, k, grid, fields, ::Nothing) =
    tangential_speed²(side, direction, i, j, k, grid, fields)

@inline near_wall_velocity(::Bottom, ::XDirection, i, j, k, fields, fv::FilteredSurfaceVelocities) = @inbounds fv.u[i, j, 1]
@inline near_wall_velocity(::Bottom, ::YDirection, i, j, k, fields, fv::FilteredSurfaceVelocities) = @inbounds fv.v[i, j, 1]

@inline wall_wind_speed²(::Bottom, ::XDirection, i, j, k, grid, fields, fv::FilteredSurfaceVelocities) = wind_speed²ᶠᶜᶜ(i, j, grid, fields, fv)
@inline wall_wind_speed²(::Bottom, ::YDirection, i, j, k, grid, fields, fv::FilteredSurfaceVelocities) = wind_speed²ᶜᶠᶜ(i, j, grid, fields, fv)
@inline wall_wind_speed²(::Bottom, ::Nothing,    i, j, k, grid, fields, fv::FilteredSurfaceVelocities) = wind_speed²ᶜᶜᶜ(i, j, grid, fields, fv)

validate_wall_filtering(side, ::Nothing) = nothing
validate_wall_filtering(::Bottom, ::FilteredSurfaceVelocities) = nothing
validate_wall_filtering(side, ::FilteredSurfaceVelocities) =
    throw(ArgumentError("Temporally filtered surface state is only supported on the bottom boundary, not on the $(typeof(side)) boundary"))
