#####
##### Wall faces
#####
##### The bulk fluxes may be placed on any of the six boundaries of a bounded domain,
##### or on the six facets of the fluid cells next to an immersed boundary.
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

# Location of a field in the direction normal to a wall
facet_normal_location(::XNormalWall, loc) = loc[1]
facet_normal_location(::YNormalWall, loc) = loc[2]
facet_normal_location(::HorizontalWall, loc) = loc[3]

# Fluxes point along the positive coordinate direction, so a positive flux enters the
# domain through a left wall and leaves it through a right wall. A flux of magnitude F
# directed out of the domain is therefore -F on a left wall and +F on a right wall.
# An immersed facet flux is instead positive into the fluid on every facet.
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

# A function is evaluated at the center of the wall face with the non-`Flat` coordinates of the
# wall followed by the time, as Oceananigans evaluates boundary-condition and forcing functions:
# `f(x, y, t)` on the bottom and top, `f(y, z, t)` on the west and east, `f(x, z, t)` on the
# south and north, with the coordinate of a `Flat` direction dropped (`node` does that)
@inline wall_value(ℓ, m, grid, ::HorizontalWall, f::Function, clock) = f(node(ℓ, m, 1, grid, Center(), Center(), nothing)..., clock.time)
@inline wall_value(ℓ, m, grid, ::XNormalWall,    f::Function, clock) = f(node(1, ℓ, m, grid, nothing, Center(), Center())..., clock.time)
@inline wall_value(ℓ, m, grid, ::YNormalWall,    f::Function, clock) = f(node(ℓ, 1, m, grid, Center(), nothing, Center())..., clock.time)

# On an immersed facet the wall state is located by the near-wall cell (i, j, k). As for
# Oceananigans' `ContinuousBoundaryFunction` on immersed facets, a function takes all three
# coordinates of the facet center followed by the time, and a field lives at cell centers.
@inline wall_value((ℓ, m)::NTuple{2}, grid, side, x, clock) = wall_value(ℓ, m, grid, side, x, clock)
@inline wall_value((i, j, k)::NTuple{3}, grid, side, x::Number, clock) = x
@inline wall_value((i, j, k)::NTuple{3}, grid, side, field::AbstractArray, clock) = @inbounds field[i, j, k]
@inline wall_value((i, j, k)::NTuple{3}, grid, side, f::Function, clock) = f(facet_node(i, j, k, grid, side)..., clock.time)

@inline facet_node(i, j, k, grid, ::West)   = node(i,     j, k, grid, Face(), Center(), Center())
@inline facet_node(i, j, k, grid, ::East)   = node(i + 1, j, k, grid, Face(), Center(), Center())
@inline facet_node(i, j, k, grid, ::South)  = node(i, j,     k, grid, Center(), Face(), Center())
@inline facet_node(i, j, k, grid, ::North)  = node(i, j + 1, k, grid, Center(), Face(), Center())
@inline facet_node(i, j, k, grid, ::Bottom) = node(i, j, k,     grid, Center(), Center(), Face())
@inline facet_node(i, j, k, grid, ::Top)    = node(i, j, k + 1, grid, Center(), Center(), Face())

# Wall-normal distance from the wall to the near-wall cell center
@inline wall_distance(i, j, k, grid, ::HorizontalWall) = Δzᶜᶜᶜ(i, j, k, grid) / 2
@inline wall_distance(i, j, k, grid, ::XNormalWall)    = Δxᶜᶜᶜ(i, j, k, grid) / 2
@inline wall_distance(i, j, k, grid, ::YNormalWall)    = Δyᶜᶜᶜ(i, j, k, grid) / 2

# Height of the wall next to the near-wall cell, for the potential energy in the static energy
@inline wall_height(i, j, k, grid, ::Bottom)       = znode(i, j, k,     grid, Center(), Center(), Face())
@inline wall_height(i, j, k, grid, ::Top)          = znode(i, j, k + 1, grid, Center(), Center(), Face())
@inline wall_height(i, j, k, grid, ::VerticalWall) = znode(i, j, k,     grid, Center(), Center(), Center())

# Buoyancy stabilizes or destabilizes a surface layer on a horizontal wall only, and the
# bulk Richardson number changes sign under the top wall (cold above warm is unstable)
@inline stability_sign(::Bottom) = 1
@inline stability_sign(::Top) = -1

#####
##### Air pressure at the wall
#####
##### The wall pressure is diagnosed from the live thermodynamic pressure and density in the
##### near-wall cell rather than stored, so that it follows both terrain and the evolving state.
##### On a horizontal wall it is extrapolated half a cell along the local hydrostatic scale
##### height; on a vertical wall the wall and the cell center sit at the same height, so the
##### cell-center pressure is used as it is.
#####

@inline wall_air_pressureᶜᶜᶜ(i, j, k, grid, ::Bottom, fields, constants) =
    surface_pressure_from_cell_center(i, j, k, grid, fields.p, fields.ρ,
                                      constants.gravitational_acceleration) # extrapolate downward

@inline wall_air_pressureᶜᶜᶜ(i, j, k, grid, ::Top, fields, constants) =
    surface_pressure_from_cell_center(i, j, k, grid, fields.p, fields.ρ,
                                      -constants.gravitational_acceleration) # extrapolate upward

@inline wall_air_pressureᶜᶜᶜ(i, j, k, grid, ::VerticalWall, fields, constants) = @inbounds fields.p[i, j, k]

# A scalar flux is evaluated at the cell center; a momentum flux at the face of the component it
# acts on, so the pressure that sets ρˢ there is interpolated to that face.
@inline wall_air_pressure(i, j, k, grid, side, ::Nothing, fields, constants) =
    wall_air_pressureᶜᶜᶜ(i, j, k, grid, side, fields, constants)

@inline wall_air_pressure(i, j, k, grid, side, ::XDirection, fields, constants) =
    ℑxᶠᵃᵃ(i, j, k, grid, wall_air_pressureᶜᶜᶜ, side, fields, constants)

@inline wall_air_pressure(i, j, k, grid, side, ::YDirection, fields, constants) =
    ℑyᵃᶠᵃ(i, j, k, grid, wall_air_pressureᶜᶜᶜ, side, fields, constants)

@inline wall_air_pressure(i, j, k, grid, side, ::ZDirection, fields, constants) =
    ℑzᵃᵃᶠ(i, j, k, grid, wall_air_pressureᶜᶜᶜ, side, fields, constants)

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

# Squared velocity averaged over the active neighboring faces only: faces on or inside an
# immersed boundary (or on a domain wall) hold no tangential wind and are left out.
const uᶠᶜᶜ = (Face(), Center(), Center())
const vᶜᶠᶜ = (Center(), Face(), Center())
const wᶜᶜᶠ = (Center(), Center(), Face())

@inline active_ϕ²(i, j, k, grid, ϕ, loc) = ifelse(peripheral_node(i, j, k, grid, loc...), zero(grid), ϕ²(i, j, k, grid, ϕ))
@inline active_face(i, j, k, grid, loc) = !peripheral_node(i, j, k, grid, loc...)

@inline function active_mean_ϕ²(ℑ, i, j, k, grid, ϕ, loc)
    n = ℑ(i, j, k, grid, active_face, loc)
    return ifelse(n > 0, ℑ(i, j, k, grid, active_ϕ², ϕ, loc) / n, zero(grid))
end

# Horizontal walls: the tangential wind is (u, v)
@inline tangential_speed²(i, j, k, grid, ::HorizontalWall, ::Nothing, fields) =
    active_mean_ϕ²(ℑxᶜᵃᵃ, i, j, k, grid, fields.u, uᶠᶜᶜ) + active_mean_ϕ²(ℑyᵃᶜᵃ, i, j, k, grid, fields.v, vᶜᶠᶜ)

@inline tangential_speed²(i, j, k, grid, ::HorizontalWall, ::XDirection, fields) =
    ϕ²(i, j, k, grid, fields.u) + active_mean_ϕ²(ℑxyᶠᶜᵃ, i, j, k, grid, fields.v, vᶜᶠᶜ)

@inline tangential_speed²(i, j, k, grid, ::HorizontalWall, ::YDirection, fields) =
    active_mean_ϕ²(ℑxyᶜᶠᵃ, i, j, k, grid, fields.u, uᶠᶜᶜ) + ϕ²(i, j, k, grid, fields.v)

# Walls normal to x: the tangential wind is (v, w)
@inline tangential_speed²(i, j, k, grid, ::XNormalWall, ::Nothing, fields) =
    active_mean_ϕ²(ℑyᵃᶜᵃ, i, j, k, grid, fields.v, vᶜᶠᶜ) + active_mean_ϕ²(ℑzᵃᵃᶜ, i, j, k, grid, fields.w, wᶜᶜᶠ)

@inline tangential_speed²(i, j, k, grid, ::XNormalWall, ::YDirection, fields) =
    ϕ²(i, j, k, grid, fields.v) + active_mean_ϕ²(ℑyzᵃᶠᶜ, i, j, k, grid, fields.w, wᶜᶜᶠ)

@inline tangential_speed²(i, j, k, grid, ::XNormalWall, ::ZDirection, fields) =
    active_mean_ϕ²(ℑyzᵃᶜᶠ, i, j, k, grid, fields.v, vᶜᶠᶜ) + ϕ²(i, j, k, grid, fields.w)

# Walls normal to y: the tangential wind is (u, w)
@inline tangential_speed²(i, j, k, grid, ::YNormalWall, ::Nothing, fields) =
    active_mean_ϕ²(ℑxᶜᵃᵃ, i, j, k, grid, fields.u, uᶠᶜᶜ) + active_mean_ϕ²(ℑzᵃᵃᶜ, i, j, k, grid, fields.w, wᶜᶜᶠ)

@inline tangential_speed²(i, j, k, grid, ::YNormalWall, ::XDirection, fields) =
    ϕ²(i, j, k, grid, fields.u) + active_mean_ϕ²(ℑxzᶠᵃᶜ, i, j, k, grid, fields.w, wᶜᶜᶠ)

@inline tangential_speed²(i, j, k, grid, ::YNormalWall, ::ZDirection, fields) =
    active_mean_ϕ²(ℑxzᶜᵃᶠ, i, j, k, grid, fields.u, uᶠᶜᶜ) + ϕ²(i, j, k, grid, fields.w)

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
