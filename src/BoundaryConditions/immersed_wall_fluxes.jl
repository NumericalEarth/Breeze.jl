#####
##### Log-law wall fluxes on the facets of a grid-fitted immersed boundary
#####
##### A facet of a GridFittedBoundary lies on a cell face, so the first fluid momentum or scalar
##### point sits a distance d = Δn/2 from it, where Δn is the grid spacing normal to the facet.
##### The neutral log law gives the wall stress from the wall-parallel speed Ũ at that point,
#####
#####     τ = - Cᴰ Ũ ρu,    Cᴰ = [κ / ln(d / z₀)]²,
#####
##### and the bulk-transfer wall heat flux
#####
#####     J = ρ Cᴴ Ũ (θʷ - θ),    Cᴴ = κ² / [ln(d / z₀) ln(d / z₀ʰ)].
#####
##### Oceananigans' immersed-flux machinery treats a facet flux as the flux *into* the fluid on
##### every facet, so neither expression carries a facet-dependent sign.
#####

struct ImmersedWallDrag{FT}
    roughness_length :: FT
    von_karman_constant :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Neutral log-law drag on every facet of a grid-fitted immersed boundary, for use as the `immersed`
boundary condition of `ρu`, `ρv` and `ρw`. Each facet acts on the momentum components tangential to it.

```jldoctest
julia> using Breeze

julia> ImmersedWallDrag(roughness_length=0.1)
ImmersedWallDrag(z₀=0.1)
```
"""
ImmersedWallDrag(; roughness_length, von_karman_constant=0.4) =
    ImmersedWallDrag(promote(roughness_length, von_karman_constant)...)

struct ImmersedWallHeatFlux{FT}
    wall_temperature :: FT
    roughness_length :: FT
    thermal_roughness_length :: FT
    von_karman_constant :: FT
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Neutral bulk-transfer flux of `ρθ` from immersed facets held at `wall_temperature`,
for use as the `immersed` boundary condition of `ρθ`.

```jldoctest
julia> using Breeze

julia> ImmersedWallHeatFlux(wall_temperature=310, roughness_length=0.1)
ImmersedWallHeatFlux(θʷ=310.0, z₀=0.1)
```
"""
ImmersedWallHeatFlux(; wall_temperature, roughness_length, thermal_roughness_length=roughness_length/10,
                       von_karman_constant=0.4) =
    ImmersedWallHeatFlux(promote(wall_temperature, roughness_length, thermal_roughness_length, von_karman_constant)...)

Base.summary(d::ImmersedWallDrag) = string("ImmersedWallDrag(z₀=", d.roughness_length, ")")
Base.summary(h::ImmersedWallHeatFlux) = string("ImmersedWallHeatFlux(θʷ=", h.wall_temperature, ", z₀=", h.roughness_length, ")")
Base.show(io::IO, f::Union{ImmersedWallDrag, ImmersedWallHeatFlux}) = print(io, summary(f))

# Facet flux conditions on a facet normal to `side`'s direction; `direction` is that of the momentum component.
struct FacetLogLawDrag{S, D, FT}
    side :: S
    direction :: D
    roughness_length :: FT
    von_karman_constant :: FT
end

struct FacetLogLawHeatFlux{S, FT}
    side :: S
    wall_temperature :: FT
    roughness_length :: FT
    thermal_roughness_length :: FT
    von_karman_constant :: FT
end

Base.summary(f::FacetLogLawDrag) = string("FacetLogLawDrag(", nameof(typeof(f.side)), ", ", nameof(typeof(f.direction)), ")")
Base.summary(f::FacetLogLawHeatFlux) = string("FacetLogLawHeatFlux(", nameof(typeof(f.side)), ")")

@inline momentum_density(i, j, k, ::XDirection, fields) = @inbounds fields.ρu[i, j, k]
@inline momentum_density(i, j, k, ::YDirection, fields) = @inbounds fields.ρv[i, j, k]
@inline momentum_density(i, j, k, ::ZDirection, fields) = @inbounds fields.ρw[i, j, k]

@inline function OceananigansBC.getbc(f::FacetLogLawDrag, i::Integer, j::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields, args...)
    d = wall_distance(i, j, k, grid, f.side)
    Cᴰ = (f.von_karman_constant / log(d / f.roughness_length))^2
    Ũ = sqrt(tangential_speed²(i, j, k, grid, f.side, f.direction, fields))
    return - Cᴰ * Ũ * momentum_density(i, j, k, f.direction, fields)
end

@inline function OceananigansBC.getbc(f::FacetLogLawHeatFlux, i::Integer, j::Integer, k::Integer,
                                      grid::AbstractGrid, clock, fields, args...)
    d = wall_distance(i, j, k, grid, f.side)
    κ = f.von_karman_constant
    Cᴴ = κ^2 / (log(d / f.roughness_length) * log(d / f.thermal_roughness_length))
    Ũ = sqrt(tangential_speed²(i, j, k, grid, f.side, nothing, fields))
    ρ = @inbounds fields.ρ[i, j, k]
    θ = @inbounds fields.θ[i, j, k]
    return ρ * Cᴴ * Ũ * (f.wall_temperature - θ)
end

momentum_direction(::Face, ::Center, ::Center) = XDirection()
momentum_direction(::Center, ::Face, ::Center) = YDirection()
momentum_direction(::Center, ::Center, ::Face) = ZDirection()

# A facet normal to the component's own direction carries no tangential stress.
facet_drag(side, direction, z₀, κ) = wall_normal_direction(side) == direction ? nothing :
    FluxBoundaryCondition(FacetLogLawDrag(side, direction, z₀, κ))

function materialize_atmosphere_boundary_condition(drag::ImmersedWallDrag, ::Nothing, loc, args...)
    direction = momentum_direction(loc...)
    z₀, κ = drag.roughness_length, drag.von_karman_constant
    x, y, z = Tuple(facet_drag(side, direction, z₀, κ) for side in (West(), South(), Bottom()))
    return ImmersedBoundaryCondition(west=x, east=x, south=y, north=y, bottom=z, top=z)
end

function materialize_atmosphere_boundary_condition(h::ImmersedWallHeatFlux, ::Nothing, loc, args...)
    facet(side) = FluxBoundaryCondition(FacetLogLawHeatFlux(side, h.wall_temperature, h.roughness_length,
                                                            h.thermal_roughness_length, h.von_karman_constant))
    x, y, z = facet(West()), facet(South()), facet(Bottom())
    return ImmersedBoundaryCondition(west=x, east=x, south=y, north=y, bottom=z, top=z)
end

# The first regularization pass in `AtmosphereModel` runs before materialization; keep the skeleton for it.
OceananigansBC.regularize_immersed_boundary_condition(skeleton::Union{ImmersedWallDrag, ImmersedWallHeatFlux}, grid, args...) = skeleton
