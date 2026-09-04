# Wall fluxes

The bulk-flux boundary conditions [`BulkDrag`](@ref Breeze.BoundaryConditions.BulkDrag),
[`BulkSensibleHeatFlux`](@ref Breeze.BoundaryConditions.BulkSensibleHeatFlux), and
[`BulkVaporFlux`](@ref Breeze.BoundaryConditions.BulkVaporFlux) compute momentum, heat, and
vapor fluxes through a wall from the wind tangential to it and the difference between the
near-wall air and the wall state. They may be placed on any of the six boundaries of a
bounded domain, so a closed box such as a laboratory cloud chamber, with a warm wet floor,
a cold wet ceiling, and side walls at their own temperature and humidity, is described by
the same conditions as a lower boundary:

```jldoctest walls
using Breeze
using Breeze.BoundaryConditions: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux

C = 6e-3
T_floor, T_ceiling, T_wall = 299, 280, 285

drag(T) = BulkDrag(coefficient=C, surface_temperature=T)
heat(T) = BulkSensibleHeatFlux(coefficient=C, surface_temperature=T)
vapor(T, ℋ) = BulkVaporFlux(coefficient=C, surface_temperature=T, surface_relative_humidity=ℋ)

# Drag acts on the two momentum components tangential to each wall
ρu_bcs = FieldBoundaryConditions(bottom=drag(T_floor), top=drag(T_ceiling), south=drag(T_wall), north=drag(T_wall))
ρθ_bcs = FieldBoundaryConditions(bottom=heat(T_floor), top=heat(T_ceiling),
                                 west=heat(T_wall), east=heat(T_wall), south=heat(T_wall), north=heat(T_wall))
ρqᵛ_bcs = FieldBoundaryConditions(bottom=vapor(T_floor, 1), top=vapor(T_ceiling, 1),
                                  west=vapor(T_wall, 0.78), east=vapor(T_wall, 0.78),
                                  south=vapor(T_wall, 0.78), north=vapor(T_wall, 0.78))
ρqᵛ_bcs.top

# output
FluxBoundaryCondition: BulkVaporFluxFunction(coefficient=0.006, gustiness=0)
```

On every wall the fluxes carry heat and vapor into the domain when the wall is warmer or
moister than the adjacent air, and remove the tangential momentum. The wall state may be a
number, a field, or a function of the two coordinates of the wall (`(x, y)` on the floor and
ceiling, `(y, z)` on the west and east walls, `(x, z)` on the south and north walls). The
transfer coefficient may be a constant or a [`PolynomialCoefficient`](@ref Breeze.BoundaryConditions.PolynomialCoefficient);
its stability correction applies on the floor and ceiling (with the sign of the bulk
Richardson number reversed under the ceiling) and is neutral on the vertical walls, along
which buoyancy acts tangentially. The temporally filtered surface state of
[`FilteredSurfaceVelocities`](@ref Breeze.BoundaryConditions.FilteredSurfaceVelocities) is
supported on the bottom boundary only.

```@docs
Breeze.BoundaryConditions.BulkDrag
Breeze.BoundaryConditions.BulkSensibleHeatFlux
Breeze.BoundaryConditions.BulkVaporFlux
```
