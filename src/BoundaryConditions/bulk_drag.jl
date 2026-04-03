#####
##### BulkDragFunction for momentum fluxes
#####

struct BulkDragFunction{D, C, G, T, FV}
    direction :: D
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    filtered_velocities :: FV  # Nothing or FilteredSurfaceVelocities
end

"""
    BulkDragFunction(; direction=nothing, coefficient=1e-3, gustiness=0,
                       surface_temperature=nothing, filtered_velocities=nothing)

Create a bulk drag function for computing surface momentum fluxes using bulk aerodynamic
formulas. The drag function computes a quadratic drag:

```math
Jᵘ = - Cᴰ |U| ρu
```

where `Cᴰ` is the drag coefficient, `|U| = √(u² + v² + gustiness²)` is the wind speed
(with gustiness to prevent division by zero), and `ρu` is the momentum density.

# Keyword Arguments

- `direction`: The direction of the momentum component (`XDirection()` or `YDirection()`).
               If `nothing`, the direction is inferred from the field location during
               boundary condition regularization.
- `coefficient`: The drag coefficient (default: `1e-3`). Can be a constant or a
  [`PolynomialCoefficient`](@ref) for wind and stability-dependent transfer coefficients.
- `gustiness`: Minimum wind speed to prevent singularities when winds are calm (default: `0`)
- `surface_temperature`: Surface temperature, required when using `PolynomialCoefficient`
  with stability correction. Can be a `Field`, `Function`, or `Number`. (default: `nothing`)
- `filtered_velocities`: A [`FilteredSurfaceVelocities`](@ref) for temporally filtered
  wind speed in the bulk formula. If `nothing` (default), instantaneous velocity is used.
"""
function BulkDragFunction(; direction=nothing, coefficient=1e-3, gustiness=0,
                            surface_temperature=nothing, filtered_velocities=nothing)
    if coefficient isa PolynomialCoefficient && isnothing(surface_temperature)
        throw(ArgumentError("surface_temperature keyword argument must be provided when configuring BulkDrag with a PolynomialCoefficient"))
    end
    return BulkDragFunction(direction, coefficient, gustiness, surface_temperature, filtered_velocities)
end

const XDirectionBulkDragFunction = BulkDragFunction{<:XDirection}
const YDirectionBulkDragFunction = BulkDragFunction{<:YDirection}

Adapt.adapt_structure(to, df::BulkDragFunction) =
    BulkDragFunction(Adapt.adapt(to, df.direction),
                     Adapt.adapt(to, df.coefficient),
                     Adapt.adapt(to, df.gustiness),
                     Adapt.adapt(to, df.surface_temperature),
                     Adapt.adapt(to, df.filtered_velocities))

function Base.summary(df::BulkDragFunction)
    s = string("BulkDragFunction(direction=", summary(df.direction),
               ", coefficient=", df.coefficient,
               ", gustiness=", df.gustiness)
    if !isnothing(df.filtered_velocities)
        s *= string(", filtered_velocities=", summary(df.filtered_velocities))
    end
    return s * ")"
end

#####
##### getbc for BulkDragFunction
#####

@inline function OceananigansBC.getbc(df::XDirectionBulkDragFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    ρu = @inbounds fields.ρu[i, j, 1]
    T₀ = surface_value(i, j, df.surface_temperature)
    U² = wind_speed²ᶠᶜᶜ(i, j, grid, fields, df.filtered_velocities)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2
    Cᴰ = bulk_coefficient(i, j, grid, df.coefficient, fields, T₀, df.filtered_velocities)
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

@inline function OceananigansBC.getbc(df::YDirectionBulkDragFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    ρv = @inbounds fields.ρv[i, j, 1]
    T₀ = surface_value(i, j, df.surface_temperature)
    U² = wind_speed²ᶜᶠᶜ(i, j, grid, fields, df.filtered_velocities)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2
    Cᴰ = bulk_coefficient(i, j, grid, df.coefficient, fields, T₀, df.filtered_velocities)
    return - Cᴰ * Ũ² * ρv / U * (U > 0)
end

const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:BulkDragFunction}

#####
##### Convenient constructor
#####

"""
    BulkDrag(; direction=nothing, coefficient=1e-3, gustiness=0, surface_temperature=nothing)

Create a `FluxBoundaryCondition` for surface momentum drag.

See [`BulkDragFunction`](@ref) for details.

# Examples

```jldoctest bulkdrag
using Breeze

drag = BulkDrag(coefficient=1e-3, gustiness=0.1)

# output
FluxBoundaryCondition: BulkDragFunction(direction=Nothing, coefficient=0.001, gustiness=0.1)
```

Or with explicit direction, e.g., `XDirection()` for u:

```jldoctest bulkdrag
using Oceananigans.Grids: XDirection

u_drag = BulkDrag(direction=XDirection(), coefficient=1e-3)
ρu_bcs = FieldBoundaryConditions(bottom=u_drag)

# output
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── east: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── south: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── north: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── bottom: FluxBoundaryCondition: BulkDragFunction(direction=XDirection(), coefficient=0.001, gustiness=0)
├── top: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
└── immersed: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
```

and similarly for `YDirection` for v.
"""
function BulkDrag(; kwargs...)
    df = BulkDragFunction(; kwargs...)
    return BoundaryCondition(Flux(), df)
end
