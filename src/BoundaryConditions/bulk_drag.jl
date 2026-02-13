#####
##### BulkDragFunction for momentum fluxes
#####

struct BulkDragFunction{D, C, G}
    direction :: D
    coefficient :: C
    gustiness :: G
end

"""
    BulkDragFunction(; direction=nothing, coefficient=1e-3, gustiness=0)

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
- `coefficient`: The drag coefficient (default: `1e-3`).
- `gustiness`: Minimum wind speed to prevent singularities when winds are calm (default: `0`)
"""
function BulkDragFunction(; direction=nothing, coefficient=1e-3, gustiness=0)
    return BulkDragFunction(direction, coefficient, gustiness)
end

const XDirectionBulkDragFunction = BulkDragFunction{<:XDirection}
const YDirectionBulkDragFunction = BulkDragFunction{<:YDirection}

Adapt.adapt_structure(to, df::BulkDragFunction) =
    BulkDragFunction(Adapt.adapt(to, df.direction),
                     Adapt.adapt(to, df.coefficient),
                     Adapt.adapt(to, df.gustiness))

Base.summary(df::BulkDragFunction) = string("BulkDragFunction(direction=", summary(df.direction),
                                            ", coefficient=", df.coefficient,
                                            ", gustiness=", df.gustiness, ")")

#####
##### Coefficient evaluation (constant vs callable)
#####

# Fallback for constant coefficients
@inline evaluate_drag_coefficient(C::Number, args...) = C

# For callable coefficients (e.g., PolynomialBulkCoefficient)
# Note: For drag, we don't have access to thermodynamic constants or surface temperature
# in the BulkDragFunction, so stability correction is limited
@inline function evaluate_drag_coefficient(C, i, j, grid, fields)
    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)

    # If coefficient doesn't have stability function, just pass wind speed
    if isnothing(C.stability_function)
        return C(U)
    end

    # For drag, we compute stability using atmospheric θ at lowest level
    # compared to a typical surface value. This is approximate since we
    # don't have access to actual surface temperature in BulkDragFunction.
    θᵥ = virtual_potential_temperature(i, j, 1, fields)

    # Use atmospheric θᵥ as proxy for surface (neutral assumption)
    # This effectively disables stability correction for drag
    # To enable it, surface temperature would need to be added to BulkDragFunction
    θᵥ₀ = θᵥ

    return C(U, θᵥ, θᵥ₀, nothing)
end

#####
##### getbc for BulkDragFunction
#####

@inline function OceananigansBC.getbc(df::XDirectionBulkDragFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    ρu = @inbounds fields.ρu[i, j, 1]
    U² = wind_speed²ᶠᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2

    # Evaluate coefficient (handles both constant and callable)
    Cᴰ = evaluate_drag_coefficient(df.coefficient, i, j, grid, fields)

    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

@inline function OceananigansBC.getbc(df::YDirectionBulkDragFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    ρv = @inbounds fields.ρv[i, j, 1]
    U² = wind_speed²ᶜᶠᶜ(i, j, grid, fields)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2

    # Evaluate coefficient (handles both constant and callable)
    Cᴰ = evaluate_drag_coefficient(df.coefficient, i, j, grid, fields)

    return - Cᴰ * Ũ² * ρv / U * (U > 0)
end

const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:BulkDragFunction}

#####
##### Convenient constructor
#####

"""
    BulkDrag(; direction=nothing, coefficient=1e-3, gustiness=0)

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
