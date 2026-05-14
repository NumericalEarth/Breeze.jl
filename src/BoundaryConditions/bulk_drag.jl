#####
##### BulkDragFunction for momentum fluxes
#####

struct BulkDragFunction{D, C, G, T, FV, P, TC}
    direction :: D
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    filtered_velocities :: FV  # Nothing or FilteredSurfaceVelocities
    surface_pressure :: P      # Set during materialization (nothing pre-materialize)
    thermodynamic_constants :: TC
end

"""
    BulkDragFunction(; direction=nothing, coefficient=1e-3, gustiness=0,
                       surface_temperature=nothing, filtered_velocities=nothing)

Create a bulk drag function for computing surface momentum fluxes using bulk aerodynamic
formulas. The momentum flux is computed in the same form as the scalar bulk fluxes,

```math
Jᵘ = - ρ₀ Cᴰ |U| u
```

where `Cᴰ` is the drag coefficient, `|U| = √(u² + v² + gustiness²)` is the wind speed
(with gustiness to prevent singularities at low wind), `u` is the velocity component
at the first cell face, and `ρ₀` is the surface density computed from the surface
pressure and surface temperature. Monin–Obukhov similarity is a profile law for `u`
(not `ρu`), so using `u` here keeps the formulation consistent with the similarity
theory underlying `Cᴰ`.

When a [`FilteredSurfaceVelocities`](@ref) is supplied via `filtered_velocities`,
*every* field entering the formula — the wind speed `|U|`, the velocity `u`, and the
virtual potential temperature `θᵥ` used in stability — is read from the filtered
state. The surface density `ρ₀` is computed from the (slowly varying) surface
temperature and pressure and is not filtered.

# Monin–Obukhov consistency

`ρ₀` is computed from surface quantities (`surface_pressure` and `surface_temperature`)
via the ideal gas law, so it is a *true surface* density — independent of the
vertical grid resolution. Using the prognostic density at the first cell would
introduce a grid-dependent ρ₀ (the first-cell height ½Δz shifts the value as the
grid is refined), which is inconsistent with the bulk-flux closure derived from
Monin–Obukhov similarity.

# Default surface temperature

If the user does not supply `surface_temperature`, materialization calls
`default_drag_surface_temperature(dynamics, …)`. The default exists for
`AnelasticDynamics` (recovered from the reference state via Exner) but raises
for `CompressibleDynamics`, which has no equivalent reference profile — pass
`surface_temperature` explicitly in that case.

# Keyword Arguments

- `direction`: The direction of the momentum component (`XDirection()` or `YDirection()`).
               If `nothing`, the direction is inferred from the field location during
               boundary condition regularization.
- `coefficient`: The drag coefficient (default: `1e-3`). Can be a constant or a
  [`PolynomialCoefficient`](@ref) for wind and stability-dependent transfer coefficients.
- `gustiness`: Minimum wind speed to prevent singularities when winds are calm (default: `0`)
- `surface_temperature`: Surface temperature, used to compute `ρ₀` and required when
  using `PolynomialCoefficient` with stability correction. Can be a `Field`,
  `Function`, or `Number`. (default: `nothing`)
- `filtered_velocities`: A [`FilteredSurfaceVelocities`](@ref) for temporally filtered
  wind speed, near-surface velocity, and `θᵥ` in the bulk formula. If `nothing`
  (default), instantaneous fields are used.
"""
function BulkDragFunction(; direction=nothing, coefficient=1e-3, gustiness=0,
                            surface_temperature=nothing, filtered_velocities=nothing)
    if coefficient isa PolynomialCoefficient && isnothing(surface_temperature)
        throw(ArgumentError("surface_temperature keyword argument must be provided when configuring BulkDrag with a PolynomialCoefficient"))
    end
    return BulkDragFunction(direction, coefficient, gustiness, surface_temperature,
                            filtered_velocities, nothing, nothing)
end

const XDirectionBulkDragFunction = BulkDragFunction{<:XDirection}
const YDirectionBulkDragFunction = BulkDragFunction{<:YDirection}

Adapt.adapt_structure(to, df::BulkDragFunction) =
    BulkDragFunction(Adapt.adapt(to, df.direction),
                     Adapt.adapt(to, df.coefficient),
                     Adapt.adapt(to, df.gustiness),
                     Adapt.adapt(to, df.surface_temperature),
                     Adapt.adapt(to, df.filtered_velocities),
                     Adapt.adapt(to, df.surface_pressure),
                     Adapt.adapt(to, df.thermodynamic_constants))

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
##### Jᵘ = -ρ₀ Cᴰ Ũ u, mirroring the scalar bulk flux form. `u` is read from the
##### filtered field at the appropriate face location when filtering is enabled.
#####

@inline function OceananigansBC.getbc(df::XDirectionBulkDragFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    T₀ = surface_value(i, j, df.surface_temperature)
    u  = near_surface_velocity(i, j, fields, df.filtered_velocities, XDirection())
    U² = wind_speed²ᶠᶜᶜ(i, j, grid, fields, df.filtered_velocities)
    Ũ  = sqrt(U² + df.gustiness^2)
    ρ₀ = surface_density(df.surface_pressure, T₀, df.thermodynamic_constants)
    Cᴰ = bulk_coefficient(i, j, grid, df.coefficient, fields, T₀, df.filtered_velocities)
    return - ρ₀ * Cᴰ * Ũ * u
end

@inline function OceananigansBC.getbc(df::YDirectionBulkDragFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    T₀ = surface_value(i, j, df.surface_temperature)
    v  = near_surface_velocity(i, j, fields, df.filtered_velocities, YDirection())
    U² = wind_speed²ᶜᶠᶜ(i, j, grid, fields, df.filtered_velocities)
    Ũ  = sqrt(U² + df.gustiness^2)
    ρ₀ = surface_density(df.surface_pressure, T₀, df.thermodynamic_constants)
    Cᴰ = bulk_coefficient(i, j, grid, df.coefficient, fields, T₀, df.filtered_velocities)
    return - ρ₀ * Cᴰ * Ũ * v
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
