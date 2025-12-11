module BoundaryConditions

export BulkDragFunction,
       XDirectionBulkDragFunction,
       YDirectionBulkDragFunction,
       BulkDrag,
       BulkDragBoundaryCondition,
       BulkSensibleHeatFluxFunction,
       BulkSensibleHeatFlux,
       BulkSensibleHeatFluxBoundaryCondition,
       BulkVaporFluxFunction,
       BulkVaporFlux,
       BulkVaporFluxBoundaryCondition

import ..AtmosphereModels: regularize_atmosphere_model_boundary_conditions

using ..Thermodynamics: saturation_specific_humidity, surface_density, PlanarLiquidSurface

using Oceananigans.Grids: Center, Face, XDirection, YDirection
using Oceananigans.Fields: Field, set!
using Oceananigans.BoundaryConditions: BoundaryCondition,
                                       Flux,
                                       DiscreteBoundaryFunction,
                                       FieldBoundaryConditions,
                                       regularize_boundary_condition

using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

using Adapt: Adapt

import Oceananigans.BoundaryConditions: regularize_boundary_condition
import Oceananigans.Architectures: on_architecture

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
- `coefficient`: The drag coefficient (default: `1e-3`)
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

on_architecture(to, df::BulkDragFunction) =
    BulkDragFunction(on_architecture(to, df.direction),
                     on_architecture(to, df.coefficient),
                     on_architecture(to, df.gustiness))

Base.summary(df::BulkDragFunction) = string("BulkDragFunction(direction=", summary(df.direction),
                                            ", coefficient=", df.coefficient,
                                            ", gustiness=", df.gustiness, ")")

#####
##### Helper function for surface values
#####

# Get surface value from a Field or a Number
@inline surface_value(field::Field, i, j) = @inbounds field[i, j, 1]
@inline surface_value(x::Number, i, j) = x

#####
##### Wind speed calculations at staggered locations
#####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

# Wind speed squared at (Face, Center, Center) - for x-momentum flux
@inline function wind_speed²ᶠᶜᶜ(i, j, grid, fields)
    u² = @inbounds fields.u[i, j, 1]^2
    v² = ℑxyᶠᶜᵃ(i, j, 1, grid, ϕ², fields.v)
    return u² + v²
end

# Wind speed squared at (Center, Face, Center) - for y-momentum flux
@inline function wind_speed²ᶜᶠᶜ(i, j, grid, fields)
    u² = ℑxyᶜᶠᵃ(i, j, 1, grid, ϕ², fields.u)
    v² = @inbounds fields.v[i, j, 1]^2
    return u² + v²
end

# Wind speed squared at (Center, Center, Center) - for scalar fluxes
@inline function wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    u² = ℑxᶜᵃᵃ(i, j, 1, grid, ϕ², fields.u)
    v² = ℑyᵃᶜᵃ(i, j, 1, grid, ϕ², fields.v)
    return u² + v²
end

#####
##### Drag flux computation
#####

@inline function (df::XDirectionBulkDragFunction)(i, j, grid, clock, fields)
    ρu = @inbounds fields.ρu[i, j, 1]
    U² = wind_speed²ᶠᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2
    Cᴰ = df.coefficient
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

@inline function (df::YDirectionBulkDragFunction)(i, j, grid, clock, fields)
    ρv = @inbounds fields.ρv[i, j, 1]
    U² = wind_speed²ᶜᶠᶜ(i, j, grid, fields)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2
    Cᴰ = df.coefficient
    return - Cᴰ * Ũ² * ρv / U * (U > 0)
end

#####
##### BulkSensibleHeatFluxFunction for temperature/potential temperature fluxes
#####

struct BulkSensibleHeatFluxFunction{C, G, T, R}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    reference_density :: R
end

"""
    BulkSensibleHeatFluxFunction(; coefficient=1e-3, gustiness=0, surface_temperature, reference_density=nothing)

Create a bulk sensible heat flux function for computing surface heat fluxes.
The flux is computed as:

```math
Jᵀ = - ρ₀ Cᵀ |U| (θ - θ₀)
```

where `Cᵀ` is the transfer coefficient, `|U|` is the wind speed, `θ` is the atmospheric
potential temperature at the surface, and `θ₀` is the surface temperature.

# Keyword Arguments

- `coefficient`: The sensible heat transfer coefficient (default: `1e-3`)
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`)
- `surface_temperature`: The surface temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Functions are converted to Fields during model construction.
- `reference_density`: Reference air density for flux calculation. If `nothing` (default),
                       the density is automatically computed from the model's reference state.
"""
function BulkSensibleHeatFluxFunction(; coefficient=1e-3, gustiness=0,
                                        surface_temperature,
                                        reference_density=nothing)
    return BulkSensibleHeatFluxFunction(coefficient, gustiness,
                                        surface_temperature, reference_density)
end

Adapt.adapt_structure(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(Adapt.adapt(to, bf.coefficient),
                                 Adapt.adapt(to, bf.gustiness),
                                 Adapt.adapt(to, bf.surface_temperature),
                                 Adapt.adapt(to, bf.reference_density))

on_architecture(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(on_architecture(to, bf.coefficient),
                                 on_architecture(to, bf.gustiness),
                                 on_architecture(to, bf.surface_temperature),
                                 on_architecture(to, bf.reference_density))

Base.summary(bf::BulkSensibleHeatFluxFunction) =
    string("BulkSensibleHeatFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

@inline function (bf::BulkSensibleHeatFluxFunction)(i, j, grid, clock, fields)
    T₀ = surface_value(bf.surface_temperature, i, j)
    θ = @inbounds fields.θ[i, j, 1]
    Δθ = θ - T₀

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    Ũ = sqrt(U² + bf.gustiness^2)

    Cᵀ = bf.coefficient
    ρ₀ = bf.reference_density
    return - ρ₀ * Cᵀ * Ũ * Δθ
end

#####
##### BulkVaporFluxFunction for moisture fluxes
#####

struct BulkVaporFluxFunction{C, G, T, Q, R}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_specific_humidity :: Q
    reference_density :: R
end

"""
    BulkVaporFluxFunction(; coefficient=1e-3, gustiness=0, surface_temperature, surface_specific_humidity=nothing, reference_density=nothing)

Create a bulk vapor flux function for computing surface moisture fluxes.
The flux is computed as:

```math
Jᵛ = - ρ₀ Cᵛ |U| (qᵗ - qᵛ₀)
```

where `Cᵛ` is the transfer coefficient, `|U|` is the wind speed, `qᵗ` is the atmospheric
specific humidity, and `qᵛ₀` is the saturation specific humidity at the surface.

# Keyword Arguments

- `coefficient`: The vapor transfer coefficient (default: `1e-3`)
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`)
- `surface_temperature`: The surface temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Used to compute `surface_specific_humidity` if not provided.
- `surface_specific_humidity`: The surface saturation specific humidity. If `nothing` (default),
                               it is computed from `surface_temperature` using the model's
                               thermodynamic constants.
- `reference_density`: Reference air density for flux calculation. If `nothing` (default),
                       the density is automatically computed from the model's reference state.
"""
function BulkVaporFluxFunction(; coefficient=1e-3, gustiness=0,
                                 surface_temperature,
                                 surface_specific_humidity=nothing,
                                 reference_density=nothing)
    return BulkVaporFluxFunction(coefficient, gustiness,
                                 surface_temperature,
                                 surface_specific_humidity,
                                 reference_density)
end

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_temperature),
                          Adapt.adapt(to, bf.surface_specific_humidity),
                          Adapt.adapt(to, bf.reference_density))

on_architecture(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(on_architecture(to, bf.coefficient),
                          on_architecture(to, bf.gustiness),
                          on_architecture(to, bf.surface_temperature),
                          on_architecture(to, bf.surface_specific_humidity),
                          on_architecture(to, bf.reference_density))

Base.summary(bf::BulkVaporFluxFunction) =
    string("BulkVaporFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

@inline function (bf::BulkVaporFluxFunction)(i, j, grid, clock, fields)
    qᵛ₀ = surface_value(bf.surface_specific_humidity, i, j)
    qᵗ = @inbounds fields.qᵗ[i, j, 1]
    Δq = qᵗ - qᵛ₀

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    Ũ = sqrt(U² + bf.gustiness^2)

    Cᵛ = bf.coefficient
    ρ₀ = bf.reference_density
    return - ρ₀ * Cᵛ * Ũ * Δq
end

#####
##### Regularization: assign direction based on field location
#####

# For DiscreteBoundaryFunction wrapping BulkDragFunction without direction, infer from field location
function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:BulkDragFunction{Nothing}},
                                       grid, loc, dim, args...)
    df = dbf.func
    LX, LY, LZ = loc

    # Determine direction from location: Face in x means x-momentum, Face in y means y-momentum
    if LX isa Face
        direction = XDirection()
    elseif LY isa Face
        direction = YDirection()
    else
        throw(ArgumentError("Cannot infer drag direction from field location $loc. " *
                            "Please specify direction=XDirection() or direction=YDirection()."))
    end

    regularized_df = BulkDragFunction(direction, df.coefficient, df.gustiness)
    return DiscreteBoundaryFunction(regularized_df, nothing)
end

# If direction is already set, pass through
function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:XDirectionBulkDragFunction},
                                       grid, loc, dim, args...)
    return dbf
end

function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:YDirectionBulkDragFunction},
                                       grid, loc, dim, args...)
    return dbf
end

# BulkSensibleHeatFluxFunction and BulkVaporFluxFunction don't need Oceananigans regularization
function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:BulkSensibleHeatFluxFunction},
                                       grid, loc, dim, args...)
    return dbf
end

function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:BulkVaporFluxFunction},
                                       grid, loc, dim, args...)
    return dbf
end

#####
##### Type aliases for FluxBoundaryCondition with these functions
#####

"""
    BulkDragBoundaryCondition

Type alias for a `FluxBoundaryCondition` with a `BulkDragFunction` condition.
"""
const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:DiscreteBoundaryFunction{Nothing, <:BulkDragFunction}}

"""
    BulkSensibleHeatFluxBoundaryCondition

Type alias for a `FluxBoundaryCondition` with a `BulkSensibleHeatFluxFunction` condition.
"""
const BulkSensibleHeatFluxBoundaryCondition = BoundaryCondition{<:Flux, <:DiscreteBoundaryFunction{Nothing, <:BulkSensibleHeatFluxFunction}}

"""
    BulkVaporFluxBoundaryCondition

Type alias for a `FluxBoundaryCondition` with a `BulkVaporFluxFunction` condition.
"""
const BulkVaporFluxBoundaryCondition = BoundaryCondition{<:Flux, <:DiscreteBoundaryFunction{Nothing, <:BulkVaporFluxFunction}}

#####
##### Convenient constructors
#####

"""
    BulkDrag(; direction=nothing, coefficient=1e-3, gustiness=0)

Create a `FluxBoundaryCondition` for surface momentum drag.

See [`BulkDragFunction`](@ref) for details.

# Examples

Drag for both u and v (direction inferred from field location)

```jldoctest bulkdrag
using Breeze

drag = BulkDrag(coefficient=1e-3, gustiness=0.1)

# output
FluxBoundaryCondition: DiscreteBoundaryFunction with BulkDragFunction(direction=Nothing, coefficient=0.001, gustiness=0.1)
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
├── bottom: FluxBoundaryCondition: DiscreteBoundaryFunction with BulkDragFunction(direction=XDirection(), coefficient=0.001, gustiness=0)
├── top: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
└── immersed: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
```

and similartly for `YDirection` for v.
"""
function BulkDrag(; kwargs...)
    df = BulkDragFunction(; kwargs...)
    condition = DiscreteBoundaryFunction(df, nothing)
    return BoundaryCondition(Flux(), condition)
end

"""
    BulkSensibleHeatFlux(; coefficient=1e-3, gustiness=0, surface_temperature, reference_density=nothing)

Create a `FluxBoundaryCondition` for surface sensible heat flux.

See [`BulkSensibleHeatFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

# Surface temperature can be a function, Field, or number
T₀(x, y) = 290 + 2 * sign(cos(2π * x / 20e3))

ρθ_bc = BulkSensibleHeatFlux(coefficient = 1e-3,
                              gustiness = 0.1,
                              surface_temperature = T₀)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_bc)

# output
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── east: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── south: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── north: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── bottom: FluxBoundaryCondition: DiscreteBoundaryFunction with BulkSensibleHeatFluxFunction(coefficient=0.001, gustiness=0.1)
├── top: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
└── immersed: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
```
"""
function BulkSensibleHeatFlux(; kwargs...)
    bf = BulkSensibleHeatFluxFunction(; kwargs...)
    condition = DiscreteBoundaryFunction(bf, nothing)
    return BoundaryCondition(Flux(), condition)
end

"""
    BulkVaporFlux(; coefficient=1e-3, gustiness=0, surface_temperature, surface_specific_humidity=nothing, reference_density=nothing)

Create a `FluxBoundaryCondition` for surface moisture flux.

The saturation specific humidity at the surface is automatically computed from
`surface_temperature` if `surface_specific_humidity` is not provided.

See [`BulkVaporFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

# Surface temperature can be a function, Field, or number
# Saturation specific humidity is computed automatically
T₀(x, y) = 290 + 2 * sign(cos(2π * x / 20e3))

ρqᵗ_bc = BulkVaporFlux(coefficient = 1e-3,
                        gustiness = 0.1,
                        surface_temperature = T₀)

ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_bc)

# output
Oceananigans.FieldBoundaryConditions, with boundary conditions
├── west: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── east: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── south: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── north: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
├── bottom: FluxBoundaryCondition: DiscreteBoundaryFunction with BulkVaporFluxFunction(coefficient=0.001, gustiness=0.1)
├── top: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
└── immersed: DefaultBoundaryCondition (FluxBoundaryCondition: Nothing)
```
"""
function BulkVaporFlux(; kwargs...)
    bf = BulkVaporFluxFunction(; kwargs...)
    condition = DiscreteBoundaryFunction(bf, nothing)
    return BoundaryCondition(Flux(), condition)
end

#####
##### Pre-regularization for AtmosphereModel
#####

"""
    regularize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation, thermodynamic_constants)

Pre-regularize boundary conditions for `AtmosphereModel` by:
1. Converting function-based surface temperatures to Fields
2. Computing reference density from the formulation's reference state
3. Computing saturation specific humidity from surface temperature for vapor fluxes

This function should be called in `AtmosphereModel` before the standard Oceananigans
boundary condition regularization.
"""
function regularize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation, thermodynamic_constants)
    # Extract reference state info and compute base density
    reference_state = formulation.reference_state
    p₀ = reference_state.surface_pressure
    θ₀ = reference_state.potential_temperature
    ρ₀ = surface_density(p₀, θ₀, thermodynamic_constants)
    constants = thermodynamic_constants

    # Regularize each field's boundary conditions
    regularized = Dict{Symbol, Any}()
    for (name, fbcs) in pairs(boundary_conditions)
        regularized[name] = regularize_field_bulk_bcs(fbcs, grid, ρ₀, constants)
    end

    return NamedTuple(regularized)
end

# Pass through non-FieldBoundaryConditions
regularize_field_bulk_bcs(fbcs, grid, ρ₀, constants) = fbcs

# Regularize FieldBoundaryConditions
function regularize_field_bulk_bcs(fbcs::FieldBoundaryConditions, grid, ρ₀, constants)
    bottom = regularize_bulk_bc(fbcs.bottom, grid, ρ₀, constants)
    top = regularize_bulk_bc(fbcs.top, grid, ρ₀, constants)
    # Keep other boundaries as-is
    return FieldBoundaryConditions(west = fbcs.west,
                                   east = fbcs.east,
                                   south = fbcs.south,
                                   north = fbcs.north,
                                   bottom = bottom,
                                   top = top,
                                   immersed = fbcs.immersed)
end

# Pass through non-bulk-flux boundary conditions
regularize_bulk_bc(bc, grid, ρ₀, constants) = bc

# Regularize BulkSensibleHeatFlux
function regularize_bulk_bc(bc::BulkSensibleHeatFluxBoundaryCondition, grid, ρ₀, constants)
    bf = bc.condition.func

    # Convert surface_temperature to Field if it's a function
    T₀ = materialize_surface_field(bf.surface_temperature, grid)

    # Use provided reference_density or compute from formulation
    ref_density = isnothing(bf.reference_density) ? ρ₀ : bf.reference_density

    new_bf = BulkSensibleHeatFluxFunction(bf.coefficient, bf.gustiness, T₀, ref_density)
    return BoundaryCondition(Flux(), DiscreteBoundaryFunction(new_bf, nothing))
end

# Regularize BulkVaporFlux
function regularize_bulk_bc(bc::BulkVaporFluxBoundaryCondition, grid, ρ₀, constants)
    bf = bc.condition.func

    # Convert surface_temperature to Field if it's a function
    T₀ = materialize_surface_field(bf.surface_temperature, grid)

    # Use provided reference_density or compute from formulation
    ref_density = isnothing(bf.reference_density) ? ρ₀ : bf.reference_density

    # Compute saturation specific humidity if not provided
    if isnothing(bf.surface_specific_humidity)
        qᵛ₀ = compute_saturation_specific_humidity(T₀, ref_density, constants, grid)
    else
        qᵛ₀ = materialize_surface_field(bf.surface_specific_humidity, grid)
    end

    new_bf = BulkVaporFluxFunction(bf.coefficient, bf.gustiness, T₀, qᵛ₀, ref_density)
    return BoundaryCondition(Flux(), DiscreteBoundaryFunction(new_bf, nothing))
end

# Helper to convert functions/numbers to Fields
materialize_surface_field(f::Field, grid) = f
materialize_surface_field(f::Number, grid) = f

function materialize_surface_field(f::Function, grid)
    field = Field{Center, Center, Nothing}(grid)
    set!(field, f)
    return field
end

function compute_saturation_specific_humidity(T₀::Field, ρ₀, constants, grid)
    surface = PlanarLiquidSurface()

    qᵛ₀ = Field{Center, Center, Nothing}(grid)
    view(qᵛ₀, :, :, 1) .= saturation_specific_humidity.(view(T₀, :, :, 1), ρ₀, constants, surface)

    return qᵛ₀
end

function compute_saturation_specific_humidity(T₀::Number, ρ₀, constants, grid)
    surface = PlanarLiquidSurface()
    return saturation_specific_humidity(T₀, ρ₀, constants, surface)
end

end # module BoundaryConditions
