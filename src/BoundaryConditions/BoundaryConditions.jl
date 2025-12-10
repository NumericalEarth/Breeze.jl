module BoundaryConditions

export DragFunction,
       XDirectionDragFunction,
       YDirectionDragFunction,
       Drag,
       DragBoundaryCondition,
       BulkSensibleHeatFluxFunction,
       BulkSensibleHeatFlux,
       BulkSensibleHeatFluxBoundaryCondition,
       BulkVaporFluxFunction,
       BulkVaporFlux,
       BulkVaporFluxBoundaryCondition

using Oceananigans.Grids: Center, Face, XDirection, YDirection
using Oceananigans.BoundaryConditions: FluxBoundaryCondition,
                                       BoundaryCondition,
                                       Flux,
                                       DiscreteBoundaryFunction,
                                       regularize_boundary_condition

using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

using Adapt

import Oceananigans.BoundaryConditions: regularize_boundary_condition
import Oceananigans.Architectures: on_architecture

#####
##### DragFunction for momentum fluxes
#####

struct DragFunction{D, C, G}
    direction :: D
    coefficient :: C
    gustiness :: G
end

"""
    DragFunction(; direction=nothing, coefficient=1e-3, gustiness=0)

Create a drag function for computing surface momentum fluxes using bulk aerodynamic
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
function DragFunction(; direction=nothing, coefficient=1e-3, gustiness=0)
    return DragFunction(direction, coefficient, gustiness)
end

const XDirectionDragFunction = DragFunction{<:XDirection}
const YDirectionDragFunction = DragFunction{<:YDirection}

Adapt.adapt_structure(to, df::DragFunction) =
    DragFunction(Adapt.adapt(to, df.direction),
                 Adapt.adapt(to, df.coefficient),
                 Adapt.adapt(to, df.gustiness))

on_architecture(to, df::DragFunction) =
    DragFunction(on_architecture(to, df.direction),
                 on_architecture(to, df.coefficient),
                 on_architecture(to, df.gustiness))

Base.summary(df::DragFunction) = string("DragFunction(direction=", summary(df.direction),
                                        ", coefficient=", df.coefficient,
                                        ", gustiness=", df.gustiness, ")")

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

@inline function (df::XDirectionDragFunction)(i, j, grid, clock, fields)
    ρu = @inbounds fields.ρu[i, j, 1]
    U² = wind_speed²ᶠᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2
    Cᴰ = df.coefficient
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

@inline function (df::YDirectionDragFunction)(i, j, grid, clock, fields)
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
    BulkSensibleHeatFluxFunction(; coefficient=1e-3, gustiness=0, surface_temperature, reference_density=1.2)

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
- `surface_temperature`: The surface temperature field or value (required)
- `reference_density`: Reference air density for flux calculation (default: `1.2` kg/m³)
"""
function BulkSensibleHeatFluxFunction(; coefficient=1e-3, gustiness=0,
                                        surface_temperature,
                                        reference_density=1.2)
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
    T₀ = @inbounds bf.surface_temperature[i, j, 1]
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

struct BulkVaporFluxFunction{C, G, Q, R}
    coefficient :: C
    gustiness :: G
    surface_specific_humidity :: Q
    reference_density :: R
end

"""
    BulkVaporFluxFunction(; coefficient=1e-3, gustiness=0, surface_specific_humidity, reference_density=1.2)

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
- `surface_specific_humidity`: The surface saturation specific humidity field or value (required)
- `reference_density`: Reference air density for flux calculation (default: `1.2` kg/m³)
"""
function BulkVaporFluxFunction(; coefficient=1e-3, gustiness=0,
                                 surface_specific_humidity,
                                 reference_density=1.2)
    return BulkVaporFluxFunction(coefficient, gustiness,
                                 surface_specific_humidity, reference_density)
end

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_specific_humidity),
                          Adapt.adapt(to, bf.reference_density))

on_architecture(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(on_architecture(to, bf.coefficient),
                          on_architecture(to, bf.gustiness),
                          on_architecture(to, bf.surface_specific_humidity),
                          on_architecture(to, bf.reference_density))

Base.summary(bf::BulkVaporFluxFunction) =
    string("BulkVaporFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

@inline function (bf::BulkVaporFluxFunction)(i, j, grid, clock, fields)
    qᵛ₀ = @inbounds bf.surface_specific_humidity[i, j, 1]
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

# For DiscreteBoundaryFunction wrapping DragFunction without direction, infer from field location
function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:DragFunction{Nothing}},
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
    
    regularized_df = DragFunction(direction, df.coefficient, df.gustiness)
    return DiscreteBoundaryFunction(regularized_df, nothing)
end

# If direction is already set, pass through
function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:XDirectionDragFunction},
                                       grid, loc, dim, args...)
    return dbf
end

function regularize_boundary_condition(dbf::DiscreteBoundaryFunction{Nothing, <:YDirectionDragFunction},
                                       grid, loc, dim, args...)
    return dbf
end

# BulkSensibleHeatFluxFunction and BulkVaporFluxFunction don't need regularization
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
    DragBoundaryCondition

Type alias for a `FluxBoundaryCondition` with a `DragFunction` condition.
"""
const DragBoundaryCondition = BoundaryCondition{<:Flux, <:DiscreteBoundaryFunction{Nothing, <:DragFunction}}

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
    Drag(; direction=nothing, coefficient=1e-3, gustiness=0)

Create a `FluxBoundaryCondition` for surface momentum drag.

See [`DragFunction`](@ref) for details.

# Example

```julia
using Breeze

# Drag for both u and v (direction inferred from field location)
drag = Drag(coefficient=1e-3, gustiness=0.1)

# Or with explicit direction
u_drag = Drag(direction=XDirection(), coefficient=1e-3)
v_drag = Drag(direction=YDirection(), coefficient=1e-3)

ρu_bcs = FieldBoundaryConditions(bottom=u_drag)
ρv_bcs = FieldBoundaryConditions(bottom=v_drag)
```
"""
function Drag(; kwargs...)
    df = DragFunction(; kwargs...)
    condition = DiscreteBoundaryFunction(df, nothing)
    return BoundaryCondition(Flux(), condition)
end

"""
    BulkSensibleHeatFlux(; coefficient=1e-3, gustiness=0, surface_temperature, reference_density=1.2)

Create a `FluxBoundaryCondition` for surface sensible heat flux.

See [`BulkSensibleHeatFluxFunction`](@ref) for details.

# Example

```julia
using Breeze

T₀ = 290  # constant surface temperature (K)
ρ₀ = 1.2  # reference density (kg/m³)

ρθ_bc = BulkSensibleHeatFlux(coefficient = 1e-3,
                              gustiness = 0.1,
                              surface_temperature = T₀,
                              reference_density = ρ₀)

ρθ_bcs = FieldBoundaryConditions(bottom=ρθ_bc)
```
"""
function BulkSensibleHeatFlux(; kwargs...)
    bf = BulkSensibleHeatFluxFunction(; kwargs...)
    condition = DiscreteBoundaryFunction(bf, nothing)
    return BoundaryCondition(Flux(), condition)
end

"""
    BulkVaporFlux(; coefficient=1e-3, gustiness=0, surface_specific_humidity, reference_density=1.2)

Create a `FluxBoundaryCondition` for surface moisture flux.

See [`BulkVaporFluxFunction`](@ref) for details.

# Example

```julia
using Breeze

qᵛ₀ = 0.015  # surface saturation specific humidity (kg/kg)
ρ₀ = 1.2     # reference density (kg/m³)

ρqᵗ_bc = BulkVaporFlux(coefficient = 1e-3,
                        gustiness = 0.1,
                        surface_specific_humidity = qᵛ₀,
                        reference_density = ρ₀)

ρqᵗ_bcs = FieldBoundaryConditions(bottom=ρqᵗ_bc)
```
"""
function BulkVaporFlux(; kwargs...)
    bf = BulkVaporFluxFunction(; kwargs...)
    condition = DiscreteBoundaryFunction(bf, nothing)
    return BoundaryCondition(Flux(), condition)
end

end # module BoundaryConditions

