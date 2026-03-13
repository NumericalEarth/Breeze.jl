#####
##### BulkSensibleHeatFluxFunction
#####

struct PotentialTemperatureFlux end
struct StaticEnergyFlux end

struct BulkSensibleHeatFluxFunction{C, G, T, P, TC, F, FV, FS}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: P
    thermodynamic_constants :: TC
    formulation :: F
    filtered_velocities :: FV  # Nothing or FilteredSurfaceVelocities
    filtered_scalar :: FS      # Nothing or FilteredSurfaceScalar
end

"""
    BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature)

A bulk sensible heat flux function. The flux is computed as:

```math
J = - ρ₀ Cᵀ |U| Δϕ
```

where `Cᵀ` is the transfer coefficient, `|U|` is the wind speed, and `Δϕ` is the
difference between the near-surface atmospheric value and the surface value of the
thermodynamic variable appropriate to the formulation:

- For `LiquidIcePotentialTemperatureFormulation`: `Δϕ = θ - θ₀` (potential temperature flux)
- For `StaticEnergyFormulation`: `Δϕ = e - cᵖᵈ T₀` (static energy flux)

The `formulation` is set automatically during model construction based on the
thermodynamic formulation.

# Keyword Arguments

- `coefficient`: The sensible heat transfer coefficient.
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`).
- `surface_temperature`: The surface temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Functions are converted to Fields during model construction.
"""
function BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature, filtered_velocities=nothing)
    return BulkSensibleHeatFluxFunction(coefficient, gustiness, surface_temperature,
                                        nothing, nothing, nothing, filtered_velocities, nothing)
end

Adapt.adapt_structure(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(Adapt.adapt(to, bf.coefficient),
                                 Adapt.adapt(to, bf.gustiness),
                                 Adapt.adapt(to, bf.surface_temperature),
                                 Adapt.adapt(to, bf.surface_pressure),
                                 Adapt.adapt(to, bf.thermodynamic_constants),
                                 bf.formulation,
                                 Adapt.adapt(to, bf.filtered_velocities),
                                 Adapt.adapt(to, bf.filtered_scalar))

Base.summary(bf::BulkSensibleHeatFluxFunction) =
    string("BulkSensibleHeatFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

# Compute the thermodynamic variable difference at the surface.
# Default to potential temperature flux when formulation is not set (ρθ BCs passed directly).
@inline bulk_sensible_heat_difference(i, j, grid, ::Nothing, T₀, constants, fields, fs) =
    bulk_sensible_heat_difference(i, j, grid, PotentialTemperatureFlux(), T₀, constants, fields, fs)

# No filtered scalar: read from 3D fields (current behavior)
@inline function bulk_sensible_heat_difference(i, j, grid, ::PotentialTemperatureFlux, T₀, constants, fields, ::Nothing)
    θ = @inbounds fields.θ[i, j, 1]
    return θ - T₀
end

# With filtered scalar: read from the 2D filtered field
@inline function bulk_sensible_heat_difference(i, j, grid, ::PotentialTemperatureFlux, T₀, constants, fields, fs::FilteredSurfaceScalar)
    θ = @inbounds fs.field[i, j, 1]
    return θ - T₀
end

@inline function bulk_sensible_heat_difference(i, j, grid, ::StaticEnergyFlux, T₀, constants, fields, ::Nothing)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    qᵛ = @inbounds fields.qᵛ[i, j, 1]
    cᵖᵐ = (1 - qᵛ) * cᵖᵈ + qᵛ * cᵖᵛ  # no condensate at the surface
    e₀ = cᵖᵐ * T₀
    e = @inbounds fields.e[i, j, 1]
    return e - e₀
end

@inline function bulk_sensible_heat_difference(i, j, grid, ::StaticEnergyFlux, T₀, constants, fields, fs::FilteredSurfaceScalar)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    qᵛ = @inbounds fields.qᵛ[i, j, 1]
    cᵖᵐ = (1 - qᵛ) * cᵖᵈ + qᵛ * cᵖᵛ  # no condensate at the surface
    e₀ = cᵖᵐ * T₀
    e = @inbounds fs.field[i, j, 1]
    return e - e₀
end

@inline function OceananigansBC.getbc(bf::BulkSensibleHeatFluxFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    T₀ = surface_value(i, j, bf.surface_temperature)

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields, bf.filtered_velocities)
    Ũ = sqrt(U² + bf.gustiness^2)

    constants = bf.thermodynamic_constants
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)

    Cᵀ = bulk_coefficient(i, j, grid, bf.coefficient, fields, T₀, bf.filtered_velocities)

    Δϕ = bulk_sensible_heat_difference(i, j, grid, bf.formulation, T₀, constants, fields, bf.filtered_scalar)
    return - ρ₀ * Cᵀ * Ũ * Δϕ
end

const BulkSensibleHeatFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}

#####
##### BulkVaporFluxFunction for moisture fluxes
#####

struct BulkVaporFluxFunction{C, G, T, F, TC, S, FV, FS}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: F
    thermodynamic_constants :: TC
    surface :: S
    filtered_velocities :: FV  # Nothing or FilteredSurfaceVelocities
    filtered_scalar :: FS      # Nothing or FilteredSurfaceScalar
end

"""
    BulkVaporFluxFunction(; coefficient, gustiness=0, surface_temperature)

Create a bulk vapor flux function for computing surface moisture fluxes.
The flux is computed as:

```math
Jᵛ = - ρ₀ Cᵛ |U| (qᵗ - qᵛ₀)
```

where `Cᵛ` is the transfer coefficient, `|U|` is the wind speed, `qᵗ` is the atmospheric
specific humidity, and `qᵛ₀` is the saturation specific humidity at the surface.

# Keyword Arguments

- `coefficient`: The vapor transfer coefficient.
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`).
- `surface_temperature`: The surface temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Used to compute saturation specific humidity at the surface.
"""
function BulkVaporFluxFunction(; coefficient, gustiness=0, surface_temperature, filtered_velocities=nothing)
    return BulkVaporFluxFunction(coefficient, gustiness, surface_temperature,
                                  nothing, nothing, nothing, filtered_velocities, nothing)
end

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_temperature),
                          Adapt.adapt(to, bf.surface_pressure),
                          Adapt.adapt(to, bf.thermodynamic_constants),
                          Adapt.adapt(to, bf.surface),
                          Adapt.adapt(to, bf.filtered_velocities),
                          Adapt.adapt(to, bf.filtered_scalar))

Base.summary(bf::BulkVaporFluxFunction) =
    string("BulkVaporFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

# getbc for BulkVaporFluxFunction
@inline function OceananigansBC.getbc(bf::BulkVaporFluxFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    constants = bf.thermodynamic_constants
    surface = bf.surface
    T₀ = surface_value(i, j, bf.surface_temperature)
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)
    qᵛ₀ = saturation_specific_humidity(T₀, ρ₀, constants, surface)

    Δq = bulk_vapor_difference(i, j, fields, bf.filtered_scalar, qᵛ₀)

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields, bf.filtered_velocities)
    Ũ = sqrt(U² + bf.gustiness^2)

    Cᵛ = bulk_coefficient(i, j, grid, bf.coefficient, fields, T₀, bf.filtered_velocities)

    return - ρ₀ * Cᵛ * Ũ * Δq
end

# Vapor difference dispatch on filtered_scalar
@inline function bulk_vapor_difference(i, j, fields, ::Nothing, qᵛ₀)
    qᵛ = @inbounds fields.qᵛ[i, j, 1]
    return qᵛ - qᵛ₀
end

@inline function bulk_vapor_difference(i, j, fields, fs::FilteredSurfaceScalar, qᵛ₀)
    qᵛ = @inbounds fs.field[i, j, 1]
    return qᵛ - qᵛ₀
end

const BulkVaporFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}

#####
##### Convenient constructors
#####

"""
    BulkSensibleHeatFlux(; coefficient, gustiness=0, surface_temperature)

Create a `FluxBoundaryCondition` for surface sensible heat flux.

The bulk formula computes `J = -ρ₀ Cᵀ |U| Δϕ`, where `Δϕ` depends on the thermodynamic
formulation: `Δθ` for potential temperature or `Δe` for static energy. The formulation
is set automatically during model construction.

See [`BulkSensibleHeatFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

T₀(x, y) = 290 + 2 * sign(cos(2π * x / 20e3))

ρe_bc = BulkSensibleHeatFlux(coefficient = 1e-3,
                             gustiness = 0.1,
                             surface_temperature = T₀)

# output
FluxBoundaryCondition: BulkSensibleHeatFluxFunction(coefficient=0.001, gustiness=0.1)
```
"""
function BulkSensibleHeatFlux(; kwargs...)
    bf = BulkSensibleHeatFluxFunction(; kwargs...)
    return BoundaryCondition(Flux(), bf)
end

"""
    BulkVaporFlux(; coefficient, surface_temperature, gustiness=0)

Create a `FluxBoundaryCondition` for surface moisture flux.

The saturation specific humidity at the surface is automatically computed from
`surface_temperature`.

See [`BulkVaporFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

T₀(x, y) = 290 + 2 * sign(cos(2π * x / 20e3))

moisture_bc = BulkVaporFlux(coefficient = 1e-3,
                            gustiness = 0.1,
                            surface_temperature = T₀)

# output
FluxBoundaryCondition: BulkVaporFluxFunction(coefficient=0.001, gustiness=0.1)
```
"""
function BulkVaporFlux(; kwargs...)
    bf = BulkVaporFluxFunction(; kwargs...)
    return BoundaryCondition(Flux(), bf)
end
