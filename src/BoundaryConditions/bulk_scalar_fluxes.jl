#####
##### BulkSensibleHeatFluxFunction for temperature/potential temperature fluxes
#####

struct BulkSensibleHeatFluxFunction{C, G, T, P, TC}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: P
    thermodynamic_constants :: TC
end

"""
    BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature)

Create a bulk sensible heat flux function for computing surface potential temperature fluxes.
The flux is computed as:

```math
Jᶿ = - ρ₀ Cᵀ |U| (θ - θ₀)
```

where `Cᵀ` is the transfer coefficient, `|U|` is the wind speed, `θ` is the atmospheric
potential temperature at the surface, and `θ₀` is the surface temperature.

This boundary condition returns a potential temperature flux (proportional to sensible heat
flux ``𝒬ᵀ = cᵖᵐ Jᶿ``) and should be applied directly to `ρθ` boundary conditions.

# Keyword Arguments

- `coefficient`: The sensible heat transfer coefficient.
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`).
- `surface_temperature`: The surface temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Functions are converted to Fields during model construction.
"""
BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature) =
    BulkSensibleHeatFluxFunction(coefficient, gustiness, surface_temperature, nothing, nothing)

Adapt.adapt_structure(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(Adapt.adapt(to, bf.coefficient),
                                 Adapt.adapt(to, bf.gustiness),
                                 Adapt.adapt(to, bf.surface_temperature),
                                 Adapt.adapt(to, bf.surface_pressure),
                                 Adapt.adapt(to, bf.thermodynamic_constants))

Base.summary(bf::BulkSensibleHeatFluxFunction) =
    string("BulkSensibleHeatFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

# getbc for BulkSensibleHeatFluxFunction: returns potential temperature flux Jᶿ
@inline function OceananigansBC.getbc(bf::BulkSensibleHeatFluxFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    T₀ = surface_value(bf.surface_temperature, i, j)
    θ = @inbounds fields.θ[i, j, 1]
    Δθ = θ - T₀

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    Ũ = sqrt(U² + bf.gustiness^2)

    constants = bf.thermodynamic_constants
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)

    Cᵀ = bf.coefficient
    return - ρ₀ * Cᵀ * Ũ * Δθ
end

const BulkSensibleHeatFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}

#####
##### BulkVaporFluxFunction for moisture fluxes
#####

struct BulkVaporFluxFunction{C, G, T, F, TC, S}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: F
    thermodynamic_constants :: TC
    surface :: S
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
BulkVaporFluxFunction(; coefficient, gustiness=0, surface_temperature) =
    BulkVaporFluxFunction(coefficient, gustiness, surface_temperature, nothing, nothing, nothing)

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_temperature),
                          Adapt.adapt(to, bf.surface_pressure),
                          Adapt.adapt(to, bf.thermodynamic_constants),
                          Adapt.adapt(to, bf.surface))

Base.summary(bf::BulkVaporFluxFunction) =
    string("BulkVaporFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

# getbc for BulkVaporFluxFunction
@inline function OceananigansBC.getbc(bf::BulkVaporFluxFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    constants = bf.thermodynamic_constants
    surface = bf.surface
    T₀ = surface_value(bf.surface_temperature, i, j)
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)
    qᵛ₀ = saturation_specific_humidity(T₀, ρ₀, constants, surface)

    qᵗ = @inbounds fields.qᵗ[i, j, 1]
    Δq = qᵗ - qᵛ₀ # neglecting condensate

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    Ũ = sqrt(U² + bf.gustiness^2)

    Cᵛ = bf.coefficient
    return - ρ₀ * Cᵛ * Ũ * Δq
end

const BulkVaporFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}

#####
##### Convenient constructors
#####

"""
    BulkSensibleHeatFlux(; coefficient, gustiness=0, surface_temperature)

Create a `FluxBoundaryCondition` for surface potential temperature flux.

This boundary condition returns a potential temperature flux `Jᶿ` (proportional to
sensible heat flux) and should be applied directly to `ρθ` boundary conditions.

See [`BulkSensibleHeatFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

T₀(x, y) = 290 + 2 * sign(cos(2π * x / 20e3))

ρθ_bc = BulkSensibleHeatFlux(coefficient = 1e-3,
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

ρqᵗ_bc = BulkVaporFlux(coefficient = 1e-3,
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
