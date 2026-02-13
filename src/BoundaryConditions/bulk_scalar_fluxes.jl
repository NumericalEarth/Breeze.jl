#####
##### BulkSensibleHeatFluxFunction
#####

struct PotentialTemperatureFlux end
struct StaticEnergyFlux end

struct BulkSensibleHeatFluxFunction{C, G, T, P, TC, F, θᵛ}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: P
    thermodynamic_constants :: TC
    formulation :: F
    virtual_potential_temperature :: θᵛ
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
BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature) =
    BulkSensibleHeatFluxFunction(coefficient, gustiness, surface_temperature, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(Adapt.adapt(to, bf.coefficient),
                                 Adapt.adapt(to, bf.gustiness),
                                 Adapt.adapt(to, bf.surface_temperature),
                                 Adapt.adapt(to, bf.surface_pressure),
                                 Adapt.adapt(to, bf.thermodynamic_constants),
                                 bf.formulation,
                                 Adapt.adapt(to, bf.virtual_potential_temperature))

Base.summary(bf::BulkSensibleHeatFluxFunction) =
    string("BulkSensibleHeatFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

# Compute the thermodynamic variable difference at the surface.
# Default to potential temperature flux when formulation is not set (ρθ BCs passed directly).
@inline bulk_sensible_heat_difference(::Nothing, i, j, T₀, constants, fields) =
    bulk_sensible_heat_difference(PotentialTemperatureFlux(), i, j, T₀, constants, fields)

@inline function bulk_sensible_heat_difference(::PotentialTemperatureFlux, i, j, T₀, constants, fields)
    θ = @inbounds fields.θ[i, j, 1]
    return θ - T₀
end

@inline function bulk_sensible_heat_difference(::StaticEnergyFlux, i, j, T₀, constants, fields)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    qᵗ = @inbounds fields.qᵗ[i, j, 1]
    cᵖᵐ = (1 - qᵗ) * cᵖᵈ + qᵗ * cᵖᵛ  # no condensate at the surface
    e₀ = cᵖᵐ * T₀
    e = @inbounds fields.e[i, j, 1]
    return e - e₀
end

#####
##### Coefficient evaluation for scalar fluxes
#####

# Fallback for constant coefficients
@inline evaluate_scalar_coefficient(C::Number, args...) = C

# For callable coefficients (e.g., PolynomialCoefficient)
@inline function evaluate_scalar_coefficient(C, i, j, grid, fields, θᵥ_op, T₀, p₀, constants, surface)
    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)

    # Virtual potential temperature at first grid level from diagnostic KernelFunctionOperation
    θᵥ = θᵥ_op[i, j, 1]
    θᵥ₀ = surface_virtual_potential_temperature(T₀, p₀, constants, surface)

    # Get measurement height from grid
    z = znode(i, j, 1, grid, Center(), Center(), Center())

    return C(U, θᵥ, θᵥ₀, z, constants)
end

@inline function OceananigansBC.getbc(bf::BulkSensibleHeatFluxFunction, i::Integer, j::Integer,
                                      grid::AbstractGrid, clock, fields)
    T₀ = surface_value(i, j, bf.surface_temperature)

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    Ũ = sqrt(U² + bf.gustiness^2)

    constants = bf.thermodynamic_constants
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)
    surface = PlanarLiquidSurface()

    # Evaluate coefficient (handles both constant and callable)
    θᵥ = bf.virtual_potential_temperature
    Cᵀ = evaluate_scalar_coefficient(bf.coefficient, i, j, grid, fields, θᵥ, T₀, p₀, constants, surface)

    Δϕ = bulk_sensible_heat_difference(bf.formulation, i, j, T₀, constants, fields)
    return - ρ₀ * Cᵀ * Ũ * Δϕ
end

const BulkSensibleHeatFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}

#####
##### BulkVaporFluxFunction for moisture fluxes
#####

struct BulkVaporFluxFunction{C, G, T, F, TC, S, θᵛ}
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: F
    thermodynamic_constants :: TC
    surface :: S
    virtual_potential_temperature :: θᵛ
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
    BulkVaporFluxFunction(coefficient, gustiness, surface_temperature, nothing, nothing, nothing, nothing)

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_temperature),
                          Adapt.adapt(to, bf.surface_pressure),
                          Adapt.adapt(to, bf.thermodynamic_constants),
                          Adapt.adapt(to, bf.surface),
                          Adapt.adapt(to, bf.virtual_potential_temperature))

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

    qᵗ = @inbounds fields.qᵗ[i, j, 1]
    Δq = qᵗ - qᵛ₀ # neglecting condensate

    U² = wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    Ũ = sqrt(U² + bf.gustiness^2)

    # Evaluate coefficient (handles both constant and callable)
    θᵥ = bf.virtual_potential_temperature
    Cᵛ = evaluate_scalar_coefficient(bf.coefficient, i, j, grid, fields, θᵥ, T₀, p₀, constants, surface)

    return - ρ₀ * Cᵛ * Ũ * Δq
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
