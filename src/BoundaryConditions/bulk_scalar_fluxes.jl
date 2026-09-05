#####
##### BulkSensibleHeatFluxFunction
#####

struct PotentialTemperatureFlux end
struct StaticEnergyFlux end

struct BulkSensibleHeatFluxFunction{S, C, G, T, P, SP, TC, F, FV, FS}
    side :: S                  # Set during materialization (nothing pre-materialize)
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_pressure :: P
    standard_pressure :: SP
    thermodynamic_constants :: TC
    formulation :: F
    filtered_velocities :: FV  # Nothing or FilteredSurfaceVelocities
    filtered_scalar :: FS      # Nothing or FilteredSurfaceScalar
end

"""
$(TYPEDSIGNATURES)

A bulk sensible heat flux function. The flux is computed as:

```math
J = - ρ₀ Cᵀ |U| Δϕ
```

where ``Cᵀ`` is the transfer coefficient, ``|U|`` is the wind speed tangential to the
wall, and ``Δϕ`` is the difference between the near-wall atmospheric value and the wall
value of the thermodynamic variable appropriate to the formulation:

- For `LiquidIcePotentialTemperatureFormulation`: ``Δϕ = θ - θ₀``, where
  ``θ₀ = T₀ / Π₀`` and ``Π₀ = (p₀ / pˢᵗ)^{Rᵈ / cᵖᵈ}`` (potential temperature flux)
- For `StaticEnergyFormulation`: ``Δϕ = s - (cᵖᵐ T₀ + g z₀)`` (static energy flux),
  with ``z₀`` the height of the wall

Here ``p₀`` is the actual surface pressure, while ``pˢᵗ`` is the fixed reference pressure
used to define potential temperature.

The flux may be placed on any of the six boundaries of a bounded domain. The sign above
is for the bottom; on every wall the flux carries heat *into* the domain when the wall is
warmer than the adjacent air.

The `formulation` is set automatically during model construction based on the
thermodynamic formulation.

# Keyword Arguments

- `coefficient`: The sensible heat transfer coefficient.
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`).
- `surface_temperature`: The wall temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Functions are evaluated at the wall at every time step with the
                         non-`Flat` coordinates of the wall followed by the time, as for
                         Oceananigans boundary conditions: `(x, y, t)` on the bottom and top,
                         `(y, z, t)` on the west and east, `(x, z, t)` on the south and north,
                         and for example `(x, t)` on the bottom of a grid that is `Flat` in `y`.
- `filtered_velocities`: Either `nothing` (default) or [`FilteredSurfaceVelocities`](@ref). Note
                         that when `filtered_velocities` is not `nothing`, then automatically
                         there is filtering in the scalar fields via [`FilteredSurfaceScalar`](@ref)
                         with the same parameters (e.g., `height`, `timescale`) as `filtered_velocities`.
                         Filtering is supported on the bottom boundary only.
"""
function BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature, filtered_velocities=nothing)
    return BulkSensibleHeatFluxFunction(nothing, coefficient, gustiness, surface_temperature,
                                        nothing, nothing, nothing, nothing, filtered_velocities, nothing)
end

Adapt.adapt_structure(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(Adapt.adapt(to, bf.side),
                                 Adapt.adapt(to, bf.coefficient),
                                 Adapt.adapt(to, bf.gustiness),
                                 Adapt.adapt(to, bf.surface_temperature),
                                 Adapt.adapt(to, bf.surface_pressure),
                                 Adapt.adapt(to, bf.standard_pressure),
                                 Adapt.adapt(to, bf.thermodynamic_constants),
                                 bf.formulation,
                                 Adapt.adapt(to, bf.filtered_velocities),
                                 Adapt.adapt(to, bf.filtered_scalar))

Base.summary(bf::BulkSensibleHeatFluxFunction) =
    string("BulkSensibleHeatFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

# Compute the thermodynamic variable difference at the wall.
# Default to potential temperature flux when formulation is not set (ρθ BCs passed directly).
@inline bulk_sensible_heat_difference(i, j, k, grid, side, ::Nothing, bf, T₀, fields, fs) =
    bulk_sensible_heat_difference(i, j, k, grid, side, PotentialTemperatureFlux(), bf, T₀, fields, fs)

@inline function wall_potential_temperature(bf, T₀)
    p₀ = bf.surface_pressure
    pˢᵗ = bf.standard_pressure
    constants = bf.thermodynamic_constants
    return potential_temperature_from_temperature(T₀, p₀, pˢᵗ, constants)
end

# No filtered scalar: read from the near-wall cell of the 3D field
@inline function bulk_sensible_heat_difference(i, j, k, grid, side, ::PotentialTemperatureFlux, bf, T₀, fields, ::Nothing)
    θ = @inbounds fields.θ[i, j, k]
    return θ - wall_potential_temperature(bf, T₀)
end

# With filtered scalar: read from the 2D filtered field (bottom only)
@inline function bulk_sensible_heat_difference(i, j, k, grid, side, ::PotentialTemperatureFlux, bf, T₀, fields, fs::FilteredSurfaceScalar)
    θ = @inbounds fs.field[i, j, 1]
    return θ - wall_potential_temperature(bf, T₀)
end

# Static energy of saturated air in contact with the wall, s₀ = cᵖᵐ T₀ + g z₀, with the
# near-wall vapor fraction in the heat capacity and no condensate at the wall
@inline function wall_static_energy(i, j, k, grid, side, bf, T₀, fields)
    constants = bf.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    g = constants.gravitational_acceleration
    qᵛ = @inbounds fields.qᵛ[i, j, k]
    cᵖᵐ = (1 - qᵛ) * cᵖᵈ + qᵛ * cᵖᵛ
    z₀ = wall_height(i, j, k, grid, side)
    return cᵖᵐ * T₀ + g * z₀
end

@inline function bulk_sensible_heat_difference(i, j, k, grid, side, ::StaticEnergyFlux, bf, T₀, fields, ::Nothing)
    s = @inbounds fields.s[i, j, k]
    return s - wall_static_energy(i, j, k, grid, side, bf, T₀, fields)
end

@inline function bulk_sensible_heat_difference(i, j, k, grid, side, ::StaticEnergyFlux, bf, T₀, fields, fs::FilteredSurfaceScalar)
    s = @inbounds fs.field[i, j, 1]
    return s - wall_static_energy(i, j, k, grid, side, bf, T₀, fields)
end

@inline function OceananigansBC.getbc(bf::BulkSensibleHeatFluxFunction, ℓ::Integer, m::Integer,
                                      grid::AbstractGrid, clock, fields)
    side = bf.side
    i, j, k = near_wall_indices(ℓ, m, grid, side)
    T₀ = wall_value(ℓ, m, grid, side, bf.surface_temperature, clock)

    U² = wall_wind_speed²(i, j, k, grid, side, nothing, fields, bf.filtered_velocities)
    Ũ = sqrt(U² + bf.gustiness^2)

    constants = bf.thermodynamic_constants
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)

    Cᵀ = bulk_coefficient(i, j, k, grid, side, bf.coefficient, fields, T₀, bf.filtered_velocities)

    Δϕ = bulk_sensible_heat_difference(i, j, k, grid, side, bf.formulation, bf, T₀, fields, bf.filtered_scalar)
    return outward_flux_sign(side) * ρ₀ * Cᵀ * Ũ * Δϕ
end

const BulkSensibleHeatFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}

#####
##### BulkVaporFluxFunction for moisture fluxes
#####

struct BulkVaporFluxFunction{S, C, G, T, H, F, TC, SF, M, FV, FS}
    side :: S                  # Set during materialization (nothing pre-materialize)
    coefficient :: C
    gustiness :: G
    surface_temperature :: T
    surface_relative_humidity :: H
    surface_pressure :: F
    thermodynamic_constants :: TC
    surface :: SF
    moisture_availability :: M # the fraction β of the wall that is wet; resolved at materialization
    filtered_velocities :: FV  # Nothing or FilteredSurfaceVelocities
    filtered_scalar :: FS      # Nothing or FilteredSurfaceScalar
end

"""
    BulkVaporFluxFunction(; coefficient, gustiness=0, surface_temperature,
                            surface_relative_humidity=1, moisture_availability=nothing,
                            filtered_velocities=nothing)

Create a bulk vapor flux function for computing wall moisture fluxes.
The flux is computed as:

```math
Jᵛ = - ρ₀ Cᵛ |U| (qᵛ - q₀), \\qquad q₀ = β ℋ₀ qᵛ⁺(T₀) + (1 - β) qᵛ,
```

where ``Cᵛ`` is the transfer coefficient, ``|U|`` is the wind speed tangential to the wall,
``qᵛ`` is the near-wall specific humidity, and ``q₀`` is the specific humidity of the air in
contact with the wall. Over the wet fraction ``β`` of the wall (the `moisture_availability`)
that is the saturation specific humidity ``qᵛ⁺`` at the wall temperature ``T₀`` times the
wall relative humidity ``ℋ₀`` (unity for a wet wall); over the dry fraction it is the
humidity of the air itself, so that ``qᵛ - q₀ = β (qᵛ - ℋ₀ qᵛ⁺)`` and the flux is ``β``
times the flux over a wet wall.

The flux may be placed on any of the six boundaries of a bounded domain. The sign above is
for the bottom; on every wall the flux carries vapor *into* the domain when the wall is
moister than the adjacent air.

# Keyword Arguments

- `coefficient`: The vapor transfer coefficient.
- `gustiness`: Minimum wind speed to prevent singularities (default: `0`).
- `surface_temperature`: The wall temperature. Can be a `Field`, a `Function`, or a `Number`.
                         Used to compute the saturation specific humidity at the wall.
                         Functions take the non-`Flat` coordinates of the wall followed by
                         the time, as for Oceananigans boundary conditions.
- `surface_relative_humidity`: The relative humidity of the air in contact with the wall,
                               between 0 and 1 (default: `1`, a saturated wall). Can be a
                               `Field`, a `Function`, or a `Number`.
- `moisture_availability`: The fraction ``β ∈ [0, 1]`` of the wall that is wet. `nothing`
                           (default) takes the value carried by a [`PolynomialCoefficient`](@ref)
                           `coefficient`, whose stability correction uses the same surface humidity,
                           and 1 (a wet wall, an ocean) for a constant coefficient. A value
                           that disagrees with a `PolynomialCoefficient` is an error. The phase of
                           the surface water follows the coefficient in the same way, and is liquid
                           for a constant coefficient.
- `filtered_velocities`: Either `nothing` (default) or [`FilteredSurfaceVelocities`](@ref). Note
                         that when `filtered_velocities` is not `nothing`, then automatically
                         there is filtering in the scalar fields via [`FilteredSurfaceScalar`](@ref)
                         with the same parameters (e.g., `height`, `timescale`) as `filtered_velocities`.
                         Filtering is supported on the bottom boundary only.
"""
function BulkVaporFluxFunction(; coefficient, gustiness=0, surface_temperature,
                                 surface_relative_humidity=1, moisture_availability=nothing,
                                 filtered_velocities=nothing)
    isnothing(moisture_availability) || 0 ≤ moisture_availability ≤ 1 ||
        throw(ArgumentError("moisture_availability must lie between 0 and 1, got $moisture_availability"))
    return BulkVaporFluxFunction(nothing, coefficient, gustiness, surface_temperature, surface_relative_humidity,
                                 nothing, nothing, nothing, moisture_availability, filtered_velocities, nothing)
end

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.side),
                          Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_temperature),
                          Adapt.adapt(to, bf.surface_relative_humidity),
                          Adapt.adapt(to, bf.surface_pressure),
                          Adapt.adapt(to, bf.thermodynamic_constants),
                          Adapt.adapt(to, bf.surface),
                          Adapt.adapt(to, bf.moisture_availability),
                          Adapt.adapt(to, bf.filtered_velocities),
                          Adapt.adapt(to, bf.filtered_scalar))

function Base.summary(bf::BulkVaporFluxFunction)
    summary_str = string("BulkVaporFluxFunction(coefficient=", prettysummary(bf.coefficient),
                         ", gustiness=", prettysummary(bf.gustiness), ")")
    if bf.filtered_velocities != nothing || bf.filtered_scalar != nothing
        summary_str *= ", with filtering"
    end
    return summary_str
end

# getbc for BulkVaporFluxFunction
@inline function OceananigansBC.getbc(bf::BulkVaporFluxFunction, ℓ::Integer, m::Integer,
                                      grid::AbstractGrid, clock, fields)
    side = bf.side
    i, j, k = near_wall_indices(ℓ, m, grid, side)
    constants = bf.thermodynamic_constants
    surface = bf.surface
    T₀ = wall_value(ℓ, m, grid, side, bf.surface_temperature, clock)
    ℋ₀ = wall_value(ℓ, m, grid, side, bf.surface_relative_humidity, clock)
    p₀ = bf.surface_pressure
    ρ₀ = surface_density(p₀, T₀, constants)
    qᵛ₀ = ℋ₀ * saturation_specific_humidity(T₀, ρ₀, constants, surface)

    Δq = bulk_vapor_difference(i, j, k, fields, bf.filtered_scalar, qᵛ₀)

    U² = wall_wind_speed²(i, j, k, grid, side, nothing, fields, bf.filtered_velocities)
    Ũ = sqrt(U² + bf.gustiness^2)

    Cᵛ = bulk_coefficient(i, j, k, grid, side, bf.coefficient, fields, T₀, bf.filtered_velocities)

    # Over the wet fraction β of the wall the air in contact with it holds qᵛ₀ = ℋ₀ qᵛ⁺(T₀), and
    # over the dry fraction the humidity of the air itself, so that qᵛ - q₀ = β (qᵛ - qᵛ₀)
    β = bf.moisture_availability
    return outward_flux_sign(side) * ρ₀ * Cᵛ * Ũ * β * Δq
end

# Vapor difference dispatch on filtered_scalar
@inline function bulk_vapor_difference(i, j, k, fields, ::Nothing, qᵛ₀)
    qᵛ = @inbounds fields.qᵛ[i, j, k]
    return qᵛ - qᵛ₀
end

@inline function bulk_vapor_difference(i, j, k, fields, fs::FilteredSurfaceScalar, qᵛ₀)
    qᵛ = @inbounds fs.field[i, j, 1]
    return qᵛ - qᵛ₀
end

const BulkVaporFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}

#####
##### Convenient constructors
#####

"""
    BulkSensibleHeatFlux(; coefficient, gustiness=0, surface_temperature)

Create a `FluxBoundaryCondition` for wall sensible heat flux, on any of the six boundaries.

The bulk formula computes
```math
J = -ρ₀ Cᵀ |U| Δϕ
```
where ``Δϕ`` depends on the thermodynamic formulation: ``Δθ`` for potential
temperature or ``Δs`` for static energy. The formulation is set automatically
during model construction.

See [`BulkSensibleHeatFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

T₀(x, y, t) = 290 + 2 * sign(cos(2π * x / 20e3))

ρs_bc = BulkSensibleHeatFlux(coefficient = 1e-3,
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
    BulkVaporFlux(; coefficient, surface_temperature, surface_relative_humidity=1,
                    moisture_availability=nothing, gustiness=0)

Create a `FluxBoundaryCondition` for wall moisture flux, on any of the six boundaries.

The specific humidity of the air in contact with the wall is computed from
`surface_temperature` and `surface_relative_humidity` (unity by default, a wet wall);
`moisture_availability` is the fraction of the wall that is wet, 1 by default.

See [`BulkVaporFluxFunction`](@ref) for details.

# Example

```jldoctest
using Breeze

T₀(x, y, t) = 290 + 2 * sign(cos(2π * x / 20e3))

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
