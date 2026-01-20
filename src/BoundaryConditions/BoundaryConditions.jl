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
       BulkVaporFluxBoundaryCondition,
       EnergyFluxBoundaryConditionFunction,
       EnergyFluxBoundaryCondition

using ..AtmosphereModels: AtmosphereModels
using ..Thermodynamics: saturation_specific_humidity, surface_density, PlanarLiquidSurface,
                        MoistureMassFractions, mixture_heat_capacity

using Oceananigans.Architectures: Architectures, on_architecture
using Oceananigans.Grids: Center, Face, XDirection, YDirection, AbstractGrid
using Oceananigans.Fields: Field, set!
using Oceananigans.BoundaryConditions: BoundaryConditions as OceananigansBC,
                                       BoundaryCondition,
                                       DefaultBoundaryCondition,
                                       Flux,
                                       FieldBoundaryConditions,
                                       Bottom, Top, West, East, South, North

using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES

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

Architectures.on_architecture(to, df::BulkDragFunction) =
    BulkDragFunction(on_architecture(to, df.direction),
                     on_architecture(to, df.coefficient),
                     on_architecture(to, df.gustiness))

Base.summary(df::BulkDragFunction) = string("BulkDragFunction(direction=", summary(df.direction),
                                            ", coefficient=", df.coefficient,
                                            ", gustiness=", df.gustiness, ")")

# getbc for BulkDragFunction

const XDBDF = XDirectionBulkDragFunction
const YDBDF = YDirectionBulkDragFunction

@inline function OceananigansBC.getbc(df::XDBDF, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
    ρu = @inbounds fields.ρu[i, j, 1]
    U² = wind_speed²ᶠᶜᶜ(i, j, grid, fields)
    U = sqrt(U²)
    Ũ² = U² + df.gustiness^2
    Cᴰ = df.coefficient
    return - Cᴰ * Ũ² * ρu / U * (U > 0)
end

@inline function OceananigansBC.getbc(df::YDBDF, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
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
function BulkSensibleHeatFluxFunction(; coefficient, gustiness=0, surface_temperature)
    return BulkSensibleHeatFluxFunction(coefficient, gustiness,
                                        surface_temperature, nothing, nothing)
end

Adapt.adapt_structure(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(Adapt.adapt(to, bf.coefficient),
                                 Adapt.adapt(to, bf.gustiness),
                                 Adapt.adapt(to, bf.surface_temperature),
                                 Adapt.adapt(to, bf.surface_pressure),
                                 Adapt.adapt(to, bf.thermodynamic_constants))

Architectures.on_architecture(to, bf::BulkSensibleHeatFluxFunction) =
    BulkSensibleHeatFluxFunction(on_architecture(to, bf.coefficient),
                                 on_architecture(to, bf.gustiness),
                                 on_architecture(to, bf.surface_temperature),
                                 on_architecture(to, bf.surface_pressure),
                                 on_architecture(to, bf.thermodynamic_constants))

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
function BulkVaporFluxFunction(; coefficient, gustiness=0, surface_temperature)
    return BulkVaporFluxFunction(coefficient, gustiness,
                                 surface_temperature, nothing, nothing, nothing)
end

Adapt.adapt_structure(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(Adapt.adapt(to, bf.coefficient),
                          Adapt.adapt(to, bf.gustiness),
                          Adapt.adapt(to, bf.surface_temperature),
                          Adapt.adapt(to, bf.surface_pressure),
                          Adapt.adapt(to, bf.thermodynamic_constants),
                          Adapt.adapt(to, bf.surface))

Architectures.on_architecture(to, bf::BulkVaporFluxFunction) =
    BulkVaporFluxFunction(on_architecture(to, bf.coefficient),
                          on_architecture(to, bf.gustiness),
                          on_architecture(to, bf.surface_temperature),
                          on_architecture(to, bf.surface_pressure),
                          on_architecture(to, bf.thermodynamic_constants),
                          on_architecture(to, bf.surface))

Base.summary(bf::BulkVaporFluxFunction) =
    string("BulkVaporFluxFunction(coefficient=", bf.coefficient,
           ", gustiness=", bf.gustiness, ")")

const BVFF = BulkVaporFluxFunction

# getbc for BulkVaporFluxFunction
@inline function OceananigansBC.getbc(bf::BVFF, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
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

#####
##### Type aliases for FluxBoundaryCondition with these functions
#####

const BulkDragBoundaryCondition = BoundaryCondition{<:Flux, <:BulkDragFunction}
const BulkSensibleHeatFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkSensibleHeatFluxFunction}
const BulkVaporFluxBoundaryCondition = BoundaryCondition{<:Flux, <:BulkVaporFluxFunction}

#####
##### EnergyFluxBoundaryConditionFunction for converting energy flux to potential temperature flux
#####

"""
    EnergyFluxBoundaryConditionFunction

A wrapper for boundary conditions that converts energy flux to potential temperature flux.

When using `LiquidIcePotentialTemperatureFormulation`, the prognostic thermodynamic variable
is `ρθ` (potential temperature density). This wrapper allows users to specify energy fluxes
(e.g., sensible heat flux in W/m²) which are converted to potential temperature fluxes by
dividing by the local mixture heat capacity `cᵖᵐ`.

The relationship is:
```math
Jᶿ = 𝒬 / cᵖᵐ
```

where `𝒬` is the energy flux and `Jᶿ` is the potential temperature flux.
"""
struct EnergyFluxBoundaryConditionFunction{C, S, TC}
    condition :: C                    # underlying BC (function, number, or Oceananigans BC type)
    side :: S                         # Bottom(), Top(), etc. to determine which k index to use
    thermodynamic_constants :: TC
end

Adapt.adapt_structure(to, ef::EnergyFluxBoundaryConditionFunction) =
    EnergyFluxBoundaryConditionFunction(Adapt.adapt(to, ef.condition),
                                        Adapt.adapt(to, ef.side),
                                        Adapt.adapt(to, ef.thermodynamic_constants))

Architectures.on_architecture(to, ef::EnergyFluxBoundaryConditionFunction) =
    EnergyFluxBoundaryConditionFunction(on_architecture(to, ef.condition),
                                        on_architecture(to, ef.side),
                                        on_architecture(to, ef.thermodynamic_constants))

function Base.summary(ef::EnergyFluxBoundaryConditionFunction)
    cond = ef.condition
    cond_str = cond isa Number ? string(cond) : summary(cond)
    return string("EnergyFluxBoundaryConditionFunction(", cond_str, ")")
end

# Type aliases for dispatch
const EFBCF = EnergyFluxBoundaryConditionFunction
const BEFBC = EFBCF{<:Any, <:Bottom}
const TEFBC = EFBCF{<:Any, <:Top}
const WEFBC = EFBCF{<:Any, <:West}
const EEFBC = EFBCF{<:Any, <:East}
const SEFBC = EFBCF{<:Any, <:South}
const NEFBC = EFBCF{<:Any, <:North}

# Helper to get underlying energy flux value
@inline _get_𝒬(c::Number, args...) = c
@inline _get_𝒬(c, args...) = OceananigansBC.getbc(c, args...)

# Convert energy flux to potential temperature flux: Jᶿ = 𝒬 / cᵖᵐ
@inline function _energy_to_θ_flux(𝒬, qᵗ, constants)
    cᵖᵐ = mixture_heat_capacity(MoistureMassFractions(qᵗ), constants)
    return 𝒬 / cᵖᵐ
end

# getbc implementations for each boundary face
@inline function OceananigansBC.getbc(ef::BEFBC, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
    𝒬 = _get_𝒬(ef.condition, i, j, grid, clock, fields)
    return _energy_to_θ_flux(𝒬, @inbounds(fields.qᵗ[i, j, 1]), ef.thermodynamic_constants)
end

@inline function OceananigansBC.getbc(ef::TEFBC, i::Integer, j::Integer, grid::AbstractGrid, clock, fields)
    𝒬 = _get_𝒬(ef.condition, i, j, grid, clock, fields)
    return _energy_to_θ_flux(𝒬, @inbounds(fields.qᵗ[i, j, grid.Nz]), ef.thermodynamic_constants)
end

@inline function OceananigansBC.getbc(ef::WEFBC, j::Integer, k::Integer, grid::AbstractGrid, clock, fields)
    𝒬 = _get_𝒬(ef.condition, j, k, grid, clock, fields)
    return _energy_to_θ_flux(𝒬, @inbounds(fields.qᵗ[1, j, k]), ef.thermodynamic_constants)
end

@inline function OceananigansBC.getbc(ef::EEFBC, j::Integer, k::Integer, grid::AbstractGrid, clock, fields)
    𝒬 = _get_𝒬(ef.condition, j, k, grid, clock, fields)
    return _energy_to_θ_flux(𝒬, @inbounds(fields.qᵗ[grid.Nx, j, k]), ef.thermodynamic_constants)
end

@inline function OceananigansBC.getbc(ef::SEFBC, i::Integer, k::Integer, grid::AbstractGrid, clock, fields)
    𝒬 = _get_𝒬(ef.condition, i, k, grid, clock, fields)
    return _energy_to_θ_flux(𝒬, @inbounds(fields.qᵗ[i, 1, k]), ef.thermodynamic_constants)
end

@inline function OceananigansBC.getbc(ef::NEFBC, i::Integer, k::Integer, grid::AbstractGrid, clock, fields)
    𝒬 = _get_𝒬(ef.condition, i, k, grid, clock, fields)
    return _energy_to_θ_flux(𝒬, @inbounds(fields.qᵗ[i, grid.Ny, k]), ef.thermodynamic_constants)
end

const EnergyFluxBC = BoundaryCondition{<:Flux, <:EFBCF}

#####
##### Convenient constructors
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

"""
    EnergyFluxBoundaryCondition(flux)

Internal function to create a boundary condition wrapper that converts an energy flux to a
potential temperature flux. Users should specify `ρe` boundary conditions instead, which
are automatically converted to `ρθ` boundary conditions when using potential temperature
formulations.

The energy flux is divided by the local mixture heat capacity `cᵖᵐ` to obtain the
potential temperature flux:

```math
Jᶿ = 𝒬 / cᵖᵐ
```

The mixture heat capacity is computed at the boundary using the local specific humidity `qᵗ`.
"""
function EnergyFluxBoundaryCondition(flux)
    # side and thermodynamic_constants are filled in during regularization
    ef = EnergyFluxBoundaryConditionFunction(flux, nothing, nothing)
    return BoundaryCondition(Flux(), ef)
end

#####
##### AtmosphereModel boundary condition regularization
#####

# Field location from field name
field_location(::Val{:ρu}) = (Face(), Center(), Center())
field_location(::Val{:ρv}) = (Center(), Face(), Center())
field_location(::Val{:ρw}) = (Center(), Center(), Face())
field_location(::Val) = (Center(), Center(), Center())  # default for scalars

"""
$(TYPEDSIGNATURES)

Regularize boundary conditions for [`AtmosphereModel`](@ref AtmosphereModels.AtmosphereModel).
This function walks through all boundary conditions and calls
`regularize_atmosphere_boundary_condition` on each one, allowing specialized handling for
bulk flux boundary conditions and other atmosphere-specific boundary condition types.

If `formulation` is `:LiquidIcePotentialTemperature` and `ρe` boundary conditions are provided,
they are automatically converted to `ρθ` boundary conditions using `EnergyFluxBoundaryCondition`.
"""
function AtmosphereModels.regularize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation, surface_pressure, thermodynamic_constants)
    # Convert ρe boundary conditions to ρθ for potential temperature formulations
    boundary_conditions = convert_energy_to_theta_bcs(boundary_conditions, formulation, thermodynamic_constants)

    regularized = Dict{Symbol, Any}()
    for (name, fbcs) in pairs(boundary_conditions)
        loc = field_location(Val(name))
        regularized[name] = regularize_atmosphere_field_bcs(fbcs, loc, grid, surface_pressure, thermodynamic_constants)
    end
    return NamedTuple(regularized)
end

#####
##### Convert ρe boundary conditions to ρθ for potential temperature formulations
#####

const θFormulation = Union{Val{:LiquidIcePotentialTemperature}, Val{:θ}}

# Check if FieldBoundaryConditions has any non-default values
has_nondefault_bcs(::Nothing) = false
has_nondefault_bcs(fbcs) = false
function has_nondefault_bcs(fbcs::FieldBoundaryConditions)
    for side in (:west, :east, :south, :north, :bottom, :top, :immersed)
        bc = getproperty(fbcs, side)
        bc isa Nothing && continue
        bc isa BoundaryCondition{<:Flux, Nothing} && continue
        bc isa DefaultBoundaryCondition && continue
        return true
    end
    return false
end

# Validate: error if BOTH ρθ and ρe have non-default BCs
function validate_thermodynamic_bcs(bcs)
    has_ρθ = :ρθ ∈ keys(bcs) && has_nondefault_bcs(bcs.ρθ)
    has_ρe = :ρe ∈ keys(bcs) && has_nondefault_bcs(bcs.ρe)
    if has_ρθ && has_ρe
        throw(ArgumentError("Cannot specify boundary conditions on both ρθ and ρe. " *
                            "Use ρe for energy fluxes or ρθ for potential temperature fluxes, but not both."))
    end
    return nothing
end

# Fallback: no conversion (but validate)
function convert_energy_to_theta_bcs(bcs, formulation, constants)
    validate_thermodynamic_bcs(bcs)
    return bcs
end

# Convert ρe → ρθ for potential temperature formulations
function convert_energy_to_theta_bcs(bcs, formulation::θFormulation, constants)
    validate_thermodynamic_bcs(bcs)
    :ρe ∈ keys(bcs) || return bcs
    has_nondefault_bcs(bcs.ρe) || return bcs

    ρθ_bcs = wrap_energy_field_bcs(bcs.ρe)
    remaining = NamedTuple(k => v for (k, v) in pairs(bcs) if k !== :ρe)
    return merge(remaining, (; ρθ=ρθ_bcs))
end

convert_energy_to_theta_bcs(bcs, f::Symbol, c) = convert_energy_to_theta_bcs(bcs, Val(f), c)

# Wrap FieldBoundaryConditions with EnergyFluxBoundaryCondition
function wrap_energy_field_bcs(fbcs::FieldBoundaryConditions)
    return FieldBoundaryConditions(; west     = wrap_energy_bc(fbcs.west),
                                     east     = wrap_energy_bc(fbcs.east),
                                     south    = wrap_energy_bc(fbcs.south),
                                     north    = wrap_energy_bc(fbcs.north),
                                     bottom   = wrap_energy_bc(fbcs.bottom),
                                     top      = wrap_energy_bc(fbcs.top),
                                     immersed = wrap_energy_bc(fbcs.immersed))
end

wrap_energy_field_bcs(fbcs) = fbcs

wrap_energy_bc(bc) = bc

# BulkSensibleHeatFlux already returns a potential temperature flux, so pass it through directly
wrap_energy_bc(bc::BulkSensibleHeatFluxBoundaryCondition) = bc

# Other flux BCs get wrapped to convert energy → potential temperature
wrap_energy_bc(bc::BoundaryCondition{<:Flux}) = EnergyFluxBoundaryCondition(bc.condition)

# Pass through non-FieldBoundaryConditions
regularize_atmosphere_field_bcs(fbcs, loc, grid, surface_pressure, constants) = fbcs

# Regularize FieldBoundaryConditions by walking through each boundary
function regularize_atmosphere_field_bcs(fbcs::FieldBoundaryConditions, loc, grid, surface_pressure, constants)
    west     = regularize_atmosphere_boundary_condition(fbcs.west, West(), loc, grid, surface_pressure, constants)
    east     = regularize_atmosphere_boundary_condition(fbcs.east, East(), loc, grid, surface_pressure, constants)
    south    = regularize_atmosphere_boundary_condition(fbcs.south, South(), loc, grid, surface_pressure, constants)
    north    = regularize_atmosphere_boundary_condition(fbcs.north, North(), loc, grid, surface_pressure, constants)
    bottom   = regularize_atmosphere_boundary_condition(fbcs.bottom, Bottom(), loc, grid, surface_pressure, constants)
    top      = regularize_atmosphere_boundary_condition(fbcs.top, Top(), loc, grid, surface_pressure, constants)
    immersed = regularize_atmosphere_boundary_condition(fbcs.immersed, nothing, loc, grid, surface_pressure, constants)

    return FieldBoundaryConditions(; west, east, south, north, bottom, top, immersed)
end

# Default: pass through unchanged
regularize_atmosphere_boundary_condition(bc, side, loc, grid, surface_pressure, constants) = bc

# Regularize BulkDrag: infer direction from field location if needed
function regularize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:BulkDragFunction{Nothing}},
                                                  side, loc, grid, surface_pressure, constants)
    df = bc.condition
    LX, LY, _ = loc

    # Determine direction from location: Face in x means x-momentum, Face in y means y-momentum
    if LX isa Face
        direction = XDirection()
    elseif LY isa Face
        direction = YDirection()
    else
        throw(ArgumentError("Can only specify BulkDrag on x-momentum or y-momentum fields!"))
    end

    regularized_df = BulkDragFunction(direction, df.coefficient, df.gustiness)
    return BoundaryCondition(Flux(), regularized_df)
end

# BulkDrag with direction already set: pass through
regularize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:XDirectionBulkDragFunction},
                                         side, loc, grid, surface_pressure, constants) = bc
regularize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:YDirectionBulkDragFunction},
                                         side, loc, grid, surface_pressure, constants) = bc

# Regularize BulkSensibleHeatFlux: populate surface_pressure and thermodynamic_constants
function regularize_atmosphere_boundary_condition(bc::BulkSensibleHeatFluxBoundaryCondition,
                                                  side, loc, grid, surface_pressure, constants)
    bf = bc.condition
    T₀ = materialize_surface_field(bf.surface_temperature, grid)
    new_bf = BulkSensibleHeatFluxFunction(bf.coefficient, bf.gustiness, T₀, surface_pressure, constants)
    return BoundaryCondition(Flux(), new_bf)
end

# Regularize BulkVaporFlux: populate surface_pressure, thermodynamic_constants, and surface
function regularize_atmosphere_boundary_condition(bc::BulkVaporFluxBoundaryCondition,
                                                  side, loc, grid, surface_pressure, constants)
    bf = bc.condition
    T₀ = materialize_surface_field(bf.surface_temperature, grid)
    surface = PlanarLiquidSurface()
    new_bf = BulkVaporFluxFunction(bf.coefficient, bf.gustiness, T₀, surface_pressure, constants, surface)
    return BoundaryCondition(Flux(), new_bf)
end

# Regularize EnergyFluxBoundaryCondition: populate side and thermodynamic_constants
const UnregularizedEFBC = BoundaryCondition{<:Flux, <:EFBCF{<:Any, Nothing}}

function regularize_atmosphere_boundary_condition(bc::UnregularizedEFBC,
                                                  side, loc, grid, surface_pressure, constants)
    ef = bc.condition
    new_ef = EnergyFluxBoundaryConditionFunction(ef.condition, side, constants)
    return BoundaryCondition(Flux(), new_ef)
end

#####
##### Utilities
#####

# Helper to convert functions to Fields
materialize_surface_field(f::Field, grid) = f
materialize_surface_field(f::Number, grid) = f

function materialize_surface_field(f::Function, grid)
    field = Field{Center, Center, Nothing}(grid)
    set!(field, f)
    return field
end

#####
##### EnergyFluxOperation: extract energy flux from EnergyFluxBoundaryCondition
#####

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: location
using Oceananigans.Models: BoundaryConditionOperation
using Oceananigans: fields

export EnergyFluxOperation

"""
    EnergyFluxKernelFunction

Kernel function that extracts the energy flux `𝒬` from an `EnergyFluxBoundaryConditionFunction`.
Unlike the default `getbc` which returns the converted potential temperature flux `Jᶿ = 𝒬/cᵖᵐ`,
this returns the original energy flux `𝒬`.
"""
struct EnergyFluxKernelFunction{S, C}
    side :: S
    condition :: C  # The underlying condition (number, function, or BC type)
end

Adapt.adapt_structure(to, ef::EnergyFluxKernelFunction) =
    EnergyFluxKernelFunction(Adapt.adapt(to, ef.side), Adapt.adapt(to, ef.condition))

# Dispatch on side to get correct indices for getbc
@inline function (kf::EnergyFluxKernelFunction{<:Bottom})(i, j, k, grid, clock, model_fields)
    return _get_𝒬(kf.condition, i, j, grid, clock, model_fields)
end

@inline function (kf::EnergyFluxKernelFunction{<:Top})(i, j, k, grid, clock, model_fields)
    return _get_𝒬(kf.condition, i, j, grid, clock, model_fields)
end

@inline function (kf::EnergyFluxKernelFunction{<:West})(i, j, k, grid, clock, model_fields)
    return _get_𝒬(kf.condition, j, k, grid, clock, model_fields)
end

@inline function (kf::EnergyFluxKernelFunction{<:East})(i, j, k, grid, clock, model_fields)
    return _get_𝒬(kf.condition, j, k, grid, clock, model_fields)
end

@inline function (kf::EnergyFluxKernelFunction{<:South})(i, j, k, grid, clock, model_fields)
    return _get_𝒬(kf.condition, i, k, grid, clock, model_fields)
end

@inline function (kf::EnergyFluxKernelFunction{<:North})(i, j, k, grid, clock, model_fields)
    return _get_𝒬(kf.condition, i, k, grid, clock, model_fields)
end

"""
$(TYPEDSIGNATURES)

Create a `KernelFunctionOperation` that returns the energy flux `𝒬` from the boundary condition
on the thermodynamic field (e.g., `ρθ` or `ρe`).

If the boundary condition is an `EnergyFluxBoundaryCondition`, this returns the underlying
energy flux before conversion to potential temperature flux. Otherwise, it returns the
flux value multiplied by the mixture heat capacity `cᵖᵐ` to convert to energy flux.

# Example

```julia
using Breeze
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), extent=(1000, 1000, 1000))
𝒬₀ = 100  # W/m²

model = AtmosphereModel(grid; boundary_conditions=(ρe=FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬₀)),))

# Get the energy flux at the bottom boundary
𝒬 = EnergyFluxOperation(model, :bottom)
```
"""
function EnergyFluxOperation end

function energy_flux_location(side, LX, LY, LZ)
    if side === :top || side === :bottom
        return LX, LY, Nothing
    elseif side === :west || side === :east
        return Nothing, LY, LZ
    elseif side === :south || side === :north
        return LX, Nothing, LZ
    end
end

function side_type(side::Symbol)
    side === :bottom && return Bottom()
    side === :top    && return Top()
    side === :west   && return West()
    side === :east   && return East()
    side === :south  && return South()
    side === :north  && return North()
    throw(ArgumentError("Unknown side: $side"))
end

using Breeze.AtmosphereModels: thermodynamic_density

# For EnergyFluxBoundaryCondition, extract the underlying condition
function EnergyFluxOperation(model, side::Symbol)
    # Get the thermodynamic field - ρθ for potential temperature formulation
    ρθ = thermodynamic_density(model.formulation)
    bc = getproperty(ρθ.boundary_conditions, side)
    return _energy_flux_operation(bc, ρθ, side, model)
end

# For EnergyFluxBoundaryCondition: extract underlying condition and return energy flux
function _energy_flux_operation(bc::EnergyFluxBC, field, side, model)
    ef = bc.condition
    grid = field.grid
    LX, LY, LZ = energy_flux_location(side, location(field)...)
    side_t = side_type(side)
    kernel_func = EnergyFluxKernelFunction(side_t, ef.condition)
    return KernelFunctionOperation{LX, LY, LZ}(kernel_func, grid, model.clock, fields(model))
end

# Helper kernel functions for cᵖᵐ at boundaries
@inline _cᵖᵐ_bottom(i, j, k, grid, qᵗ, constants) =
    mixture_heat_capacity(MoistureMassFractions(@inbounds qᵗ[i, j, 1]), constants)

@inline _cᵖᵐ_top(i, j, k, grid, qᵗ, constants, Nz) =
    mixture_heat_capacity(MoistureMassFractions(@inbounds qᵗ[i, j, Nz]), constants)

# For other boundary conditions: multiply by cᵖᵐ to get energy flux
function _energy_flux_operation(bc::BoundaryCondition, field, side, model)
    # Fall back to BoundaryConditionOperation and multiply by cᵖᵐ
    Jᶿ = BoundaryConditionOperation(field, side, model)
    constants = model.thermodynamic_constants
    qᵗ = model.specific_moisture
    # Project cᵖᵐ to boundary location
    if side === :bottom
        cᵖᵐ_bc = KernelFunctionOperation{Center, Center, Nothing}(_cᵖᵐ_bottom, field.grid, qᵗ, constants)
        return cᵖᵐ_bc * Jᶿ
    elseif side === :top
        Nz = field.grid.Nz
        cᵖᵐ_bc = KernelFunctionOperation{Center, Center, Nothing}(_cᵖᵐ_top, field.grid, qᵗ, constants, Nz)
        return cᵖᵐ_bc * Jᶿ
    else
        throw(ArgumentError("EnergyFluxOperation for side $side not yet implemented"))
    end
end

end # module BoundaryConditions
