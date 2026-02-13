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
       EnergyFluxBoundaryCondition,
       ThetaFluxBoundaryConditionFunction,
       ThetaFluxBoundaryCondition,
       PolynomialCoefficient,
       DefaultStabilityFunction,
       default_neutral_drag_polynomial,
       default_neutral_sensible_heat_polynomial,
       default_neutral_latent_heat_polynomial

using ..AtmosphereModels: AtmosphereModels, grid_moisture_fractions, dynamics_density
using ..AtmosphereModels.Diagnostics: VirtualPotentialTemperature, saturation_total_specific_moisture
using ..Thermodynamics: saturation_specific_humidity, surface_density, PlanarLiquidSurface,
                        mixture_heat_capacity, dry_air_gas_constant, vapor_gas_constant

using Oceananigans.Architectures: Architectures
using Oceananigans.Grids: Center, Face, XDirection, YDirection, AbstractGrid, znode
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
##### Helper functions
#####

# Get surface value from a Field or a Number
@inline surface_value(i, j, field::AbstractArray) = @inbounds field[i, j, 1]
@inline surface_value(i, j, x::Number) = x

#####
##### Wind speed calculations at staggered locations
#####

@inline ϕ²(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

# Wind speed squared at (Face, Center, Center) - for x-momentum flux
@inline function wind_speed²ᶠᶜᶜ(i, j, grid, fields)
    u² = surface_value(i, j, fields.u)^2
    v² = ℑxyᶠᶜᵃ(i, j, 1, grid, ϕ², fields.v)
    return u² + v²
end

# Wind speed squared at (Center, Face, Center) - for y-momentum flux
@inline function wind_speed²ᶜᶠᶜ(i, j, grid, fields)
    u² = ℑxyᶜᶠᵃ(i, j, 1, grid, ϕ², fields.u)
    v² = surface_value(i, j, fields.v)^2
    return u² + v²
end

# Wind speed squared at (Center, Center, Center) - for scalar fluxes
@inline function wind_speed²ᶜᶜᶜ(i, j, grid, fields)
    u² = ℑxᶜᵃᵃ(i, j, 1, grid, ϕ², fields.u)
    v² = ℑyᵃᶜᵃ(i, j, 1, grid, ϕ², fields.v)
    return u² + v²
end

#####
##### Boundary condition implementations
#####

include("polynomial_bulk_coefficient.jl")
include("bulk_drag.jl")
include("bulk_scalar_fluxes.jl")
include("thermodynamic_variable_bcs.jl")

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
`materialize_atmosphere_boundary_condition` on each one, allowing specialized handling for
bulk flux boundary conditions and other atmosphere-specific boundary condition types.

If `formulation` is `:LiquidIcePotentialTemperature` and `ρe` boundary conditions are provided,
they are automatically converted to `ρθ` boundary conditions using `EnergyFluxBoundaryCondition`.
"""
function AtmosphereModels.materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                                          dynamics, microphysics, surface_pressure,
                                                                          thermodynamic_constants,
                                                                          microphysical_fields, specific_moisture, temperature)

    # Convert ρe boundary conditions to ρθ for potential temperature formulations
    boundary_conditions = convert_energy_to_theta_bcs(boundary_conditions, formulation, thermodynamic_constants)

    materialized = Dict{Symbol, Any}()
    for (name, fbcs) in pairs(boundary_conditions)
        loc = field_location(Val(name))
        materialized[name] = materialize_atmosphere_field_bcs(fbcs, loc, grid, dynamics, microphysics,
                                                              surface_pressure, thermodynamic_constants,
                                                              microphysical_fields, specific_moisture, temperature)
    end
    return NamedTuple(materialized)
end

#####
##### Convert ρe boundary conditions to ρθ for potential temperature formulations
#####

const θFormulation = Union{Val{:LiquidIcePotentialTemperature}, Val{:θ}}
const eFormulation = Union{Val{:StaticEnergy}, Val{:e}, Val{:ρe}}

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

    ρe_bcs = set_sensible_heat_formulation_bcs(bcs.ρe, PotentialTemperatureFlux())
    ρθ_bcs = energy_to_theta_bcs(ρe_bcs)
    remaining = NamedTuple(k => v for (k, v) in pairs(bcs) if k !== :ρe)
    return merge(remaining, (; ρθ=ρθ_bcs))
end

# Set formulation on BulkSensibleHeatFlux for static energy formulations
function convert_energy_to_theta_bcs(bcs, formulation::eFormulation, constants)
    validate_thermodynamic_bcs(bcs)
    :ρe ∈ keys(bcs) || return bcs
    has_nondefault_bcs(bcs.ρe) || return bcs

    ρe_bcs = set_sensible_heat_formulation_bcs(bcs.ρe, StaticEnergyFlux())
    remaining = NamedTuple(k => v for (k, v) in pairs(bcs) if k !== :ρe)
    return merge(remaining, (; ρe=ρe_bcs))
end

convert_energy_to_theta_bcs(bcs, f::Symbol, c) = convert_energy_to_theta_bcs(bcs, Val(f), c)

# Materialize FieldBoundaryConditions by walking through each boundary
function materialize_atmosphere_field_bcs(fbcs::FieldBoundaryConditions, loc, grid, dynam, micro, p₀, consts,
                                          microphysical_fields, specific_moisture, temperature)
    args = (loc, grid, dynam, micro, p₀, consts, microphysical_fields, specific_moisture, temperature)
    west     = materialize_atmosphere_boundary_condition(fbcs.west,     West(),   args...)
    east     = materialize_atmosphere_boundary_condition(fbcs.east,     East(),   args...)
    south    = materialize_atmosphere_boundary_condition(fbcs.south,    South(),  args...)
    north    = materialize_atmosphere_boundary_condition(fbcs.north,    North(),  args...)
    bottom   = materialize_atmosphere_boundary_condition(fbcs.bottom,   Bottom(), args...)
    top      = materialize_atmosphere_boundary_condition(fbcs.top,      Top(),    args...)
    immersed = materialize_atmosphere_boundary_condition(fbcs.immersed, nothing,  args...)

    return FieldBoundaryConditions(; west, east, south, north, bottom, top, immersed)
end

# Default: pass through unchanged
materialize_atmosphere_boundary_condition(bc, side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                          microphysical_fields, specific_moisture, temperature) = bc

#####
##### Materialize coefficient: fill in VPT/pressure/constants for PolynomialCoefficient
#####

# Default: pass through unchanged (constant coefficients, etc.)
materialize_coefficient(C, grid, dynamics, microphysics, surface_pressure, constants,
                        microphysical_fields, specific_moisture, temperature) = C

# For PolynomialCoefficient: create VPT and return a fully-populated coefficient
function materialize_coefficient(coef::PolynomialCoefficient, grid, dynamics, microphysics,
                                 surface_pressure, constants,
                                 microphysical_fields, specific_moisture, temperature)
    reference_state = dynamics.reference_state
    θᵥ = VirtualPotentialTemperature(grid;
        reference_state, microphysics, microphysical_fields,
        specific_moisture, temperature, thermodynamic_constants=constants)

    return PolynomialCoefficient(coef.neutral_coefficients,
                                 coef.roughness_length,
                                 coef.minimum_wind_speed,
                                 coef.stability_function,
                                 θᵥ, surface_pressure, constants)
end

#####
##### Materialize BulkDrag: convert surface field and materialize coefficient
#####

function materialize_bulk_drag(df, grid, dynamics, microphysics, surface_pressure, constants,
                               microphysical_fields, specific_moisture, temperature)
    T₀ = materialize_surface_field(df.surface_temperature, grid)
    coef = materialize_coefficient(df.coefficient, grid, dynamics, microphysics,
                                   surface_pressure, constants,
                                   microphysical_fields, specific_moisture, temperature)
    new_df = BulkDragFunction(df.direction, coef, df.gustiness, T₀)
    return BoundaryCondition(Flux(), new_df)
end

# BulkDrag with no direction: infer direction from field location, then materialize
function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:BulkDragFunction{Nothing}},
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                  microphysical_fields, specific_moisture, temperature)
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

    directed_df = BulkDragFunction(direction, df.coefficient, df.gustiness, df.surface_temperature)
    return materialize_bulk_drag(directed_df, grid, dynamics, microphysics, surface_pressure, constants,
                                 microphysical_fields, specific_moisture, temperature)
end

# BulkDrag with direction already set: materialize
function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:XDirectionBulkDragFunction},
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                  microphysical_fields, specific_moisture, temperature)
    return materialize_bulk_drag(bc.condition, grid, dynamics, microphysics, surface_pressure, constants,
                                 microphysical_fields, specific_moisture, temperature)
end

function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:YDirectionBulkDragFunction},
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                  microphysical_fields, specific_moisture, temperature)
    return materialize_bulk_drag(bc.condition, grid, dynamics, microphysics, surface_pressure, constants,
                                 microphysical_fields, specific_moisture, temperature)
end

# Materialize BulkSensibleHeatFlux: populate surface_pressure, thermodynamic_constants, preserve formulation
function materialize_atmosphere_boundary_condition(bc::BulkSensibleHeatFluxBoundaryCondition,
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                  microphysical_fields, specific_moisture, temperature)
    bf = bc.condition
    T₀ = materialize_surface_field(bf.surface_temperature, grid)
    coef = materialize_coefficient(bf.coefficient, grid, dynamics, microphysics,
                                   surface_pressure, constants,
                                   microphysical_fields, specific_moisture, temperature)
    new_bf = BulkSensibleHeatFluxFunction(coef, bf.gustiness, T₀, surface_pressure, constants, bf.formulation)
    return BoundaryCondition(Flux(), new_bf)
end

# Materialize BulkVaporFlux: populate surface_pressure, thermodynamic_constants, and surface
function materialize_atmosphere_boundary_condition(bc::BulkVaporFluxBoundaryCondition,
                                                  side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                  microphysical_fields, specific_moisture, temperature)
    bf = bc.condition
    T₀ = materialize_surface_field(bf.surface_temperature, grid)
    surface = PlanarLiquidSurface()
    coef = materialize_coefficient(bf.coefficient, grid, dynamics, microphysics,
                                   surface_pressure, constants,
                                   microphysical_fields, specific_moisture, temperature)
    new_bf = BulkVaporFluxFunction(coef, bf.gustiness, T₀, surface_pressure, constants, surface)
    return BoundaryCondition(Flux(), new_bf)
end

#####
##### Utilities
#####

# Helper to convert functions to Fields
materialize_surface_field(::Nothing, grid) = nothing
materialize_surface_field(f::Field, grid) = f
materialize_surface_field(f::Number, grid) = f

function materialize_surface_field(f::Function, grid)
    field = Field{Center, Center, Nothing}(grid)
    set!(field, f)
    return field
end

end # module BoundaryConditions
