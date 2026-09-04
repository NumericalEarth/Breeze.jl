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
       FilteredSurfaceVelocities,
       FilteredSurfaceScalar,
       PolynomialCoefficient,
       FittedStabilityFunction,
       StabilityFunctionParameters,
       RichardsonNumberMapping,
       default_neutral_drag_polynomial,
       default_neutral_sensible_heat_polynomial,
       default_neutral_latent_heat_polynomial

using ..AtmosphereModels: AtmosphereModels, grid_moisture_fractions, dynamics_density,
                          standard_pressure, boundary_conditions_reference_state,
                          default_drag_surface_temperature
using ..AtmosphereModels.Diagnostics: VirtualPotentialTemperature, saturation_total_specific_moisture
using ..Thermodynamics: saturation_specific_humidity, surface_density, PlanarLiquidSurface,
                        mixture_heat_capacity, dry_air_gas_constant, vapor_gas_constant,
                        potential_temperature_from_temperature

using Oceananigans: Oceananigans
using Oceananigans.Architectures: Architectures
using Oceananigans.BoundaryConditions: BoundaryConditions as OceananigansBC,
                                       BoundaryCondition,
                                       DefaultBoundaryCondition,
                                       Flux,
                                       FieldBoundaryConditions,
                                       Bottom, Top, West, East, South, North
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face, XDirection, YDirection, ZDirection, AbstractGrid, node, znode
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶜᶠ, ℑxzᶠᵃᶜ, ℑxzᶜᵃᶠ,
                              Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES

#####
##### Boundary condition implementations
#####

include("filtered_surface_state.jl")
include("wall_faces.jl")
include("polynomial_bulk_coefficient.jl")
include("bulk_drag.jl")
include("bulk_scalar_fluxes.jl")
include("thermodynamic_variable_bcs.jl")
include("update_boundary_conditions.jl")

#####
##### Wind speed at the bottom wall

@inline function wind_speed²ᶠᶜᶜ(i, j, grid, fields, fv::FilteredSurfaceVelocities)
    u² = @inbounds fv.u[i, j, 1]^2
    v² = ℑxyᶠᶜᵃ(i, j, 1, grid, ϕ², fv.v)
    return u² + v²
end

@inline function wind_speed²ᶜᶠᶜ(i, j, grid, fields, fv::FilteredSurfaceVelocities)
    u² = ℑxyᶜᶠᵃ(i, j, 1, grid, ϕ², fv.u)
    v² = @inbounds fv.v[i, j, 1]^2
    return u² + v²
end

@inline function wind_speed²ᶜᶜᶜ(i, j, grid, fields, fv::FilteredSurfaceVelocities)
    u² = ℑxᶜᵃᵃ(i, j, 1, grid, ϕ², fv.u)
    v² = ℑyᵃᶜᵃ(i, j, 1, grid, ϕ², fv.v)
    return u² + v²
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
`materialize_atmosphere_boundary_condition` on each one, allowing specialized handling for
bulk flux boundary conditions and other atmosphere-specific boundary condition types.

If `formulation` is `:LiquidIcePotentialTemperature` and `ρs` boundary conditions are provided,
they are automatically converted to `ρθ` boundary conditions using `EnergyFluxBoundaryCondition`.
"""
function AtmosphereModels.materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                                           dynamics, microphysics, surface_pressure,
                                                                           thermodynamic_constants,
                                                                           microphysical_fields, specific_prognostic_moisture, temperature)

    # Convert ρs boundary conditions to ρθ for potential temperature formulations
    boundary_conditions = convert_energy_to_theta_bcs(boundary_conditions, formulation, thermodynamic_constants)

    materialized = Dict{Symbol, Any}()
    for (name, fbcs) in pairs(boundary_conditions)
        loc = field_location(Val(name))
        materialized[name] = materialize_atmosphere_field_bcs(fbcs, loc, grid, dynamics, microphysics,
                                                              surface_pressure, thermodynamic_constants,
                                                              microphysical_fields, specific_prognostic_moisture, temperature)
    end
    return NamedTuple(materialized)
end

#####
##### Convert ρs boundary conditions to ρθ for potential temperature formulations
#####

const θFormulation = Union{Val{:LiquidIcePotentialTemperature}, Val{:θ}}
const sFormulation = Union{Val{:StaticEnergy}, Val{:s}, Val{:ρs}}

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

# Validate: error if BOTH ρθ and ρs have non-default BCs
function validate_thermodynamic_bcs(bcs)
    has_ρθ = :ρθ ∈ keys(bcs) && has_nondefault_bcs(bcs.ρθ)
    has_ρs = :ρs ∈ keys(bcs) && has_nondefault_bcs(bcs.ρs)
    if has_ρθ && has_ρs
        throw(ArgumentError("Cannot specify boundary conditions on both ρθ and ρs. " *
                            "Use ρs for energy fluxes or ρθ for potential temperature fluxes, but not both."))
    end
    return nothing
end

# Fallback: no conversion (but validate)
function convert_energy_to_theta_bcs(bcs, formulation, constants)
    validate_thermodynamic_bcs(bcs)
    return bcs
end

# Convert ρs → ρθ for potential temperature formulations
function convert_energy_to_theta_bcs(bcs, formulation::θFormulation, constants)
    validate_thermodynamic_bcs(bcs)
    :ρs ∈ keys(bcs) || return bcs
    has_nondefault_bcs(bcs.ρs) || return bcs

    ρs_bcs = set_sensible_heat_formulation_bcs(bcs.ρs, PotentialTemperatureFlux())
    ρθ_bcs = energy_to_theta_bcs(ρs_bcs)
    remaining = NamedTuple(k => v for (k, v) in pairs(bcs) if k !== :ρs)
    return merge(remaining, (; ρθ=ρθ_bcs))
end

# Set formulation on BulkSensibleHeatFlux for static energy formulations
function convert_energy_to_theta_bcs(bcs, formulation::sFormulation, constants)
    validate_thermodynamic_bcs(bcs)
    :ρs ∈ keys(bcs) || return bcs
    has_nondefault_bcs(bcs.ρs) || return bcs

    ρs_bcs = set_sensible_heat_formulation_bcs(bcs.ρs, StaticEnergyFlux())
    remaining = NamedTuple(k => v for (k, v) in pairs(bcs) if k !== :ρs)
    return merge(remaining, (; ρs=ρs_bcs))
end

convert_energy_to_theta_bcs(bcs, f::Symbol, c) = convert_energy_to_theta_bcs(bcs, Val(f), c)

# Materialize FieldBoundaryConditions by walking through each boundary
function materialize_atmosphere_field_bcs(fbcs::FieldBoundaryConditions, loc, grid, dynam, micro, p₀, consts,
                                          microphysical_fields, specific_prognostic_moisture, temperature)
    args = (loc, grid, dynam, micro, p₀, consts, microphysical_fields, specific_prognostic_moisture, temperature)
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
                                          microphysical_fields, specific_prognostic_moisture, temperature) = bc

#####
##### Materialize coefficient: fill in VPT/pressure/constants for PolynomialCoefficient
#####

# Default: pass through unchanged (constant coefficients, etc.)
materialize_coefficient(C, grid, dynamics, microphysics, surface_pressure, constants,
                        microphysical_fields, specific_prognostic_moisture, temperature, transfer_type) = C

# For PolynomialCoefficient: create VPT and return a fully-populated coefficient
function materialize_coefficient(coef::PolynomialCoefficient, grid, dynamics, microphysics,
                                 surface_pressure, constants,
                                 microphysical_fields, specific_prognostic_moisture, temperature,
                                 transfer_type)
    reference_state = boundary_conditions_reference_state(dynamics, grid, constants)
    θᵥ = VirtualPotentialTemperature(grid;
        reference_state, microphysics, microphysical_fields,
        specific_prognostic_moisture, temperature, thermodynamic_constants=constants)

    return PolynomialCoefficient(coef.polynomial,
                                 coef.roughness_length,
                                 coef.minimum_wind_speed,
                                 coef.stability_function,
                                 coef.surface,
                                 θᵥ, surface_pressure, constants,
                                 transfer_type)
end

#####
##### Materialize BulkDrag: convert surface field and materialize coefficient
#####

# Bulk fluxes are evaluated on the six walls of the domain, not on immersed boundaries
validate_wall(side) = nothing
validate_wall(::Nothing) = throw(ArgumentError("Bulk flux boundary conditions are not supported on immersed boundaries"))

# Drag acts on the momentum components tangential to the wall
function validate_drag_direction(side, direction)
    if direction isa typeof(wall_normal_direction(side))
        throw(ArgumentError("BulkDrag cannot act on the momentum component normal to the $(typeof(side)) boundary"))
    end
    return nothing
end

function materialize_bulk_drag(df, side, grid, dynamics, microphysics, surface_pressure, constants,
                               microphysical_fields, specific_prognostic_moisture, temperature)
    validate_wall(side)
    validate_drag_direction(side, df.direction)
    validate_wall_filtering(side, df.filtered_velocities)

    # The momentum-drag formula `Jᵘ = -ρ₀ Cᴰ |U| u` needs a surface temperature to
    # compute ρ₀. When the user did not supply one (allowed for constant `coefficient`),
    # fall back to the reference-state surface temperature derived from the dynamics.
    T₀_input = if isnothing(df.surface_temperature)
        default_drag_surface_temperature(dynamics, grid, constants)
    else
        df.surface_temperature
    end
    T₀ = materialize_surface_field(T₀_input, grid, side)
    coef = materialize_coefficient(df.coefficient, grid, dynamics, microphysics,
                                   surface_pressure, constants,
                                   microphysical_fields, specific_prognostic_moisture, temperature,
                                   Val(:momentum))
    new_df = BulkDragFunction(df.direction, side, coef, df.gustiness, T₀, df.filtered_velocities,
                              surface_pressure, constants)
    return BoundaryCondition(Flux(), new_df)
end

# BulkDrag with no direction: infer direction from field location, then materialize
function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:BulkDragFunction{Nothing}},
                                                   side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                   microphysical_fields, specific_prognostic_moisture, temperature)
    df = bc.condition
    LX, LY, LZ = loc

    # Determine direction from location: the momentum component lives at a Face in its direction
    if LX isa Face
        direction = XDirection()
    elseif LY isa Face
        direction = YDirection()
    elseif LZ isa Face
        direction = ZDirection()
    else
        throw(ArgumentError("Can only specify BulkDrag on momentum fields (ρu, ρv, ρw)!"))
    end

    directed_df = BulkDragFunction(direction, df.side, df.coefficient, df.gustiness, df.surface_temperature,
                                   df.filtered_velocities, df.surface_pressure, df.thermodynamic_constants)
    return materialize_bulk_drag(directed_df, side, grid, dynamics, microphysics, surface_pressure, constants,
                                 microphysical_fields, specific_prognostic_moisture, temperature)
end

# BulkDrag with direction already set: materialize
function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:DirectedBulkDragFunction},
                                                   side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                   microphysical_fields, specific_prognostic_moisture, temperature)
    return materialize_bulk_drag(bc.condition, side, grid, dynamics, microphysics, surface_pressure, constants,
                                 microphysical_fields, specific_prognostic_moisture, temperature)
end

# Materialize BulkSensibleHeatFlux: populate pressure data, thermodynamic_constants, preserve formulation
function materialize_atmosphere_boundary_condition(bc::BulkSensibleHeatFluxBoundaryCondition,
                                                   side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                   microphysical_fields, specific_prognostic_moisture, temperature)

    bf = bc.condition
    validate_wall(side)
    validate_wall_filtering(side, bf.filtered_velocities)
    T₀ = materialize_surface_field(bf.surface_temperature, grid, side)
    pˢᵗ = standard_pressure(dynamics)
    coef = materialize_coefficient(bf.coefficient, grid, dynamics, microphysics,
                                   surface_pressure, constants,
                                   microphysical_fields, specific_prognostic_moisture, temperature,
                                   Val(:scalar))
    # Auto-create FilteredSurfaceScalar if filtered_velocities is provided
    fs = if isnothing(bf.filtered_velocities)
        nothing
    else
        FilteredSurfaceScalar(grid; height=bf.filtered_velocities.height,
                              filter_timescale=bf.filtered_velocities.filter_timescale)
    end

    new_bf = BulkSensibleHeatFluxFunction(side, coef, bf.gustiness, T₀, surface_pressure, pˢᵗ, constants,
                                          bf.formulation, bf.filtered_velocities, fs)
    return BoundaryCondition(Flux(), new_bf)
end

# Materialize BulkVaporFlux: populate surface_pressure, thermodynamic_constants, and surface
function materialize_atmosphere_boundary_condition(bc::BulkVaporFluxBoundaryCondition,
                                                   side, loc, grid, dynamics, microphysics, surface_pressure, constants,
                                                   microphysical_fields, specific_prognostic_moisture, temperature)

    bf = bc.condition
    validate_wall(side)
    validate_wall_filtering(side, bf.filtered_velocities)
    T₀ = materialize_surface_field(bf.surface_temperature, grid, side)
    ℋ₀ = materialize_surface_field(bf.surface_relative_humidity, grid, side)
    surface = PlanarLiquidSurface()
    coef = materialize_coefficient(bf.coefficient, grid, dynamics, microphysics,
                                   surface_pressure, constants,
                                   microphysical_fields, specific_prognostic_moisture, temperature,
                                   Val(:scalar))

    # Auto-create FilteredSurfaceScalar if filtered_velocities is provided
    fs = if isnothing(bf.filtered_velocities)
        nothing
    else
        FilteredSurfaceScalar(grid; height=bf.filtered_velocities.height,
                              filter_timescale=bf.filtered_velocities.filter_timescale)
    end

    new_bf = BulkVaporFluxFunction(side, coef, bf.gustiness, T₀, ℋ₀, surface_pressure, constants, surface,
                                   bf.filtered_velocities, fs)

    return BoundaryCondition(Flux(), new_bf)
end

#####
##### Utilities
#####

#####
##### Bottom-wall wind speeds at the three staggers, for the temporally filtered surface state
#####

@inline wind_speed²ᶠᶜᶜ(i, j, grid, fields, ::Nothing) = tangential_speed²(i, j, 1, grid, Bottom(), XDirection(), fields)
@inline wind_speed²ᶜᶠᶜ(i, j, grid, fields, ::Nothing) = tangential_speed²(i, j, 1, grid, Bottom(), YDirection(), fields)
@inline wind_speed²ᶜᶜᶜ(i, j, grid, fields, ::Nothing) = tangential_speed²(i, j, 1, grid, Bottom(), nothing,      fields)

# The wall state may be a number, a field on the wall, or a function of the non-`Flat` wall
# coordinates and the time, evaluated at every call (see `wall_value`)
materialize_surface_field(f, grid) = materialize_surface_field(f, grid, Bottom())
materialize_surface_field(::Nothing, grid, side) = nothing
materialize_surface_field(f::Field, grid, side) = f
materialize_surface_field(f::Number, grid, side) = f
materialize_surface_field(f::Function, grid, side) = f

#####
##### Default polynomial filling for Function constructors
#####
##### When a PolynomialCoefficient with `polynomial = nothing` is passed as the
##### coefficient, fill in the appropriate Large & Yeager (2009) default polynomial
##### before constructing the Function struct. This way the user interface is the
##### same regardless of coefficient type: BulkDrag(coefficient=..., gustiness=...).
#####
##### These must be defined after the struct definitions (BulkDragFunction, etc.)
##### so that they add methods to the existing constructors.
#####

BulkDragFunction(d, side, coef::NothingPolynomialCoefficient, g, t, fv, p, c) =
    BulkDragFunction(d, side, fill_polynomial(coef, default_neutral_drag_polynomial, Val(:momentum)), g, t, fv, p, c)

BulkSensibleHeatFluxFunction(side, coef::NothingPolynomialCoefficient, g, t, p, s, c, f, fv, fs) =
    BulkSensibleHeatFluxFunction(side, fill_polynomial(coef, default_neutral_sensible_heat_polynomial, Val(:scalar)),
                                 g, t, p, s, c, f, fv, fs)

BulkVaporFluxFunction(side, coef::NothingPolynomialCoefficient, g, t, h, p, c, s, fv, fs) =
    BulkVaporFluxFunction(side, fill_polynomial(coef, default_neutral_latent_heat_polynomial, Val(:scalar)), g, t, h, p, c, s, fv, fs)

end # module BoundaryConditions
