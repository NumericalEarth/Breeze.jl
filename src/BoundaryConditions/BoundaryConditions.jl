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
       StabilityFunction,
       RichardsonNumberMapping,
       default_neutral_drag_polynomial,
       default_neutral_sensible_heat_polynomial,
       default_neutral_latent_heat_polynomial

using ..AtmosphereModels: AtmosphereModels, grid_moisture_fractions, dynamics_density,
                          dynamics_thermodynamic_fields,
                          standard_pressure, default_drag_surface_temperature,
                          moisture_specific_name, thermodynamic_density_name,
                          total_energy_density_name, moisture_prognostic_name,
                          total_moisture_density_name
using ..AtmosphereModels.Diagnostics: saturation_total_specific_moisture,
                                      virtual_potential_temperature
using ..Thermodynamics: saturation_specific_humidity, surface_density, PlanarLiquidSurface,
                        mixture_heat_capacity, MoistureMassFractions,
                        potential_temperature_from_temperature, surface_pressure_from_cell_center

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
                              ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES

#####
##### The surface-layer field tuple
#####

"""
$(TYPEDSIGNATURES)

The field tuple the wall diagnostics read: the model fields merged with the thermodynamic pressure
`p` and density `ρ`.

Those two arrive separately, in the tuple `boundary_condition_args` passes after the model fields,
because under `AnelasticDynamics` they are dimension-reduced fields. Admitting them to
`Oceananigans.fields(model)` would make the positional lookup that user forcings and boundary
functions perform on it non-concrete, and the GPU compiler then rejects every kernel that performs
one — see `AtmosphereModels.dynamics_thermodynamic_fields`. Merging the two here is resolved at
compile time, and everything downstream reads its fields by name, which stays type-stable however
heterogeneous the merged tuple is.

The single-argument method assembles the same tuple from a model, for host-side callers: the
filtered Δθᵥ update and the tests.
"""
@inline surface_layer_state(model_fields, dynamics_fields) = merge(model_fields, dynamics_fields)

surface_layer_state(model) = surface_layer_state(Oceananigans.fields(model),
                                                 dynamics_thermodynamic_fields(model.dynamics))

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

Boundary conditions supplied under an interface key are first routed onto the prognostic field
that carries them: the energy key `ρE` onto the thermodynamic variable of `formulation` by
[`convert_energy_bcs`](@ref), and the moisture key `ρqᵗ` onto the moisture density of
`microphysics` by [`convert_moisture_bcs`](@ref).
"""
function AtmosphereModels.materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                                           dynamics, microphysics,
                                                                           thermodynamic_constants)

    # Route interface keys onto the prognostic fields that carry them
    boundary_conditions = convert_energy_bcs(boundary_conditions, formulation)
    boundary_conditions = convert_moisture_bcs(boundary_conditions, microphysics)

    materialized = Dict{Symbol, Any}()
    for (name, fbcs) in pairs(boundary_conditions)
        loc = field_location(Val(name))
        materialized[name] = materialize_atmosphere_field_bcs(fbcs, loc, grid, dynamics, microphysics,
                                                              thermodynamic_constants)
    end
    return NamedTuple(materialized)
end

#####
##### Route interface keys (ρE, ρqᵗ) onto the prognostic fields that carry them
#####

const boundary_sides = (:west, :east, :south, :north, :bottom, :top, :immersed)

# Whether the caller wrote a condition on a side, as opposed to the constructor filling it in.
# Distinct from `nondefault_bc`, which asks whether an entry carries anything worth converting and
# so treats an explicit no-flux as nothing at all.
specified_bc(bc) = !(bc isa DefaultBoundaryCondition)

nondefault_bc(::Nothing) = false
nondefault_bc(::BoundaryCondition{<:Flux, Nothing}) = false
nondefault_bc(::DefaultBoundaryCondition) = false
nondefault_bc(bc) = true

# Check if FieldBoundaryConditions has any non-default values
has_nondefault_bcs(::Nothing) = false
has_nondefault_bcs(fbcs) = false
has_nondefault_bcs(fbcs::FieldBoundaryConditions) =
    any(side -> nondefault_bc(getproperty(fbcs, side)), boundary_sides)

# Error if an interface key and the prognostic field it routes onto both carry a non-default
# condition on the same side: the two would be summed into one flux there, which is never what a
# caller means. Different sides are two halves of one specification and are merged.
function validate_interface_bcs(bcs, interface_name, target_name)
    interface_bcs = get(bcs, interface_name, nothing)
    target_bcs = get(bcs, target_name, nothing)

    interface_bcs isa FieldBoundaryConditions && target_bcs isa FieldBoundaryConditions || return nothing

    contested = Tuple(side for side in boundary_sides
                      if specified_bc(getproperty(interface_bcs, side)) &&
                         specified_bc(getproperty(target_bcs, side)))

    if !isempty(contested)
        throw(ArgumentError("Cannot specify boundary conditions on both $target_name and $interface_name " *
                            "on the same side, but both carry one on $contested. Both are applied to " *
                            "$target_name, so supplying both would sum them into a single flux there. " *
                            "Use $interface_name, which is valid whatever the formulation and " *
                            "microphysics, or $target_name, but not both on one side."))
    end

    return nothing
end

# Take each side the caller wrote under the interface key, and the target's own condition on every
# other side. `validate_interface_bcs` has already rejected any side written under both.
merge_interface_sides(interface_bcs, target_bcs) = interface_bcs

merge_interface_sides(interface_bcs, target_bcs::FieldBoundaryConditions) =
    FieldBoundaryConditions(; (side => (specified_bc(getproperty(interface_bcs, side)) ?
                                        getproperty(interface_bcs, side) :
                                        getproperty(target_bcs, side))
                               for side in boundary_sides)...)

# Strip `interface_name` from `bcs`, returning the remainder together with the conditions bound for
# `target_name`: each side taken from the interface entry, put by `adapt` into the units
# `target_name` requires, or from what was supplied under `target_name` itself.
function route_interface_bcs(bcs, interface_name, target_name, adapt)
    validate_interface_bcs(bcs, interface_name, target_name)

    interface_bcs = get(bcs, interface_name, nothing)
    bcs = NamedTuple(k => v for (k, v) in pairs(bcs) if k !== interface_name)

    target_bcs = get(bcs, target_name, FieldBoundaryConditions())
    has_nondefault_bcs(interface_bcs) || return bcs, target_bcs

    return bcs, merge_interface_sides(adapt(interface_bcs), target_bcs)
end

"""
$(TYPEDSIGNATURES)

Assemble the boundary conditions of the prognostic thermodynamic density of `formulation`:
move any conditions supplied under the energy key `ρE` (see
[`total_energy_density_name`](@ref AtmosphereModels.total_energy_density_name)) onto it,
converting the energy flux as that variable requires, and tell any bulk sensible-heat flux
which surface difference to form. The `ρE` entry is dropped — it names an interface, not a
field.
"""
function convert_energy_bcs(bcs, formulation)
    ρᵡ_name = thermodynamic_density_name(formulation)
    ρᵡ = Val(ρᵡ_name)
    bcs, ρᵡ_bcs = route_interface_bcs(bcs, total_energy_density_name, ρᵡ_name,
                                      ρE_bcs -> energy_bcs_to_thermodynamic_bcs(ρE_bcs, ρᵡ))

    # `BulkSensibleHeatFlux` forms its surface difference in the prognostic variable itself,
    # whether it arrived under `ρE` or under that variable's own key.
    ρᵡ_bcs = set_sensible_heat_formulation_bcs(ρᵡ_bcs, sensible_heat_flux_formulation(ρᵡ))

    return merge(bcs, NamedTuple{(ρᵡ_name,)}((ρᵡ_bcs,)))
end

"""
$(TYPEDSIGNATURES)

Assemble the boundary conditions of the prognostic moisture density of `microphysics`, moving
any conditions supplied under the moisture key `ρqᵗ` (see
[`total_moisture_density_name`](@ref AtmosphereModels.total_moisture_density_name)) onto it.
Water enters that variable unconverted whatever the scheme calls it, so unlike the energy key
this is a pure re-key. The `ρqᵗ` entry is dropped — it names an interface, not a field.
"""
function convert_moisture_bcs(bcs, microphysics)
    ρq_name = moisture_prognostic_name(microphysics)
    bcs, ρq_bcs = route_interface_bcs(bcs, total_moisture_density_name, ρq_name, identity)
    return merge(bcs, NamedTuple{(ρq_name,)}((ρq_bcs,)))
end

# ρθ: an energy flux 𝒬 enters as the potential temperature flux Jᶿ = 𝒬 / cᵖᵐ, applied by
# `EnergyFluxBoundaryCondition`.
energy_bcs_to_thermodynamic_bcs(ρE_bcs, ::Val{:ρθ}) = energy_to_theta_bcs(ρE_bcs)

# ρs: static energy is an energy per unit mass, so an energy flux needs no conversion.
energy_bcs_to_thermodynamic_bcs(ρE_bcs, ::Val{:ρs}) = ρE_bcs

# A new thermodynamic formulation must say how an energy flux enters its prognostic variable.
energy_bcs_to_thermodynamic_bcs(ρE_bcs, ::Val{ρᵡ_name}) where ρᵡ_name =
    throw(ArgumentError("Energy boundary conditions (ρE) are not implemented for the prognostic " *
                        "thermodynamic variable $ρᵡ_name. Set boundary conditions on $ρᵡ_name directly."))

# The surface difference Δϕ that a bulk sensible-heat flux forms, per prognostic variable.
sensible_heat_flux_formulation(::Val{:ρθ}) = PotentialTemperatureFlux()
sensible_heat_flux_formulation(::Val{:ρs}) = StaticEnergyFlux()

# Materialize FieldBoundaryConditions by walking through each boundary
function materialize_atmosphere_field_bcs(fbcs::FieldBoundaryConditions, loc, grid, dynam, micro, consts)
    args = (loc, grid, dynam, micro, consts)
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
materialize_atmosphere_boundary_condition(bc, side, loc, grid, dynamics, microphysics, constants) = bc

#####
##### Materialize coefficient: fill in VPT/pressure/constants for PolynomialCoefficient
#####

# Default: pass through unchanged (constant coefficients, etc.)
materialize_coefficient(C, grid, dynamics, microphysics, constants, transfer_type) = C

# For PolynomialCoefficient: create the surface-layer θᵥ and return a fully-populated coefficient
function materialize_coefficient(coef::PolynomialCoefficient, grid, dynamics, microphysics,
                                 constants, transfer_type)
    pˢᵗ = standard_pressure(dynamics)
    moisture_name = Val(moisture_specific_name(microphysics))
    θᵥ = BoundaryVirtualPotentialTemperature(microphysics, moisture_name, pˢᵗ, constants)

    return PolynomialCoefficient(coef.polynomial,
                                 coef.roughness_length,
                                 coef.minimum_wind_speed,
                                 coef.stability_function,
                                 coef.surface,
                                 coef.moisture_availability,
                                 θᵥ, pˢᵗ, constants,
                                 transfer_type)
end

# The surface phase and moisture availability of a bulk vapor flux follow its coefficient when
# that is a `PolynomialCoefficient`, so that evaporation and the stability correction see the
# same surface humidity. A constant coefficient implies a saturated liquid surface unless a
# `moisture_availability` is given.
coefficient_surface(::Number) = PlanarLiquidSurface()
coefficient_surface(coef::PolynomialCoefficient) = coef.surface

coefficient_moisture_availability(::Number) = 1
coefficient_moisture_availability(coef::PolynomialCoefficient) = coef.moisture_availability

resolve_moisture_availability(::Nothing, coefficient) = coefficient_moisture_availability(coefficient)
resolve_moisture_availability(β::Number, ::Number) = β

function resolve_moisture_availability(β::Number, coef::PolynomialCoefficient)
    βᶜ = coef.moisture_availability
    convert(typeof(βᶜ), β) == βᶜ ||
        throw(ArgumentError("BulkVaporFlux was given moisture_availability = $β, but its " *
                            "PolynomialCoefficient has moisture_availability = $βᶜ"))
    return β
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

function materialize_bulk_drag(df, side, grid, dynamics, microphysics, constants)
    validate_wall(side)
    validate_drag_direction(side, df.direction)
    validate_wall_filtering(side, df.filtered_velocities)

    # The momentum-drag formula `Jᵘ = -ρˢ Cᴰ |U| u` needs a surface temperature to
    # compute ρˢ. When the user did not supply one (allowed for constant `coefficient`),
    # fall back to the reference-state surface temperature derived from the dynamics.
    Tˢ_input = if isnothing(df.surface_temperature)
        default_drag_surface_temperature(dynamics, grid, constants)
    else
        df.surface_temperature
    end
    Tˢ = materialize_surface_field(Tˢ_input, grid, side)
    coef = materialize_coefficient(df.coefficient, grid, dynamics, microphysics, constants, Val(:momentum))
    new_df = BulkDragFunction(df.direction, side, coef, df.gustiness, Tˢ, df.filtered_velocities, constants)
    return BoundaryCondition(Flux(), new_df)
end

# BulkDrag with no direction: infer direction from field location, then materialize
function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:BulkDragFunction{Nothing}},
                                                   side, loc, grid, dynamics, microphysics, constants)
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
                                   df.filtered_velocities, df.thermodynamic_constants)
    return materialize_bulk_drag(directed_df, side, grid, dynamics, microphysics, constants)
end

# BulkDrag with direction already set: materialize
function materialize_atmosphere_boundary_condition(bc::BoundaryCondition{<:Flux, <:DirectedBulkDragFunction},
                                                   side, loc, grid, dynamics, microphysics, constants)
    return materialize_bulk_drag(bc.condition, side, grid, dynamics, microphysics, constants)
end

# Materialize BulkSensibleHeatFlux: populate constants and preserve the formulation
function materialize_atmosphere_boundary_condition(bc::BulkSensibleHeatFluxBoundaryCondition,
                                                   side, loc, grid, dynamics, microphysics, constants)

    bf = bc.condition
    validate_wall(side)
    validate_wall_filtering(side, bf.filtered_velocities)
    Tˢ = materialize_surface_field(bf.surface_temperature, grid, side)
    pˢᵗ = standard_pressure(dynamics)
    coef = materialize_coefficient(bf.coefficient, grid, dynamics, microphysics, constants, Val(:scalar))
    # Auto-create FilteredSurfaceScalar if filtered_velocities is provided
    fs = if isnothing(bf.filtered_velocities)
        nothing
    else
        FilteredSurfaceScalar(grid; height=bf.filtered_velocities.height,
                              filter_timescale=bf.filtered_velocities.filter_timescale)
    end

    new_bf = BulkSensibleHeatFluxFunction(side, coef, bf.gustiness, Tˢ, pˢᵗ, constants,
                                          bf.formulation, bf.filtered_velocities, fs)
    return BoundaryCondition(Flux(), new_bf)
end

# Materialize BulkVaporFlux: populate thermodynamic_constants and the surface type
function materialize_atmosphere_boundary_condition(bc::BulkVaporFluxBoundaryCondition,
                                                   side, loc, grid, dynamics, microphysics, constants)

    bf = bc.condition
    validate_wall(side)
    validate_wall_filtering(side, bf.filtered_velocities)
    Tˢ = materialize_surface_field(bf.surface_temperature, grid, side)
    ℋˢ = materialize_surface_field(bf.surface_relative_humidity, grid, side)
    surface = coefficient_surface(bf.coefficient)
    β = convert(eltype(grid), resolve_moisture_availability(bf.moisture_availability, bf.coefficient))
    coef = materialize_coefficient(bf.coefficient, grid, dynamics, microphysics, constants, Val(:scalar))

    # Auto-create FilteredSurfaceScalar if filtered_velocities is provided
    fs = if isnothing(bf.filtered_velocities)
        nothing
    else
        FilteredSurfaceScalar(grid; height=bf.filtered_velocities.height,
                              filter_timescale=bf.filtered_velocities.filter_timescale)
    end

    new_bf = BulkVaporFluxFunction(side, coef, bf.gustiness, Tˢ, ℋˢ, constants, surface, β,
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

BulkDragFunction(d, side, coef::NothingPolynomialCoefficient, g, t, fv, c) =
    BulkDragFunction(d, side, fill_polynomial(coef, default_neutral_drag_polynomial, Val(:momentum)), g, t, fv, c)

BulkSensibleHeatFluxFunction(side, coef::NothingPolynomialCoefficient, g, t, s, c, f, fv, fs) =
    BulkSensibleHeatFluxFunction(side, fill_polynomial(coef, default_neutral_sensible_heat_polynomial, Val(:scalar)),
                                 g, t, s, c, f, fv, fs)

BulkVaporFluxFunction(side, coef::NothingPolynomialCoefficient, g, t, h, c, s, β, fv, fs) =
    BulkVaporFluxFunction(side, fill_polynomial(coef, default_neutral_latent_heat_polynomial, Val(:scalar)), g, t, h, c, s, β, fv, fs)

end # module BoundaryConditions
