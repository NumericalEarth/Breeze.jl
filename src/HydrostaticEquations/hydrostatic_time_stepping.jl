#####
##### Model initialization and state setting for HydrostaticDynamics
#####

unsupported_by_hydrostatic_dynamics(what) =
    throw(ArgumentError("HydrostaticDynamics does not support $what yet."))

# The Coriolis force is evaluated with the horizontal velocities alone. A rotation with a
# horizontal component would be handed `w`, which here is the velocity through the coordinate
# surfaces rather than the vertical velocity, and would silently compute the wrong force.
validate_hydrostatic_coriolis(::Nothing) = nothing
validate_hydrostatic_coriolis(::Union{FPlane, BetaPlane, HydrostaticSphericalCoriolis}) = nothing

validate_hydrostatic_coriolis(coriolis) =
    throw(ArgumentError("HydrostaticDynamics supports the traditional Coriolis force only (FPlane, BetaPlane or " *
                        "HydrostaticSphericalCoriolis), got $(summary(coriolis)). The vertical velocity of " *
                        "HydrostaticDynamics is the velocity through the coordinate surfaces, which the " *
                        "non-traditional terms must not be given."))

# The equations are those of dry air: the moist pressure-gradient force and the moist layer
# mass are not derived, so water in any form is rejected rather than carried with dry constants.
# Three guards keep the model dry: no forcing of the moisture density other than one that is
# identically zero (`validate_dry_forcing`), no boundary condition that is a source of it
# (`validate_dry_boundary_conditions`), and a moisture density that is zero each time the state
# is updated (`validate_dry_state`), whatever the route by which water might have entered.
function validate_dry_forcing(model)
    for name in keys(model.forcing)
        (name ∈ dry_prognostic_names(model) || is_identically_zero(model.forcing[name])) ||
            throw(ArgumentError("The equations of HydrostaticDynamics are those of dry air, so the forcing " *
                                "of $name is rejected: moisture is not supported yet."))
    end
    return nothing
end

dry_prognostic_names(model) = (:pˢ, :u, :v, :ρθ, :ρE, keys(model.tracers)...)

# A forcing or boundary value is known to add nothing only if it is a zero that can be read
# off: the default forcing `Returns(0)`, a number, or no condition at all. A function is not.
is_identically_zero(::Nothing) = true
is_identically_zero(value::Number) = iszero(value)
is_identically_zero(forcing::Returns) = is_identically_zero(forcing.value)
is_identically_zero(_) = false

# Periodic, no-flux and other boundary conditions that are not a source carry no condition, or
# a zero one
is_dry_boundary_condition(::Nothing) = true
is_dry_boundary_condition(bc::BoundaryCondition) = is_identically_zero(bc.condition)
is_dry_boundary_condition(_) = false

# The exchange between the regions of a multi-region grid is not a source either
is_dry_boundary_condition(::BoundaryCondition{<:MultiRegionCommunication}) = true

# On a multi-region grid every region has boundary conditions of its own
function validate_dry_boundary_conditions(model::AtmosphereModel)
    @apply_regionally validate_dry_boundary_conditions(model.moisture_density.boundary_conditions)
    return nothing
end

function validate_dry_boundary_conditions(boundary_conditions)
    for side in (:west, :east, :south, :north, :bottom, :top, :immersed)
        is_dry_boundary_condition(getproperty(boundary_conditions, side)) ||
            throw(ArgumentError("The equations of HydrostaticDynamics are those of dry air, so the $side " *
                                "boundary condition of the moisture density is rejected: moisture is not " *
                                "supported yet."))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Throw an `ArgumentError` unless the moisture density of `model` is zero everywhere.

`HydrostaticDynamics` integrates the equations of dry air. This is checked when the state is
set and each time it is updated, that is after every stage of a time step, so that water that
enters by a route the checks of the forcing and of the boundary conditions do not cover stops
the integration instead of being carried with the constants of dry air.
"""
function validate_dry_state(model)
    ρqᵛ = model.moisture_density
    (iszero(minimum(ρqᵛ)) && iszero(maximum(ρqᵛ))) ||
        throw(ArgumentError("The equations of HydrostaticDynamics are those of dry air, and the moisture " *
                            "density must remain zero: moisture is not supported yet."))
    return nothing
end

# Radiation heats the air through the flux divergence in the tendency of ρθ, and its level
# pressures are those of the coordinate surfaces. The radiative transfer models index their
# columns with the size of one grid, so they are not given a multi-region grid, and a model that
# builds its level pressures from the layer pressures would leave out half of the lowest and of
# the highest layer.
validate_hydrostatic_radiation(::Nothing, grid) = nothing

function validate_hydrostatic_radiation(radiation, grid)
    grid isa MultiRegion.MultiRegionGrids && unsupported_by_hydrostatic_dynamics("radiation on a multi-region grid")

    uses_level_pressure(radiation) ||
        unsupported_by_hydrostatic_dynamics("$(summary(radiation)), which does not take the pressures of " *
                                            "the coordinate surfaces from the dynamics,")
    return nothing
end

# Only the forcings whose effect on a prognostic velocity has been verified are accepted
function validate_velocity_forcing(forcing, name)
    supports_prognostic_velocity(forcing) ||
        throw(ArgumentError("HydrostaticDynamics does not support the forcing $(summary(forcing)) of $name yet. " *
                            "Supported: a function of the coordinates and time, without parameters or " *
                            "field dependencies, under `$name` (an acceleration) " *
                            "or under `ρ$name` (a force per unit volume), and `held_suarez_forcings`."))
    return nothing
end

# Momentum advection is either vector invariant or in flux form.
#
# Vector invariant: the gradient of the kinetic energy is `EnergyConserving`, or both it and the
# vorticity flux are upwind-biased (`WENOVectorInvariant`). The vertical advection is that of the
# scheme's vertical scheme: the kinetic-energy-conserving form unless that scheme is upwind-biased.
# The scheme's divergence scheme is not used: the divergence of the mass flux is that of continuity.
#
# Flux form: a centered or an upwind-biased scheme reconstructs the velocity on the faces of the
# velocity cells, in all three directions (see `momentum_advection_u`).
const FluxFormScheme = Union{AbstractCenteredAdvectionScheme, AbstractUpwindBiasedAdvectionScheme}
const SupportedVerticalScheme = Union{EnergyConserving, AbstractUpwindBiasedAdvectionScheme}

validate_hydrostatic_momentum_advection(::FluxFormScheme) = nothing

function validate_hydrostatic_momentum_advection(advection::VectorInvariant)
    supported_horizontal = advection isa VectorInvariantKEGradientEnergyConserving ||
                           (advection isa VectorInvariantKineticEnergyUpwinding &&
                            advection isa VectorInvariantUpwindVorticity)

    (supported_horizontal && advection.vertical_advection_scheme isa SupportedVerticalScheme) ||
        throw(ArgumentError("HydrostaticDynamics requires a VectorInvariant momentum_advection to have an " *
                            "EnergyConserving kinetic energy gradient, or to be a WENOVectorInvariant, with a " *
                            "vertical scheme that is EnergyConserving or upwind-biased, got " *
                            "$(summary(advection))."))
    return nothing
end

validate_hydrostatic_momentum_advection(advection) =
    throw(ArgumentError("HydrostaticDynamics requires momentum_advection to be VectorInvariant, or a centered " *
                        "or upwind-biased scheme for the momentum equations in flux form, got " *
                        "$(summary(advection))."))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Check that `model` is one `HydrostaticDynamics` can integrate, and diagnose the layer mass from
the initial surface pressure so that the densities are valid before the first `set!`.
"""
function AtmosphereModels.initialize_model_thermodynamics!(model::HydrostaticModel)
    model.formulation isa LiquidIcePotentialTemperatureFormulation ||
        unsupported_by_hydrostatic_dynamics("the $(summary(model.formulation))")

    validate_hydrostatic_momentum_advection(model.advection.momentum)

    isnothing(model.microphysics) || unsupported_by_hydrostatic_dynamics("microphysics")
    validate_dry_forcing(model)
    validate_dry_boundary_conditions(model)
    isnothing(model.closure) || unsupported_by_hydrostatic_dynamics("turbulence closures")
    validate_hydrostatic_radiation(model.radiation, model.grid)

    validate_hydrostatic_coriolis(model.coriolis)
    validate_velocity_forcing(model.forcing.u, :u)
    validate_velocity_forcing(model.forcing.v, :v)

    diagnose_layer_mass_of_new_state!(model)

    return nothing
end

# Diagnose the layer mass from a surface pressure the user has just supplied, and report a
# layer without mass as the invalid argument it is.
function diagnose_layer_mass_of_new_state!(model)
    try
        AtmosphereModels.compute_total_density!(model)
    catch err
        err isa DomainError || rethrow()
        throw(ArgumentError("The surface pressure gives a layer of HydrostaticDynamics a mass that is not " *
                            "positive: the pressure thickness Δp = ΔA + ΔB pˢ must be positive in every " *
                            "layer of every column. Check the hybrid coefficients against the lowest " *
                            "surface pressure."))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set variables of an `AtmosphereModel` with `HydrostaticDynamics`. The surface pressure `pˢ` is
set first, because it determines the layer mass that weights the other prognostic variables;
the remaining keyword arguments are set as for any `AtmosphereModel`.
"""
function Fields.set!(model::HydrostaticModel; pˢ=nothing, kw...)
    (haskey(kw, :ρu) || haskey(kw, :ρv)) &&
        throw(ArgumentError("Cannot set the momentum density of HydrostaticDynamics, whose prognostic " *
                            "variables are the velocities u and v."))

    isnothing(pˢ) || set!(model.dynamics.surface_pressure, pˢ)
    diagnose_layer_mass_of_new_state!(model)
    invoke(set!, Tuple{AtmosphereModel}, model; kw...)
    validate_dry_state(model)
    return nothing
end

# `u` and `v` are prognostic: on a grid that takes geographic components, setting the
# velocities sets the prognostic variables
AtmosphereModels.set_horizontal_momentum_from_velocities!(::HydrostaticModel) = nothing

# `u` and `v` are prognostic and set as such; `w` is diagnosed from continuity
AtmosphereModels.set_velocity!(::HydrostaticModel, name::Symbol, value) =
    throw(ArgumentError("Cannot set the velocity component $name of HydrostaticDynamics, " *
                        "which is diagnosed from continuity."))
