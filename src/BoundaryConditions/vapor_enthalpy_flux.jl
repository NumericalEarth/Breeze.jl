#####
##### Enthalpy of the water crossing a wall, for a prognostic total energy
#####
##### A water flux through a wall carries its enthalpy. Potential temperature and static energy are
##### intensive variables that the vapor leaves nearly unchanged at fixed temperature, so their
##### boundary conditions ignore it; a prognostic total energy density ρE must receive it. At a wall
##### where the moisture density carries a flux Jᵛ, the flux of ρE is
#####
#####   𝒬ᴱ = 𝒬ᵀ + Jᵛ hᵛ(T₁),   hᵛ(T) = cᵖᵛ T + ℒˡ₀,
#####
##### with 𝒬ᵀ the sensible (or any user-supplied) energy flux, T₁ the temperature of the wall-adjacent
##### cell and ℒˡ₀ the latent heat of condensation extrapolated to zero kelvin, so that the enthalpy
##### hᵛ is measured from liquid water at zero kelvin, the origin of the total energy.
#####

"""
$(TYPEDEF)

The flux of a total energy density through a wall: the energy flux `energy` (the boundary
condition supplied for `ρE`, or `nothing`) plus the enthalpy `cᵖᵛ T₁ + ℒˡ₀` of the water flux
`vapor` (the moisture density's boundary condition on the same wall), with `T₁` the temperature of
the wall-adjacent cell. Built by the `AtmosphereModel` constructor for `TotalEnergyFormulation`;
the moisture density keeps its own boundary condition, and `vapor` evaluates the same condition.
"""
struct VaporEnthalpyFluxFunction{E, V, S, TC}
    energy :: E
    vapor :: V
    side :: S
    thermodynamic_constants :: TC
end

Adapt.adapt_structure(to, vf::VaporEnthalpyFluxFunction) =
    VaporEnthalpyFluxFunction(Adapt.adapt(to, vf.energy),
                              Adapt.adapt(to, vf.vapor),
                              Adapt.adapt(to, vf.side),
                              Adapt.adapt(to, vf.thermodynamic_constants))

Base.summary(vf::VaporEnthalpyFluxFunction) =
    string("VaporEnthalpyFluxFunction(", flux_condition_summary(vf.energy), " + hᵛ × ",
           flux_condition_summary(vf.vapor), ")")

flux_condition_summary(::Nothing) = "0"
flux_condition_summary(condition::Number) = string(condition)
flux_condition_summary(condition) = summary(condition)

const VaporEnthalpyFluxBoundaryCondition = BoundaryCondition{<:Flux, <:VaporEnthalpyFluxFunction}

@inline energy_flux(::Nothing, ℓ, m, grid, clock, model_fields, dynamics_fields) = zero(grid)
@inline energy_flux(condition, ℓ, m, grid, clock, model_fields, dynamics_fields) =
    OceananigansBC.getbc(condition, ℓ, m, grid, clock, model_fields, dynamics_fields)

@inline function OceananigansBC.getbc(vf::VaporEnthalpyFluxFunction, ℓ::Integer, m::Integer,
                                      grid::AbstractGrid, clock, model_fields, dynamics_fields)
    𝒬ᵀ = energy_flux(vf.energy, ℓ, m, grid, clock, model_fields, dynamics_fields)
    Jᵛ = OceananigansBC.getbc(vf.vapor, ℓ, m, grid, clock, model_fields, dynamics_fields)
    i, j, k = near_wall_indices(ℓ, m, grid, vf.side)
    T₁ = @inbounds model_fields.T[i, j, k]
    constants = vf.thermodynamic_constants
    cᵖᵛ = constants.vapor.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    return 𝒬ᵀ + Jᵛ * (cᵖᵛ * T₁ + ℒˡ₀)
end

# The bulk sensible-heat flux inside the wrapper keeps its own filtered state. The water flux is the
# moisture density's own condition, which that field's boundary conditions update.
OceananigansBC.update_boundary_condition!(bc::VaporEnthalpyFluxBoundaryCondition, side, field, model) =
    OceananigansBC.update_boundary_condition!(energy_boundary_condition(bc), side, field, model)

initialize_boundary_condition!(bc::VaporEnthalpyFluxBoundaryCondition, side, field, model) =
    initialize_boundary_condition!(energy_boundary_condition(bc), side, field, model)

"""
$(TYPEDSIGNATURES)

Return the energy-flux part of a wall boundary condition of a total energy density, stripped of the
enthalpy of the water crossing the wall: the sensible heat flux of a `VaporEnthalpyFluxFunction`,
and any other boundary condition unchanged.
"""
energy_boundary_condition(bc) = bc
energy_boundary_condition(bc::VaporEnthalpyFluxBoundaryCondition) = flux_boundary_condition(bc.condition.energy)

flux_boundary_condition(::Nothing) = OceananigansBC.NoFluxBoundaryCondition()
flux_boundary_condition(condition) = BoundaryCondition(Flux(), condition)

"""
$(TYPEDSIGNATURES)

Return `fbcs` with every wall flux of a total energy density reduced to its energy-flux part (see
[`energy_boundary_condition`](@ref)), for diagnostics of the sensible heat flux alone.
"""
energy_boundary_conditions(fbcs::FieldBoundaryConditions) =
    map_field_boundary_conditions(energy_boundary_condition, fbcs)

#####
##### Coupling the moisture flux into the energy flux
#####

# Only a prognostic total energy takes the enthalpy of the water crossing a wall.
couple_vapor_enthalpy_bcs(bcs, ρᵡ, microphysics, constants) = bcs

function couple_vapor_enthalpy_bcs(bcs, ::Val{:ρE}, microphysics, constants)
    ρq_name = moisture_prognostic_name(microphysics)
    ρq_bcs = get(bcs, ρq_name, nothing)
    ρE_bcs = get(bcs, total_energy_density_name, nothing)
    ρq_bcs isa FieldBoundaryConditions && ρE_bcs isa FieldBoundaryConditions || return bcs

    sides = (west = West(), east = East(), south = South(), north = North(), bottom = Bottom(), top = Top())
    coupled = NamedTuple(name => couple_vapor_enthalpy(getproperty(ρE_bcs, name), getproperty(ρq_bcs, name),
                                                       side, constants)
                         for (name, side) in pairs(sides))

    coupled_ρE_bcs = FieldBoundaryConditions(; coupled..., immersed = ρE_bcs.immersed)
    return merge(bcs, NamedTuple{(total_energy_density_name,)}((coupled_ρE_bcs,)))
end

# No water flux through this wall: the energy condition stands.
couple_vapor_enthalpy(ρE_bc, ρq_bc, side, constants) = ρE_bc

couple_vapor_enthalpy(ρE_bc, ρq_bc::BoundaryCondition{<:Flux}, side, constants) =
    couple_vapor_enthalpy(ρE_bc, ρq_bc, ρq_bc.condition, side, constants)

couple_vapor_enthalpy(ρE_bc, ρq_bc, ::Nothing, side, constants) = ρE_bc

function couple_vapor_enthalpy(ρE_bc, ρq_bc, vapor_condition, side, constants)
    energy = energy_flux_condition(ρE_bc)
    vf = VaporEnthalpyFluxFunction(energy, vapor_condition, side, constants)
    return BoundaryCondition(Flux(), vf)
end

energy_flux_condition(::Nothing) = nothing
energy_flux_condition(::DefaultBoundaryCondition) = nothing
energy_flux_condition(bc::BoundaryCondition{<:Flux}) = bc.condition

energy_flux_condition(bc) =
    throw(ArgumentError("The moisture density carries a flux through a wall on which the total energy " *
                        "density has the non-flux boundary condition $(summary(bc)). The enthalpy of " *
                        "the water crossing the wall can only be added to a flux boundary condition."))
