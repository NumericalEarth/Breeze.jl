#####
##### Enthalpy of the water crossing a wall, for a prognostic energy
#####
##### A water flux through a wall carries its enthalpy. Potential temperature is an intensive variable
##### that the vapor leaves unchanged at fixed temperature, so its boundary conditions ignore it; an
##### energy density must receive it. At a wall where the moisture density carries a flux Jᵛ, the flux
##### of the energy density is
#####
#####   𝒬ᴱ = 𝒬ᵀ + Jᵛ hᵛ(T₁),   hᵛ(T) = cᵛ T + ℓᵛ,
#####
##### with 𝒬ᵀ the sensible (or any user-supplied) energy flux, T₁ the temperature of the wall-adjacent
##### cell, and (cᵛ, ℓᵛ) set by the energy variable:
#####
#####   ρE (total energy):          cᵛ = cᵖᵛ,         ℓᵛ = ℒˡ₀ − cˡ T₀
#####       the enthalpy of vapor measured from the energy origin T₀ of the formulation (liquid water
#####       has zero internal energy at T₀), with ℒˡ₀ the latent heat extrapolated to zero kelvin;
#####   ρs (moist static energy):   cᵛ = cᵖᵛ − cᵖᵈ,   ℓᵛ = 0
#####       s = cᵖᵐ T + g z − ℒˡ₀ qˡ − ℒⁱ₀ qⁱ carries no latent energy on the vapor, and on the
#####       anelastic core the vapor displaces dry air at fixed density, so evaporating Jᵛ at T₁ changes
#####       ρs by (cᵖᵛ − cᵖᵈ) T₁ Jᵛ. Without it the evaporated vapor would cool the wall-adjacent cell.

"""
$(TYPEDEF)

The flux of an energy density through a wall: the energy flux `energy` (the boundary condition
supplied for the energy, or `nothing`) plus the enthalpy `heat_capacity × T₁ + enthalpy_offset` of
the water flux `vapor` (the moisture density's boundary condition on the same wall), with `T₁` the
temperature of the wall-adjacent cell. For `TotalEnergyFormulation` that enthalpy is
`cᵖᵛ T₁ + ℒˡ₀ - cˡ T₀`, with `T₀` the energy origin; for `StaticEnergyFormulation` it is
`(cᵖᵛ - cᵖᵈ) T₁`. Built by the `AtmosphereModel` constructor; the moisture density keeps its own
boundary condition, and `vapor` evaluates the same condition.
"""
struct VaporEnthalpyFluxFunction{E, V, S, TC, T}
    energy :: E
    vapor :: V
    side :: S
    thermodynamic_constants :: TC
    heat_capacity :: T      # cᵛ in hᵛ = cᵛ T₁ + ℓᵛ
    enthalpy_offset :: T    # ℓᵛ
end

Adapt.adapt_structure(to, vf::VaporEnthalpyFluxFunction) =
    VaporEnthalpyFluxFunction(Adapt.adapt(to, vf.energy),
                              Adapt.adapt(to, vf.vapor),
                              Adapt.adapt(to, vf.side),
                              Adapt.adapt(to, vf.thermodynamic_constants),
                              vf.heat_capacity,
                              vf.enthalpy_offset)

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
    return 𝒬ᵀ + Jᵛ * (vf.heat_capacity * T₁ + vf.enthalpy_offset)
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

# Only an energy density (ρE or ρs) takes the enthalpy of the water crossing a wall.
couple_vapor_enthalpy_bcs(bcs, ρᵡ, formulation, microphysics, constants) = bcs

# hᵛ = cᵛ T₁ + ℓᵛ: (cᵛ, ℓᵛ) for each energy density; see the header of this file
function vapor_enthalpy_coefficients(::Val{:ρE}, formulation, constants)
    FT = typeof(constants.liquid.heat_capacity)
    T₀ = convert(FT, reference_temperature(formulation))
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    return constants.vapor.heat_capacity, convert(FT, ℒˡ₀ - constants.liquid.heat_capacity * T₀)
end

function vapor_enthalpy_coefficients(::Val{:ρs}, formulation, constants)
    cᵖᵛ = constants.vapor.heat_capacity
    return cᵖᵛ - constants.dry_air.heat_capacity, zero(cᵖᵛ)
end

const VaporEnthalpyCoupledName = Union{Val{:ρE}, Val{:ρs}}

function couple_vapor_enthalpy_bcs(bcs, ρᵡ::VaporEnthalpyCoupledName, formulation, microphysics, constants)
    name = ρᵡ isa Val{:ρE} ? total_energy_density_name : :ρs
    ρq_name = moisture_prognostic_name(microphysics)
    ρq_bcs = get(bcs, ρq_name, nothing)
    ρE_bcs = get(bcs, name, nothing)
    ρq_bcs isa FieldBoundaryConditions && ρE_bcs isa FieldBoundaryConditions || return bcs

    cᵛ, ℓᵛ = vapor_enthalpy_coefficients(ρᵡ, formulation, constants)
    sides = (west = West(), east = East(), south = South(), north = North(), bottom = Bottom(), top = Top())
    coupled = NamedTuple(side_name => couple_vapor_enthalpy(getproperty(ρE_bcs, side_name), getproperty(ρq_bcs, side_name),
                                                            side, constants, cᵛ, ℓᵛ)
                         for (side_name, side) in pairs(sides))

    coupled_ρE_bcs = FieldBoundaryConditions(; coupled..., immersed = ρE_bcs.immersed)
    return merge(bcs, NamedTuple{(name,)}((coupled_ρE_bcs,)))
end

# No water flux through this wall: the energy condition stands.
couple_vapor_enthalpy(ρE_bc, ρq_bc, side, constants, cᵛ, ℓᵛ) = ρE_bc

couple_vapor_enthalpy(ρE_bc, ρq_bc::BoundaryCondition{<:Flux}, side, constants, cᵛ, ℓᵛ) =
    couple_vapor_enthalpy(ρE_bc, ρq_bc, ρq_bc.condition, side, constants, cᵛ, ℓᵛ)

couple_vapor_enthalpy(ρE_bc, ρq_bc, ::Nothing, side, constants, cᵛ, ℓᵛ) = ρE_bc

function couple_vapor_enthalpy(ρE_bc, ρq_bc, vapor_condition, side, constants, cᵛ, ℓᵛ)
    energy = energy_flux_condition(ρE_bc)
    vf = VaporEnthalpyFluxFunction(energy, vapor_condition, side, constants, cᵛ, ℓᵛ)
    return BoundaryCondition(Flux(), vf)
end

energy_flux_condition(::Nothing) = nothing
energy_flux_condition(::DefaultBoundaryCondition) = nothing
energy_flux_condition(bc::BoundaryCondition{<:Flux}) = bc.condition

energy_flux_condition(bc) =
    throw(ArgumentError("The moisture density carries a flux through a wall on which the total energy " *
                        "density has the non-flux boundary condition $(summary(bc)). The enthalpy of " *
                        "the water crossing the wall can only be added to a flux boundary condition."))
