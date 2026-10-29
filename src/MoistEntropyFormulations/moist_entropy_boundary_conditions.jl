#####
##### Boundary conditions on the moist entropy density
#####
#
# An energy flux 𝒬ᵀ (W m⁻²) supplied under the energy key ρE, and a vapor flux Jᵛ (kg m⁻² s⁻¹) supplied
# under the moisture key ρqᵗ (or ρqᵛ), together carry the entropy flux
#
#   Jη = 𝒬ᵀ / T + Jᵛ ηᵛ(T, ρqᵛ) ,
#
# evaluated with the temperature T and vapor density ρqᵛ of the cell adjacent to the wall. The
# evaporated vapor brings its enthalpy hᵛ = cᵖᵛ T + ℒˡ₀ into that cell, where its entropy is the
# partial specific entropy ηᵛ = (hᵛ − 𝒢ᵛ) / T: adding the vapor at that entropy raises the cell's
# internal energy by exactly hᵛ per unit mass, so the energy input is 𝒬ᵀ + hᵛ Jᵛ. A bulk sensible
# heat flux forms its energy flux from the static-energy difference across the surface layer, as
# for `StaticEnergyFormulation`.

"""
$(TYPEDEF)

Wraps an energy flux and a vapor flux into the moist entropy flux
``Jη = 𝒬ᵀ / T + Jᵛ ηᵛ`` at the cell adjacent to the wall. Either flux may be `nothing`.
"""
struct EntropyFluxBoundaryConditionFunction{E, V, S, TC, B}
    energy_flux :: E
    vapor_flux :: V
    side :: S
    thermodynamic_constants :: TC
    water_entropy_offset :: B
end

EntropyFluxBoundaryConditionFunction(energy_flux, vapor_flux, water_entropy_offset) =
    EntropyFluxBoundaryConditionFunction(energy_flux, vapor_flux, nothing, nothing, water_entropy_offset)

Adapt.adapt_structure(to, ef::EntropyFluxBoundaryConditionFunction) =
    EntropyFluxBoundaryConditionFunction(adapt(to, ef.energy_flux), adapt(to, ef.vapor_flux),
                                         adapt(to, ef.side), adapt(to, ef.thermodynamic_constants),
                                         ef.water_entropy_offset)

flux_summary(::Nothing) = "nothing"
flux_summary(flux::Number) = string(flux)
flux_summary(flux) = summary(flux)

Base.summary(ef::EntropyFluxBoundaryConditionFunction) =
    string("EntropyFluxBoundaryConditionFunction(energy=", flux_summary(ef.energy_flux),
           ", vapor=", flux_summary(ef.vapor_flux), ")")

const EntropyFluxBoundaryCondition = BoundaryCondition{<:Flux, <:EntropyFluxBoundaryConditionFunction}

entropy_flux_boundary_condition(energy_flux, vapor_flux, b = default_water_entropy_offset) =
    BoundaryCondition(Flux(), EntropyFluxBoundaryConditionFunction(energy_flux, vapor_flux, b))

@inline wall_flux(::Nothing, ℓ, m, grid, clock, fields, dynamics_fields) = 0
@inline wall_flux(flux, ℓ, m, grid, clock, fields, dynamics_fields) =
    OceananigansBC.getbc(flux, ℓ, m, grid, clock, fields, dynamics_fields)

@inline function OceananigansBC.getbc(ef::EntropyFluxBoundaryConditionFunction, ℓ::Integer, m::Integer,
                                      grid::AbstractGrid, clock, fields, dynamics_fields)
    i, j, k = near_wall_indices(ℓ, m, grid, ef.side)
    𝒬ᵀ = wall_flux(ef.energy_flux, ℓ, m, grid, clock, fields, dynamics_fields)
    Jᵛ = wall_flux(ef.vapor_flux, ℓ, m, grid, clock, fields, dynamics_fields)
    @inbounds begin
        T = fields.T[i, j, k]
        ρqᵛ = fields.ρqᵛ[i, j, k]
    end
    # the vapor carries its partial specific entropy plus the water offset b
    ηᵛ = vapor_entropy(T, ρqᵛ, ef.thermodynamic_constants) + ef.water_entropy_offset
    return 𝒬ᵀ / T + Jᵛ * ηᵛ
end

#####
##### The energy difference a bulk sensible heat flux forms for the entropy
#####

"""
$(TYPEDEF)

Tells a [`BulkSensibleHeatFlux`](@ref) on the moist entropy to form an energy flux from the
static-energy difference ``cᵖᵐ (T - Tˢ) + g (z - zˢ)`` between the adjacent cell and the wall, which
[`EntropyFluxBoundaryConditionFunction`](@ref) then divides by the cell temperature.
"""
struct TemperatureEnergyFlux end

# The static-energy difference s − sˢ = cᵖᵐ (T − Tˢ) + g (z − zˢ) at the composition of the cell,
# which `StaticEnergyFormulation` forms from its prognostic s
@inline function AtmosphereBoundaryConditions.bulk_sensible_heat_difference(i, j, k, grid, side, ::TemperatureEnergyFlux,
                                                                           bf, Tˢ, fields, pˢ, ::Nothing)
    constants = bf.thermodynamic_constants
    q = wall_moisture_fractions(i, j, k, grid, bf.microphysics, fields)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    k₀ = ifelse(side isa Bottom, 1, k)
    zˢ = wall_height(i, j, k₀, grid, side)
    z = znode(i, j, k, grid, c, c, c)
    @inbounds T = fields.T[i, j, k]
    return cᵖᵐ * (T - Tˢ) + constants.gravitational_acceleration * (z - zˢ)
end

#####
##### Routing the energy and water fluxes onto ρη
#####

# ρE → ρη: each energy flux becomes the energy part of an entropy flux
energy_to_entropy_bc(bc) = bc
energy_to_entropy_bc(bc::BoundaryCondition{<:Flux, Nothing}) = bc

function energy_to_entropy_bc(bc::BulkSensibleHeatFluxBoundaryCondition)
    bf = with_formulation(bc.condition, TemperatureEnergyFlux())
    return entropy_flux_boundary_condition(bf, nothing)
end

energy_to_entropy_bc(bc::BoundaryCondition{<:Flux}) = entropy_flux_boundary_condition(bc.condition, nothing)

AtmosphereBoundaryConditions.energy_bcs_to_thermodynamic_bcs(ρE_bcs, ::Val{:ρη}) =
    map_field_boundary_conditions(energy_to_entropy_bc, ρE_bcs)

# A bulk sensible heat flux is an energy flux: it is accepted under ρE only (handled above)
AtmosphereBoundaryConditions.sensible_heat_flux_formulation(::Val{:ρη}, formulation) = TemperatureEnergyFlux()

with_formulation(bf::BulkSensibleHeatFluxFunction, formulation) =
    BulkSensibleHeatFluxFunction(bf.side, bf.coefficient, bf.gustiness, bf.surface_temperature,
                                 bf.standard_pressure, bf.thermodynamic_constants,
                                 formulation, bf.filtered_velocities, bf.filtered_scalar, bf.microphysics)

# The water fluxes: the vapor flux on each side becomes the vapor part of the entropy flux there
function AtmosphereBoundaryConditions.add_moisture_flux_contribution(ρη_bcs, ::Val{:ρη}, bcs, formulation)
    moisture_bcs = get(bcs, :ρqᵗ, get(bcs, :ρqᵛ, nothing))
    moisture_bcs isa FieldBoundaryConditions || return ρη_bcs
    b = water_entropy_offset(formulation)
    sides = map(boundary_sides) do side
        side => with_vapor_flux(getproperty(ρη_bcs, side), getproperty(moisture_bcs, side), side, b)
    end
    return FieldBoundaryConditions(; sides...)
end

vapor_flux_condition(bc) = nothing
vapor_flux_condition(::BoundaryCondition{<:Flux, Nothing}) = nothing
vapor_flux_condition(bc::BoundaryCondition{<:Flux}) = bc.condition

is_entropy_flux(bc) = false
is_entropy_flux(::EntropyFluxBoundaryCondition) = true

# Whether a boundary condition on ρη leaves room for the water flux's entropy: unspecified or no-flux
is_open_for_vapor(bc) = true
is_open_for_vapor(::BoundaryCondition{<:Flux, Nothing}) = true
is_open_for_vapor(bc::BoundaryCondition) = false

function with_vapor_flux(ρη_bc, moisture_bc, side, b)
    Jᵛ = vapor_flux_condition(moisture_bc)
    isnothing(Jᵛ) && return ρη_bc
    is_entropy_flux(ρη_bc) && return entropy_flux_boundary_condition(ρη_bc.condition.energy_flux, Jᵛ, b)
    is_open_for_vapor(ρη_bc) && return entropy_flux_boundary_condition(nothing, Jᵛ, b)
    throw(ArgumentError("The $side boundary condition on ρη, $(summary(ρη_bc)), cannot be combined with " *
                        "the water flux on that side, whose entropy it would have to include. " *
                        "Supply the energy flux under ρE instead."))
end

# Materialize the energy and vapor fluxes inside the entropy flux as they would be materialized alone
function AtmosphereBoundaryConditions.materialize_atmosphere_boundary_condition(bc::EntropyFluxBoundaryCondition,
                                                                                side, loc, grid, dynamics,
                                                                                microphysics, constants)
    ef = bc.condition
    args = (side, loc, grid, dynamics, microphysics, constants)
    energy_flux = materialize_wall_flux(ef.energy_flux, args...)
    vapor_flux = materialize_wall_flux(ef.vapor_flux, args...)
    b = convert(eltype(grid), ef.water_entropy_offset)
    new_ef = EntropyFluxBoundaryConditionFunction(energy_flux, vapor_flux, side, constants, b)
    return BoundaryCondition(Flux(), new_ef)
end

materialize_wall_flux(::Nothing, args...) = nothing
materialize_wall_flux(flux, args...) =
    materialize_atmosphere_boundary_condition(BoundaryCondition(Flux(), flux), args...).condition

# The energy flux 𝒬ᵀ inside an entropy flux, for the diagnostic static energy density
entropy_to_energy_bc(bc) = bc
entropy_to_energy_bc(bc::EntropyFluxBoundaryCondition) = energy_flux_bc(bc.condition.energy_flux)
energy_flux_bc(::Nothing) = BoundaryCondition(Flux(), nothing)
energy_flux_bc(flux) = BoundaryCondition(Flux(), flux)
