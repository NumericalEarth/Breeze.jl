#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics:
    AbstractThermodynamicState,
    temperature,
    MoistureMassFractions

#####
##### Definition of the microphysics interface, with methods for "Nothing" microphysics
#####

"""
$(TYPEDSIGNATURES)

Return `tuple()` - zero-moment scheme has no prognostic variables.
"""
prognostic_field_names(::Nothing) = tuple()

"""
$(TYPEDSIGNATURES)

Build microphysical fields associated with `microphysics` on `grid` and with
user defined `boundary_conditions`.
"""
materialize_microphysical_fields(microphysics::Nothing, grid, boundary_conditions) = NamedTuple()

"""
$(TYPEDSIGNATURES)

Update microphysical fields for `microphysics_scheme` given the thermodynamic `state` and
`thermo`dynamic parameters.
"""
@inline update_microphysical_fields!(microphysical_fields, microphysics::Nothing, i, j, k, grid, density, state, thermo) = nothing

"""
$(TYPEDSIGNATURES)

Return the temperature associated with the thermodynamic `state`,
`microphysics` scheme, and `thermo`dynamic constants.
"""
compute_temperature(state, microphysics::Nothing, thermo) = temperature(state, thermo)

"""
$(TYPEDSIGNATURES)

Return a possibly adjusted thermodynamic state associated with the
`microphysics` scheme, and `thermo`dynamic constants.
"""
@inline compute_thermodynamic_state(state::AbstractThermodynamicState, ::Nothing, thermo) = state

"""
$(TYPEDSIGNATURES)

Build and return [`MoistureMassFractions`](@ref) at `(i, j, k)` for the given `grid`,
`microphysics`, `microphysical_fields`, and total moisture mass fraction `qᵗ`.

Dispatch is provided for `::Nothing` microphysics here. Specific microphysics
schemes may extend this method to provide tailored behavior.
"""
@inline compute_moisture_fractions(i, j, k, grid, microphysics::Nothing, ρ, qᵗ, μ) = @inbounds MoistureMassFractions(qᵗ)

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities associated with `microphysics` and `name`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
"""
@inline microphysical_velocities(microphysics::Nothing, name) = nothing

"""
$(TYPEDSIGNATURES)

Compute the thermodynamic state associated with `microphysics` and `thermo`dynamic constants,
and then return the temperature associated with that state.
"""
@inline function compute_temperature(𝒰₀::AbstractThermodynamicState, microphysics, thermo)
    𝒰₁ = compute_thermodynamic_state(𝒰₀, microphysics, thermo)
    return temperature(𝒰₁, thermo)
end

"""
$(TYPEDSIGNATURES)

Return the tendency of the microphysical field `name` associated with `microphysics` and `thermo`dynamic constants.
"""
@inline microphysical_tendency(i, j, k, grid, microphysics::Nothing, name, ρ, μ, thermo) = zero(grid)
