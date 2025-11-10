#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics:
    AbstractThermodynamicState,
    temperature

prognostic_field_names(::Nothing) = tuple()
materialize_microphysical_fields(microphysics, grid, bcs) = NamedTuple()
@inline update_microphysical_fields(microphysical_fields, ::Nothing, i, j, k, grid, 𝒰₁, thermo) = nothing

"""
$(TYPEDSIGNATURES)

Return the temperature associated with the thermodynamic `state`,
`microphysics` scheme, and `thermo`dynamic constants.
"""
function compute_temperature(state, microphysics, thermo) end

"""
$(TYPEDSIGNATURES)

Return a possibly adjusted thermodynamic state associated with the
`microphysics` scheme and `thermo`dynamic constants.
"""
@inline compute_thermodynamic_state(state::AbstractThermodynamicState, ::Nothing, thermo) = state

@inline function compute_temperature(𝒰₀::AbstractThermodynamicState, microphysics, thermo)
    𝒰₁ = compute_thermodynamic_state(𝒰₀, microphysics, thermo)
    return temperature(𝒰₁, thermo)
end
