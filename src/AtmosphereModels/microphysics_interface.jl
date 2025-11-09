#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics:
    AnelasticThermodynamicState

materialize_condenstates(microphysics, grid) = NamedTuple()

@inline function compute_temperature(state::AnelasticThermodynamicState, ::Nothing)
    # Default behavior without microphysics: dry T = Π * θ
    return state.exner_function * state.potential_temperature
end

