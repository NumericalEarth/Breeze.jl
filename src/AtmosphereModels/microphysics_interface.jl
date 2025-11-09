#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics:
    MoistStaticEnergyState,
    mixture_heat_capacity

materialize_microphysical_fields(microphysics, grid, bcs) = NamedTuple()
prognostic_field_names(::Nothing) = tuple()

@inline function compute_temperature(state::MoistStaticEnergyState, ::Nothing, thermo)
    cᵖᵐ = mixture_heat_capacity(state.moisture_mass_fractions, thermo)
    e = state.moist_static_energy
    return e / cᵖᵐ
end
