#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics:
    MoistStaticEnergyState,
    mixture_heat_capacity

materialize_condenstates(microphysics, grid) = NamedTuple()

@inline function compute_temperature(state::MoistStaticEnergyState, ::Nothing, thermo)
    cᵖᵐ = mixture_heat_capacity(state.moisture_fractions, thermo)
    e = state.moist_static_energy
    return e / cᵖᵐ
end
