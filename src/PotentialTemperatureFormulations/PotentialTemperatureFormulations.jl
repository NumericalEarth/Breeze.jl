"""
    PotentialTemperatureFormulations

Submodule defining the liquid-ice potential temperature thermodynamic formulation for atmosphere models.

`LiquidIcePotentialTemperatureFormulation` uses liquid-ice potential temperature density `ρθ`
as the prognostic thermodynamic variable.
"""
module PotentialTemperatureFormulations

export LiquidIcePotentialTemperatureFormulation

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, znode
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Utils: prettysummary, launch!

using Breeze.AtmosphereModels: AtmosphereModels, diagnose_thermodynamic_state,
    set_thermodynamic_variable!, dynamics_density, dynamics_pressure, standard_pressure,
    dynamics_prognostic_fields, compute_moisture_fractions, maybe_adjust_thermodynamic_state,
    div_ρUc, ∇_dot_Jᶜ, AtmosphereModelBuoyancy, microphysical_tendency
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, StaticEnergyState, with_temperature, exner_function, mixture_heat_capacity

# The lowercase c is a singleton instance of Center
const c = Center()

include("potential_temperature_formulation.jl")
include("potential_temperature_tendency.jl")

# Kernel wrapper for launching potential_temperature_tendency
# (needs to be defined after potential_temperature_tendency is defined)
@kernel function compute_potential_temperature_tendency!(Gρθ, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] = potential_temperature_tendency(i, j, k, grid, args...)
end

end # module

