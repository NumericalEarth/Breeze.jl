"""
    PotentialTemperatureFormulationModule

Submodule defining the liquid-ice potential temperature thermodynamic formulation for atmosphere models.

`LiquidIcePotentialTemperatureFormulation` uses liquid-ice potential temperature density `ρθ`
as the prognostic thermodynamic variable.
"""
module PotentialTemperatureFormulationModule

export LiquidIcePotentialTemperatureFormulation

using DocStringExtensions: TYPEDSIGNATURES
using Adapt: Adapt, adapt
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans, CenterField, Center, znode
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Utils: prettysummary, launch!

using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, StaticEnergyState, with_temperature, exner_function, mixture_heat_capacity

# The lowercase c is a singleton instance of Center
const c = Center()

# Import interface functions from parent module
import Breeze.AtmosphereModels: materialize_formulation,
                                prognostic_thermodynamic_field_names,
                                additional_thermodynamic_field_names,
                                thermodynamic_density_name,
                                thermodynamic_density,
                                collect_prognostic_fields,
                                compute_auxiliary_thermodynamic_variables!,
                                diagnose_thermodynamic_state,
                                compute_thermodynamic_tendency!,
                                set_thermodynamic_variable!,
                                static_energy,
                                static_energy_density,
                                liquid_ice_potential_temperature,
                                liquid_ice_potential_temperature_density,
                                dynamics_density,
                                dynamics_pressure,
                                dynamics_standard_pressure,
                                dynamics_prognostic_fields,
                                compute_moisture_fractions,
                                maybe_adjust_thermodynamic_state,
                                div_ρUc,
                                ∇_dot_Jᶜ,
                                AtmosphereModelBuoyancy,
                                microphysical_tendency

include("potential_temperature_formulation_type.jl")
include("potential_temperature_tendency.jl")

end # module

